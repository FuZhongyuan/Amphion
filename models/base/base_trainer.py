# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import random
import time
import torch
import numpy as np
from utils.util import Logger, ValueWindow
from torch.utils.data import DataLoader

import torch.nn.functional as F
from transformers import get_inverse_sqrt_schedule, get_constant_schedule

import accelerate
from accelerate.utils import ProjectConfiguration

from models.base.base_sampler import VariableSampler


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


class BaseTrainer:
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name

        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # Init logger
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                os.makedirs(os.path.join(self.exp_dir, "checkpoint"), exist_ok=True)
                self.log_file = os.path.join(
                    os.path.join(self.exp_dir, "checkpoint"), "train.log"
                )
                self.logger = Logger(self.log_file, level=self.args.log_level).logger

        self.time_window = ValueWindow(100)

        if self.accelerator.is_main_process:
            # Log some info
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        self.checkpoint_backup_dir = os.path.join(self.exp_dir, "checkpoint_backup")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.checkpoint_backup_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )

        # Calculate steps per epoch for progress display (will be updated after dataloader creation)
        self.steps_per_epoch = 0
        self.valid_steps_per_epoch = 0
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            if self.accelerator.is_main_process:
                self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            self.train_dataloader = self.accelerator.prepare(
                self.train_dataloader,
            )

        # Calculate steps per epoch for progress display (after accelerator prepare)
        self.steps_per_epoch = len(self.train_dataloader) // self.cfg.train.gradient_accumulation_step
        if self.valid_dataloader is not None:
            self.valid_steps_per_epoch = len(self.valid_dataloader) // self.cfg.train.gradient_accumulation_step
        else:
            self.valid_steps_per_epoch = 0
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # Resume or Finetune
        try:
            with self.accelerator.main_process_first():
                if args.resume:
                    ## Automatically resume according to the current exprimental name
                    print(
                        "Automatically resuming from latest checkpoint in {}...".format(
                            self.checkpoint_dir
                        )
                    )
                    start = time.monotonic_ns()
                    ckpt_path = self._load_model(
                        checkpoint_dir=self.checkpoint_dir, resume_type=args.resume_type
                    )
                    end = time.monotonic_ns()
                    print(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
        except Exception as e:
            print(e)
            import traceback

            print(traceback.format_exc())
            print("Resume failed")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # self.task_type = "VC"
        # if self.accelerator.is_main_process:
        #     self.logger.info("Task type: {}".format(self.task_type))

    def _check_basic_configs(self):
        if self.cfg.train.gradient_accumulation_step <= 0:
            self.logger.fatal("Invalid gradient_accumulation_step value!")
            self.logger.error(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )
            self.accelerator.end_training()
            raise ValueError(
                f"Invalid gradient_accumulation_step value: {self.cfg.train.gradient_accumulation_step}. It should be positive."
            )

    @staticmethod
    def _set_random_seed(seed):
        r"""Set random seed for all possible random modules."""
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)

    def _load_model(
        self,
        checkpoint_dir: str = None,
        checkpoint_path: str = None,
        resume_type: str = "",
    ):
        r"""Load model from checkpoint. If checkpoint_path is None, it will
        load the latest checkpoint in checkpoint_dir. If checkpoint_path is not
        None, it will load the checkpoint specified by checkpoint_path. **Only use this
        method after** ``accelerator.prepare()``.
        """
        if checkpoint_path is None:
            all_ckpts = os.listdir(checkpoint_dir)
            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            ls = list(all_ckpts)
            ls = [os.path.join(checkpoint_dir, i) for i in ls]
            ls.sort(key=lambda x: int(x.split("_")[-2].split("-")[-1]), reverse=True)
            checkpoint_path = ls[0]
            if self.accelerator.is_main_process:
                self.logger.info("Resume from {}".format(checkpoint_path))

        if resume_type in ["resume", ""]:
            # Load all the things, including model weights, optimizer, scheduler, and random states.
            self.accelerator.load_state(input_dir=checkpoint_path)

            # set epoch and step
            self.epoch = int(checkpoint_path.split("_")[-3].split("-")[-1])
            self.step = int(checkpoint_path.split("_")[-2].split("-")[-1])

            if self.accelerator.is_main_process:
                self.logger.info(
                    "Resume from {}, epoch: {}, step: {}".format(
                        checkpoint_path, self.epoch, self.step
                    )
                )

        elif resume_type == "finetune":
            # Load only the model weights
            accelerate.load_checkpoint_and_dispatch(
                self.accelerator.unwrap_model(self.model),
                os.path.join(checkpoint_path, "pytorch_model.bin"),
            )
            if self.accelerator.is_main_process:
                self.logger.info("Load model weights for finetune...")

        else:
            raise ValueError("Resume_type must be `resume` or `finetune`.")

        return checkpoint_path

    def _count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(
                    p.numel() for p in model[key].parameters() if p.requires_grad
                )
        else:
            model_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return model_param

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        # from accelerate import DistributedDataParallelKwargs
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
            # kwargs_handlers=[ddp_kwargs]
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        # IMPORTANT:
        # Only initialize trackers on the main process. Initializing (e.g. TensorBoard)
        # trackers on every distributed process can create multiple event writer queues
        # and lead to steadily increasing host RAM usage in multi-GPU runs.
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.accelerator.init_trackers(self.args.exp_name)
        self.accelerator.wait_for_everyone()

    def _build_model(self):
        raise NotImplementedError

    def _build_dataset(self):
        raise NotImplementedError

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()
            # Check if using new dataset factory pattern (libritts, ljspeech, etc.)
            use_factory_pattern = (
                hasattr(self.cfg.preprocess, "dataset_type")
                and self.cfg.preprocess.dataset_type in ["libritts", "ljspeech"]
            ) or (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            )
            
            if use_factory_pattern:
                train_dataset = Dataset(cfg=self.cfg, is_valid=False)
            else:
                # Legacy pattern: assume dataset is a list
                if isinstance(self.cfg.dataset, (list, tuple)):
                    train_dataset = Dataset(cfg=self.cfg, dataset=self.cfg.dataset[0], is_valid=False)
                else:
                    # If dataset is dict, try to use first key or use factory pattern
                    train_dataset = Dataset(cfg=self.cfg, is_valid=False)
            train_collate = Collator(self.cfg)

            t = time.time()
            if self.accelerator.is_main_process:
                print("Start batching...")

            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )

            if self.accelerator.is_main_process:
                info = "Time taken to batch: {:.1f}s, #bacthes = {}".format(
                    time.time() - t, len(batch_sampler)
                )
                print(info)
                self.logger.info(info)

            np.random.seed(self.cfg.train.random_seed)
            np.random.shuffle(batch_sampler)

            if self.accelerator.is_main_process:
                print(batch_sampler[:1])

            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
                prefetch_factor=self.cfg.train.dataloader.prefetch_factor
            )
            self.accelerator.wait_for_everyone()

            # Build validation dataloader
            valid_loader = None
            # Check if validation is enabled in config (default: True)
            use_validation = getattr(self.cfg.preprocess, "use_validation", True)
            
            if use_factory_pattern and use_validation:
                try:
                    if self.accelerator.is_main_process:
                        print("Building validation dataset...")
                    valid_dataset = Dataset(cfg=self.cfg, is_valid=True)
                    valid_collate = Collator(self.cfg)
                    
                    if len(valid_dataset) > 0:
                        # Use dynamic batch size for validation as well
                        t_valid = time.time()
                        if self.accelerator.is_main_process:
                            print("Start batching validation dataset...")
                        
                        valid_batch_sampler = batch_by_size(
                            valid_dataset.num_frame_indices,
                            valid_dataset.get_num_frames,
                            max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                            max_sentences=self.cfg.train.max_sentences * self.accelerator.num_processes,
                            required_batch_size_multiple=self.accelerator.num_processes,
                        )
                        
                        if self.accelerator.is_main_process:
                            info = "Time taken to batch validation: {:.1f}s, #batches = {}".format(
                                time.time() - t_valid, len(valid_batch_sampler)
                            )
                            print(info)
                            self.logger.info(info)
                        
                        valid_batches = [
                            x[
                                self.accelerator.local_process_index :: self.accelerator.num_processes
                            ]
                            for x in valid_batch_sampler
                            if len(x) % self.accelerator.num_processes == 0
                        ]
                        
                        valid_loader = DataLoader(
                            valid_dataset,
                            collate_fn=valid_collate,
                            num_workers=self.cfg.train.dataloader.num_worker,
                            batch_sampler=VariableSampler(
                                valid_batches, drop_last=False, use_random_sampler=False
                            ),
                            pin_memory=self.cfg.train.dataloader.pin_memory,
                            prefetch_factor=self.cfg.train.dataloader.prefetch_factor
                        )
                        
                        if self.accelerator.is_main_process:
                            print(f"Validation dataset built successfully with {len(valid_dataset)} samples")
                    else:
                        if self.accelerator.is_main_process:
                            print("Warning: Validation dataset is empty, skipping validation")
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"Warning: Failed to build validation dataset: {e}")
                        import traceback
                        traceback.print_exc()
            elif not use_validation and self.accelerator.is_main_process:
                print("Validation is disabled in config (use_validation=false)")

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            # Check if using new dataset factory pattern (libritts, ljspeech, etc.)
            use_factory_pattern = (
                hasattr(self.cfg.preprocess, "dataset_type")
                and self.cfg.preprocess.dataset_type in ["libritts", "ljspeech"]
            ) or (
                hasattr(self.cfg.train, "use_emilia_dataset")
                and self.cfg.train.use_emilia_dataset
            )
            
            if use_factory_pattern:
                train_dataset = Dataset(cfg=self.cfg, is_valid=False)
            else:
                # Legacy pattern: assume dataset is a list
                if isinstance(self.cfg.dataset, (list, tuple)):
                    train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
                else:
                    # If dataset is dict, try to use first key or use factory pattern
                    train_dataset = Dataset(cfg=self.cfg, is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            # Build validation dataloader
            valid_loader = None
            # Check if validation is enabled in config (default: True)
            use_validation = getattr(self.cfg.preprocess, "use_validation", True)
            
            if use_factory_pattern and use_validation:
                try:
                    if self.accelerator.is_main_process:
                        print("Building validation dataset...")
                    valid_dataset = Dataset(cfg=self.cfg, is_valid=True)
                    valid_collate = Collator(self.cfg)
                    
                    if len(valid_dataset) > 0:
                        valid_loader = DataLoader(
                            valid_dataset,
                            collate_fn=valid_collate,
                            batch_size=self.cfg.train.batch_size,
                            num_workers=self.cfg.train.dataloader.num_worker,
                            pin_memory=self.cfg.train.dataloader.pin_memory,
                            shuffle=False,
                        )
                        if self.accelerator.is_main_process:
                            print(f"Validation dataset built successfully with {len(valid_dataset)} samples")
                    else:
                        if self.accelerator.is_main_process:
                            print("Warning: Validation dataset is empty, skipping validation")
                except Exception as e:
                    if self.accelerator.is_main_process:
                        print(f"Warning: Failed to build validation dataset: {e}")
                        import traceback
                        traceback.print_exc()
            elif not use_validation and self.accelerator.is_main_process:
                print("Validation is disabled in config (use_validation=false)")
            
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer=self.optimizer,
            # num_warmup_steps=self.cfg.train.lr_warmup_steps,  # TODO: need to check wheather need to multiply by num_processes
            num_warmup_steps=self.cfg.train.lr_warmup_steps
            * self.accelerator.num_processes,
        )
        return lr_scheduler

    def _build_criterion(self):
        criteria = dict()
        criteria["l1_loss"] = torch.nn.L1Loss(reduction="mean")
        criteria["l2_loss"] = torch.nn.MSELoss(reduction="mean")
        criteria["ce_loss"] = torch.nn.CrossEntropyLoss(reduction="none")
        return criteria

    def write_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def write_valid_summary(self, losses, stats):
        for key, value in losses.items():
            self.sw.add_scalar(key, value, self.step)

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self, batch):
        raise NotImplementedError

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()

        epoch_sum_loss: float = 0.0
        epoch_losses: dict = {}
        epoch_step: int = 0
        ema_loss = None

        # Calculate the number of batches to skip, only skip when resume_skip_steps is enabled in the configuration
        steps_to_skip = 0
        if (
            hasattr(self.cfg.train, "resume_skip_steps")
            and self.cfg.train.resume_skip_steps
            and hasattr(self, "step")
            and self.step > 0
        ):
            steps_to_skip = self.step
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Resume skip steps enabled, skipping first {steps_to_skip} steps..."
                )

            # If dynamic batch size is used, we need to modify batch_sampler
            if self.cfg.train.use_dynamic_batchsize:
                if hasattr(self.train_dataloader, "batch_sampler"):
                    self.train_dataloader.batch_sampler.skip_steps(steps_to_skip)
            # If normal batch size is used, we need to modify sampler
            else:
                if hasattr(self.train_dataloader, "sampler"):
                    # Calculate the number of samples to skip
                    samples_to_skip = (
                        steps_to_skip
                        * self.cfg.train.batch_size
                        * self.accelerator.num_processes
                    )
                    if isinstance(
                        self.train_dataloader.sampler,
                        torch.utils.data.DistributedSampler,
                    ):
                        self.train_dataloader.sampler.set_start_index(samples_to_skip)
                    elif hasattr(self.train_dataloader.sampler, "skip_samples"):
                        self.train_dataloader.sampler.skip_samples(samples_to_skip)

        # Track the current number of batches processed
        current_batch = steps_to_skip

        for batch in self.train_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Record step start time for performance monitoring (after data loading)
            step_start_time = time.time()

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, training_stats = self._train_step(batch)
            self.batch_count += 1
            ema_loss = (
                0.98 * ema_loss + 0.02 * self.current_loss
                if ema_loss is not None
                else self.current_loss
            )
            # Update info for each step
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss = total_loss
                for key, value in train_losses.items():
                    epoch_losses[key] = value

                if self.accelerator.is_main_process and isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Steps/Train {}".format(key): loss},
                            step=self.step,
                        )

                # Log learning rate(s)
                if self.accelerator.is_main_process:
                    if isinstance(self.optimizer, dict):
                        for key in self.optimizer.keys():
                            lr = self.optimizer[key].param_groups[0]["lr"]
                            self.accelerator.log(
                                {"Steps/LR_{}".format(key): lr},
                                step=self.step,
                            )
                    else:
                        lr = self.optimizer.param_groups[0]["lr"]
                        self.accelerator.log(
                            {"Steps/LR": lr},
                            step=self.step,
                        )

                # Log gradient norm
                if self.accelerator.sync_gradients:
                    if self.accelerator.is_main_process:
                        if isinstance(self.model, dict):
                            for key in self.model.keys():
                                total_norm = 0.0
                                for p in self.model[key].parameters():
                                    if p.grad is not None:
                                        param_norm = p.grad.data.norm(2)
                                        total_norm += param_norm.item() ** 2
                                total_norm = total_norm**0.5
                                self.accelerator.log(
                                    {"Steps/GradNorm_{}".format(key): total_norm},
                                    step=self.step,
                                )
                        else:
                            total_norm = 0.0
                            for p in self.model.parameters():
                                if p.grad is not None:
                                    param_norm = p.grad.data.norm(2)
                                    total_norm += param_norm.item() ** 2
                            total_norm = total_norm**0.5
                            self.accelerator.log(
                                {"Steps/GradNorm": total_norm},
                                step=self.step,
                            )

                # Log training stats if available
                if self.accelerator.is_main_process and isinstance(training_stats, dict):
                    for key, value in training_stats.items():
                        self.accelerator.log(
                            {"Steps/Stats_{}".format(key): value},
                            step=self.step,
                        )

                if (
                    self.accelerator.is_main_process
                    and self.batch_count
                    % (10 * self.cfg.train.gradient_accumulation_step)
                    == 0
                ):
                    self.echo_log(train_losses, mode="Training")

                self.step += 1
                epoch_step += 1

                # Record step time for performance monitoring
                step_time = time.time() - step_start_time
                self.time_window.append(step_time)
                
                # Log performance metrics
                if self.accelerator.is_main_process:
                    self.accelerator.log(
                        {"Steps/StepTime": step_time},
                        step=self.step,
                    )
                    if len(self.time_window._values) > 0:
                        self.accelerator.log(
                            {"Steps/AvgStepTime": self.time_window.average},
                            step=self.step,
                        )

                if self.step % self.cfg.train.save_checkpoints_steps == 0:
                    self.save_checkpoint()

                if self.accelerator.is_main_process:
                    if self.step % 100 == 0:
                        print(f"EMA Loss: {ema_loss:.6f}")

            current_batch += 1

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def save_checkpoint(self):
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)

            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))

            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))
            self.accelerator.save_state(path, safe_serialization=False)
            self.logger.info("Finished saving state.")

            if (
                hasattr(self.cfg.train, "save_checkpoints_backup_steps")
                and self.step % self.cfg.train.save_checkpoints_backup_steps == 0
            ):
                try:
                    backup_path = os.path.join(
                        self.checkpoint_backup_dir, checkpoint_filename
                    )
                    shutil.copytree(path, backup_path)
                    self.logger.info("Saving backup state to {}...".format(backup_path))
                except Exception as e:
                    self.logger.error("Failed to save backup state: {}".format(e))

    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        # if self.accelerator.is_main_process:
        #     self._dump_cfg(self.config_save_path)

        # self.optimizer.zero_grad()

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        while self.epoch < self.max_epoch:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 50)
                max_epoch_display = "∞" if self.max_epoch == float("inf") else str(int(self.max_epoch))
                self.logger.info("Epoch {}/{} (Steps per epoch: {}): ".format(
                    self.epoch + 1, max_epoch_display, self.steps_per_epoch
                ))

            # Do training & validating epoch
            train_total_loss, train_losses = self._train_epoch()
            if isinstance(train_losses, dict):
                for key, loss in train_losses.items():
                    if self.accelerator.is_main_process:
                        self.logger.info("  |- Train/{} Loss: {:.6f}".format(key, loss))
                    if self.accelerator.is_main_process:
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.epoch,
                        )

            # Run validation if valid_dataloader exists
            if self.valid_dataloader is not None:
                valid_total_loss, valid_losses = self._valid_epoch()
                if isinstance(valid_losses, dict):
                    for key, loss in valid_losses.items():
                        if self.accelerator.is_main_process:
                            self.logger.info("  |- Valid/{} Loss: {:.6f}".format(key, loss))
                        if self.accelerator.is_main_process:
                            self.accelerator.log(
                                {"Epoch/Valid {} Loss".format(key): loss},
                                step=self.epoch,
                            )
            else:
                valid_total_loss, valid_losses = 0.0, 0.0

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss))
            if self.accelerator.is_main_process:
                self.accelerator.log(
                    {
                        "Epoch/Train Loss": train_total_loss,
                        "Epoch/Valid Loss": valid_total_loss,
                    },
                    step=self.epoch,
                )

            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

            # Update info for each epoch
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}".format(self.epoch, self.step),
                )
            )
        self.accelerator.end_training()

    def _valid_epoch(self):
        """Validation epoch for T2S model.

        Aggregates losses and stats across all validation batches.
        """
        self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = {}
        epoch_stats = {}
        num_batches = 0

        for batch in self.valid_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss += total_loss
            num_batches += 1
            if (
                self.accelerator.is_main_process
                and num_batches
                % (10 * self.cfg.train.gradient_accumulation_step)
                == 0
            ):
                self.echo_log(valid_losses, mode="Validing")
            # Aggregate losses
            if isinstance(valid_losses, dict):
                for key, value in valid_losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

            # Aggregate stats
            if isinstance(valid_stats, dict):
                for key, value in valid_stats.items():
                    if key not in epoch_stats:
                        epoch_stats[key] = value
                    else:
                        epoch_stats[key] += value

        # Average over batches
        if num_batches > 0:
            epoch_sum_loss = epoch_sum_loss / num_batches
            for key in epoch_losses:
                epoch_losses[key] = epoch_losses[key] / num_batches
            for key in epoch_stats:
                epoch_losses[key] = epoch_stats[key] / num_batches  # Add stats to losses for logging

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

    def echo_log(self, losses, mode="Training"):
        max_epoch_display = "∞" if self.max_epoch == float("inf") else str(int(self.max_epoch))
        if mode == "Training":
            message = [
                "{} - Epoch {} / {} (Step {} / {}): [{:.5f} s/step]".format(
                    mode, self.epoch + 1, max_epoch_display, self.step, self.steps_per_epoch, self.time_window.average
                )
            ]
        else:
            message = [
                "{} - Epoch {} / {} (Step {} / {}): [{:.5f} s/step]".format(
                    mode, self.epoch + 1, max_epoch_display, self.step, self.valid_steps_per_epoch, self.time_window.average
                )
            ]
        # print(self.valid_steps_per_epoch)
        for key in sorted(losses.keys()):
            if isinstance(losses[key], dict):
                for k, v in losses[key].items():
                    message.append(
                        str(k).split("/")[-1] + "=" + str(round(float(v), 5))
                    )
            else:
                message.append(str(key) + "=" + str(round(float(losses[key]), 5)))
        self.logger.info(", ".join(message))
