# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from tqdm import tqdm

from models.tts.base import TTSTrainer
from models.tts.tacotron2.tacotron2 import Tacotron2, Tacotron2Loss
from models.tts.tacotron2.tacotron2_dataset import Tacotron2Dataset, Tacotron2Collator


class Tacotron2Trainer(TTSTrainer):
    """Trainer for Tacotron2 model."""

    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)
        self.cfg = cfg

    def _build_dataset(self):
        return Tacotron2Dataset, Tacotron2Collator

    def _build_model(self):
        self.model = Tacotron2(self.cfg)
        return self.model

    def _build_criterion(self):
        return Tacotron2Loss(self.cfg)

    def _build_optimizer(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.train.adam.lr,
            betas=(self.cfg.train.adam.betas[0], self.cfg.train.adam.betas[1]),
            eps=self.cfg.train.adam.eps,
            weight_decay=self.cfg.train.adam.weight_decay,
        )
        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler.

        Uses exponential decay scheduler by default.
        """
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=self.cfg.train.lr_scheduler.gamma,
        )
        return scheduler

    def _write_summary(self, losses, stats):
        """Write training summary to tensorboard."""
        for key, value in losses.items():
            self.sw.add_scalar("train/" + key, value, self.step)
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar("learning_rate", lr, self.step)

    def _write_valid_summary(self, losses, stats):
        """Write validation summary to tensorboard."""
        for key, value in losses.items():
            self.sw.add_scalar("val/" + key, value, self.step)

    def get_state_dict(self):
        """Get state dict for saving checkpoint."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict

    def _train_epoch(self):
        """Training epoch.

        Returns:
            Average loss over one epoch.
        """
        self.model.train()
        epoch_sum_loss: float = 0.0
        epoch_step: int = 0
        epoch_losses: dict = {}

        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, train_stats = self._train_step(batch)
                self.accelerator.backward(total_loss)

                # Only clip gradients and step optimizer/scheduler when accumulation is complete
                if self.accelerator.sync_gradients:
                    grad_clip_thresh = self.cfg.train.grad_clip_thresh
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_thresh)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.batch_count += 1

            # Update info for each step (when gradients are actually applied)
            if self.accelerator.sync_gradients:
                epoch_sum_loss += total_loss.item()
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Train Loss": total_loss.item(),
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

        # Update learning rate at the end of epoch
        self.scheduler.step()

        self.accelerator.wait_for_everyone()

        epoch_sum_loss = (
            epoch_sum_loss
            / len(self.train_dataloader)
            * self.cfg.train.gradient_accumulation_step
        )

        for key in epoch_losses.keys():
            epoch_losses[key] = (
                epoch_losses[key]
                / len(self.train_dataloader)
                * self.cfg.train.gradient_accumulation_step
            )

        return epoch_sum_loss, epoch_losses

    def _train_step(self, data):
        """Training step.

        Args:
            data: Batch data from dataloader.

        Returns:
            Tuple of (total_loss, loss_dict, stats_dict)
        """
        train_losses = {}
        train_stats = {}

        # Forward pass
        predictions = self.model(data)

        # Calculate loss
        losses = self.criterion(data, predictions)

        total_loss = losses["loss"]
        for key, value in losses.items():
            train_losses[key] = value.item()

        return total_loss, train_losses, train_stats

    @torch.no_grad()
    def _valid_step(self, data):
        """Validation step.

        Args:
            data: Batch data from dataloader.

        Returns:
            Tuple of (total_loss, loss_dict, stats_dict)
        """
        valid_losses = {}
        valid_stats = {}

        # Forward pass
        predictions = self.model(data)

        # Calculate loss
        losses = self.criterion(data, predictions)

        total_loss = losses["loss"]
        for key, value in losses.items():
            valid_losses[key] = value.item()

        return total_loss, valid_losses, valid_stats
