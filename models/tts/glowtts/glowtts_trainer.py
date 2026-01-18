# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GlowTTS Trainer

Trainer class for GlowTTS model. Inherits from TTSTrainer and implements
GlowTTS-specific training logic including MLE loss and duration loss.
"""

import os
import torch
import torch.nn as nn
from tqdm import tqdm

from models.tts.base.tts_trainer import TTSTrainer
from models.tts.glowtts.glowtts import GlowTTS, GlowTTSLoss
from models.tts.glowtts.glowtts_dataset import GlowTTSDataset, GlowTTSCollator
from optimizer.optimizers import NoamLR


class GlowTTSTrainer(TTSTrainer):
    """
    Trainer for GlowTTS model.

    Implements the training loop with MLE loss for the flow
    and duration prediction loss.
    """

    def __init__(self, args, cfg):
        TTSTrainer.__init__(self, args, cfg)
        self.cfg = cfg
        self.ddi_done = False

    def _build_dataset(self):
        """Build dataset and collator."""
        return GlowTTSDataset, GlowTTSCollator

    def _build_model(self):
        """Build GlowTTS model with DDI support."""
        # Check if DDI is enabled in config
        ddi_enabled = getattr(self.cfg.train, "ddi", True)
        self.model = GlowTTS(self.cfg, ddi=ddi_enabled)
        return self.model

    def _build_criterion(self):
        """Build loss function."""
        return GlowTTSLoss(self.cfg)

    def _build_optimizer(self):
        """Build optimizer."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            **self.cfg.train.adam
        )
        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler."""
        scheduler = NoamLR(
            self.optimizer,
            **self.cfg.train.lr_scheduler
        )
        return scheduler

    def _init_ddi(self):
        """
        Run DDI (Data-Dependent Initialization) for ActNorm layers.

        This runs one forward pass to initialize ActNorm parameters
        based on the data statistics, which helps stabilize training.
        """
        if self.ddi_done:
            return

        self.accelerator.print("Running DDI (Data-Dependent Initialization)...")

        # Set DDI mode
        self.model.set_ddi(True)
        self.model.train()

        # Run one forward pass to initialize ActNorm
        for batch in self.train_dataloader:
            with torch.no_grad():
                _ = self.model(batch, gen=False)
            break

        # Disable DDI mode (ActNorm is now initialized)
        self.model.set_ddi(False)
        self.ddi_done = True

        self.accelerator.print("DDI initialization completed.")

    def get_state_dict(self):
        """Get state dict for checkpointing."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
            "ddi_done": self.ddi_done,
        }
        return state_dict

    def load_model(self, checkpoint):
        """Load model from checkpoint."""
        super().load_model(checkpoint)
        # Check if DDI was already done when checkpoint was saved
        if "ddi_done" in checkpoint:
            self.ddi_done = checkpoint["ddi_done"]
        else:
            # If loading an old checkpoint, assume DDI is done
            self.ddi_done = True

    def _write_summary(self, losses, stats):
        """Write training summary to tensorboard."""
        for key, value in losses.items():
            self.sw.add_scalar(f"train/{key}", value, self.step)
        lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        self.sw.add_scalar("learning_rate", lr, self.step)

    def _write_valid_summary(self, losses, stats):
        """Write validation summary to tensorboard."""
        for key, value in losses.items():
            self.sw.add_scalar(f"val/{key}", value, self.step)

    def _train_epoch(self):
        """
        Training epoch for GlowTTS.

        Returns average loss over one epoch.
        """
        # Run DDI initialization before first training epoch
        if not self.ddi_done:
            self._init_ddi()

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
                loss, train_losses = self._train_step(batch)
                self.accelerator.backward(loss)

                # Only clip gradients and step optimizer/scheduler when accumulation is complete
                if self.accelerator.sync_gradients:
                    grad_clip_thresh = self.cfg.train.grad_clip_thresh
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), grad_clip_thresh
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

            self.batch_count += 1

            # Update info for each step (when gradients are actually applied)
            if self.accelerator.sync_gradients:
                epoch_sum_loss += loss
                for key, value in train_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

                self.accelerator.log(
                    {
                        "Step/Train Loss": loss,
                        "Step/Learning Rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.step,
                )
                self.step += 1
                epoch_step += 1

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
        """
        Single training step.

        Args:
            data: Batch data from dataloader

        Returns:
            total_loss: Total loss value
            train_losses: Dictionary of individual losses
        """
        train_losses = {}

        # Forward pass
        outputs = self.model(data, gen=False)

        # Compute losses
        train_losses = self.criterion(data, outputs)

        total_loss = train_losses["loss"]

        # Convert to float for logging
        for key, value in train_losses.items():
            train_losses[key] = value.item()

        return total_loss, train_losses

    @torch.no_grad()
    def _valid_step(self, data):
        """
        Single validation step.

        Args:
            data: Batch data from dataloader

        Returns:
            total_loss: Total loss value
            valid_losses: Dictionary of individual losses
            valid_stats: Empty dict (for compatibility)
        """
        valid_stats = {}

        # Forward pass
        outputs = self.model(data, gen=False)

        # Compute losses
        valid_losses = self.criterion(data, outputs)

        total_valid_loss = valid_losses["loss"]

        # Convert to float for logging
        for key, value in valid_losses.items():
            valid_losses[key] = value.item()

        return total_valid_loss, valid_losses, valid_stats
