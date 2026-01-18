# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
T2S Curriculum Trainer for Small Datasets

This trainer extends T2STrainer with curriculum learning support.
It automatically updates the model's training step to adjust difficulty.
"""

import torch
import torch.nn.functional as F

from models.tts.maskgct.t2s_trainer import T2STrainer
from models.tts.maskgct.maskgct_t2s_curriculum import MaskGCT_T2S_Curriculum


class T2SCurriculumTrainer(T2STrainer):
    """
    Trainer for T2S model with curriculum learning.

    Extends T2STrainer to:
    1. Use MaskGCT_T2S_Curriculum model
    2. Update model's current_step during training
    3. Log curriculum stage information
    """

    def _build_model(self):
        """Build T2S model with curriculum learning"""
        model = MaskGCT_T2S_Curriculum(cfg=self.cfg.model.t2s_model)

        # Configure gradient checkpointing
        gradient_checkpointing = getattr(self.cfg.model.t2s_model, 'gradient_checkpointing', False)
        if hasattr(model.diff_estimator, 'gradient_checkpointing'):
            model.diff_estimator.gradient_checkpointing = gradient_checkpointing
        elif hasattr(model.diff_estimator, 'gradient_checkpointing_enable'):
            if gradient_checkpointing:
                model.diff_estimator.gradient_checkpointing_enable()
        elif hasattr(model.diff_estimator, 'enable_gradient_checkpointing'):
            if gradient_checkpointing:
                model.diff_estimator.enable_gradient_checkpointing()

        return model

    def _train_step(self, batch):
        """Training step with curriculum step update"""
        # Update model's current step for curriculum scheduling
        if hasattr(self.model, 'module'):
            # DDP wrapped
            self.model.module.set_step(self.step)
        else:
            self.model.set_step(self.step)

        # Call parent's train step
        total_loss, train_losses, train_stats = super()._train_step(batch)

        # Add curriculum information to stats
        if hasattr(self.model, 'module'):
            curriculum_params = self.model.module.get_curriculum_params()
        else:
            curriculum_params = self.model.get_curriculum_params()

        train_stats["curriculum_stage"] = curriculum_params["stage"]
        train_stats["curriculum_progress"] = curriculum_params["progress"]
        train_stats["mask_ratio_min"] = curriculum_params["mask_ratio_range"][0]
        train_stats["mask_ratio_max"] = curriculum_params["mask_ratio_range"][1]
        train_stats["prompt_ratio_min"] = curriculum_params["prompt_ratio_range"][0]
        train_stats["prompt_ratio_max"] = curriculum_params["prompt_ratio_range"][1]
        train_stats["cfg_dropout"] = curriculum_params["cfg_dropout"]

        return total_loss, train_losses, train_stats

    def _valid_step(self, batch):
        """Validation step with curriculum step update"""
        # Update model's current step
        if hasattr(self.model, 'module'):
            self.model.module.set_step(self.step)
        else:
            self.model.set_step(self.step)

        # Call parent's valid step
        total_loss, valid_losses, valid_stats = super()._valid_step(batch)

        # Add curriculum information
        if hasattr(self.model, 'module'):
            curriculum_params = self.model.module.get_curriculum_params()
        else:
            curriculum_params = self.model.get_curriculum_params()

        valid_stats["curriculum_stage"] = curriculum_params["stage"]

        return total_loss, valid_losses, valid_stats
