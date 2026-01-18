# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
T2S Fine-tuning Trainer for MaskGCT

This trainer fine-tunes a pre-trained T2S model (with 8192 codebook) to work with
a smaller codebook (512). It freezes the DiffLlamaPrefix backbone and only trains
the codebook-related parameters (cond_emb and to_logit).

Key features:
1. Loads pre-trained teacher model with large codebook
2. Initializes student model with small codebook
3. Copies DiffLlamaPrefix parameters from teacher
4. Freezes all parameters except codebook-related ones
5. Supports mixed training modes for better alignment
"""

import torch
import torch.nn.functional as F
import os
import safetensors

from models.tts.maskgct.t2s_trainer import T2STrainer
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S


class T2SFineTuneTrainer(T2STrainer):
    """
    Fine-tuning trainer for T2S model with reduced codebook size.

    This trainer:
    1. Loads a pre-trained teacher model (large codebook)
    2. Creates a student model (small codebook)
    3. Transfers DiffLlamaPrefix weights from teacher to student
    4. Freezes all parameters except codebook-related ones
    5. Fine-tunes only the codebook embeddings and output projection
    """

    def __init__(self, args, cfg):
        # Initialize parent class
        super(T2SFineTuneTrainer, self).__init__(args, cfg)

        # Load teacher model weights after model initialization
        self._load_teacher_weights()

        # Freeze parameters that should not be trained
        self._freeze_pretrained_parameters()

        # Log trainable parameters
        self._log_trainable_parameters()

    def _load_teacher_weights(self):
        """
        Load pre-trained teacher model weights and transfer to student model.

        The teacher model has a larger codebook (8192), so we only copy the
        parameters that are not codebook-dependent:
        - DiffLlamaPrefix (diff_estimator)
        - phone_emb
        - mask_emb
        """
        teacher_path = getattr(self.cfg.model, "teacher_model_path", None)
        if not teacher_path or not os.path.exists(teacher_path):
            self.accelerator.print(
                f"Warning: Teacher model path not found: {teacher_path}. "
                "Training from scratch."
            )
            return

        self.accelerator.print(f"Loading teacher model from: {teacher_path}")

        # Load teacher weights
        if teacher_path.endswith(".safetensors"):
            teacher_state_dict = {}
            with safetensors.safe_open(teacher_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    teacher_state_dict[key] = f.get_tensor(key)
        else:
            checkpoint = torch.load(teacher_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                teacher_state_dict = checkpoint["model"]
            else:
                teacher_state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        teacher_state_dict = {
            k.replace("module.", ""): v for k, v in teacher_state_dict.items()
        }

        # Get unwrapped student model (without DDP wrapper) to access correct parameter names
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        student_state_dict = unwrapped_model.state_dict()

        # Copy compatible parameters (those not dependent on codebook size)
        transferred_params = []
        skipped_params = []

        for name, param in teacher_state_dict.items():
            if name in student_state_dict:
                # Check if shapes match
                if student_state_dict[name].shape == param.shape:
                    student_state_dict[name].copy_(param)
                    transferred_params.append(name)
                else:
                    # Skip codebook-related parameters with different shapes
                    skipped_params.append(f"{name} (shape mismatch: "
                                        f"teacher {param.shape} vs "
                                        f"student {student_state_dict[name].shape})")
            else:
                skipped_params.append(f"{name} (not in student model)")

        # Load the updated state dict into unwrapped model
        unwrapped_model.load_state_dict(student_state_dict)

        # Log transfer statistics
        self.accelerator.print(f"\nParameter Transfer Summary:")
        self.accelerator.print(f"  Transferred: {len(transferred_params)} parameters")
        self.accelerator.print(f"  Skipped: {len(skipped_params)} parameters")

        if self.accelerator.is_main_process:
            self.accelerator.print(f"\nTransferred parameters:")
            for name in transferred_params[:10]:  # Show first 10
                self.accelerator.print(f"  - {name}")
            if len(transferred_params) > 10:
                self.accelerator.print(f"  ... and {len(transferred_params) - 10} more")

            self.accelerator.print(f"\nSkipped parameters:")
            for name in skipped_params[:10]:  # Show first 10
                self.accelerator.print(f"  - {name}")
            if len(skipped_params) > 10:
                self.accelerator.print(f"  ... and {len(skipped_params) - 10} more")

    def _freeze_pretrained_parameters(self):
        """
        Freeze all parameters except codebook-related ones.

        Trainable parameters (codebook-dependent):
        - cond_emb: Embedding layer for semantic tokens
        - to_logit: Output projection layer

        Frozen parameters (transferred from teacher):
        - diff_estimator: DiffLlamaPrefix backbone
        - phone_emb: Phone embedding layer
        - mask_emb: Mask token embedding
        """
        frozen_params = []
        trainable_params = []

        # Use unwrapped model to get correct parameter names (without DDP 'module.' prefix)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        for name, param in unwrapped_model.named_parameters():
            # Only train codebook-related parameters
            if "cond_emb" in name or "to_logit" in name:
            # if False:
                param.requires_grad = True
                trainable_params.append(name)
            else:
                param.requires_grad = False
                frozen_params.append(name)

        self.accelerator.print(f"\nParameter Freeze Summary:")
        self.accelerator.print(f"  Trainable: {len(trainable_params)} parameters")
        self.accelerator.print(f"  Frozen: {len(frozen_params)} parameters")

    def _log_trainable_parameters(self):
        """Log trainable parameter statistics."""
        # Use unwrapped model for accurate parameter counting
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        total_params = sum(p.numel() for p in unwrapped_model.parameters())
        trainable_params = sum(
            p.numel() for p in unwrapped_model.parameters() if p.requires_grad
        )
        frozen_params = total_params - trainable_params

        self.accelerator.print(f"\nModel Parameter Statistics:")
        self.accelerator.print(f"  Total parameters: {total_params:,}")
        self.accelerator.print(f"  Trainable parameters: {trainable_params:,} "
                             f"({100 * trainable_params / total_params:.2f}%)")
        self.accelerator.print(f"  Frozen parameters: {frozen_params:,} "
                             f"({100 * frozen_params / total_params:.2f}%)")

        # Log trainable parameter names
        if self.accelerator.is_main_process:
            self.accelerator.print(f"\nTrainable parameters:")
            for name, param in unwrapped_model.named_parameters():
                if param.requires_grad:
                    self.accelerator.print(f"  - {name}: {param.shape}")
