# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
T2S Fine-Tuning Trainer for MaskGCT

This trainer implements fine-tuning of the T2S model with a reduced codebook size.
The teacher model (with 8192 codebook size) is frozen, and only the codebook-dependent
layers (cond_emb and to_logit) of the student model (with 512 codebook size) are trained.

Key features:
1. Load pretrained teacher model with DiffLlamaPrefix parameters
2. Initialize student model with reduced codebook size
3. Copy teacher parameters to student (except codebook-dependent layers)
4. Freeze all parameters except cond_emb and to_logit layers
5. Train only the codebook-dependent layers to adapt to the new codebook size

Trainable layers:
- cond_emb: Semantic token embedding (512 x hidden_size) ~786K params
- to_logit: Final projection (hidden_size x 512) ~786K params
Total: ~1.57M trainable parameters (~0.8% of total)
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import safetensors

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from models.base.base_trainer import BaseTrainer


class T2SFineTuneTrainer(BaseTrainer):
    """Trainer for fine-tuning MaskGCT-T2S model with reduced codebook size.
    
    This trainer loads a pretrained teacher model and fine-tunes only the 
    codebook-dependent layers (cond_emb and to_logit) to work with a 
    reduced codebook size (8192 -> 512).
    """
    
    # def _init_accelerator(self):
    #     """Override to enable find_unused_parameters for DDP with frozen parameters"""
    #     import accelerate
    #     from accelerate import DistributedDataParallelKwargs
    #     from accelerate.utils import ProjectConfiguration
    #     import os
        
    #     self.exp_dir = os.path.join(
    #         os.path.abspath(self.cfg.log_dir), self.args.exp_name
    #     )
    #     project_config = ProjectConfiguration(
    #         project_dir=self.exp_dir,
    #         logging_dir=os.path.join(self.exp_dir, "log"),
    #     )
        
    #     # Enable find_unused_parameters for DDP since we freeze most parameters
    #     ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
    #     self.accelerator = accelerate.Accelerator(
    #         gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
    #         log_with=self.cfg.train.tracker,
    #         project_config=project_config,
    #         kwargs_handlers=[ddp_kwargs]
    #     )
        
    #     if self.accelerator.is_main_process:
    #         os.makedirs(project_config.project_dir, exist_ok=True)
    #         os.makedirs(project_config.logging_dir, exist_ok=True)
    #     with self.accelerator.main_process_first():
    #         self.accelerator.init_trackers(self.args.exp_name)
    
    def __init__(self, args, cfg):
        super(T2SFineTuneTrainer, self).__init__(args, cfg)
        
        # Setup pretrained models for feature extraction
        self._build_input_models()
        
        # Load teacher model for initialization
        self._load_teacher_model()
    
    def _build_model(self):
        """Build student T2S model with reduced codebook size"""
        # Build student model with reduced codebook size (512)
        student_model = MaskGCT_T2S(cfg=self.cfg.model.student_model)

        # Configure gradient checkpointing for memory efficiency
        gradient_checkpointing = getattr(self.cfg.model.student_model, 'gradient_checkpointing', False)
        if hasattr(student_model.diff_estimator, 'gradient_checkpointing'):
            student_model.diff_estimator.gradient_checkpointing = gradient_checkpointing
        elif hasattr(student_model.diff_estimator, 'gradient_checkpointing_enable'):
            if gradient_checkpointing:
                student_model.diff_estimator.gradient_checkpointing_enable()
        elif hasattr(student_model.diff_estimator, 'enable_gradient_checkpointing'):
            if gradient_checkpointing:
                student_model.diff_estimator.enable_gradient_checkpointing()

        return student_model
    
    def _load_teacher_model(self):
        """Load pretrained teacher model and initialize student model"""
        # Build teacher model with original codebook size (8192)
        self.teacher_model = MaskGCT_T2S(cfg=self.cfg.model.teacher_model)
        
        # Load teacher model weights
        teacher_path = self.cfg.model.teacher_model.pretrained_path
        if teacher_path and os.path.exists(teacher_path):
            self.accelerator.print(f"Loading teacher model from {teacher_path}")
            if teacher_path.endswith(".safetensors"):
                safetensors.torch.load_model(self.teacher_model, teacher_path)
            else:
                checkpoint = torch.load(teacher_path, map_location=self.accelerator.device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.teacher_model.load_state_dict(checkpoint["model"], strict=False)
                else:
                    self.teacher_model.load_state_dict(checkpoint, strict=False)
            self.accelerator.print("Teacher model loaded successfully")
        else:
            self.accelerator.print(f"Warning: Teacher model path not found: {teacher_path}")
        
        # Freeze teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Copy teacher parameters to student (except final layer and embeddings)
        self.accelerator.print("Copying teacher parameters to student model...")
        
        # Copy DiffLlamaPrefix parameters (the backbone)
        # Note: student_dict may have 'module.' prefix if wrapped by DDP
        student_dict = self.model.state_dict()
        teacher_dict = self.teacher_model.state_dict()
        
        # Check if student model is wrapped by DDP (has 'module.' prefix)
        has_module_prefix = any(k.startswith('module.') for k in student_dict.keys())
        
        self.accelerator.print(f"Student model wrapped by DDP: {has_module_prefix}")
        self.accelerator.print("Copying teacher parameters to student model...")
        
        copied_count = 0
        skipped_count = 0
        
        # Copy all parameters except those related to codebook size
        for name, param in teacher_dict.items():
            # Skip parameters that depend on codebook size (need to be trained)
            # cond_emb: embedding layer for semantic tokens (8192 -> 512)
            # to_logit: final projection layer (hidden_size -> 512)
            if 'to_logit' in name or 'cond_emb' in name:
                self.accelerator.print(f"Skipping parameter (codebook-dependent, will be trained): {name}")
                skipped_count += 1
                continue
            
            # Add 'module.' prefix if student is wrapped by DDP
            student_key = f"module.{name}" if has_module_prefix else name
            
            # Copy matching parameters
            if student_key in student_dict:
                if param.shape == student_dict[student_key].shape:
                    student_dict[student_key].copy_(param)
                    copied_count += 1
                    if copied_count <= 10:  # Only print first 10 to avoid spam
                        self.accelerator.print(f"Copied parameter: {name} -> {student_key}")
                else:
                    self.accelerator.print(f"Shape mismatch for {name}: teacher {param.shape} vs student {student_dict[student_key].shape}")
            else:
                self.accelerator.print(f"Parameter not found in student: {student_key}")
        
        self.accelerator.print(f"\nParameter copying summary:")
        self.accelerator.print(f"  Copied: {copied_count} parameters")
        self.accelerator.print(f"  Skipped (codebook-dependent): {skipped_count} parameters")
        
        # Freeze all parameters except cond_emb and to_logit
        self.accelerator.print("\nFreezing all parameters except cond_emb and to_logit layers...")
        trainable_params = 0
        frozen_params = 0
        
        for name, param in self.model.named_parameters():
            # Train both cond_emb (semantic token embedding) and to_logit (final projection)
            # These are the only layers that depend on codebook size
            if 'to_logit' in name or 'cond_emb' in name:
                param.requires_grad = True
                trainable_params += param.numel()
                self.accelerator.print(f"Trainable: {name} - {param.shape}")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        self.accelerator.print(f"\nTotal trainable parameters: {trainable_params:,}")
        self.accelerator.print(f"Total frozen parameters: {frozen_params:,}")
        self.accelerator.print(f"Percentage trainable: {100 * trainable_params / (trainable_params + frozen_params):.2f}%")
        
        # Clean up teacher model to save memory (no longer needed after parameter copying)
        self.accelerator.print("\nCleaning up teacher model to save memory...")
        del self.teacher_model
        torch.cuda.empty_cache()
        self.accelerator.print("Teacher model removed from memory")
    
    def _build_input_models(self):
        """Build pretrained models for extracting semantic tokens"""
        # Load teacher's semantic codec for extracting semantic tokens from speech
        if hasattr(self.cfg.model, "semantic_codec") and hasattr(self.cfg.model.semantic_codec, "pretrained_path"):
            from models.codec.kmeans.repcodec_model import RepCodec
            self.semantic_codec = RepCodec(cfg=self.cfg.model.semantic_codec)
            
            # Load pretrained weights
            pretrained_path = self.cfg.model.semantic_codec.pretrained_path
            if pretrained_path:
                if os.path.isdir(pretrained_path):
                    # Accelerate checkpoint format
                    model_path = os.path.join(pretrained_path, "model.safetensors")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(pretrained_path, "pytorch_model.bin")
                else:
                    model_path = pretrained_path
                
                if os.path.exists(model_path):
                    self.accelerator.print(f"Loading semantic codec from {model_path}")
                    if model_path.endswith(".safetensors"):
                        safetensors.torch.load_model(self.semantic_codec, model_path)
                    else:
                        checkpoint = torch.load(model_path, map_location=self.accelerator.device)
                        if isinstance(checkpoint, dict) and "model" in checkpoint:
                            self.semantic_codec.load_state_dict(checkpoint["model"], strict=False)
                        else:
                            self.semantic_codec.load_state_dict(checkpoint, strict=False)
                    self.accelerator.print("Semantic codec loaded successfully")
            
            self.semantic_codec.eval()
            self.semantic_codec.to(self.accelerator.device)
            
            # # Freeze semantic codec
            # for param in self.semantic_codec.parameters():
            #     param.requires_grad = False
            
            # Load semantic model (w2v-bert-2.0) for feature extraction
            from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
            self.semantic_model.eval()
            self.semantic_model.to(self.accelerator.device)
            
            # Freeze semantic model
            for param in self.semantic_model.parameters():
                param.requires_grad = False
            
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
            
            # Load normalization statistics
            if hasattr(self.cfg.model.semantic_codec, "representation_stat_mean_var_path"):
                stat_path = self.cfg.model.semantic_codec.representation_stat_mean_var_path
                if os.path.exists(stat_path):
                    stat = torch.load(stat_path, weights_only=False)
                    self.semantic_mean = stat["mean"].clone().detach().to(self.accelerator.device)
                    self.semantic_std = torch.sqrt(
                        stat["var"].clone().detach()
                    ).to(self.accelerator.device)

                else:
                    self.accelerator.print(f"Warning: Stat file not found: {stat_path}")
                    self.semantic_mean = None
                    self.semantic_std = None
            else:
                self.semantic_mean = None
                self.semantic_std = None
    
    @torch.no_grad()
    def _extract_semantic_tokens(self, batch, output_layer=17):
        """Extract semantic tokens from pre-processed features using teacher's codec"""
        if not hasattr(self, 'semantic_codec'):
            return None

        # Check if pre-processed semantic features are available
        if "semantic_hidden_states" in batch:
            # Use pre-processed hidden states from dataset (most efficient)
            feat = batch["semantic_hidden_states"].to(self.accelerator.device)  # [B, T, D]
        elif "semantic_model_input_features" in batch and "semantic_model_attention_mask" in batch:
            # Use pre-processed features from dataset
            input_features = batch["semantic_model_input_features"]  # [B, T, 80]
            attention_mask = batch["semantic_model_attention_mask"]  # [B, T]
            
            # Extract semantic features using pre-built model
            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            feat = outputs.hidden_states[output_layer]  # [B, T, D]
        else:
            # Fallback: extract features manually (for backward compatibility)
            wavs_16k = batch.get("wav_16000", None)
            if wavs_16k is None:
                raise ValueError("wav_16000 is required for semantic token extraction")

            # Process audio
            inputs = self.processor(
                wavs_16k.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            input_features = inputs["input_features"].to(self.accelerator.device)
            attention_mask = inputs["attention_mask"].to(self.accelerator.device)

            # Extract features
            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            feat = outputs.hidden_states[output_layer]  # [B, T, D]

        # Normalize
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        # Quantize to semantic tokens using teacher's codec
        semantic_tokens, _ = self.semantic_codec.quantize(feat)  # [B, T]
        return semantic_tokens
    
    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)
    
    def _train_step(self, batch):
        """Training step for fine-tuning.

        This method:
        1. Extracts semantic tokens from audio using teacher's codec (8192 codebook)
        2. Gets phone IDs from the batch
        3. Runs forward pass through student model (512 codebook)
        4. Computes cross-entropy loss on masked tokens
        5. Only updates the to_logit layer parameters
        """
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # Get semantic tokens (ground truth) from teacher's codec
        semantic_tokens = batch.get("semantic_code", None)  # [B, T]
        if semantic_tokens is None:
            # Extract from batch data if not provided
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")

        # Get phone IDs (text tokens)
        phone_id = batch.get("phone_id", None)  # [B, T_phone]
        if phone_id is None:
            raise ValueError("phone_id is required for T2S training")

        # Get masks
        semantic_mask = batch.get("semantic_mask", None)  # [B, T], 1 for valid, 0 for padding
        if semantic_mask is None:
            semantic_mask = torch.ones_like(semantic_tokens, dtype=torch.float32)

        phone_mask = batch.get("phone_mask", None)  # [B, T_phone]
        if phone_mask is None:
            phone_mask = torch.ones_like(phone_id, dtype=torch.float32)

        torch.cuda.empty_cache()  # Removed: causes GPU sync, slows down training

        # Forward through student model (with 512 codebook)
        # Note: The model internally uses its own cond_emb which has 512 entries
        # But we need to map the teacher's 8192 tokens to student's 512 space
        # For fine-tuning, we use modulo operation to map tokens
        student_semantic_tokens = semantic_tokens % self.cfg.model.student_model.cond_codebook_size
        
        logits, final_mask, x0, prompt_len, mask_prob = self.model(
            student_semantic_tokens, semantic_mask, phone_id, phone_mask
        )  # logits: [B, T, 512], final_mask: [B, T, 1]

        # Compute cross-entropy loss on masked tokens
        logits_flat = logits.reshape(-1, logits.shape[-1])  # [B*T, 512]
        targets = student_semantic_tokens.reshape(-1)  # [B*T]
        mask = final_mask.reshape(-1)  # [B*T]

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_flat, targets, reduction='none')
        ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

        total_loss += ce_loss
        train_losses["ce_loss"] = ce_loss

        # Compute token accuracy for monitoring
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)  # [B, T]
            correct = (predictions == student_semantic_tokens).float()
            # Accuracy on masked tokens only
            masked_correct = (correct * final_mask.squeeze(-1)).sum()
            masked_total = final_mask.sum()
            token_accuracy = masked_correct / (masked_total + 1e-8)
            train_stats["token_accuracy"] = token_accuracy.item()

            # Top-5 accuracy
            top5_preds = logits.topk(5, dim=-1).indices  # [B, T, 5]
            top5_correct = (top5_preds == student_semantic_tokens.unsqueeze(-1)).any(dim=-1).float()
            top5_masked_correct = (top5_correct * final_mask.squeeze(-1)).sum()
            top5_accuracy = top5_masked_correct / (masked_total + 1e-8)
            train_stats["top5_accuracy"] = top5_accuracy.item()
            
            # Logits statistics
            train_stats["logits_mean"] = logits.mean().item()
            train_stats["logits_std"] = logits.std().item()
            train_stats["logits_max"] = logits.max().item()
            train_stats["logits_min"] = logits.min().item()
            
            # Prediction confidence (max probability)
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            train_stats["pred_confidence"] = (max_probs * final_mask.squeeze(-1)).sum() / (masked_total + 1e-8)
            train_stats["pred_confidence"] = train_stats["pred_confidence"].item()
            
            # Entropy of predictions
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            train_stats["pred_entropy"] = (entropy * final_mask.squeeze(-1)).sum() / (masked_total + 1e-8)
            train_stats["pred_entropy"] = train_stats["pred_entropy"].item()

        # Backward pass
        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            # Only clip gradients for trainable parameters
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            self.accelerator.clip_grad_norm_(trainable_params, 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Convert losses to items
        for item in train_losses:
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()
        train_losses["mask_ratio"] = mask_prob.mean().item() if isinstance(mask_prob, torch.Tensor) else mask_prob
        
        # Add batch statistics
        train_stats["batch_size"] = semantic_tokens.shape[0]
        train_stats["seq_length"] = semantic_tokens.shape[1]
        train_stats["phone_length"] = phone_id.shape[1]
        train_stats["masked_tokens"] = masked_total.item()

        return (total_loss.item(), train_losses, train_stats)

    def _valid_step(self, batch):
        """Validation step for fine-tuning.

        Similar to _train_step but without gradient computation.
        """
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Get semantic tokens (ground truth) from teacher's codec
        semantic_tokens = batch.get("semantic_code", None)  # [B, T]
        if semantic_tokens is None:
            # Extract from batch data if not provided
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")

        # Get phone IDs (text tokens)
        phone_id = batch.get("phone_id", None)  # [B, T_phone]
        if phone_id is None:
            raise ValueError("phone_id is required for T2S training")

        # Get masks
        semantic_mask = batch.get("semantic_mask", None)  # [B, T], 1 for valid, 0 for padding
        if semantic_mask is None:
            semantic_mask = torch.ones_like(semantic_tokens, dtype=torch.float32)

        phone_mask = batch.get("phone_mask", None)  # [B, T_phone]
        if phone_mask is None:
            phone_mask = torch.ones_like(phone_id, dtype=torch.float32)

        torch.cuda.empty_cache()  # Removed: causes GPU sync, slows down training

        # Map teacher tokens to student space
        student_semantic_tokens = semantic_tokens % self.cfg.model.student_model.cond_codebook_size

        # Forward through student model (no gradients)
        with torch.no_grad():
            logits, final_mask, x0, prompt_len, mask_prob = self.model(
                student_semantic_tokens, semantic_mask, phone_id, phone_mask
            )  # logits: [B, T, 512], final_mask: [B, T, 1]

            # Compute cross-entropy loss on masked tokens
            logits_flat = logits.reshape(-1, logits.shape[-1])  # [B*T, 512]
            targets = student_semantic_tokens.reshape(-1)  # [B*T]
            mask = final_mask.reshape(-1)  # [B*T]

            # Cross-entropy loss
            ce_loss = F.cross_entropy(logits_flat, targets, reduction='none')
            ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

            total_loss += ce_loss
            valid_losses["ce_loss"] = ce_loss.item()
            valid_losses["mask_ratio"] = mask_prob.mean().item() if isinstance(mask_prob, torch.Tensor) else mask_prob

            # Compute token accuracy
            predictions = logits.argmax(dim=-1)  # [B, T]
            correct = (predictions == student_semantic_tokens).float()
            masked_correct = (correct * final_mask.squeeze(-1)).sum()
            masked_total = final_mask.sum()
            token_accuracy = masked_correct / (masked_total + 1e-8)
            valid_stats["token_accuracy"] = token_accuracy.item()

            # Top-5 accuracy
            top5_preds = logits.topk(5, dim=-1).indices  # [B, T, 5]
            top5_correct = (top5_preds == student_semantic_tokens.unsqueeze(-1)).any(dim=-1).float()
            top5_masked_correct = (top5_correct * final_mask.squeeze(-1)).sum()
            top5_accuracy = top5_masked_correct / (masked_total + 1e-8)
            valid_stats["top5_accuracy"] = top5_accuracy.item()

        return (total_loss.item(), valid_losses, valid_stats)

    def _valid_epoch(self):
        """Validation epoch for fine-tuning.

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

