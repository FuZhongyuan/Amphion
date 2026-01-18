# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from models.base.base_trainer import BaseTrainer


class T2STrainer(BaseTrainer):
    """Trainer for MaskGCT-T2S model.
    
    The T2S model predicts semantic tokens with text and prompt semantic tokens.
    """
    
    def __init__(self, args, cfg):
        super(T2STrainer, self).__init__(args, cfg)
        
        # Setup pretrained models for feature extraction
        self._build_input_models()
    
    def _build_model(self):
        """Build T2S model"""
        model = MaskGCT_T2S(cfg=self.cfg.model.t2s_model)

        # Configure gradient checkpointing for memory efficiency
        # MaskGCT_T2S uses DiffLlamaPrefix as diff_estimator
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
    
    def _build_input_models(self):
        """Build pretrained models for extracting semantic tokens"""
        # Load semantic codec for extracting semantic tokens from speech
        if hasattr(self.cfg.model, "semantic_codec") and hasattr(self.cfg.model.semantic_codec, "pretrained_path"):
            from models.codec.kmeans.repcodec_model import RepCodec
            self.semantic_codec = RepCodec(cfg=self.cfg.model.semantic_codec)
            # Load pretrained weights if available
            pretrained_path = self.cfg.model.semantic_codec.pretrained_path
            if pretrained_path:
                import os
                import safetensors
                # Check if path is a directory (accelerate checkpoint) or file
                if os.path.isdir(pretrained_path):
                    # Accelerate checkpoint format
                    model_path = os.path.join(pretrained_path, "model.safetensors")
                    if not os.path.exists(model_path):
                        model_path = os.path.join(pretrained_path, "pytorch_model.bin")
                else:
                    model_path = pretrained_path
                
                if os.path.exists(model_path):
                    if model_path.endswith(".safetensors"):
                        safetensors.torch.load_model(self.semantic_codec, model_path)
                    else:
                        checkpoint = torch.load(model_path, map_location=self.accelerator.device)
                        if isinstance(checkpoint, dict) and "model" in checkpoint:
                            self.semantic_codec.load_state_dict(checkpoint["model"], strict=False)
                        else:
                            self.semantic_codec.load_state_dict(checkpoint, strict=False)
            self.semantic_codec.eval()
            self.semantic_codec.to(self.accelerator.device)
            
            # Load semantic model (w2v-bert-2.0) for feature extraction
            from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
            self.semantic_model.eval()
            self.semantic_model.to(self.accelerator.device)
            
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
            
            # Load normalization statistics
            if hasattr(self.cfg.model.semantic_codec, "representation_stat_mean_var_path"):
                stat_path = self.cfg.model.semantic_codec.representation_stat_mean_var_path
                stat = torch.load(stat_path, weights_only=False)
                self.semantic_mean = torch.tensor(stat["mean"]).to(self.accelerator.device)
                self.semantic_std = torch.sqrt(torch.tensor(stat["var"])).to(self.accelerator.device)
            else:
                self.semantic_mean = None
                self.semantic_std = None
    
    @torch.no_grad()
    def _extract_semantic_tokens(self, batch):
        """Extract semantic tokens from pre-processed features"""
        if not hasattr(self, 'semantic_codec'):
            return None

        # Check if pre-processed semantic features are available
        if "semantic_model_input_features" in batch and "semantic_model_attention_mask" in batch:
            # Use pre-processed features from dataset
            input_features = batch["semantic_model_input_features"]  # [B, T, 80]
            attention_mask = batch["semantic_model_attention_mask"]  # [B, T]

            # Extract semantic features using pre-built model
            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            feat = outputs.hidden_states[17]  # [B, T, D]
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
            feat = outputs.hidden_states[17]  # [B, T, D]

        # Normalize
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        # Quantize to semantic tokens
        semantic_tokens, _ = self.semantic_codec.quantize(feat)  # [B, T]
        return semantic_tokens
    
    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)
    
    def _train_step(self, batch):
        """Training step for T2S model"""
        train_losses = {}
        total_loss = 0
        train_stats = {}
        
        # Get semantic tokens (ground truth)
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
        
        torch.cuda.empty_cache()
        
        # Forward through T2S model
        logits, final_mask, x0, prompt_len, mask_prob = self.model(
            semantic_tokens, semantic_mask, phone_id, phone_mask
        )  # logits: [B, T, codebook_size], final_mask: [B, T, 1]
        
        # Compute cross-entropy loss on masked tokens
        logits = logits.reshape(-1, logits.shape[-1])  # [B*T, codebook_size]
        targets = x0.reshape(-1)  # [B*T]
        mask = final_mask.reshape(-1)  # [B*T]
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss += ce_loss
        train_losses["ce_loss"] = ce_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 1.0
            )
        self.optimizer.step()
        self.scheduler.step()
        
        # Convert losses to items
        for item in train_losses:
            train_losses[item] = train_losses[item].item()
        
        self.current_loss = total_loss.item()
        train_losses["batch_size"] = semantic_tokens.shape[0]
        train_losses["mask_ratio"] = mask_prob.mean().item() if isinstance(mask_prob, torch.Tensor) else mask_prob
        
        return (total_loss.item(), train_losses, train_stats)

