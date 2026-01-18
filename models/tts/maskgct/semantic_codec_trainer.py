# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torchaudio
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
from tqdm import tqdm

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.codec.kmeans.repcodec_model import RepCodec
from models.base.base_trainer import BaseTrainer


class SemanticCodecTrainer(BaseTrainer):
    """Trainer for Semantic Codec model.
    
    The semantic codec converts speech to semantic tokens using w2v-bert-2.0 features.
    """
    
    def __init__(self, args, cfg):
        super(SemanticCodecTrainer, self).__init__(args, cfg)
        
        # Setup input model (w2v-bert-2.0)
        self._build_input_model()
    
    def _build_model(self):
        """Build semantic codec model (RepCodec)"""
        model = RepCodec(cfg=self.cfg.model.semantic_codec)

        # Note: RepCodec doesn't support gradient checkpointing
        # as it's a custom model, not a standard transformer

        return model
    
    def _build_input_model(self):
        """Build w2v-bert-2.0 model for feature extraction"""
        if self.cfg.model.representation_type == "w2v-bert-2.0":
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
            self.semantic_model.eval()
            self.semantic_model.to(self.accelerator.device)
            
            self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
            
            # Load normalization statistics if available
            if getattr(self.cfg.model, "use_norm_feat", False):
                stat_path = self.cfg.model.representation_stat_mean_var_path
                stat = torch.load(stat_path)
                self.feat_norm_mean = torch.tensor(stat["mean"]).to(self.accelerator.device)
                self.feat_norm_std = torch.sqrt(torch.tensor(stat["var"])).to(self.accelerator.device)
    
    @torch.no_grad()
    def _extract_semantic_feature(self, batch, output_layer=17):
        """
        Extract semantic features from w2v-bert-2.0 model.

        Args:
            batch: batch data containing audio or pre-processed features
            output_layer: which layer to extract features from (default: 17)

        Returns:
            feat: [B, T, D] semantic features
        """
        # Check if pre-processed semantic features are available
        if "semantic_hidden_states" in batch:
            # Use pre-processed hidden states from dataset (most efficient)
            feat = batch["semantic_hidden_states"].to(self.accelerator.device)  # [B, T, D]
        elif "semantic_model_input_features" in batch and "semantic_model_attention_mask" in batch:
            # Use pre-processed features from dataset
            input_features = batch["semantic_model_input_features"].to(self.accelerator.device)  # [B, T, 80]
            attention_mask = batch["semantic_model_attention_mask"].to(self.accelerator.device)  # [B, T]

            # input_features = inputs["input_features"].to(self.accelerator.device)
            # attention_mask = inputs["attention_mask"].to(self.accelerator.device)

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
                # Try alternative keys
                wavs_16k = batch.get("wav_16k", None)
                if wavs_16k is None:
                    raise ValueError("wav_16000 or wav_16k is required for semantic feature extraction")

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

        return feat
    
    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)
    
    def _train_step(self, batch):
        """Training step for semantic codec"""
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # Extract semantic features
        feat = self._extract_semantic_feature(batch)

        # Gaussian normalization if enabled
        if getattr(self.cfg.model, "use_norm_feat", False):
            feat = (feat - self.feat_norm_mean.to(feat)) / self.feat_norm_std.to(feat)

        # torch.cuda.empty_cache()
        
        # Forward through semantic codec
        feat_rec, codebook_loss, _ = self.model(feat)
        
        # Reconstruction loss
        rec_loss = torch.nn.functional.l1_loss(feat_rec, feat)
        total_loss += rec_loss * getattr(self.cfg.model.semantic_codec, "rec_loss_weight", 32.0)
        train_losses["rec_loss"] = rec_loss
        
        # Codebook loss
        total_loss += codebook_loss
        train_losses["codebook_loss"] = codebook_loss
        
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

        return (total_loss.item(), train_losses, train_stats)

    def _valid_step(self, batch):
        """Validation step for semantic codec"""
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Extract semantic features
        feat = self._extract_semantic_feature(batch)

        # Gaussian normalization if enabled
        if getattr(self.cfg.model, "use_norm_feat", False):
            feat = (feat - self.feat_norm_mean.to(feat)) / self.feat_norm_std.to(feat)

        # torch.cuda.empty_cache()

        # Forward through semantic codec (no gradients)
        with torch.no_grad():
            feat_rec, codebook_loss, _ = self.model(feat)

        # Reconstruction loss
        rec_loss = torch.nn.functional.l1_loss(feat_rec, feat)
        total_loss += rec_loss * getattr(self.cfg.model.semantic_codec, "rec_loss_weight", 32.0)
        valid_losses["rec_loss"] = rec_loss.item()

        # Codebook loss
        total_loss += codebook_loss
        valid_losses["codebook_loss"] = codebook_loss.item()

        return (total_loss.item(), valid_losses, valid_stats)

    def _valid_epoch(self):
        """Validation epoch for semantic codec"""
        self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = {}
        for batch in self.valid_dataloader:
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, valid_losses, valid_stats = self._valid_step(batch)
            epoch_sum_loss += total_loss
            if isinstance(valid_losses, dict):
                for key, value in valid_losses.items():
                    if key not in epoch_losses.keys():
                        epoch_losses[key] = value
                    else:
                        epoch_losses[key] += value

        epoch_sum_loss = epoch_sum_loss / len(self.valid_dataloader)
        for key in epoch_losses.keys():
            epoch_losses[key] = epoch_losses[key] / len(self.valid_dataloader)

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, epoch_losses

