# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torchaudio
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

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
    def _extract_semantic_feature(self, wavs, wav_lens=None, output_layer=17):
        """
        Extract semantic features from w2v-bert-2.0 model.
        
        Args:
            wavs: [B, T] audio waveforms at 16kHz
            wav_lens: [B,] actual lengths of waveforms
            output_layer: which layer to extract features from (default: 17)
        
        Returns:
            feats: [B, T, D] semantic features
        """
        with torch.no_grad():
            # Process audio with feature extractor
            inputs = self.processor(
                wavs.cpu().numpy(), 
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
            # Use layer 17 features (as in maskgct_utils.py)
            feats = outputs.hidden_states[output_layer]  # [B, T, D]
            
        return feats
    
    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)
    
    def _train_step(self, batch):
        """Training step for semantic codec"""
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # Get pre-processed semantic features from dataset
        if "semantic_model_input_features" in batch and "semantic_model_attention_mask" in batch:
            # Use pre-processed features from dataset
            input_features = batch["semantic_model_input_features"]  # [B, T, 80]
            attention_mask = batch["semantic_model_attention_mask"]  # [B, T]

            # Extract semantic features using pre-built model
            with torch.no_grad():
                outputs = self.semantic_model(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                # Use layer 17 features (as in maskgct_utils.py)
                feat = outputs.hidden_states[17]  # [B, T, D]
        else:
            # Fallback: extract features manually (for backward compatibility)
            speech_16k = batch.get("wav_16k", None)  # [B, T] at 16kHz
            speech_lens = batch.get("wav_16k_len", None)  # [B,]

            if speech_16k is None:
                # Fallback to regular wav if wav_16k not available
                speech_16k = batch["wav"]  # [B, T]
                # Resample to 16kHz if needed
                if self.cfg.preprocess.sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(
                        self.cfg.preprocess.sample_rate,
                        16000
                    ).to(speech_16k.device)
                    speech_16k = resampler(speech_16k)
                speech_lens = batch.get("wav_len", None)

            # Extract semantic features
            feat = self._extract_semantic_feature(speech_16k, speech_lens)  # [B, T, D]
        
        # Gaussian normalization if enabled
        if getattr(self.cfg.model, "use_norm_feat", False):
            feat = (feat - self.feat_norm_mean.to(feat)) / self.feat_norm_std.to(feat)
        
        torch.cuda.empty_cache()
        
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
        train_losses["batch_size"] = feat.shape[0]
        
        return (total_loss.item(), train_losses, train_stats)

