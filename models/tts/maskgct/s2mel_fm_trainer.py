# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer for Flow Matching Semantic-to-Mel model.

This trainer handles the training of the FlowMatchingS2Mel model which:
1. Takes codebook-quantized discrete semantic units as input
2. Incorporates acoustic information from prompt audio for zero-shot voice cloning
3. Generates mel spectrograms using Flow Matching
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import os
from tqdm import tqdm

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.tts.maskgct.maskgct_s2mel_fm import FlowMatchingS2Mel, FlowMatchingS2MelWithPrompt
from models.base.base_trainer import BaseTrainer
from utils.mel import extract_mel_features


class S2MelFMTrainer(BaseTrainer):
    """Trainer for Flow Matching Semantic-to-Mel model.

    The model predicts mel spectrograms from semantic tokens with prompt-based
    voice cloning using Flow Matching.
    """

    def __init__(self, args, cfg):
        super(S2MelFMTrainer, self).__init__(args, cfg)

        # Setup pretrained models for feature extraction
        self._build_input_models()

        # Mel spectrogram configuration
        self.n_mel = getattr(self.cfg.preprocess, "n_mel", 100)
        self.sample_rate = getattr(self.cfg.preprocess, "sample_rate", 16000)
        self.hop_size = getattr(self.cfg.preprocess, "hop_size", 320)

        # Mel normalization (from MelVQGAN config)
        self.mel_mean = getattr(self.cfg.preprocess, "mel_mean", 0.0)
        self.mel_var = getattr(self.cfg.preprocess, "mel_var", 1.0)

    def _build_model(self):
        """Build Flow Matching S2Mel model"""
        use_prompt_model = getattr(self.cfg.model.s2mel_fm, "use_prompt_model", False)

        if use_prompt_model:
            model = FlowMatchingS2MelWithPrompt(cfg=self.cfg.model.s2mel_fm)
        else:
            model = FlowMatchingS2Mel(cfg=self.cfg.model.s2mel_fm)

        # Configure gradient checkpointing for memory efficiency
        gradient_checkpointing = getattr(self.cfg.model.s2mel_fm, 'gradient_checkpointing', False)
        if hasattr(model.diff_estimator, 'gradient_checkpointing'):
            model.diff_estimator.gradient_checkpointing = gradient_checkpointing
        elif hasattr(model.diff_estimator, 'gradient_checkpointing_enable'):
            if gradient_checkpointing:
                model.diff_estimator.gradient_checkpointing_enable()

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
                        checkpoint = torch.load(model_path, map_location=self.accelerator.device, weights_only=False)
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
    def _extract_semantic_tokens(self, batch, output_layer=17):
        """Extract semantic tokens from pre-processed features or audio"""
        if not hasattr(self, 'semantic_codec'):
            return None

        # Check if pre-processed semantic features are available
        if "semantic_hidden_states" in batch:
            # Use pre-processed hidden states from dataset (most efficient)
            feat = batch["semantic_hidden_states"].to(self.accelerator.device)  # [B, T, D]
        elif "semantic_model_input_features" in batch and "semantic_model_attention_mask" in batch:
            # Use pre-processed features from dataset
            input_features = batch["semantic_model_input_features"].to(self.accelerator.device)  # [B, T, 80]
            attention_mask = batch["semantic_model_attention_mask"].to(self.accelerator.device)  # [B, T]

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
                wavs_16k = batch.get("wav_16k", None)
                if wavs_16k is None:
                    raise ValueError("wav_16000 or wav_16k is required for semantic token extraction")

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

        # Quantize to semantic tokens
        semantic_tokens, _ = self.semantic_codec.quantize(feat)  # [B, T]
        return semantic_tokens

    def _extract_mel_spectrogram(self, batch):
        """Extract mel spectrogram from audio"""
        # Check if mel spectrogram is already in batch
        if "mel_spectrogram" in batch:
            mel = batch["mel_spectrogram"].to(self.accelerator.device)
            # Ensure shape is (B, T, n_mel)
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            if mel.shape[-1] != self.n_mel and mel.shape[1] == self.n_mel:
                # Shape is (B, n_mel, T), transpose
                mel = mel.transpose(1, 2)
            return mel

        # Extract from audio
        wav = batch.get("wav", None)
        if wav is None:
            wav = batch.get("wav_16000", None)

        if wav is None:
            raise ValueError("wav or wav_16000 is required for mel spectrogram extraction")

        # Extract mel spectrogram using project utility
        mel = extract_mel_features(wav, self.cfg.preprocess, center=False)

        # mel shape: (B, n_mel, T) -> (B, T, n_mel)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.transpose(1, 2)

        # Normalize mel
        if self.mel_mean != 0.0 or self.mel_var != 1.0:
            mel = (mel - self.mel_mean) / math.sqrt(self.mel_var)

        return mel

    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)

    def _train_step(self, batch):
        """Training step for Flow Matching S2Mel model"""
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # Get semantic tokens
        semantic_tokens = batch.get("semantic_code", None)  # [B, T]
        if semantic_tokens is None:
            # Extract from batch data if not provided
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")

        # Get mel spectrogram (target)
        mel_target = self._extract_mel_spectrogram(batch)  # [B, T, n_mel]

        # Align lengths between semantic tokens and mel spectrogram
        semantic_len = semantic_tokens.shape[1]
        mel_len = mel_target.shape[1]

        # Interpolate to match lengths if needed
        if semantic_len != mel_len:
            # Interpolate semantic tokens to match mel length
            semantic_tokens_float = semantic_tokens.float().unsqueeze(1)  # [B, 1, T]
            semantic_tokens_interp = F.interpolate(
                semantic_tokens_float, size=mel_len, mode='nearest'
            ).squeeze(1).long()  # [B, T_mel]
            semantic_tokens = semantic_tokens_interp

        # Get masks
        x_mask = batch.get("mask", None)  # [B, T]
        if x_mask is None:
            x_mask = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.float32).to(mel_target.device)

        # Align mask length
        if x_mask.shape[1] != mel_len:
            x_mask = F.interpolate(
                x_mask.float().unsqueeze(1), size=mel_len, mode='nearest'
            ).squeeze(1)

        torch.cuda.empty_cache()

        # Forward through model
        noise, x, flow_pred, final_mask, prompt_len = self.model(
            mel_target=mel_target,
            semantic_tokens=semantic_tokens,
            x_mask=x_mask,
        )

        # Compute flow matching loss
        # Flow GT: x - (1 - sigma) * noise
        sigma = getattr(self.cfg.model.s2mel_fm, "sigma", 1e-5)
        flow_gt = x - (1 - sigma) * noise

        # L1 loss on flow prediction
        final_mask_squeezed = final_mask.squeeze(-1)  # [B, T]
        diff_loss = F.l1_loss(flow_pred, flow_gt, reduction='none').float() * final_mask  # [B, T, mel_dim]
        diff_loss = torch.mean(diff_loss, dim=2).sum() / (final_mask_squeezed.sum() + 1e-8)

        total_loss += diff_loss
        train_losses["flow_loss"] = diff_loss
        
        # Add flow statistics
        with torch.no_grad():
            train_stats["flow_pred_mean"] = flow_pred.mean().item()
            train_stats["flow_pred_std"] = flow_pred.std().item()
            train_stats["flow_gt_mean"] = flow_gt.mean().item()
            train_stats["flow_gt_std"] = flow_gt.std().item()
            train_stats["noise_mean"] = noise.mean().item()
            train_stats["noise_std"] = noise.std().item()
            train_stats["mel_target_mean"] = mel_target.mean().item()
            train_stats["mel_target_std"] = mel_target.std().item()
            
            # Flow prediction error statistics
            flow_error = torch.abs(flow_pred - flow_gt)
            train_stats["flow_error_mean"] = (flow_error * final_mask).sum() / (final_mask.sum() + 1e-8)
            train_stats["flow_error_mean"] = train_stats["flow_error_mean"].item()
            train_stats["flow_error_max"] = flow_error.max().item()

        # Backward pass
        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()), 0.2
            )
        self.optimizer.step()
        self.scheduler.step()

        # Convert losses to items
        for item in train_losses:
            if isinstance(train_losses[item], torch.Tensor):
                train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()
        
        # Add batch statistics
        train_stats["batch_size"] = mel_target.shape[0]
        train_stats["mel_length"] = mel_target.shape[1]
        train_stats["semantic_length"] = semantic_tokens.shape[1]
        train_stats["valid_frames"] = final_mask_squeezed.sum().item()
        if prompt_len is not None:
            train_stats["prompt_length"] = prompt_len if isinstance(prompt_len, (int, float)) else prompt_len.float().mean().item()

        return (total_loss.item(), train_losses, train_stats)

    def _valid_step(self, batch):
        """Validation step for Flow Matching S2Mel model"""
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Get semantic tokens
        semantic_tokens = batch.get("semantic_code", None)
        if semantic_tokens is None:
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")

        # Get mel spectrogram (target)
        mel_target = self._extract_mel_spectrogram(batch)

        # Align lengths
        semantic_len = semantic_tokens.shape[1]
        mel_len = mel_target.shape[1]

        if semantic_len != mel_len:
            semantic_tokens_float = semantic_tokens.float().unsqueeze(1)
            semantic_tokens_interp = F.interpolate(
                semantic_tokens_float, size=mel_len, mode='nearest'
            ).squeeze(1).long()
            semantic_tokens = semantic_tokens_interp

        # Get masks
        x_mask = batch.get("mask", None)
        if x_mask is None:
            x_mask = torch.ones(mel_target.shape[0], mel_target.shape[1], dtype=torch.float32).to(mel_target.device)

        if x_mask.shape[1] != mel_len:
            x_mask = F.interpolate(
                x_mask.float().unsqueeze(1), size=mel_len, mode='nearest'
            ).squeeze(1)

        torch.cuda.empty_cache()

        # Forward through model (no gradients)
        with torch.no_grad():
            noise, x, flow_pred, final_mask, prompt_len = self.model(
                mel_target=mel_target,
                semantic_tokens=semantic_tokens,
                x_mask=x_mask,
            )

            # Compute flow matching loss
            sigma = getattr(self.cfg.model.s2mel_fm, "sigma", 1e-5)
            flow_gt = x - (1 - sigma) * noise

            final_mask_squeezed = final_mask.squeeze(-1)
            diff_loss = F.l1_loss(flow_pred, flow_gt, reduction='none').float() * final_mask
            diff_loss = torch.mean(diff_loss, dim=2).sum() / (final_mask_squeezed.sum() + 1e-8)

        total_loss = diff_loss
        valid_losses["flow_loss"] = diff_loss.item()

        return (total_loss.item(), valid_losses, valid_stats)

    def _valid_epoch(self):
        """Validation epoch for Flow Matching S2Mel model"""
        self.model.eval()

        epoch_sum_loss = 0.0
        epoch_losses = {}

        if self.valid_dataloader is None:
            return 0.0, {}

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

    @torch.no_grad()
    def generate_mel(
        self,
        semantic_tokens,
        prompt_mel,
        prompt_semantic_tokens,
        semantic_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        """
        Generate mel spectrogram from semantic tokens with voice cloning.

        Args:
            semantic_tokens: [B, T] semantic token indices for generation
            prompt_mel: [B, T_prompt, n_mel] prompt mel for voice cloning
            prompt_semantic_tokens: [B, T_prompt] semantic tokens for prompt
            semantic_mask: [B, T] mask (optional)
            prompt_mask: [B, T_prompt] prompt mask (optional)
            n_timesteps: number of ODE solver steps
            cfg: CFG scale
            rescale_cfg: CFG rescaling factor

        Returns:
            mel: [B, T, n_mel] generated mel spectrogram
        """
        self.model.eval()

        if isinstance(self.model, FlowMatchingS2MelWithPrompt):
            mel = self.model.generate(
                semantic_tokens=semantic_tokens,
                prompt_mel=prompt_mel,
                prompt_semantic_tokens=prompt_semantic_tokens,
                semantic_mask=semantic_mask,
                prompt_mask=prompt_mask,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
            )
        else:
            # Concatenate for base model
            full_semantic = torch.cat([prompt_semantic_tokens, semantic_tokens], dim=1)
            mel = self.model.reverse_diffusion(
                semantic_tokens=full_semantic,
                prompt_mel=prompt_mel,
                semantic_mask=None,
                prompt_mask=prompt_mask,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
            )

        return mel
