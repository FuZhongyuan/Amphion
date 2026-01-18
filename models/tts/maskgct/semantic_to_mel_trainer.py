# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer for Semantic-to-Mel model.

This trainer handles the training of the SemanticToMel model which:
1. Takes codebook-quantized discrete semantic units as input
2. Incorporates acoustic information from reference audio for zero-shot voice cloning
3. Generates mel spectrograms using diffusion-based denoising
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.tts.maskgct.maskgct_semantic_to_mel import SemanticToMel, SemanticToMelWithPrompt
from models.base.base_trainer import BaseTrainer
from utils.mel import extract_mel_features


class SemanticToMelTrainer(BaseTrainer):
    """Trainer for Semantic-to-Mel model.

    The model predicts mel spectrograms from semantic tokens with acoustic conditioning
    from reference audio for zero-shot voice cloning.
    """

    def __init__(self, args, cfg):
        super(SemanticToMelTrainer, self).__init__(args, cfg)

        # Setup pretrained models for feature extraction
        self._build_input_models()

        # Mel spectrogram configuration
        self.n_mel = getattr(self.cfg.preprocess, "n_mel", 80)
        self.sample_rate = getattr(self.cfg.preprocess, "sample_rate", 16000)
        self.hop_size = getattr(self.cfg.preprocess, "hop_size", 320)

    def _build_model(self):
        """Build Semantic-to-Mel model"""
        use_prompt = getattr(self.cfg.model.semantic_to_mel, "use_prompt", False)

        if use_prompt:
            model = SemanticToMelWithPrompt(cfg=self.cfg.model.semantic_to_mel)
        else:
            model = SemanticToMel(cfg=self.cfg.model.semantic_to_mel)

        # Configure gradient checkpointing for memory efficiency
        gradient_checkpointing = getattr(self.cfg.model.semantic_to_mel, 'gradient_checkpointing', False)
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
            return batch["mel_spectrogram"].to(self.accelerator.device)

        # Extract from audio
        wav = batch.get("wav", None)
        if wav is None:
            wav = batch.get("wav_16000", None)

        if wav is None:
            raise ValueError("wav or wav_16000 is required for mel spectrogram extraction")

        # Extract mel spectrogram
        mel = extract_mel_features(wav, self.cfg.preprocess, center=False)

        # mel shape: (B, n_mel, T) -> (B, T, n_mel)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.transpose(1, 2)

        return mel

    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)

    def _get_reference_mel(self, batch, mel_target):
        """
        Get reference mel for acoustic conditioning.

        For training, we use a random segment from the same utterance or
        a different utterance in the batch as reference.
        """
        B, T, n_mel = mel_target.shape

        # Strategy 1: Use a random segment from the same utterance
        # This helps the model learn speaker characteristics
        ref_mel_list = []
        ref_mask_list = []

        for i in range(B):
            mel_i = mel_target[i]  # [T, n_mel]

            # Random segment length (10-50% of total length)
            min_len = max(10, int(T * 0.1))
            max_len = max(min_len + 1, int(T * 0.5))
            ref_len = torch.randint(min_len, max_len, (1,)).item()

            # Random start position
            max_start = max(0, T - ref_len)
            start = torch.randint(0, max_start + 1, (1,)).item()

            ref_mel = mel_i[start:start + ref_len]  # [ref_len, n_mel]
            ref_mask = torch.ones(ref_len, device=mel_target.device)

            ref_mel_list.append(ref_mel)
            ref_mask_list.append(ref_mask)

        # Pad to same length
        max_ref_len = max(r.shape[0] for r in ref_mel_list)
        ref_mel_padded = torch.zeros(B, max_ref_len, n_mel, device=mel_target.device)
        ref_mask_padded = torch.zeros(B, max_ref_len, device=mel_target.device)

        for i, (ref_mel, ref_mask) in enumerate(zip(ref_mel_list, ref_mask_list)):
            ref_len = ref_mel.shape[0]
            ref_mel_padded[i, :ref_len] = ref_mel
            ref_mask_padded[i, :ref_len] = ref_mask

        return ref_mel_padded, ref_mask_padded

    def _train_step(self, batch):
        """Training step for Semantic-to-Mel model"""
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
        # Semantic tokens are at ~50Hz (16kHz / 320 hop), mel is at sample_rate / hop_size
        semantic_len = semantic_tokens.shape[1]
        mel_len = mel_target.shape[1]

        # Interpolate to match lengths if needed
        if semantic_len != mel_len:
            # Interpolate semantic tokens to match mel length
            # We need to handle discrete tokens - use nearest neighbor
            semantic_tokens_float = semantic_tokens.float().unsqueeze(1)  # [B, 1, T]
            semantic_tokens_interp = F.interpolate(
                semantic_tokens_float, size=mel_len, mode='nearest'
            ).squeeze(1).long()  # [B, T_mel]
            semantic_tokens = semantic_tokens_interp

        # Get masks
        semantic_mask = batch.get("semantic_mask", None)  # [B, T]
        if semantic_mask is None:
            semantic_mask = batch.get("mask", None)
        if semantic_mask is None:
            semantic_mask = torch.ones_like(semantic_tokens, dtype=torch.float32)

        # Align mask length
        if semantic_mask.shape[1] != mel_len:
            semantic_mask = F.interpolate(
                semantic_mask.float().unsqueeze(1), size=mel_len, mode='nearest'
            ).squeeze(1)

        # Get reference mel for acoustic conditioning
        ref_mel, ref_mel_mask = self._get_reference_mel(batch, mel_target)

        torch.cuda.empty_cache()

        # Forward through model
        loss, loss_dict = self.model(
            mel_target=mel_target,
            semantic_tokens=semantic_tokens,
            semantic_mask=semantic_mask,
            ref_mel=ref_mel,
            ref_mel_mask=ref_mel_mask,
        )

        total_loss = loss

        # Record losses
        for key, value in loss_dict.items():
            train_losses[key] = value

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
            if isinstance(train_losses[item], torch.Tensor):
                train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        return (total_loss.item(), train_losses, train_stats)

    def _valid_step(self, batch):
        """Validation step for Semantic-to-Mel model"""
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Get semantic tokens
        semantic_tokens = batch.get("semantic_code", None)  # [B, T]
        if semantic_tokens is None:
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")

        # Get mel spectrogram (target)
        mel_target = self._extract_mel_spectrogram(batch)  # [B, T, n_mel]

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
        semantic_mask = batch.get("semantic_mask", None)
        if semantic_mask is None:
            semantic_mask = batch.get("mask", None)
        if semantic_mask is None:
            semantic_mask = torch.ones_like(semantic_tokens, dtype=torch.float32)

        if semantic_mask.shape[1] != mel_len:
            semantic_mask = F.interpolate(
                semantic_mask.float().unsqueeze(1), size=mel_len, mode='nearest'
            ).squeeze(1)

        # Get reference mel
        ref_mel, ref_mel_mask = self._get_reference_mel(batch, mel_target)

        torch.cuda.empty_cache()

        # Forward through model (no gradients)
        with torch.no_grad():
            loss, loss_dict = self.model(
                mel_target=mel_target,
                semantic_tokens=semantic_tokens,
                semantic_mask=semantic_mask,
                ref_mel=ref_mel,
                ref_mel_mask=ref_mel_mask,
            )

        total_loss = loss

        # Record losses
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                valid_losses[key] = value.item()
            else:
                valid_losses[key] = value

        return (total_loss.item(), valid_losses, valid_stats)

    def _valid_epoch(self):
        """Validation epoch for Semantic-to-Mel model"""
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
    def generate_mel(self, semantic_tokens, ref_mel, semantic_mask=None, ref_mel_mask=None,
                     n_timesteps=50, cfg_scale=2.0):
        """
        Generate mel spectrogram from semantic tokens.

        Args:
            semantic_tokens: [B, T] semantic token indices
            ref_mel: [B, T_ref, n_mel] reference mel for voice cloning
            semantic_mask: [B, T] mask (optional)
            ref_mel_mask: [B, T_ref] reference mel mask (optional)
            n_timesteps: number of diffusion steps
            cfg_scale: classifier-free guidance scale

        Returns:
            mel: [B, T, n_mel] generated mel spectrogram
        """
        self.model.eval()

        mel = self.model.generate(
            semantic_tokens=semantic_tokens,
            semantic_mask=semantic_mask,
            ref_mel=ref_mel,
            ref_mel_mask=ref_mel_mask,
            n_timesteps=n_timesteps,
            cfg_scale=cfg_scale,
        )

        return mel
