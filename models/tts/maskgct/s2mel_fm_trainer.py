# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
S2Mel Flow Matching Trainer for MaskGCT

Trains the Semantic-to-Mel Flow Matching model that converts
discrete semantic tokens to continuous mel spectrograms.
"""

import torch
import torch.nn.functional as F
import math

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.tts.maskgct.maskgct_s2mel_fm import SemanticToMelFM
from models.base.base_trainer import BaseTrainer
from utils.mel import extract_mel_features


class S2MelFMTrainer(BaseTrainer):
    """Trainer for Semantic-to-Mel Flow Matching model."""

    def __init__(self, args, cfg):
        super(S2MelFMTrainer, self).__init__(args, cfg)
        self._build_input_models()

    def _build_model(self):
        """Build S2Mel Flow Matching model."""
        model = SemanticToMelFM(cfg=self.cfg.model.s2mel_fm)

        # Configure gradient checkpointing
        gradient_checkpointing = getattr(self.cfg.model.s2mel_fm, 'gradient_checkpointing', False)
        if hasattr(model.diff_estimator, 'gradient_checkpointing'):
            model.diff_estimator.gradient_checkpointing = gradient_checkpointing

        return model

    def _build_input_models(self):
        """Build pretrained models for extracting semantic tokens."""
        # Load semantic codec for extracting semantic tokens
        if hasattr(self.cfg.model, "semantic_codec") and hasattr(self.cfg.model.semantic_codec, "pretrained_path"):
            from models.codec.kmeans.repcodec_model import RepCodec
            self.semantic_codec = RepCodec(cfg=self.cfg.model.semantic_codec)
            pretrained_path = self.cfg.model.semantic_codec.pretrained_path
            if pretrained_path:
                import os
                import safetensors
                if os.path.isdir(pretrained_path):
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

            # Load semantic model (w2v-bert-2.0)
            from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
            self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
            self.semantic_model.eval()
            self.semantic_model.to(self.accelerator.device)

            self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

            # Load normalization statistics
            if hasattr(self.cfg.model.semantic_codec, "representation_stat_mean_var_path"):
                stat_path = self.cfg.model.semantic_codec.representation_stat_mean_var_path
                stat = torch.load(stat_path)
                self.semantic_mean = torch.tensor(stat["mean"]).to(self.accelerator.device)
                self.semantic_std = torch.sqrt(torch.tensor(stat["var"])).to(self.accelerator.device)
            else:
                self.semantic_mean = None
                self.semantic_std = None

    @torch.no_grad()
    def _extract_semantic_tokens(self, batch, output_layer=17):
        """Extract semantic tokens from pre-processed features."""
        if not hasattr(self, 'semantic_codec'):
            return None

        if "semantic_hidden_states" in batch:
            feat = batch["semantic_hidden_states"].to(self.accelerator.device)
        elif "semantic_model_input_features" in batch and "semantic_model_attention_mask" in batch:
            input_features = batch["semantic_model_input_features"]
            attention_mask = batch["semantic_model_attention_mask"]

            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            feat = outputs.hidden_states[output_layer]
        else:
            wavs_16k = batch.get("wav_16000", None)
            if wavs_16k is None:
                raise ValueError("wav_16000 is required for semantic token extraction")

            inputs = self.processor(
                wavs_16k.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            input_features = inputs["input_features"].to(self.accelerator.device)
            attention_mask = inputs["attention_mask"].to(self.accelerator.device)

            outputs = self.semantic_model(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            feat = outputs.hidden_states[output_layer]

        # Normalize
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        # Quantize to semantic tokens
        semantic_tokens, _ = self.semantic_codec.quantize(feat)
        return semantic_tokens

    @torch.no_grad()
    def _extract_mel(self, batch):
        """Extract mel spectrogram from audio."""
        if "mel_spectrogram" in batch:
            return batch["mel_spectrogram"]

        speech = batch["wav"]
        mel = extract_mel_features(speech, self.cfg.preprocess, center=False)
        # mel shape: (B, n_mel, T) -> (B, T, n_mel)
        mel = mel.transpose(1, 2)

        # Normalize mel
        if hasattr(self.cfg.preprocess, "mel_mean") and hasattr(self.cfg.preprocess, "mel_var"):
            mel = (mel - self.cfg.preprocess.mel_mean) / math.sqrt(self.cfg.preprocess.mel_var)

        return mel

    def _build_dataset(self):
        """Build dataset for training."""
        return get_maskgct_dataset_class(self.cfg)

    def _train_step(self, batch):
        """Training step for S2Mel FM model."""
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # Get semantic tokens
        semantic_tokens = batch.get("semantic_code", None)
        if semantic_tokens is None:
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens.")

        # Get mel spectrogram
        mel = self._extract_mel(batch)

        # Align lengths
        target_len = min(semantic_tokens.shape[1], mel.shape[1])
        semantic_tokens = semantic_tokens[:, :target_len]
        mel = mel[:, :target_len, :]

        # Create mask
        x_mask = batch.get("mask", None)
        if x_mask is None:
            x_mask = torch.ones(mel.shape[0], mel.shape[1], dtype=torch.float32).to(mel.device)
        else:
            x_mask = x_mask[:, :target_len]

        # torch.cuda.empty_cache()

        # Forward through model
        noise, x, flow_pred, final_mask, prompt_len = self.model(
            x=mel,
            x_mask=x_mask,
            semantic_tokens=semantic_tokens,
        )

        final_mask = final_mask.squeeze(-1)

        # Compute flow matching loss
        sigma = self.cfg.model.s2mel_fm.sigma if hasattr(self.cfg.model.s2mel_fm, "sigma") else 1e-5
        flow_gt = x - (1 - sigma) * noise

        diff_loss = F.l1_loss(flow_pred, flow_gt, reduction="none").float() * final_mask.unsqueeze(-1)
        diff_loss = torch.mean(diff_loss, dim=2).sum() / (final_mask.sum())

        total_loss += diff_loss
        train_losses["diff_loss"] = diff_loss

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
            train_losses[item] = train_losses[item].item()

        self.current_loss = total_loss.item()

        train_stats["batch_size"] = mel.shape[0]
        train_stats["seq_length"] = mel.shape[1]

        return (total_loss.item(), train_losses, train_stats)

    def _valid_step(self, batch):
        """Validation step for S2Mel FM model."""
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Get semantic tokens
        semantic_tokens = batch.get("semantic_code", None)
        if semantic_tokens is None:
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens.")

        # Get mel spectrogram
        mel = self._extract_mel(batch)

        # Align lengths
        target_len = min(semantic_tokens.shape[1], mel.shape[1])
        semantic_tokens = semantic_tokens[:, :target_len]
        mel = mel[:, :target_len, :]

        # Create mask
        x_mask = batch.get("mask", None)
        if x_mask is None:
            x_mask = torch.ones(mel.shape[0], mel.shape[1], dtype=torch.float32).to(mel.device)
        else:
            x_mask = x_mask[:, :target_len]

        # torch.cuda.empty_cache()

        with torch.no_grad():
            noise, x, flow_pred, final_mask, prompt_len = self.model(
                x=mel,
                x_mask=x_mask,
                semantic_tokens=semantic_tokens,
            )

            final_mask = final_mask.squeeze(-1)

            sigma = self.cfg.model.s2mel_fm.sigma if hasattr(self.cfg.model.s2mel_fm, "sigma") else 1e-5
            flow_gt = x - (1 - sigma) * noise

            diff_loss = F.l1_loss(flow_pred, flow_gt, reduction="none").float() * final_mask.unsqueeze(-1)
            diff_loss = torch.mean(diff_loss, dim=2).sum() / (final_mask.sum())

            total_loss += diff_loss
            valid_losses["diff_loss"] = diff_loss.item()

        return (total_loss.item(), valid_losses, valid_stats)

    # def _valid_epoch(self):
    #     """Validation epoch."""
    #     self.model.eval()

    #     epoch_sum_loss = 0.0
    #     val_batch_count = 0
    #     epoch_losses = {}
    #     for batch in self.valid_dataloader:
    #         device = self.accelerator.device
    #         for k, v in batch.items():
    #             if isinstance(v, torch.Tensor):
    #                 batch[k] = v.to(device)

    #         total_loss, valid_losses, valid_stats = self._valid_step(batch)
    #         val_batch_count += 1
    #         if (
    #             self.accelerator.is_main_process
    #             and val_batch_count
    #             % (10 * self.cfg.train.gradient_accumulation_step)
    #             == 0
    #         ):
    #             self.echo_log(valid_losses, mode="Validing")
            
    #         epoch_sum_loss += total_loss
    #         if isinstance(valid_losses, dict):
    #             for key, value in valid_losses.items():
    #                 if key not in epoch_losses:
    #                     epoch_losses[key] = value
    #                 else:
    #                     epoch_losses[key] += value

    #     epoch_sum_loss = epoch_sum_loss / len(self.valid_dataloader)
    #     for key in epoch_losses:
    #         epoch_losses[key] = epoch_losses[key] / len(self.valid_dataloader)

    #     self.accelerator.wait_for_everyone()

    #     return epoch_sum_loss, epoch_losses
