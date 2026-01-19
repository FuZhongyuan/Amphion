# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
from models.base.base_trainer import BaseTrainer


class AcousticCodecTrainer(BaseTrainer):
    """Trainer for Acoustic Codec model.

    The acoustic codec converts speech to acoustic tokens and reconstructs
    waveform from acoustic tokens.
    """

    def __init__(self, args, cfg):
        super(AcousticCodecTrainer, self).__init__(args, cfg)

    def _build_model(self):
        """Build acoustic codec model (encoder + decoder)"""
        encoder = CodecEncoder(cfg=self.cfg.model.acoustic_codec.encoder)
        decoder = CodecDecoder(cfg=self.cfg.model.acoustic_codec.decoder)

        # Return as a dict to handle both encoder and decoder
        model = nn.ModuleDict({
            'encoder': encoder,
            'decoder': decoder
        })
        return model

    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)

    def _train_step(self, batch):
        """Training step for acoustic codec"""
        train_losses = {}
        total_loss = 0
        train_stats = {}

        # Get audio data at 24kHz (acoustic codec uses 24kHz)
        speech = batch.get("wav_24k", None)  # [B, T] at 24kHz
        if speech is None:
            speech = batch["wav"]  # [B, T]
            # Resample to 24kHz if needed
            if self.cfg.preprocess.sample_rate != 24000:
                import torchaudio
                resampler = torchaudio.transforms.Resample(
                    self.cfg.preprocess.sample_rate,
                    24000
                ).to(speech.device)
                speech = resampler(speech)

        # Add channel dimension: [B, T] -> [B, 1, T]
        speech = speech.unsqueeze(1)

        # torch.cuda.empty_cache()

        # Forward through encoder
        # Handle both DDP and non-DDP cases
        if hasattr(self.model, 'module'):
            # DDP case: access through .module
            encoded = self.model.module['encoder'](speech)  # [B, C, T']
            # Quantize and decode
            quantized_out, all_indices, all_commit_losses, all_codebook_losses, all_quantized = \
                self.model.module['decoder'](encoded, vq=True, n_quantizers=None)
            # Decode to waveform
            reconstructed = self.model.module['decoder'](quantized_out, vq=False)  # [B, 1, T]
        else:
            # Non-DDP case: direct access
            encoded = self.model['encoder'](speech)  # [B, C, T']
            # Quantize and decode
            quantized_out, all_indices, all_commit_losses, all_codebook_losses, all_quantized = \
                self.model['decoder'](encoded, vq=True, n_quantizers=None)
            # Decode to waveform
            reconstructed = self.model['decoder'](quantized_out, vq=False)  # [B, 1, T]

        # Reconstruction loss (L1 loss)
        rec_loss = F.l1_loss(reconstructed, speech)
        rec_loss_weight = getattr(self.cfg.model.acoustic_codec.decoder, "rec_loss_weight", 1.0)
        total_loss += rec_loss * rec_loss_weight
        train_losses["rec_loss"] = rec_loss

        # Commitment loss
        commit_loss = torch.sum(all_commit_losses)
        commit_weight = getattr(self.cfg.model.acoustic_codec.decoder, "commitment", 0.25)
        total_loss += commit_loss * commit_weight
        train_losses["commit_loss"] = commit_loss

        # Codebook loss
        codebook_loss = torch.sum(all_codebook_losses)
        codebook_loss_weight = getattr(self.cfg.model.acoustic_codec.decoder, "codebook_loss_weight", 1.0)
        total_loss += codebook_loss * codebook_loss_weight
        train_losses["codebook_loss"] = codebook_loss

        # Optional: Mel-spectrogram loss for better perceptual quality
        if getattr(self.cfg.model.acoustic_codec.decoder, "use_mel_loss", False):
            from utils.audio import mel_spectrogram
            mel_gt = mel_spectrogram(
                speech.squeeze(1),
                n_fft=1024,
                num_mels=80,
                sampling_rate=24000,
                hop_size=240,
                win_size=1024,
            )
            mel_pred = mel_spectrogram(
                reconstructed.squeeze(1),
                n_fft=1024,
                num_mels=80,
                sampling_rate=24000,
                hop_size=240,
                win_size=1024,
            )
            mel_loss = F.l1_loss(mel_pred, mel_gt)
            mel_loss_weight = getattr(self.cfg.model.acoustic_codec.decoder, "mel_loss_weight", 45.0)
            total_loss += mel_loss * mel_loss_weight
            train_losses["mel_loss"] = mel_loss

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
        """Validation step for acoustic codec"""
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Get audio data at 24kHz (acoustic codec uses 24kHz)
        speech = batch.get("wav_24k", None)  # [B, T] at 24kHz
        if speech is None:
            speech = batch["wav"]  # [B, T]
            # Resample to 24kHz if needed
            if self.cfg.preprocess.sample_rate != 24000:
                import torchaudio
                resampler = torchaudio.transforms.Resample(
                    self.cfg.preprocess.sample_rate,
                    24000
                ).to(speech.device)
                speech = resampler(speech)

        # Add channel dimension: [B, T] -> [B, 1, T]
        speech = speech.unsqueeze(1)

        # torch.cuda.empty_cache()

        # Forward through encoder and decoder (no gradients)
        with torch.no_grad():
            # Handle both DDP and non-DDP cases
            if hasattr(self.model, 'module'):
                # DDP case: access through .module
                encoded = self.model.module['encoder'](speech)  # [B, C, T']
                # Quantize and decode
                quantized_out, all_indices, all_commit_losses, all_codebook_losses, all_quantized = \
                    self.model.module['decoder'](encoded, vq=True, n_quantizers=None)
                # Decode to waveform
                reconstructed = self.model.module['decoder'](quantized_out, vq=False)  # [B, 1, T]
            else:
                # Non-DDP case: direct access
                encoded = self.model['encoder'](speech)  # [B, C, T']
                # Quantize and decode
                quantized_out, all_indices, all_commit_losses, all_codebook_losses, all_quantized = \
                    self.model['decoder'](encoded, vq=True, n_quantizers=None)
                # Decode to waveform
                reconstructed = self.model['decoder'](quantized_out, vq=False)  # [B, 1, T]

        # Reconstruction loss (L1 loss)
        rec_loss = F.l1_loss(reconstructed, speech)
        rec_loss_weight = getattr(self.cfg.model.acoustic_codec.decoder, "rec_loss_weight", 1.0)
        total_loss += rec_loss * rec_loss_weight
        valid_losses["rec_loss"] = rec_loss.item()

        # Commitment loss
        commit_loss = torch.sum(all_commit_losses)
        commit_weight = getattr(self.cfg.model.acoustic_codec.decoder, "commitment", 0.25)
        total_loss += commit_loss * commit_weight
        valid_losses["commit_loss"] = commit_loss.item()

        # Codebook loss
        codebook_loss = torch.sum(all_codebook_losses)
        codebook_loss_weight = getattr(self.cfg.model.acoustic_codec.decoder, "codebook_loss_weight", 1.0)
        total_loss += codebook_loss * codebook_loss_weight
        valid_losses["codebook_loss"] = codebook_loss.item()

        # Optional: Mel-spectrogram loss for better perceptual quality
        if getattr(self.cfg.model.acoustic_codec.decoder, "use_mel_loss", False):
            from utils.audio import mel_spectrogram
            mel_gt = mel_spectrogram(
                speech.squeeze(1),
                n_fft=1024,
                num_mels=80,
                sampling_rate=24000,
                hop_size=240,
                win_size=1024,
            )
            mel_pred = mel_spectrogram(
                reconstructed.squeeze(1),
                n_fft=1024,
                num_mels=80,
                sampling_rate=24000,
                hop_size=240,
                win_size=1024,
            )
            mel_loss = F.l1_loss(mel_pred, mel_gt)
            mel_loss_weight = getattr(self.cfg.model.acoustic_codec.decoder, "mel_loss_weight", 45.0)
            total_loss += mel_loss * mel_loss_weight
            valid_losses["mel_loss"] = mel_loss.item()

        return (total_loss.item(), valid_losses, valid_stats)

    # def _valid_epoch(self):
    #     """Validation epoch for acoustic codec"""
    #     # Set both encoder and decoder to eval mode
    #     if hasattr(self.model, 'module'):
    #         self.model.module['encoder'].eval()
    #         self.model.module['decoder'].eval()
    #     else:
    #         self.model['encoder'].eval()
    #         self.model['decoder'].eval()

    #     epoch_sum_loss = 0.0
    #     epoch_losses = {}
    #     for batch in self.valid_dataloader:
    #         # Put the data to cuda device
    #         device = self.accelerator.device
    #         for k, v in batch.items():
    #             if isinstance(v, torch.Tensor):
    #                 batch[k] = v.to(device)

    #         total_loss, valid_losses, valid_stats = self._valid_step(batch)
    #         epoch_sum_loss += total_loss
    #         if isinstance(valid_losses, dict):
    #             for key, value in valid_losses.items():
    #                 if key not in epoch_losses.keys():
    #                     epoch_losses[key] = value
    #                 else:
    #                     epoch_losses[key] += value

    #     epoch_sum_loss = epoch_sum_loss / len(self.valid_dataloader)
    #     for key in epoch_losses.keys():
    #         epoch_losses[key] = epoch_losses[key] / len(self.valid_dataloader)

    #     self.accelerator.wait_for_everyone()

    #     return epoch_sum_loss, epoch_losses

    def _save_auxiliary_states(self):
        """Save auxiliary states for acoustic codec checkpoint"""
        # Acoustic codec doesn't need special auxiliary states
        pass

    def save_checkpoint(self):
        """Override save_checkpoint to save encoder and decoder separately"""
        if self.accelerator.is_main_process:
            keep_last = self.keep_last[0]
            # 读取self.checkpoint_dir所有的folder
            all_ckpts = os.listdir(self.checkpoint_dir)

            all_ckpts = filter(lambda x: x.startswith("epoch"), all_ckpts)
            all_ckpts = list(all_ckpts)
            if len(all_ckpts) > keep_last:
                # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                all_ckpts = sorted(
                    all_ckpts, key=lambda x: int(x.split("_")[1].split("-")[1])
                )
                for ckpt in all_ckpts[:-keep_last]:
                    import shutil
                    shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))

            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)
            self.logger.info("Saving state to {}...".format(path))

            # Save encoder and decoder separately (like in the pretrained model)
            encoder_path = os.path.join(path, "model.safetensors")  # encoder
            decoder_path = os.path.join(path, "model_1.safetensors")  # decoder

            import safetensors.torch
            os.makedirs(path, exist_ok=True)

            # Save encoder
            self.logger.info("Saving encoder to {}...".format(encoder_path))
            # Handle both DDP and non-DDP cases
            if hasattr(self.model, 'module'):
                encoder_state = self.accelerator.get_state_dict(self.model.module['encoder'])
            else:
                encoder_state = self.accelerator.get_state_dict(self.model['encoder'])
            safetensors.torch.save_file(encoder_state, encoder_path)

            # Save decoder
            self.logger.info("Saving decoder to {}...".format(decoder_path))
            # Handle both DDP and non-DDP cases
            if hasattr(self.model, 'module'):
                decoder_state = self.accelerator.get_state_dict(self.model.module['decoder'])
            else:
                decoder_state = self.accelerator.get_state_dict(self.model['decoder'])
            safetensors.torch.save_file(decoder_state, decoder_path)

            # Also save optimizer and scheduler states
            import torch
            optimizer_path = os.path.join(path, "optimizer.bin")
            scheduler_path = os.path.join(path, "scheduler.bin")

            torch.save(self.optimizer.state_dict(), optimizer_path)
            torch.save(self.scheduler.state_dict(), scheduler_path)

            self.logger.info("Finished saving acoustic codec state.")
