# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.tts.base.dataset_factory import get_maskgct_dataset_class
from models.tts.maskgct.maskgct_s2a import MaskGCT_S2A
from models.base.base_trainer import BaseTrainer


class S2ATrainer(BaseTrainer):
    """Trainer for MaskGCT-S2A model.
    
    The S2A model predicts acoustic tokens conditioned on semantic tokens.
    """
    
    def __init__(self, args, cfg):
        super(S2ATrainer, self).__init__(args, cfg)
        
        # Setup pretrained models for feature extraction
        self._build_input_models()
    
    def _build_model(self):
        """Build S2A model"""
        # Determine which S2A model to use (1layer or full)
        model_type = getattr(self.cfg.model, "s2a_model_type", "full")
        if model_type == "1layer":
            model = MaskGCT_S2A(cfg=self.cfg.model.s2a_model.s2a_1layer)
        else:
            model = MaskGCT_S2A(cfg=self.cfg.model.s2a_model.s2a_full)

        # Configure gradient checkpointing for memory efficiency
        # MaskGCT_S2A uses DiffLlama as diff_estimator
        if model_type == "1layer":
            gradient_checkpointing = getattr(self.cfg.model.s2a_model.s2a_1layer, 'gradient_checkpointing', False)
        else:
            gradient_checkpointing = getattr(self.cfg.model.s2a_model.s2a_full, 'gradient_checkpointing', False)

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
        """Build pretrained models for extracting semantic and acoustic tokens"""
        # Load semantic codec for extracting semantic tokens
        if hasattr(self.cfg.model, "semantic_codec") and hasattr(self.cfg.model.semantic_codec, "pretrained_path"):
            from models.codec.kmeans.repcodec_model import RepCodec
            self.semantic_codec = RepCodec(cfg=self.cfg.model.semantic_codec)
            pretrained_path = self.cfg.model.semantic_codec.pretrained_path
            if pretrained_path:
                import os
                import safetensors
                # Check if path is a directory (accelerate checkpoint) or file
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
        
        # Load acoustic codec for extracting acoustic tokens
        if hasattr(self.cfg.model, "acoustic_codec") and hasattr(self.cfg.model.acoustic_codec, "pretrained_path"):
            from models.codec.amphion_codec.codec import CodecEncoder, CodecDecoder
            self.codec_encoder = CodecEncoder(cfg=self.cfg.model.acoustic_codec.encoder)
            self.codec_decoder = CodecDecoder(cfg=self.cfg.model.acoustic_codec.decoder)
            
            # Load pretrained weights
            pretrained_path = self.cfg.model.acoustic_codec.pretrained_path
            if pretrained_path:
                import safetensors
                import os
                # Try safetensors first, then pytorch_model.bin
                encoder_path = os.path.join(pretrained_path, "model.safetensors")
                decoder_path = os.path.join(pretrained_path, "model_1.safetensors")
                if not os.path.exists(encoder_path):
                    encoder_path = os.path.join(pretrained_path, "pytorch_model.bin")
                if not os.path.exists(decoder_path):
                    decoder_path = os.path.join(pretrained_path, "pytorch_model_1.bin")
                
                if os.path.exists(encoder_path):
                    if encoder_path.endswith(".safetensors"):
                        safetensors.torch.load_model(self.codec_encoder, encoder_path)
                    else:
                        checkpoint = torch.load(encoder_path, map_location=self.accelerator.device)
                        if isinstance(checkpoint, dict) and "model" in checkpoint:
                            self.codec_encoder.load_state_dict(checkpoint["model"], strict=False)
                        else:
                            self.codec_encoder.load_state_dict(checkpoint, strict=False)
                
                if os.path.exists(decoder_path):
                    if decoder_path.endswith(".safetensors"):
                        safetensors.torch.load_model(self.codec_decoder, decoder_path)
                    else:
                        checkpoint = torch.load(decoder_path, map_location=self.accelerator.device)
                        if isinstance(checkpoint, dict) and "model" in checkpoint:
                            self.codec_decoder.load_state_dict(checkpoint["model"], strict=False)
                        else:
                            self.codec_decoder.load_state_dict(checkpoint, strict=False)
            
            self.codec_encoder.eval()
            self.codec_decoder.eval()
            self.codec_encoder.to(self.accelerator.device)
            self.codec_decoder.to(self.accelerator.device)
    
    @torch.no_grad()
    def _extract_semantic_tokens(self, wavs_16k):
        """Extract semantic tokens from audio"""
        if not hasattr(self, 'semantic_codec'):
            return None
        
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
        feat = outputs.hidden_states[17]
        
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)
        
        semantic_tokens, _ = self.semantic_codec.quantize(feat)
        return semantic_tokens
    
    @torch.no_grad()
    def _extract_acoustic_tokens(self, wavs_24k):
        """Extract acoustic tokens from audio"""
        if not hasattr(self, 'codec_encoder'):
            return None
        
        # Add channel dimension
        wavs_24k = wavs_24k.unsqueeze(1)  # [B, 1, T]
        
        # Encode
        encoded = self.codec_encoder(wavs_24k)
        
        # Quantize
        _, vq, _, _, _ = self.codec_decoder.quantizer(encoded)
        
        # Reshape: [num_quantizers, B, T] -> [B, T, num_quantizers]
        acoustic_tokens = vq.permute(1, 2, 0)
        return acoustic_tokens
    
    def _build_dataset(self):
        """Build dataset for training"""
        return get_maskgct_dataset_class(self.cfg)
    
    def _train_step(self, batch):
        """Training step for S2A model"""
        train_losses = {}
        total_loss = 0
        train_stats = {}
        
        # Get acoustic tokens (ground truth)
        acoustic_tokens = batch.get("acoustic_code", None)  # [B, T, num_quantizers]
        if acoustic_tokens is None:
            # Extract from audio if not provided
            wav_24k = batch.get("wav_24k", None)
            if wav_24k is None:
                wav_24k = batch["wav"]
                # Resample to 24kHz if needed
                if self.cfg.preprocess.sample_rate != 24000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(
                        self.cfg.preprocess.sample_rate, 24000
                    ).to(wav_24k.device)
                    wav_24k = resampler(wav_24k)
            acoustic_tokens = self._extract_acoustic_tokens(wav_24k)
            if acoustic_tokens is None:
                raise ValueError("Cannot extract acoustic tokens. Please provide pretrained acoustic codec.")
        
        # Get semantic tokens (condition)
        semantic_tokens = batch.get("semantic_code", None)  # [B, T]
        if semantic_tokens is None:
            # Extract from batch data if not provided
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")
        
        # Get masks
        acoustic_mask = batch.get("acoustic_mask", None)  # [B, T], 1 for valid, 0 for padding
        if acoustic_mask is None:
            acoustic_mask = torch.ones(acoustic_tokens.shape[0], acoustic_tokens.shape[1], dtype=torch.float32).to(acoustic_tokens.device)
        
        # torch.cuda.empty_cache()
        
        # Forward through S2A model
        logits, mask_layer, final_mask, x0, prompt_len, mask_prob = self.model(
            acoustic_tokens, acoustic_mask, semantic_tokens
        )  # logits: [B, T, codebook_size], mask_layer: which layer is being predicted
        
        # Compute cross-entropy loss on masked tokens
        # x0: [B, T, num_quantizers], we need to select the target layer
        target_layer = mask_layer.item() if isinstance(mask_layer, torch.Tensor) else mask_layer
        targets = x0[:, :, target_layer]  # [B, T]
        
        logits_reshaped = logits.reshape(-1, logits.shape[-1])  # [B*T, codebook_size]
        targets_reshaped = targets.reshape(-1)  # [B*T]
        mask = final_mask.reshape(-1)  # [B*T]
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_reshaped, targets_reshaped, reduction='none')
        ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)
        
        total_loss += ce_loss
        train_losses["ce_loss"] = ce_loss
        train_losses["mask_layer"] = target_layer
        
        # Compute token accuracy for monitoring
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)  # [B, T]
            correct = (predictions == targets).float()
            # Accuracy on masked tokens only
            masked_correct = (correct * final_mask.squeeze(-1)).sum()
            masked_total = final_mask.sum()
            token_accuracy = masked_correct / (masked_total + 1e-8)
            train_stats["token_accuracy"] = token_accuracy.item()
            
            # Top-5 accuracy
            top5_preds = logits.topk(5, dim=-1).indices  # [B, T, 5]
            top5_correct = (top5_preds == targets.unsqueeze(-1)).any(dim=-1).float()
            top5_masked_correct = (top5_correct * final_mask.squeeze(-1)).sum()
            top5_accuracy = top5_masked_correct / (masked_total + 1e-8)
            train_stats["top5_accuracy"] = top5_accuracy.item()
            
            # Logits statistics
            train_stats["logits_mean"] = logits.mean().item()
            train_stats["logits_std"] = logits.std().item()
            train_stats["logits_max"] = logits.max().item()
            train_stats["logits_min"] = logits.min().item()
            
            # Prediction confidence
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            train_stats["pred_confidence"] = (max_probs * final_mask.squeeze(-1)).sum() / (masked_total + 1e-8)
            train_stats["pred_confidence"] = train_stats["pred_confidence"].item()
        
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
        train_losses["mask_ratio"] = mask_prob.mean().item() if isinstance(mask_prob, torch.Tensor) else mask_prob
        
        # Add batch statistics
        train_stats["batch_size"] = acoustic_tokens.shape[0]
        train_stats["seq_length"] = acoustic_tokens.shape[1]
        train_stats["num_quantizers"] = acoustic_tokens.shape[2]
        train_stats["semantic_length"] = semantic_tokens.shape[1]
        train_stats["masked_tokens"] = masked_total.item()

        return (total_loss.item(), train_losses, train_stats)

    def _valid_step(self, batch):
        """Validation step for S2A model"""
        valid_losses = {}
        total_loss = 0
        valid_stats = {}

        # Get acoustic tokens (ground truth)
        acoustic_tokens = batch.get("acoustic_code", None)  # [B, T, num_quantizers]
        if acoustic_tokens is None:
            # Extract from audio if not provided
            wav_24k = batch.get("wav_24k", None)
            if wav_24k is None:
                wav_24k = batch["wav"]
                # Resample to 24kHz if needed
                if self.cfg.preprocess.sample_rate != 24000:
                    import torchaudio
                    resampler = torchaudio.transforms.Resample(
                        self.cfg.preprocess.sample_rate, 24000
                    ).to(wav_24k.device)
                    wav_24k = resampler(wav_24k)
            acoustic_tokens = self._extract_acoustic_tokens(wav_24k)
            if acoustic_tokens is None:
                raise ValueError("Cannot extract acoustic tokens. Please provide pretrained acoustic codec.")

        # Get semantic tokens (condition)
        semantic_tokens = batch.get("semantic_code", None)  # [B, T]
        if semantic_tokens is None:
            # Extract from batch data if not provided
            semantic_tokens = self._extract_semantic_tokens(batch)
            if semantic_tokens is None:
                raise ValueError("Cannot extract semantic tokens. Please provide pretrained semantic codec.")

        # Get masks
        acoustic_mask = batch.get("acoustic_mask", None)  # [B, T], 1 for valid, 0 for padding
        if acoustic_mask is None:
            acoustic_mask = torch.ones(acoustic_tokens.shape[0], acoustic_tokens.shape[1], dtype=torch.float32).to(acoustic_tokens.device)

        # torch.cuda.empty_cache()

        # Forward through S2A model (no gradients)
        with torch.no_grad():
            logits, mask_layer, final_mask, x0, prompt_len, mask_prob = self.model(
                acoustic_tokens, acoustic_mask, semantic_tokens
            )  # logits: [B, T, codebook_size], mask_layer: which layer is being predicted

        # Compute cross-entropy loss on masked tokens
        # x0: [B, T, num_quantizers], we need to select the target layer
        target_layer = mask_layer.item() if isinstance(mask_layer, torch.Tensor) else mask_layer
        targets = x0[:, :, target_layer]  # [B, T]

        logits = logits.reshape(-1, logits.shape[-1])  # [B*T, codebook_size]
        targets = targets.reshape(-1)  # [B*T]
        mask = final_mask.reshape(-1)  # [B*T]

        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        ce_loss = (ce_loss * mask).sum() / (mask.sum() + 1e-8)

        total_loss += ce_loss
        valid_losses["ce_loss"] = ce_loss.item()
        valid_losses["mask_layer"] = target_layer
        valid_losses["mask_ratio"] = mask_prob.mean().item() if isinstance(mask_prob, torch.Tensor) else mask_prob

        return (total_loss.item(), valid_losses, valid_stats)

    def _valid_epoch(self):
        """Validation epoch for S2A model"""
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

