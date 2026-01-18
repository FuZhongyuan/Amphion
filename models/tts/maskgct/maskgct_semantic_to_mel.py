# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Semantic-to-Mel Model for MaskGCT

This model takes codebook-quantized discrete semantic units as input,
incorporates acoustic information from a reference audio (for zero-shot voice cloning),
and generates mel spectrograms using a diffusion-based approach.

The generated mel spectrograms can be converted to waveforms using HiFi-GAN or similar vocoders.
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from models.tts.maskgct.llama_nar import DiffLlama, DiffLlamaPrefix


def top_k(logits, thres=0.9):
    """Apply top-k filtering to logits."""
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = logits.topk(k, dim=-1)
    probs = torch.full_like(logits, float("-inf"))
    probs.scatter_(2, ind, val)
    return probs


def log(t, eps=1e-10):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.0, dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class MelProjection(nn.Module):
    """Projects hidden states to mel spectrogram dimensions."""

    def __init__(self, hidden_size, n_mel, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_size, n_mel))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class AcousticEncoder(nn.Module):
    """
    Encodes reference audio mel spectrogram into acoustic embeddings.
    This captures speaker characteristics for zero-shot voice cloning.
    """

    def __init__(self, n_mel=80, hidden_size=1024, num_layers=4, num_heads=8):
        super().__init__()
        self.n_mel = n_mel
        self.hidden_size = hidden_size

        # Project mel to hidden size
        self.mel_proj = nn.Linear(n_mel, hidden_size)

        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, 4096, hidden_size) * 0.02)

        # Transformer encoder for acoustic features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling to get speaker embedding
        self.global_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, mel, mel_mask=None):
        """
        Args:
            mel: [B, T, n_mel] mel spectrogram of reference audio
            mel_mask: [B, T] mask (1 for valid, 0 for padding)

        Returns:
            acoustic_emb: [B, hidden_size] global speaker embedding
            acoustic_seq: [B, T, hidden_size] sequence-level acoustic features
        """
        B, T, _ = mel.shape

        # Project mel to hidden size
        x = self.mel_proj(mel)  # [B, T, hidden_size]

        # Add positional encoding
        x = x + self.pos_emb[:, :T, :]

        # Create attention mask for transformer
        if mel_mask is not None:
            # Convert to attention mask format (True = ignore)
            attn_mask = ~mel_mask.bool()
        else:
            attn_mask = None

        # Encode
        acoustic_seq = self.encoder(x, src_key_padding_mask=attn_mask)  # [B, T, hidden_size]

        # Global pooling for speaker embedding
        if mel_mask is not None:
            # Masked mean pooling
            mask_expanded = mel_mask.unsqueeze(-1)  # [B, T, 1]
            acoustic_emb = (acoustic_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            acoustic_emb = acoustic_seq.mean(dim=1)  # [B, hidden_size]

        acoustic_emb = self.global_proj(acoustic_emb)  # [B, hidden_size]

        return acoustic_emb, acoustic_seq


class SemanticToMel(nn.Module):
    """
    Semantic-to-Mel diffusion model.

    Takes semantic tokens and reference audio, generates mel spectrograms.
    Uses masked diffusion similar to MaskGCT for iterative refinement.
    """

    def __init__(
        self,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        n_mel=80,
        semantic_codebook_size=8192,
        cfg_scale=0.15,
        use_acoustic_prompt=True,
        acoustic_encoder_layers=4,
        mel_proj_layers=2,
        cfg=None,
    ):
        super().__init__()

        # Parse config
        hidden_size = cfg.hidden_size if cfg is not None and hasattr(cfg, "hidden_size") else hidden_size
        num_layers = cfg.num_layers if cfg is not None and hasattr(cfg, "num_layers") else num_layers
        num_heads = cfg.num_heads if cfg is not None and hasattr(cfg, "num_heads") else num_heads
        n_mel = cfg.n_mel if cfg is not None and hasattr(cfg, "n_mel") else n_mel
        semantic_codebook_size = cfg.semantic_codebook_size if cfg is not None and hasattr(cfg, "semantic_codebook_size") else semantic_codebook_size
        cfg_scale = cfg.cfg_scale if cfg is not None and hasattr(cfg, "cfg_scale") else cfg_scale
        use_acoustic_prompt = cfg.use_acoustic_prompt if cfg is not None and hasattr(cfg, "use_acoustic_prompt") else use_acoustic_prompt
        acoustic_encoder_layers = cfg.acoustic_encoder_layers if cfg is not None and hasattr(cfg, "acoustic_encoder_layers") else acoustic_encoder_layers
        mel_proj_layers = cfg.mel_proj_layers if cfg is not None and hasattr(cfg, "mel_proj_layers") else mel_proj_layers

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.n_mel = n_mel
        self.semantic_codebook_size = semantic_codebook_size
        self.cfg_scale = cfg_scale
        self.use_acoustic_prompt = use_acoustic_prompt

        # Semantic token embedding
        self.semantic_emb = nn.Embedding(semantic_codebook_size, hidden_size)

        # Acoustic encoder for reference audio
        if use_acoustic_prompt:
            self.acoustic_encoder = AcousticEncoder(
                n_mel=n_mel,
                hidden_size=hidden_size,
                num_layers=acoustic_encoder_layers,
                num_heads=num_heads,
            )
            # Acoustic conditioning projection
            self.acoustic_cond_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.SiLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            )

        # Mel input projection (for noisy mel during diffusion)
        self.mel_input_proj = nn.Linear(n_mel, hidden_size)

        # Mel output projection
        self.mel_output_proj = MelProjection(hidden_size, n_mel, num_layers=mel_proj_layers)

        # Diffusion backbone (using DiffLlama for consistency with MaskGCT)
        self.diff_estimator = DiffLlama(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        def _reset_parameters(m):
            if isinstance(m, nn.MultiheadAttention):
                if m._qkv_same_embed_dim:
                    nn.init.normal_(m.in_proj_weight, std=0.02)
                else:
                    nn.init.normal_(m.q_proj_weight, std=0.02)
                    nn.init.normal_(m.k_proj_weight, std=0.02)
                    nn.init.normal_(m.v_proj_weight, std=0.02)
                if m.in_proj_bias is not None:
                    nn.init.constant_(m.in_proj_bias, 0.0)
                    nn.init.constant_(m.out_proj.bias, 0.0)
            elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    def get_noise_schedule(self, t):
        """Cosine noise schedule for diffusion."""
        return torch.cos(t * math.pi / 2)

    def forward_diffusion(self, mel_target, semantic_tokens, t, acoustic_emb=None):
        """
        Forward diffusion process: add noise to mel spectrogram.

        Args:
            mel_target: [B, T, n_mel] target mel spectrogram
            semantic_tokens: [B, T] semantic token indices
            t: [B] diffusion timestep (0 to 1)
            acoustic_emb: [B, hidden_size] acoustic embedding from reference

        Returns:
            mel_noisy: [B, T, n_mel] noisy mel spectrogram
            noise: [B, T, n_mel] the noise added
        """
        B, T, _ = mel_target.shape

        # Get noise level from schedule
        alpha = self.get_noise_schedule(t)  # [B]
        alpha = alpha.view(B, 1, 1)  # [B, 1, 1]

        # Sample noise
        noise = torch.randn_like(mel_target)

        # Add noise: x_t = alpha * x_0 + (1 - alpha) * noise
        mel_noisy = alpha * mel_target + (1 - alpha) * noise

        return mel_noisy, noise

    def compute_loss(self, mel_target, semantic_tokens, semantic_mask,
                     ref_mel=None, ref_mel_mask=None):
        """
        Compute training loss.

        Args:
            mel_target: [B, T, n_mel] target mel spectrogram
            semantic_tokens: [B, T] semantic token indices
            semantic_mask: [B, T] mask (1 for valid, 0 for padding)
            ref_mel: [B, T_ref, n_mel] reference mel for acoustic conditioning
            ref_mel_mask: [B, T_ref] reference mel mask

        Returns:
            loss: scalar loss value
            loss_dict: dictionary of individual losses
        """
        B, T, _ = mel_target.shape

        # Sample random timesteps
        t = torch.rand(B, device=mel_target.device)
        t = torch.clamp(t, 1e-5, 1.0)

        # Get acoustic embedding from reference
        # Always compute acoustic embedding to ensure gradients flow through acoustic_encoder
        acoustic_emb = None
        cfg_drop_mask = None
        if self.use_acoustic_prompt and ref_mel is not None:
            acoustic_emb, _ = self.acoustic_encoder(ref_mel, ref_mel_mask)

            # Classifier-free guidance: randomly drop acoustic conditioning per sample
            if self.training:
                # Create per-sample drop mask
                cfg_drop_mask = (torch.rand(B, device=mel_target.device) < self.cfg_scale).float()
                # Zero out acoustic embedding for dropped samples (but keep gradient flow)
                acoustic_emb = acoustic_emb * (1 - cfg_drop_mask.unsqueeze(-1))

        # Forward diffusion
        mel_noisy, noise = self.forward_diffusion(mel_target, semantic_tokens, t, acoustic_emb)

        # Predict noise (or clean mel)
        mel_pred = self.denoise(mel_noisy, semantic_tokens, semantic_mask, t, acoustic_emb)

        # Compute loss (predict clean mel directly)
        # L1 loss on mel spectrogram
        loss_mel = F.l1_loss(mel_pred, mel_target, reduction='none')

        # Apply mask
        mask_expanded = semantic_mask.unsqueeze(-1)  # [B, T, 1]
        loss_mel = (loss_mel * mask_expanded).sum() / (mask_expanded.sum() * self.n_mel + 1e-8)

        # Optional: add spectral convergence loss
        loss_sc = self.spectral_convergence_loss(mel_pred, mel_target, semantic_mask)

        total_loss = loss_mel + 0.5 * loss_sc

        loss_dict = {
            "mel_loss": loss_mel,
            "sc_loss": loss_sc,
            "total_loss": total_loss,
        }

        return total_loss, loss_dict

    def spectral_convergence_loss(self, pred, target, mask):
        """Spectral convergence loss for better mel quality."""
        mask_expanded = mask.unsqueeze(-1)

        # Frobenius norm of difference / Frobenius norm of target
        diff = (pred - target) * mask_expanded
        target_masked = target * mask_expanded

        sc_loss = torch.norm(diff, p='fro') / (torch.norm(target_masked, p='fro') + 1e-8)
        return sc_loss

    def denoise(self, mel_noisy, semantic_tokens, semantic_mask, t, acoustic_emb=None):
        """
        Denoise mel spectrogram given semantic tokens and timestep.

        Args:
            mel_noisy: [B, T, n_mel] noisy mel spectrogram
            semantic_tokens: [B, T] semantic token indices
            semantic_mask: [B, T] mask (1 for valid, 0 for padding)
            t: [B] diffusion timestep
            acoustic_emb: [B, hidden_size] acoustic embedding

        Returns:
            mel_pred: [B, T, n_mel] predicted clean mel spectrogram
        """
        B, T, _ = mel_noisy.shape

        # Get semantic embeddings
        semantic_emb = self.semantic_emb(semantic_tokens)  # [B, T, hidden_size]

        # Project noisy mel to hidden size
        mel_hidden = self.mel_input_proj(mel_noisy)  # [B, T, hidden_size]

        # Combine semantic and mel information
        x = semantic_emb + mel_hidden  # [B, T, hidden_size]

        # Add acoustic conditioning
        if acoustic_emb is not None:
            acoustic_cond = self.acoustic_cond_proj(acoustic_emb)  # [B, hidden_size]
            x = x + acoustic_cond.unsqueeze(1)  # [B, T, hidden_size]

        # Create condition for diffusion (semantic embeddings)
        cond = semantic_emb
        if acoustic_emb is not None:
            cond = cond + acoustic_cond.unsqueeze(1)

        # Run through diffusion backbone
        hidden = self.diff_estimator(x, t, cond, semantic_mask)  # [B, T, hidden_size]

        # Project to mel
        mel_pred = self.mel_output_proj(hidden)  # [B, T, n_mel]

        return mel_pred

    @torch.no_grad()
    def generate(
        self,
        semantic_tokens,
        semantic_mask=None,
        ref_mel=None,
        ref_mel_mask=None,
        n_timesteps=50,
        cfg_scale=2.0,
        temperature=1.0,
    ):
        """
        Generate mel spectrogram from semantic tokens.

        Args:
            semantic_tokens: [B, T] semantic token indices
            semantic_mask: [B, T] mask (1 for valid, 0 for padding)
            ref_mel: [B, T_ref, n_mel] reference mel for voice cloning
            ref_mel_mask: [B, T_ref] reference mel mask
            n_timesteps: number of diffusion steps
            cfg_scale: classifier-free guidance scale
            temperature: sampling temperature

        Returns:
            mel: [B, T, n_mel] generated mel spectrogram
        """
        B, T = semantic_tokens.shape
        device = semantic_tokens.device

        if semantic_mask is None:
            semantic_mask = torch.ones(B, T, device=device)

        # Get acoustic embedding from reference
        acoustic_emb = None
        if self.use_acoustic_prompt and ref_mel is not None:
            acoustic_emb, _ = self.acoustic_encoder(ref_mel, ref_mel_mask)

        # Initialize with noise
        mel = torch.randn(B, T, self.n_mel, device=device) * temperature

        # Diffusion sampling (DDPM-style)
        timesteps = torch.linspace(1.0, 0.0, n_timesteps + 1, device=device)

        for i in range(n_timesteps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = t_curr.expand(B)

            # Predict clean mel
            mel_pred = self.denoise(mel, semantic_tokens, semantic_mask, t_batch, acoustic_emb)

            # Classifier-free guidance
            if cfg_scale > 1.0 and acoustic_emb is not None:
                mel_pred_uncond = self.denoise(mel, semantic_tokens, semantic_mask, t_batch, None)
                mel_pred = mel_pred_uncond + cfg_scale * (mel_pred - mel_pred_uncond)

            # Get noise levels
            alpha_curr = self.get_noise_schedule(t_curr)
            alpha_next = self.get_noise_schedule(t_next)

            if t_next > 0:
                # DDPM update step
                # x_t = alpha * x_0 + (1 - alpha) * noise
                # Estimate noise from current state
                noise_pred = (mel - alpha_curr * mel_pred) / (1 - alpha_curr + 1e-8)

                # Update to next timestep
                mel = alpha_next * mel_pred + (1 - alpha_next) * noise_pred

                # Add small noise for stochasticity (optional)
                if i < n_timesteps - 1:
                    noise_scale = 0.1 * (1 - alpha_next)
                    mel = mel + noise_scale * torch.randn_like(mel)
            else:
                # Final step: use predicted clean mel
                mel = mel_pred

        return mel

    def forward(self, mel_target, semantic_tokens, semantic_mask,
                ref_mel=None, ref_mel_mask=None):
        """
        Forward pass for training.

        Args:
            mel_target: [B, T, n_mel] target mel spectrogram
            semantic_tokens: [B, T] semantic token indices
            semantic_mask: [B, T] mask (1 for valid, 0 for padding)
            ref_mel: [B, T_ref, n_mel] reference mel for acoustic conditioning
            ref_mel_mask: [B, T_ref] reference mel mask

        Returns:
            loss: scalar loss value
            loss_dict: dictionary of individual losses
        """
        return self.compute_loss(mel_target, semantic_tokens, semantic_mask,
                                 ref_mel, ref_mel_mask)


class SemanticToMelWithPrompt(SemanticToMel):
    """
    Extended SemanticToMel model with prompt-based generation.

    Supports using a portion of the target as prompt for better continuity.
    Note: This class inherits all functionality from SemanticToMel and adds
    prompt-based generation methods. No additional parameters are added to
    avoid DDP unused parameter issues.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # No additional parameters - reuse acoustic_encoder for prompt encoding
        # This avoids DDP unused parameter issues during training

    @torch.no_grad()
    def generate_with_prompt(
        self,
        semantic_tokens,
        prompt_mel,
        prompt_semantic_tokens,
        semantic_mask=None,
        prompt_mask=None,
        ref_mel=None,
        ref_mel_mask=None,
        n_timesteps=50,
        cfg_scale=2.0,
        temperature=1.0,
    ):
        """
        Generate mel spectrogram with mel prompt for better continuity.

        Args:
            semantic_tokens: [B, T] semantic tokens for generation
            prompt_mel: [B, T_prompt, n_mel] mel prompt
            prompt_semantic_tokens: [B, T_prompt] semantic tokens for prompt
            semantic_mask: [B, T] mask for generation part
            prompt_mask: [B, T_prompt] mask for prompt
            ref_mel: [B, T_ref, n_mel] reference mel for voice cloning
            ref_mel_mask: [B, T_ref] reference mel mask
            n_timesteps: number of diffusion steps
            cfg_scale: classifier-free guidance scale
            temperature: sampling temperature

        Returns:
            mel: [B, T_prompt + T, n_mel] generated mel spectrogram (including prompt)
        """
        B, T = semantic_tokens.shape
        T_prompt = prompt_mel.shape[1]
        device = semantic_tokens.device

        if semantic_mask is None:
            semantic_mask = torch.ones(B, T, device=device)
        if prompt_mask is None:
            prompt_mask = torch.ones(B, T_prompt, device=device)

        # Concatenate prompt and generation parts
        full_semantic = torch.cat([prompt_semantic_tokens, semantic_tokens], dim=1)
        full_mask = torch.cat([prompt_mask, semantic_mask], dim=1)

        # Get acoustic embedding from reference
        acoustic_emb = None
        if self.use_acoustic_prompt and ref_mel is not None:
            acoustic_emb, _ = self.acoustic_encoder(ref_mel, ref_mel_mask)

        # Initialize: prompt is fixed, generation part is noise
        mel_gen = torch.randn(B, T, self.n_mel, device=device) * temperature

        # Diffusion sampling
        timesteps = torch.linspace(1.0, 0.0, n_timesteps + 1, device=device)

        for i in range(n_timesteps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_batch = t_curr.expand(B)

            # Concatenate prompt and current generation
            full_mel = torch.cat([prompt_mel, mel_gen], dim=1)

            # Predict clean mel
            mel_pred = self.denoise(full_mel, full_semantic, full_mask, t_batch, acoustic_emb)

            # Only update generation part
            mel_pred_gen = mel_pred[:, T_prompt:, :]

            # Classifier-free guidance
            if cfg_scale > 1.0 and acoustic_emb is not None:
                mel_pred_uncond = self.denoise(full_mel, full_semantic, full_mask, t_batch, None)
                mel_pred_gen_uncond = mel_pred_uncond[:, T_prompt:, :]
                mel_pred_gen = mel_pred_gen_uncond + cfg_scale * (mel_pred_gen - mel_pred_gen_uncond)

            # Get noise levels
            alpha_curr = self.get_noise_schedule(t_curr)
            alpha_next = self.get_noise_schedule(t_next)

            if t_next > 0:
                # DDPM update step
                noise_pred = (mel_gen - alpha_curr * mel_pred_gen) / (1 - alpha_curr + 1e-8)
                mel_gen = alpha_next * mel_pred_gen + (1 - alpha_next) * noise_pred

                if i < n_timesteps - 1:
                    noise_scale = 0.1 * (1 - alpha_next)
                    mel_gen = mel_gen + noise_scale * torch.randn_like(mel_gen)
            else:
                mel_gen = mel_pred_gen

        # Concatenate prompt and generated mel
        full_mel = torch.cat([prompt_mel, mel_gen], dim=1)

        return full_mel
