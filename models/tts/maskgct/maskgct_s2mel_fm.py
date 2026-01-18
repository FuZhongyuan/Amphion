# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flow Matching Semantic-to-Mel Model for MaskGCT

This model converts codebook-quantized discrete semantic units to continuous mel spectrograms
using Flow Matching. It supports zero-shot voice cloning by conditioning on prompt mel
spectrograms and semantic tokens.

Key features:
1. Flow Matching for continuous mel generation (more stable than diffusion)
2. Prompt-based voice cloning (timbre, F0, prosody)
3. Semantic token conditioning for content information
4. Classifier-free guidance (CFG) for improved quality
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange
from models.tts.maskgct.llama_nar import DiffLlama


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 1.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MelProjection(nn.Module):
    """Projects hidden states to/from mel spectrogram dimensions."""

    def __init__(self, hidden_size, mel_dim, num_layers=2):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_size, mel_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x):
        return self.proj(x)


class FlowMatchingS2Mel(nn.Module):
    """
    Flow Matching Semantic-to-Mel Model.

    Takes semantic tokens as content condition and prompt mel/semantic tokens
    for speaker information, generates target mel spectrogram using Flow Matching.

    Args:
        mel_dim: Mel spectrogram dimension (default: 100)
        hidden_size: Transformer hidden dimension (default: 1024)
        num_layers: Number of transformer layers (default: 16)
        num_heads: Number of attention heads (default: 16)
        semantic_codebook_size: Size of semantic token vocabulary (default: 8192)
        cfg_scale: Probability of CFG dropout during training (default: 0.2)
        sigma: Flow matching noise parameter (default: 1e-5)
        time_scheduler: Timestep scheduler ('linear' or 'cos') (default: 'linear')
        cfg: Config object (optional)
    """

    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        semantic_codebook_size=8192,
        cfg_scale=0.2,
        sigma=1e-5,
        time_scheduler="linear",
        min_prompt_ratio=0.1,
        max_prompt_ratio=0.5,
        cfg=None,
    ):
        super().__init__()

        # Parse config
        mel_dim = cfg.mel_dim if cfg is not None and hasattr(cfg, "mel_dim") else mel_dim
        hidden_size = cfg.hidden_size if cfg is not None and hasattr(cfg, "hidden_size") else hidden_size
        num_layers = cfg.num_layers if cfg is not None and hasattr(cfg, "num_layers") else num_layers
        num_heads = cfg.num_heads if cfg is not None and hasattr(cfg, "num_heads") else num_heads
        semantic_codebook_size = cfg.semantic_codebook_size if cfg is not None and hasattr(cfg, "semantic_codebook_size") else semantic_codebook_size
        cfg_scale = cfg.cfg_scale if cfg is not None and hasattr(cfg, "cfg_scale") else cfg_scale
        sigma = cfg.sigma if cfg is not None and hasattr(cfg, "sigma") else sigma
        time_scheduler = cfg.time_scheduler if cfg is not None and hasattr(cfg, "time_scheduler") else time_scheduler
        min_prompt_ratio = cfg.min_prompt_ratio if cfg is not None and hasattr(cfg, "min_prompt_ratio") else min_prompt_ratio
        max_prompt_ratio = cfg.max_prompt_ratio if cfg is not None and hasattr(cfg, "max_prompt_ratio") else max_prompt_ratio

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.semantic_codebook_size = semantic_codebook_size
        self.cfg_scale = cfg_scale
        self.sigma = sigma
        self.time_scheduler = time_scheduler
        self.min_prompt_ratio = min_prompt_ratio
        self.max_prompt_ratio = max_prompt_ratio

        # Semantic token embedding
        self.semantic_emb = nn.Embedding(semantic_codebook_size, hidden_size)

        # Mel input projection (for noisy mel during flow matching)
        self.mel_input_mlp = nn.Sequential(
            nn.Linear(mel_dim, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        # Mel output projection
        self.mel_output_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, mel_dim),
        )

        # Flow backbone (DiffLlama from MaskGCT)
        self.diff_estimator = DiffLlama(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Initialize weights
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters."""
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

    @torch.no_grad()
    def forward_diffusion(self, x, t):
        """
        Forward diffusion (flow) process.

        Flow Matching: x_t = (1 - (1 - sigma) * t) * z + t * x
        where z ~ N(0, 1) and x is the target mel.

        Args:
            x: Target mel spectrogram (B, T, mel_dim)
            t: Diffusion timestep (B,)

        Returns:
            xt: Noisy mel (B, T, mel_dim)
            z: Noise (B, T, mel_dim)
            new_t: Timestep (B,)
            prompt_len: Prompt lengths (B,)
            mask: Mask for loss computation (B, T, 1)
        """
        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        # Sample noise
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)

        # Determine prompt length
        cfg_scale = self.cfg_scale
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if torch.rand(1) > cfg_scale:
            # Use prompt (most training steps)
            min_prompt = max(int(seq_len * self.min_prompt_ratio), 1)
            max_prompt = max(int(seq_len * self.max_prompt_ratio), min_prompt + 1)
            prompt_len = torch.randint(min_prompt, max_prompt, (batch_size,)).to(x.device)
        else:
            # CFG dropout: no prompt
            prompt_len = torch.zeros(batch_size, dtype=torch.long).to(x.device)

        # Create is_prompt mask (1 for prompt positions, 0 for target)
        is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
        col_indices = torch.arange(is_prompt.shape[1]).repeat(is_prompt.shape[0], 1).to(prompt_len)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # (B, T)

        # Create generation mask (1 for positions to generate, 0 for prompt)
        mask = torch.ones_like(x[:, :, 0])
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]  # (B, T, 1)

        # Flow matching: x_t = (1 - (1 - sigma) * t) * z + t * x
        # For prompt positions, keep original x (no noise)
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def compute_loss(
        self,
        mel_target,
        semantic_tokens,
        x_mask,
    ):
        """
        Compute flow matching loss.

        Args:
            mel_target: Target mel spectrogram (B, T, mel_dim)
            semantic_tokens: Semantic token indices (B, T)
            x_mask: Mask for valid positions (B, T), 1 for valid, 0 for padding

        Returns:
            noise: z (B, T, mel_dim)
            x: Target mel (B, T, mel_dim)
            flow_pred: Predicted flow (B, T, mel_dim)
            final_mask: Mask for loss computation (B, T, 1)
            prompt_len: Prompt lengths (B,)
        """
        # Sample random timesteps
        t = torch.rand(mel_target.shape[0], device=mel_target.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)

        # Apply time scheduler
        if self.time_scheduler == "cos":
            # Cosine scheduler: harder at beginning
            t = 1 - torch.cos(t * math.pi * 0.5)

        # Forward diffusion
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(mel_target, t)

        noise = z

        # Get semantic embeddings as condition
        cond = self.semantic_emb(semantic_tokens)  # (B, T, hidden_size)

        # Drop condition for CFG when prompt_len is 0
        cond = cond * torch.where(
            prompt_len > 0,
            torch.ones_like(prompt_len, dtype=cond.dtype),
            torch.zeros_like(prompt_len, dtype=cond.dtype),
        ).unsqueeze(-1).unsqueeze(-1).to(cond.device)

        # Project noisy mel to hidden space
        mel_hidden = self.mel_input_mlp(xt)  # (B, T, hidden_size)

        # Run through flow estimator
        hidden_states = self.diff_estimator(mel_hidden, new_t, cond, x_mask)  # (B, T, hidden_size)

        # Project back to mel space
        flow_pred = self.mel_output_mlp(hidden_states)  # (B, T, mel_dim)

        # Final mask for loss computation
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return noise, mel_target, flow_pred, final_mask, prompt_len

    @torch.no_grad()
    def reverse_diffusion(
        self,
        semantic_tokens,
        prompt_mel,
        semantic_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        """
        Reverse diffusion (flow) process for inference.

        Args:
            semantic_tokens: Semantic token indices for full sequence (B, T_total)
            prompt_mel: Prompt mel spectrogram (B, T_prompt, mel_dim)
            semantic_mask: Mask for semantic tokens (B, T_total)
            prompt_mask: Mask for prompt (B, T_prompt)
            n_timesteps: Number of ODE solver steps
            cfg: CFG scale (0 means no CFG)
            rescale_cfg: CFG rescaling factor

        Returns:
            mel: Generated mel spectrogram (B, T_target, mel_dim)
        """
        h = 1.0 / n_timesteps
        prompt_len = prompt_mel.shape[1]
        total_len = semantic_tokens.shape[1]
        target_len = total_len - prompt_len

        batch_size = semantic_tokens.shape[0]
        device = semantic_tokens.device

        if semantic_mask is None:
            semantic_mask = torch.ones(batch_size, total_len, device=device)
        if prompt_mask is None:
            prompt_mask = torch.ones(batch_size, prompt_len, device=device)

        target_mask = semantic_mask[:, prompt_len:]

        # Get semantic condition
        cond = self.semantic_emb(semantic_tokens)  # (B, T_total, hidden_size)

        # Initialize target with noise
        z = torch.randn(
            (batch_size, target_len, self.mel_dim),
            dtype=prompt_mel.dtype,
            device=device,
            requires_grad=False,
        )
        xt = z

        # ODE solver: t from 0 to 1
        for i in range(n_timesteps):
            t = (0 + (i + 0.5) * h) * torch.ones(batch_size, dtype=z.dtype, device=device)

            # Concatenate prompt and current generation
            xt_input = torch.cat([prompt_mel, xt], dim=1)  # (B, T_total, mel_dim)
            xt_mask = torch.cat([prompt_mask, target_mask], dim=1)

            # Project mel to hidden space
            mel_hidden = self.mel_input_mlp(xt_input)  # (B, T_total, hidden_size)

            # Run through flow estimator
            hidden_states = self.diff_estimator(mel_hidden, t, cond, xt_mask)

            # Project back to mel and extract target part
            flow_pred = self.mel_output_mlp(hidden_states)
            flow_pred = flow_pred[:, prompt_len:, :]  # (B, T_target, mel_dim)

            # CFG
            if cfg > 0:
                # Unconditional prediction (no prompt context)
                mel_hidden_uncond = self.mel_input_mlp(xt)
                hidden_states_uncond = self.diff_estimator(
                    mel_hidden_uncond, t,
                    torch.zeros_like(cond)[:, prompt_len:, :],
                    target_mask
                )
                uncond_flow_pred = self.mel_output_mlp(hidden_states_uncond)

                # Apply CFG with rescaling
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescale_flow_pred = flow_pred_cfg * pos_flow_pred_std / (flow_pred_cfg.std() + 1e-8)
                flow_pred = rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg

            # Euler step
            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    def forward(
        self,
        mel_target,
        semantic_tokens,
        x_mask,
    ):
        """
        Forward pass for training.

        Args:
            mel_target: Target mel spectrogram (B, T, mel_dim)
            semantic_tokens: Semantic token indices (B, T)
            x_mask: Mask for valid positions (B, T)

        Returns:
            noise: z (B, T, mel_dim)
            x: Target mel (B, T, mel_dim)
            flow_pred: Predicted flow (B, T, mel_dim)
            final_mask: Mask for loss computation (B, T, 1)
            prompt_len: Prompt lengths (B,)
        """
        return self.compute_loss(mel_target, semantic_tokens, x_mask)


class FlowMatchingS2MelWithPrompt(FlowMatchingS2Mel):
    """
    Extended Flow Matching S2Mel model with explicit prompt input during inference.

    This variant allows passing prompt mel and semantic tokens separately,
    enabling more flexible inference scenarios for zero-shot voice cloning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        semantic_tokens,
        prompt_mel,
        prompt_semantic_tokens,
        semantic_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
        temperature=1.0,
    ):
        """
        Generate mel spectrogram with prompt-based voice cloning.

        Args:
            semantic_tokens: Semantic tokens for generation (B, T_target)
            prompt_mel: Prompt mel spectrogram (B, T_prompt, mel_dim)
            prompt_semantic_tokens: Semantic tokens for prompt (B, T_prompt)
            semantic_mask: Mask for target semantic tokens (B, T_target)
            prompt_mask: Mask for prompt (B, T_prompt)
            n_timesteps: Number of ODE solver steps
            cfg: CFG scale
            rescale_cfg: CFG rescaling factor
            temperature: Initial noise temperature

        Returns:
            mel: Generated mel spectrogram (B, T_target, mel_dim)
        """
        batch_size = semantic_tokens.shape[0]
        target_len = semantic_tokens.shape[1]
        prompt_len = prompt_mel.shape[1]
        device = semantic_tokens.device

        if semantic_mask is None:
            semantic_mask = torch.ones(batch_size, target_len, device=device)
        if prompt_mask is None:
            prompt_mask = torch.ones(batch_size, prompt_len, device=device)

        # Concatenate semantic tokens
        full_semantic = torch.cat([prompt_semantic_tokens, semantic_tokens], dim=1)
        full_mask = torch.cat([prompt_mask, semantic_mask], dim=1)

        # Get semantic condition
        cond = self.semantic_emb(full_semantic)  # (B, T_total, hidden_size)

        # Initialize with noise
        h = 1.0 / n_timesteps
        z = torch.randn(
            (batch_size, target_len, self.mel_dim),
            dtype=prompt_mel.dtype,
            device=device,
        ) * temperature
        xt = z

        # ODE solver
        for i in range(n_timesteps):
            t = (0 + (i + 0.5) * h) * torch.ones(batch_size, dtype=z.dtype, device=device)

            # Concatenate prompt and current generation
            xt_input = torch.cat([prompt_mel, xt], dim=1)

            # Project mel to hidden space
            mel_hidden = self.mel_input_mlp(xt_input)

            # Run through flow estimator
            hidden_states = self.diff_estimator(mel_hidden, t, cond, full_mask)

            # Project back to mel and extract target part
            flow_pred = self.mel_output_mlp(hidden_states)
            flow_pred = flow_pred[:, prompt_len:, :]

            # CFG
            if cfg > 0:
                # Unconditional prediction
                mel_hidden_uncond = self.mel_input_mlp(xt)
                hidden_states_uncond = self.diff_estimator(
                    mel_hidden_uncond, t,
                    torch.zeros_like(cond)[:, prompt_len:, :],
                    semantic_mask
                )
                uncond_flow_pred = self.mel_output_mlp(hidden_states_uncond)

                # Apply CFG
                pos_flow_pred_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow_pred)
                rescale_flow_pred = flow_pred_cfg * pos_flow_pred_std / (flow_pred_cfg.std() + 1e-8)
                flow_pred = rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg

            # Euler step
            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    @torch.no_grad()
    def generate_full(
        self,
        semantic_tokens,
        prompt_mel,
        prompt_semantic_tokens,
        semantic_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
        temperature=1.0,
    ):
        """
        Generate mel spectrogram and return full sequence (prompt + generated).

        Same as generate() but returns the full mel spectrogram including prompt.

        Returns:
            full_mel: Full mel spectrogram (B, T_prompt + T_target, mel_dim)
        """
        generated_mel = self.generate(
            semantic_tokens=semantic_tokens,
            prompt_mel=prompt_mel,
            prompt_semantic_tokens=prompt_semantic_tokens,
            semantic_mask=semantic_mask,
            prompt_mask=prompt_mask,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            temperature=temperature,
        )

        # Concatenate prompt and generated mel
        full_mel = torch.cat([prompt_mel, generated_mel], dim=1)
        return full_mel
