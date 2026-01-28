# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Semantic-to-Mel Flow Matching Model for MaskGCT

This model converts discrete semantic tokens to continuous mel spectrograms
using flow matching, with support for zero-shot voice cloning via prompt mel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.tts.maskgct.llama_nar import DiffLlama


class SemanticToMelFM(nn.Module):
    """
    Flow Matching model for Semantic-to-Mel conversion.

    Takes discrete semantic tokens as condition and generates mel spectrograms.
    Supports zero-shot voice cloning by using prompt mel as reference.
    """

    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cfg_scale=0.2,
        cond_codebook_size=8192,
        sigma=1e-5,
        time_scheduler="linear",
        cfg=None,
    ):
        super().__init__()

        # Load from config if provided
        mel_dim = cfg.mel_dim if cfg is not None and hasattr(cfg, "mel_dim") else mel_dim
        hidden_size = cfg.hidden_size if cfg is not None and hasattr(cfg, "hidden_size") else hidden_size
        num_layers = cfg.num_layers if cfg is not None and hasattr(cfg, "num_layers") else num_layers
        num_heads = cfg.num_heads if cfg is not None and hasattr(cfg, "num_heads") else num_heads
        cfg_scale = cfg.cfg_scale if cfg is not None and hasattr(cfg, "cfg_scale") else cfg_scale
        cond_codebook_size = cfg.cond_codebook_size if cfg is not None and hasattr(cfg, "cond_codebook_size") else cond_codebook_size
        sigma = cfg.sigma if cfg is not None and hasattr(cfg, "sigma") else sigma
        time_scheduler = cfg.time_scheduler if cfg is not None and hasattr(cfg, "time_scheduler") else time_scheduler

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.cond_codebook_size = cond_codebook_size
        self.time_scheduler = time_scheduler

        # Semantic token embedding (discrete -> continuous)
        self.cond_emb = nn.Embedding(cond_codebook_size, hidden_size)
        
        self.reset_parameters()

        # Flow estimator (DiffLlama from llama_nar.py)
        # Note: DiffLlama now has internal mel_mlp and mel_out_mlp, so we don't need separate projections
        self.diff_estimator = DiffLlama(
            mel_dim=mel_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.sigma = sigma


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
                if m.bias_k is not None:
                    nn.init.xavier_normal_(m.bias_k)
                if m.bias_v is not None:
                    nn.init.xavier_normal_(m.bias_v)

            elif (
                isinstance(m, nn.Conv1d)
                or isinstance(m, nn.ConvTranspose1d)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.ConvTranspose2d)
            ):
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
        Forward diffusion process for flow matching.

        Args:
            x: target mel spectrogram (B, T, mel_dim)
            t: diffusion timestep (B,)

        Returns:
            xt: noised mel (B, T, mel_dim)
            z: noise (B, T, mel_dim)
            new_t: timestep (B,)
            prompt_len: prompt length for each sample (B,)
            mask: mask indicating which frames are target (not prompt) (B, T, 1)
        """
        new_t = t
        t = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        # Sample noise
        z = torch.randn(x.shape, dtype=x.dtype, device=x.device, requires_grad=False)

        cfg_scale = self.cfg_scale

        # Determine prompt length for CFG training
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)
            ).to(x.device)
        else:
            prompt_len = torch.zeros(x.shape[0]).to(x)

        # Create prompt mask
        is_prompt = torch.zeros_like(x[:, :, 0])  # (B, T)
        col_indices = torch.arange(is_prompt.shape[1]).repeat(is_prompt.shape[0], 1).to(prompt_len)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1  # 1 if prompt

        mask = torch.ones_like(x[:, :, 0])  # 1 for target, 0 for prompt
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]  # (B, T, 1)

        # Flow matching: xt = (1 - (1 - sigma) * t) * z + t * x
        # For prompt frames, keep original mel
        xt = ((1 - (1 - self.sigma) * t) * z + t * x) * mask + x * (1 - mask)

        return xt, z, new_t, prompt_len, mask

    def loss_t(self, x, x_mask, t, cond):
        """
        Compute loss at timestep t.

        Args:
            x: target mel (B, T, mel_dim)
            x_mask: mask for valid frames (B, T)
            t: timestep (B,)
            cond: semantic condition embedding (B, T, hidden_size)

        Returns:
            noise: ground truth noise
            x: target mel
            flow_pred: predicted flow
            final_mask: mask for loss computation
            prompt_len: prompt lengths
        """
        xt, z, new_t, prompt_len, mask = self.forward_diffusion(x, t)

        noise = z

        # Drop condition for CFG (when prompt_len is 0)
        if cond is not None:
            cond = cond * torch.where(
                prompt_len > 0,
                torch.ones_like(prompt_len),
                torch.zeros_like(prompt_len),
            ).to(cond.device).unsqueeze(-1).unsqueeze(-1)

        # Run through flow estimator (DiffLlama now handles mel projection internally)
        flow_pred = self.diff_estimator(xt, new_t, cond, x_mask)  # (B, T, mel_dim)

        # Final mask for loss
        final_mask = mask * x_mask[..., None]  # (B, T, 1)

        return noise, x, flow_pred, final_mask, prompt_len

    def compute_loss(self, x, x_mask, cond):
        """
        Compute training loss.

        Args:
            x: target mel (B, T, mel_dim)
            x_mask: mask (B, T)
            cond: semantic condition embedding (B, T, hidden_size)
        """
        t = torch.rand(x.shape[0], device=x.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)

        # Cosine scheduler for timestep
        if self.time_scheduler == "cos":
            t = 1 - torch.cos(t * math.pi * 0.5)

        return self.loss_t(x, x_mask, t, cond)

    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,
        prompt_mel,
        x_mask=None,
        prompt_mask=None,
        n_timesteps=10,
        cfg=1.0,
        rescale_cfg=0.75,
    ):
        """
        Reverse diffusion for inference.

        Args:
            cond: semantic condition embedding (B, T_total, hidden_size)
            prompt_mel: prompt mel spectrogram (B, T_prompt, mel_dim)
            x_mask: mask for target frames (B, T_target)
            prompt_mask: mask for prompt frames (B, T_prompt)
            n_timesteps: number of diffusion steps
            cfg: classifier-free guidance scale
            rescale_cfg: CFG rescaling factor

        Returns:
            Generated mel spectrogram (B, T_target, mel_dim)
        """
        h = 1.0 / n_timesteps
        prompt_len = prompt_mel.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)
        if prompt_mask is None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(cond.device)

        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)

        # Initialize with noise
        z = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
            requires_grad=False,
        )
        xt = z

        # Iterative denoising
        for i in range(n_timesteps):
            # Concatenate prompt and current estimate
            xt_input = torch.cat([prompt_mel, xt], dim=1)  # (B, T_total, mel_dim)

            t = (0 + (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)

            # Predict flow (DiffLlama now handles mel projection internally)
            flow_pred = self.diff_estimator(xt_input, t, cond, xt_mask)
            flow_pred = flow_pred[:, prompt_len:, :]  # Only target frames

            # Classifier-free guidance
            if cfg > 0:
                # Unconditional prediction
                uncond_flow = self.diff_estimator(
                    xt, t, torch.zeros_like(cond)[:, :xt.shape[1], :], x_mask
                )

                # Apply CFG
                pos_flow_std = flow_pred.std()
                flow_pred_cfg = flow_pred + cfg * (flow_pred - uncond_flow)
                rescale_flow = flow_pred_cfg * pos_flow_std / flow_pred_cfg.std()
                flow_pred = rescale_cfg * rescale_flow + (1 - rescale_cfg) * flow_pred_cfg

            # Update estimate
            dxt = flow_pred * h
            xt = xt + dxt

        return xt

    def forward(self, x, x_mask, semantic_tokens):
        """
        Forward pass for training.

        Args:
            x: target mel spectrogram (B, T, mel_dim)
            x_mask: mask for valid frames (B, T)
            semantic_tokens: discrete semantic tokens (B, T)

        Returns:
            noise: ground truth noise
            x: target mel
            flow_pred: predicted flow
            final_mask: mask for loss
            prompt_len: prompt lengths
        """
        # Embed semantic tokens
        cond = self.cond_emb(semantic_tokens)  # (B, T, hidden_size)

        noise, x, flow_pred, final_mask, prompt_len = self.compute_loss(x, x_mask, cond)
        return noise, x, flow_pred, final_mask, prompt_len
