# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Semantic-to-Mel DiT (Diffusion Transformer) Model for MaskGCT

This model converts discrete semantic tokens to continuous mel spectrograms
using diffusion, with support for zero-shot voice cloning via prompt mel.
"""

import torch
import torch.nn as nn
import math
from models.tts.maskgct.llama_nar import DiffLlama


class SemanticToMelDiT(nn.Module):
    """
    Diffusion Transformer model for Semantic-to-Mel conversion.

    Takes discrete semantic tokens as condition and generates mel spectrograms
    using DDPM-style diffusion. Supports zero-shot voice cloning.
    """

    def __init__(
        self,
        mel_dim=100,
        hidden_size=1024,
        num_layers=12,
        num_heads=16,
        cfg_scale=0.2,
        cond_codebook_size=8192,
        num_diffusion_steps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
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
        num_diffusion_steps = cfg.num_diffusion_steps if cfg is not None and hasattr(cfg, "num_diffusion_steps") else num_diffusion_steps
        beta_start = cfg.beta_start if cfg is not None and hasattr(cfg, "beta_start") else beta_start
        beta_end = cfg.beta_end if cfg is not None and hasattr(cfg, "beta_end") else beta_end
        beta_schedule = cfg.beta_schedule if cfg is not None and hasattr(cfg, "beta_schedule") else beta_schedule

        self.mel_dim = mel_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.cond_codebook_size = cond_codebook_size
        self.num_diffusion_steps = num_diffusion_steps

        # Setup diffusion schedule
        self.register_diffusion_schedule(beta_start, beta_end, beta_schedule, num_diffusion_steps)

        # Semantic token embedding
        self.cond_emb = nn.Embedding(cond_codebook_size, hidden_size)

        # Mel input/output projections
        self.mel_in_proj = nn.Linear(mel_dim, hidden_size)
        self.mel_out_proj = nn.Linear(hidden_size, mel_dim)

        # Noise estimator (DiffLlama)
        self.diff_estimator = DiffLlama(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.reset_parameters()

    def register_diffusion_schedule(self, beta_start, beta_end, beta_schedule, num_steps):
        """Register diffusion schedule buffers."""
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "cosine":
            steps = num_steps + 1
            s = 0.008
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer("posterior_variance", betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))

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
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

        self.apply(_reset_parameters)

    def q_sample(self, x_start, t, noise=None):
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    @torch.no_grad()
    def forward_diffusion(self, x, t_idx):
        """
        Forward diffusion for training.

        Args:
            x: target mel (B, T, mel_dim)
            t_idx: timestep indices (B,)

        Returns:
            x_t: noised mel
            noise: added noise
            t_normalized: normalized timestep for model input
            prompt_len: prompt lengths
            mask: target mask
        """
        cfg_scale = self.cfg_scale

        # Determine prompt length for CFG
        if torch.rand(1) > cfg_scale:
            prompt_len = torch.randint(
                min(x.shape[1] // 4, 5), int(x.shape[1] * 0.4), (x.shape[0],)
            ).to(x.device)
        else:
            prompt_len = torch.zeros(x.shape[0]).to(x)

        # Create prompt mask
        is_prompt = torch.zeros_like(x[:, :, 0])
        col_indices = torch.arange(is_prompt.shape[1]).repeat(is_prompt.shape[0], 1).to(prompt_len)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1

        mask = torch.ones_like(x[:, :, 0])
        mask[is_prompt.bool()] = 0
        mask = mask[:, :, None]

        # Sample noise and create noisy input
        noise = torch.randn_like(x)
        x_t = self.q_sample(x, t_idx, noise)

        # Keep prompt frames clean
        x_t = x_t * mask + x * (1 - mask)
        noise = noise * mask  # Only predict noise for target frames

        # Normalize timestep for model
        t_normalized = t_idx.float() / self.num_diffusion_steps

        return x_t, noise, t_normalized, prompt_len, mask

    def loss_t(self, x, x_mask, t_idx, cond):
        """Compute loss at timestep t."""
        x_t, noise, t_normalized, prompt_len, mask = self.forward_diffusion(x, t_idx)

        # Drop condition for CFG
        if cond is not None:
            cond = cond * torch.where(
                prompt_len > 0,
                torch.ones_like(prompt_len),
                torch.zeros_like(prompt_len),
            ).to(cond.device).unsqueeze(-1).unsqueeze(-1)

        # Project to hidden space
        x_t_hidden = self.mel_in_proj(x_t)

        # Predict noise
        hidden_out = self.diff_estimator(x_t_hidden, t_normalized, cond, x_mask)
        noise_pred = self.mel_out_proj(hidden_out)

        # Final mask
        final_mask = mask * x_mask[..., None]

        return noise, noise_pred, final_mask, prompt_len

    def compute_loss(self, x, x_mask, cond):
        """Compute training loss."""
        # Sample random timesteps
        t_idx = torch.randint(0, self.num_diffusion_steps, (x.shape[0],), device=x.device)
        return self.loss_t(x, x_mask, t_idx, cond)

    @torch.no_grad()
    def p_sample(self, x_t, t_idx, cond, x_mask, cfg=1.0):
        """Single denoising step: p(x_{t-1} | x_t)."""
        t_normalized = t_idx.float() / self.num_diffusion_steps

        # Project to hidden
        x_t_hidden = self.mel_in_proj(x_t)

        # Predict noise
        hidden_out = self.diff_estimator(x_t_hidden, t_normalized, cond, x_mask)
        noise_pred = self.mel_out_proj(hidden_out)

        # CFG
        if cfg > 0:
            uncond_hidden = self.diff_estimator(
                x_t_hidden, t_normalized, torch.zeros_like(cond), x_mask
            )
            uncond_noise = self.mel_out_proj(uncond_hidden)
            noise_pred = noise_pred + cfg * (noise_pred - uncond_noise)

        # Compute x_{t-1}
        betas_t = self.betas[t_idx][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_idx][:, None, None]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_idx][:, None, None]

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

        if t_idx[0] > 0:
            posterior_variance_t = self.posterior_variance[t_idx][:, None, None]
            noise = torch.randn_like(x_t)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def reverse_diffusion(
        self,
        cond,
        prompt_mel,
        x_mask=None,
        prompt_mask=None,
        n_timesteps=50,
        cfg=1.0,
        use_ddim=True,
        ddim_eta=0.0,
    ):
        """
        Reverse diffusion for inference using DDIM or DDPM sampling.

        Args:
            cond: semantic condition (B, T_total, hidden_size)
            prompt_mel: prompt mel (B, T_prompt, mel_dim)
            x_mask: target mask (B, T_target)
            prompt_mask: prompt mask (B, T_prompt)
            n_timesteps: number of denoising steps (can be much smaller with DDIM)
            cfg: classifier-free guidance scale
            use_ddim: whether to use DDIM sampling (faster, fewer steps needed)
            ddim_eta: DDIM eta parameter (0=deterministic, 1=DDPM)

        Returns:
            Generated mel (B, T_target, mel_dim)
        """
        prompt_len = prompt_mel.shape[1]
        target_len = cond.shape[1] - prompt_len

        if x_mask is None:
            x_mask = torch.ones(cond.shape[0], target_len).to(cond.device)
        if prompt_mask is None:
            prompt_mask = torch.ones(cond.shape[0], prompt_len).to(cond.device)

        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)

        # Start from noise
        x_t = torch.randn(
            (cond.shape[0], target_len, self.mel_dim),
            dtype=cond.dtype,
            device=cond.device,
        )

        # Create timestep sequence for DDIM
        step_size = self.num_diffusion_steps // n_timesteps
        timesteps = list(range(self.num_diffusion_steps - 1, -1, -step_size))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((cond.shape[0],), t, device=cond.device, dtype=torch.long)

            # Concatenate prompt and current estimate
            x_full = torch.cat([prompt_mel, x_t], dim=1)
            x_full_hidden = self.mel_in_proj(x_full)

            t_normalized = t_batch.float() / self.num_diffusion_steps

            # Predict noise
            hidden_out = self.diff_estimator(x_full_hidden, t_normalized, cond, xt_mask)
            noise_pred = self.mel_out_proj(hidden_out)
            noise_pred = noise_pred[:, prompt_len:, :]

            # CFG
            if cfg > 0:
                x_t_hidden = self.mel_in_proj(x_t)
                uncond_hidden = self.diff_estimator(
                    x_t_hidden, t_normalized, torch.zeros_like(cond)[:, :target_len, :], x_mask
                )
                uncond_noise = self.mel_out_proj(uncond_hidden)
                noise_pred = noise_pred + cfg * (noise_pred - uncond_noise)

            if use_ddim:
                # DDIM sampling step
                alpha_cumprod_t = self.alphas_cumprod[t]

                # Get alpha_cumprod for previous timestep
                if i + 1 < len(timesteps):
                    t_prev = timesteps[i + 1]
                    alpha_cumprod_t_prev = self.alphas_cumprod[t_prev]
                else:
                    alpha_cumprod_t_prev = torch.tensor(1.0, device=cond.device)

                # Predict x_0
                pred_x0 = (x_t - math.sqrt(1 - alpha_cumprod_t) * noise_pred) / math.sqrt(alpha_cumprod_t)

                # Compute sigma for DDIM
                sigma_t = ddim_eta * math.sqrt(
                    (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                )

                # Direction pointing to x_t
                dir_xt = math.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * noise_pred

                # DDIM update
                x_t = math.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt

                if sigma_t > 0 and t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = x_t + sigma_t * noise
            else:
                # Original DDPM sampling
                betas_t = self.betas[t]
                sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
                sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]

                model_mean = sqrt_recip_alphas_t * (x_t - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

                if t > 0:
                    posterior_variance_t = self.posterior_variance[t]
                    noise = torch.randn_like(x_t)
                    x_t = model_mean + math.sqrt(posterior_variance_t) * noise
                else:
                    x_t = model_mean

        return x_t

    def forward(self, x, x_mask, semantic_tokens):
        """
        Forward pass for training.

        Args:
            x: target mel (B, T, mel_dim)
            x_mask: mask (B, T)
            semantic_tokens: discrete semantic tokens (B, T)

        Returns:
            noise: ground truth noise
            noise_pred: predicted noise
            final_mask: loss mask
            prompt_len: prompt lengths
        """
        cond = self.cond_emb(semantic_tokens)
        noise, noise_pred, final_mask, prompt_len = self.compute_loss(x, x_mask, cond)
        return noise, noise_pred, final_mask, prompt_len
