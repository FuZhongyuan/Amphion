# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
MaskGCT T2S Model with Curriculum Learning for Small Datasets

This variant implements a progressive training curriculum that:
1. Starts with low mask ratios and gradually increases
2. Uses longer prompts initially, then reduces
3. Implements smooth difficulty progression
4. Better suited for small-scale datasets (< 50 hours)

Key improvements over standard T2S:
- Curriculum-based mask scheduling
- Adaptive prompt length scheduling
- Gradual CFG dropout introduction
- Better convergence on limited data
"""

import torch
import numpy as np
import torch.nn as nn
import math
from einops import rearrange
from models.tts.maskgct.llama_nar import DiffLlamaPrefix


def top_k(logits, thres=0.9):
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


class MaskGCT_T2S_Curriculum(nn.Module):
    """
    MaskGCT T2S with Curriculum Learning for Small Datasets

    Training curriculum (3 stages):

    Stage 1 (0-30% steps): Easy - Low mask ratio, long prompts
    - Mask ratio: 10-30%
    - Prompt length: 40-60% of sequence
    - CFG dropout: 0% (always use prompt)
    - Goal: Learn basic phone-semantic alignment

    Stage 2 (30-70% steps): Medium - Moderate mask ratio, medium prompts
    - Mask ratio: 20-50%
    - Prompt length: 20-50% of sequence
    - CFG dropout: 5%
    - Goal: Improve generation quality

    Stage 3 (70-100% steps): Hard - High mask ratio, short prompts
    - Mask ratio: 30-80%
    - Prompt length: 10-40% of sequence
    - CFG dropout: 15%
    - Goal: Match original T2S capability
    """

    def __init__(
        self,
        hidden_size=1024,
        num_layers=16,
        num_heads=16,
        cfg_scale=0.15,
        cond_codebook_size=8192,
        cond_dim=1024,
        use_phone_cond=True,
        # Curriculum parameters
        curriculum_stages=3,
        stage1_steps=0.3,  # 30% of training
        stage2_steps=0.7,  # 70% of training
        total_steps=100000,
        min_mask_ratio=0.1,
        max_mask_ratio=0.8,
        cfg=None,
    ):
        super().__init__()

        # Parse config
        hidden_size = cfg.hidden_size if cfg is not None and hasattr(cfg, "hidden_size") else hidden_size
        num_layers = cfg.num_layers if cfg is not None and hasattr(cfg, "num_layers") else num_layers
        num_heads = cfg.num_heads if cfg is not None and hasattr(cfg, "num_heads") else num_heads
        cfg_scale = cfg.cfg_scale if cfg is not None and hasattr(cfg, "cfg_scale") else cfg_scale
        cond_codebook_size = cfg.cond_codebook_size if cfg is not None and hasattr(cfg, "cond_codebook_size") else cond_codebook_size
        cond_dim = cfg.cond_dim if cfg is not None and hasattr(cfg, "cond_dim") else cond_dim
        use_phone_cond = cfg.use_phone_cond if cfg is not None and hasattr(cfg, "use_phone_cond") else use_phone_cond

        # Curriculum parameters
        total_steps = cfg.total_steps if cfg is not None and hasattr(cfg, "total_steps") else total_steps
        stage1_steps = cfg.stage1_steps if cfg is not None and hasattr(cfg, "stage1_steps") else stage1_steps
        stage2_steps = cfg.stage2_steps if cfg is not None and hasattr(cfg, "stage2_steps") else stage2_steps
        min_mask_ratio = cfg.min_mask_ratio if cfg is not None and hasattr(cfg, "min_mask_ratio") else min_mask_ratio
        max_mask_ratio = cfg.max_mask_ratio if cfg is not None and hasattr(cfg, "max_mask_ratio") else max_mask_ratio

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cfg_scale = cfg_scale
        self.cond_codebook_size = cond_codebook_size
        self.cond_dim = cond_dim
        self.use_phone_cond = use_phone_cond

        # Curriculum settings
        self.total_steps = total_steps
        self.stage1_steps = int(total_steps * stage1_steps)
        self.stage2_steps = int(total_steps * stage2_steps)
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.current_step = 0

        # Model components (same as original)
        self.mask_emb = nn.Embedding(1, self.hidden_size)
        self.to_logit = nn.Linear(self.hidden_size, self.cond_codebook_size)
        self.cond_emb = nn.Embedding(cond_codebook_size, self.hidden_size)

        if self.use_phone_cond:
            self.phone_emb = nn.Embedding(1024, hidden_size, padding_idx=1023)
            torch.nn.init.normal_(self.phone_emb.weight, mean=0.0, std=0.02)

        self.reset_parameters()

        self.diff_estimator = DiffLlamaPrefix(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            use_phone_cond=use_phone_cond,
        )

    def set_step(self, step):
        """Update current training step for curriculum scheduling"""
        self.current_step = step

    def get_curriculum_params(self):
        """
        Get curriculum parameters based on current training step.

        Returns:
            dict with keys:
            - mask_ratio_range: (min, max) for mask ratio
            - prompt_ratio_range: (min, max) for prompt length
            - cfg_dropout: probability of dropping prompt
            - stage: current stage number (1, 2, or 3)
        """
        step = self.current_step

        if step < self.stage1_steps:
            # Stage 1: Easy - Learn basic alignment
            progress = step / self.stage1_steps
            return {
                "mask_ratio_range": (0.10, 0.30),
                "prompt_ratio_range": (0.40, 0.60),
                "cfg_dropout": 0.0,
                "stage": 1,
                "progress": progress
            }
        elif step < self.stage2_steps:
            # Stage 2: Medium - Improve quality
            progress = (step - self.stage1_steps) / (self.stage2_steps - self.stage1_steps)
            return {
                "mask_ratio_range": (0.20, 0.50),
                "prompt_ratio_range": (0.20, 0.50),
                "cfg_dropout": 0.05,
                "stage": 2,
                "progress": progress
            }
        else:
            # Stage 3: Hard - Match original capability
            progress = (step - self.stage2_steps) / (self.total_steps - self.stage2_steps)
            return {
                "mask_ratio_range": (0.30, 0.80),
                "prompt_ratio_range": (0.10, 0.40),
                "cfg_dropout": 0.15,
                "stage": 3,
                "progress": progress
            }

    def mask_prob(self, t, curriculum_params):
        """
        Compute mask probability with curriculum adjustment.

        Original: mask_prob = sin(t * π/2), range [0, 1]
        Curriculum: scale to current stage's range
        """
        # Base mask probability from diffusion timestep
        base_prob = torch.sin(t * np.pi / 2).to(t.device)

        # Scale to curriculum range
        min_ratio, max_ratio = curriculum_params["mask_ratio_range"]
        scaled_prob = min_ratio + (max_ratio - min_ratio) * base_prob

        return scaled_prob

    def forward_diffusion(self, x0, t):
        """
        Apply forward diffusion with curriculum-based masking.

        Args:
            x0: Semantic tokens (B, T)
            t: Diffusion timestep (B,)

        Returns:
            xt: Masked embeddings (B, T, hidden_size)
            new_t: Timestep (B,)
            mask: Binary mask (B, T, 1)
            prompt_len: Prompt lengths (B,)
            mask_prob: Masking probabilities (B,)
        """
        new_t = t
        curriculum_params = self.get_curriculum_params()

        # Curriculum-adjusted mask probability
        mask_prob = self.mask_prob(new_t, curriculum_params)

        mask_token = self.mask_emb(torch.LongTensor([0]).to(x0.device))
        xt = torch.zeros(x0.shape[0], x0.shape[1], self.hidden_size).to(x0.device)

        # Curriculum-adjusted CFG dropout
        cfg_dropout = curriculum_params["cfg_dropout"]

        # Determine prompt length based on curriculum
        if torch.rand(1) > cfg_dropout:
            # Use prompt
            min_prompt_ratio, max_prompt_ratio = curriculum_params["prompt_ratio_range"]
            seq_len = x0.shape[1]
            min_prompt = max(int(seq_len * min_prompt_ratio), 1)
            max_prompt = max(int(seq_len * max_prompt_ratio), min_prompt + 1)
            prompt_len = torch.randint(min_prompt, max_prompt, (x0.shape[0],)).to(x0.device)
        else:
            # CFG dropout: no prompt
            prompt_len = torch.zeros(x0.shape[0]).to(x0.device)

        # Create prompt mask
        is_prompt = torch.zeros_like(x0[:, :])
        col_indices = torch.arange(is_prompt.shape[1]).repeat(is_prompt.shape[0], 1).to(prompt_len)
        is_prompt[col_indices < prompt_len.unsqueeze(1)] = 1

        # Apply masking (don't mask prompt tokens)
        mask = torch.bernoulli(torch.ones_like(x0[:, :]) * mask_prob[..., None])
        mask[is_prompt.bool()] = 0

        # Ensure at least one token is masked
        mask_num = mask.sum(dim=1, keepdim=False)
        all_zero_mask = (mask_num == 0).bool()
        row_indices_to_modify = torch.nonzero(all_zero_mask).squeeze(-1)
        for idx in row_indices_to_modify:
            mask_pos = prompt_len[idx].long().item()
            if mask_pos < x0.shape[1]:
                mask[idx, mask_pos] = 1

        mask = mask[..., None]
        xt = xt + mask * mask_token[:, None, :] + (1 - mask) * self.cond_emb(x0[:, :])

        return xt, new_t, mask, prompt_len, mask_prob

    def loss_t(self, x0, x_mask, t, phone_embedding=None, phone_mask=None):
        """Compute loss (same interface as original)"""
        xt, new_t, mask, prompt_len, mask_prob = self.forward_diffusion(x0, t)

        embeds = self.diff_estimator(
            xt, new_t, x_mask, phone_embedding=phone_embedding, phone_mask=phone_mask
        )
        logits = self.to_logit(embeds)

        final_mask = mask * x_mask[..., None]

        return logits, final_mask, x0, prompt_len, mask_prob

    def compute_loss(self, x0, x_mask, phone_embedding=None, phone_mask=None):
        """Compute loss (same interface as original)"""
        t = torch.rand(x0.shape[0], device=x0.device, requires_grad=False)
        t = torch.clamp(t, 1e-5, 1.0)
        return self.loss_t(x0, x_mask, t, phone_embedding, phone_mask)

    def reset_parameters(self):
        """Initialize parameters (same as original)"""
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
    def reverse_diffusion(
        self,
        prompt,
        target_len,
        phone_id,
        prompt_mask=None,
        temp=0.9,
        filter_thres=0.98,
        n_timesteps=40,
        cfg=1.0,
        rescale_cfg=1.0,
    ):
        """Inference (same interface as original T2S)"""
        phone_embedding = self.phone_emb(phone_id)
        prompt_code = prompt
        prompt_len = prompt_code.shape[1]

        x_mask = torch.ones(prompt_code.shape[0], target_len).to(prompt_code.device)
        phone_mask = torch.ones_like(phone_id)

        if prompt_mask is None:
            prompt_mask = torch.ones(prompt_code.shape[0], prompt_len).to(prompt_code.device)

        cum = torch.zeros(x_mask.shape[0], x_mask.shape[1], self.hidden_size).to(x_mask.device)
        bsz, seq_len, _ = cum.shape

        choice_temp = 1.0
        start_temp = temp
        start_choice_temp = choice_temp

        xt = torch.LongTensor(bsz, seq_len).to(x_mask.device)
        steps = n_timesteps
        to_logit = self.to_logit
        cond_emb = self.cond_emb

        mask_token = self.mask_emb(torch.LongTensor([0]).to(xt.device))
        mask = torch.full((bsz, seq_len, 1), True).to(x_mask.device)
        seq = torch.full((bsz, seq_len), 0).to(x_mask.device)
        h = 1.0 / steps

        cur_prompt = cond_emb(prompt_code)

        t_list = [1.0 - i * h for i in range(steps)]
        t_list.append(0.0)

        for i in range(steps):
            t = t_list[i] * torch.ones(bsz).to(x_mask.device)
            token = cond_emb(seq)
            cur = cum + mask * mask_token[:, None, :] + (~mask) * token

            xt_input = torch.cat([cur_prompt, cur], dim=1)
            xt_mask = torch.cat([prompt_mask, x_mask], dim=1)

            embeds = self.diff_estimator(
                xt_input, t, xt_mask,
                phone_embedding=phone_embedding,
                phone_mask=phone_mask,
            )
            embeds = embeds[:, prompt_len:, :]

            # Classifier-free guidance
            if cfg > 0:
                mask_embeds = self.diff_estimator(
                    cur, t, x_mask,
                    phone_embedding=phone_embedding[:, phone_embedding.shape[1]:, :],
                    phone_mask=phone_mask[:, prompt_len:],
                )
                pos_emb_std = embeds.std()
                embeds = embeds + cfg * (embeds - mask_embeds)
                rescale_embeds = embeds * pos_emb_std / embeds.std()
                embeds = rescale_cfg * rescale_embeds + (1 - rescale_cfg) * embeds

            logits = to_logit(embeds)
            annealing_scale = t_list[i]

            choice_temp = start_choice_temp * annealing_scale
            temp = start_temp * annealing_scale
            logits = top_k(logits, filter_thres)

            if i == steps - 1:
                if steps == 1:
                    temp = 0.2
                    sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))
                else:
                    sampled_ids = logits.argmax(dim=-1)
            else:
                sampled_ids = gumbel_sample(logits, temperature=max(temp, 1e-3))

            seq = torch.where(mask.squeeze(-1), sampled_ids, seq)

            scores = logits.softmax(dim=-1)
            scores = scores.gather(2, rearrange(sampled_ids, "b n -> b n 1"))
            scores = rearrange(scores, "b n 1 -> b n")
            scores = choice_temp * gumbel_noise(scores) + scores
            scores = 1 - scores

            next_t = t_list[i + 1] * torch.ones(bsz).to(x_mask.device)

            # Use curriculum-adjusted mask probability for inference
            curriculum_params = self.get_curriculum_params()
            next_mask_num = (self.mask_prob(next_t, curriculum_params) * seq_len).long()[0].item()

            if next_mask_num == 0:
                break

            scores = scores.masked_fill(~mask.squeeze(-1), -torch.finfo(scores.dtype).max)
            mask_indices = scores.topk(next_mask_num, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool).scatter(1, mask_indices, True)
            seq = seq.masked_fill(mask, 0)
            mask = mask.unsqueeze(-1)

        cum = cum + cond_emb(seq)
        xt = seq
        return xt

    def forward(self, x0, x_mask, phone_id=None, phone_mask=None):
        """Forward pass (same interface as original)"""
        phone_embedding = self.phone_emb(phone_id)
        logits, final_mask, x0, prompt_len, mask_prob = self.compute_loss(
            x0, x_mask, phone_embedding, phone_mask=phone_mask
        )
        return logits, final_mask, x0, prompt_len, mask_prob
