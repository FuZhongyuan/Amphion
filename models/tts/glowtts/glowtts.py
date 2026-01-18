# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GlowTTS Model

GlowTTS is a flow-based generative model for parallel text-to-speech synthesis.
It combines the following components:
- Text Encoder: Transformer-based encoder with relative position attention
- Duration Predictor: Predicts duration for each phoneme
- Flow Decoder: Normalizing flow for mel-spectrogram generation

Reference:
    Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search
    https://arxiv.org/abs/2005.11129
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .glowtts_modules import (
    Encoder,
    DurationPredictor,
    ConvReluNorm,
    ActNorm,
    InvConvNear,
    CouplingBlock,
    LayerNorm,
    sequence_mask,
    generate_path,
    squeeze,
    unsqueeze,
)
from models.tts.glowtts.monotonic_align import maximum_path
from text.symbols import symbols
from text.symbol_table import SymbolTable


class TextEncoder(nn.Module):
    """
    Text Encoder for GlowTTS.

    Encodes phoneme sequence into hidden representation with
    mean and log variance for the prior distribution.
    """

    def __init__(
        self,
        n_vocab,
        out_channels,
        hidden_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        window_size=None,
        block_length=None,
        mean_only=False,
        prenet=False,
        gin_channels=0,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.block_length = block_length
        self.mean_only = mean_only
        self.prenet = prenet
        self.gin_channels = gin_channels

        # Embedding layer
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels ** -0.5)

        # Optional prenet
        if prenet:
            self.pre = ConvReluNorm(
                hidden_channels, hidden_channels, hidden_channels,
                kernel_size=5, n_layers=3, p_dropout=0.5
            )

        # Transformer encoder
        self.encoder = Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
        )

        # Projection for mean
        self.proj_m = nn.Conv1d(hidden_channels, out_channels, 1)

        # Projection for log variance (if not mean_only)
        if not mean_only:
            self.proj_s = nn.Conv1d(hidden_channels, out_channels, 1)

        # Duration predictor
        self.proj_w = DurationPredictor(
            hidden_channels + gin_channels,
            filter_channels_dp,
            kernel_size,
            p_dropout
        )

    def forward(self, x, x_lengths, g=None):
        # x: [B, T_x]
        # x_lengths: [B]
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T_x, H]
        x = torch.transpose(x, 1, -1)  # [B, H, T_x]
        x_mask = torch.unsqueeze(
            sequence_mask(x_lengths, x.size(2)), 1
        ).to(x.dtype)

        if self.prenet:
            x = self.pre(x, x_mask)

        x = self.encoder(x, x_mask)

        # Duration predictor input (optionally with speaker embedding)
        if g is not None:
            g_exp = g.expand(-1, -1, x.size(-1))
            x_dp = torch.cat([torch.detach(x), g_exp], 1)
        else:
            x_dp = torch.detach(x)

        # Project to mean and log variance
        x_m = self.proj_m(x) * x_mask
        if not self.mean_only:
            x_logs = self.proj_s(x) * x_mask
        else:
            x_logs = torch.zeros_like(x_m)

        # Predict log duration
        logw = self.proj_w(x_dp, x_mask)

        return x_m, x_logs, logw, x_mask


class FlowSpecDecoder(nn.Module):
    """
    Flow-based Spectrogram Decoder.

    Uses normalizing flows to transform between the prior distribution
    and the mel-spectrogram distribution.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
        ddi=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz, ddi=ddi))
            self.flows.append(
                InvConvNear(channels=in_channels * n_sqz, n_split=n_split)
            )
            self.flows.append(
                CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = squeeze(x, x_mask, self.n_sqz)

        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)

        if self.n_sqz > 1:
            x, x_mask = unsqueeze(x, x_mask, self.n_sqz)

        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()

    def set_ddi(self, ddi):
        """Set DDI (Data-Dependent Initialization) mode for all ActNorm layers."""
        for f in self.flows:
            if hasattr(f, "set_ddi"):
                f.set_ddi(ddi)


class GlowTTS(nn.Module):
    """
    GlowTTS Model.

    Combines text encoder, duration predictor, and flow decoder
    for parallel text-to-speech synthesis.
    """

    def __init__(self, cfg, ddi=False):
        super().__init__()

        self.cfg = cfg
        model_cfg = cfg.model.glowtts

        # Get vocabulary size
        if cfg.preprocess.phone_extractor == "lexicon":
            self.n_vocab = len(symbols)
        else:
            # Load vocabulary size from symbols dictionary
            symbols_dict_file = os.path.join(
                cfg.preprocess.processed_dir,
                cfg.dataset[0],
                cfg.preprocess.symbols_dict,
            )
            if os.path.exists(symbols_dict_file):
                symbol_table = SymbolTable.from_file(symbols_dict_file)
                self.n_vocab = len(symbol_table.symbols)
            else:
                # Fallback to default symbols if file doesn't exist
                self.n_vocab = len(symbols)
        self.n_speakers = getattr(model_cfg, "n_speakers", 0)
        self.gin_channels = getattr(model_cfg, "gin_channels", 0)

        # Encoder parameters
        hidden_channels = model_cfg.hidden_channels
        filter_channels = model_cfg.filter_channels
        filter_channels_dp = model_cfg.filter_channels_dp
        out_channels = cfg.preprocess.n_mel
        n_heads = model_cfg.n_heads
        n_layers_enc = model_cfg.n_layers_enc
        kernel_size = model_cfg.kernel_size
        p_dropout = model_cfg.p_dropout
        window_size = getattr(model_cfg, "window_size", 4)
        block_length = getattr(model_cfg, "block_length", None)
        mean_only = getattr(model_cfg, "mean_only", False)
        prenet = getattr(model_cfg, "prenet", False)

        # Decoder parameters
        hidden_channels_dec = getattr(model_cfg, "hidden_channels_dec", hidden_channels)
        n_blocks_dec = model_cfg.n_blocks_dec
        kernel_size_dec = model_cfg.kernel_size_dec
        dilation_rate = model_cfg.dilation_rate
        n_block_layers = model_cfg.n_block_layers
        p_dropout_dec = getattr(model_cfg, "p_dropout_dec", 0.)
        n_split = getattr(model_cfg, "n_split", 4)
        n_sqz = getattr(model_cfg, "n_sqz", 2)
        sigmoid_scale = getattr(model_cfg, "sigmoid_scale", False)

        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_sqz = n_sqz

        # Text Encoder
        self.encoder = TextEncoder(
            self.n_vocab,
            out_channels,
            hidden_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_layers_enc,
            kernel_size,
            p_dropout,
            window_size=window_size,
            block_length=block_length,
            mean_only=mean_only,
            prenet=prenet,
            gin_channels=self.gin_channels,
        )

        # Flow Decoder
        self.decoder = FlowSpecDecoder(
            out_channels,
            hidden_channels_dec,
            kernel_size_dec,
            dilation_rate,
            n_blocks_dec,
            n_block_layers,
            p_dropout=p_dropout_dec,
            n_split=n_split,
            n_sqz=n_sqz,
            sigmoid_scale=sigmoid_scale,
            gin_channels=self.gin_channels,
            ddi=ddi,
        )

        # Speaker embedding
        if self.n_speakers > 1:
            self.emb_g = nn.Embedding(self.n_speakers, self.gin_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    def forward(
        self,
        data,
        gen=False,
        noise_scale=1.0,
        length_scale=1.0,
    ):
        """
        Forward pass for GlowTTS.

        Args:
            data: Dictionary containing:
                - phone_seq: [B, T_x] phoneme sequence
                - phone_len: [B] phoneme lengths
                - mel: [B, T_y, n_mel] mel spectrogram (training only)
                - target_len: [B] mel lengths (training only)
                - spk_id: [B] speaker ids (multi-speaker only)
            gen: If True, generate mel spectrogram (inference mode)
            noise_scale: Scale for sampling noise (inference only)
            length_scale: Scale for duration (inference only)

        Returns:
            Dictionary containing model outputs and losses
        """
        x = data["phone_seq"]
        x_lengths = data["phone_len"]

        # Get speaker embedding if multi-speaker
        g = None
        if self.n_speakers > 1 and "spk_id" in data:
            g = F.normalize(self.emb_g(data["spk_id"])).unsqueeze(-1)  # [B, gin, 1]

        # Encode text
        x_m, x_logs, logw, x_mask = self.encoder(x, x_lengths, g=g)

        if gen:
            # Inference mode: generate mel spectrogram
            w = torch.exp(logw) * x_mask * length_scale
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_max_length = None
            y = None
        else:
            # Training mode: use ground truth mel
            y = data["mel"].transpose(1, 2)  # [B, n_mel, T_y]
            y_lengths = data["target_len"]
            y_max_length = y.size(2)

        # Preprocess y for flow (make sure it's divisible by n_sqz)
        y, y_lengths, y_max_length = self.preprocess(y, y_lengths, y_max_length)

        # Create masks
        z_mask = torch.unsqueeze(
            sequence_mask(y_lengths, y_max_length), 1
        ).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(z_mask, 2)

        if gen:
            # Generate alignment using duration prediction
            attn = generate_path(
                w_ceil.squeeze(1), attn_mask.squeeze(1)
            ).unsqueeze(1)

            # Expand encoder output to mel length
            z_m = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
            ).transpose(1, 2)
            z_logs = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
            ).transpose(1, 2)
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            # Sample from prior and decode
            z = (z_m + torch.exp(z_logs) * torch.randn_like(z_m) * noise_scale) * z_mask
            y, logdet = self.decoder(z, z_mask, g=g, reverse=True)

            return {
                "mel_out": y.transpose(1, 2),  # [B, T_y, n_mel]
                "mel_lens": y_lengths,
                "attn": attn,
                "logw": logw,
                "logw_": logw_,
            }
        else:
            # Training mode: encode mel and compute loss
            z, logdet = self.decoder(y, z_mask, g=g, reverse=False)

            # Compute alignment using MAS
            with torch.no_grad():
                x_s_sq_r = torch.exp(-2 * x_logs)
                logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - x_logs, [1]).unsqueeze(-1)
                logp2 = torch.matmul(x_s_sq_r.transpose(1, 2), -0.5 * (z ** 2))
                logp3 = torch.matmul((x_m * x_s_sq_r).transpose(1, 2), z)
                logp4 = torch.sum(-0.5 * (x_m ** 2) * x_s_sq_r, [1]).unsqueeze(-1)
                logp = logp1 + logp2 + logp3 + logp4  # [B, T_x, T_y]

                attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

            # Expand encoder output to mel length
            z_m = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_m.transpose(1, 2)
            ).transpose(1, 2)
            z_logs = torch.matmul(
                attn.squeeze(1).transpose(1, 2), x_logs.transpose(1, 2)
            ).transpose(1, 2)
            logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * x_mask

            return {
                "z": z,
                "z_m": z_m,
                "z_logs": z_logs,
                "logdet": logdet,
                "z_mask": z_mask,
                "x_m": x_m,
                "x_logs": x_logs,
                "x_mask": x_mask,
                "attn": attn,
                "logw": logw,
                "logw_": logw_,
                "mel_lens": y_lengths,
            }

    def preprocess(self, y, y_lengths, y_max_length):
        """Preprocess mel spectrogram to be divisible by n_sqz."""
        if y_max_length is not None:
            y_max_length = (y_max_length // self.n_sqz) * self.n_sqz
            y = y[:, :, :y_max_length]
        y_lengths = (y_lengths // self.n_sqz) * self.n_sqz
        return y, y_lengths, y_max_length

    def store_inverse(self):
        """Store inverse weights for faster inference."""
        self.decoder.store_inverse()

    def set_ddi(self, ddi):
        """Set DDI (Data-Dependent Initialization) mode for ActNorm layers."""
        self.decoder.set_ddi(ddi)


class GlowTTSLoss(nn.Module):
    """
    GlowTTS Loss Function.

    Combines MLE loss for the flow and duration loss.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, data, outputs):
        """
        Compute GlowTTS loss.

        Args:
            data: Dictionary containing input data
            outputs: Dictionary containing model outputs

        Returns:
            Dictionary containing loss values
        """
        z = outputs["z"]
        z_m = outputs["z_m"]
        z_logs = outputs["z_logs"]
        logdet = outputs["logdet"]
        z_mask = outputs["z_mask"]
        logw = outputs["logw"]
        logw_ = outputs["logw_"]
        x_mask = outputs["x_mask"]

        # MLE loss
        mle_loss = self.mle_loss(z, z_m, z_logs, logdet, z_mask)

        # Duration loss (use x_mask for proper normalization)
        dur_loss = self.duration_loss(logw, logw_, x_mask)

        total_loss = mle_loss + dur_loss

        return {
            "loss": total_loss,
            "mle_loss": mle_loss,
            "dur_loss": dur_loss,
        }

    def mle_loss(self, z, m, logs, logdet, mask):
        """
        Compute MLE loss for the flow.

        This is the negative log-likelihood under the flow model.
        """
        # Negative log-likelihood without the constant term
        l = torch.sum(logs) + 0.5 * torch.sum(
            torch.exp(-2 * logs) * ((z - m) ** 2)
        )
        # Subtract log Jacobian determinant
        l = l - torch.sum(logdet)
        # Average across batch, channel and time
        l = l / torch.sum(torch.ones_like(z) * mask)
        # Add the remaining constant term
        l = l + 0.5 * math.log(2 * math.pi)
        return l

    def duration_loss(self, logw, logw_, x_mask):
        """
        Compute duration loss (MSE in log domain) with proper mask normalization.

        Args:
            logw: Predicted log duration [B, 1, T_x] (masked)
            logw_: Ground truth log duration from MAS [B, 1, T_x] (masked)
            x_mask: Mask for valid phoneme positions [B, 1, T_x]

        Returns:
            Duration loss normalized by the number of valid positions
        """
        # Compute squared difference only at valid positions
        l = torch.sum(x_mask * (logw - logw_) ** 2) / torch.sum(x_mask)
        return l
