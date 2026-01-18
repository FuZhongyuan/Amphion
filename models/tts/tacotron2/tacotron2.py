# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_mask_from_lengths(lengths, max_len=None):
    """Create mask from lengths tensor.

    Args:
        lengths: Tensor of shape (batch_size,) containing lengths
        max_len: Maximum length (optional)

    Returns:
        Boolean mask of shape (batch_size, max_len) where True indicates valid positions
    """
    device = lengths.device
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, device=device).unsqueeze(0).expand(batch_size, -1)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)
    return mask


class LinearNorm(nn.Module):
    """Linear layer with Xavier initialization."""

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(nn.Module):
    """1D Convolution layer with Xavier initialization."""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        return self.conv(signal)


class Prenet(nn.Module):
    """Pre-network: a stack of fully connected layers with ReLU and dropout."""

    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([
            LinearNorm(in_size, out_size, bias=False)
            for (in_size, out_size) in zip(in_sizes, sizes)
        ])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class LocationLayer(nn.Module):
    """Location-sensitive attention layer."""

    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2, attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding, bias=False, stride=1,
            dilation=1
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim,
            bias=False, w_init_gain='tanh'
        )

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    """Location-sensitive attention mechanism."""

    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim,
            bias=False, w_init_gain='tanh'
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim,
            bias=False, w_init_gain='tanh'
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(
            attention_location_n_filters,
            attention_location_kernel_size,
            attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """Calculate alignment energies.

        Args:
            query: decoder output (batch, n_mel_channels * n_frames_per_step)
            processed_memory: processed encoder outputs (B, T_in, attention_dim)
            attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        Returns:
            alignment: (batch, max_time)
        """
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory
        ))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """Forward pass for attention.

        Args:
            attention_hidden_state: attention rnn last output
            memory: encoder outputs
            processed_memory: processed encoder outputs
            attention_weights_cat: previous and cumulative attention weights
            mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Encoder(nn.Module):
    """Encoder module: Three 1-d convolution banks + Bidirectional LSTM."""

    def __init__(self, cfg):
        super(Encoder, self).__init__()

        encoder_embedding_dim = cfg.model.encoder.encoder_embedding_dim
        encoder_n_convolutions = cfg.model.encoder.encoder_n_convolutions
        encoder_kernel_size = cfg.model.encoder.encoder_kernel_size

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim, encoder_embedding_dim,
                    kernel_size=encoder_kernel_size, stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1, w_init_gain='relu'
                ),
                nn.BatchNorm1d(encoder_embedding_dim)
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            encoder_embedding_dim,
            int(encoder_embedding_dim / 2), 1,
            batch_first=True, bidirectional=True
        )

    def forward(self, x, input_lengths):
        """Forward pass for encoder.

        Args:
            x: embedded inputs (B, embed_dim, T)
            input_lengths: input lengths

        Returns:
            encoder outputs (B, T, embed_dim)
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True, enforce_sorted=False
        )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )

        return outputs

    def inference(self, x):
        """Inference mode forward pass."""
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    """Autoregressive decoder with location-sensitive attention."""

    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.n_mel_channels = cfg.preprocess.n_mel
        self.n_frames_per_step = cfg.model.decoder.n_frames_per_step
        self.encoder_embedding_dim = cfg.model.encoder.encoder_embedding_dim
        self.attention_rnn_dim = cfg.model.attention.attention_rnn_dim
        self.decoder_rnn_dim = cfg.model.decoder.decoder_rnn_dim
        self.prenet_dim = cfg.model.decoder.prenet_dim
        self.max_decoder_steps = cfg.model.decoder.max_decoder_steps
        self.gate_threshold = cfg.model.decoder.gate_threshold
        self.p_attention_dropout = cfg.model.decoder.p_attention_dropout
        self.p_decoder_dropout = cfg.model.decoder.p_decoder_dropout

        attention_dim = cfg.model.attention.attention_dim
        attention_location_n_filters = cfg.model.location_layer.attention_location_n_filters
        attention_location_kernel_size = cfg.model.location_layer.attention_location_kernel_size

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim]
        )

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim
        )

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step
        )

        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid'
        )

    def get_go_frame(self, memory):
        """Get all zeros frames to use as first decoder input."""
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step
        ).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """Initialize attention rnn states, decoder rnn states, attention weights, etc."""
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim
        ).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim
        ).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim
        ).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim
        ).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME
        ).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME
        ).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim
        ).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """Prepare decoder inputs (mel outputs for teacher forcing)."""
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        # Grouping multiple frames: (B, T_out, n_mel_channels) -> (B, T_out/r, n_mel_channels*r)
        decoder_inputs = decoder_inputs.contiguous().view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step), -1
        )
        # (B, T_out/r, n_mel_channels*r) -> (T_out/r, B, n_mel_channels*r)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """Prepare decoder outputs for output."""
        # (T_out/r, B) -> (B, T_out/r)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out/r, B) -> (B, T_out/r)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        # tile gate_outputs to make frames per step
        B = gate_outputs.size(0)
        gate_outputs = gate_outputs.contiguous().view(-1, 1).repeat(
            1, self.n_frames_per_step
        ).view(B, -1)

        # (T_out/r, B, n_mel_channels*r) -> (B, T_out/r, n_mel_channels*r)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames: (B, T_out/r, n_mel_channels*r) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels
        )
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """Decoder step using stored states, attention and memory."""
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask
        )

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1
        )
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context
        )

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """Decoder forward pass for training.

        Args:
            memory: Encoder outputs (B, T_in, embed_dim)
            decoder_inputs: Decoder inputs for teacher forcing (B, n_mel, T_out)
            memory_lengths: Encoder output lengths for attention masking

        Returns:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """Decoder inference.

        Args:
            memory: Encoder outputs

        Returns:
            mel_outputs: mel outputs from the decoder
            gate_outputs: gate outputs from the decoder
            alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps // self.n_frames_per_step:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )

        return mel_outputs, gate_outputs, alignments


class Postnet(nn.Module):
    """Postnet: Five 1-d convolution with 512 channels and kernel size 5."""

    def __init__(self, cfg):
        super(Postnet, self).__init__()
        n_mel_channels = cfg.preprocess.n_mel
        postnet_embedding_dim = cfg.model.postnet.postnet_embedding_dim
        postnet_kernel_size = cfg.model.postnet.postnet_kernel_size
        postnet_n_convolutions = cfg.model.postnet.postnet_n_convolutions

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_mel_channels, postnet_embedding_dim,
                    kernel_size=postnet_kernel_size, stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1, w_init_gain='tanh'
                ),
                nn.BatchNorm1d(postnet_embedding_dim)
            )
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim, postnet_embedding_dim,
                        kernel_size=postnet_kernel_size, stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1, w_init_gain='tanh'
                    ),
                    nn.BatchNorm1d(postnet_embedding_dim)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim, n_mel_channels,
                    kernel_size=postnet_kernel_size, stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1, w_init_gain='linear'
                ),
                nn.BatchNorm1d(n_mel_channels)
            )
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x


class Tacotron2(nn.Module):
    """Tacotron2 TTS Model."""

    def __init__(self, cfg):
        super(Tacotron2, self).__init__()
        self.cfg = cfg
        self.n_mel_channels = cfg.preprocess.n_mel
        self.mask_padding = cfg.train.mask_padding

        # Number of symbols (phonemes/characters)
        n_symbols = cfg.model.text_token_num
        symbols_embedding_dim = cfg.model.encoder.symbols_embedding_dim

        self.embedding = nn.Embedding(n_symbols, symbols_embedding_dim)
        std = sqrt(2.0 / (n_symbols + symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.postnet = Postnet(cfg)

        # Speaker embedding for multi-speaker
        self.speaker_emb = None
        if cfg.train.multi_speaker_training:
            if os.path.exists(os.path.join(
                cfg.preprocess.processed_dir, cfg.dataset[0], "spk2id.json"
            )):
                with open(
                    os.path.join(
                        cfg.preprocess.processed_dir, cfg.dataset[0], "spk2id.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    cfg.model.encoder.encoder_embedding_dim,
                )

    def parse_output(self, outputs, output_lengths=None):
        """Parse and mask outputs."""
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, outputs[0].size(2))
            mask = mask.unsqueeze(1).expand(mask.size(0), self.n_mel_channels, mask.size(1))

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs[0] = outputs[0].transpose(-2, -1)
        outputs[1] = outputs[1].transpose(-2, -1)
        return outputs

    def forward(self, data):
        """Forward pass for training.

        Args:
            data: Dictionary containing:
                - texts: Input text sequences (B, T_text)
                - text_len: Text lengths (B,)
                - mel: Target mel spectrograms (B, T_mel, n_mel)
                - target_len: Mel lengths (B,)
                - spk_id: Speaker IDs (B,) for multi-speaker
        """
        texts = data["texts"]
        src_lens = data["text_len"]
        mels = data["mel"]
        mel_lens = data["target_len"]

        # Get speaker IDs if available
        speakers = data.get("spk_id", None)

        max_src_len = texts.size(1)

        # Embed input text
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, src_lens)

        # Add speaker embedding if multi-speaker
        if self.speaker_emb is not None and speakers is not None:
            encoder_outputs = encoder_outputs + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        # Transpose mel for decoder (B, T, n_mel) -> (B, n_mel, T)
        mels_transposed = mels.transpose(1, 2)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels_transposed, memory_lengths=src_lens
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            mel_lens
        )

        return {
            "mel_outputs": outputs[0],
            "mel_outputs_postnet": outputs[1],
            "gate_outputs": outputs[2],
            "alignments": outputs[3],
            "mel_lens": mel_lens,
        }

    def inference(self, data):
        """Inference mode forward pass.

        Args:
            data: Dictionary containing:
                - texts: Input text sequences (B, T_text)
                - text_len: Text lengths (B,)
                - spk_id: Speaker IDs (B,) for multi-speaker (optional)
        """
        texts = data["texts"]
        src_lens = data.get("text_len", None)
        speakers = data.get("spk_id", None)

        max_src_len = texts.size(1)

        # Embed input text
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        # Add speaker embedding if multi-speaker
        if self.speaker_emb is not None and speakers is not None:
            encoder_outputs = encoder_outputs + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )

        return {
            "mel_outputs": outputs[0],
            "mel_outputs_postnet": outputs[1],
            "gate_outputs": outputs[2],
            "alignments": outputs[3],
        }


class GuidedAttentionLoss(nn.Module):
    """Guided attention loss for encouraging diagonal attention.

    Based on "Efficiently Trainable Text-to-Speech System Based on
    Deep Convolutional Networks with Guided Attention".
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate guided attention loss.

        Args:
            att_ws: Batch of attention weights (B, T_max_out, T_max_in)
            ilens: Batch of input lengths (B,)
            olens: Batch of output lengths (B,)

        Returns:
            Guided attention loss value
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens, device=att_ws.device)
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens, device=att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens, device=None):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen), device=device)
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma, device=device
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma, device=None):
        grid_x, grid_y = torch.meshgrid(
            torch.arange(olen, device=device), torch.arange(ilen, device=device), indexing='ij'
        )
        grid_x, grid_y = grid_x.float(), grid_y.float()
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens, device=None):
        in_masks = self._make_non_pad_mask(ilens, device=device)
        out_masks = self._make_non_pad_mask(olens, device=device)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)

    def _make_non_pad_mask(self, lengths, device=None):
        return ~self._make_pad_mask(lengths, device=device)

    def _make_pad_mask(self, lengths, device=None):
        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        maxlen = int(max(lengths))

        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return mask


class Tacotron2Loss(nn.Module):
    """Tacotron2 Loss with optional guided attention."""

    def __init__(self, cfg):
        super(Tacotron2Loss, self).__init__()
        self.n_frames_per_step = cfg.model.decoder.n_frames_per_step
        self.use_guided_attn_loss = cfg.train.use_guided_attn_loss

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

        if self.use_guided_attn_loss:
            self.guided_attn_loss = GuidedAttentionLoss(
                sigma=cfg.train.guided_sigma,
                alpha=cfg.train.guided_lambda,
            )

    def forward(self, data, predictions):
        """Calculate Tacotron2 loss.

        Args:
            data: Dictionary containing target data
            predictions: Dictionary containing model predictions
        """
        mel_target = data["mel"]
        input_lengths = data["text_len"]
        output_lengths = data["target_len"]

        mel_out = predictions["mel_outputs"]
        mel_out_postnet = predictions["mel_outputs_postnet"]
        gate_out = predictions["gate_outputs"]
        alignments = predictions["alignments"]

        # Create gate target (1 at the end of sequence)
        gate_target = torch.zeros(gate_out.size(), device=gate_out.device)
        for i, length in enumerate(output_lengths):
            gate_target[i, length - 1:] = 1.0

        gate_target = gate_target.view(-1, 1)
        gate_out = gate_out.view(-1, 1)

        mel_loss = self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_out_postnet, mel_target)
        gate_loss = self.bce_loss(gate_out, gate_target)

        if self.use_guided_attn_loss:
            attn_loss = self.guided_attn_loss(
                alignments, input_lengths,
                (output_lengths + self.n_frames_per_step - 1) // self.n_frames_per_step
            )
            total_loss = mel_loss + gate_loss + attn_loss
            return {
                "loss": total_loss,
                "mel_loss": mel_loss,
                "gate_loss": gate_loss,
                "attn_loss": attn_loss,
            }
        else:
            total_loss = mel_loss + gate_loss
            return {
                "loss": total_loss,
                "mel_loss": mel_loss,
                "gate_loss": gate_loss,
                "attn_loss": torch.tensor([0.], device=mel_target.device),
            }
