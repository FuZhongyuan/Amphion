# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GlowTTS Inference

Inference class for GlowTTS model. Supports both batch inference
and single utterance synthesis.
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path

from models.tts.base.tts_inferece import TTSInference
from models.tts.glowtts.glowtts import GlowTTS
from models.tts.glowtts.glowtts_dataset import GlowTTSTestDataset, GlowTTSTestCollator
from models.vocoders.vocoder_inference import synthesis
from utils.util import load_config
from utils.io import save_audio
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation
from text import text_to_sequence


class GlowTTSInference(TTSInference):
    """
    Inference class for GlowTTS model.

    Supports:
    - Batch inference on test dataset
    - Single utterance synthesis from text
    """

    def __init__(self, args, cfg):
        TTSInference.__init__(self, args, cfg)
        self.args = args
        self.cfg = cfg
        self.infer_type = args.mode

    def _build_model(self):
        """Build GlowTTS model."""
        self.model = GlowTTS(self.cfg)
        return self.model

    def load_model(self, state_dict):
        """Load model weights from state dict."""
        raw_dict = state_dict.get("model", state_dict)
        clean_dict = OrderedDict()

        for k, v in raw_dict.items():
            if k.startswith("module."):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v

        self.model.load_state_dict(clean_dict)

    def _build_test_dataset(self):
        """Build test dataset and collator."""
        return GlowTTSTestDataset, GlowTTSTestCollator

    @staticmethod
    def _parse_vocoder(vocoder_dir):
        """Parse vocoder config and checkpoint path."""
        vocoder_dir = os.path.abspath(vocoder_dir)
        ckpt_list = [ckpt for ckpt in Path(vocoder_dir).glob("*.pt")]

        # Sort by step number (different formats supported)
        try:
            ckpt_list.sort(
                key=lambda x: int(x.stem.split("_")[-2].split("-")[-1]),
                reverse=True
            )
        except (ValueError, IndexError):
            ckpt_list.sort(key=lambda x: int(x.stem), reverse=True)

        ckpt_path = str(ckpt_list[0])
        vocoder_cfg = load_config(
            os.path.join(vocoder_dir, "args.json"), lowercase=True
        )
        return vocoder_cfg, ckpt_path

    @torch.inference_mode()
    def inference_for_batches(self):
        """
        Run batch inference on test dataset.

        Generates mel spectrograms for all samples in the test set
        and synthesizes audio using vocoder.
        """
        for i, batch in tqdm(enumerate(self.test_dataloader)):
            y_pred, mel_lens = self._inference_each_batch(batch)

            y_ls = y_pred.chunk(self.test_batch_size)
            tgt_ls = mel_lens.chunk(self.test_batch_size)

            for j, (mel, length) in enumerate(zip(y_ls, tgt_ls)):
                length = length.item()
                mel = mel.squeeze(0)[:length].detach().cpu()

                uid = self.test_dataset.metadata[
                    i * self.test_batch_size + j
                ]["Uid"]
                torch.save(mel, os.path.join(self.args.output_dir, f"{uid}.pt"))

        # Synthesize audio using vocoder
        vocoder_cfg, vocoder_ckpt = self._parse_vocoder(self.args.vocoder_dir)
        res = synthesis(
            cfg=vocoder_cfg,
            vocoder_weight_file=vocoder_ckpt,
            n_samples=None,
            pred=[
                torch.load(
                    os.path.join(
                        self.args.output_dir, f"{item['Uid']}.pt"
                    )
                ).numpy()
                for item in self.test_dataset.metadata
            ],
        )

        # Save audio files
        for item, wav in zip(self.test_dataset.metadata, res):
            uid = item["Uid"]
            save_audio(
                os.path.join(self.args.output_dir, f"{uid}.wav"),
                wav.numpy(),
                self.cfg.preprocess.sample_rate,
                add_silence=True,
                turn_up=True,
            )
            # Clean up temporary mel file
            os.remove(os.path.join(self.args.output_dir, f"{uid}.pt"))

    @torch.inference_mode()
    def _inference_each_batch(self, batch_data):
        """
        Run inference on a single batch.

        Args:
            batch_data: Batch data from dataloader

        Returns:
            mel_pred: Predicted mel spectrograms [B, T, n_mel]
            mel_lens: Mel spectrogram lengths [B]
        """
        device = self.accelerator.device

        # Move batch to device
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)

        # Get inference parameters
        noise_scale = getattr(self.args, 'noise_scale', 0.667)
        length_scale = getattr(self.args, 'length_scale', 1.0)

        # Run inference
        output = self.model(
            batch_data,
            gen=True,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )

        mel_pred = output["mel_out"]
        mel_lens = output["mel_lens"].cpu()

        return mel_pred, mel_lens

    def inference_for_single_utterance(self):
        """
        Synthesize audio from a single text input.

        Returns:
            audio: Generated audio waveform
        """
        text = self.args.text
        noise_scale = getattr(self.args, 'noise_scale', 0.667)
        length_scale = getattr(self.args, 'length_scale', 1.0)

        # Get phone symbol file
        phone_symbol_file = None
        if self.cfg.preprocess.use_phone and self.cfg.preprocess.phone_extractor != "lexicon":
            phone_symbol_file = os.path.join(
                self.exp_dir, self.cfg.preprocess.symbols_dict
            )
            assert os.path.exists(phone_symbol_file), \
                f"Phone symbol file not found: {phone_symbol_file}"

        # Convert text to phone sequence
        if self.cfg.preprocess.use_phone:
            phone_extractor = phoneExtractor(self.cfg)
            phone_seq = phone_extractor.extract_phone(text)  # list

            phon_id_collator = phoneIDCollation(
                self.cfg, symbols_dict_file=phone_symbol_file
            )
            # phone_seq = ["{"] + phone_seq + ["}"]
            # phone_seq is already a list of phones, get_phone_id_sequence will wrap with braces
            phone_id_seq = phon_id_collator.get_phone_id_sequence(
                self.cfg, phone_seq
            )
        else:
            phone_id_seq = text_to_sequence(
                text, self.cfg.preprocess.text_cleaners
            )

        # Convert to tensor
        phone_id_seq = np.array(phone_id_seq)
        phone_id_seq = torch.from_numpy(phone_id_seq)

        # Get speaker ID if multi-speaker
        speaker_id = None
        if (
            self.cfg.preprocess.use_spkid
            and self.cfg.train.multi_speaker_training
        ):
            spk2id_file = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            with open(spk2id_file, "r") as f:
                spk2id = json.load(f)
                speaker_id = spk2id.get(self.args.speaker_name, 0)
                speaker_id = torch.LongTensor([speaker_id])
        else:
            speaker_id = torch.LongTensor([0])

        # Prepare input
        with torch.no_grad():
            phone_seq = phone_id_seq.to(self.device).unsqueeze(0)
            phone_len = torch.LongTensor([phone_id_seq.size(0)]).to(self.device)
            speaker_id = speaker_id.to(self.device)

            data = {
                "phone_seq": phone_seq,
                "phone_len": phone_len,
                "spk_id": speaker_id,
            }

            # Run inference
            output = self.model(
                data,
                gen=True,
                noise_scale=noise_scale,
                length_scale=length_scale,
            )

            mel_pred = output["mel_out"]

            # Synthesize audio using vocoder
            vocoder_cfg, vocoder_ckpt = self._parse_vocoder(self.args.vocoder_dir)
            audio = synthesis(
                cfg=vocoder_cfg,
                vocoder_weight_file=vocoder_ckpt,
                n_samples=None,
                pred=mel_pred.cpu(),
            )

        return audio[0]

    @staticmethod
    def add_arguments(parser):
        """Add GlowTTS-specific arguments to parser."""
        parser.add_argument(
            "--noise_scale",
            type=float,
            default=0.667,
            help="Scale for sampling noise during inference",
        )
        parser.add_argument(
            "--length_scale",
            type=float,
            default=1.0,
            help="Scale for duration during inference (larger = slower)",
        )
        return parser
