# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from pathlib import Path

from models.tts.base.tts_inferece import TTSInference
from models.tts.tacotron2.tacotron2_dataset import (
    Tacotron2TestDataset,
    Tacotron2TestCollator,
)
from models.tts.tacotron2.tacotron2 import Tacotron2
from utils.util import load_config
from utils.io import save_audio
from models.vocoders.vocoder_inference import synthesis
from processors.phone_extractor import phoneExtractor
from text.text_token_collation import phoneIDCollation
from text import text_to_sequence


class Tacotron2Inference(TTSInference):
    """Inference class for Tacotron2 model."""

    def __init__(self, args, cfg):
        TTSInference.__init__(self, args, cfg)
        self.args = args
        self.cfg = cfg
        self.infer_type = args.mode

    def _build_model(self):
        self.model = Tacotron2(self.cfg)
        return self.model

    def load_model(self, state_dict):
        """Load model from state dict."""
        raw_dict = state_dict.get("model", state_dict)
        clean_dict = OrderedDict()
        for k, v in raw_dict.items():
            if k.startswith("module."):
                clean_dict[k[7:]] = v
            else:
                clean_dict[k] = v

        self.model.load_state_dict(clean_dict)

    def _build_test_dataset(self):
        return Tacotron2TestDataset, Tacotron2TestCollator

    @staticmethod
    def _parse_vocoder(vocoder_dir):
        """Parse vocoder config."""
        vocoder_dir = os.path.abspath(vocoder_dir)
        ckpt_list = [ckpt for ckpt in Path(vocoder_dir).glob("*.pt")]
        # Sort by step number
        ckpt_list.sort(
            key=lambda x: int(x.stem.split("_")[-2].split("-")[-1])
            if "_" in x.stem else int(x.stem),
            reverse=True
        )
        ckpt_path = str(ckpt_list[0])
        vocoder_cfg = load_config(
            os.path.join(vocoder_dir, "args.json"), lowercase=True
        )
        return vocoder_cfg, ckpt_path

    @torch.inference_mode()
    def inference_for_batches(self):
        """Run inference for batch of samples."""
        y_pred = []
        for i, batch in tqdm(enumerate(self.test_dataloader)):
            y_pred, mel_lens, _ = self._inference_each_batch(batch)
            y_ls = y_pred.chunk(self.test_batch_size)
            tgt_ls = mel_lens.chunk(self.test_batch_size)
            j = 0
            for it, l in zip(y_ls, tgt_ls):
                l = l.item()
                it = it.squeeze(0)[:l].detach().cpu()

                uid = self.test_dataset.metadata[i * self.test_batch_size + j]["Uid"]
                torch.save(it, os.path.join(self.args.output_dir, f"{uid}.pt"))
                j += 1

        vocoder_cfg, vocoder_ckpt = self._parse_vocoder(self.args.vocoder_dir)
        res = synthesis(
            cfg=vocoder_cfg,
            vocoder_weight_file=vocoder_ckpt,
            n_samples=None,
            pred=[
                torch.load(
                    os.path.join(self.args.output_dir, "{}.pt".format(item["Uid"]))
                ).numpy()
                for item in self.test_dataset.metadata
            ],
        )
        for it, wav in zip(self.test_dataset.metadata, res):
            uid = it["Uid"]
            save_audio(
                os.path.join(self.args.output_dir, f"{uid}.wav"),
                wav.numpy(),
                self.cfg.preprocess.sample_rate,
                add_silence=True,
                turn_up=True,
            )
            os.remove(os.path.join(self.args.output_dir, f"{uid}.pt"))

    @torch.inference_mode()
    def _inference_each_batch(self, batch_data):
        """Run inference for a single batch."""
        device = self.accelerator.device
        for k, v in batch_data.items():
            if isinstance(v, torch.Tensor):
                batch_data[k] = v.to(device)

        output = self.model.inference(batch_data)
        pred_res = output["mel_outputs_postnet"]

        # Calculate mel lengths from gate outputs
        gate_outputs = output["gate_outputs"]
        mel_lens = torch.zeros(gate_outputs.size(0), dtype=torch.long, device=device)
        for i in range(gate_outputs.size(0)):
            gate_probs = torch.sigmoid(gate_outputs[i])
            # Find first position where gate > threshold
            end_indices = (gate_probs > self.cfg.model.decoder.gate_threshold).nonzero()
            if len(end_indices) > 0:
                mel_lens[i] = end_indices[0].item() + 1
            else:
                mel_lens[i] = gate_outputs.size(1)

        return pred_res, mel_lens.cpu(), 0

    def inference_for_single_utterance(self):
        """Run inference for a single text input."""
        text = self.args.text

        # Get phone symbol file
        phone_symbol_file = None
        if self.cfg.preprocess.use_phone and self.cfg.preprocess.phone_extractor != "lexicon":
            phone_symbol_file = os.path.join(
                self.exp_dir, self.cfg.preprocess.symbols_dict
            )
            if os.path.exists(phone_symbol_file):
                pass
            else:
                phone_symbol_file = None

        # Convert text to phone sequence
        if self.cfg.preprocess.use_phone:
            phone_extractor = phoneExtractor(self.cfg)
            phone_seq = phone_extractor.extract_phone(text)
            phon_id_collator = phoneIDCollation(
                self.cfg, symbols_dict_file=phone_symbol_file
            )
            # phone_seq = ["{"] + phone_seq + ["}"]
            phone_id_seq = phon_id_collator.get_phone_id_sequence(self.cfg, phone_seq)
        else:
            # Use text directly with cleaners
            phone_id_seq = text_to_sequence(text, self.cfg.preprocess.text_cleaners)

        # Convert to tensor
        phone_id_seq = np.array(phone_id_seq)
        phone_id_seq = torch.from_numpy(phone_id_seq)

        # Get speaker ID if multi-speaker
        speaker_id = None
        if self.cfg.preprocess.use_spkid and self.cfg.train.multi_speaker_training:
            spk2id_file = os.path.join(self.exp_dir, self.cfg.preprocess.spk2id)
            if os.path.exists(spk2id_file):
                with open(spk2id_file, "r") as f:
                    spk2id = json.load(f)
                    if self.args.speaker_name and self.args.speaker_name in spk2id:
                        speaker_id = spk2id[self.args.speaker_name]
                        speaker_id = torch.from_numpy(np.array([speaker_id], dtype=np.int32))
                    else:
                        speaker_id = torch.tensor([0], dtype=torch.int32)
            else:
                speaker_id = torch.tensor([0], dtype=torch.int32)
        else:
            speaker_id = torch.tensor([0], dtype=torch.int32)

        with torch.no_grad():
            x_tst = phone_id_seq.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([phone_id_seq.size(0)]).to(self.device)
            if speaker_id is not None:
                speaker_id = speaker_id.to(self.device)

            data = {
                "texts": x_tst,
                "text_len": x_tst_lengths,
                "spk_id": speaker_id,
            }

            output = self.model.inference(data)
            pred_res = output["mel_outputs_postnet"]

            vocoder_cfg, vocoder_ckpt = self._parse_vocoder(self.args.vocoder_dir)
            audio = synthesis(
                cfg=vocoder_cfg,
                vocoder_weight_file=vocoder_ckpt,
                n_samples=None,
                pred=pred_res.cpu().numpy(),
            )

        return audio[0]
