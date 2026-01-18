# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
GlowTTS Dataset and Collator

Provides dataset and collator classes for training and testing GlowTTS.
Supports LJSpeech and LibriTTS datasets.
"""

import os
import json
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from models.base.base_dataset import (
    BaseOfflineDataset,
    BaseOfflineCollator,
    BaseTestDataset,
    BaseTestCollator,
)
from utils.data_utils import align_length
from text import text_to_sequence
from text.text_token_collation import phoneIDCollation


class GlowTTSDataset(BaseOfflineDataset):
    """
    Dataset for GlowTTS training.

    Loads phoneme sequences and mel spectrograms for training.
    Supports both LJSpeech and LibriTTS datasets.
    """

    def __init__(self, cfg, dataset, is_valid=False):
        BaseOfflineDataset.__init__(self, cfg, dataset, is_valid=is_valid)
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.is_valid = is_valid

        # Noise augmentation settings (only for training)
        self.add_noise = getattr(cfg.train, "add_noise", True) and not is_valid
        self.noise_scale = getattr(cfg.train, "noise_scale", 1.0)

        # Load speaker mapping if multi-speaker
        self.speaker_map = {}
        spk2id_path = os.path.join(
            cfg.preprocess.processed_dir, dataset, "spk2id.json"
        )
        if os.path.exists(spk2id_path):
            with open(spk2id_path, "r") as f:
                self.speaker_map = json.load(f)

        # Check and filter metadata
        self.metadata = self._check_metadata()

    def _check_metadata(self):
        """Filter out samples with missing files."""
        new_metadata = []
        for utt_info in self.metadata:
            dataset = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = f"{dataset}_{uid}"

            # Check if mel exists
            if hasattr(self, 'utt2mel_path') and utt in self.utt2mel_path:
                if not os.path.exists(self.utt2mel_path[utt]):
                    continue
            new_metadata.append(utt_info)
        return new_metadata

    def __getitem__(self, index):
        # Get base features from parent class
        single_feature = BaseOfflineDataset.__getitem__(self, index)

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = f"{dataset}_{uid}"

        # Speaker ID
        if len(self.speaker_map) > 0:
            speaker_key = utt_info.get("Singer", utt_info.get("Speaker", dataset))
            speaker_id = self.speaker_map.get(speaker_key, 0)
        else:
            speaker_id = 0

        single_feature.update({
            "spk_id": speaker_id,
            "uid": uid,
        })

        # Add noise augmentation to mel spectrogram (training only)
        if self.add_noise and "mel" in single_feature:
            mel = single_feature["mel"]
            # Add random uniform noise scaled by noise_scale
            noise = np.random.rand(*mel.shape).astype(mel.dtype) * self.noise_scale
            single_feature["mel"] = mel + noise

        return self._clip_if_too_long(single_feature)

    def _random_select(self, feature_seq_len, max_seq_len, ending_ts=2812):
        """Select a random segment from long sequences."""
        ts = max(feature_seq_len - max_seq_len, 0)
        ts = min(ts, ending_ts - max_seq_len) if ending_ts > max_seq_len else ts
        start = random.randint(0, max(ts, 0))
        end = start + max_seq_len
        return start, end

    def _clip_if_too_long(self, sample, max_seq_len=1000):
        """Clip sequences that are too long."""
        if sample.get("target_len", 0) <= max_seq_len:
            return sample

        start, end = self._random_select(sample["target_len"], max_seq_len)
        sample["target_len"] = end - start

        # Clip mel and other frame-level features
        for k in ["mel", "frame_pitch", "frame_energy"]:
            if k in sample:
                sample[k] = sample[k][start:end]

        return sample

    def __len__(self):
        return len(self.metadata)


class GlowTTSCollator(BaseOfflineCollator):
    """
    Collator for GlowTTS training.

    Zero-pads model inputs and targets based on the longest sequence in batch.
    """

    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long)
                    for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "phone_len":
                packed_batch_features["phone_len"] = torch.LongTensor(
                    [b["phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["phone_len"], 1), dtype=torch.long)
                    for b in batch
                ]
                packed_batch_features["phone_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "spk_id":
                packed_batch_features["spk_id"] = torch.LongTensor(
                    [b["spk_id"] for b in batch]
                )
            elif key == "uid":
                packed_batch_features[key] = [b["uid"] for b in batch]
            else:
                if isinstance(batch[0][key], np.ndarray):
                    values = [torch.from_numpy(b[key]) for b in batch]
                else:
                    values = [b[key] for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features


class GlowTTSTestDataset(BaseTestDataset):
    """
    Test Dataset for GlowTTS inference.
    """

    def __init__(self, args, cfg, infer_type=None):
        self.args = args
        self.cfg = cfg

        datasets = cfg.dataset
        is_bigdata = False

        assert len(datasets) >= 1
        if len(datasets) > 1:
            datasets.sort()
            bigdata_version = "_".join(datasets)
            processed_data_dir = os.path.join(
                cfg.preprocess.processed_dir, bigdata_version
            )
            is_bigdata = True
        else:
            processed_data_dir = os.path.join(
                cfg.preprocess.processed_dir, args.dataset
            )

        if args.test_list_file:
            self.metafile_path = args.test_list_file
            self.metadata = self._get_metadata()
        else:
            assert args.testing_set
            source_metafile_path = os.path.join(
                cfg.preprocess.processed_dir,
                args.dataset,
                f"{args.testing_set}.json",
            )
            with open(source_metafile_path, "r") as f:
                self.metadata = json.load(f)

        self.datasets = datasets
        self.data_root = processed_data_dir
        self.is_bigdata = is_bigdata
        self.source_dataset = args.dataset

        # Load speaker mapping
        self.speaker_map = {}
        spk2id_path = os.path.join(processed_data_dir, "spk2id.json")
        if os.path.exists(spk2id_path):
            with open(spk2id_path, "r") as f:
                self.speaker_map = json.load(f)

        # Setup phone sequence paths
        if cfg.preprocess.use_phone:
            self.utt2phone_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = f"{dataset}_{uid}"
                self.utt2phone_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.phone_dir,
                    f"{uid}.phone",
                )

    def _get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata

    def __getitem__(self, index):
        single_feature = {}

        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = f"{dataset}_{uid}"

        # Load phone sequence
        if self.cfg.preprocess.use_phone:
            phone_path = self.utt2phone_path[utt]
            with open(phone_path, "r") as f:
                phones = f.readlines()[0].strip()
            phones_seq = phones.split(" ")

            phon_id_collator = phoneIDCollation(self.cfg, dataset=dataset)
            phone_seq = phon_id_collator.get_phone_id_sequence(
                self.cfg, phones_seq
            )
            phone_seq = np.array(phone_seq)
        elif self.cfg.preprocess.use_text:
            text = utt_info["Text"]
            phone_seq = np.array(
                text_to_sequence(text, self.cfg.preprocess.text_cleaners)
            )
        else:
            raise ValueError(
                "Either use_phone or use_text must be True in config"
            )

        phone_len = len(phone_seq)

        # Speaker ID
        if len(self.speaker_map) > 0:
            speaker_key = utt_info.get("Singer", utt_info.get("Speaker", dataset))
            speaker_id = self.speaker_map.get(speaker_key, 0)
        else:
            speaker_id = 0

        single_feature.update({
            "phone_seq": phone_seq,
            "phone_len": phone_len,
            "spk_id": speaker_id,
            "uid": uid,
        })

        return single_feature

    def __len__(self):
        return len(self.metadata)


class GlowTTSTestCollator(BaseTestCollator):
    """
    Collator for GlowTTS inference.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "phone_len":
                packed_batch_features["phone_len"] = torch.LongTensor(
                    [b["phone_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["phone_len"], 1), dtype=torch.long)
                    for b in batch
                ]
                packed_batch_features["phone_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "spk_id":
                packed_batch_features["spk_id"] = torch.LongTensor(
                    [b["spk_id"] for b in batch]
                )
            elif key == "uid":
                packed_batch_features[key] = [b["uid"] for b in batch]
            else:
                if isinstance(batch[0][key], np.ndarray):
                    values = [torch.from_numpy(b[key]) for b in batch]
                else:
                    values = [b[key] for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
