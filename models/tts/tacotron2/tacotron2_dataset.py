# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils.data_utils import *
from models.base.base_dataset import (
    BaseOfflineCollator,
    BaseOfflineDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from text import text_to_sequence
from text.text_token_collation import phoneIDCollation


class Tacotron2Dataset(BaseOfflineDataset):
    """Dataset for Tacotron2 training."""

    def __init__(self, cfg, dataset, is_valid=False):
        BaseOfflineDataset.__init__(self, cfg, dataset, is_valid=is_valid)
        self.cfg = cfg
        self.batch_size = cfg.train.batch_size
        self.n_frames_per_step = cfg.model.decoder.n_frames_per_step

        # Build utt2lab path for phoneme/text labels
        self.utt2lab_path = {}
        for utt_info in self.metadata:
            dataset_name = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset_name, uid)

            if cfg.preprocess.use_phone:
                self.utt2lab_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset_name,
                    cfg.preprocess.phone_dir,
                    uid + ".phone",
                )
            else:
                self.utt2lab_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset_name,
                    cfg.preprocess.lab_dir,
                    uid + ".txt",
                )

        # Load speaker map if exists
        self.speaker_map = {}
        spk2id_path = os.path.join(cfg.preprocess.processed_dir, dataset, "spk2id.json")
        if os.path.exists(spk2id_path):
            with open(spk2id_path, "r") as f:
                self.speaker_map = json.load(f)

        # Filter metadata to only include valid samples
        self.metadata = self.check_metadata()

    def __getitem__(self, index):
        single_feature = BaseOfflineDataset.__getitem__(self, index)

        utt_info = self.metadata[index]
        dataset_name = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset_name, uid)

        # Load phoneme/text sequence
        if self.cfg.preprocess.use_phone:
            with open(self.utt2lab_path[utt], "r") as f:
                phones = f.readlines()[0].strip()
            phones_seq = phones.split(" ")
            phon_id_collator = phoneIDCollation(self.cfg, dataset=dataset_name)
            phones_ids = np.array(
                phon_id_collator.get_phone_id_sequence(self.cfg, phones_seq)
            )
        else:
            with open(self.utt2lab_path[utt], "r") as f:
                text = f.readlines()[0].strip()
            phones_ids = np.array(
                text_to_sequence(text, self.cfg.preprocess.text_cleaners)
            )

        text_len = len(phones_ids)

        # Get speaker ID
        if len(self.speaker_map) > 0:
            speaker_key = utt_info.get("Singer", utt_info.get("Speaker", dataset_name))
            speaker_id = self.speaker_map.get(speaker_key, 0)
        else:
            speaker_id = 0

        single_feature.update({
            "texts": phones_ids,
            "spk_id": speaker_id,
            "text_len": text_len,
            "uid": uid,
        })

        return self.pad_for_step(single_feature)

    def pad_for_step(self, sample):
        """Pad mel spectrogram to be divisible by n_frames_per_step."""
        if "mel" in sample and self.n_frames_per_step > 1:
            mel = sample["mel"]
            target_len = sample["target_len"]

            # Calculate padding needed
            remainder = target_len % self.n_frames_per_step
            if remainder != 0:
                pad_len = self.n_frames_per_step - remainder
                mel_padded = np.pad(mel, ((0, pad_len), (0, 0)), mode='constant')
                sample["mel"] = mel_padded
                sample["target_len"] = target_len + pad_len
                sample["r_len_pad"] = pad_len
            else:
                sample["r_len_pad"] = 0

        return sample

    def __len__(self):
        return len(self.metadata)

    def check_metadata(self):
        """Filter metadata to only include samples with valid files."""
        new_metadata = []
        for utt_info in self.metadata:
            dataset_name = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset_name, uid)

            # Check if mel file exists
            if not os.path.exists(self.utt2mel_path.get(utt, "")):
                continue

            # Check if label file exists
            if not os.path.exists(self.utt2lab_path.get(utt, "")):
                continue

            new_metadata.append(utt_info)

        return new_metadata


class Tacotron2Collator(BaseOfflineCollator):
    """Collator for Tacotron2 training."""

    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)
        self.n_frames_per_step = cfg.model.decoder.n_frames_per_step

    def __call__(self, batch):
        """Collate batch of samples.

        Returns dictionary with:
            - texts: (B, T_text)
            - text_len: (B,)
            - mel: (B, T_mel, n_mel)
            - target_len: (B,)
            - spk_id: (B,)
            - gate_target: (B, T_mel)
            - text_mask: (B, T_text, 1)
            - mask: (B, T_mel, 1)
        """
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "target_len":
                packed_batch_features["target_len"] = torch.LongTensor(
                    [b["target_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["target_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "text_len":
                packed_batch_features["text_len"] = torch.LongTensor(
                    [b["text_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["text_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["text_mask"] = pad_sequence(
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
            elif key == "r_len_pad":
                packed_batch_features["r_len_pad"] = torch.LongTensor(
                    [b["r_len_pad"] for b in batch]
                )
            elif key == "uid":
                packed_batch_features[key] = [b["uid"] for b in batch]
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        # Create gate target (1 at the end of each sequence)
        max_mel_len = packed_batch_features["mel"].size(1)
        gate_target = torch.zeros(len(batch), max_mel_len)
        for i, length in enumerate(packed_batch_features["target_len"]):
            gate_target[i, length - 1:] = 1.0
        packed_batch_features["gate_target"] = gate_target

        return packed_batch_features


class Tacotron2TestDataset(BaseTestDataset):
    """Test dataset for Tacotron2 inference."""

    def __init__(self, args, cfg, infer_type=None):
        self.args = args
        self.cfg = cfg
        self.infer_type = infer_type

        datasets = cfg.dataset
        is_bigdata = False

        assert len(datasets) >= 1
        if len(datasets) > 1:
            datasets.sort()
            bigdata_version = "_".join(datasets)
            processed_data_dir = os.path.join(cfg.preprocess.processed_dir, bigdata_version)
            is_bigdata = True
        else:
            processed_data_dir = os.path.join(cfg.preprocess.processed_dir, args.dataset)

        if args.test_list_file:
            self.metafile_path = args.test_list_file
            self.metadata = self.get_metadata()
        else:
            assert args.testing_set
            source_metafile_path = os.path.join(
                cfg.preprocess.processed_dir,
                args.dataset,
                "{}.json".format(args.testing_set),
            )
            with open(source_metafile_path, "r") as f:
                self.metadata = json.load(f)

        self.datasets = datasets
        self.data_root = processed_data_dir
        self.is_bigdata = is_bigdata
        self.source_dataset = args.dataset

        # Build utt2lab path
        self.utt2lab_path = {}
        for utt_info in self.metadata:
            dataset_name = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset_name, uid)

            if cfg.preprocess.use_phone:
                self.utt2lab_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset_name,
                    cfg.preprocess.phone_dir,
                    uid + ".phone",
                )
            else:
                self.utt2lab_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset_name,
                    cfg.preprocess.lab_dir,
                    uid + ".txt",
                )

        # Load speaker map
        self.speaker_map = {}
        spk2id_path = os.path.join(processed_data_dir, "spk2id.json")
        if os.path.exists(spk2id_path):
            with open(spk2id_path, "r") as f:
                self.speaker_map = json.load(f)

    def __getitem__(self, index):
        single_feature = {}

        utt_info = self.metadata[index]
        dataset_name = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset_name, uid)

        # Load phoneme/text sequence
        if self.cfg.preprocess.use_phone:
            with open(self.utt2lab_path[utt], "r") as f:
                phones = f.readlines()[0].strip()
            phones_seq = phones.split(" ")
            phon_id_collator = phoneIDCollation(self.cfg, dataset=dataset_name)
            phones_ids = np.array(
                phon_id_collator.get_phone_id_sequence(self.cfg, phones_seq)
            )
        else:
            with open(self.utt2lab_path[utt], "r") as f:
                text = f.readlines()[0].strip()
            phones_ids = np.array(
                text_to_sequence(text, self.cfg.preprocess.text_cleaners)
            )

        text_len = len(phones_ids)

        # Get speaker ID
        if len(self.speaker_map) > 0:
            speaker_key = utt_info.get("Singer", utt_info.get("Speaker", dataset_name))
            speaker_id = self.speaker_map.get(speaker_key, 0)
        else:
            speaker_id = 0

        single_feature.update({
            "texts": phones_ids,
            "spk_id": speaker_id,
            "text_len": text_len,
        })

        return single_feature

    def __len__(self):
        return len(self.metadata)

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return metadata


class Tacotron2TestCollator(BaseTestCollator):
    """Test collator for Tacotron2 inference."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        for key in batch[0].keys():
            if key == "text_len":
                packed_batch_features["text_len"] = torch.LongTensor(
                    [b["text_len"] for b in batch]
                )
                masks = [
                    torch.ones((b["text_len"], 1), dtype=torch.long) for b in batch
                ]
                packed_batch_features["text_mask"] = pad_sequence(
                    masks, batch_first=True, padding_value=0
                )
            elif key == "spk_id":
                packed_batch_features["spk_id"] = torch.LongTensor(
                    [b["spk_id"] for b in batch]
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
