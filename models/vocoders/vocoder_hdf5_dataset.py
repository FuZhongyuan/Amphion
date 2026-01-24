# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HDF5-based vocoder dataset for efficient data loading.
"""

from typing import Iterable
import os
import json
import torch
import numpy as np
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from torch.utils.data import ConcatDataset, Dataset
from utils.hdf5_utils import HDF5Reader


class VocoderHDF5Dataset(torch.utils.data.Dataset):
    """Vocoder dataset that reads features from HDF5 files."""

    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """
        assert isinstance(dataset, str)

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()

        self.data_root = processed_data_dir
        self.cfg = cfg
        self.dataset_name = dataset

        # Initialize HDF5 reader
        hdf5_cache_size = getattr(cfg.preprocess, 'hdf5_cache_size', 5)
        self.hdf5_reader = HDF5Reader(processed_data_dir, cache_size=hdf5_cache_size)

        # Build uid to sample key mapping
        self.utt2sample_key = {}
        for utt_info in self.metadata:
            dataset_name = utt_info["Dataset"]
            uid = utt_info["Uid"]
            utt = "{}_{}".format(dataset_name, uid)
            # Sample key in HDF5 is just the uid
            self.utt2sample_key[utt] = uid

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)
        sample_key = self.utt2sample_key[utt]

        single_feature = dict()

        # Read all features for this sample from HDF5
        sample_data = self.hdf5_reader.read_sample(sample_key)

        if sample_data is None:
            raise ValueError(f"Sample {sample_key} not found in HDF5 files")

        if self.cfg.preprocess.use_mel:
            mel = sample_data.get("mel")
            if mel is None:
                raise ValueError(f"Mel feature not found for sample {sample_key}")

            # Ensure correct shape [n_mels, T]
            if mel.shape[0] != self.cfg.preprocess.n_mel:
                mel = mel.T

            assert mel.shape[0] == self.cfg.preprocess.n_mel

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]

            single_feature["mel"] = mel

        if self.cfg.preprocess.use_frame_pitch:
            frame_pitch = sample_data.get("pitch")
            if frame_pitch is None:
                raise ValueError(f"Pitch feature not found for sample {sample_key}")

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = len(frame_pitch)

            aligned_frame_pitch = align_length(
                frame_pitch, single_feature["target_len"]
            )

            single_feature["frame_pitch"] = aligned_frame_pitch

        if self.cfg.preprocess.use_audio:
            audio = sample_data.get("audio")
            if audio is None:
                raise ValueError(f"Audio feature not found for sample {sample_key}")

            single_feature["audio"] = audio

        if self.cfg.preprocess.use_uv:
            uv = sample_data.get("uv")
            if uv is not None:
                single_feature["uv"] = uv

        if self.cfg.preprocess.use_amplitude_phase:
            log_amplitude = sample_data.get("log_amplitude")
            phase = sample_data.get("phase")
            real = sample_data.get("real")
            imaginary = sample_data.get("imaginary")

            if log_amplitude is not None:
                single_feature["log_amplitude"] = log_amplitude
            if phase is not None:
                single_feature["phase"] = phase
            if real is not None:
                single_feature["real"] = real
            if imaginary is not None:
                single_feature["imaginary"] = imaginary

        return single_feature

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return metadata

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def __len__(self):
        return len(self.metadata)

    def __del__(self):
        """Close HDF5 reader on deletion."""
        if hasattr(self, 'hdf5_reader'):
            self.hdf5_reader.close()


class VocoderHDF5ConcatDataset(ConcatDataset):
    """Concatenate multiple HDF5-based vocoder datasets."""

    def __init__(self, datasets: Iterable[Dataset], full_audio_inference=False):
        """Concatenate a series of datasets with their random inference audio merged."""
        super().__init__(datasets)

        self.cfg = self.datasets[0].cfg

        self.metadata = []

        # Merge metadata
        for dataset in self.datasets:
            self.metadata += dataset.metadata

        # Merge random inference features
        if full_audio_inference:
            self.eval_audios = []
            self.eval_dataset_names = []
            if self.cfg.preprocess.use_mel:
                self.eval_mels = []
            if self.cfg.preprocess.use_frame_pitch:
                self.eval_pitchs = []
            for dataset in self.datasets:
                self.eval_audios.append(dataset.eval_audio)
                self.eval_dataset_names.append(dataset.get_dataset_name())
                if self.cfg.preprocess.use_mel:
                    self.eval_mels.append(dataset.eval_mel)
                if self.cfg.preprocess.use_frame_pitch:
                    self.eval_pitchs.append(dataset.eval_pitch)


class VocoderHDF5Collator(object):
    """Zero-pads model inputs and targets based on number of frames per step."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, n_mels, frame]
        # frame_pitch: [b, frame]
        # audios: [b, frame * hop_size]

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
            elif key == "mel":
                values = [torch.from_numpy(b[key]).T for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
