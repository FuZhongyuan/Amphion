# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HDF5-based GAN vocoder dataset for efficient data loading.
"""

import os
import json
import torch
import random
import numpy as np

from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from utils.hdf5_utils import HDF5Reader


class GANVocoderHDF5Dataset(torch.utils.data.Dataset):
    """GAN Vocoder dataset that reads features from HDF5 files."""

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
            self.utt2sample_key[utt] = uid

        # Load eval sample for validation
        eval_index = random.randint(0, len(self.metadata) - 1)
        eval_utt_info = self.metadata[eval_index]
        eval_uid = eval_utt_info["Uid"]

        eval_data = self.hdf5_reader.read_sample(eval_uid)
        if eval_data is not None:
            self.eval_audio = eval_data.get("audio")
            if cfg.preprocess.use_mel:
                self.eval_mel = eval_data.get("mel")
            if cfg.preprocess.use_frame_pitch:
                self.eval_pitch = eval_data.get("pitch")
        else:
            # Fallback: try to find another valid sample
            for i in range(len(self.metadata)):
                eval_utt_info = self.metadata[i]
                eval_uid = eval_utt_info["Uid"]
                eval_data = self.hdf5_reader.read_sample(eval_uid)
                if eval_data is not None:
                    self.eval_audio = eval_data.get("audio")
                    if cfg.preprocess.use_mel:
                        self.eval_mel = eval_data.get("mel")
                    if cfg.preprocess.use_frame_pitch:
                        self.eval_pitch = eval_data.get("pitch")
                    break

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

            if single_feature["target_len"] <= self.cfg.preprocess.cut_mel_frame:
                mel = np.pad(
                    mel,
                    ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0, mel.shape[-1] - self.cfg.preprocess.cut_mel_frame
                    )
                    end = start + self.cfg.preprocess.cut_mel_frame
                    single_feature["start"] = start
                    single_feature["end"] = end
                mel = mel[:, single_feature["start"] : single_feature["end"]]
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

            if single_feature["target_len"] <= self.cfg.preprocess.cut_mel_frame:
                aligned_frame_pitch = np.pad(
                    aligned_frame_pitch,
                    (
                        (
                            0,
                            self.cfg.preprocess.cut_mel_frame
                            * self.cfg.preprocess.hop_size
                            - audio.shape[-1],
                        )
                    ),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0,
                        aligned_frame_pitch.shape[-1]
                        - self.cfg.preprocess.cut_mel_frame,
                    )
                    end = start + self.cfg.preprocess.cut_mel_frame
                    single_feature["start"] = start
                    single_feature["end"] = end
                aligned_frame_pitch = aligned_frame_pitch[
                    single_feature["start"] : single_feature["end"]
                ]
            single_feature["frame_pitch"] = aligned_frame_pitch

        if self.cfg.preprocess.use_audio:
            audio = sample_data.get("audio")
            if audio is None:
                raise ValueError(f"Audio feature not found for sample {sample_key}")

            assert "target_len" in single_feature.keys()

            if (
                audio.shape[-1]
                <= self.cfg.preprocess.cut_mel_frame * self.cfg.preprocess.hop_size
            ):
                audio = np.pad(
                    audio,
                    (
                        (
                            0,
                            self.cfg.preprocess.cut_mel_frame
                            * self.cfg.preprocess.hop_size
                            - audio.shape[-1],
                        )
                    ),
                    mode="constant",
                )
            else:
                if "start" not in single_feature.keys():
                    audio = audio[
                        0 : self.cfg.preprocess.cut_mel_frame
                        * self.cfg.preprocess.hop_size
                    ]
                else:
                    audio = audio[
                        single_feature["start"]
                        * self.cfg.preprocess.hop_size : single_feature["end"]
                        * self.cfg.preprocess.hop_size,
                    ]
            single_feature["audio"] = audio

        if self.cfg.preprocess.use_amplitude_phase:
            logamp = sample_data.get("log_amplitude")
            pha = sample_data.get("phase")
            rea = sample_data.get("real")
            imag = sample_data.get("imaginary")

            assert "target_len" in single_feature.keys()

            if logamp is not None and pha is not None and rea is not None and imag is not None:
                if single_feature["target_len"] <= self.cfg.preprocess.cut_mel_frame:
                    logamp = np.pad(
                        logamp,
                        ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                        mode="constant",
                    )
                    pha = np.pad(
                        pha,
                        ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                        mode="constant",
                    )
                    rea = np.pad(
                        rea,
                        ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                        mode="constant",
                    )
                    imag = np.pad(
                        imag,
                        ((0, 0), (0, self.cfg.preprocess.cut_mel_frame - mel.shape[-1])),
                        mode="constant",
                    )
                else:
                    logamp = logamp[:, single_feature["start"] : single_feature["end"]]
                    pha = pha[:, single_feature["start"] : single_feature["end"]]
                    rea = rea[:, single_feature["start"] : single_feature["end"]]
                    imag = imag[:, single_feature["start"] : single_feature["end"]]
                single_feature["logamp"] = logamp
                single_feature["pha"] = pha
                single_feature["rea"] = rea
                single_feature["imag"] = imag

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


class GANVocoderHDF5Collator(object):
    """Zero-pads model inputs and targets based on number of frames per step."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, n_mels, frame]
        # frame_pitch: [b, frame]
        # audios: [b, frame * hop_size]

        for key in batch[0].keys():
            if key in ["target_len", "start", "end"]:
                continue
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
