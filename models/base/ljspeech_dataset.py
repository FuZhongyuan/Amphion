# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa
import torch
import json
import numpy as np
import logging
import pickle
import os
import csv
from pathlib import Path
import random
from tqdm import tqdm


class WarningFilter(logging.Filter):
    def filter(self, record):
        if record.name == "phonemizer" and record.levelno == logging.WARNING:
            return False
        if record.name == "qcloud_cos.cos_client" and record.levelno == logging.INFO:
            return False
        if record.name == "jieba" and record.levelno == logging.DEBUG:
            return False
        return True


filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)
logging.getLogger("qcloud_cos.cos_client").addFilter(filter)
logging.getLogger("jieba").addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LJSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_type="path",
        cfg=None,
    ):  # 'path' or 'meta'

        assert cfg is not None

        self.cache_type = cache_type
        self.cfg = cfg

        self.dataset_ratio_dict = self.cfg.dataset
        # JsonHParams doesn't have .get() method, use getattr instead
        self.ljspeech_ratio = getattr(self.dataset_ratio_dict, "ljspeech", 1.0)

        self.wav_paths = []
        self.metadata = {}  # wav_id -> {text, normalized_text, language}
        
        # Get data root path from cfg or use default
        if hasattr(self.cfg.preprocess, "ljspeech_data_root"):
            self.data_root = self.cfg.preprocess.ljspeech_data_root
        else:
            self.data_root = "[Please fill out your ljspeech data root path]"

        self.language_list = ["en"]  # LJSpeech is English only
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.index2num_frames = []

        self.wav_id2meta = {}

        # Get cache path from cfg or use default
        if hasattr(self.cfg.preprocess, "ljspeech_cache_path"):
            self.cache_folder = self.cfg.preprocess.ljspeech_cache_path
        else:
            self.cache_folder = "[Please fill out your ljspeech cache path]"
        
        Path(self.cache_folder).mkdir(parents=True, exist_ok=True)

        self.wav_paths_cache = os.path.join(self.cache_folder, "wav_paths_cache.pkl")
        self.metadata_cache = os.path.join(self.cache_folder, "metadata_cache.pkl")
        self.duration_cache = os.path.join(self.cache_folder, "duration_cache.pkl")
        self.phone_count_cache = os.path.join(
            self.cache_folder, "phone_count_cache.pkl"
        )
        self.wav_id2meta_cache = os.path.join(
            self.cache_folder, "wav_id2meta.pkl"
        )

        if cache_type == "path":
            if (
                os.path.exists(self.wav_paths_cache)
                and os.path.exists(self.metadata_cache)
                and os.path.exists(self.duration_cache)
                and os.path.exists(self.phone_count_cache)
            ):
                self.load_cached_paths()
            else:
                logger.info("Cache files not found. Building cache...")
                self.build_cache()
                self.load_cached_paths()

        if cache_type == "meta":
            if os.path.exists(self.wav_id2meta_cache):
                self.load_path2meta()
            else:
                logger.info("Meta cache not found. Building cache...")
                self.build_meta_cache()
                self.load_path2meta()

        self.num_frame_indices = np.array(
            sorted(
                range(len(self.index2num_frames)),
                key=lambda k: self.index2num_frames[k],
            )
        )

        self.duration_setting = {"min": 3, "max": 30}
        if hasattr(self.cfg.preprocess, "min_dur"):
            self.duration_setting["min"] = self.cfg.preprocess.min_dur
        if hasattr(self.cfg.preprocess, "max_dur"):
            self.duration_setting["max"] = self.cfg.preprocess.max_dur

    def g2p(self, text, language):
        """G2P conversion - to be implemented in VC version"""
        pass

    def load_metadata_from_csv(self):
        """Load metadata from metadata.csv"""
        metadata_path = os.path.join(self.data_root, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.csv not found at {metadata_path}")
        
        self.metadata = {}
        with open(metadata_path, "r", encoding="utf-8") as f:
            # Count total lines for progress bar
            total_lines = sum(1 for _ in f)
            f.seek(0)
            reader = csv.reader(f, delimiter="|")
            for row in tqdm(reader, total=total_lines, desc="Loading metadata from CSV"):
                if len(row) >= 3:
                    wav_id = row[0].strip()
                    text = row[1].strip()
                    normalized_text = row[2].strip()
                    self.metadata[wav_id] = {
                        "text": text,
                        "normalized_text": normalized_text,
                        "language": "en"
                    }
        
        logger.info(f"Loaded {len(self.metadata)} metadata entries from CSV")

    def build_cache(self):
        """Build cache files for path cache type"""
        logger.info("Building cache for LJSpeech dataset...")
        
        # Load metadata
        self.load_metadata_from_csv()
        
        # Get all wav files
        wav_dir = os.path.join(self.data_root, "wavs")
        if not os.path.exists(wav_dir):
            raise FileNotFoundError(f"wavs directory not found at {wav_dir}")
        
        all_wav_paths = []
        all_durations = []
        all_phone_counts = []
        
        # Process each wav file
        for wav_id in tqdm(sorted(self.metadata.keys()), desc="Building cache"):
            wav_file = os.path.join(wav_dir, f"{wav_id}.wav")
            if not os.path.exists(wav_file):
                logger.warning(f"Wav file not found: {wav_file}")
                continue
            
            # Get relative path
            rel_path = os.path.join("wavs", f"{wav_id}.wav")
            all_wav_paths.append(rel_path)
            
            # Load audio to get duration
            try:
                audio, sr = librosa.load(wav_file, sr=None)
                duration = len(audio) / sr
                
                # Get phone count using G2P
                text = self.metadata[wav_id]["normalized_text"]
                try:
                    from models.tts.maskgct.g2p.g2p_generation import chn_eng_g2p
                    _, phone_ids = chn_eng_g2p(text)
                    phone_count = len(phone_ids)
                except Exception as e:
                    logger.warning(f"G2P failed for {wav_id}: {e}")
                    phone_count = len(text.split()) * 2  # Rough estimate
                
                all_durations.append(duration)
                all_phone_counts.append(phone_count)
                
            except Exception as e:
                logger.warning(f"Failed to process {wav_file}: {e}")
                continue
        
        # Save cache
        with open(self.wav_paths_cache, "wb") as f:
            pickle.dump(all_wav_paths, f)
        with open(self.metadata_cache, "wb") as f:
            pickle.dump(self.metadata, f)
        with open(self.duration_cache, "wb") as f:
            pickle.dump(all_durations, f)
        with open(self.phone_count_cache, "wb") as f:
            pickle.dump(all_phone_counts, f)
        
        logger.info(f"Cache built successfully. Total samples: {len(all_wav_paths)}")

    def build_meta_cache(self):
        """Build meta cache for meta cache type"""
        logger.info("Building meta cache for LJSpeech dataset...")
        
        # Load metadata
        self.load_metadata_from_csv()
        
        # Get all wav files
        wav_dir = os.path.join(self.data_root, "wavs")
        if not os.path.exists(wav_dir):
            raise FileNotFoundError(f"wavs directory not found at {wav_dir}")
        
        all_wav_paths = []
        wav_id2meta = {}
        
        # Process each wav file
        for wav_id in tqdm(sorted(self.metadata.keys()), desc="Building meta cache"):
            wav_file = os.path.join(wav_dir, f"{wav_id}.wav")
            if not os.path.exists(wav_file):
                logger.warning(f"Wav file not found: {wav_file}")
                continue
            
            # Get relative path
            rel_path = os.path.join("wavs", f"{wav_id}.wav")
            all_wav_paths.append(rel_path)
            
            # Load audio to get duration
            try:
                audio, sr = librosa.load(wav_file, sr=None)
                duration = len(audio) / sr
                
                # Get phone count using G2P
                text = self.metadata[wav_id]["normalized_text"]
                phone_ids = []
                try:
                    from models.tts.maskgct.g2p.g2p_generation import chn_eng_g2p
                    _, phone_ids = chn_eng_g2p(text)
                    phone_count = len(phone_ids)
                except Exception as e:
                    logger.warning(f"G2P failed for {wav_id}: {e}")
                    phone_count = len(text.split()) * 2  # Rough estimate
                
                # Store meta
                wav_id2meta[wav_id] = {
                    "text": self.metadata[wav_id]["text"],
                    "normalized_text": self.metadata[wav_id]["normalized_text"],
                    "language": "en",
                    "duration": duration,
                    "phone_count": phone_count,
                    "phone_id": phone_ids
                }
                
            except Exception as e:
                logger.warning(f"Failed to process {wav_file}: {e}")
                continue
        
        # Save cache
        with open(self.wav_paths_cache, "wb") as f:
            pickle.dump(all_wav_paths, f)
        with open(self.wav_id2meta_cache, "wb") as f:
            pickle.dump(wav_id2meta, f)
        
        logger.info(f"Meta cache built successfully. Total samples: {len(all_wav_paths)}")

    def load_cached_paths(self):
        logger.info("Loaded paths from cache files")
        with open(self.wav_paths_cache, "rb") as f:
            all_wav_paths = pickle.load(f)
        with open(self.metadata_cache, "rb") as f:
            all_metadata = pickle.load(f)

        # Select part of data according to ljspeech_ratio
        if self.ljspeech_ratio < 1.0:
            total_samples = len(all_wav_paths)
            num_samples = int(total_samples * self.ljspeech_ratio)
            selected_indices = random.sample(range(total_samples), num_samples)

            self.wav_paths = [all_wav_paths[i] for i in selected_indices]

            if self.cache_type == "path":
                with open(self.duration_cache, "rb") as f:
                    all_durations = pickle.load(f)
                with open(self.phone_count_cache, "rb") as f:
                    all_phone_counts = pickle.load(f)

                self.wav_path_index2duration = [
                    all_durations[i] for i in selected_indices
                ]
                self.wav_path_index2phonelen = [
                    all_phone_counts[i] for i in selected_indices
                ]
        else:
            assert self.ljspeech_ratio == 1.0

            self.wav_paths = all_wav_paths
            self.metadata = all_metadata
            if self.cache_type == "path":
                with open(self.duration_cache, "rb") as f:
                    self.wav_path_index2duration = pickle.load(f)
                with open(self.phone_count_cache, "rb") as f:
                    self.wav_path_index2phonelen = pickle.load(f)

        # Calculate the number of frames
        if self.cache_type == "path":
            self.index2num_frames = []
            for duration, phone_count in zip(
                self.wav_path_index2duration, self.wav_path_index2phonelen
            ):
                self.index2num_frames.append(duration * 50 + phone_count)

        logger.info("All LJSpeech paths got successfully, ratio: %f" % self.ljspeech_ratio)
        logger.info("Number of wavs: %d" % len(self.wav_paths))

    def save_cached_paths(self):
        with open(self.wav_paths_cache, "wb") as f:
            pickle.dump(self.wav_paths, f)
        with open(self.metadata_cache, "wb") as f:
            pickle.dump(self.metadata, f)
        if self.cache_type == "path":
            with open(self.duration_cache, "wb") as f:
                pickle.dump(self.wav_path_index2duration, f)
            with open(self.phone_count_cache, "wb") as f:
                pickle.dump(self.wav_path_index2phonelen, f)
        logger.info("Saved paths to cache files")

    # Only 'meta' cache type use
    def load_path2meta(self):
        logger.info("Loaded meta from cache files")
        with open(self.wav_paths_cache, "rb") as f:
            all_wav_paths = pickle.load(f)
        
        with open(self.wav_id2meta_cache, "rb") as f:
            all_wav_id2meta = pickle.load(f)
        
        # Select part of data according to ljspeech_ratio
        if self.ljspeech_ratio < 1.0:
            total_samples = len(all_wav_paths)
            num_samples = int(total_samples * self.ljspeech_ratio)
            selected_indices = random.sample(range(total_samples), num_samples)
            self.wav_paths = [all_wav_paths[i] for i in selected_indices]
        else:
            assert self.ljspeech_ratio == 1.0
            self.wav_paths = all_wav_paths
        
        # Build wav_id2meta for selected samples
        self.wav_id2meta = {}
        for path in tqdm(self.wav_paths, desc="Loading meta from paths"):
            wav_id = os.path.basename(path).replace(".wav", "")
            if wav_id in all_wav_id2meta:
                self.wav_id2meta[wav_id] = all_wav_id2meta[wav_id]
        
        # Build index mappings
        for path in tqdm(self.wav_paths, desc="Building index mappings"):
            wav_id = os.path.basename(path).replace(".wav", "")
            if wav_id in self.wav_id2meta:
                meta = self.wav_id2meta[wav_id]
                duration = meta["duration"]
                phone_count = meta["phone_count"]
                self.wav_path_index2duration.append(duration)
                self.wav_path_index2phonelen.append(phone_count)
                self.index2num_frames.append(duration * 50)

    def get_meta_from_wav_path(self, wav_path):
        """Get metadata from wav path"""
        # Extract wav_id from path (handle both relative and absolute paths)
        wav_id = os.path.basename(wav_path).replace(".wav", "")
        
        if self.cache_type == "meta":
            if wav_id in self.wav_id2meta:
                return self.wav_id2meta[wav_id]
        elif self.cache_type == "path":
            if wav_id in self.metadata:
                meta = self.metadata[wav_id].copy()
                # Get duration and phone_count from indices
                # Try to find the index in wav_paths
                rel_path = os.path.join("wavs", f"{wav_id}.wav")
                try:
                    idx = self.wav_paths.index(rel_path)
                    if idx < len(self.wav_path_index2duration):
                        meta["duration"] = self.wav_path_index2duration[idx]
                        meta["phone_count"] = self.wav_path_index2phonelen[idx]
                except ValueError:
                    # If not found, try to get from first matching entry
                    logger.warning(f"Could not find index for {wav_id}, using metadata only")
                return meta
        
        return None

    def __len__(self):
        return len(self.wav_paths)

    def get_num_frames(self, index):
        return self.wav_path_index2duration[index] * 50

    def __getitem__(self, idx):
        wav_path = self.wav_paths[idx]
        file_bytes = None
        try:
            full_wav_path = os.path.join(self.data_root, wav_path)
            file_bytes = full_wav_path
        except:
            logger.info("Get data from failed. Get another.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)

        meta = self.get_meta_from_wav_path(wav_path)
        if file_bytes is not None and meta is not None:
            buffer = file_bytes
            try:
                speech, sr = librosa.load(buffer, sr=self.cfg.preprocess.sample_rate)
                duration = len(speech) / sr
                
                # Check duration constraints
                if duration < self.duration_setting["min"]:
                    position = np.where(self.num_frame_indices == idx)[0][0]
                    random_index = np.random.choice(self.num_frame_indices[:position])
                    del position
                    return self.__getitem__(random_index)
                
                if (
                    len(speech)
                    > self.duration_setting["max"] * self.cfg.preprocess.sample_rate
                ):
                    position = np.where(self.num_frame_indices == idx)[0][0]
                    random_index = np.random.choice(self.num_frame_indices[:position])
                    del position
                    return self.__getitem__(random_index)
            except:
                logger.info("Failed to load file. Get another.")
                position = np.where(self.num_frame_indices == idx)[0][0]
                random_index = np.random.choice(self.num_frame_indices[:position])
                del position
                return self.__getitem__(random_index)

            single_feature = dict()

            # pad the speech to the multiple of hop_size
            speech = np.pad(
                speech,
                (
                    0,
                    self.cfg.preprocess.hop_size
                    - len(speech) % self.cfg.preprocess.hop_size,
                ),
                mode="constant",
            )

            # get speech mask
            speech_frames = len(speech) // self.cfg.preprocess.hop_size
            mask = np.ones(speech_frames)

            single_feature.update(
                {
                    "speech": speech,
                    "mask": mask,
                }
            )

            return single_feature

        else:
            logger.info("Failed to get file after retries.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            del position
            return self.__getitem__(random_index)

