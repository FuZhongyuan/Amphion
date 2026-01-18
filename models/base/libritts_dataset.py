# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchaudio
import json
import numpy as np
import logging
import pickle
import os
import glob
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


class LibriTTSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cache_type="meta",
        cfg=None,
        is_valid=False,
    ):  # 'path' or 'meta'

        assert cfg is not None

        self.cache_type = cache_type
        self.cfg = cfg
        self.is_valid = is_valid

        self.dataset_ratio_dict = self.cfg.dataset
        # JsonHParams doesn't have .get() method, use getattr instead
        self.libritts_ratio = getattr(self.dataset_ratio_dict, "libritts", 1.0)

        self.wav_paths = []
        self.metadata = {}  # wav_id -> {text, normalized_text, language}
        
        # Get data root path from cfg or use default
        if hasattr(self.cfg.preprocess, "libritts_data_root"):
            self.data_root = self.cfg.preprocess.libritts_data_root
        else:
            self.data_root = "[Please fill out your libritts data root path]"

        self.language_list = ["en"]  # LibriTTS is English only
        self.wav_path_index2duration = []
        self.wav_path_index2phonelen = []
        self.index2num_frames = []

        self.wav_id2meta = {}

        # Get cache path from cfg or use default
        if hasattr(self.cfg.preprocess, "libritts_cache_path"):
            self.cache_folder = self.cfg.preprocess.libritts_cache_path
        else:
            self.cache_folder = "[Please fill out your libritts cache path]"
        
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

        # Error files tracking
        self.error_files_path = os.path.join(self.cache_folder, "error_files.txt")
        self.error_files = set()
        self._load_error_files()
        self.max_retries = getattr(self.cfg.preprocess, "max_retries", 3)

        # Get validation split ratio from config (default 0.05 = 5%)
        self.valid_split_ratio = getattr(self.cfg.preprocess, "valid_split_ratio", 0.05)

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

    def _load_error_files(self):
        """Load error files list from disk."""
        if os.path.exists(self.error_files_path):
            with open(self.error_files_path, "r") as f:
                self.error_files = set(line.strip() for line in f if line.strip())
            logger.info(f"Loaded {len(self.error_files)} error files to skip")
            print(f"error files: {self.error_files_path}")

    def _save_error_file(self, wav_path):
        """Save an error file path to the error files list."""
        self.error_files.add(wav_path)
        with open(self.error_files_path, "a") as f:
            f.write(f"{wav_path}\n")
        logger.warning(f"Added {wav_path} to error files list")

    def _load_audio_with_torchaudio(self, file_path, target_sr=None):
        """Load audio using torchaudio with resampling support."""
        waveform, sr = torchaudio.load(file_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if target_sr is not None and sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        # Convert to numpy array (mono)
        audio = waveform.squeeze(0).numpy()
        return audio, sr

    def load_metadata_from_files(self):
        """Load metadata from .normalized.txt files"""
        logger.info("Loading LibriTTS metadata from files...")
        
        # Find all normalized text files
        # LibriTTS structure: train/{speaker_id}/{book_id}/*.normalized.txt
        subsets = ["train-clean-100", "train-clean-360", "train-other-500",
                   "dev-clean", "dev-other", "test-clean", "test-other"]
        
        self.metadata = {}
        
        for subset in tqdm(subsets, desc="Loading subsets"):
            subset_dir = os.path.join(self.data_root, subset)
            if not os.path.exists(subset_dir):
                logger.warning(f"Subset directory not found: {subset_dir}")
                continue
            
            # Find all normalized.txt files
            pattern = os.path.join(subset_dir, "**", "*.normalized.txt")
            txt_files = glob.glob(pattern, recursive=True)
            
            for txt_file in tqdm(txt_files, desc=f"Processing {subset}", leave=False):
                # Get corresponding wav file
                wav_file = txt_file.replace(".normalized.txt", ".wav")
                if not os.path.exists(wav_file):
                    continue
                
                # Extract wav_id from filename
                # Format: {speaker_id}_{book_id}_{chapter_id}_{utterance_id}
                wav_id = os.path.basename(wav_file).replace(".wav", "")
                
                # Read normalized text
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        normalized_text = f.read().strip()
                    
                    # Read original text if available
                    original_file = txt_file.replace(".normalized.txt", ".original.txt")
                    original_text = normalized_text
                    if os.path.exists(original_file):
                        with open(original_file, "r", encoding="utf-8") as f:
                            original_text = f.read().strip()
                    
                    self.metadata[wav_id] = {
                        "text": original_text,
                        "normalized_text": normalized_text,
                        "language": "en",
                        "subset": subset,
                        "wav_path": os.path.relpath(wav_file, self.data_root)
                    }
                except Exception as e:
                    logger.warning(f"Failed to read {txt_file}: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.metadata)} metadata entries from LibriTTS files")

    def build_cache(self):
        """Build cache files for path cache type"""
        logger.info("Building cache for LibriTTS dataset...")
        
        # Load metadata
        self.load_metadata_from_files()
        
        all_wav_paths = []
        all_durations = []
        all_phone_counts = []
        
        # Process each wav file
        for wav_id, meta in tqdm(sorted(self.metadata.items()), desc="Building cache"):
            wav_file = os.path.join(self.data_root, meta["wav_path"])
            if not os.path.exists(wav_file):
                logger.warning(f"Wav file not found: {wav_file}")
                continue
            
            # Get relative path
            rel_path = meta["wav_path"]
            all_wav_paths.append(rel_path)
            
            # Load audio to get duration
            try:
                audio, sr = self._load_audio_with_torchaudio(wav_file)
                duration = len(audio) / sr
                
                # Get phone count using G2P
                text = meta["normalized_text"]
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
        logger.info("Building meta cache for LibriTTS dataset...")
        
        # Load metadata
        self.load_metadata_from_files()
        
        all_wav_paths = []
        wav_id2meta = {}
        
        # Process each wav file
        for wav_id, meta in tqdm(sorted(self.metadata.items()), desc="Building meta cache"):
            wav_file = os.path.join(self.data_root, meta["wav_path"])
            if not os.path.exists(wav_file):
                logger.warning(f"Wav file not found: {wav_file}")
                continue
            
            # Get relative path
            rel_path = meta["wav_path"]
            all_wav_paths.append(rel_path)
            
            # Load audio to get duration
            try:
                audio, sr = self._load_audio_with_torchaudio(wav_file)
                duration = len(audio) / sr
                
                # Get phone count using G2P
                text = meta["normalized_text"]
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
                    "text": meta["text"],
                    "normalized_text": meta["normalized_text"],
                    "language": "en",
                    "duration": duration,
                    "phone_count": phone_count,
                    "phone_id": phone_ids,
                    "subset": meta.get("subset", "train")
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

        # Load duration and phone count data
        if self.cache_type == "path":
            with open(self.duration_cache, "rb") as f:
                all_durations = pickle.load(f)
            with open(self.phone_count_cache, "rb") as f:
                all_phone_counts = pickle.load(f)

        # First, apply libritts_ratio if needed
        if self.libritts_ratio < 1.0:
            total_samples = len(all_wav_paths)
            num_samples = int(total_samples * self.libritts_ratio)
            # Use fixed seed for reproducibility
            rng = random.Random(self.cfg.train.random_seed)
            selected_indices = rng.sample(range(total_samples), num_samples)
            selected_indices.sort()  # Keep order for consistency

            all_wav_paths = [all_wav_paths[i] for i in selected_indices]
            # Update metadata
            all_metadata = {k: v for i, (k, v) in enumerate(all_metadata.items()) if i in selected_indices}
            
            if self.cache_type == "path":
                all_durations = [all_durations[i] for i in selected_indices]
                all_phone_counts = [all_phone_counts[i] for i in selected_indices]

        # Then, split into train/valid sets
        total_samples = len(all_wav_paths)
        num_valid = int(total_samples * self.valid_split_ratio)
        
        # Use fixed seed for reproducibility
        rng = random.Random(self.cfg.train.random_seed)
        all_indices = list(range(total_samples))
        rng.shuffle(all_indices)
        
        if self.is_valid:
            # Use first num_valid samples for validation
            selected_indices = all_indices[:num_valid]
            split_name = "validation"
        else:
            # Use remaining samples for training
            selected_indices = all_indices[num_valid:]
            split_name = "training"
        
        selected_indices.sort()  # Keep order for consistency
        
        # Select data based on split
        self.wav_paths = [all_wav_paths[i] for i in selected_indices]
        
        if self.cache_type == "path":
            self.wav_path_index2duration = [all_durations[i] for i in selected_indices]
            self.wav_path_index2phonelen = [all_phone_counts[i] for i in selected_indices]
            
            # Calculate the number of frames
            self.index2num_frames = []
            for duration, phone_count in zip(
                self.wav_path_index2duration, self.wav_path_index2phonelen
            ):
                self.index2num_frames.append(duration * 50 + phone_count)
        
        # Filter metadata to only include selected samples
        if self.metadata:
            selected_wav_ids = set()
            for path in self.wav_paths:
                wav_id = os.path.basename(path).replace(".wav", "")
                selected_wav_ids.add(wav_id)
            self.metadata = {k: v for k, v in all_metadata.items() if k in selected_wav_ids}
        else:
            self.metadata = all_metadata

        logger.info(f"LibriTTS {split_name} set loaded successfully, ratio: {self.libritts_ratio:.2f}")
        logger.info(f"Number of {split_name} samples: {len(self.wav_paths)}")

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
        
        # Select part of data according to libritts_ratio
        if self.libritts_ratio < 1.0:
            total_samples = len(all_wav_paths)
            num_samples = int(total_samples * self.libritts_ratio)
            selected_indices = random.sample(range(total_samples), num_samples)
            self.wav_paths = [all_wav_paths[i] for i in selected_indices]
        else:
            assert self.libritts_ratio == 1.0
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
                try:
                    idx = self.wav_paths.index(wav_path)
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
        full_wav_path = os.path.join(self.data_root, wav_path)

        # Skip known error files
        if wav_path in self.error_files or full_wav_path in self.error_files:
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            return self.__getitem__(random_index)

        meta = self.get_meta_from_wav_path(wav_path)
        if meta is not None:
            # Try loading audio with retries
            speech = None
            for attempt in range(self.max_retries):
                try:
                    speech, sr = self._load_audio_with_torchaudio(
                        full_wav_path, target_sr=self.cfg.preprocess.sample_rate
                    )
                    duration = len(speech) / sr

                    # Check duration constraints
                    if duration < self.duration_setting["min"]:
                        position = np.where(self.num_frame_indices == idx)[0][0]
                        random_index = np.random.choice(self.num_frame_indices[:position])
                        return self.__getitem__(random_index)

                    if len(speech) > self.duration_setting["max"] * self.cfg.preprocess.sample_rate:
                        position = np.where(self.num_frame_indices == idx)[0][0]
                        random_index = np.random.choice(self.num_frame_indices[:position])
                        return self.__getitem__(random_index)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        logger.warning(f"Failed to load {full_wav_path} after {self.max_retries} attempts: {e}")
                        self._save_error_file(wav_path)
                        position = np.where(self.num_frame_indices == idx)[0][0]
                        random_index = np.random.choice(self.num_frame_indices[:position])
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
            # Keep mask in float32 to avoid mixing float64 in training
            mask = np.ones(speech_frames, dtype=np.float32)

            single_feature.update(
                {
                    "speech": speech,
                    "mask": mask,
                }
            )

            return single_feature

        else:
            logger.info("Failed to get metadata.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            return self.__getitem__(random_index)

