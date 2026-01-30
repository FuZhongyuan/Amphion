# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numpy as np
import torch
import torchaudio
import os
import h5py
import json
from torch.nn.utils.rnn import pad_sequence

from models.base.libritts_dataset import LibriTTSDataset, WarningFilter


filter = WarningFilter()
logging.getLogger("phonemizer").addFilter(filter)
logging.getLogger("qcloud_cos.cos_client").addFilter(filter)
logging.getLogger("jieba").addFilter(filter)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaskgctLibriTTSDataset(LibriTTSDataset):
    def __init__(self, cfg, is_valid=False):
        super(MaskgctLibriTTSDataset, self).__init__(cfg=cfg, is_valid=is_valid)

        self.sample_rate = self.cfg.preprocess.sample_rate

        # Audio pretrained models' sample rates
        self.all_sample_rates = {self.sample_rate}
        if hasattr(self.cfg.model, "cond_sample_rate"):
            self.all_sample_rates.add(self.cfg.model.cond_sample_rate)

        self.load_phone = getattr(self.cfg.preprocess, "load_phone", False)
        self.load_wav_path = getattr(self.cfg.preprocess, "load_wav_path", False)
        self.load_semantic_features = getattr(
            self.cfg.preprocess, "load_semantic_features", False
        )
        self.load_chromagram = getattr(self.cfg.preprocess, "load_chromagram", False)
        self.load_mel_spectrogram = getattr(
            self.cfg.preprocess, "load_mel_spectrogram", False
        )
        
        # Audio loading switch: when disabled, skip loading raw audio to reduce disk I/O
        # Useful for trainers that only use cached features (e.g., semantic_codec_trainer)
        self.load_audio = getattr(self.cfg.preprocess, "load_audio", True)

        # Semantic feature caching configuration
        self.use_semantic_cache = getattr(self.cfg.preprocess, "use_semantic_cache", False)
        self.processed_dir = getattr(self.cfg.preprocess, "processed_dir", "")

        # Semantic code (discrete tokens) loading configuration
        self.load_semantic_code = getattr(self.cfg.preprocess, "load_semantic_code", False)
        self.semantic_code_dir = getattr(self.cfg.preprocess, "semantic_code_dir", "")

        # HDF5 cache index for semantic features
        self.hdf5_index = {}
        self.hdf5_files = {}  # Cache for opened HDF5 files
        if self.use_semantic_cache and self.processed_dir:
            self._load_hdf5_index()

        # HDF5 index for semantic codes (discrete tokens)
        self.semantic_code_index = {}
        self.semantic_code_files = {}  # Cache for opened HDF5 files
        if self.load_semantic_code and self.semantic_code_dir:
            self._load_semantic_code_index()

        if self.load_semantic_features:
            from transformers import SeamlessM4TFeatureExtractor

            self.semantic_model_processor = SeamlessM4TFeatureExtractor.from_pretrained(
                "facebook/w2v-bert-2.0"
            )
    
    def _load_hdf5_index(self):
        """Load HDF5 index for fast lookup."""
        index_path = os.path.join(self.processed_dir, "hdf5_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self.hdf5_index = data.get("sample_index", {})
                logger.info(f"Loaded HDF5 index with {len(self.hdf5_index)} samples")
            except Exception as e:
                logger.warning(f"Failed to load HDF5 index: {e}")

    def _load_semantic_code_index(self):
        """Load HDF5 index for semantic codes."""
        index_path = os.path.join(self.semantic_code_dir, "hdf5_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    data = json.load(f)
                    self.semantic_code_index = data.get("sample_index", {})
                logger.info(f"Loaded semantic code index with {len(self.semantic_code_index)} samples")
            except Exception as e:
                logger.warning(f"Failed to load semantic code index: {e}")
    
    def _get_hdf5_file(self, file_idx: int):
        """Get HDF5 file handle (with caching)."""
        if file_idx not in self.hdf5_files:
            hdf5_path = os.path.join(self.processed_dir, f"features_{file_idx:05d}.h5")
            if os.path.exists(hdf5_path):
                self.hdf5_files[file_idx] = h5py.File(hdf5_path, 'r')
                # print(f"{len(self.hdf5_files)} files open")
            else:
                return None
        return self.hdf5_files[file_idx]

    def _get_semantic_code_file(self, file_idx: int):
        """Get semantic code HDF5 file handle (with caching)."""
        if file_idx not in self.semantic_code_files:
            hdf5_path = os.path.join(self.semantic_code_dir, f"semantic_code_{file_idx:05d}.h5")
            if os.path.exists(hdf5_path):
                self.semantic_code_files[file_idx] = h5py.File(hdf5_path, 'r')
            else:
                return None
        return self.semantic_code_files[file_idx]

    def __del__(self):
        """Close all HDF5 files."""
        for f in self.hdf5_files.values():
            try:
                f.close()
            except:
                pass
        for f in self.semantic_code_files.values():
            try:
                f.close()
            except:
                pass

    def g2p(self, text, language):
        from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p

        if language in ["zh", "en"]:
            return chn_eng_g2p(text)
        else:
            return g2p(text, sentence=None, language=language)

    def load_cached_semantic_code(self, wav_path):
        """Load preprocessed semantic codes (discrete tokens) from HDF5 cache if available."""
        if not self.load_semantic_code or not self.semantic_code_dir:
            return None

        try:
            # Convert wav path to sample key
            # For LibriTTS: data_root/train-clean-100/123/456/123_456_789.wav
            # -> train-clean-100/123/456/123_456_789
            rel_path = os.path.relpath(wav_path, self.data_root)
            sample_key = os.path.splitext(rel_path)[0].replace(os.sep, '/')

            # Look up in semantic code index
            if sample_key not in self.semantic_code_index:
                return None

            index_info = self.semantic_code_index[sample_key]
            file_idx = index_info["file_idx"]
            group_name = index_info["group_name"]

            # Get HDF5 file
            hdf5_file = self._get_semantic_code_file(file_idx)
            if hdf5_file is None or group_name not in hdf5_file:
                return None

            group = hdf5_file[group_name]

            # Load semantic code if available
            if "semantic_code" in group:
                # Semantic codes are stored as int16, convert to int64 for PyTorch
                semantic_code = group["semantic_code"][:].astype(np.int64)
                return {"semantic_code": semantic_code}

            return None

        except Exception as e:
            logger.warning(f"Failed to load cached semantic code for {wav_path}: {e}")

        return None

    def load_cached_semantic_features(self, wav_path):
        """Load preprocessed semantic features from HDF5 cache if available."""
        # if not self.use_semantic_cache or not self.processed_dir:
        #     return None

        try:
            # Convert wav path to sample key
            # For LibriTTS: data_root/train-clean-100/123/456/123_456_789.wav
            # -> train-clean-100/123/456/123_456_789
            rel_path = os.path.relpath(wav_path, self.data_root)
            sample_key = os.path.splitext(rel_path)[0].replace(os.sep, '/')
            
            # Look up in HDF5 index
            if sample_key not in self.hdf5_index:
                return None
            
            index_info = self.hdf5_index[sample_key]
            file_idx = index_info["file_idx"]
            group_name = index_info["group_name"]
            
            # Get HDF5 file
            hdf5_file = self._get_hdf5_file(file_idx)
            if hdf5_file is None or group_name not in hdf5_file:
                return None
            
            group = hdf5_file[group_name]
            cached_data = {}
            
            # Load hidden states if available
            if self.load_semantic_features and "hidden_states" in group:
                # NOTE: cached features may be stored as float16; cast to float32 for training stability
                cached_data["semantic_hidden_states"] = group["hidden_states"][:].astype(
                    np.float32, copy=False
                )
            
            # Load mel spectrogram if available and requested
            if self.load_mel_spectrogram and "mel_spectrogram" in group:
                # NOTE: cached features may be stored as float16; cast to float32 for training stability
                cached_data["mel_spectrogram"] = group["mel_spectrogram"][:].astype(
                    np.float32, copy=False
                )
            
            return cached_data if cached_data else None
            
        except Exception as e:
            logger.warning(f"Failed to load cached features for {wav_path}: {e}")

        return None

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
            single_feature = dict()
            
            # Load audio only if enabled (to reduce disk I/O when using cached features)
            speech = None
            if self.load_audio:
                # Try loading audio with retries
                for attempt in range(self.max_retries):
                    try:
                        speech, _ = self._load_audio_with_torchaudio(
                            full_wav_path, target_sr=self.sample_rate
                        )
                        if len(speech) > self.duration_setting["max"] * self.sample_rate:
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

            # Process audio if loaded
            if self.load_audio and speech is not None:
                # pad the speech to the multiple of vocos hop_size for acoustic codec alignment
                vocos_hop_size = getattr(self.cfg.preprocess, "hop_size", 320)
                speech = np.pad(
                    speech,
                    (
                        0,
                        vocos_hop_size - len(speech) % vocos_hop_size,
                    ),
                    mode="constant",
                )

                # For all the sample rates
                for tgt_sr in self.all_sample_rates:
                    if tgt_sr != self.sample_rate:
                        assert tgt_sr < self.sample_rate
                        speech_tensor = torch.from_numpy(speech).float()
                        tgt_speech = torchaudio.functional.resample(
                            speech_tensor, self.sample_rate, tgt_sr
                        ).numpy()
                    else:
                        tgt_speech = speech
                    single_feature.update(
                        {
                            f"wav_{tgt_sr}": tgt_speech,
                            f"wav_{tgt_sr}_len": len(tgt_speech),
                        }
                    )

                # [Note] Mask is (n_frames,) but not (T,)
                speech_frames = len(speech) // self.cfg.preprocess.hop_size
                mask = np.ones(speech_frames, dtype=np.float32)

                single_feature.update(
                    {
                        "wav": speech,
                        "wav_len": len(speech),
                        "mask": mask,
                    }
                )

            ## Load Semantic Model Input Features ##
            if self.load_semantic_features or self.load_mel_spectrogram:
                # Try to load from cache first
                cached_features = self.load_cached_semantic_features(full_wav_path)
                if cached_features is not None:
                    single_feature.update(cached_features)
                else:
                    # Fallback to real-time extraction (requires audio to be loaded)
                    if not self.load_audio:
                        logger.warning(
                            f"Cache miss for {full_wav_path} but load_audio=False. "
                            f"Cannot extract features in real-time. Skipping sample."
                        )
                        position = np.where(self.num_frame_indices == idx)[0][0]
                        random_index = np.random.choice(self.num_frame_indices[:position])
                        return self.__getitem__(random_index)
                    
                    logger.debug(f"Cache miss for {full_wav_path}, extracting features in real-time")

                    # Extract semantic features if needed
                    if self.load_semantic_features:
                        speech_16k = single_feature["wav_16000"]
                        inputs = self.semantic_model_processor(speech_16k, sampling_rate=16000)
                        input_features = inputs["input_features"][0]
                        attention_mask = inputs["attention_mask"][0]

                        single_feature.update(
                            {
                                "semantic_model_input_features": input_features,
                                "semantic_model_attention_mask": attention_mask,
                            }
                        )

                    # Note: Mel spectrogram extraction will be handled by trainer if not cached

            ## Load Semantic Code (discrete tokens) ##
            if self.load_semantic_code:
                cached_semantic_code = self.load_cached_semantic_code(full_wav_path)
                if cached_semantic_code is not None:
                    single_feature.update(cached_semantic_code)
                    # Also create semantic_mask based on semantic_code length
                    semantic_code_len = len(cached_semantic_code["semantic_code"])
                    single_feature["semantic_mask"] = np.ones(semantic_code_len, dtype=np.float32)
                else:
                    logger.warning(f"Semantic code not found for {full_wav_path}")

            if self.load_wav_path:
                single_feature.update({"wav_path": full_wav_path})

            if not self.load_phone:
                return single_feature

            ## Load phone using G2P ##
            try:
                text = meta.get("normalized_text", meta.get("text", ""))
                language = meta.get("language", "en")
                
                phone_id = (
                    self.g2p(text, language)[1]
                    if self.cache_type == "path"
                    else meta.get("phone_id", [])
                )
                
                if len(phone_id) == 0:
                    # Fallback: compute phone_id if not in meta
                    phone_id = self.g2p(text, language)[1]
                
                if len(phone_id) > 512:
                    raise Exception("too long phone seq")
            except Exception as e:
                print(e)
                print(f"Loading phone failed for {full_wav_path}")
                print(text, language)
                
                position = np.where(self.num_frame_indices == idx)[0][0]
                random_index = np.random.choice(self.num_frame_indices[:position])
                del position
                return self.__getitem__(random_index)
            
            # Check phone_id length against speech_frames (only if audio was loaded)
            if self.load_audio and speech is not None:
                speech_frames = len(speech) // self.cfg.preprocess.hop_size
                if len(phone_id) >= speech_frames:
                    position = np.where(self.num_frame_indices == idx)[0][0]
                    random_index = np.random.choice(self.num_frame_indices[:position])
                    del position
                    return self.__getitem__(random_index)

            phone_id = torch.tensor(np.array(phone_id), dtype=torch.long)
            phone_mask = np.ones(len(phone_id), dtype=np.float32)

            single_feature.update({"phone_id": phone_id, "phone_mask": phone_mask})
            return single_feature

        else:
            logger.info("Failed to get metadata.")
            position = np.where(self.num_frame_indices == idx)[0][0]
            random_index = np.random.choice(self.num_frame_indices[:position])
            return self.__getitem__(random_index)


class MaskgctLibriTTSCollator:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        """
        MaskgctLibriTTSDataset.__getitem__:
            wav: (T,)
            wav_len: int
            mask: (n_frames,)

            wav_{sr}: (T,)
            wav_{sr}_len: int

            phone_id: (n_phones,)
            phone_mask: (n_phones,)

        Returns:
            wav: (B, T), torch.float32
            wav_len: (B), torch.long
            mask: (B, n_frames), torch.float32

            wav_{sr}: (B, T)
            wav_{sr}_len: (B), torch.long

            phone_id: (B, n_phones), torch.long
            phone_mask: (B, n_phones), torch.float32
        """

        packed_batch_features = dict()

        for key in batch[0].keys():
            if "_len" in key:
                packed_batch_features[key] = torch.LongTensor([b[key] for b in batch])
            elif key == "phone_id":
                packed_batch_features[key] = pad_sequence(
                    [utt[key].long() for utt in batch],
                    batch_first=True,
                    padding_value=1023,  # phone vocab size is 1024
                )
            elif key == "semantic_code":
                # semantic_code is discrete tokens, pad with 0
                packed_batch_features[key] = pad_sequence(
                    [torch.as_tensor(b[key]).long() for b in batch],
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "wav_path":
                packed_batch_features[key] = [b[key] for b in batch]
            elif key == "mel_spectrogram":
                # mel_spectrogram has shape (1, T, D) or (T, D), remove batch dim before padding
                packed_batch_features[key] = pad_sequence(
                    [torch.as_tensor(b[key]).squeeze(0) if len(b[key].shape) == 3 else torch.as_tensor(b[key]) for b in batch],
                    batch_first=True,
                    padding_value=0.0,
                )
            else:
                packed_batch_features[key] = pad_sequence(
                    [torch.as_tensor(b[key]) for b in batch],
                    batch_first=True,
                    padding_value=0.0,
                )
        return packed_batch_features

