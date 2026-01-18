#!/usr/bin/env python3
"""
Optimized preprocessing pipeline with async prefetching and pipelining.
"""

import os
import json
import numpy as np
import torch
import torchaudio
import h5py
from pathlib import Path
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from utils.mel import extract_mel_features
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import time


class AudioLoader:
    """Handles audio file loading with thread pool prefetching using torchaudio."""

    def __init__(self, num_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    @staticmethod
    def load_audio(file_path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Load audio file using torchaudio and resample to target sample rate."""
        try:
            waveform, sr = torchaudio.load(file_path)
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            # Resample if needed
            if sr != sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            # Convert to numpy array (mono)
            audio = waveform.squeeze(0).numpy()
            return audio
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None
    
    def load_audio_async(self, file_path: str, sample_rate: int = 16000):
        """Submit audio loading task to thread pool."""
        return self.executor.submit(self.load_audio, file_path, sample_rate)
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class SemanticFeatureExtractor:
    """Extracts semantic features using w2v-bert-2.0 model with GPU streams."""
    
    def __init__(self, model_name: str, output_layer: int = 17, device: str = "cuda"):
        print("Loading semantic model...")
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2BertModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_layer = output_layer
        
        # Create CUDA stream for async GPU operations
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
        
        print(f"Semantic model loaded on device: {self.device}")
    
    def extract(self, audio: np.ndarray, sampling_rate: int = 16000, 
                use_stream: bool = True) -> Optional[Dict[str, np.ndarray]]:
        """Extract semantic features with optional CUDA stream."""
        try:
            inputs = self.processor(audio, sampling_rate=sampling_rate)
            input_features = inputs["input_features"][0]
            attention_mask = inputs["attention_mask"][0]
            
            # Use CUDA stream for async execution
            if use_stream and self.stream is not None:
                with torch.cuda.stream(self.stream):
                    with torch.no_grad():
                        outputs = self.model(
                            input_features=torch.tensor(input_features).unsqueeze(0).to(self.device, non_blocking=True),
                            attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(self.device, non_blocking=True),
                            output_hidden_states=True
                        )
                        hidden_states = outputs.hidden_states[self.output_layer][0]
                
                # Synchronize stream before returning
                self.stream.synchronize()
                hidden_states = hidden_states.cpu().numpy()
            else:
                with torch.no_grad():
                    outputs = self.model(
                        input_features=torch.tensor(input_features).unsqueeze(0).to(self.device),
                        attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(self.device),
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[self.output_layer][0].cpu().numpy()
            
            return {"hidden_states": hidden_states}
        except Exception as e:
            print(f"Failed to extract semantic features: {e}")
            return None


class MelSpectrogramExtractor:
    """Extracts mel-spectrogram features."""
    
    def __init__(self, mel_config: Dict[str, Any]):
        self.cfg = self._create_config_object(mel_config)
    
    @staticmethod
    def _create_config_object(config_dict: Dict[str, Any]) -> Any:
        """Convert config dictionary to object with attributes."""
        class ConfigObj:
            pass
        cfg = ConfigObj()
        for key, value in config_dict.items():
            setattr(cfg, key, value)
        return cfg
    
    def extract(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Extract mel-spectrogram from audio."""
        try:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            mel = extract_mel_features(audio_tensor, self.cfg)
            
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.permute(0, 2, 1)
            
            return mel.numpy()
        except Exception as e:
            print(f"Failed to extract mel spectrogram: {e}")
            return None


class HDF5Writer:
    """Manages HDF5 file writing with multi-file support."""
    
    def __init__(self, base_dir: Path, samples_per_file: int = 20000, use_float16: bool = True):
        self.base_dir = Path(base_dir)
        self.samples_per_file = samples_per_file
        self.use_float16 = use_float16
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping from sample key to HDF5 file location
        self.sample_index = {}  # {sample_key: (file_idx, dataset_name)}
        self.current_file_idx = 0
        self.current_file_count = 0
        self.current_file = None
        self.lock = threading.Lock()
        
        # Index file path
        self.index_path = self.base_dir / "hdf5_index.json"
        self.load_index()
    
    def load_index(self):
        """Load existing HDF5 index if available."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)
                self.sample_index = data.get("sample_index", {})
                self.current_file_idx = data.get("current_file_idx", 0)
                self.current_file_count = data.get("current_file_count", 0)
    
    def save_index(self):
        """Save HDF5 index to disk."""
        with open(self.index_path, 'w') as f:
            json.dump({
                "sample_index": self.sample_index,
                "current_file_idx": self.current_file_idx,
                "current_file_count": self.current_file_count,
                "samples_per_file": self.samples_per_file,
                "use_float16": self.use_float16
            }, f, indent=2)
    
    def get_hdf5_path(self, file_idx: int) -> Path:
        """Get HDF5 file path for a given file index."""
        return self.base_dir / f"features_{file_idx:05d}.h5"
    
    def write_sample(self, sample_key: str, data: Dict[str, np.ndarray]) -> bool:
        """Write a single sample to HDF5 file."""
        with self.lock:
            try:
                # Check if we need to create a new file
                if self.current_file_count >= self.samples_per_file or self.current_file is None:
                    if self.current_file is not None:
                        self.current_file.close()
                    
                    # Only increment if we're not at the initial state
                    if self.current_file is not None or self.current_file_count > 0:
                        self.current_file_idx += 1
                        self.current_file_count = 0
                    elif self.current_file_idx == 0:
                        # First file, start from 1
                        self.current_file_idx = 1
                        self.current_file_count = 0
                    
                    hdf5_path = self.get_hdf5_path(self.current_file_idx)
                    self.current_file = h5py.File(hdf5_path, 'a')
                
                # Create group for this sample
                if sample_key in self.current_file:
                    del self.current_file[sample_key]
                
                sample_group = self.current_file.create_group(sample_key)
                
                # Write each dataset
                for key, value in data.items():
                    # Convert to float16 if requested and data is floating point
                    if self.use_float16 and np.issubdtype(value.dtype, np.floating):
                        value = value.astype(np.float16)
                    
                    sample_group.create_dataset(
                        key,
                        data=value,
                        compression="gzip",
                        compression_opts=4
                    )
                
                # Update index
                self.sample_index[sample_key] = {
                    "file_idx": self.current_file_idx,
                    "group_name": sample_key
                }
                
                self.current_file_count += 1
                
                # Periodically save index
                if self.current_file_count % 100 == 0:
                    self.save_index()
                
                return True
            except Exception as e:
                print(f"Failed to write sample {sample_key}: {e}")
                return False
    
    def close(self):
        """Close current HDF5 file and save index."""
        with self.lock:
            if self.current_file is not None:
                self.current_file.close()
                self.current_file = None
            self.save_index()


class AsyncFileSaver:
    """Handles asynchronous file saving with thread pool for HDF5."""
    
    def __init__(self, num_workers: int = 4, hdf5_writer: HDF5Writer = None):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.hdf5_writer = hdf5_writer
    
    def save_file(self, sample_key: str, data: Dict[str, np.ndarray]) -> Tuple[bool, str]:
        """Save data to HDF5 file."""
        try:
            success = self.hdf5_writer.write_sample(sample_key, data)
            if success:
                return True, f"Saved {sample_key}"
            else:
                return False, f"Failed to save {sample_key}"
        except Exception as e:
            return False, f"Failed to save {sample_key}: {e}"
    
    def save_async(self, sample_key: str, data: Dict[str, np.ndarray]):
        """Submit save task to thread pool."""
        return self.executor.submit(self.save_file, sample_key, data)
    
    def shutdown(self):
        """Shutdown the executor and close HDF5 writer."""
        self.executor.shutdown(wait=True)
        if self.hdf5_writer is not None:
            self.hdf5_writer.close()


class PipelineQueue:
    """Multi-stage pipeline queue for prefetching and processing."""
    
    def __init__(self, max_size: int = 10):
        self.queue = Queue(maxsize=max_size)
        self.max_size = max_size
    
    def put(self, item):
        """Put item in queue."""
        self.queue.put(item)
    
    def get(self):
        """Get item from queue."""
        return self.queue.get()
    
    def empty(self):
        """Check if queue is empty."""
        return self.queue.empty()
    
    def qsize(self):
        """Get queue size."""
        return self.queue.qsize()


class DatasetPreprocessor(ABC):
    """Abstract base class for dataset-specific preprocessing with async pipeline."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_config = config["datasets"][dataset_name]
        
        self.data_root = Path(self.dataset_config["data_root"])
        self.processed_dir = Path(self.dataset_config["processed_dir"])
        self.overwrite_existing = config["preprocessing"]["overwrite_existing"]
        self.extract_mel = config["preprocessing"].get("extract_mel", False)
        
        # HDF5 configuration
        self.use_float16 = config["preprocessing"].get("use_float16", True)
        self.samples_per_hdf5 = config["preprocessing"].get("samples_per_hdf5", 20000)
        
        # Pipeline configuration
        self.num_audio_workers = config["preprocessing"].get("num_audio_workers", 8)
        self.num_save_workers = config["preprocessing"].get("num_save_workers", 4)
        self.prefetch_size = config["preprocessing"].get("prefetch_size", 20)
        
        # Retry configuration
        self.max_retries = config["preprocessing"].get("max_retries", 3)
        self.retry_delay = config["preprocessing"].get("retry_delay", 1.0)
        
        # Error tracking
        self.error_log = []  # List of (file_path, error_message, retry_count)
        
        # Initialize HDF5 writer
        self.hdf5_writer = HDF5Writer(
            base_dir=self.processed_dir,
            samples_per_file=self.samples_per_hdf5,
            use_float16=self.use_float16
        )
        
        # Initialize components
        self.audio_loader = AudioLoader(num_workers=self.num_audio_workers)
        self.file_saver = AsyncFileSaver(
            num_workers=self.num_save_workers,
            hdf5_writer=self.hdf5_writer
        )
        
        self.semantic_extractor = SemanticFeatureExtractor(
            model_name=config["semantic_model"]["model_name"],
            output_layer=config["semantic_model"].get("output_layer", 17),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.mel_extractor = None
        if self.extract_mel:
            mel_config = config["preprocessing"]["mel_config"]
            self.mel_extractor = MelSpectrogramExtractor(mel_config)
    
    @abstractmethod
    def find_audio_files(self) -> List[Path]:
        """Find all audio files in the dataset."""
        pass
    
    @abstractmethod
    def get_sample_key(self, wav_file: Path) -> str:
        """Get unique sample key for HDF5 storage."""
        pass
    
    def process_single_file(self, wav_path: Path, audio: np.ndarray, 
                           sample_key: str) -> Tuple[bool, str, Dict[str, np.ndarray]]:
        """Process a single audio file (GPU processing stage)."""
        try:
            # Extract semantic features
            semantic_features = self.semantic_extractor.extract(audio, sampling_rate=16000)
            if semantic_features is None:
                raise RuntimeError(f"Failed to extract semantic features for {wav_path}")
            
            # Prepare data to save
            save_data = {
                "hidden_states": semantic_features["hidden_states"]
            }
            
            # Extract mel spectrogram if requested
            if self.extract_mel and self.mel_extractor is not None:
                mel_spectrogram = self.mel_extractor.extract(audio)
                if mel_spectrogram is None:
                    raise RuntimeError(f"Failed to extract mel spectrogram for {wav_path}")
                save_data["mel_spectrogram"] = mel_spectrogram
            
            return True, f"Processed {wav_path}", save_data
        except Exception as e:
            return False, f"Exception during processing {wav_path}: {str(e)}", None
    
    def process_file_with_retry(self, wav_file: Path, sample_key: str) -> Tuple[bool, str, Dict[str, np.ndarray]]:
        """Process a file with retry mechanism."""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Load audio
                audio = AudioLoader.load_audio(str(wav_file), sample_rate=16000)
                if audio is None:
                    last_error = "Failed to load audio"
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        break
                
                # Process the file
                success, msg, save_data = self.process_single_file(wav_file, audio, sample_key)
                
                if success:
                    return True, msg, save_data
                else:
                    last_error = msg
                    if attempt < self.max_retries - 1:
                        print(f"\nRetry {attempt + 1}/{self.max_retries} for {wav_file}: {msg}")
                        time.sleep(self.retry_delay)
                    
            except Exception as e:
                last_error = f"Exception: {str(e)}"
                if attempt < self.max_retries - 1:
                    print(f"\nRetry {attempt + 1}/{self.max_retries} for {wav_file}: {last_error}")
                    time.sleep(self.retry_delay)
        
        # All retries failed
        error_info = f"Failed after {self.max_retries} attempts: {last_error}"
        self.error_log.append({
            "file_path": str(wav_file),
            "sample_key": sample_key,
            "error_message": last_error,
            "retry_count": self.max_retries
        })
        
        return False, error_info, None
    
    def save_error_log(self):
        """Save error log to file."""
        if len(self.error_log) > 0:
            error_log_path = self.processed_dir / "error_log.json"
            with open(error_log_path, 'w') as f:
                json.dump({
                    "total_errors": len(self.error_log),
                    "max_retries": self.max_retries,
                    "errors": self.error_log
                }, f, indent=2)
            print(f"\nError log saved to {error_log_path}")
            print(f"Total failed files: {len(self.error_log)}")
            
            # Also save a simple text file with failed file paths
            error_files_path = self.processed_dir / "error_files.txt"
            with open(error_files_path, 'w') as f:
                for error in self.error_log:
                    f.write(f"{error['file_path']}\n")
            print(f"Failed file list saved to {error_files_path}")
    
    def run(self) -> Dict[str, Any]:
        """Run the preprocessing pipeline with async prefetching."""
        print(f"Processing {self.dataset_name.upper()} dataset...")
        print(f"Data root: {self.data_root}")
        print(f"Processed dir: {self.processed_dir}")
        print(f"Pipeline config: {self.num_audio_workers} audio workers, "
              f"{self.num_save_workers} save workers, prefetch={self.prefetch_size}")
        print(f"HDF5 config: use_float16={self.use_float16}, samples_per_hdf5={self.samples_per_hdf5}")
        print("Using GPU acceleration with async pipeline and HDF5 storage")
        
        if self.extract_mel:
            print("Also extracting and caching mel spectrograms")
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        print("Finding audio files...")
        wav_files = self.find_audio_files()
        print(f"Found {len(wav_files)} audio files")
        
        if len(wav_files) == 0:
            raise ValueError(f"No audio files found in {self.data_root}")
        
        # Filter files that need processing
        files_to_process = []
        for wav_file in wav_files:
            sample_key = self.get_sample_key(wav_file)
            # Check if already processed in HDF5
            if sample_key not in self.hdf5_writer.sample_index or self.overwrite_existing:
                files_to_process.append((wav_file, sample_key))
        
        print(f"Files to process: {len(files_to_process)}")
        
        success_count = 0
        error_count = 0
        save_futures = []
        
        # Start prefetching audio files
        audio_futures = {}
        prefetch_idx = 0
        
        with tqdm(total=len(files_to_process), desc="Processing") as pbar:
            for i, (wav_file, sample_key) in enumerate(files_to_process):
                # Prefetch next batch of audio files
                while prefetch_idx < len(files_to_process) and \
                      prefetch_idx < i + self.prefetch_size and \
                      prefetch_idx not in audio_futures:
                    future_wav, _ = files_to_process[prefetch_idx]
                    audio_futures[prefetch_idx] = self.audio_loader.load_audio_async(
                        str(future_wav), sample_rate=16000
                    )
                    prefetch_idx += 1
                
                # Get prefetched audio
                if i in audio_futures:
                    audio_future = audio_futures.pop(i)
                    audio = audio_future.result()
                else:
                    # Fallback: load synchronously if not prefetched
                    audio = AudioLoader.load_audio(str(wav_file), sample_rate=16000)
                
                if audio is None:
                    # Retry loading and processing with retry mechanism
                    success, msg, save_data = self.process_file_with_retry(wav_file, sample_key)
                    if not success:
                        error_count += 1
                        if error_count <= 10:
                            print(f"\nError: {msg}")
                        pbar.update(1)
                        continue
                else:
                    # Process on GPU
                    success, msg, save_data = self.process_single_file(
                        wav_file, audio, sample_key
                    )
                    
                    # If processing failed, retry with full retry mechanism
                    if not success:
                        success, msg, save_data = self.process_file_with_retry(wav_file, sample_key)
                        if not success:
                            error_count += 1
                            if error_count <= 10:
                                print(f"\nError: {msg}")
                            pbar.update(1)
                            continue
                
                # Async save to HDF5
                save_future = self.file_saver.save_async(sample_key, save_data)
                save_futures.append(save_future)
                success_count += 1
                
                # Check completed saves periodically
                if len(save_futures) > self.num_save_workers * 2:
                    completed = [f for f in save_futures if f.done()]
                    for f in completed:
                        save_futures.remove(f)
                        try:
                            f.result()  # Check for exceptions
                        except Exception as e:
                            print(f"\nSave error: {e}")
                
                pbar.update(1)
                pbar.set_postfix({
                    "success": success_count, 
                    "errors": error_count,
                    "prefetch": len(audio_futures),
                    "save_queue": len(save_futures)
                })
        
        # Wait for all saves to complete
        print("\nWaiting for file saves to complete...")
        for future in tqdm(save_futures, desc="Finalizing saves"):
            try:
                future.result()
            except Exception as e:
                print(f"\nSave error: {e}")
        
        # Cleanup
        self.audio_loader.shutdown()
        self.file_saver.shutdown()
        
        # Save error log if there are any errors
        if error_count > 0:
            self.save_error_log()
        
        print("\nPreprocessing completed!")
        print(f"Successfully processed: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Total HDF5 files created: {self.hdf5_writer.current_file_idx}")
        
        # Prepare and save metadata
        metadata = {
            "dataset": self.dataset_name,
            "total_files": len(wav_files),
            "processed_files": success_count,
            "error_files": error_count,
            "processed_dir": str(self.processed_dir),
            "semantic_model": self.config["semantic_model"]["model_name"],
            "extract_mel": self.extract_mel,
            "storage_format": "hdf5",
            "use_float16": self.use_float16,
            "samples_per_hdf5": self.samples_per_hdf5,
            "num_hdf5_files": self.hdf5_writer.current_file_idx,
            "pipeline_config": {
                "num_audio_workers": self.num_audio_workers,
                "num_save_workers": self.num_save_workers,
                "prefetch_size": self.prefetch_size
            }
        }
        
        if self.extract_mel and self.mel_extractor is not None:
            metadata["mel_config"] = self.config["preprocessing"]["mel_config"]
        
        metadata_path = self.processed_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_path}")
        
        return metadata


class LJSpeechPreprocessor(DatasetPreprocessor):
    """Preprocessor for LJSpeech dataset."""
    
    def find_audio_files(self) -> List[Path]:
        """Find all wav files in LJSpeech dataset."""
        wav_files = []
        wavs_dir = self.data_root / "wavs"
        
        if not wavs_dir.exists():
            raise FileNotFoundError(f"LJSpeech wavs directory not found: {wavs_dir}")
        
        for wav_file in wavs_dir.glob("*.wav"):
            wav_files.append(wav_file)
        
        return sorted(wav_files)
    
    def get_sample_key(self, wav_file: Path) -> str:
        """Get unique sample key for HDF5 storage.
        For LJSpeech: LJ001-0001.wav -> LJ001-0001
        """
        return wav_file.stem


class LibriTTSPreprocessor(DatasetPreprocessor):
    """Preprocessor for LibriTTS dataset."""
    
    def find_audio_files(self) -> List[Path]:
        """Find all wav files in LibriTTS dataset."""
        wav_files = []
        
        subdirs = ["train-clean-100", "train-clean-360", "train-other-500",
                   "dev-clean", "dev-other", "test-clean", "test-other"]
        
        for subdir in subdirs:
            subdir_path = self.data_root / subdir
            if not subdir_path.exists():
                print(f"Warning: {subdir_path} not found, skipping...")
                continue
            
            for speaker_dir in subdir_path.iterdir():
                if not speaker_dir.is_dir():
                    continue
                
                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue
                    
                    for wav_file in chapter_dir.glob("*.wav"):
                        wav_files.append(wav_file)
        
        return sorted(wav_files)
    
    def get_sample_key(self, wav_file: Path) -> str:
        """Get unique sample key for HDF5 storage.
        For LibriTTS: train-clean-100/123/456/123_456_789.wav -> train-clean-100/123/456/123_456_789
        """
        relative_path = wav_file.relative_to(self.data_root)
        # Replace path separators with forward slashes for consistent HDF5 group naming
        return str(relative_path.with_suffix('')).replace(os.sep, '/')


def create_preprocessor(config: Dict[str, Any], dataset_name: str) -> DatasetPreprocessor:
    """Factory function to create appropriate preprocessor for dataset."""
    preprocessors = {
        "ljspeech": LJSpeechPreprocessor,
        "libritts": LibriTTSPreprocessor,
    }
    
    if dataset_name not in preprocessors:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported datasets: {list(preprocessors.keys())}")
    
    return preprocessors[dataset_name](config, dataset_name)

# 🚀 性能提升技术
# 1. 异步音频加载 (AudioLoader)

# 使用 ThreadPoolExecutor 预加载音频文件
# 配置 num_audio_workers=8 个线程并行加载

# 2. 智能预取队列

# 在处理当前文件时，提前加载后续 prefetch_size=20 个文件
# 避免GPU等待I/O的空闲时间

# 3. 异步文件保存 (AsyncFileSaver)

# 使用独立的线程池 num_save_workers=4 异步保存
# GPU处理完立即提交保存任务，不阻塞主流程

# 4. CUDA Stream优化

# 使用 torch.cuda.Stream() 实现GPU异步执行
# non_blocking=True 数据传输

# 5. 流水线并行
# [Audio Load] → [GPU Process] → [Async Save]
#      ↓              ↓               ↓
#   Thread Pool    CUDA Stream    Thread Pool