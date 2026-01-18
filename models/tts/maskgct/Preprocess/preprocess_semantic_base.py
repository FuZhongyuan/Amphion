#!/usr/bin/env python3
"""
Unified preprocessing pipeline for semantic features extraction.
This module provides a common interface for preprocessing different datasets (LJSpeech, LibriTTS, etc.)
with semantic features from w2v-bert-2.0 model and optional mel-spectrogram extraction.
"""

import os
import json
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel
from utils.mel import extract_mel_features
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class AudioLoader:
    """Handles audio file loading and resampling."""
    
    @staticmethod
    def load_audio(file_path: str, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """Load audio file and resample to target sample rate."""
        try:
            audio, _ = librosa.load(file_path, sr=sample_rate)
            return audio
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None


class SemanticFeatureExtractor:
    """Extracts semantic features using w2v-bert-2.0 model."""
    
    def __init__(self, model_name: str, output_layer: int = 17, device: str = "cuda"):
        """
        Initialize semantic feature extractor.
        
        Args:
            model_name: HuggingFace model name for w2v-bert-2.0
            output_layer: Which hidden layer to extract features from
            device: Device to run model on ('cuda' or 'cpu')
        """
        print("Loading semantic model...")
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2BertModel.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_layer = output_layer
        
        print(f"Semantic model loaded on device: {self.device}")
    
    def extract(self, audio: np.ndarray, sampling_rate: int = 16000) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract semantic features from audio.
        
        Args:
            audio: Audio waveform as numpy array
            sampling_rate: Sampling rate of the audio
            
        Returns:
            Dictionary containing hidden_states, or None if extraction fails
        """
        try:
            inputs = self.processor(audio, sampling_rate=sampling_rate)
            input_features = inputs["input_features"][0]
            attention_mask = inputs["attention_mask"][0]
            
            # Extract hidden states from semantic model on GPU
            with torch.no_grad():
                outputs = self.model(
                    input_features=torch.tensor(input_features).unsqueeze(0).to(self.device),
                    attention_mask=torch.tensor(attention_mask).unsqueeze(0).to(self.device),
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[self.output_layer][0].cpu().numpy()
            
            return {
                "hidden_states": hidden_states
            }
        except Exception as e:
            print(f"Failed to extract semantic features: {e}")
            return None


class MelSpectrogramExtractor:
    """Extracts mel-spectrogram features."""
    
    def __init__(self, mel_config: Dict[str, Any]):
        """
        Initialize mel-spectrogram extractor.
        
        Args:
            mel_config: Dictionary containing mel configuration parameters
        """
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
        """
        Extract mel-spectrogram from audio.
        
        Args:
            audio: Audio waveform as numpy array
            
        Returns:
            Mel-spectrogram as numpy array [B, T_mel, n_mel], or None if extraction fails
        """
        try:
            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)  # [1, T]
            
            # Extract mel spectrogram
            mel = extract_mel_features(audio_tensor, self.cfg)  # [n_mel, T_mel]
            
            # Transpose to [B, T_mel, n_mel]
            if len(mel.shape) == 2:
                mel = mel.unsqueeze(0)
            mel = mel.permute(0, 2, 1)  # [B, T_mel, n_mel]
            
            return mel.numpy()
        except Exception as e:
            print(f"Failed to extract mel spectrogram: {e}")
            return None


class DatasetPreprocessor(ABC):
    """Abstract base class for dataset-specific preprocessing."""
    
    def __init__(self, config: Dict[str, Any], dataset_name: str):
        """
        Initialize dataset preprocessor.
        
        Args:
            config: Full preprocessing configuration
            dataset_name: Name of the dataset (e.g., 'ljspeech', 'libritts')
        """
        self.config = config
        self.dataset_name = dataset_name
        self.dataset_config = config["datasets"][dataset_name]
        
        self.data_root = Path(self.dataset_config["data_root"])
        self.processed_dir = Path(self.dataset_config["processed_dir"])
        self.overwrite_existing = config["preprocessing"]["overwrite_existing"]
        self.extract_mel = config["preprocessing"].get("extract_mel", False)
        
        # Initialize feature extractors
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
        """
        Find all audio files in the dataset.
        
        Returns:
            List of Path objects pointing to audio files
        """
        pass
    
    @abstractmethod
    def get_output_path(self, wav_file: Path) -> Path:
        """
        Get output path for processed features.
        
        Args:
            wav_file: Path to input audio file
            
        Returns:
            Path where processed features should be saved
        """
        pass
    
    def process_single_file(self, wav_path: Path, output_path: Path) -> str:
        """
        Process a single audio file.
        
        Args:
            wav_path: Path to input audio file
            output_path: Path to save processed features
            
        Returns:
            Status message string
        """
        # Check if output already exists
        if output_path.exists() and not self.overwrite_existing:
            return f"Skipped {wav_path} (already exists)"
        
        # Load audio
        audio = AudioLoader.load_audio(str(wav_path), sample_rate=16000)
        if audio is None:
            return f"Failed to load {wav_path}"
        
        # Extract semantic features
        semantic_features = self.semantic_extractor.extract(audio, sampling_rate=16000)
        if semantic_features is None:
            return f"Failed to extract semantic features for {wav_path}"
        
        # Prepare data to save
        save_data = {
            "hidden_states": semantic_features["hidden_states"]
        }
        
        # Extract mel spectrogram if requested
        if self.extract_mel and self.mel_extractor is not None:
            mel_spectrogram = self.mel_extractor.extract(audio)
            if mel_spectrogram is None:
                return f"Failed to extract mel spectrogram for {wav_path}"
            save_data["mel_spectrogram"] = mel_spectrogram
        
        # Save features
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, **save_data)
            return f"Processed {wav_path}"
        except Exception as e:
            return f"Failed to save {output_path}: {e}"
    
    def run(self) -> Dict[str, Any]:
        """
        Run the preprocessing pipeline.
        
        Returns:
            Dictionary containing processing statistics and metadata
        """
        print(f"Processing {self.dataset_name.upper()} dataset...")
        print(f"Data root: {self.data_root}")
        print(f"Processed dir: {self.processed_dir}")
        print("Using GPU acceleration for semantic feature extraction")
        
        if self.extract_mel:
            print("Also extracting and caching mel spectrograms")
        else:
            print("Only extracting semantic features (mel spectrograms will be computed at training time)")
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all audio files
        print("Finding audio files...")
        wav_files = self.find_audio_files()
        print(f"Found {len(wav_files)} audio files")
        
        if len(wav_files) == 0:
            raise ValueError(f"No audio files found in {self.data_root}")
        
        # Process files
        print("Processing files with GPU acceleration...")
        success_count = 0
        error_count = 0
        
        with tqdm(total=len(wav_files), desc="Processing") as pbar:
            for wav_file in wav_files:
                output_path = self.get_output_path(wav_file)
                result = self.process_single_file(wav_file, output_path)
                
                if "Processed" in result:
                    success_count += 1
                else:
                    error_count += 1
                    if error_count <= 10:  # Only print first 10 errors to avoid spam
                        print(f"\nError: {result}")
                
                pbar.update(1)
                pbar.set_postfix({"success": success_count, "errors": error_count})
        
        print("\nPreprocessing completed!")
        print(f"Successfully processed: {success_count}")
        print(f"Errors: {error_count}")
        
        # Prepare metadata
        metadata = {
            "dataset": self.dataset_name,
            "total_files": len(wav_files),
            "processed_files": success_count,
            "error_files": error_count,
            "processed_dir": str(self.processed_dir),
            "semantic_model": self.config["semantic_model"]["model_name"],
            "extract_mel": self.extract_mel
        }
        
        if self.extract_mel and self.mel_extractor is not None:
            metadata["mel_config"] = self.config["preprocessing"]["mel_config"]
        
        # Save metadata
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
    
    def get_output_path(self, wav_file: Path) -> Path:
        """Get output path for LJSpeech processed features."""
        # Create relative path for output
        relative_path = wav_file.relative_to(self.data_root / "wavs")
        output_path = self.processed_dir / relative_path.with_suffix('.npz')
        return output_path


class LibriTTSPreprocessor(DatasetPreprocessor):
    """Preprocessor for LibriTTS dataset."""
    
    def find_audio_files(self) -> List[Path]:
        """Find all wav files in LibriTTS dataset."""
        wav_files = []
        
        # LibriTTS has multiple subdirectories
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
    
    def get_output_path(self, wav_file: Path) -> Path:
        """Get output path for LibriTTS processed features."""
        # Create relative path for output, preserving LibriTTS directory structure
        relative_path = wav_file.relative_to(self.data_root)
        output_path = self.processed_dir / relative_path.with_suffix('.npz')
        return output_path


def create_preprocessor(config: Dict[str, Any], dataset_name: str) -> DatasetPreprocessor:
    """
    Factory function to create appropriate preprocessor for dataset.
    
    Args:
        config: Full preprocessing configuration
        dataset_name: Name of the dataset ('ljspeech' or 'libritts')
        
    Returns:
        DatasetPreprocessor instance
        
    Raises:
        ValueError: If dataset_name is not supported
    """
    preprocessors = {
        "ljspeech": LJSpeechPreprocessor,
        "libritts": LibriTTSPreprocessor,
    }
    
    if dataset_name not in preprocessors:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(preprocessors.keys())}")
    
    return preprocessors[dataset_name](config, dataset_name)
