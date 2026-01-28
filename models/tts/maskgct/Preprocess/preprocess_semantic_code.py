#!/usr/bin/env python3
"""
Semantic Code Preprocessing Script

This script quantizes pre-extracted semantic hidden states into discrete semantic codes
using the RepCodec model. The quantized codes are stored in HDF5 files for efficient
loading during training.

Usage:
    python preprocess_semantic_code.py --config preprocess_semantic_code_config.json

The input HDF5 files should contain 'hidden_states' datasets (from preprocess_semantic_base.py).
The output will add 'semantic_code' datasets to the same or new HDF5 files.
"""

import os
import json
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List
import threading
from concurrent.futures import ThreadPoolExecutor
from utils.util import load_config


class SemanticCodec:
    """Wrapper for RepCodec model to quantize semantic features."""

    def __init__(
        self,
        cfg,
        device: str = "cuda"
    ):
        """Initialize the semantic codec.

        Args:
            cfg: Configuration object containing codec parameters (e.g., codebook_size, hidden_size, etc.)
            device: Device to run the model on
        """
        from models.codec.kmeans.repcodec_model import RepCodec
        import safetensors.torch

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize model with config
        self.model = RepCodec(cfg=cfg)

        # Load pretrained weights
        pretrained_path = getattr(cfg, "pretrained_path", None)
        if pretrained_path:
            if os.path.isdir(pretrained_path):
                model_path = os.path.join(pretrained_path, "model.safetensors")
                if not os.path.exists(model_path):
                    model_path = os.path.join(pretrained_path, "pytorch_model.bin")
            else:
                model_path = pretrained_path

            if os.path.exists(model_path):
                print(f"Loading semantic codec from: {model_path}")
                if model_path.endswith(".safetensors"):
                    safetensors.torch.load_model(self.model, model_path)
                else:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "model" in checkpoint:
                        self.model.load_state_dict(checkpoint["model"], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.eval()
        self.model.to(self.device)

        # Load normalization statistics
        self.semantic_mean = None
        self.semantic_std = None
        stat_path = getattr(cfg, "representation_stat_mean_var_path", None)
        if stat_path and os.path.exists(stat_path):
            print(f"Loading normalization statistics from: {stat_path}")
            stat = torch.load(stat_path, map_location=self.device)
            self.semantic_mean = torch.tensor(stat["mean"]).to(self.device)
            self.semantic_std = torch.sqrt(torch.tensor(stat["var"])).to(self.device)

        print(f"Semantic codec loaded on device: {self.device}")
        codebook_size = getattr(cfg, "codebook_size", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        print(f"Codebook size: {codebook_size}, Hidden size: {hidden_size}")

    @torch.no_grad()
    def quantize(self, hidden_states: np.ndarray) -> np.ndarray:
        """Quantize hidden states to semantic codes.

        Args:
            hidden_states: Input features of shape [T, D] or [B, T, D]

        Returns:
            Semantic codes of shape [T] or [B, T]
        """
        # Convert to tensor
        feat = torch.from_numpy(hidden_states).float().to(self.device)

        # Add batch dimension if needed
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normalize
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        # Quantize
        semantic_codes, _ = self.model.quantize(feat)

        # Remove batch dimension if added
        if squeeze_output:
            semantic_codes = semantic_codes.squeeze(0)

        return semantic_codes.cpu().numpy()


class HDF5SemanticCodeProcessor:
    """Process HDF5 files to add semantic codes."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        semantic_codec: SemanticCodec,
        batch_size: int = 32,
        overwrite: bool = False,
        samples_per_file: int = 20000,
    ):
        """Initialize the processor.

        Args:
            input_dir: Directory containing input HDF5 files with hidden_states
            output_dir: Directory to save output HDF5 files with semantic_code
            semantic_codec: SemanticCodec instance for quantization
            batch_size: Batch size for processing
            overwrite: Whether to overwrite existing semantic_code
            samples_per_file: Number of samples per output HDF5 file
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.semantic_codec = semantic_codec
        self.batch_size = batch_size
        self.overwrite = overwrite
        self.samples_per_file = samples_per_file

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load input index
        self.input_index_path = self.input_dir / "hdf5_index.json"
        if not self.input_index_path.exists():
            raise FileNotFoundError(f"Input HDF5 index not found: {self.input_index_path}")

        with open(self.input_index_path, 'r') as f:
            self.input_index = json.load(f)

        # Initialize output index
        self.output_index_path = self.output_dir / "hdf5_index.json"
        self.output_index = {"sample_index": {}, "current_file_idx": 0, "current_file_count": 0}
        if self.output_index_path.exists():
            with open(self.output_index_path, 'r') as f:
                self.output_index = json.load(f)

        self.current_output_file = None
        self.current_output_file_idx = self.output_index.get("current_file_idx", 0)
        self.current_output_file_count = self.output_index.get("current_file_count", 0)
        self.lock = threading.Lock()

    def get_output_hdf5_path(self, file_idx: int) -> Path:
        """Get output HDF5 file path."""
        return self.output_dir / f"semantic_code_{file_idx:05d}.h5"

    def save_output_index(self):
        """Save output index to disk."""
        self.output_index["current_file_idx"] = self.current_output_file_idx
        self.output_index["current_file_count"] = self.current_output_file_count
        with open(self.output_index_path, 'w') as f:
            json.dump(self.output_index, f, indent=2)

    def write_sample(self, sample_key: str, semantic_code: np.ndarray) -> bool:
        """Write a single sample to output HDF5 file."""
        with self.lock:
            try:
                # Check if we need a new file
                if self.current_output_file_count >= self.samples_per_file or self.current_output_file is None:
                    if self.current_output_file is not None:
                        self.current_output_file.close()

                    if self.current_output_file is not None or self.current_output_file_count > 0:
                        self.current_output_file_idx += 1
                        self.current_output_file_count = 0
                    elif self.current_output_file_idx == 0:
                        self.current_output_file_idx = 1
                        self.current_output_file_count = 0

                    hdf5_path = self.get_output_hdf5_path(self.current_output_file_idx)
                    self.current_output_file = h5py.File(hdf5_path, 'a')

                # Create group for this sample
                if sample_key in self.current_output_file:
                    del self.current_output_file[sample_key]

                sample_group = self.current_output_file.create_group(sample_key)

                # Write semantic code as int16 (sufficient for codebook indices)
                sample_group.create_dataset(
                    "semantic_code",
                    data=semantic_code.astype(np.int16),
                    compression="gzip",
                    compression_opts=4
                )

                # Update index
                self.output_index["sample_index"][sample_key] = {
                    "file_idx": self.current_output_file_idx,
                    "group_name": sample_key
                }

                self.current_output_file_count += 1

                # Periodically save index
                if self.current_output_file_count % 100 == 0:
                    self.save_output_index()

                return True
            except Exception as e:
                print(f"Failed to write sample {sample_key}: {e}")
                return False

    def close(self):
        """Close current output file and save index."""
        with self.lock:
            if self.current_output_file is not None:
                self.current_output_file.close()
                self.current_output_file = None
            self.save_output_index()

    def process(self) -> Dict[str, Any]:
        """Process all samples in input HDF5 files."""
        sample_index = self.input_index.get("sample_index", {})
        total_samples = len(sample_index)

        print(f"Processing {total_samples} samples...")
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")

        # Group samples by input file for efficient reading
        file_to_samples: Dict[int, List[str]] = {}
        for sample_key, info in sample_index.items():
            file_idx = info["file_idx"]
            if file_idx not in file_to_samples:
                file_to_samples[file_idx] = []
            file_to_samples[file_idx].append(sample_key)

        success_count = 0
        error_count = 0
        skip_count = 0

        # Process each input file
        for file_idx in tqdm(sorted(file_to_samples.keys()), desc="Processing HDF5 files"):
            input_path = self.input_dir / f"features_{file_idx:05d}.h5"
            if not input_path.exists():
                print(f"Warning: Input file not found: {input_path}")
                continue

            sample_keys = file_to_samples[file_idx]

            with h5py.File(input_path, 'r') as input_file:
                for sample_key in tqdm(sample_keys, desc=f"File {file_idx}", leave=False):
                    # Skip if already processed
                    if not self.overwrite and sample_key in self.output_index["sample_index"]:
                        skip_count += 1
                        continue

                    try:
                        # Read hidden states
                        if sample_key not in input_file:
                            print(f"Warning: Sample {sample_key} not found in {input_path}")
                            error_count += 1
                            continue

                        sample_group = input_file[sample_key]
                        if "hidden_states" not in sample_group:
                            print(f"Warning: hidden_states not found for {sample_key}")
                            error_count += 1
                            continue

                        hidden_states = sample_group["hidden_states"][:]

                        # Convert float16 to float32 if needed
                        if hidden_states.dtype == np.float16:
                            hidden_states = hidden_states.astype(np.float32)

                        # Quantize
                        semantic_code = self.semantic_codec.quantize(hidden_states)

                        # Write to output
                        if self.write_sample(sample_key, semantic_code):
                            success_count += 1
                        else:
                            error_count += 1

                    except Exception as e:
                        print(f"Error processing {sample_key}: {e}")
                        error_count += 1

        # Close output file
        self.close()

        print(f"\nProcessing completed!")
        print(f"Successfully processed: {success_count}")
        print(f"Skipped (already exists): {skip_count}")
        print(f"Errors: {error_count}")

        # Save metadata
        metadata = {
            "total_samples": total_samples,
            "processed_samples": success_count,
            "skipped_samples": skip_count,
            "error_samples": error_count,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "num_output_files": self.current_output_file_idx,
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return metadata




def main():
    parser = argparse.ArgumentParser(
        description="Quantize semantic hidden states to discrete codes using RepCodec"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration JSON file"
    )

    args = parser.parse_args()

    # Load config if provided
    if not args.config:
        raise ValueError("--config is required")
    cfg = load_config(args.config)

    # Get parameters from config
    input_dir = getattr(cfg, "input_dir", None)
    output_dir = getattr(cfg, "output_dir", None)
    device = getattr(cfg, "device", "cuda")
    batch_size = getattr(cfg, "batch_size", 32)
    samples_per_file = getattr(cfg, "samples_per_file", 20000)
    overwrite = getattr(cfg, "overwrite", False)

    # Validate required parameters
    if not input_dir:
        raise ValueError("input_dir is required in config file")
    if not output_dir:
        raise ValueError("output_dir is required in config file")
    if not hasattr(cfg, "semantic_codec"):
        raise ValueError("semantic_codec is required in config file")
    if not hasattr(cfg.semantic_codec, "pretrained_path"):
        raise ValueError("semantic_codec.pretrained_path is required in config file")

    print("=" * 60)
    print("Semantic Code Preprocessing")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Pretrained path: {cfg.semantic_codec.pretrained_path}")
    stat_path = getattr(cfg.semantic_codec, "representation_stat_mean_var_path", None)
    print(f"Stat path: {stat_path}")
    print(f"Codebook size: {getattr(cfg.semantic_codec, 'codebook_size', None)}")
    print(f"Hidden size: {getattr(cfg.semantic_codec, 'hidden_size', None)}")
    print(f"Device: {device}")
    print("=" * 60)

    # Initialize semantic codec
    semantic_codec = SemanticCodec(
        cfg=cfg.semantic_codec,
        device=device
    )

    # Initialize processor
    processor = HDF5SemanticCodeProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        semantic_codec=semantic_codec,
        batch_size=batch_size,
        overwrite=overwrite,
        samples_per_file=samples_per_file
    )

    # Run processing
    metadata = processor.process()

    print("\nDone!")
    return metadata


if __name__ == "__main__":
    main()
