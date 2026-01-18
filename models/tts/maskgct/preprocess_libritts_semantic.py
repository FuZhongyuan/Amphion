#!/usr/bin/env python3
"""
Preprocess semantic features for LibriTTS dataset to avoid real-time extraction during training.
This script extracts and caches semantic features from w2v-bert-2.0 model for faster training.
"""

import os
import json
import argparse
import numpy as np
import librosa
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import SeamlessM4TFeatureExtractor


def load_audio(file_path, sample_rate=16000):
    """Load audio file and resample to target sample rate."""
    try:
        audio, _ = librosa.load(file_path, sr=sample_rate)
        return audio
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return None


def extract_semantic_features(audio, processor):
    """Extract semantic features from audio using w2v-bert-2.0."""
    try:
        inputs = processor(audio, sampling_rate=16000)
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return {
            "input_features": input_features,
            "attention_mask": attention_mask
        }
    except Exception as e:
        print(f"Failed to extract features: {e}")
        return None


def process_single_file(args):
    """Process a single audio file."""
    wav_path, output_path, processor, overwrite_existing = args

    # Check if output already exists
    if output_path.exists() and not overwrite_existing:
        return f"Skipped {wav_path} (already exists)"

    # Load audio
    audio = load_audio(str(wav_path), sample_rate=16000)
    if audio is None:
        return f"Failed to load {wav_path}"

    # Extract features
    features = extract_semantic_features(audio, processor)
    if features is None:
        return f"Failed to extract features for {wav_path}"

    # Save features
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            input_features=features["input_features"],
            attention_mask=features["attention_mask"]
        )
        return f"Processed {wav_path}"
    except Exception as e:
        return f"Failed to save {output_path}: {e}"


def find_audio_files(data_root):
    """Find all wav files in LibriTTS dataset."""
    wav_files = []

    # LibriTTS has multiple subdirectories like train-clean-100, train-clean-360, etc.
    for subdir in ["train"]:
        subdir_path = Path(data_root) / subdir
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


def main():
    parser = argparse.ArgumentParser(description="Preprocess semantic features for LibriTTS dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing config JSON")
    parser.add_argument("--dataset", type=str, default="libritts", choices=["ljspeech", "libritts"],
                       help="Dataset type")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset_config = config["datasets"][args.dataset]

    data_root = dataset_config["data_root"]
    processed_dir = Path(dataset_config["processed_dir"])
    max_workers = dataset_config["max_workers"]
    overwrite_existing = config["preprocessing"]["overwrite_existing"]

    print(f"Processing {args.dataset.upper()} dataset...")
    print(f"Data root: {data_root}")
    print(f"Processed dir: {processed_dir}")
    print(f"Max workers: {max_workers}")

    # Create processed directory
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files
    print("Finding audio files...")
    wav_files = find_audio_files(data_root)
    print(f"Found {len(wav_files)} audio files")

    # Initialize semantic processor
    print("Loading semantic model...")
    processor = SeamlessM4TFeatureExtractor.from_pretrained(
        config["semantic_model"]["model_name"]
    )

    # Prepare processing tasks
    tasks = []
    for wav_file in wav_files:
        # Create relative path for output, preserving LibriTTS directory structure
        relative_path = wav_file.relative_to(data_root)
        output_path = processed_dir / relative_path.with_suffix('.npz')
        tasks.append((wav_file, output_path, processor, overwrite_existing))

    # Process files in parallel
    print("Processing files...")
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if "Processed" in result:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"Error: {result}")
                pbar.update(1)
                pbar.set_postfix({"success": success_count, "errors": error_count})

    print("Preprocessing completed!")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")

    # Save metadata
    metadata = {
        "dataset": args.dataset,
        "total_files": len(wav_files),
        "processed_files": success_count,
        "error_files": error_count,
        "processed_dir": str(processed_dir),
        "semantic_model": config["semantic_model"]["model_name"]
    }

    metadata_path = processed_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


if __name__ == "__main__":
    main()
