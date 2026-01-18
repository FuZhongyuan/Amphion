#!/usr/bin/env python3
"""
Preprocess semantic features for LibriTTS dataset to avoid real-time extraction during training.
This script extracts and caches semantic features from w2v-bert-2.0 model for faster training.

This is a wrapper script that uses the unified preprocessing pipeline from preprocess_semantic_base.py
"""

import json
import argparse
import sys
from pathlib import Path

# Add the script directory to Python path to import preprocess_semantic_base
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from preprocess_semantic_base import create_preprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess semantic features for LibriTTS dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to preprocessing config JSON")
    parser.add_argument("--dataset", type=str, default="libritts", choices=["ljspeech", "libritts"],
                       help="Dataset type")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Create and run preprocessor
    preprocessor = create_preprocessor(config, args.dataset)
    metadata = preprocessor.run()
    
    print(f"\nPreprocessing summary:")
    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Total files: {metadata['total_files']}")
    print(f"  Successfully processed: {metadata['processed_files']}")
    print(f"  Errors: {metadata['error_files']}")
    print(f"  Output directory: {metadata['processed_dir']}")


if __name__ == "__main__":
    main()
