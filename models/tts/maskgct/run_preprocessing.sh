#!/bin/bash

# Script to run semantic feature preprocessing for MaskGCT training
# This script preprocesses semantic features to avoid real-time extraction during training

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"
export CUDA_VISIBLE_DEVICES="1"
set -e  # Exit on any error

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

echo "=== MaskGCT Semantic Feature Preprocessing ==="

# Set paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$PROJECT_ROOT/egs/tts/MaskGCT/preprocess_semantic_features.json"
SCRIPT_DIR="$PROJECT_ROOT/models/tts/maskgct"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Config file: $CONFIG_FILE"
echo "Project root: $PROJECT_ROOT"

# Function to run preprocessing for a dataset
run_preprocessing() {
    local dataset=$1
    local script_name="preprocess_${dataset}_semantic.py"
    local script_path="$SCRIPT_DIR/$script_name"

    echo ""
    echo "=== Processing $dataset ==="

    if [ ! -f "$script_path" ]; then
        echo "Error: Script not found: $script_path"
        return 1
    fi

    echo "Running: python $script_path --config $CONFIG_FILE --dataset $dataset"

    # Run the preprocessing script
    python "$script_path" --config "$CONFIG_FILE" --dataset "$dataset"

    if [ $? -eq 0 ]; then
        echo "✓ $dataset preprocessing completed successfully"
    else
        echo "✗ $dataset preprocessing failed"
        return 1
    fi
}

# Run preprocessing for both datasets
echo "Starting semantic feature preprocessing..."

run_preprocessing "ljspeech"
# run_preprocessing "libritts"

echo ""
echo "=== All preprocessing tasks completed! ==="
echo ""
echo "Next steps:"
echo "1. Verify that processed features are saved in the configured directories"
echo "2. Update your training config to use semantic feature caching"
echo "3. Run training with improved GPU utilization"
echo ""
echo "To check processed data:"
echo "  ls -la data/processed_maskgct/"
