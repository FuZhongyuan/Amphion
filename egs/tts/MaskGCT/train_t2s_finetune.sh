#!/bin/bash

# Fine-tuning script for MaskGCT T2S model with reduced codebook size
# This script fine-tunes a pretrained T2S model (8192 codebook) to work with
# a reduced codebook size (512), training only the final projection layer.

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/t2s_finetune.json"
exp_name="t2s_finetune"

# Check if checkpoint exists for resume training
checkpoint_dir="${work_dir}/ckpts/maskgct/${exp_name}_model/checkpoint"
resume_args=""
if [ -d "$checkpoint_dir" ] && [ "$(ls -A $checkpoint_dir/epoch* 2>/dev/null)" ]; then
    echo "Found existing checkpoint in $checkpoint_dir, enabling resume mode..."
    resume_args="--resume --resume_type resume"
else
    echo "No existing checkpoint found, starting new training..."
fi

####### Train Model ###########
# Note: Using bf16 mixed precision for better training stability
# Adjust CUDA_VISIBLE_DEVICES to use your available GPU(s)
CUDA_VISIBLE_DEVICES="2,3" accelerate launch --main_process_port 14559 --mixed_precision="bf16" \
    "${work_dir}"/bins/tts/maskgct_train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    $resume_args

