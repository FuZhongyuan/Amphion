#!/bin/bash
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/semantic_to_mel_mini.json"
exp_name="semantic_to_mel_mini"

######## Train Model ###########
echo "Starting Semantic-to-Mel training (mini model)..."
echo "Config: $exp_config"
echo "Experiment name: $exp_name"

# Multi-GPU training with accelerate
CUDA_VISIBLE_DEVICES="4,5" accelerate launch --main_process_port 13546 --mixed_precision="bf16" \
    "${work_dir}"/bins/tts/maskgct_train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    $resume_args

# For single GPU training, use:
# python "${work_dir}"/bins/tts/maskgct_train.py \
#     --config=$exp_config \
#     --exp_name=$exp_name \
#     --log_level=info

# For resuming training, add --resume flag:
# accelerate launch --config_file "${work_dir}"/accelerate_config.yaml \
#     "${work_dir}"/bins/tts/maskgct_train.py \
#     --config=$exp_config \
#     --exp_name=$exp_name \
#     --log_level=info \
#     --resume \
#     --resume_type=resume
