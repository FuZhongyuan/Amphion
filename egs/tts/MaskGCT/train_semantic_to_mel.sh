#!/bin/bash
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/semantic_to_mel_mini.json"
exp_name="semantic_to_mel"

######## Train Model ###########
echo "Starting Semantic-to-Mel training..."
echo "Config: $exp_config"
echo "Experiment name: $exp_name"

# Single GPU training
python "${work_dir}"/bins/tts/maskgct_train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level=info

# For multi-GPU training with accelerate, use:
# accelerate launch --config_file "${work_dir}"/accelerate_config.yaml \
#     "${work_dir}"/bins/tts/maskgct_train.py \
#     --config=$exp_config \
#     --exp_name=$exp_name \
#     --log_level=info

# For resuming training, add --resume flag:
# python "${work_dir}"/bins/tts/maskgct_train.py \
#     --config=$exp_config \
#     --exp_name=$exp_name \
#     --log_level=info \
#     --resume \
#     --resume_type=resume
