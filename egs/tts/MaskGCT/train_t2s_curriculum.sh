#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/t2s_curriculum.json"
exp_name="t2s_curriculum"

# Check if checkpoint exists for resume training
checkpoint_dir="${work_dir}/ckpts/maskgct_mini/${exp_name}/checkpoint"
resume_args=""
if [ -d "$checkpoint_dir" ] && [ "$(ls -A $checkpoint_dir/epoch* 2>/dev/null)" ]; then
    echo "Found existing checkpoint in $checkpoint_dir, enabling resume mode..."
    resume_args="--resume --resume_type resume"
else
    echo "No existing checkpoint found, starting curriculum training..."
fi

####### Train Model ###########
CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch --main_process_port 16494 \
    "${work_dir}"/bins/tts/maskgct_train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    $resume_args
