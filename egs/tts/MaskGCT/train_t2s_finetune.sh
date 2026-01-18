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
exp_config="$exp_dir/t2s_finetune.json"
exp_name="t2s_finetune"

# Check if teacher model exists
teacher_model_path="${work_dir}/ckpts/MaskGCT-ckpt/t2s_model/model.safetensors"
if [ ! -f "$teacher_model_path" ]; then
    echo "ERROR: Teacher model not found at: $teacher_model_path"
    echo "Please ensure the pre-trained T2S model is available before fine-tuning."
    exit 1
fi

echo "Teacher model found: $teacher_model_path"

# Check if checkpoint exists for resume training
checkpoint_dir="${work_dir}/ckpts/maskgct/${exp_name}_model/checkpoint"
resume_args=""
if [ -d "$checkpoint_dir" ] && [ "$(ls -A $checkpoint_dir/epoch* 2>/dev/null)" ]; then
    echo "Found existing checkpoint in $checkpoint_dir, enabling resume mode..."
    resume_args="--resume --resume_type resume"
else
    echo "No existing checkpoint found, starting fine-tuning from teacher model..."
fi

####### Train Model ###########
# Use single GPU for fine-tuning (you can adjust CUDA_VISIBLE_DEVICES as needed)
CUDA_VISIBLE_DEVICES="4,5" accelerate launch --main_process_port 14559 --mixed_precision="bf16" \
    "${work_dir}"/bins/tts/maskgct_train.py \
    --config=$exp_config \
    --exp_name=$exp_name \
    --log_level debug \
    $resume_args
