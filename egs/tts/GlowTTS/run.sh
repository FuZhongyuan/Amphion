#!/bin/bash
# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

# Build monotonic align module
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

######## Parse the Given Parameters from the Command ###########
options=$(getopt -o c:n:s --long gpu:,config:,infer_expt_dir:,infer_output_dir:,infer_mode:,infer_dataset:,infer_testing_set:,infer_text:,name:,stage:,vocoder_dir:,noise_scale:,length_scale: -- "$@")
eval set -- "$options"

while true; do
  case $1 in
    # Experimental Configuration File
    -c | --config) shift; exp_config=$1 ; shift ;;
    # Experimental Name
    -n | --name) shift; exp_name=$1 ; shift ;;
    # Running Stage
    -s | --stage) shift; running_stage=$1 ; shift ;;
    # Visible GPU machines. The default value is "0".
    --gpu) shift; gpu=$1 ; shift ;;

    # [Only for Inference] The experiment dir
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;
    # [Only for Inference] The inference mode: "batch" or "single"
    --infer_mode) shift; infer_mode=$1 ; shift ;;
    # [Only for Inference] The inference dataset
    --infer_dataset) shift; infer_dataset=$1 ; shift ;;
    # [Only for Inference] The inference testing set
    --infer_testing_set) shift; infer_testing_set=$1 ; shift ;;
    # [Only for Inference] The text to be synthesized
    --infer_text) shift; infer_text=$1 ; shift ;;
    # [Only for Inference] The vocoder directory
    --vocoder_dir) shift; vocoder_dir=$1 ; shift ;;
    # [Only for Inference] Noise scale for sampling
    --noise_scale) shift; noise_scale=$1 ; shift ;;
    # [Only for Inference] Length scale for duration
    --length_scale) shift; length_scale=$1 ; shift ;;

    --) shift ; break ;;
    *) echo "Invalid option: $1" exit 1 ;;
  esac
done


### Value check ###
if [ -z "$running_stage" ]; then
    echo "[Error] Please specify the running stage"
    exit 1
fi

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config.json
fi
echo "Experimental Configuration File: $exp_config"

if [ -z "$gpu" ]; then
    gpu="0"
fi

if [ -z "$noise_scale" ]; then
    noise_scale=0.667
fi

if [ -z "$length_scale" ]; then
    length_scale=1.0
fi

######## Features Extraction ###########
if [ $running_stage -eq 1 ]; then
    echo "Stage 1: Feature Extraction"
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/tts/preprocess.py \
        --config=$exp_config \
        --num_workers=8
fi

######## Training ###########
if [ $running_stage -eq 2 ]; then
    if [ -z "$exp_name" ]; then
        echo "[Error] Please specify the experiments name"
        exit 1
    fi
    echo "Experimental Name: $exp_name"
    echo "Stage 2: Training"
    export CUDA_LAUNCH_BLOCKING=1
    CUDA_VISIBLE_DEVICES=$gpu accelerate launch "${work_dir}"/bins/tts/train.py \
        --config $exp_config \
        --exp_name $exp_name \
        --log_level debug
fi

######## Inference ###########
if [ $running_stage -eq 3 ]; then
    echo "Stage 3: Inference"

    if [ -z "$infer_expt_dir" ]; then
        echo "[Error] Please specify the experimental directory"
        exit 1
    fi

    if [ -z "$infer_output_dir" ]; then
        infer_output_dir="$infer_expt_dir/result"
    fi

    if [ -z "$vocoder_dir" ]; then
        echo "[Error] Please specify the vocoder directory"
        exit 1
    fi

    if [ -z "$infer_mode" ]; then
        echo "[Error] Please specify the inference mode: 'batch' or 'single'"
        exit 1
    fi

    if [ "$infer_mode" = "batch" ] && [ -z "$infer_dataset" ]; then
        echo "[Error] Please specify the dataset for batch inference"
        exit 1
    fi

    if [ "$infer_mode" = "batch" ] && [ -z "$infer_testing_set" ]; then
        echo "[Error] Please specify the testing set for batch inference"
        exit 1
    fi

    if [ "$infer_mode" = "single" ] && [ -z "$infer_text" ]; then
        echo "[Error] Please specify the text for single inference"
        exit 1
    fi

    if [ "$infer_mode" = "single" ]; then
        echo "Text: ${infer_text}"
        infer_dataset=None
        infer_testing_set=None
    elif [ "$infer_mode" = "batch" ]; then
        infer_text=""
    fi

    CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/tts/inference.py \
        --config $exp_config \
        --acoustics_dir $infer_expt_dir \
        --output_dir $infer_output_dir \
        --mode $infer_mode \
        --dataset $infer_dataset \
        --testing_set $infer_testing_set \
        --text "$infer_text" \
        --log_level debug \
        --vocoder_dir $vocoder_dir \
        --noise_scale $noise_scale \
        --length_scale $length_scale
fi


# Usage Examples:
#
# Stage 1: Feature Extraction
# bash egs/tts/GlowTTS/run.sh --stage 1 --gpu 0
#
# Stage 2: Training
# bash egs/tts/GlowTTS/run.sh --stage 2 --name glowtts_ljspeech --gpu 0
#
# Stage 3: Inference (Single)
# bash egs/tts/GlowTTS/run.sh --stage 3 \
#     --infer_expt_dir ckpts/glowtts/glowtts_ljspeech \
#     --infer_output_dir ckpts/glowtts/glowtts_ljspeech/result \
#     --infer_mode "single" \
#     --infer_text "This is a test of GlowTTS synthesis." \
#     --vocoder_dir ckpts/vocoder/hifigan_ljspeech/checkpoints \
#     --gpu 0
#
# Stage 3: Inference (Batch)
# bash egs/tts/GlowTTS/run.sh --stage 3 \
#     --infer_expt_dir ckpts/glowtts/glowtts_ljspeech \
#     --infer_output_dir ckpts/glowtts/glowtts_ljspeech/result \
#     --infer_mode "batch" \
#     --infer_dataset LJSpeech \
#     --infer_testing_set test \
#     --vocoder_dir ckpts/vocoder/hifigan_ljspeech/checkpoints \
#     --gpu 0
