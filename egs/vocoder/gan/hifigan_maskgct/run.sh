# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Parse the Given Parameters from the Commond ###########
options=$(getopt -o c:n:s --long gpu:,config:,name:,stage:,checkpoint:,resume_type:,main_process_port:,infer_mode:,infer_datasets:,infer_feature_dir:,infer_audio_dir:,infer_expt_dir:,infer_output_dir: -- "$@")
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

    # [Only for Training] The specific checkpoint path that you want to resume from.
    --checkpoint) shift; checkpoint=$1 ; shift ;;
    # [Only for Training] `resume` for loading all the things (including model weights, optimizer, scheduler, and random states). `finetune` for loading only the model weights.
    --resume_type) shift; resume_type=$1 ; shift ;;
    # [Only for Traiing] `main_process_port` for multi gpu training
    --main_process_port) shift; main_process_port=$1 ; shift ;;

    # [Only for Inference] The inference mode
    --infer_mode) shift; infer_mode=$1 ; shift ;;
    # [Only for Inference] The inferenced datasets
    --infer_datasets) shift; infer_datasets=$1 ; shift ;;
    # [Only for Inference] The feature dir for inference
    --infer_feature_dir) shift; infer_feature_dir=$1 ; shift ;;
    # [Only for Inference] The audio dir for inference
    --infer_audio_dir) shift; infer_audio_dir=$1 ; shift ;;
    # [Only for Inference] The experiment dir. The value is like "[Your path to save logs and checkpoints]/[YourExptName]"
    --infer_expt_dir) shift; infer_expt_dir=$1 ; shift ;;
    # [Only for Inference] The output dir to save inferred audios. Its default value is "$expt_dir/result"
    --infer_output_dir) shift; infer_output_dir=$1 ; shift ;;

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
echo "Exprimental Configuration File: $exp_config"

if [ -z "$gpu" ]; then
    gpu="0"
fi

if [ -z "$main_process_port" ]; then
    main_process_port=29500
fi
echo "Main Process Port: $main_process_port"

######## Features Extraction ###########
if [ $running_stage -eq 1 ]; then
    echo "Stage 1: Preprocessing for HiFi-GAN (MaskGCT) with HDF5 storage"
    echo "Sample Rate: 16000, Hop Size: 320, n_mel: 80"
    echo "Using HDF5 format for efficient feature storage and loading"
    CUDA_VISIBLE_DEVICES=$gpu python "${work_dir}"/bins/vocoder/preprocess_hdf5.py \
        --config $exp_config \
        --num_workers 8
fi

######## Training ###########
if [ $running_stage -eq 2 ]; then
    if [ -z "$exp_name" ]; then
        echo "[Error] Please specify the experiments name"
        exit 1
    fi
    echo "Exprimental Name: $exp_name"
    echo "Training HiFi-GAN for MaskGCT (16kHz, hop_size=320)"

    CUDA_VISIBLE_DEVICES=$gpu accelerate launch \
        --main_process_port "$main_process_port" \
        "${work_dir}"/bins/vocoder/train.py \
        --config "$exp_config" \
        --exp_name "$exp_name" \
        --log_level info \
        --checkpoint "$checkpoint" \
        --resume_type "$resume_type"
fi

######## Inference/Conversion ###########
if [ $running_stage -eq 3 ]; then
    if [ -z "$infer_expt_dir" ]; then
        echo "[Error] Please specify the experimental directionary. The value is like [Your path to save logs and checkpoints]/[YourExptName]"
        exit 1
    fi

    if [ -z "$infer_output_dir" ]; then
        infer_output_dir="$infer_expt_dir/result"
    fi

    if [ $infer_mode = "infer_from_dataset" ]; then
        CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/vocoder/inference.py \
            --config $exp_config \
            --infer_mode $infer_mode \
            --infer_datasets $infer_datasets \
            --vocoder_dir $infer_expt_dir \
            --output_dir $infer_output_dir  \
            --log_level debug
    fi

    if [ $infer_mode = "infer_from_feature" ]; then
        CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/vocoder/inference.py \
            --config $exp_config \
            --infer_mode $infer_mode \
            --feature_folder $infer_feature_dir \
            --vocoder_dir $infer_expt_dir \
            --output_dir $infer_output_dir  \
            --log_level debug
    fi

    if [ $infer_mode = "infer_from_audio" ]; then
        CUDA_VISIBLE_DEVICES=$gpu accelerate launch "$work_dir"/bins/vocoder/inference.py \
            --config $exp_config \
            --infer_mode $infer_mode \
            --audio_folder $infer_audio_dir \
            --vocoder_dir $infer_expt_dir \
            --output_dir $infer_output_dir  \
            --log_level debug
    fi
fi

echo "
============================================================
MaskGCT HiFi-GAN Vocoder Training/Inference Script

This HiFi-GAN is configured specifically for MaskGCT:
- Sample Rate: 16000 Hz
- Hop Size: 320 (matching semantic-to-mel model)
- n_mel: 80
- Upsample rates: [10, 8, 2, 2] -> 320 total

HDF5 Storage Configuration (in exp_config.json):
- samples_per_hdf5: Number of samples per HDF5 file (default: 10000)
- use_float16: Use float16 for storage (default: false)
- hdf5_compression: Compression algorithm (gzip/lzf/none)
- hdf5_compression_opts: Compression level 1-9 (default: 4)
- hdf5_cache_size: Number of HDF5 files to cache (default: 5)
- use_hdf5: Enable HDF5 storage (default: true)

Usage Examples:

1. Preprocess (with HDF5 storage):
   bash run.sh --stage 1 --gpu 0

2. Train:
   bash run.sh --stage 2 --name hifigan_maskgct --gpu 0

3. Inference from dataset:
   bash run.sh --stage 3 --gpu 0 \\
       --infer_mode infer_from_dataset \\
       --infer_datasets ljspeech \\
       --infer_expt_dir ckpts/vocoder/hifigan_maskgct

4. Inference from mel features (e.g., from semantic-to-mel output):
   bash run.sh --stage 3 --gpu 0 \\
       --infer_mode infer_from_feature \\
       --infer_feature_dir outputs/semantic_to_mel \\
       --infer_expt_dir ckpts/vocoder/hifigan_maskgct

============================================================
"
