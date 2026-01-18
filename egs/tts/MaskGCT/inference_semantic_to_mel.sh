export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES="1"

python egs/tts/MaskGCT/inference_semantic_to_mel.py \
      --checkpoint_path ckpts/maskgct_mini/semantic_to_mel_mini/checkpoint/epoch-0087_step-0018000_loss-0.211765 \
      --config_path egs/tts/MaskGCT/semantic_to_mel_mini.json \
      --input_audio data/LJSpeech-1.1/wavs/LJ001-0001.wav \
      --reference_audio data/LJSpeech-1.1/wavs/LJ001-0002.wav \
      --output_dir ckpts/maskgct_mini/semantic_to_mel_mini/result