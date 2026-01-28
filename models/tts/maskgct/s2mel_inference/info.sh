export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES="5"


# python models/tts/maskgct/s2mel_inference/s2mel_inference.py \
#     --config models/tts/maskgct/inference/s2mel_dit_inference.json \
#     --s2mel_ckpt ckpts/maskgct_mini/s2mel_dit_mini/checkpoint/epoch-0001_step-0006000_loss-0.141610 \
#     --vocoder_ckpt ckpts/vocoder/hifigan_libritts/checkpoint/epoch-0020_step-0058191_loss-34.994965 \
#     --vocoder_config egs/vocoder/gan/hifigan_maskgct/exp_config.json \
#     --input data/LibriTTS/test-other/367/130732/367_130732_000002_000000.wav \
#     --output_dir models/tts/maskgct/s2mel_inference/result_s2mel_dit

python models/tts/maskgct/s2mel_inference/s2mel_inference.py \
    --config models/tts/maskgct/inference/s2mel_fm_inference.json \
    --s2mel_ckpt ckpts/maskgct_mini/s2mel_fm_mini/checkpoint/epoch-0023_step-0070000_loss-0.594491 \
    --vocoder_ckpt ckpts/vocoder/hifigan_libritts/checkpoint/epoch-0020_step-0058191_loss-34.994965 \
    --vocoder_config egs/vocoder/gan/hifigan_maskgct/exp_config.json \
    --input data/LJSpeech-1.1/wavs/LJ001-0006.wav \
    --output_dir models/tts/maskgct/s2mel_inference/result_s2mel_fm

