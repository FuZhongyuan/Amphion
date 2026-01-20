export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES="0"


python models/tts/maskgct/s2mel_inference/s2mel_inference.py \
    --config egs/tts/MaskGCT/s2mel_dit_mini.json \
    --s2mel_ckpt ckpts/maskgct_mini/s2mel_dit_mini_backup/checkpoint/epoch-0033_step-0232000_loss-0.098823 \
    --vocoder_ckpt ckpts/vocoder/hifigan_maskgct/hifigan_maskgct/checkpoint/epoch-0550_step-0213788_loss-34.959272 \
    --vocoder_config egs/vocoder/gan/hifigan_maskgct/exp_config.json \
    --input data/LibriTTS/test-other/367/130732/367_130732_000000_000000.wav \
    --output_dir ckpts/maskgct_mini/s2mel_dit_mini_backup/result

# python models/tts/maskgct/s2mel_inference/s2mel_inference.py \
#     --config egs/tts/MaskGCT/s2mel_fm_mini.json \
#     --s2mel_ckpt ckpts/maskgct_mini/s2mel_fm_mini_backup/checkpoint/epoch-0031_step-0220000_loss-0.659933 \
#     --vocoder_ckpt ckpts/vocoder/hifigan_maskgct/hifigan_maskgct/checkpoint/epoch-0550_step-0213788_loss-34.959272 \
#     --vocoder_config egs/vocoder/gan/hifigan_maskgct/exp_config.json \
#     --input data/LibriTTS/test-other/367/130732/367_130732_000000_000000.wav \
#     --output_dir ckpts/maskgct_mini/s2mel_fm_mini_backup/result