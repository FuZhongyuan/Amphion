export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES="2"

# Configuration
CFG_PATH="models/tts/maskgct/inference/s2mel_dit_inference.json"
SEMANTIC_CODEC_CKPT="ckpts/maskgct_mini/semantic_codec_mini_ljspeech/checkpoint/epoch-1476_step-0302000_loss-14.263197"
T2S_CKPT="ckpts/maskgct_mini/t2s_curriculum_ljspeech_org/checkpoint_backup/epoch-0107_step-0014000_loss-3.183115"
S2MEL_CKPT="ckpts/maskgct_mini/s2mel_dit_mini/checkpoint/epoch-0276_step-0036000_loss-0.129780"
VOCODER_CKPT="ckpts/vocoder/hifigan_maskgct/hifigan_maskgct/checkpoint/epoch-0550_step-0213788_loss-34.959272"

# Input
PROMPT_WAV="data/LJSpeech-1.1/wavs/LJ001-0016.wav"
PROMPT_TEXT="The Middle Ages brought calligraphy to perfection, and it was natural therefore."
TARGET_TEXT="The Middle Ages brought calligraphy to perfection, and it was natural therefore."

# Output
mkdir -p "ckpts/maskgct_mini/s2mel_dit_mini/outputs"
OUTPUT_PATH="ckpts/maskgct_mini/s2mel_dit_mini/outputs/output_s2mel_dit.wav"

python models/tts/maskgct/inference/s2mel_dit_inference.py \
    --cfg_path ${CFG_PATH} \
    --semantic_codec_ckpt ${SEMANTIC_CODEC_CKPT} \
    --t2s_ckpt ${T2S_CKPT} \
    --s2mel_ckpt ${S2MEL_CKPT} \
    --vocoder_ckpt ${VOCODER_CKPT} \
    --prompt_wav ${PROMPT_WAV} \
    --prompt_text "${PROMPT_TEXT}" \
    --target_text "${TARGET_TEXT}" \
    --output_path ${OUTPUT_PATH} \
    --device cuda:0
