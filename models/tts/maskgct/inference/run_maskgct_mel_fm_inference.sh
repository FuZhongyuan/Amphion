export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES="5"

# Configuration
CFG_PATH="models/tts/maskgct/inference/s2mel_fm_inference.json"
SEMANTIC_CODEC_CKPT="ckpts/maskgct_mini/semantic_codec_mini_160k_128/checkpoint/epoch-0170_step-0160000_loss-43.150742"
T2S_CKPT="ckpts/maskgct_mini/t2s_curriculum_180k/checkpoint/epoch-0039_step-0180000_loss-3.452544"
S2MEL_CKPT="ckpts/maskgct_mini/s2mel_fm_mini/checkpoint/epoch-0023_step-0070000_loss-0.594491"
VOCODER_CKPT="ckpts/vocoder/hifigan_libritts/checkpoint/epoch-0020_step-0058191_loss-34.994965"

# Input
PROMPT_WAV="data/LibriTTS/dev-other/700/122866/700_122866_000002_000001.wav"
PROMPT_TEXT="To Anne in particular things seemed fearfully flat, stale, and unprofitable after the goblet of excitement she had been sipping for weeks."
TARGET_TEXT="To Anne in particular things seemed fearfully flat, stale, and unprofitable after the goblet of excitement she had been sipping for weeks."

# Output
mkdir -p "models/tts/maskgct/inference/result_s2mel_fm"
OUTPUT_PATH="models/tts/maskgct/inference/result_s2mel_fm/output_s2mel_fm.wav"

python models/tts/maskgct/inference/s2mel_fm_inference.py \
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
