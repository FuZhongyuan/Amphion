export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8
export CUDA_VISIBLE_DEVICES="3"

python models/tts/maskgct/maskgct_inference.py \
  --cfg_path ./models/tts/maskgct/config/maskgct.json \
  --ckpt_dir ./ckpts/MaskGCT-ckpt \
  --prompt_wav_path ./models/tts/maskgct/wav/prompt.wav \
  --prompt_text "We do not break. We never give in. We never back down." \
  --target_text "We do not break. We never give in. We never back down." \
  --save_path ckpts/MaskGCT-ckpt/output/maskgct_generated_audio_t2s_64k.wav \
  --target_len 6.3 \
  --device cuda:0 \
  --log_level debug \
  --debug \
  --semantic_codec_ckpt ckpts/MaskGCT-ckpt/semantic_codec/model.safetensors \
  --codec_encoder_ckpt ckpts/MaskGCT-ckpt/acoustic_codec/model.safetensors \
  --codec_decoder_ckpt ckpts/MaskGCT-ckpt/acoustic_codec/model_1.safetensors \
  --t2s_model_ckpt ckpts/MaskGCT-ckpt/t2s_model/model.safetensors \
  --s2a_1layer_ckpt ckpts/MaskGCT-ckpt/s2a_model/s2a_model_1layer/model.safetensors \
  --s2a_full_ckpt ckpts/MaskGCT-ckpt/s2a_model/s2a_model_full/model.safetensors

  # --target_text "In this paper, we introduce MaskGCT, a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision." \