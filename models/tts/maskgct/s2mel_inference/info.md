  1. Supports both DiT and FM models - Auto-detects model type from config or accepts --model_type argument
  2. Simulates training logic:
    - Extracts semantic tokens from audio using w2v-bert-2.0 + semantic codec
    - Uses a portion of the audio as prompt (configurable via --prompt_ratio)
    - Generates mel spectrograms from semantic tokens using the S2Mel model
    - Converts mel to audio using HiFi-GAN vocoder
  3. Saves comprehensive outputs:
    - {sample}_generated.wav - Generated audio
    - {sample}_reconstructed.wav - GT mel through vocoder (for comparison)
    - {sample}_gt.wav - Original ground truth audio
    - {sample}_generated_mel.npy - Generated mel spectrogram
    - {sample}_gt_mel.npy - Ground truth mel spectrogram
    - {sample}_metadata.json - Metadata (lengths, shapes)

  Usage Example:

  # For DiT model
  python models/tts/maskgct/s2mel_inference/s2mel_inference.py \
      --config egs/tts/MaskGCT/s2mel_dit_mini.json \
      --s2mel_ckpt ckpts/maskgct_mini/s2mel_dit_mini_backup/checkpoint/epoch-0033_step-0232000_loss-0.098823 \
      --vocoder_ckpt ckpts/vocoder/hifigan_maskgct/hifigan_maskgct/checkpoint/epoch-0550_step-0213788_loss-34.959272 \
      --vocoder_config egs/vocoder/gan/hifigan_maskgct/exp_config.json \
      --input data/LibriTTS/test-other/367/130732/367_130732_000000_000000.wav \
      --output_dir ckpts/maskgct_mini/s2mel_dit_mini_backup/result

  # For FM model
  python models/tts/maskgct/s2mel_inference/s2mel_inference.py \
      --config egs/tts/MaskGCT/s2mel_fm_mini.json \
      --s2mel_ckpt ckpts/maskgct_mini/s2mel_fm_mini_backup/checkpoint/epoch-0031_step-0220000_loss-0.659933 \
      --vocoder_ckpt ckpts/vocoder/hifigan_maskgct/hifigan_maskgct/checkpoint/epoch-0550_step-0213788_loss-34.959272 \
      --vocoder_config egs/vocoder/gan/hifigan_maskgct/exp_config.json \
      --input data/LibriTTS/test-other/367/130732/367_130732_000000_000000.wav \
      --output_dir ckpts/maskgct_mini/s2mel_fm_mini_backup/result

  Key Arguments:
  - --prompt_ratio: Ratio of audio to use as prompt (default: 0.3)
  - --n_timesteps: Diffusion/flow steps (default: 50 for DiT, 10 for FM)
  - --cfg_scale: Classifier-free guidance scale (default: 1.0)
  - --input: Can be a single file or directory of audio files