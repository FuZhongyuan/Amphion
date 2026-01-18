# GlowTTS Recipe

This is the Amphion implementation of GlowTTS, a flow-based generative model for parallel text-to-speech synthesis.

## Model Overview

GlowTTS is based on the paper:
> **Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search**
> Jaehyeon Kim, Sungwon Kim, Jungil Kong, Sungroh Yoon
> NeurIPS 2020

### Key Features

- **Parallel Generation**: Unlike autoregressive models, GlowTTS generates mel-spectrograms in parallel
- **Monotonic Alignment Search (MAS)**: Uses dynamic programming to find optimal alignment between text and mel
- **Flow-based Decoder**: Uses normalizing flows for high-quality mel-spectrogram generation
- **Duration Prediction**: Explicitly models phoneme durations for controllable synthesis

### Architecture

```
Input Text → Text Encoder → Duration Predictor → Length Regulator
                ↓                                      ↓
           Prior (μ, σ)  ←←←←←←←←←←←←←←←←←←←←←←←←←   MAS
                ↓                                      ↑
         Flow Decoder  ←←←←←←←←← (reverse) ←←←←←←  Mel-spectrogram
```

## Dataset Preparation

### LJSpeech

1. Download [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/) dataset
2. Place it in `data/LJSpeech-1.1/`

### LibriTTS (Multi-speaker)

1. Download [LibriTTS](https://www.openslr.org/60/) dataset
2. Place it in `data/LibriTTS/`
3. Use `exp_config_libritts.json` for multi-speaker training

## Training

### Step 1: Build Monotonic Alignment Module

```bash
cd modules/monotonic_align
python setup.py build_ext --inplace
cd ../..
```

### Step 2: Feature Extraction

```bash
bash egs/tts/GlowTTS/run.sh --stage 1 --gpu 0
```

### Step 3: Training

```bash
bash egs/tts/GlowTTS/run.sh --stage 2 --name glowtts_ljspeech --gpu 0
```

For LibriTTS (multi-speaker):
```bash
bash egs/tts/GlowTTS/run.sh --stage 2 \
    --config egs/tts/GlowTTS/exp_config_libritts.json \
    --name glowtts_libritts \
    --gpu 0
```

## Inference

### Single Sentence

```bash
bash egs/tts/GlowTTS/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/glowtts_ljspeech \
    --infer_output_dir ckpts/tts/glowtts_ljspeech/result \
    --infer_mode "single" \
    --infer_text "Hello, this is a test of GlowTTS synthesis." \
    --vocoder_dir ckpts/vocoder/hifigan_ljspeech/checkpoints \
    --gpu 0
```

### Batch Inference

```bash
bash egs/tts/GlowTTS/run.sh --stage 3 \
    --infer_expt_dir ckpts/tts/glowtts_ljspeech \
    --infer_output_dir ckpts/tts/glowtts_ljspeech/result \
    --infer_mode "batch" \
    --infer_dataset LJSpeech \
    --infer_testing_set test \
    --vocoder_dir ckpts/vocoder/hifigan_ljspeech/checkpoints \
    --gpu 0
```

### Inference Parameters

- `--noise_scale`: Scale for sampling noise (default: 0.667). Lower values give more deterministic outputs.
- `--length_scale`: Scale for duration (default: 1.0). Values > 1.0 slow down speech, < 1.0 speed up.

## Configuration

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_channels` | 192 | Hidden dimension of encoder/decoder |
| `filter_channels` | 768 | Filter channels in encoder FFN |
| `n_heads` | 2 | Number of attention heads |
| `n_layers_enc` | 6 | Number of encoder layers |
| `n_blocks_dec` | 12 | Number of flow blocks in decoder |
| `kernel_size_dec` | 5 | Kernel size for decoder WaveNet |
| `n_sqz` | 2 | Squeeze factor for flow |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 32 | Training batch size |
| `lr` | 1.0 | Learning rate (with Noam scheduler) |
| `warmup` | 4000 | Warmup steps for Noam scheduler |
| `grad_clip` | 1.0 | Gradient clipping threshold |

## Pretrained Models

Coming soon...

## References

```bibtex
@inproceedings{kim2020glowtts,
  title={Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search},
  author={Kim, Jaehyeon and Kim, Sungwon and Kong, Jungil and Yoon, Sungroh},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

## License

This implementation is released under the MIT License.
