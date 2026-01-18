# MaskGCT Project Structure and Workflow Documentation

## Project Overview

MaskGCT (Masked Generative Codec Transformer) is a fully non-autoregressive TTS model implemented in the Amphion project. This document provides a comprehensive guide to the MaskGCT implementation, including project structure, training workflows, and inference processes.

**IMPORTANT NOTES:**
1. MaskGCT has its own **dedicated training scripts, configuration formats, and dataset loaders** that are different from other TTS models in Amphion
2. **DO NOT confuse** MaskGCT implementations with other models like VITS, FastSpeech2, NaturalSpeech2, etc.
3. MaskGCT uses specialized components that should be used consistently throughout the pipeline

---

## 1. Complete Project Structure

### 1.1 Core Training Components

```
Amphion/
├── bins/tts/
│   └── maskgct_train.py                    # Main training factory for ALL MaskGCT components
│
├── models/tts/maskgct/                     # MaskGCT model implementations
│   ├── semantic_codec_trainer.py           # Trainer for semantic codec
│   ├── acoustic_codec_trainer.py           # Trainer for acoustic codec(unused)
│   ├── t2s_trainer.py                      # Trainer for T2S (Text-to-Semantic)
│   ├── s2a_trainer.py                      # Trainer for S2A (Semantic-to-Acoustic)(unused)
│   ├── s2mel_dit_trainer.py          # Trainer for Semantic-to-Mel (diffusion)
│   ├── s2mel_fm_trainer.py                 # Trainer for Semantic-to-Mel (flow matching)
│   ├── t2s_finetune_trainer.py             # Trainer for T2S fine-tuning
│   ├── t2s_curriculum_trainer.py           # Trainer for T2S curriculum learning
│   │
│   ├── maskgct_t2s.py                      # T2S model architecture
│   ├── maskgct_s2a.py                      # S2A model architecture(unused)
│   ├── maskgct_s2mel_dit.py                # Semantic-to-Mel (diffusion) architecture
│   ├── maskgct_s2mel_fm.py                 # Semantic-to-Mel (flow matching) architecture
│   ├── maskgct_t2s_curriculum.py           # T2S curriculum learning model
│   ├── llama_nar.py                        # Non-autoregressive LLaMA backbone
│   │
│   ├── maskgct_inference.py                # Main inference script(unused)
│   ├── s2mel_dit_inference.py              # Semantic-to-Mel (diffusion) inference script
│   ├── run_maskgct_mel_dit_inference.sh    # Semantic-to-Mel (diffusion) inference runner
│   ├── s2mel_fm_inference.py                # Semantic-to-Mel (flow matching) inference script
│   ├── run_maskgct_mel_fm_inference.sh    # Semantic-to-Mel (flow matching) inference runner
│   ├── maskgct_utils.py                    # Utility functions for inference
│   ├── gradio_demo.py                      # Gradio web interface
│   ├── maskgct_demo.ipynb                  # Jupyter notebook demo
│   │
│   ├── g2p/                                # Grapheme-to-Phoneme conversion
│   │   └── g2p_generation.py
│   │
│   ├── config/
│   │   └── maskgct.json                    # Default inference configuration
│   │
│   ├── preprocess_ljspeech_semantic.py     # Preprocessing script for LJSpeech
│   ├── preprocess_libritts_semantic.py     # Preprocessing script for LibriTTS
│   ├── run_preprocessing.sh                # Preprocessing runner
│   │
│   └── README.md                           # MaskGCT documentation
│
├── models/tts/base/                        # Dataset loaders (MaskGCT-specific)
│   ├── dataset_factory.py                  # Factory for selecting dataset type
│   ├── maskgct_ljspeech_dataset.py         # LJSpeech dataset loader
│   ├── maskgct_libritts_dataset.py         # LibriTTS dataset loader
│   └── maskgct_emilia_dataset.py           # Emilia dataset loader
│
├── models/codec/                           # Codec implementations
│   ├── kmeans/
│   │   └── repcodec_model.py               # Semantic codec (RepCodec)
│   └── amphion_codec/
│       └── codec.py                        # Acoustic codec (encoder & decoder)
│
└── models/vocoders/gan/
    └── gan_vocoder_trainer.py              # HiFi-GAN vocoder trainer (optional)
```

### 1.2 Configuration Files

```
Amphion/
└── egs/tts/MaskGCT/                        # Training configurations
    ├── semantic_codec.json                 # Semantic codec training config (full)
    ├── semantic_codec_mini.json            # Semantic codec training config (mini)
    ├── acoustic_codec.json                 # Acoustic codec training config (full)(unused)
    ├── acoustic_codec_mini.json            # Acoustic codec training config (mini)(unused)
    ├── t2s.json                            # T2S training config (full)
    ├── t2s_mini.json                       # T2S training config (mini)
    ├── t2s_curriculum.json                 # T2S curriculum learning config
    ├── t2s_finetune.json                   # T2S fine-tuning config
    ├── s2a.json                            # S2A training config (full)(unused)
    ├── s2a_mini.json                       # S2A training config (mini)(unused)
    ├── s2mel_dit.json                # Semantic-to-Mel (diffusion) config
    ├── s2mel_dit_mini.json           # Semantic-to-Mel mini config
    ├── s2mel_fm.json                       # Semantic-to-Mel (flow matching) config
    ├── s2mel_fm_mini.json                  # Semantic-to-Mel FM mini config
    │
    ├── train_semantic_codec.sh             # Training scripts
    ├── train_semantic_codec_mini.sh
    ├── train_acoustic_codec.sh             # (unused)
    ├── train_acoustic_codec_mini.sh        # (unused)
    ├── train_t2s.sh
    ├── train_t2s_mini.sh
    ├── train_t2s_curriculum.sh
    ├── train_t2s_finetune.sh
    ├── train_s2a.sh                       # (unused)
    ├── train_s2a_mini.sh                  # (unused)
    ├── train_s2mel_dit.sh
    ├── train_s2mel_dit_mini.sh
    ├── train_s2mel_fm.sh
    └── train_s2mel_fm_mini.sh
```

### 1.3 Supporting Components

```
Amphion/
├── models/base/
│   └── base_trainer.py                     # Base trainer class (can be reused)
│
├── utils/                                  # Utility functions (can be reused)
│   ├── util.py                             # Config loading, logging
│   ├── mel.py                              # Mel spectrogram extraction
│   └── audio.py                            # Audio processing utilities
│
├── modules/                                # Reusable neural network modules
│   ├── transformer/                        # Transformer blocks
│   ├── attention/                          # Attention mechanisms
│   └── encoder_decoder/                    # Encoder-decoder architectures
│
└── text/                                   # Text processing (can be reused)
    └── g2p/                                # Grapheme-to-phoneme conversion
```

---

## 2. MaskGCT Training Pipeline

### 2.1 Overview

MaskGCT training consists of **4 sequential stages**, each building upon the previous:

```
Stage 1: Semantic Codec Training
    ↓ (produces semantic tokens)
Stage 2: T2S Model Training
    ↓ (predicts semantic tokens from text)
Stage 3: Semantic-to-Mel (S2Mel) Model Training
    ↓ (predicts mel from semantic tokens)
Final: Vocoder (HiFi-GAN) or direct mel-to-wav
```

**Alternative Pipeline (Flow Matching):**
```
Stage 1: Semantic Codec Training
    ↓
Stage 3: T2S Model Training
    ↓
Stage 3.5: Semantic-to-Mel (S2Mel) Training with Flow Matching
    ↓
Final: Vocoder (HiFi-GAN)
```

### 2.2 Stage-by-Stage Training Process

#### Stage 1: Semantic Codec Training

**Purpose:** Train a model to quantize continuous semantic features (from w2v-bert-2.0) into discrete semantic tokens.

**Input:** 
- Raw audio waveforms (16kHz)
- Extracted w2v-bert-2.0 features (layer 17, 1024-dim)

**Output:**
- Semantic codec model checkpoint
- Codebook (512-8192 tokens)

---

#### Stage 2: T2S (Text-to-Semantic) Training

**Purpose:** Train model to predict semantic tokens from phoneme sequences using masked language modeling.

**Prerequisites:**
- Trained semantic codec from Stage 1

**Input:**
- Phoneme IDs (from G2P conversion)
- Ground truth semantic tokens (from semantic codec)

**Output:**
- T2S model checkpoint

---

#### Stage 3: S2mel Training 

---

### 2.3 Training Factory System

**Central Training Entry Point:** `bins/tts/maskgct_train.py`

This factory automatically selects the correct trainer based on `model_type` in the configuration.

---

### 2.4 Dataset Configuration

**Supported Datasets:**
1. **LibriTTS** (default): Multi-speaker English audiobook dataset
2. **LJSpeech-1.1**: Single-speaker English dataset
3. **Emilia**: Large-scale multilingual dataset (100K hours)

**Dataset Loaders:**
- `models/tts/base/maskgct_libritts_dataset.py`
- `models/tts/base/maskgct_ljspeech_dataset.py`
- `models/tts/base/maskgct_emilia_dataset.py`

---

## 3. MaskGCT Inference Pipeline

Input Text → G2P → Phone IDs → (T2S Model → Semantic Tokens) → S2Mel-FM Model/S2Mel-DIT Model → Mel Spectrogram → Vocoder → Waveform

---

## 4. Key Design Principles

### 4.1 MaskGCT-Specific Features

**1. Masked Generative Training:**
- Uses mask-and-predict paradigm (not autoregressive)
- Dynamic masking ratio during training
- Parallel generation during inference

**2. Classifier-Free Guidance (CFG):**
- Random dropout of conditioning (text/prompt) during training
- Enhanced controllability during inference
- CFG scale tunable at inference time

**3. Prompt-Based Voice Cloning:**
- Uses reference audio for zero-shot voice cloning
- Prompt semantic tokens guide speaking style
- Prompt Mel guide timbre/prosody

**4. Hierarchical Token Representation:**
- Semantic tokens: linguistic content (text → semantic)
- Mel: acoustic details (semantic → Mel)
- Separation enables better generalization

### 4.2 Reusable Components

**ALL Functions in Amphion**

**MaskGCT-Specific (DO NOT mix with other models):**
- All trainers in `models/tts/maskgct/`
- All model architectures in `models/tts/maskgct/`
- Dataset loaders in `models/tts/base/maskgct_*_dataset.py`
- Training configurations in `egs/tts/MaskGCT/`

## 5. Summary

MaskGCT is a **multi-stage, non-autoregressive TTS system** with the following key characteristics:

1. **Dedicated Infrastructure:**
   - Unique training factory (`bins/tts/maskgct_train.py`)
   - Specialized trainers for each component
   - Custom dataset loaders
   - Specific configuration format

2. **Training Pipeline:**
   - 4 sequential stages (Semantic Codec → T2S → S2mel)
   - Each stage depends on previous checkpoints

3. **Reusable Components:**
   - Base trainer class
   - Utility functions (mel, audio, config loading)
   - Neural network modules (transformer, attention)
   - Can be adapted for similar codec-based TTS systems

4. **Best Practices:**
   - Always use the correct dataset loader for your dataset
   - Train stages in order
   - Use mini configs for limited GPU memory
   - Pre-extract semantic features for faster training
   - Monitor metrics appropriate for each stage

5. **DO NOT:**
   - Mix MaskGCT configs with other TTS models
   - Skip training stages
   - Use inconsistent sample rates across stages
   - Modify dataset factory without proper registration

This documentation provides a complete reference for working with MaskGCT in the Amphion project. 
