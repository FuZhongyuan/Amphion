# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Flow Matching Semantic-to-Mel Inference Script.

This script demonstrates inference using the Flow Matching S2Mel model.
It generates mel spectrograms from semantic tokens with prompt-based voice cloning
and saves them as image files for visualization.

Pipeline:
1. Prompt Audio -> Semantic Tokens (w2v-bert-2.0 + RepCodec)
2. Semantic Tokens + Prompt Mel -> Target Mel Spectrogram (S2Mel Flow Matching)
3. Mel Spectrogram -> Visualization (matplotlib)
4. (Optional) Mel Spectrogram -> Waveform (HiFi-GAN vocoder)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
import argparse
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.util import load_config, Logger
from utils.mel import extract_mel_features

from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.maskgct.maskgct_s2mel_fm import FlowMatchingS2Mel, FlowMatchingS2MelWithPrompt
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor


def build_semantic_model(device, stat_path="./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt"):
    """Build w2v-bert-2.0 semantic model."""
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)

    stat_mean_var = torch.load(stat_path, weights_only=False)
    semantic_mean = stat_mean_var["mean"].to(device)
    semantic_std = torch.sqrt(stat_mean_var["var"]).to(device)

    return semantic_model, semantic_mean, semantic_std


def build_semantic_codec(cfg, device):
    """Build semantic codec (RepCodec)."""
    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec


def build_s2mel_fm_model(cfg, device):
    """Build Flow Matching Semantic-to-Mel model."""
    use_prompt = getattr(cfg, "use_prompt_model", False)

    if use_prompt:
        model = FlowMatchingS2MelWithPrompt(cfg=cfg)
    else:
        model = FlowMatchingS2Mel(cfg=cfg)

    model.eval()
    model.to(device)
    return model


def load_checkpoint(model, ckpt_path, device):
    """Load checkpoint for a model."""
    import safetensors

    if os.path.isdir(ckpt_path):
        # Accelerate checkpoint format
        model_path = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.exists(model_path):
            model_path = os.path.join(ckpt_path, "pytorch_model.bin")
    else:
        model_path = ckpt_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    if model_path.endswith(".safetensors"):
        safetensors.torch.load_model(model, model_path)
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"], strict=False)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    return model


def save_mel_spectrogram_image(
    mel,
    save_path,
    title=None,
    figsize=(12, 4),
    cmap='viridis',
    dpi=150,
    sample_rate=16000,
    hop_size=320,
):
    """
    Save mel spectrogram as image file.

    Args:
        mel: Mel spectrogram (T, n_mel) or (B, T, n_mel)
        save_path: Path to save image
        title: Optional title for the plot
        figsize: Figure size
        cmap: Colormap for spectrogram
        dpi: Image resolution
        sample_rate: Audio sample rate
        hop_size: Hop size for mel extraction
    """
    if isinstance(mel, torch.Tensor):
        mel = mel.cpu().numpy()

    # Handle batch dimension
    if len(mel.shape) == 3:
        mel = mel[0]  # Take first sample in batch

    # mel shape: (T, n_mel) -> transpose for visualization
    mel = mel.T  # (n_mel, T)

    fig, ax = plt.subplots(figsize=figsize)

    # Create time axis in seconds
    duration = mel.shape[1] * hop_size / sample_rate
    extent = [0, duration, 0, mel.shape[0]]

    img = ax.imshow(
        mel,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap=cmap,
    )

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency Bin')
    if title:
        ax.set_title(title)

    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_mel_comparison_image(
    mel_list,
    titles,
    save_path,
    figsize=None,
    cmap='viridis',
    dpi=150,
    sample_rate=16000,
    hop_size=320,
):
    """
    Save multiple mel spectrograms side by side for comparison.

    Args:
        mel_list: List of mel spectrograms
        titles: List of titles for each spectrogram
        save_path: Path to save image
        figsize: Figure size (auto-computed if None)
        cmap: Colormap for spectrogram
        dpi: Image resolution
        sample_rate: Audio sample rate
        hop_size: Hop size for mel extraction
    """
    n_plots = len(mel_list)
    if figsize is None:
        figsize = (5 * n_plots, 4)

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    for i, (mel, title) in enumerate(zip(mel_list, titles)):
        if isinstance(mel, torch.Tensor):
            mel = mel.cpu().numpy()

        if len(mel.shape) == 3:
            mel = mel[0]

        mel = mel.T  # (n_mel, T)

        duration = mel.shape[1] * hop_size / sample_rate
        extent = [0, duration, 0, mel.shape[0]]

        img = axes[i].imshow(
            mel,
            aspect='auto',
            origin='lower',
            extent=extent,
            cmap=cmap,
        )

        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Mel Frequency Bin')
        axes[i].set_title(title)
        plt.colorbar(img, ax=axes[i], format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


class S2MelFMInferencePipeline:
    """
    Inference pipeline for Flow Matching Semantic-to-Mel model.

    This pipeline extracts semantic tokens from audio and generates
    mel spectrograms using Flow Matching.
    """

    def __init__(
        self,
        semantic_model,
        semantic_codec,
        s2mel_fm_model,
        semantic_mean,
        semantic_std,
        preprocess_cfg,
        device,
        vocoder=None,
    ):
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.s2mel_fm_model = s2mel_fm_model
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.preprocess_cfg = preprocess_cfg
        self.device = device
        self.vocoder = vocoder

        # Mel spectrogram parameters
        self.sample_rate = getattr(preprocess_cfg, "sample_rate", 16000)
        self.hop_size = getattr(preprocess_cfg, "hop_size", 320)
        self.n_mel = getattr(preprocess_cfg, "n_mel", 100)

    @torch.no_grad()
    def extract_features(self, speech):
        """Extract features for w2v-bert-2.0."""
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"][0]
        attention_mask = inputs["attention_mask"][0]
        return input_features, attention_mask

    @torch.no_grad()
    def extract_semantic_code(self, input_features, attention_mask):
        """Extract semantic tokens from audio features."""
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        semantic_code, rec_feat = self.semantic_codec.quantize(feat)  # (B, T)
        return semantic_code, rec_feat

    @torch.no_grad()
    def extract_mel_spectrogram(self, speech):
        """Extract mel spectrogram from audio."""
        speech_tensor = torch.tensor(speech).float().unsqueeze(0).to(self.device)

        mel = extract_mel_features(speech_tensor, self.preprocess_cfg, center=False)

        # mel shape: (B, n_mel, T) -> (B, T, n_mel)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.transpose(1, 2)

        return mel

    @torch.no_grad()
    def semantic_to_mel(
        self,
        semantic_tokens,
        prompt_mel,
        prompt_semantic_tokens=None,
        n_timesteps=20,
        cfg=1.0,
        rescale_cfg=0.75,
        temperature=1.0,
    ):
        """
        Generate mel spectrogram from semantic tokens.

        Args:
            semantic_tokens: Semantic tokens for target generation (B, T)
            prompt_mel: Prompt mel spectrogram (B, T_prompt, mel_dim)
            prompt_semantic_tokens: Semantic tokens for prompt (B, T_prompt)
            n_timesteps: Number of ODE solver steps
            cfg: CFG scale
            rescale_cfg: CFG rescaling factor
            temperature: Initial noise temperature

        Returns:
            mel: Generated mel spectrogram (B, T, mel_dim)
        """
        if isinstance(self.s2mel_fm_model, FlowMatchingS2MelWithPrompt) and prompt_semantic_tokens is not None:
            # Use prompt-based generation
            mel = self.s2mel_fm_model.generate(
                semantic_tokens=semantic_tokens,
                prompt_mel=prompt_mel,
                prompt_semantic_tokens=prompt_semantic_tokens,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
                temperature=temperature,
            )
        else:
            # Use base model with concatenated semantic tokens
            full_semantic = torch.cat([prompt_semantic_tokens, semantic_tokens], dim=1) if prompt_semantic_tokens is not None else semantic_tokens
            mel = self.s2mel_fm_model.reverse_diffusion(
                semantic_tokens=full_semantic,
                prompt_mel=prompt_mel,
                n_timesteps=n_timesteps,
                cfg=cfg,
                rescale_cfg=rescale_cfg,
            )

        return mel

    @torch.no_grad()
    def mel_to_audio(self, mel):
        """Convert mel spectrogram to audio using vocoder."""
        if self.vocoder is None:
            raise ValueError("Vocoder not provided. Cannot convert mel to audio.")

        # mel shape: (B, T, n_mel) -> (B, n_mel, T)
        mel = mel.transpose(1, 2)

        # Generate audio
        audio = self.vocoder(mel)

        # audio shape: (B, 1, T) -> (B, T)
        audio = audio.squeeze(1)

        return audio

    def inference(
        self,
        prompt_speech_path,
        target_speech_path=None,
        n_timesteps=20,
        cfg=1.0,
        rescale_cfg=0.75,
        temperature=1.0,
    ):
        """
        Run inference to generate mel spectrogram.

        If target_speech_path is provided, extract target semantic tokens from it.
        Otherwise, duplicate the prompt semantic tokens for demonstration.

        Args:
            prompt_speech_path: Path to prompt audio file
            target_speech_path: Path to target audio (for extracting semantic tokens)
            n_timesteps: Number of ODE solver steps
            cfg: CFG scale
            rescale_cfg: CFG rescaling factor
            temperature: Initial noise temperature

        Returns:
            generated_mel: Generated mel spectrogram (B, T, mel_dim)
            prompt_mel: Prompt mel spectrogram (B, T_prompt, mel_dim)
            target_mel: Target mel spectrogram if target provided (B, T, mel_dim)
            prompt_semantic: Prompt semantic tokens (B, T_prompt)
            target_semantic: Target semantic tokens (B, T)
        """
        # Load prompt audio
        print("Loading prompt audio...")
        prompt_speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]

        # Extract prompt semantic tokens
        print("Extracting prompt semantic tokens...")
        prompt_input_features, prompt_attention_mask = self.extract_features(prompt_speech_16k)
        prompt_input_features = prompt_input_features.unsqueeze(0).to(self.device)
        prompt_attention_mask = prompt_attention_mask.unsqueeze(0).to(self.device)
        prompt_semantic, _ = self.extract_semantic_code(prompt_input_features, prompt_attention_mask)
        print(f"  Prompt semantic shape: {prompt_semantic.shape}")

        # Extract prompt mel spectrogram
        print("Extracting prompt mel spectrogram...")
        prompt_mel = self.extract_mel_spectrogram(prompt_speech_16k)
        print(f"  Prompt mel shape: {prompt_mel.shape}")

        # Extract target semantic tokens
        target_mel = None
        if target_speech_path is not None:
            print("Loading target audio...")
            target_speech_16k = librosa.load(target_speech_path, sr=16000)[0]

            print("Extracting target semantic tokens...")
            target_input_features, target_attention_mask = self.extract_features(target_speech_16k)
            target_input_features = target_input_features.unsqueeze(0).to(self.device)
            target_attention_mask = target_attention_mask.unsqueeze(0).to(self.device)
            target_semantic, _ = self.extract_semantic_code(target_input_features, target_attention_mask)
            print(f"  Target semantic shape: {target_semantic.shape}")

            # Extract target mel for comparison
            print("Extracting target mel spectrogram...")
            target_mel = self.extract_mel_spectrogram(target_speech_16k)
            print(f"  Target mel shape: {target_mel.shape}")
        else:
            # Use prompt semantic as target for demonstration
            print("No target audio provided. Using prompt semantic tokens as target...")
            target_semantic = prompt_semantic

        # Align mel and semantic lengths
        # Semantic tokens: ~50 Hz, Mel: sample_rate / hop_size Hz
        semantic_rate = 50
        mel_rate = self.sample_rate / self.hop_size
        prompt_semantic_len = prompt_semantic.shape[1]
        target_semantic_len = target_semantic.shape[1]

        # Interpolate prompt mel to match prompt semantic length
        expected_prompt_mel_len = int(prompt_semantic_len * mel_rate / semantic_rate)
        if prompt_mel.shape[1] != expected_prompt_mel_len:
            print(f"  Aligning prompt mel from {prompt_mel.shape[1]} to {expected_prompt_mel_len}")
            prompt_mel = F.interpolate(
                prompt_mel.transpose(1, 2), size=expected_prompt_mel_len, mode='linear', align_corners=False
            ).transpose(1, 2)

        # Generate mel spectrogram
        print("Generating mel spectrogram using Flow Matching...")
        generated_mel = self.semantic_to_mel(
            semantic_tokens=target_semantic,
            prompt_mel=prompt_mel,
            prompt_semantic_tokens=prompt_semantic,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            temperature=temperature,
        )
        print(f"  Generated mel shape: {generated_mel.shape}")

        return generated_mel, prompt_mel, target_mel, prompt_semantic, target_semantic


def main():
    parser = argparse.ArgumentParser(description="Flow Matching S2Mel Inference")

    # Config paths
    parser.add_argument("--config", type=str, required=True,
                        help="Path to S2Mel FM config (e.g., s2mel_fm.json or s2mel_fm_mini.json)")
    parser.add_argument("--semantic_stat_path", type=str,
                        default="./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
                        help="Path to semantic model statistics")

    # Checkpoint paths
    parser.add_argument("--semantic_codec_ckpt", type=str, required=True,
                        help="Path to semantic codec checkpoint")
    parser.add_argument("--s2mel_fm_ckpt", type=str, required=True,
                        help="Path to S2Mel Flow Matching checkpoint")
    parser.add_argument("--vocoder_ckpt", type=str, default=None,
                        help="Path to HiFi-GAN vocoder checkpoint (optional)")
    parser.add_argument("--vocoder_config", type=str, default=None,
                        help="Path to HiFi-GAN vocoder config (optional)")

    # Input/Output
    parser.add_argument("--prompt_wav_path", type=str, required=True,
                        help="Path to prompt audio file")
    parser.add_argument("--target_wav_path", type=str, default=None,
                        help="Path to target audio (for extracting semantic tokens)")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save outputs")

    # Generation parameters
    parser.add_argument("--n_timesteps", type=int, default=20,
                        help="Number of ODE solver steps")
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="CFG scale")
    parser.add_argument("--rescale_cfg", type=float, default=0.75,
                        help="CFG rescale factor")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Initial noise temperature")

    # Visualization options
    parser.add_argument("--save_mel_image", action="store_true", default=True,
                        help="Save mel spectrogram as image")
    parser.add_argument("--colormap", type=str, default="viridis",
                        choices=["viridis", "magma", "inferno", "plasma", "cividis"],
                        help="Colormap for mel visualization")
    parser.add_argument("--dpi", type=int, default=150,
                        help="Image resolution (DPI)")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logger
    log_file = os.path.join(args.output_dir, "s2mel_fm_inference.log")
    logger = Logger(log_file, level=args.log_level).logger

    logger.info("=" * 60)
    logger.info("||  Flow Matching S2Mel Inference Started  ||")
    logger.info("=" * 60)

    device = torch.device(args.device)

    # Load config
    logger.info("Loading configuration...")
    cfg = load_config(args.config)

    # Build models
    logger.info("Building semantic model (w2v-bert-2.0)...")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        device, args.semantic_stat_path
    )

    logger.info("Building semantic codec...")
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    load_checkpoint(semantic_codec, args.semantic_codec_ckpt, device)

    logger.info("Building S2Mel Flow Matching model...")
    s2mel_fm_model = build_s2mel_fm_model(cfg.model.s2mel_fm, device)
    load_checkpoint(s2mel_fm_model, args.s2mel_fm_ckpt, device)

    # Optionally build vocoder
    vocoder = None
    if args.vocoder_ckpt and args.vocoder_config:
        logger.info("Building HiFi-GAN vocoder...")
        from models.vocoders.gan.generator.hifigan import HiFiGAN
        vocoder_cfg = load_config(args.vocoder_config)
        vocoder = HiFiGAN(vocoder_cfg)
        vocoder.eval()
        vocoder.to(device)
        load_checkpoint(vocoder, args.vocoder_ckpt, device)

    logger.info("All models loaded successfully!")

    # Create inference pipeline
    logger.info("Creating inference pipeline...")
    pipeline = S2MelFMInferencePipeline(
        semantic_model=semantic_model,
        semantic_codec=semantic_codec,
        s2mel_fm_model=s2mel_fm_model,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        preprocess_cfg=cfg.preprocess,
        device=device,
        vocoder=vocoder,
    )

    # Run inference
    logger.info("Starting inference...")
    logger.info(f"Prompt audio: {args.prompt_wav_path}")
    if args.target_wav_path:
        logger.info(f"Target audio: {args.target_wav_path}")

    generated_mel, prompt_mel, target_mel, prompt_semantic, target_semantic = pipeline.inference(
        prompt_speech_path=args.prompt_wav_path,
        target_speech_path=args.target_wav_path,
        n_timesteps=args.n_timesteps,
        cfg=args.cfg,
        rescale_cfg=args.rescale_cfg,
        temperature=args.temperature,
    )

    # Save mel spectrogram images
    if args.save_mel_image:
        logger.info("Saving mel spectrogram images...")

        # Save individual mel spectrograms
        prompt_mel_path = os.path.join(args.output_dir, "prompt_mel.png")
        save_mel_spectrogram_image(
            prompt_mel, prompt_mel_path,
            title="Prompt Mel Spectrogram",
            cmap=args.colormap, dpi=args.dpi,
            sample_rate=pipeline.sample_rate, hop_size=pipeline.hop_size,
        )
        logger.info(f"  Saved: {prompt_mel_path}")

        generated_mel_path = os.path.join(args.output_dir, "generated_mel.png")
        save_mel_spectrogram_image(
            generated_mel, generated_mel_path,
            title="Generated Mel Spectrogram",
            cmap=args.colormap, dpi=args.dpi,
            sample_rate=pipeline.sample_rate, hop_size=pipeline.hop_size,
        )
        logger.info(f"  Saved: {generated_mel_path}")

        if target_mel is not None:
            target_mel_path = os.path.join(args.output_dir, "target_mel.png")
            save_mel_spectrogram_image(
                target_mel, target_mel_path,
                title="Target (Ground Truth) Mel Spectrogram",
                cmap=args.colormap, dpi=args.dpi,
                sample_rate=pipeline.sample_rate, hop_size=pipeline.hop_size,
            )
            logger.info(f"  Saved: {target_mel_path}")

            # Save comparison image
            comparison_path = os.path.join(args.output_dir, "mel_comparison.png")
            save_mel_comparison_image(
                [prompt_mel, target_mel, generated_mel],
                ["Prompt", "Target (GT)", "Generated"],
                comparison_path,
                cmap=args.colormap, dpi=args.dpi,
                sample_rate=pipeline.sample_rate, hop_size=pipeline.hop_size,
            )
            logger.info(f"  Saved: {comparison_path}")
        else:
            # Save comparison without target
            comparison_path = os.path.join(args.output_dir, "mel_comparison.png")
            save_mel_comparison_image(
                [prompt_mel, generated_mel],
                ["Prompt", "Generated"],
                comparison_path,
                cmap=args.colormap, dpi=args.dpi,
                sample_rate=pipeline.sample_rate, hop_size=pipeline.hop_size,
            )
            logger.info(f"  Saved: {comparison_path}")

    # Save mel as numpy array for further processing
    mel_npy_path = os.path.join(args.output_dir, "generated_mel.npy")
    np.save(mel_npy_path, generated_mel.cpu().numpy())
    logger.info(f"Saved mel spectrogram as numpy: {mel_npy_path}")

    # Optionally generate audio
    if vocoder is not None:
        logger.info("Generating audio using HiFi-GAN vocoder...")
        audio = pipeline.mel_to_audio(generated_mel)
        audio = audio[0].cpu().numpy()

        audio_path = os.path.join(args.output_dir, "generated_audio.wav")
        sf.write(audio_path, audio, pipeline.sample_rate)
        logger.info(f"Saved audio: {audio_path}")

    logger.info("Inference completed successfully!")
    logger.info("=" * 60)
    logger.info("||  Flow Matching S2Mel Inference Finished  ||")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
