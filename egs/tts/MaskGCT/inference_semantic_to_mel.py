#!/usr/bin/env python3
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for Semantic-to-Mel model.

This script:
1. Loads a trained SemanticToMel model
2. Extracts semantic tokens from input audio
3. Uses reference audio for voice cloning
4. Generates mel spectrograms
5. Saves mel spectrograms as images

Usage:
    python inference_semantic_to_mel.py \
        --checkpoint_path /path/to/checkpoint \
        --config_path /path/to/config.json \
        --input_audio /path/to/input.wav \
        --reference_audio /path/to/reference.wav \
        --output_dir /path/to/output
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from utils.util import load_config
from models.tts.maskgct.maskgct_semantic_to_mel import SemanticToMel, SemanticToMelWithPrompt
from models.codec.kmeans.repcodec_model import RepCodec
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor
from utils.mel import extract_mel_features


class SemanticToMelInference:
    """Inference class for Semantic-to-Mel model."""

    def __init__(self, checkpoint_path, config_path, device="cuda"):
        """
        Initialize inference.

        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config JSON file
            device: Device to run inference on
        """
        self.device = device
        self.cfg = load_config(config_path)

        # Load models
        self._load_semantic_codec()
        self._load_semantic_model()
        self._load_semantic_to_mel_model(checkpoint_path)

        print(f"Models loaded successfully on {device}")

    def _load_semantic_codec(self):
        """Load semantic codec for extracting semantic tokens."""
        self.semantic_codec = RepCodec(cfg=self.cfg.model.semantic_codec)

        pretrained_path = self.cfg.model.semantic_codec.pretrained_path
        if pretrained_path and os.path.exists(pretrained_path):
            if os.path.isdir(pretrained_path):
                model_path = os.path.join(pretrained_path, "model.safetensors")
                if not os.path.exists(model_path):
                    model_path = os.path.join(pretrained_path, "pytorch_model.bin")
            else:
                model_path = pretrained_path

            if os.path.exists(model_path):
                if model_path.endswith(".safetensors"):
                    import safetensors.torch
                    safetensors.torch.load_model(self.semantic_codec, model_path)
                else:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    if isinstance(checkpoint, dict) and "model" in checkpoint:
                        self.semantic_codec.load_state_dict(checkpoint["model"], strict=False)
                    else:
                        self.semantic_codec.load_state_dict(checkpoint, strict=False)

        self.semantic_codec.eval()
        self.semantic_codec.to(self.device)

        # Load normalization statistics
        if hasattr(self.cfg.model.semantic_codec, "representation_stat_mean_var_path"):
            stat_path = self.cfg.model.semantic_codec.representation_stat_mean_var_path
            if os.path.exists(stat_path):
                stat = torch.load(stat_path, weights_only=False)
                self.semantic_mean = torch.tensor(stat["mean"]).to(self.device)
                self.semantic_std = torch.sqrt(torch.tensor(stat["var"])).to(self.device)
            else:
                self.semantic_mean = None
                self.semantic_std = None
        else:
            self.semantic_mean = None
            self.semantic_std = None

    def _load_semantic_model(self):
        """Load w2v-bert-2.0 for feature extraction."""
        print("Loading w2v-bert-2.0 model...")
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model.eval()
        self.semantic_model.to(self.device)

        self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        print("w2v-bert-2.0 model loaded")

    def _load_semantic_to_mel_model(self, checkpoint_path):
        """Load Semantic-to-Mel model."""
        print(f"Loading Semantic-to-Mel model from {checkpoint_path}...")

        use_prompt = getattr(self.cfg.model.semantic_to_mel, "use_prompt", False)
        if use_prompt:
            self.model = SemanticToMelWithPrompt(cfg=self.cfg.model.semantic_to_mel)
        else:
            self.model = SemanticToMel(cfg=self.cfg.model.semantic_to_mel)

        # Load checkpoint
        if os.path.isdir(checkpoint_path):
            # Accelerate checkpoint format
            model_path = os.path.join(checkpoint_path, "model.safetensors")
            if not os.path.exists(model_path):
                model_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        else:
            model_path = checkpoint_path

        if os.path.exists(model_path):
            if model_path.endswith(".safetensors"):
                import safetensors.torch
                safetensors.torch.load_model(self.model, model_path)
            else:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    self.model.load_state_dict(checkpoint["model"], strict=False)
                else:
                    self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()
        self.model.to(self.device)
        print("Semantic-to-Mel model loaded")

    @torch.no_grad()
    def extract_semantic_tokens(self, audio_path, output_layer=17):
        """
        Extract semantic tokens from audio file.

        Args:
            audio_path: Path to audio file
            output_layer: Which layer of w2v-bert to use

        Returns:
            semantic_tokens: [1, T] tensor of semantic token indices
        """
        # Load audio at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)

        # Process with w2v-bert processor
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Extract features
        outputs = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        feat = outputs.hidden_states[output_layer]  # [1, T, D]

        # Normalize
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean) / self.semantic_std

        # Quantize to semantic tokens
        semantic_tokens, _ = self.semantic_codec.quantize(feat)  # [1, T]

        return semantic_tokens

    @torch.no_grad()
    def extract_mel_spectrogram(self, audio_path):
        """
        Extract mel spectrogram from audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            mel: [1, T, n_mel] tensor of mel spectrogram
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.cfg.preprocess.sample_rate)
        audio = torch.tensor(audio).unsqueeze(0).to(self.device)

        # Extract mel spectrogram
        mel = extract_mel_features(audio, self.cfg.preprocess, center=False)

        # mel shape: (B, n_mel, T) -> (B, T, n_mel)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.transpose(1, 2)

        return mel

    @torch.no_grad()
    def generate(self, input_audio_path, reference_audio_path,
                 n_timesteps=50, cfg_scale=2.0, temperature=1.0):
        """
        Generate mel spectrogram from input audio with voice cloning.

        Args:
            input_audio_path: Path to input audio (provides semantic content)
            reference_audio_path: Path to reference audio (provides voice characteristics)
            n_timesteps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            temperature: Sampling temperature

        Returns:
            mel_generated: [1, T, n_mel] generated mel spectrogram
            semantic_tokens: [1, T] semantic tokens
            ref_mel: [1, T_ref, n_mel] reference mel spectrogram
        """
        print(f"Extracting semantic tokens from {input_audio_path}...")
        semantic_tokens = self.extract_semantic_tokens(input_audio_path)

        print(f"Extracting reference mel from {reference_audio_path}...")
        ref_mel = self.extract_mel_spectrogram(reference_audio_path)

        print(f"Generating mel spectrogram with {n_timesteps} diffusion steps...")
        mel_generated = self.model.generate(
            semantic_tokens=semantic_tokens,
            semantic_mask=None,
            ref_mel=ref_mel,
            ref_mel_mask=None,
            n_timesteps=n_timesteps,
            cfg_scale=cfg_scale,
            temperature=temperature,
        )

        return mel_generated, semantic_tokens, ref_mel


def save_mel_as_image(mel, output_path, title="Mel Spectrogram", figsize=(12, 4)):
    """
    Save mel spectrogram as image.

    Args:
        mel: [T, n_mel] or [1, T, n_mel] mel spectrogram tensor
        output_path: Path to save image
        title: Title for the plot
        figsize: Figure size
    """
    if isinstance(mel, torch.Tensor):
        mel = mel.cpu().numpy()

    if len(mel.shape) == 3:
        mel = mel.squeeze(0)

    # Transpose to [n_mel, T] for visualization
    mel = mel.T

    plt.figure(figsize=figsize)
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Magnitude (dB)')
    plt.xlabel('Time (frames)')
    plt.ylabel('Mel Frequency Bin')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mel spectrogram to {output_path}")


def save_mel_comparison(mel_generated, mel_target, output_path, figsize=(12, 8)):
    """
    Save comparison of generated and target mel spectrograms.

    Args:
        mel_generated: Generated mel spectrogram
        mel_target: Target mel spectrogram
        output_path: Path to save image
        figsize: Figure size
    """
    if isinstance(mel_generated, torch.Tensor):
        mel_generated = mel_generated.cpu().numpy()
    if isinstance(mel_target, torch.Tensor):
        mel_target = mel_target.cpu().numpy()

    if len(mel_generated.shape) == 3:
        mel_generated = mel_generated.squeeze(0)
    if len(mel_target.shape) == 3:
        mel_target = mel_target.squeeze(0)

    # Transpose to [n_mel, T]
    mel_generated = mel_generated.T
    mel_target = mel_target.T

    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Generated mel
    im1 = axes[0].imshow(mel_generated, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_xlabel('Time (frames)')
    axes[0].set_ylabel('Mel Bin')
    axes[0].set_title('Generated Mel Spectrogram')
    plt.colorbar(im1, ax=axes[0], label='Magnitude (dB)')

    # Target mel
    im2 = axes[1].imshow(mel_target, aspect='auto', origin='lower', cmap='viridis')
    axes[1].set_xlabel('Time (frames)')
    axes[1].set_ylabel('Mel Bin')
    axes[1].set_title('Target Mel Spectrogram')
    plt.colorbar(im2, ax=axes[1], label='Magnitude (dB)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved mel comparison to {output_path}")


def save_mel_as_numpy(mel, output_path):
    """
    Save mel spectrogram as numpy file.

    Args:
        mel: Mel spectrogram tensor
        output_path: Path to save numpy file
    """
    if isinstance(mel, torch.Tensor):
        mel = mel.cpu().numpy()

    np.save(output_path, mel)
    print(f"Saved mel spectrogram to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Semantic-to-Mel Inference")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to config JSON file")
    parser.add_argument("--input_audio", type=str, required=True,
                        help="Path to input audio (provides semantic content)")
    parser.add_argument("--reference_audio", type=str, required=True,
                        help="Path to reference audio (provides voice characteristics)")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory")
    parser.add_argument("--n_timesteps", type=int, default=50,
                        help="Number of diffusion steps")
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")
    parser.add_argument("--save_numpy", action="store_true",
                        help="Also save mel as numpy file")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize inference
    inference = SemanticToMelInference(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        device=args.device
    )

    # Generate mel spectrogram
    mel_generated, semantic_tokens, ref_mel = inference.generate(
        input_audio_path=args.input_audio,
        reference_audio_path=args.reference_audio,
        n_timesteps=args.n_timesteps,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature
    )

    # Get base name for output files
    input_name = os.path.splitext(os.path.basename(args.input_audio))[0]
    ref_name = os.path.splitext(os.path.basename(args.reference_audio))[0]
    output_base = f"{input_name}_voice_{ref_name}"

    # Save generated mel as image
    save_mel_as_image(
        mel_generated,
        os.path.join(args.output_dir, f"{output_base}_generated.png"),
        title=f"Generated Mel (Input: {input_name}, Voice: {ref_name})"
    )

    # Save reference mel as image
    save_mel_as_image(
        ref_mel,
        os.path.join(args.output_dir, f"{output_base}_reference.png"),
        title=f"Reference Mel ({ref_name})"
    )

    # Optionally save as numpy
    if args.save_numpy:
        save_mel_as_numpy(
            mel_generated,
            os.path.join(args.output_dir, f"{output_base}_generated.npy")
        )

    # Also extract and save target mel from input audio for comparison
    target_mel = inference.extract_mel_spectrogram(args.input_audio)
    save_mel_as_image(
        target_mel,
        os.path.join(args.output_dir, f"{output_base}_target.png"),
        title=f"Target Mel ({input_name})"
    )

    # Save comparison
    # Note: lengths may differ, so we truncate to minimum length
    min_len = min(mel_generated.shape[1], target_mel.shape[1])
    save_mel_comparison(
        mel_generated[:, :min_len, :],
        target_mel[:, :min_len, :],
        os.path.join(args.output_dir, f"{output_base}_comparison.png")
    )

    print(f"\nInference complete! Results saved to {args.output_dir}")
    print(f"  - Generated mel: {output_base}_generated.png")
    print(f"  - Reference mel: {output_base}_reference.png")
    print(f"  - Target mel: {output_base}_target.png")
    print(f"  - Comparison: {output_base}_comparison.png")


if __name__ == "__main__":
    main()
