# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
S2Mel Inference Script for Evaluating Training Performance

This script evaluates S2Mel models (DiT or FM) trained with:
- train_s2mel_dit_mini.sh
- train_s2mel_fm_mini.sh

It simulates the training logic by:
1. Loading audio samples and extracting semantic tokens
2. Using the S2Mel model to convert semantic tokens to mel spectrograms
3. Using HiFi-GAN vocoder to generate audio from mel spectrograms
4. Saving both mel spectrograms and audio for evaluation
5. Visualizing mel spectrograms as images
"""

import argparse
import os
import sys
import json
import math
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from utils.util import load_config
from utils.mel import extract_mel_features


def plot_mel_spectrogram(mel_spec, title="Mel Spectrogram", save_path=None, figsize=(12, 6)):
    """
    Plot mel spectrogram and save as image.
    
    Args:
        mel_spec: Mel spectrogram array (T, mel_dim) or (1, T, mel_dim)
        title: Plot title
        save_path: Path to save the image
        figsize: Figure size (width, height)
    """
    # Handle different input shapes
    if len(mel_spec.shape) == 3:
        mel_spec = mel_spec.squeeze(0)  # Remove batch dimension (B, T, mel_dim) -> (T, mel_dim)
    
    # Ensure shape is (mel_dim, T) for plotting
    # mel_spec is typically (T, mel_dim), need to transpose to (mel_dim, T)
    if len(mel_spec.shape) == 2:
        # If second dimension is larger, assume it's (T, mel_dim) format
        if mel_spec.shape[1] > mel_spec.shape[0] or mel_spec.shape[1] >= 50:
            mel_spec = mel_spec.T
    
    plt.figure(figsize=figsize)
    plt.imshow(mel_spec, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    plt.colorbar(format='%+2.0f')
    plt.title(title)
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mel spectrogram plot: {save_path}")
    
    plt.close()


def plot_mel_comparison(mel_list, titles, save_path=None, figsize=(18, 10)):
    """
    Plot multiple mel spectrograms for comparison.
    
    Args:
        mel_list: List of mel spectrogram arrays
        titles: List of titles for each mel spectrogram
        save_path: Path to save the image
        figsize: Figure size (width, height)
    """
    n_mels = len(mel_list)
    fig, axes = plt.subplots(n_mels, 1, figsize=figsize)
    
    if n_mels == 1:
        axes = [axes]
    
    for idx, (mel_spec, title) in enumerate(zip(mel_list, titles)):
        # Handle different input shapes
        if len(mel_spec.shape) == 3:
            mel_spec = mel_spec.squeeze(0)  # (B, T, mel_dim) -> (T, mel_dim)
        
        # Ensure shape is (mel_dim, T) for plotting
        if len(mel_spec.shape) == 2:
            # If second dimension is larger, assume it's (T, mel_dim) format
            if mel_spec.shape[1] > mel_spec.shape[0] or mel_spec.shape[1] >= 50:
                mel_spec = mel_spec.T
        
        im = axes[idx].imshow(mel_spec, aspect='auto', origin='lower', 
                             interpolation='none', cmap='viridis')
        axes[idx].set_title(title)
        axes[idx].set_xlabel('Time Frame')
        axes[idx].set_ylabel('Mel Bin')
        plt.colorbar(im, ax=axes[idx], format='%+2.0f')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mel comparison plot: {save_path}")
    
    plt.close()


def plot_mel_difference(mel_pred, mel_gt, save_path=None, figsize=(18, 8)):
    """
    Plot predicted mel, ground truth mel, and their difference.
    
    Args:
        mel_pred: Predicted mel spectrogram
        mel_gt: Ground truth mel spectrogram
        save_path: Path to save the image
        figsize: Figure size (width, height)
    """
    # Handle different input shapes
    if len(mel_pred.shape) == 3:
        mel_pred = mel_pred.squeeze(0)  # (B, T, mel_dim) -> (T, mel_dim)
    if len(mel_gt.shape) == 3:
        mel_gt = mel_gt.squeeze(0)  # (B, T, mel_dim) -> (T, mel_dim)
    
    # Ensure shape is (mel_dim, T) for plotting
    if len(mel_pred.shape) == 2:
        if mel_pred.shape[1] > mel_pred.shape[0] or mel_pred.shape[1] >= 50:
            mel_pred = mel_pred.T
    if len(mel_gt.shape) == 2:
        if mel_gt.shape[1] > mel_gt.shape[0] or mel_gt.shape[1] >= 50:
            mel_gt = mel_gt.T
    
    # Ensure same shape for difference calculation
    min_t = min(mel_pred.shape[1], mel_gt.shape[1])
    min_mel = min(mel_pred.shape[0], mel_gt.shape[0])
    mel_pred_aligned = mel_pred[:min_mel, :min_t]
    mel_gt_aligned = mel_gt[:min_mel, :min_t]
    
    mel_diff = mel_pred_aligned - mel_gt_aligned
    
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot predicted
    im1 = axes[0].imshow(mel_pred, aspect='auto', origin='lower', 
                        interpolation='none', cmap='viridis')
    axes[0].set_title('Generated Mel Spectrogram')
    axes[0].set_ylabel('Mel Bin')
    plt.colorbar(im1, ax=axes[0], format='%+2.0f')
    
    # Plot ground truth
    im2 = axes[1].imshow(mel_gt, aspect='auto', origin='lower', 
                        interpolation='none', cmap='viridis')
    axes[1].set_title('Ground Truth Mel Spectrogram')
    axes[1].set_ylabel('Mel Bin')
    plt.colorbar(im2, ax=axes[1], format='%+2.0f')
    
    # Plot difference
    im3 = axes[2].imshow(mel_diff, aspect='auto', origin='lower', 
                        interpolation='none', cmap='RdBu_r', vmin=-2, vmax=2)
    axes[2].set_title('Difference (Generated - Ground Truth)')
    axes[2].set_xlabel('Time Frame')
    axes[2].set_ylabel('Mel Bin')
    plt.colorbar(im3, ax=axes[2], format='%+2.1f')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved mel difference plot: {save_path}")
    
    plt.close()


def load_checkpoint(model, ckpt_path, device):
    """Load checkpoint for a model."""
    import safetensors

    if os.path.isdir(ckpt_path):
        model_path = os.path.join(ckpt_path, "model.safetensors")
        if not os.path.exists(model_path):
            model_path = os.path.join(ckpt_path, "pytorch_model.bin")
    else:
        model_path = ckpt_path

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    print(f"Loading checkpoint from {model_path}")
    if model_path.endswith(".safetensors"):
        safetensors.torch.load_model(model, model_path)
    else:
        checkpoint = torch.load(model_path, map_location=device)
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


def build_semantic_model(device):
    """Build w2v-bert-2.0 model for semantic feature extraction."""
    from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

    print("Loading w2v-bert-2.0 model...")
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)

    processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    return semantic_model, processor


def build_semantic_codec(cfg, device):
    """Build semantic codec for quantizing semantic features."""
    from models.codec.kmeans.repcodec_model import RepCodec

    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec


def build_s2mel_dit_model(cfg, device):
    """Build S2Mel DiT model."""
    from models.tts.maskgct.maskgct_s2mel_dit import SemanticToMelDiT

    s2mel_model = SemanticToMelDiT(cfg=cfg)
    s2mel_model.eval()
    s2mel_model.to(device)
    return s2mel_model


def build_s2mel_fm_model(cfg, device):
    """Build S2Mel Flow Matching model."""
    from models.tts.maskgct.maskgct_s2mel_fm import SemanticToMelFM

    s2mel_model = SemanticToMelFM(cfg=cfg)
    s2mel_model.eval()
    s2mel_model.to(device)
    return s2mel_model


def build_vocoder(cfg, device):
    """Build HiFi-GAN vocoder."""
    from models.vocoders.gan.generator.hifigan import HiFiGAN

    vocoder = HiFiGAN(cfg)
    vocoder.eval()
    vocoder.to(device)
    return vocoder


@torch.no_grad()
def extract_semantic_tokens(
    wav_path,
    semantic_model,
    processor,
    semantic_codec,
    semantic_mean,
    semantic_std,
    device,
    output_layer=17,
):
    """Extract semantic tokens from audio file."""
    speech, _ = librosa.load(wav_path, sr=16000)

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = semantic_model(
        input_features=input_features,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    feat = outputs.hidden_states[output_layer]

    if semantic_mean is not None:
        feat = (feat - semantic_mean.to(feat)) / semantic_std.to(feat)

    semantic_tokens, _ = semantic_codec.quantize(feat)
    return semantic_tokens


@torch.no_grad()
def extract_mel(wav_path, cfg, device):
    """Extract mel spectrogram from audio file."""
    speech, _ = librosa.load(wav_path, sr=cfg.preprocess.sample_rate)
    speech = torch.tensor(speech).unsqueeze(0).to(device)

    mel = extract_mel_features(speech, cfg.preprocess, center=False)

    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    mel = mel.transpose(1, 2)

    if hasattr(cfg.preprocess, "mel_mean") and hasattr(cfg.preprocess, "mel_var"):
        mel = (mel - cfg.preprocess.mel_mean) / math.sqrt(cfg.preprocess.mel_var)

    return mel


class S2MelInferencePipeline:
    """
    Inference pipeline for evaluating S2Mel models.

    This pipeline simulates the training logic:
    1. Extract semantic tokens from audio
    2. Extract ground truth mel from audio (for prompt)
    3. Use S2Mel model to generate mel from semantic tokens
    4. Use vocoder to generate audio from mel
    """

    def __init__(
        self,
        semantic_model,
        processor,
        semantic_codec,
        s2mel_model,
        vocoder,
        semantic_mean,
        semantic_std,
        cfg,
        device,
        model_type="dit",
    ):
        self.semantic_model = semantic_model
        self.processor = processor
        self.semantic_codec = semantic_codec
        self.s2mel_model = s2mel_model
        self.vocoder = vocoder
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.cfg = cfg
        self.device = device
        self.model_type = model_type

    @torch.no_grad()
    def inference(
        self,
        wav_path,
        prompt_ratio=0.3,
        n_timesteps=None,
        cfg_scale=1.0,
        use_ddim=True,
        ddim_eta=0.0,
        rescale_cfg=0.75,
    ):
        """
        Run inference on a single audio file.

        This simulates the training process:
        1. Extract semantic tokens from the full audio
        2. Use a portion as prompt (prompt_ratio)
        3. Generate the remaining portion using S2Mel

        Args:
            wav_path: Path to input audio file
            prompt_ratio: Ratio of audio to use as prompt (0.0-1.0)
            n_timesteps: Number of diffusion/flow steps
            cfg_scale: Classifier-free guidance scale
            use_ddim: Use DDIM sampling (DiT only)
            ddim_eta: DDIM eta parameter (DiT only)
            rescale_cfg: CFG rescaling factor (FM only)

        Returns:
            dict with:
                - generated_mel: Generated mel spectrogram
                - generated_audio: Generated audio waveform
                - gt_mel: Ground truth mel spectrogram
                - gt_audio: Ground truth audio
                - prompt_mel: Prompt mel spectrogram
                - semantic_tokens: Extracted semantic tokens
        """
        # Set default timesteps based on model type
        if n_timesteps is None:
            n_timesteps = 50 if self.model_type == "dit" else 10

        # Extract semantic tokens from full audio
        semantic_tokens = extract_semantic_tokens(
            wav_path,
            self.semantic_model,
            self.processor,
            self.semantic_codec,
            self.semantic_mean,
            self.semantic_std,
            self.device,
        )

        # Extract ground truth mel from full audio
        gt_mel = extract_mel(wav_path, self.cfg, self.device)

        # Load ground truth audio
        gt_audio, _ = librosa.load(wav_path, sr=self.cfg.preprocess.sample_rate)

        # Align lengths between semantic tokens and mel
        target_len = min(semantic_tokens.shape[1], gt_mel.shape[1])
        semantic_tokens = semantic_tokens[:, :target_len]
        gt_mel = gt_mel[:, :target_len, :]

        # Split into prompt and target
        prompt_len = int(target_len * prompt_ratio)
        prompt_len = max(prompt_len, 5)  # Minimum prompt length

        prompt_semantic = semantic_tokens[:, :prompt_len]
        prompt_mel = gt_mel[:, :prompt_len, :]

        target_semantic = semantic_tokens[:, prompt_len:]
        target_mel_gt = gt_mel[:, prompt_len:, :]

        # Combine semantic tokens for conditioning
        combined_semantic = semantic_tokens

        # Get semantic embeddings
        cond = self.s2mel_model.cond_emb(combined_semantic)

        # Generate target mel using S2Mel model
        if self.model_type == "dit":
            generated_mel = self.s2mel_model.reverse_diffusion(
                cond=cond,
                prompt_mel=prompt_mel,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                use_ddim=use_ddim,
                ddim_eta=ddim_eta,
            )
        else:  # FM
            generated_mel = self.s2mel_model.reverse_diffusion(
                cond=cond,
                prompt_mel=prompt_mel,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                rescale_cfg=rescale_cfg,
            )

        # Combine prompt and generated mel for full output
        full_generated_mel = torch.cat([prompt_mel, generated_mel], dim=1)

        # Denormalize mel for vocoder
        if hasattr(self.cfg.preprocess, "mel_mean") and hasattr(self.cfg.preprocess, "mel_var"):
            full_generated_mel_denorm = full_generated_mel * math.sqrt(self.cfg.preprocess.mel_var) + self.cfg.preprocess.mel_mean
            gt_mel_denorm = gt_mel * math.sqrt(self.cfg.preprocess.mel_var) + self.cfg.preprocess.mel_mean
        else:
            full_generated_mel_denorm = full_generated_mel
            gt_mel_denorm = gt_mel

        # Generate audio using vocoder
        # Vocoder expects (B, mel_dim, T)
        generated_audio = self.vocoder(full_generated_mel_denorm.transpose(1, 2))
        generated_audio = generated_audio.squeeze(0).squeeze(0).cpu().numpy()

        # Also generate audio from GT mel for comparison
        reconstructed_audio = self.vocoder(gt_mel_denorm.transpose(1, 2))
        reconstructed_audio = reconstructed_audio.squeeze(0).squeeze(0).cpu().numpy()

        return {
            "generated_mel": full_generated_mel.cpu().numpy(),
            "generated_audio": generated_audio,
            "gt_mel": gt_mel.cpu().numpy(),
            "gt_audio": gt_audio,
            "reconstructed_audio": reconstructed_audio,
            "prompt_mel": prompt_mel.cpu().numpy(),
            "semantic_tokens": semantic_tokens.cpu().numpy(),
            "prompt_len": prompt_len,
            "target_len": target_len,
        }

    @torch.no_grad()
    def inference_from_semantic(
        self,
        semantic_tokens,
        prompt_mel,
        n_timesteps=None,
        cfg_scale=1.0,
        use_ddim=True,
        ddim_eta=0.0,
        rescale_cfg=0.75,
    ):
        """
        Run inference from pre-extracted semantic tokens and prompt mel.

        Args:
            semantic_tokens: Semantic tokens tensor (B, T)
            prompt_mel: Prompt mel tensor (B, T_prompt, mel_dim)
            n_timesteps: Number of diffusion/flow steps
            cfg_scale: Classifier-free guidance scale
            use_ddim: Use DDIM sampling (DiT only)
            ddim_eta: DDIM eta parameter (DiT only)
            rescale_cfg: CFG rescaling factor (FM only)

        Returns:
            dict with generated_mel and generated_audio
        """
        if n_timesteps is None:
            n_timesteps = 50 if self.model_type == "dit" else 10

        # Ensure tensors are on device
        if not isinstance(semantic_tokens, torch.Tensor):
            semantic_tokens = torch.tensor(semantic_tokens)
        if not isinstance(prompt_mel, torch.Tensor):
            prompt_mel = torch.tensor(prompt_mel)

        semantic_tokens = semantic_tokens.to(self.device)
        prompt_mel = prompt_mel.to(self.device)

        if len(semantic_tokens.shape) == 1:
            semantic_tokens = semantic_tokens.unsqueeze(0)
        if len(prompt_mel.shape) == 2:
            prompt_mel = prompt_mel.unsqueeze(0)

        # Get semantic embeddings
        cond = self.s2mel_model.cond_emb(semantic_tokens)

        # Generate mel
        if self.model_type == "dit":
            generated_mel = self.s2mel_model.reverse_diffusion(
                cond=cond,
                prompt_mel=prompt_mel,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                use_ddim=use_ddim,
                ddim_eta=ddim_eta,
            )
        else:
            generated_mel = self.s2mel_model.reverse_diffusion(
                cond=cond,
                prompt_mel=prompt_mel,
                n_timesteps=n_timesteps,
                cfg=cfg_scale,
                rescale_cfg=rescale_cfg,
            )

        # Combine prompt and generated
        full_mel = torch.cat([prompt_mel, generated_mel], dim=1)

        # Denormalize
        if hasattr(self.cfg.preprocess, "mel_mean") and hasattr(self.cfg.preprocess, "mel_var"):
            full_mel_denorm = full_mel * math.sqrt(self.cfg.preprocess.mel_var) + self.cfg.preprocess.mel_mean
        else:
            full_mel_denorm = full_mel

        # Generate audio
        audio = self.vocoder(full_mel_denorm.transpose(1, 2))
        audio = audio.squeeze(0).squeeze(0).cpu().numpy()

        return {
            "generated_mel": full_mel.cpu().numpy(),
            "generated_audio": audio,
        }


def save_results(results, output_dir, sample_name, sample_rate, visualize=True):
    """Save inference results to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save generated audio
    generated_audio_path = os.path.join(output_dir, f"{sample_name}_generated.wav")
    sf.write(generated_audio_path, results["generated_audio"], sample_rate)
    print(f"Saved generated audio: {generated_audio_path}")

    # Save reconstructed audio (GT mel -> vocoder)
    if "reconstructed_audio" in results:
        reconstructed_audio_path = os.path.join(output_dir, f"{sample_name}_reconstructed.wav")
        sf.write(reconstructed_audio_path, results["reconstructed_audio"], sample_rate)
        print(f"Saved reconstructed audio: {reconstructed_audio_path}")

    # Save ground truth audio
    if "gt_audio" in results:
        gt_audio_path = os.path.join(output_dir, f"{sample_name}_gt.wav")
        sf.write(gt_audio_path, results["gt_audio"], sample_rate)
        print(f"Saved ground truth audio: {gt_audio_path}")

    # Save mel spectrograms as numpy files
    generated_mel_path = os.path.join(output_dir, f"{sample_name}_generated_mel.npy")
    np.save(generated_mel_path, results["generated_mel"])
    print(f"Saved generated mel: {generated_mel_path}")

    if "gt_mel" in results:
        gt_mel_path = os.path.join(output_dir, f"{sample_name}_gt_mel.npy")
        np.save(gt_mel_path, results["gt_mel"])
        print(f"Saved ground truth mel: {gt_mel_path}")

    # Visualize mel spectrograms
    if visualize:
        print("\nGenerating mel spectrogram visualizations...")
        
        # Plot individual mel spectrograms
        generated_mel_img_path = os.path.join(output_dir, f"{sample_name}_generated_mel.png")
        plot_mel_spectrogram(
            results["generated_mel"],
            title="Generated Mel Spectrogram",
            save_path=generated_mel_img_path
        )
        
        if "gt_mel" in results:
            gt_mel_img_path = os.path.join(output_dir, f"{sample_name}_gt_mel.png")
            plot_mel_spectrogram(
                results["gt_mel"],
                title="Ground Truth Mel Spectrogram",
                save_path=gt_mel_img_path
            )
        
        if "prompt_mel" in results:
            prompt_mel_img_path = os.path.join(output_dir, f"{sample_name}_prompt_mel.png")
            plot_mel_spectrogram(
                results["prompt_mel"],
                title="Prompt Mel Spectrogram",
                save_path=prompt_mel_img_path
            )
        
        # Plot comparison
        if "gt_mel" in results:
            comparison_path = os.path.join(output_dir, f"{sample_name}_comparison.png")
            mel_list = [results["generated_mel"], results["gt_mel"]]
            titles = ["Generated Mel Spectrogram", "Ground Truth Mel Spectrogram"]
            
            if "prompt_mel" in results:
                mel_list.insert(0, results["prompt_mel"])
                titles.insert(0, "Prompt Mel Spectrogram")
            
            plot_mel_comparison(
                mel_list,
                titles,
                save_path=comparison_path
            )
            
            # Plot difference
            diff_path = os.path.join(output_dir, f"{sample_name}_difference.png")
            plot_mel_difference(
                results["generated_mel"],
                results["gt_mel"],
                save_path=diff_path
            )

    # Save metadata
    metadata = {
        "prompt_len": results.get("prompt_len", 0),
        "target_len": results.get("target_len", 0),
        "generated_mel_shape": list(results["generated_mel"].shape),
    }
    if "gt_mel" in results:
        metadata["gt_mel_shape"] = list(results["gt_mel"].shape)

    metadata_path = os.path.join(output_dir, f"{sample_name}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="S2Mel Inference Script for Evaluating Training Performance"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config file (e.g., s2mel_dit_mini.json or s2mel_fm_mini.json)",
    )
    parser.add_argument(
        "--s2mel_ckpt",
        type=str,
        required=True,
        help="Path to S2Mel model checkpoint",
    )
    parser.add_argument(
        "--semantic_codec_ckpt",
        type=str,
        default=None,
        help="Path to semantic codec checkpoint (uses config path if not specified)",
    )
    parser.add_argument(
        "--vocoder_ckpt",
        type=str,
        required=True,
        help="Path to HiFi-GAN vocoder checkpoint",
    )
    parser.add_argument(
        "--vocoder_config",
        type=str,
        required=True,
        help="Path to HiFi-GAN vocoder config",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input audio file or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/s2mel_inference",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["dit", "fm"],
        default=None,
        help="Model type (auto-detected from config if not specified)",
    )
    parser.add_argument(
        "--prompt_ratio",
        type=float,
        default=0.3,
        help="Ratio of audio to use as prompt (0.0-1.0)",
    )
    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=None,
        help="Number of diffusion/flow steps (default: 50 for DiT, 10 for FM)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--no_visualize",
        action="store_true",
        help="Disable mel spectrogram visualization",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    cfg = load_config(args.config)

    # Auto-detect model type from config
    model_type = args.model_type
    if model_type is None:
        if hasattr(cfg, "model_type"):
            if "DiT" in cfg.model_type or "dit" in cfg.model_type.lower():
                model_type = "dit"
            elif "FM" in cfg.model_type or "fm" in cfg.model_type.lower():
                model_type = "fm"
        if model_type is None:
            if hasattr(cfg.model, "s2mel_dit"):
                model_type = "dit"
            elif hasattr(cfg.model, "s2mel_fm"):
                model_type = "fm"
            else:
                raise ValueError("Cannot auto-detect model type. Please specify --model_type")
    print(f"Model type: {model_type}")

    # Load vocoder config
    vocoder_cfg = load_config(args.vocoder_config)

    # Build models
    print("Building models...")

    # Semantic model
    semantic_model, processor = build_semantic_model(device)

    # Semantic codec
    semantic_codec_cfg = cfg.model.semantic_codec
    semantic_codec = build_semantic_codec(semantic_codec_cfg, device)

    # Load semantic codec checkpoint
    semantic_codec_ckpt = args.semantic_codec_ckpt
    if semantic_codec_ckpt is None:
        semantic_codec_ckpt = getattr(semantic_codec_cfg, "pretrained_path", None)
    if semantic_codec_ckpt:
        load_checkpoint(semantic_codec, semantic_codec_ckpt, device)

    # S2Mel model
    if model_type == "dit":
        s2mel_model = build_s2mel_dit_model(cfg.model.s2mel_dit, device)
    else:
        s2mel_model = build_s2mel_fm_model(cfg.model.s2mel_fm, device)
    load_checkpoint(s2mel_model, args.s2mel_ckpt, device)

    # Vocoder
    vocoder = build_vocoder(vocoder_cfg, device)
    load_checkpoint(vocoder, args.vocoder_ckpt, device)

    # Load semantic normalization stats
    semantic_mean, semantic_std = None, None
    if hasattr(semantic_codec_cfg, "representation_stat_mean_var_path"):
        stat_path = semantic_codec_cfg.representation_stat_mean_var_path
        if os.path.exists(stat_path):
            stat = torch.load(stat_path, map_location=device)
            semantic_mean = torch.tensor(stat["mean"], device=device)
            semantic_std = torch.sqrt(torch.tensor(stat["var"], device=device))
            print(f"Loaded semantic stats from {stat_path}")

    # Create pipeline
    pipeline = S2MelInferencePipeline(
        semantic_model=semantic_model,
        processor=processor,
        semantic_codec=semantic_codec,
        s2mel_model=s2mel_model,
        vocoder=vocoder,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        cfg=cfg,
        device=device,
        model_type=model_type,
    )

    # Process input
    if os.path.isfile(args.input):
        input_files = [args.input]
    elif os.path.isdir(args.input):
        input_files = [
            os.path.join(args.input, f)
            for f in os.listdir(args.input)
            if f.endswith((".wav", ".flac", ".mp3"))
        ]
    else:
        raise ValueError(f"Input path does not exist: {args.input}")

    print(f"Processing {len(input_files)} file(s)...")

    for wav_path in input_files:
        print(f"\nProcessing: {wav_path}")
        sample_name = os.path.splitext(os.path.basename(wav_path))[0]

        try:
            results = pipeline.inference(
                wav_path=wav_path,
                prompt_ratio=args.prompt_ratio,
                n_timesteps=args.n_timesteps,
                cfg_scale=args.cfg_scale,
            )

            save_results(
                results,
                args.output_dir,
                sample_name,
                cfg.preprocess.sample_rate,
                visualize=not args.no_visualize,
            )
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()