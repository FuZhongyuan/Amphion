# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Complete TTS Inference Pipeline using Semantic-to-Mel model.

This pipeline replaces the S2A + acoustic codec approach with a direct
semantic-to-mel diffusion model, followed by HiFi-GAN vocoder.

Pipeline:
1. Text -> Phone IDs (G2P)
2. Prompt Audio -> Semantic Tokens (w2v-bert-2.0 + RepCodec)
3. Phone IDs + Prompt Semantic -> Target Semantic Tokens (T2S model)
4. Semantic Tokens + Prompt Mel -> Target Mel Spectrogram (Semantic-to-Mel model)
5. Mel Spectrogram -> Waveform (HiFi-GAN vocoder)
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
from tqdm import tqdm

from utils.util import load_config, Logger
from utils.mel import extract_mel_features

from models.codec.kmeans.repcodec_model import RepCodec
from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S
from models.tts.maskgct.maskgct_semantic_to_mel import SemanticToMel, SemanticToMelWithPrompt
from models.vocoders.gan.generator.hifigan import HiFiGAN
from transformers import Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

from models.tts.maskgct.g2p.g2p_generation import g2p, chn_eng_g2p


def g2p_(text, language):
    """Convert text to phone IDs."""
    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p(text, sentence=None, language=language)


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


def build_t2s_model(cfg, device):
    """Build Text-to-Semantic model."""
    t2s_model = MaskGCT_T2S(cfg=cfg)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model


def build_semantic_to_mel_model(cfg, device):
    """Build Semantic-to-Mel model."""
    use_prompt = getattr(cfg, "use_prompt", False)

    if use_prompt:
        model = SemanticToMelWithPrompt(cfg=cfg)
    else:
        model = SemanticToMel(cfg=cfg)

    model.eval()
    model.to(device)
    return model


def build_hifigan_vocoder(cfg, device):
    """Build HiFi-GAN vocoder."""
    vocoder = HiFiGAN(cfg)
    vocoder.eval()
    vocoder.to(device)
    return vocoder


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


class SemanticToMelInferencePipeline:
    """
    Complete TTS inference pipeline using Semantic-to-Mel model.

    This pipeline converts text to speech using:
    1. G2P for text-to-phone conversion
    2. w2v-bert-2.0 + RepCodec for semantic token extraction
    3. T2S model for text-to-semantic generation
    4. Semantic-to-Mel model for mel spectrogram generation
    5. HiFi-GAN for waveform synthesis
    """

    def __init__(
        self,
        semantic_model,
        semantic_codec,
        t2s_model,
        semantic_to_mel_model,
        vocoder,
        semantic_mean,
        semantic_std,
        preprocess_cfg,
        device,
    ):
        self.processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model = semantic_model
        self.semantic_codec = semantic_codec
        self.t2s_model = t2s_model
        self.semantic_to_mel_model = semantic_to_mel_model
        self.vocoder = vocoder
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.preprocess_cfg = preprocess_cfg
        self.device = device

        # Mel spectrogram parameters
        self.sample_rate = getattr(preprocess_cfg, "sample_rate", 16000)
        self.hop_size = getattr(preprocess_cfg, "hop_size", 320)
        self.n_mel = getattr(preprocess_cfg, "n_mel", 80)

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
        # speech: numpy array at sample_rate
        speech_tensor = torch.tensor(speech).float().unsqueeze(0).to(self.device)

        mel = extract_mel_features(speech_tensor, self.preprocess_cfg, center=False)

        # mel shape: (B, n_mel, T) -> (B, T, n_mel)
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.transpose(1, 2)

        return mel

    @torch.no_grad()
    def text2semantic(
        self,
        prompt_speech,
        prompt_text,
        prompt_language,
        target_text,
        target_language,
        target_len=None,
        n_timesteps=50,
        cfg=2.5,
        rescale_cfg=0.75,
    ):
        """Convert text to semantic tokens."""
        prompt_phone_id = g2p_(prompt_text, prompt_language)[1]
        target_phone_id = g2p_(target_text, target_language)[1]

        if target_len is None:
            target_len = int(
                (len(prompt_speech) * len(target_phone_id) / len(prompt_phone_id))
                / 16000
                * 50
            )
        else:
            target_len = int(target_len * 50)

        prompt_phone_id = torch.tensor(prompt_phone_id, dtype=torch.long).to(self.device)
        target_phone_id = torch.tensor(target_phone_id, dtype=torch.long).to(self.device)

        phone_id = torch.cat([prompt_phone_id, target_phone_id])

        input_features, attention_mask = self.extract_features(prompt_speech)
        input_features = input_features.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        semantic_code, _ = self.extract_semantic_code(input_features, attention_mask)

        predict_semantic = self.t2s_model.reverse_diffusion(
            semantic_code[:, :],
            target_len,
            phone_id.unsqueeze(0),
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        combine_semantic_code = torch.cat(
            [semantic_code[:, :], predict_semantic], dim=-1
        )
        prompt_semantic_code = semantic_code

        return combine_semantic_code, prompt_semantic_code

    @torch.no_grad()
    def semantic2mel(
        self,
        semantic_tokens,
        prompt_mel,
        prompt_semantic_tokens=None,
        n_timesteps=50,
        cfg_scale=2.0,
        temperature=1.0,
    ):
        """Convert semantic tokens to mel spectrogram."""
        B = semantic_tokens.shape[0]
        T = semantic_tokens.shape[1]

        # Create mask
        semantic_mask = torch.ones(B, T, device=self.device)

        # Check if model supports prompt-based generation
        if isinstance(self.semantic_to_mel_model, SemanticToMelWithPrompt) and prompt_semantic_tokens is not None:
            # Use prompt-based generation for better acoustic cloning
            prompt_mask = torch.ones(B, prompt_semantic_tokens.shape[1], device=self.device)

            # Get target semantic tokens (excluding prompt)
            prompt_len = prompt_semantic_tokens.shape[1]
            target_semantic = semantic_tokens[:, prompt_len:]
            target_mask = torch.ones(B, target_semantic.shape[1], device=self.device)

            mel = self.semantic_to_mel_model.generate_with_prompt(
                semantic_tokens=target_semantic,
                prompt_mel=prompt_mel,
                prompt_semantic_tokens=prompt_semantic_tokens,
                semantic_mask=target_mask,
                prompt_mask=prompt_mask,
                n_timesteps=n_timesteps,
                cfg_scale=cfg_scale,
                temperature=temperature,
            )
        else:
            # Use reference-based generation
            ref_mel_mask = torch.ones(B, prompt_mel.shape[1], device=self.device)

            mel = self.semantic_to_mel_model.generate(
                semantic_tokens=semantic_tokens,
                semantic_mask=semantic_mask,
                ref_mel=prompt_mel,
                ref_mel_mask=ref_mel_mask,
                n_timesteps=n_timesteps,
                cfg_scale=cfg_scale,
                temperature=temperature,
            )

        return mel

    @torch.no_grad()
    def reconstruct_prompt_from_semantic(
        self,
        prompt_semantic_tokens,
        target_semantic_tokens=None,
        prompt_mel=None,
        n_timesteps=50,
        cfg_scale=2.0,
        temperature=1.0,
    ):
        """
        Reconstruct prompt mel spectrogram from discrete semantic tokens.

        This generates a mel spectrogram for the prompt part using the discrete
        semantic tokens (optionally with global acoustic conditioning from the
        prompt mel), instead of directly reusing the original prompt mel frames.
        """
        # If target semantic tokens are provided, reconstruct prompt + target
        if target_semantic_tokens is not None:
            semantic_tokens = torch.cat(
                [prompt_semantic_tokens, target_semantic_tokens], dim=1
            )
        else:
            semantic_tokens = prompt_semantic_tokens

        B = semantic_tokens.shape[0]
        T = semantic_tokens.shape[1]

        # Create mask for prompt semantic tokens
        semantic_mask = torch.ones(B, T, device=self.device)

        # Optional: use prompt mel only as a reference for global speaker embedding
        ref_mel = None
        ref_mel_mask = None
        if prompt_mel is not None:
            ref_mel = prompt_mel
            ref_mel_mask = torch.ones(B, prompt_mel.shape[1], device=self.device)

        mel = self.semantic_to_mel_model.generate(
            semantic_tokens=semantic_tokens,
            semantic_mask=semantic_mask,
            ref_mel=ref_mel,
            ref_mel_mask=ref_mel_mask,
            n_timesteps=n_timesteps,
            cfg_scale=cfg_scale,
            temperature=temperature,
        )

        return mel

    @torch.no_grad()
    def mel2audio(self, mel):
        """Convert mel spectrogram to audio using HiFi-GAN."""
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
        prompt_text,
        target_text,
        language="en",
        target_language="en",
        target_len=None,
        n_timesteps_t2s=50,
        cfg_t2s=2.5,
        rescale_cfg_t2s=0.75,
        n_timesteps_s2m=50,
        cfg_s2m=2.0,
        temperature=1.0,
    ):
        """
        Run complete TTS inference.

        Args:
            prompt_speech_path: Path to prompt audio file
            prompt_text: Text corresponding to prompt audio
            target_text: Target text to synthesize
            language: Language of prompt text
            target_language: Language of target text
            target_len: Target duration in seconds (None for auto)
            n_timesteps_t2s: Number of diffusion steps for T2S
            cfg_t2s: Classifier-free guidance scale for T2S
            rescale_cfg_t2s: CFG rescale factor for T2S
            n_timesteps_s2m: Number of diffusion steps for Semantic-to-Mel
            cfg_s2m: Classifier-free guidance scale for Semantic-to-Mel
            temperature: Sampling temperature

        Returns:
            audio: Generated audio (prompt + target) as numpy array
            sample_rate: Sample rate of generated audio
            prompt_recon_audio: Prompt audio reconstructed from semantic tokens
        """
        # Load prompt audio
        speech_16k = librosa.load(prompt_speech_path, sr=16000)[0]

        # Step 1: Text to Semantic
        print("Step 1: Text to Semantic...")
        combine_semantic_code, prompt_semantic_code = self.text2semantic(
            speech_16k,
            prompt_text,
            language,
            target_text,
            target_language,
            target_len,
            n_timesteps_t2s,
            cfg_t2s,
            rescale_cfg_t2s,
        )
        print(f"  Combined semantic shape: {combine_semantic_code.shape}")
        print(f"  Prompt semantic shape: {prompt_semantic_code.shape}")

        # Step 2: Extract prompt mel spectrogram
        print("Step 2: Extracting prompt mel spectrogram...")
        prompt_mel = self.extract_mel_spectrogram(speech_16k)
        print(f"  Prompt mel shape: {prompt_mel.shape}")

        # Step 2.5: Reconstruct prompt from semantic tokens
        print("Step 2.5: Reconstructing prompt (and optional target) from semantic tokens...")
        prompt_semantic_len = prompt_semantic_code.shape[1]
        target_semantic_tokens = combine_semantic_code[:, prompt_semantic_len:]
        prompt_recon_mel = self.reconstruct_prompt_from_semantic(
            prompt_semantic_tokens=prompt_semantic_code,
            target_semantic_tokens=target_semantic_tokens,
            prompt_mel=prompt_mel,
            n_timesteps=n_timesteps_s2m,
            cfg_scale=cfg_s2m,
            temperature=temperature,
        )
        print(f"  Reconstructed prompt mel shape: {prompt_recon_mel.shape}")

        # Align semantic tokens with mel length if needed
        semantic_len = combine_semantic_code.shape[1]
        prompt_mel_len = prompt_mel.shape[1]
        prompt_semantic_len = prompt_semantic_code.shape[1]

        # Calculate expected target mel length based on semantic tokens
        target_semantic_len = semantic_len - prompt_semantic_len
        # Semantic tokens are at ~50Hz, mel is at sample_rate / hop_size
        mel_rate = self.sample_rate / self.hop_size
        semantic_rate = 50  # w2v-bert-2.0 output rate
        target_mel_len = int(target_semantic_len * mel_rate / semantic_rate)

        print(f"  Target semantic length: {target_semantic_len}")
        print(f"  Expected target mel length: {target_mel_len}")

        # Step 3: Semantic to Mel
        print("Step 3: Semantic to Mel...")
        mel = self.semantic2mel(
            combine_semantic_code,
            prompt_mel,
            prompt_semantic_tokens=prompt_semantic_code,
            n_timesteps=n_timesteps_s2m,
            cfg_scale=cfg_s2m,
            temperature=temperature,
        )
        print(f"  Generated mel shape: {mel.shape}")

        # Step 4: Mel to Audio
        print("Step 4: Mel to Audio (HiFi-GAN)...")
        audio = self.mel2audio(mel)
        audio = audio[0].cpu().numpy()
        print(f"  Generated audio length: {len(audio)} samples ({len(audio)/self.sample_rate:.2f}s)")

        # Step 4b: Mel to Audio for reconstructed prompt
        print("Step 4b: Mel to Audio (HiFi-GAN) for reconstructed prompt...")
        prompt_recon_audio = self.mel2audio(prompt_recon_mel)
        prompt_recon_audio = prompt_recon_audio[0].cpu().numpy()
        print(
            f"  Reconstructed prompt audio length: {len(prompt_recon_audio)} samples "
            f"({len(prompt_recon_audio)/self.sample_rate:.2f}s)"
        )

        return audio, self.sample_rate, prompt_recon_audio


def main():
    parser = argparse.ArgumentParser(description="Semantic-to-Mel TTS Inference")

    # Config paths
    parser.add_argument("--semantic_codec_cfg", type=str, required=True,
                        help="Path to semantic codec config (e.g., semantic_codec_mini.json)")
    parser.add_argument("--t2s_cfg", type=str, required=True,
                        help="Path to T2S model config (e.g., t2s_mini.json)")
    parser.add_argument("--semantic_to_mel_cfg", type=str, required=True,
                        help="Path to Semantic-to-Mel config (e.g., semantic_to_mel_mini.json)")
    parser.add_argument("--vocoder_cfg", type=str, required=True,
                        help="Path to HiFi-GAN vocoder config")

    # Checkpoint paths
    parser.add_argument("--semantic_codec_ckpt", type=str, required=True,
                        help="Path to semantic codec checkpoint")
    parser.add_argument("--t2s_ckpt", type=str, required=True,
                        help="Path to T2S model checkpoint")
    parser.add_argument("--semantic_to_mel_ckpt", type=str, required=True,
                        help="Path to Semantic-to-Mel checkpoint")
    parser.add_argument("--vocoder_ckpt", type=str, required=True,
                        help="Path to HiFi-GAN vocoder checkpoint")
    parser.add_argument("--semantic_stat_path", type=str,
                        default="./models/tts/maskgct/ckpt/wav2vec2bert_stats.pt",
                        help="Path to semantic model statistics")

    # Input/Output
    parser.add_argument("--prompt_wav_path", type=str, required=True,
                        help="Path to prompt audio file")
    parser.add_argument("--prompt_text", type=str, required=True,
                        help="Text corresponding to prompt audio")
    parser.add_argument("--target_text", type=str, required=True,
                        help="Target text to synthesize")
    parser.add_argument("--save_path", type=str, default="generated_audio.wav",
                        help="Path to save generated audio")
    parser.add_argument("--save_prompt_recon_path", type=str, default=None,
                        help="Path to save prompt audio reconstructed from semantic tokens")

    # Generation parameters
    parser.add_argument("--language", type=str, default="en",
                        help="Language of prompt text")
    parser.add_argument("--target_language", type=str, default="en",
                        help="Language of target text")
    parser.add_argument("--target_len", type=float, default=None,
                        help="Target duration in seconds (None for auto)")
    parser.add_argument("--n_timesteps_t2s", type=int, default=50,
                        help="Number of diffusion steps for T2S")
    parser.add_argument("--cfg_t2s", type=float, default=2.5,
                        help="CFG scale for T2S")
    parser.add_argument("--n_timesteps_s2m", type=int, default=50,
                        help="Number of diffusion steps for Semantic-to-Mel")
    parser.add_argument("--cfg_s2m", type=float, default=2.0,
                        help="CFG scale for Semantic-to-Mel")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")

    # Device
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--log_level", type=str, default="info",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join(os.path.dirname(args.save_path), "semantic_to_mel_inference.log")
    logger = Logger(log_file, level=args.log_level).logger

    logger.info("=" * 60)
    logger.info("||    Semantic-to-Mel TTS Inference Started    ||")
    logger.info("=" * 60)

    device = torch.device(args.device)

    # Load configs
    logger.info("Loading configurations...")
    semantic_codec_cfg = load_config(args.semantic_codec_cfg)
    t2s_cfg = load_config(args.t2s_cfg)
    semantic_to_mel_cfg = load_config(args.semantic_to_mel_cfg)
    vocoder_cfg = load_config(args.vocoder_cfg)

    # Build models
    logger.info("Building semantic model (w2v-bert-2.0)...")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(
        device, args.semantic_stat_path
    )

    logger.info("Building semantic codec...")
    semantic_codec = build_semantic_codec(semantic_codec_cfg.model.semantic_codec, device)
    load_checkpoint(semantic_codec, args.semantic_codec_ckpt, device)

    logger.info("Building T2S model...")
    t2s_model = build_t2s_model(t2s_cfg.model.t2s_model, device)
    load_checkpoint(t2s_model, args.t2s_ckpt, device)

    logger.info("Building Semantic-to-Mel model...")
    semantic_to_mel_model = build_semantic_to_mel_model(
        semantic_to_mel_cfg.model.semantic_to_mel, device
    )
    load_checkpoint(semantic_to_mel_model, args.semantic_to_mel_ckpt, device)

    logger.info("Building HiFi-GAN vocoder...")
    vocoder = build_hifigan_vocoder(vocoder_cfg, device)
    load_checkpoint(vocoder, args.vocoder_ckpt, device)

    logger.info("All models loaded successfully!")

    # Create inference pipeline
    logger.info("Creating inference pipeline...")
    pipeline = SemanticToMelInferencePipeline(
        semantic_model=semantic_model,
        semantic_codec=semantic_codec,
        t2s_model=t2s_model,
        semantic_to_mel_model=semantic_to_mel_model,
        vocoder=vocoder,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        preprocess_cfg=semantic_to_mel_cfg.preprocess,
        device=device,
    )

    # Run inference
    logger.info("Starting inference...")
    logger.info(f"Prompt audio: {args.prompt_wav_path}")
    logger.info(f"Prompt text: {args.prompt_text}")
    logger.info(f"Target text: {args.target_text}")

    audio, sample_rate, prompt_recon_audio = pipeline.inference(
        prompt_speech_path=args.prompt_wav_path,
        prompt_text=args.prompt_text,
        target_text=args.target_text,
        language=args.language,
        target_language=args.target_language,
        target_len=args.target_len,
        n_timesteps_t2s=args.n_timesteps_t2s,
        cfg_t2s=args.cfg_t2s,
        n_timesteps_s2m=args.n_timesteps_s2m,
        cfg_s2m=args.cfg_s2m,
        temperature=args.temperature,
    )

    # Save main audio
    logger.info(f"Saving generated audio to {args.save_path}")
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    sf.write(args.save_path, audio, sample_rate)

    # Save reconstructed prompt audio
    if args.save_prompt_recon_path is None:
        base_dir = os.path.dirname(args.save_path) or "."
        base_name = os.path.splitext(os.path.basename(args.save_path))[0]
        save_prompt_recon_path = os.path.join(base_dir, base_name + "_prompt_recon.wav")
    else:
        save_prompt_recon_path = args.save_prompt_recon_path

    logger.info(f"Saving reconstructed prompt audio to {save_prompt_recon_path}")
    os.makedirs(os.path.dirname(save_prompt_recon_path) or ".", exist_ok=True)
    sf.write(save_prompt_recon_path, prompt_recon_audio, sample_rate)

    logger.info("Inference completed successfully!")
    logger.info("=" * 60)
    logger.info("||    Semantic-to-Mel TTS Inference Finished    ||")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
