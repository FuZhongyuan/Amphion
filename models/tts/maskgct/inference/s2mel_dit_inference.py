# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
S2Mel DiT Inference Script for MaskGCT

This script performs text-to-speech synthesis using:
1. T2S model: Text -> Semantic tokens
2. S2Mel DiT model: Semantic tokens + Prompt mel -> Target mel
3. Vocoder: Mel -> Waveform
"""

import argparse
import os
import torch
import soundfile as sf
import math

from utils.util import load_config

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

    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
    semantic_model.eval()
    semantic_model.to(device)

    processor = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    return semantic_model, processor


def build_semantic_codec(cfg, device):
    """Build semantic codec."""
    from models.codec.kmeans.repcodec_model import RepCodec

    semantic_codec = RepCodec(cfg=cfg)
    semantic_codec.eval()
    semantic_codec.to(device)
    return semantic_codec


def build_t2s_model(cfg, device):
    """Build T2S model."""
    from models.tts.maskgct.maskgct_t2s import MaskGCT_T2S

    t2s_model = MaskGCT_T2S(cfg=cfg)
    t2s_model.eval()
    t2s_model.to(device)
    return t2s_model


def build_s2mel_dit_model(cfg, device):
    """Build S2Mel DiT model."""
    from models.tts.maskgct.maskgct_s2mel_dit import SemanticToMelDiT

    s2mel_model = SemanticToMelDiT(cfg=cfg)
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
    import librosa

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
    import librosa
    from utils.mel import extract_mel_features

    speech, _ = librosa.load(wav_path, sr=cfg.preprocess.sample_rate)
    speech = torch.tensor(speech).unsqueeze(0).to(device)

    mel = extract_mel_features(speech, cfg.preprocess, center=False)
    
    # mel shape: (B, n_mel, T) -> (B, T, n_mel)
    if len(mel.shape) == 2:
        mel = mel.unsqueeze(0)
    mel = mel.transpose(1, 2)

    if hasattr(cfg.preprocess, "mel_mean") and hasattr(cfg.preprocess, "mel_var"):
        mel = (mel - cfg.preprocess.mel_mean) / math.sqrt(cfg.preprocess.mel_var)

    return mel


def g2p(text, language="en"):
    """Convert text to phone IDs."""
    from models.tts.maskgct.g2p.g2p_generation import g2p as g2p_func, chn_eng_g2p

    if language in ["zh", "en"]:
        return chn_eng_g2p(text)
    else:
        return g2p_func(text, sentence=None, language=language)


class S2MelDiTInferencePipeline:
    """Inference pipeline for S2Mel DiT TTS."""

    def __init__(
        self,
        semantic_model,
        processor,
        semantic_codec,
        t2s_model,
        s2mel_model,
        vocoder,
        semantic_mean,
        semantic_std,
        cfg,
        device,
    ):
        self.semantic_model = semantic_model
        self.processor = processor
        self.semantic_codec = semantic_codec
        self.t2s_model = t2s_model
        self.s2mel_model = s2mel_model
        self.vocoder = vocoder
        self.semantic_mean = semantic_mean
        self.semantic_std = semantic_std
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def inference(
        self,
        prompt_wav_path,
        prompt_text,
        target_text,
        prompt_language="en",
        target_language="en",
        target_len=None,
        t2s_n_timesteps=None,
        s2mel_n_timesteps=None,
        t2s_cfg=None,
        s2mel_cfg=None,
        use_ddim=None,
        ddim_eta=None,
    ):
        """Run TTS inference."""
        # Get inference params from config or use defaults
        inf_cfg = getattr(self.cfg, "inference", None)
        t2s_n_timesteps = t2s_n_timesteps if t2s_n_timesteps is not None else (getattr(inf_cfg, "t2s_n_timesteps", 60) if inf_cfg else 60)
        s2mel_n_timesteps = s2mel_n_timesteps if s2mel_n_timesteps is not None else (getattr(inf_cfg, "s2mel_n_timesteps", 50) if inf_cfg else 50)
        t2s_cfg = t2s_cfg if t2s_cfg is not None else (getattr(inf_cfg, "t2s_cfg", 1.0) if inf_cfg else 1.0)
        s2mel_cfg = s2mel_cfg if s2mel_cfg is not None else (getattr(inf_cfg, "s2mel_cfg", 1.0) if inf_cfg else 1.0)
        use_ddim = use_ddim if use_ddim is not None else (getattr(inf_cfg, "use_ddim", True) if inf_cfg else True)
        ddim_eta = ddim_eta if ddim_eta is not None else (getattr(inf_cfg, "ddim_eta", 0.0) if inf_cfg else 0.0)
        # Extract prompt semantic tokens
        prompt_semantic = extract_semantic_tokens(
            prompt_wav_path,
            self.semantic_model,
            self.processor,
            self.semantic_codec,
            self.semantic_mean,
            self.semantic_std,
            self.device,
        )

        # Extract prompt mel
        prompt_mel = extract_mel(prompt_wav_path, self.cfg, self.device)

        # Get phone IDs
        _, prompt_phone_id = g2p(prompt_text, prompt_language)
        _, target_phone_id = g2p(target_text, target_language)

        prompt_phone_id = torch.tensor(prompt_phone_id).unsqueeze(0).to(self.device)
        target_phone_id = torch.tensor(target_phone_id).unsqueeze(0).to(self.device)

        combined_phone_id = torch.cat([prompt_phone_id, target_phone_id], dim=1)

        # Estimate target length
        if target_len is not None:
            target_semantic_len = int(target_len * 50)
        else:
            ratio = len(target_phone_id[0]) / max(len(prompt_phone_id[0]), 1)
            target_semantic_len = int(prompt_semantic.shape[1] * ratio * 1.2)

        # T2S: Generate target semantic tokens
        target_semantic = self.t2s_model.reverse_diffusion(
            prompt=prompt_semantic,
            target_len=target_semantic_len,
            phone_id=combined_phone_id,
            n_timesteps=t2s_n_timesteps,
            cfg=t2s_cfg,
        )

        # Combine semantic tokens
        combined_semantic = torch.cat([prompt_semantic, target_semantic], dim=1)

        # Get semantic embeddings
        cond = self.s2mel_model.cond_emb(combined_semantic)

        # S2Mel: Generate target mel
        target_mel = self.s2mel_model.reverse_diffusion(
            cond=cond,
            prompt_mel=prompt_mel,
            n_timesteps=s2mel_n_timesteps,
            cfg=s2mel_cfg,
            use_ddim=use_ddim,
            ddim_eta=ddim_eta,
        )

        # Denormalize mel
        if hasattr(self.cfg.preprocess, "mel_mean") and hasattr(self.cfg.preprocess, "mel_var"):
            target_mel = target_mel * math.sqrt(self.cfg.preprocess.mel_var) + self.cfg.preprocess.mel_mean

        # Vocoder: Generate waveform
        target_mel = target_mel.transpose(1, 2)
        audio = self.vocoder(target_mel)
        audio = audio.squeeze(0).squeeze(0).cpu().numpy()

        return audio


def main():
    parser = argparse.ArgumentParser(description="S2Mel DiT Inference Script")
    parser.add_argument("--cfg_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--semantic_codec_ckpt", type=str, required=True, help="Semantic codec checkpoint")
    parser.add_argument("--t2s_ckpt", type=str, required=True, help="T2S model checkpoint")
    parser.add_argument("--s2mel_ckpt", type=str, required=True, help="S2Mel model checkpoint")
    parser.add_argument("--vocoder_ckpt", type=str, required=True, help="Vocoder checkpoint")
    parser.add_argument("--prompt_wav", type=str, required=True, help="Prompt audio path")
    parser.add_argument("--prompt_text", type=str, required=True, help="Prompt text")
    parser.add_argument("--target_text", type=str, required=True, help="Target text")
    parser.add_argument("--output_path", type=str, default="output.wav", help="Output path")
    parser.add_argument("--target_len", type=float, default=None, help="Target length in seconds")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.cfg_path)

    print("Building models...")

    semantic_model, processor = build_semantic_model(device)
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    s2mel_model = build_s2mel_dit_model(cfg.model.s2mel_dit, device)
    vocoder = build_vocoder(cfg, device)

    load_checkpoint(semantic_codec, args.semantic_codec_ckpt, device)
    load_checkpoint(t2s_model, args.t2s_ckpt, device)
    load_checkpoint(s2mel_model, args.s2mel_ckpt, device)
    load_checkpoint(vocoder, args.vocoder_ckpt, device)

    semantic_mean, semantic_std = None, None
    if hasattr(cfg.model.semantic_codec, "representation_stat_mean_var_path"):
        stat = torch.load(cfg.model.semantic_codec.representation_stat_mean_var_path, map_location=device)
        semantic_mean = torch.tensor(stat["mean"], device=device)
        semantic_std = torch.sqrt(torch.tensor(stat["var"], device=device))

    pipeline = S2MelDiTInferencePipeline(
        semantic_model=semantic_model,
        processor=processor,
        semantic_codec=semantic_codec,
        t2s_model=t2s_model,
        s2mel_model=s2mel_model,
        vocoder=vocoder,
        semantic_mean=semantic_mean,
        semantic_std=semantic_std,
        cfg=cfg,
        device=device,
    )

    print("Running inference...")
    audio = pipeline.inference(
        prompt_wav_path=args.prompt_wav,
        prompt_text=args.prompt_text,
        target_text=args.target_text,
        target_len=args.target_len,
    )

    print(f"Saving to {args.output_path}")
    sf.write(args.output_path, audio, cfg.preprocess.sample_rate)
    print("Done!")


if __name__ == "__main__":
    main()
