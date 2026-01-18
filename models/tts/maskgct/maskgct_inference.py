# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.tts.maskgct.maskgct_utils import *
import safetensors
import soundfile as sf
import argparse
import os
import sys
import torch
from utils.util import Logger

def main():
    parser = argparse.ArgumentParser(description="MaskGCT Inference Script")
    parser.add_argument("--cfg_path", type=str, default="./models/tts/maskgct/config/maskgct.json",
                        help="Path to config file")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts/MaskGCT-ckpt",
                        help="Directory containing local checkpoints")
    parser.add_argument("--semantic_codec_ckpt", type=str,
                        help="Path to semantic codec checkpoint")
    parser.add_argument("--codec_encoder_ckpt", type=str,
                        help="Path to acoustic codec encoder checkpoint")
    parser.add_argument("--codec_decoder_ckpt", type=str,
                        help="Path to acoustic codec decoder checkpoint")
    parser.add_argument("--t2s_model_ckpt", type=str,
                        help="Path to T2S model checkpoint")
    parser.add_argument("--s2a_1layer_ckpt", type=str,
                        help="Path to S2A 1-layer model checkpoint")
    parser.add_argument("--s2a_full_ckpt", type=str,
                        help="Path to S2A full model checkpoint")
    parser.add_argument("--prompt_wav_path", type=str, default="./models/tts/maskgct/wav/prompt.wav",
                        help="Path to prompt audio file")
    parser.add_argument("--prompt_text", type=str,
                        default=" We do not break. We never give in. We never back down.",
                        help="Prompt text")
    parser.add_argument("--target_text", type=str,
                        default="In this paper, we introduce MaskGCT, a fully non-autoregressive TTS model that eliminates the need for explicit alignment information between text and speech supervision.",
                        help="Target text to generate")
    parser.add_argument("--save_path", type=str, default="generated_audio.wav",
                        help="Path to save generated audio")
    parser.add_argument("--target_len", type=float, default=18.0,
                        help="Target duration in seconds (None for auto)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run inference on")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["debug", "info", "warning", "error"],
                        help="Logging level")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with model info saving")

    args = parser.parse_args()

    # Setup logger
    log_file = os.path.join(os.path.dirname(args.save_path), "maskgct_inference.log")
    logger = Logger(log_file, level=args.log_level).logger

    logger.info("=" * 56)
    logger.info("||\t\tMaskGCT Inference Started\t\t||")
    logger.info("=" * 56)
    logger.info(f"Config path: {args.cfg_path}")
    logger.info(f"Checkpoint directory: {args.ckpt_dir}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Debug mode: {args.debug}")

    # build model
    device = torch.device(args.device)
    cfg = load_config(args.cfg_path)

    logger.info("Building models...")
    # 1. build semantic model (w2v-bert-2.0)
    logger.info("Building semantic model...")
    semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
    if args.debug:
        logger.debug("Semantic model structure:")
        logger.debug(semantic_model)

    # 2. build semantic codec
    logger.info("Building semantic codec...")
    semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
    if args.debug:
        logger.debug("Semantic codec structure:")
        logger.debug(semantic_codec)

    # 3. build acoustic codec
    logger.info("Building acoustic codec...")
    codec_encoder, codec_decoder = build_acoustic_codec(
        cfg.model.acoustic_codec, device
    )
    if args.debug:
        logger.debug("Acoustic codec encoder structure:")
        logger.debug(codec_encoder)
        logger.debug("Acoustic codec decoder structure:")
        logger.debug(codec_decoder)

    # 4. build t2s model
    logger.info("Building T2S model...")
    t2s_model = build_t2s_model(cfg.model.t2s_model, device)
    if args.debug:
        logger.debug("T2S model structure:")
        logger.debug(t2s_model)

    # 5. build s2a model
    logger.info("Building S2A models...")
    s2a_model_1layer = build_s2a_model(cfg.model.s2a_model.s2a_1layer, device)
    s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)
    if args.debug:
        logger.debug("S2A 1-layer model structure:")
        logger.debug(s2a_model_1layer)
        logger.debug("S2A full model structure:")
        logger.debug(s2a_model_full)

    # Save model info if debug mode
    if args.debug:
        model_info_path = os.path.join(os.path.dirname(args.save_path), "model_info.txt")
        logger.info(f"Saving model information to {model_info_path}")
        with open(model_info_path, "w", encoding="utf-8") as f:
            f.write("MaskGCT Model Information\n")
            f.write("=" * 50 + "\n\n")
            f.write("Semantic Model:\n")
            f.write(str(semantic_model) + "\n\n")
            f.write("Semantic Codec:\n")
            f.write(str(semantic_codec) + "\n\n")
            f.write("Acoustic Codec Encoder:\n")
            f.write(str(codec_encoder) + "\n\n")
            f.write("Acoustic Codec Decoder:\n")
            f.write(str(codec_decoder) + "\n\n")
            f.write("T2S Model:\n")
            f.write(str(t2s_model) + "\n\n")
            f.write("S2A 1-Layer Model:\n")
            f.write(str(s2a_model_1layer) + "\n\n")
            f.write("S2A Full Model:\n")
            f.write(str(s2a_model_full) + "\n\n")
        logger.info("Model information saved successfully.")

    # Load checkpoints from local directory
    logger.info("Loading checkpoints from local directory...")

    # Define checkpoint paths - use individual paths if provided, otherwise use directory structure
    semantic_codec_ckpt = (args.semantic_codec_ckpt or
                          os.path.join(args.ckpt_dir, "semantic_codec", "model.safetensors"))
    codec_encoder_ckpt = (args.codec_encoder_ckpt or
                         os.path.join(args.ckpt_dir, "acoustic_codec", "model.safetensors"))
    codec_decoder_ckpt = (args.codec_decoder_ckpt or
                         os.path.join(args.ckpt_dir, "acoustic_codec", "model_1.safetensors"))
    t2s_model_ckpt = (args.t2s_model_ckpt or
                     os.path.join(args.ckpt_dir, "t2s_model", "model.safetensors"))
    s2a_1layer_ckpt = (args.s2a_1layer_ckpt or
                      os.path.join(args.ckpt_dir, "s2a_model", "s2a_model_1layer", "model.safetensors"))
    s2a_full_ckpt = (args.s2a_full_ckpt or
                    os.path.join(args.ckpt_dir, "s2a_model", "s2a_model_full", "model.safetensors"))

    # Check if checkpoints exist
    checkpoints = [
        ("semantic_codec", semantic_codec_ckpt),
        ("codec_encoder", codec_encoder_ckpt),
        ("codec_decoder", codec_decoder_ckpt),
        ("t2s_model", t2s_model_ckpt),
        ("s2a_1layer", s2a_1layer_ckpt),
        ("s2a_full", s2a_full_ckpt)
    ]

    missing_checkpoints = []
    for name, path in checkpoints:
        if not os.path.exists(path):
            missing_checkpoints.append(f"{name}: {path}")

    if missing_checkpoints:
        logger.error("Missing checkpoints:")
        for missing in missing_checkpoints:
            logger.error(f"  {missing}")
        logger.error("Please ensure all checkpoints are downloaded and placed in the correct directories.")
        sys.exit(1)

    # Load checkpoints
    logger.info("Loading semantic codec checkpoint...")
    safetensors.torch.load_model(semantic_codec, semantic_codec_ckpt)

    logger.info("Loading acoustic codec checkpoints...")
    safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
    safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)

    logger.info("Loading T2S model checkpoint...")
    safetensors.torch.load_model(t2s_model, t2s_model_ckpt)

    logger.info("Loading S2A model checkpoints...")
    safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
    safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)

    logger.info("All checkpoints loaded successfully.")

    # Create inference pipeline
    logger.info("Creating MaskGCT inference pipeline...")
    maskgct_inference_pipeline = MaskGCT_Inference_Pipeline(
        semantic_model,
        semantic_codec,
        codec_encoder,
        codec_decoder,
        t2s_model,
        s2a_model_1layer,
        s2a_model_full,
        semantic_mean,
        semantic_std,
        device,
    )

    # Run inference
    logger.info("Starting inference...")
    logger.info(f"Prompt audio: {args.prompt_wav_path}")
    logger.info(f"Prompt text: {args.prompt_text}")
    logger.info(f"Target text: {args.target_text}")
    logger.info(f"Target length: {args.target_len} seconds")
    logger.info(f"Save path: {args.save_path}")

    recovered_audio = maskgct_inference_pipeline.maskgct_inference(
        args.prompt_wav_path, args.prompt_text, args.target_text, "en", "en",
        target_len=args.target_len if args.target_len > 0 else None
    )

    # Save audio
    logger.info(f"Saving generated audio to {args.save_path}")
    sf.write(args.save_path, recovered_audio, 24000)
    logger.info("Inference completed successfully!")

    logger.info("=" * 56)
    logger.info("||\t\tMaskGCT Inference Finished\t\t||")
    logger.info("=" * 56)


if __name__ == "__main__":
    main()
