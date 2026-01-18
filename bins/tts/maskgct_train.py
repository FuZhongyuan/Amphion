# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
from utils.util import load_config
from models.tts.maskgct.semantic_codec_trainer import SemanticCodecTrainer
from models.tts.maskgct.acoustic_codec_trainer import AcousticCodecTrainer
from models.tts.maskgct.t2s_trainer import T2STrainer
from models.tts.maskgct.s2a_trainer import S2ATrainer
from models.tts.maskgct.s2mel_dit_trainer import S2MelDiTTrainer
from models.tts.maskgct.s2mel_fm_trainer import S2MelFMTrainer
from models.tts.maskgct.t2s_finetune_trainer import T2SFineTuneTrainer
from models.tts.maskgct.t2s_curriculum_trainer import T2SCurriculumTrainer


def build_trainer(args, cfg):
    """Build trainer based on model type"""
    supported_trainer = {
        "SemanticCodec": SemanticCodecTrainer,
        "AcousticCodec": AcousticCodecTrainer,
        "T2S": T2STrainer,
        "S2A": S2ATrainer,
        "S2MelDiT": S2MelDiTTrainer,
        "S2MelFM": S2MelFMTrainer,
        "T2S_FineTune": T2SFineTuneTrainer,
        "T2S_Curriculum": T2SCurriculumTrainer,
    }
    
    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def cuda_relevant(deterministic=False):
    """CUDA settings"""
    torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--resume", action="store_true", help="The model name to restore"
    )
    parser.add_argument(
        "--resume_type",
        type=str,
        help="resume for continue to train, finetune for finetuning",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="checkpoint to resume",
    )
    parser.add_argument(
        "--log_level", default="warning", help="logging level (debug, info, warning)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    
    # CUDA settings
    cuda_relevant()
    
    # Build trainer
    trainer = build_trainer(args, cfg)
    
    trainer.train_loop()


if __name__ == "__main__":
    main()

