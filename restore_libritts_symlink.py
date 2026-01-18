#!/usr/bin/env python3
import os
import sys

def ensure_symlink(src, dst):
    """创建软链接，若已存在则跳过"""
    if os.path.islink(dst) or os.path.exists(dst):
        return
    os.symlink(src, dst)

def parse_speakers(speakers_txt):
    """
    解析 SPEAKERS.txt
    返回：dict {speaker_id(str): subset_name(str)}
    """
    mapping = {}
    with open(speakers_txt, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            # 格式：
            # ID | SEX | SUBSET | MINUTES | NAME
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            speaker_id = parts[0]
            subset = parts[2]
            mapping[speaker_id] = subset
    return mapping

def main(link_root, original_root):
    link_root = os.path.abspath(link_root)
    original_root = os.path.abspath(original_root)

    os.makedirs(link_root, exist_ok=True)

    speakers_txt = os.path.join(original_root, "SPEAKERS.txt")
    train_dir = os.path.join(original_root, "train")

    if not os.path.isfile(speakers_txt):
        raise FileNotFoundError(f"SPEAKERS.txt not found: {speakers_txt}")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"train dir not found: {train_dir}")

    speaker2subset = parse_speakers(speakers_txt)

    # 需要还原的三个子集
    subsets = {
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    }

    # 创建目标子集目录
    for subset in subsets:
        os.makedirs(os.path.join(link_root, subset), exist_ok=True)

    # 处理 train 内的 speaker
    for speaker_id in os.listdir(train_dir):
        speaker_src = os.path.join(train_dir, speaker_id)
        if not os.path.isdir(speaker_src):
            continue

        subset = speaker2subset.get(speaker_id)
        if subset not in subsets:
            continue

        speaker_dst = os.path.join(link_root, subset, speaker_id)
        ensure_symlink(speaker_src, speaker_dst)

    # 其余目录和文件：直接软链接
    for name in os.listdir(original_root):
        if name == "train":
            continue
        if name in subsets:
            continue

        src = os.path.join(original_root, name)
        dst = os.path.join(link_root, name)
        ensure_symlink(src, dst)

    print("✅ LibriTTS 软链接结构还原完成")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python restore_libritts_symlink.py <link_root> <original_root>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
