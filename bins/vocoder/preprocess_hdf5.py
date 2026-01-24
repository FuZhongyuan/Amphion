# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
HDF5-based preprocessing for vocoder training.
Extracts acoustic features and stores them in HDF5 format for efficient loading.
"""

import faulthandler
faulthandler.enable()

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.util import load_config
from utils.hdf5_utils import HDF5Writer, AsyncHDF5Saver
from preprocessors.processor import preprocess_dataset
from preprocessors.metadata import cal_metadata


def extract_vocoder_features_single(cfg, utt, dataset_output):
    """Extract acoustic features for a single utterance.

    Args:
        cfg: Configuration object
        utt: Utterance metadata dict
        dataset_output: Output directory

    Returns:
        Tuple of (uid, features_dict) or (uid, None) on failure
    """
    from utils import audio
    from utils.mel import extract_mel_features

    uid = utt["Uid"]
    wav_path = utt["Path"]

    try:
        with torch.no_grad():
            # Load audio
            wav_torch, _ = audio.load_audio_torch(wav_path, cfg.preprocess.sample_rate)
            wav = wav_torch.cpu().numpy()

            features = {}

            # Extract mel spectrogram
            if cfg.preprocess.extract_mel:
                mel = extract_mel_features(wav_torch.unsqueeze(0), cfg.preprocess)
                features["mel"] = mel.cpu().numpy()

            # Extract energy
            if cfg.preprocess.extract_energy:
                if (
                    cfg.preprocess.energy_extract_mode == "from_mel"
                    and cfg.preprocess.extract_mel
                ):
                    mel_tensor = torch.from_numpy(features["mel"])
                    energy = (mel_tensor.exp() ** 2).sum(0).sqrt().numpy()
                elif cfg.preprocess.energy_extract_mode == "from_waveform":
                    energy = audio.energy(wav, cfg.preprocess)
                else:
                    energy = None

                if energy is not None:
                    features["energy"] = energy

            # Extract pitch
            if cfg.preprocess.extract_pitch:
                from utils import f0
                pitch = f0.get_f0(wav, cfg.preprocess)
                features["pitch"] = pitch

                if cfg.preprocess.extract_uv:
                    uv = (pitch != 0).astype(np.float32)
                    features["uv"] = uv

            # Extract amplitude and phase
            if cfg.preprocess.extract_amplitude_phase:
                from utils.mel import amplitude_phase_spectrum
                log_amplitude, phase, real, imaginary = amplitude_phase_spectrum(
                    wav_torch.unsqueeze(0), cfg.preprocess
                )
                features["log_amplitude"] = log_amplitude
                features["phase"] = phase
                features["real"] = real
                features["imaginary"] = imaginary

            # Extract audio waveform
            if cfg.preprocess.extract_audio:
                features["audio"] = wav

            return uid, features

    except Exception as e:
        print(f"Error processing {uid}: {e}")
        return uid, None


def extract_acoustic_features_hdf5(dataset, output_path, cfg, n_workers=1):
    """Extract acoustic features and save to HDF5 format.

    Args:
        dataset: Dataset name
        output_path: Output directory path
        cfg: Configuration object
        n_workers: Number of worker threads
    """
    types = ["train", "test"] if "eval" not in dataset else ["test"]
    metadata = []

    dataset_output = os.path.join(output_path, dataset)

    for dataset_type in tqdm(types, desc=f"Loading {dataset} metadata"):
        dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    print(f"Total utterances to process: {len(metadata)}")

    # Get HDF5 configuration
    samples_per_hdf5 = getattr(cfg.preprocess, 'samples_per_hdf5', 10000)
    use_float16 = getattr(cfg.preprocess, 'use_float16', False)
    compression = getattr(cfg.preprocess, 'hdf5_compression', 'gzip')
    compression_opts = getattr(cfg.preprocess, 'hdf5_compression_opts', 4)

    print(f"HDF5 config: samples_per_file={samples_per_hdf5}, "
          f"use_float16={use_float16}, compression={compression}")

    # Initialize HDF5 writer
    hdf5_writer = HDF5Writer(
        base_dir=dataset_output,
        samples_per_file=samples_per_hdf5,
        use_float16=use_float16,
        compression=compression,
        compression_opts=compression_opts
    )

    # Check which samples already exist
    samples_to_process = []
    overwrite = getattr(cfg.preprocess, 'overwrite_existing', False)

    for utt in metadata:
        uid = utt["Uid"]
        if not hdf5_writer.sample_exists(uid) or overwrite:
            samples_to_process.append(utt)

    print(f"Samples to process (new/overwrite): {len(samples_to_process)}")

    if len(samples_to_process) == 0:
        print("All samples already processed. Skipping...")
        hdf5_writer.close()
        return

    # Process samples
    success_count = 0
    error_count = 0

    if n_workers <= 1:
        # Single-threaded processing
        for utt in tqdm(samples_to_process, desc="Extracting features"):
            uid, features = extract_vocoder_features_single(cfg, utt, dataset_output)
            if features is not None:
                if hdf5_writer.write_sample(uid, features):
                    success_count += 1
                else:
                    error_count += 1
            else:
                error_count += 1
    else:
        # Multi-threaded processing with async saving
        num_save_workers = min(4, n_workers)
        async_saver = AsyncHDF5Saver(hdf5_writer, num_workers=num_save_workers)
        save_futures = []

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit extraction tasks
            future_to_utt = {
                executor.submit(
                    extract_vocoder_features_single, cfg, utt, dataset_output
                ): utt
                for utt in samples_to_process
            }

            with tqdm(total=len(samples_to_process), desc="Extracting features") as pbar:
                for future in as_completed(future_to_utt):
                    uid, features = future.result()
                    if features is not None:
                        # Async save
                        save_future = async_saver.save_async(uid, features)
                        save_futures.append((uid, save_future))
                        success_count += 1
                    else:
                        error_count += 1
                    pbar.update(1)

        # Wait for all saves to complete
        print("\nWaiting for saves to complete...")
        for uid, save_future in tqdm(save_futures, desc="Finalizing saves"):
            try:
                result = save_future.result()
                if not result:
                    print(f"Failed to save {uid}")
            except Exception as e:
                print(f"Save error for {uid}: {e}")

        async_saver.shutdown()

    # Close HDF5 writer (saves index)
    hdf5_writer.close()

    print(f"\nExtraction completed!")
    print(f"Success: {success_count}, Errors: {error_count}")
    print(f"HDF5 files saved to: {dataset_output}")


def cal_mel_min_max_hdf5(dataset, output_path, cfg):
    """Calculate mel min/max statistics from HDF5 files.

    Args:
        dataset: Dataset name
        output_path: Output directory path
        cfg: Configuration object
    """
    from utils.hdf5_utils import HDF5Reader

    dataset_output = os.path.join(output_path, dataset)

    # Load metadata
    metadata = []
    for dataset_type in ["train", "test"] if "eval" not in dataset else ["test"]:
        dataset_file = os.path.join(dataset_output, "{}.json".format(dataset_type))
        with open(dataset_file, "r") as f:
            metadata.extend(json.load(f))

    # Initialize HDF5 reader
    hdf5_reader = HDF5Reader(dataset_output)

    tmp_mel_min = []
    tmp_mel_max = []

    for item in tqdm(metadata, desc="Calculating mel statistics"):
        uid = item["Uid"]
        mel = hdf5_reader.read_feature(uid, "mel")

        if mel is None:
            continue

        if mel.shape[0] != cfg.preprocess.n_mel:
            mel = mel.T

        assert mel.shape[0] == cfg.preprocess.n_mel

        tmp_mel_min.append(np.min(mel, axis=-1))
        tmp_mel_max.append(np.max(mel, axis=-1))

    hdf5_reader.close()

    if len(tmp_mel_min) == 0:
        print("No mel features found!")
        return

    mel_min = np.min(tmp_mel_min, axis=0)
    mel_max = np.max(tmp_mel_max, axis=0)

    # Save mel min max data
    mel_min_max_dir = os.path.join(dataset_output, cfg.preprocess.mel_min_max_stats_dir)
    os.makedirs(mel_min_max_dir, exist_ok=True)

    mel_min_path = os.path.join(mel_min_max_dir, "mel_min.npy")
    mel_max_path = os.path.join(mel_min_max_dir, "mel_max.npy")
    np.save(mel_min_path, mel_min)
    np.save(mel_max_path, mel_max)

    print(f"Mel statistics saved to {mel_min_max_dir}")


def preprocess_hdf5(cfg, args):
    """Preprocess raw data and save to HDF5 format.

    Args:
        cfg: Configuration object
        args: Command line arguments
    """
    output_path = cfg.preprocess.processed_dir
    os.makedirs(output_path, exist_ok=True)

    # Split train and test sets
    for dataset in tqdm(cfg.dataset, desc="Preprocessing datasets"):
        print(f"Preprocess {dataset}...")

        preprocess_dataset(
            dataset,
            cfg.dataset_path[dataset],
            output_path,
            cfg.preprocess,
            cfg.task_type,
            is_custom_dataset=dataset in cfg.use_custom_dataset
            if hasattr(cfg, 'use_custom_dataset') else False,
        )

    # Dump metadata
    cal_metadata(cfg)

    # Extract acoustic features to HDF5
    for dataset in tqdm(cfg.dataset, desc="Extracting acoustic features"):
        if (
            "pitch_shift" in dataset
            or "formant_shift" in dataset
            or "equalizer" in dataset
        ):
            continue

        print(f"\nExtracting acoustic features for {dataset} using {args.num_workers} workers...")
        extract_acoustic_features_hdf5(dataset, output_path, cfg, args.num_workers)

        # Calculate mel statistics
        if cfg.preprocess.mel_min_max_norm:
            cal_mel_min_max_hdf5(dataset, output_path, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config.json", help="json files for configurations."
    )
    parser.add_argument("--num_workers", type=int, default=int(cpu_count()))

    args = parser.parse_args()
    cfg = load_config(args.config)

    preprocess_hdf5(cfg, args)


if __name__ == "__main__":
    main()
