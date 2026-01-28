#!/usr/bin/env python3
"""
Quantization Consistency Checker

This script quantizes the same hidden states multiple times and compares the results
to detect any non-deterministic behavior in the quantization process.

Usage:
    python check_quantization_consistency.py --config config.json --num_runs 5

The script will:
1. Load hidden states from HDF5 files
2. Quantize them multiple times
3. Compare results across runs
4. Generate detailed statistics and reports
"""

import os
import json
import argparse
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from utils.util import load_config


class SemanticCodec:
    """Wrapper for RepCodec model to quantize semantic features."""

    def __init__(
        self,
        cfg,
        device: str = "cuda"
    ):
        """Initialize the semantic codec.
        
        Args:
            cfg: Configuration object containing codec parameters (e.g., codebook_size, hidden_size, etc.)
            device: Device to run the model on
        """
        from models.codec.kmeans.repcodec_model import RepCodec
        import safetensors.torch

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize model with config
        self.model = RepCodec(cfg=cfg)

        # Load pretrained weights
        pretrained_path = getattr(cfg, "pretrained_path", None)
        if pretrained_path:
            if os.path.isdir(pretrained_path):
                model_path = os.path.join(pretrained_path, "model.safetensors")
                if not os.path.exists(model_path):
                    model_path = os.path.join(pretrained_path, "pytorch_model.bin")
            else:
                model_path = pretrained_path

            if os.path.exists(model_path):
                print(f"Loading semantic codec from: {model_path}")
                if model_path.endswith(".safetensors"):
                    safetensors.torch.load_model(self.model, model_path)
                else:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    if isinstance(checkpoint, dict) and "model" in checkpoint:
                        self.model.load_state_dict(checkpoint["model"], strict=False)
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model.eval()
        self.model.to(self.device)

        # Load normalization statistics
        self.semantic_mean = None
        self.semantic_std = None
        stat_path = getattr(cfg, "representation_stat_mean_var_path", None)
        if stat_path and os.path.exists(stat_path):
            print(f"Loading normalization statistics from: {stat_path}")
            stat = torch.load(stat_path, map_location=self.device)
            self.semantic_mean = torch.tensor(stat["mean"]).to(self.device)
            self.semantic_std = torch.sqrt(torch.tensor(stat["var"])).to(self.device)

        print(f"Semantic codec loaded on device: {self.device}")
        codebook_size = getattr(cfg, "codebook_size", None)
        hidden_size = getattr(cfg, "hidden_size", None)
        print(f"Codebook size: {codebook_size}, Hidden size: {hidden_size}")

    @torch.no_grad()
    def quantize(self, hidden_states: np.ndarray) -> np.ndarray:
        """Quantize hidden states to semantic codes."""
        # Convert to tensor
        feat = torch.from_numpy(hidden_states).float().to(self.device)

        # Add batch dimension if needed
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Normalize
        if self.semantic_mean is not None:
            feat = (feat - self.semantic_mean.to(feat)) / self.semantic_std.to(feat)

        # Quantize
        semantic_codes, _ = self.model.quantize(feat)

        # Remove batch dimension if added
        if squeeze_output:
            semantic_codes = semantic_codes.squeeze(0)

        return semantic_codes.cpu().numpy()


class QuantizationConsistencyChecker:
    """Check consistency of quantization across multiple runs."""

    def __init__(
        self,
        input_dir: str,
        semantic_codec: SemanticCodec,
        num_runs: int = 5,
        num_samples: int = 100,
        output_dir: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        """Initialize the consistency checker.

        Args:
            input_dir: Directory containing input HDF5 files with hidden_states
            semantic_codec: SemanticCodec instance for quantization
            num_runs: Number of times to quantize each sample
            num_samples: Number of samples to test (None = all samples)
            output_dir: Directory to save analysis results
            random_seed: Random seed for reproducibility (if None, tests randomness)
        """
        self.input_dir = Path(input_dir)
        self.semantic_codec = semantic_codec
        self.num_runs = num_runs
        self.num_samples = num_samples
        self.random_seed = random_seed

        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path("consistency_check_results")
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load input index
        self.input_index_path = self.input_dir / "hdf5_index.json"
        if not self.input_index_path.exists():
            raise FileNotFoundError(f"Input HDF5 index not found: {self.input_index_path}")

        with open(self.input_index_path, 'r') as f:
            self.input_index = json.load(f)

        # Results storage
        self.results = {
            "sample_consistency": {},  # Per-sample consistency metrics
            "global_stats": {},  # Overall statistics
            "inconsistent_samples": [],  # Samples with differences
        }

    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

    def compare_codes(self, codes_list: List[np.ndarray]) -> Dict[str, Any]:
        """Compare multiple quantization results for the same input.

        Args:
            codes_list: List of semantic codes from different runs

        Returns:
            Dictionary with comparison metrics
        """
        num_runs = len(codes_list)
        
        # Check if all codes are identical
        all_identical = all(np.array_equal(codes_list[0], codes) for codes in codes_list[1:])

        # Calculate pairwise differences
        pairwise_diffs = []
        for i in range(num_runs):
            for j in range(i + 1, num_runs):
                diff = np.sum(codes_list[i] != codes_list[j])
                diff_ratio = diff / len(codes_list[i])
                pairwise_diffs.append({
                    "run_pair": (i, j),
                    "num_differences": int(diff),
                    "difference_ratio": float(diff_ratio)
                })

        # Calculate per-position variance
        codes_array = np.stack(codes_list, axis=0)  # [num_runs, T]
        position_variance = []
        
        for pos in range(codes_array.shape[1]):
            unique_codes = len(np.unique(codes_array[:, pos]))
            position_variance.append(unique_codes)

        # Statistics
        max_diff = max([d["num_differences"] for d in pairwise_diffs])
        mean_diff = np.mean([d["num_differences"] for d in pairwise_diffs])
        max_diff_ratio = max([d["difference_ratio"] for d in pairwise_diffs])
        mean_diff_ratio = np.mean([d["difference_ratio"] for d in pairwise_diffs])

        # Count positions with variance
        positions_with_variance = sum([1 for v in position_variance if v > 1])
        variance_ratio = positions_with_variance / len(position_variance)

        return {
            "all_identical": all_identical,
            "num_runs": num_runs,
            "sequence_length": len(codes_list[0]),
            "pairwise_differences": pairwise_diffs,
            "max_differences": int(max_diff),
            "mean_differences": float(mean_diff),
            "max_difference_ratio": float(max_diff_ratio),
            "mean_difference_ratio": float(mean_diff_ratio),
            "positions_with_variance": int(positions_with_variance),
            "position_variance_ratio": float(variance_ratio),
            "position_variance_details": position_variance
        }

    def check_sample(self, sample_key: str, hidden_states: np.ndarray) -> Dict[str, Any]:
        """Quantize a single sample multiple times and check consistency."""
        codes_list = []

        for run_idx in range(self.num_runs):
            # Set seed for this run if specified
            if self.random_seed is not None:
                self.set_random_seed(self.random_seed + run_idx)

            # Quantize
            codes = self.semantic_codec.quantize(hidden_states)
            codes_list.append(codes)

        # Compare results
        comparison = self.compare_codes(codes_list)
        comparison["sample_key"] = sample_key
        
        return comparison

    def check_consistency(self) -> Dict[str, Any]:
        """Check quantization consistency across multiple samples."""
        sample_index = self.input_index.get("sample_index", {})
        
        # Select samples to test
        sample_keys = list(sample_index.keys())
        if self.num_samples and self.num_samples < len(sample_keys):
            # Randomly select samples
            np.random.seed(42)  # Fixed seed for sample selection
            sample_keys = np.random.choice(sample_keys, self.num_samples, replace=False)

        print(f"Testing {len(sample_keys)} samples with {self.num_runs} runs each...")
        print(f"Random seed: {self.random_seed if self.random_seed is not None else 'None (testing randomness)'}")

        # Group samples by file
        file_to_samples: Dict[int, List[str]] = defaultdict(list)
        for sample_key in sample_keys:
            file_idx = sample_index[sample_key]["file_idx"]
            file_to_samples[file_idx].append(sample_key)

        # Process samples
        all_identical_count = 0
        has_differences_count = 0
        total_samples = 0

        for file_idx in tqdm(sorted(file_to_samples.keys()), desc="Processing files"):
            input_path = self.input_dir / f"features_{file_idx:05d}.h5"
            if not input_path.exists():
                print(f"Warning: Input file not found: {input_path}")
                continue

            samples = file_to_samples[file_idx]

            with h5py.File(input_path, 'r') as input_file:
                for sample_key in tqdm(samples, desc=f"File {file_idx}", leave=False):
                    try:
                        # Read hidden states
                        if sample_key not in input_file:
                            continue

                        sample_group = input_file[sample_key]
                        if "hidden_states" not in sample_group:
                            continue

                        hidden_states = sample_group["hidden_states"][:]
                        
                        # Convert float16 to float32 if needed
                        if hidden_states.dtype == np.float16:
                            hidden_states = hidden_states.astype(np.float32)

                        # Check consistency
                        result = self.check_sample(sample_key, hidden_states)
                        
                        # Store result
                        self.results["sample_consistency"][sample_key] = result

                        # Update counters
                        total_samples += 1
                        if result["all_identical"]:
                            all_identical_count += 1
                        else:
                            has_differences_count += 1
                            self.results["inconsistent_samples"].append({
                                "sample_key": sample_key,
                                "max_differences": result["max_differences"],
                                "mean_differences": result["mean_differences"],
                                "max_difference_ratio": result["max_difference_ratio"],
                            })

                    except Exception as e:
                        print(f"Error processing {sample_key}: {e}")

        # Calculate global statistics
        if total_samples > 0:
            self.results["global_stats"] = {
                "total_samples_tested": total_samples,
                "all_identical_samples": all_identical_count,
                "samples_with_differences": has_differences_count,
                "consistency_rate": all_identical_count / total_samples,
                "inconsistency_rate": has_differences_count / total_samples,
                "num_runs_per_sample": self.num_runs,
                "random_seed_used": self.random_seed,
            }

            # Calculate aggregate statistics for inconsistent samples
            if has_differences_count > 0:
                all_max_diffs = [s["max_differences"] for s in self.results["inconsistent_samples"]]
                all_mean_diffs = [s["mean_differences"] for s in self.results["inconsistent_samples"]]
                all_max_ratios = [s["max_difference_ratio"] for s in self.results["inconsistent_samples"]]

                self.results["global_stats"]["inconsistency_statistics"] = {
                    "max_differences_across_samples": {
                        "min": float(np.min(all_max_diffs)),
                        "max": float(np.max(all_max_diffs)),
                        "mean": float(np.mean(all_max_diffs)),
                        "median": float(np.median(all_max_diffs)),
                        "std": float(np.std(all_max_diffs)),
                    },
                    "max_difference_ratios": {
                        "min": float(np.min(all_max_ratios)),
                        "max": float(np.max(all_max_ratios)),
                        "mean": float(np.mean(all_max_ratios)),
                        "median": float(np.median(all_max_ratios)),
                        "std": float(np.std(all_max_ratios)),
                    }
                }

        return self.results

    def generate_report(self):
        """Generate detailed report with visualizations."""
        print("\n" + "=" * 80)
        print("QUANTIZATION CONSISTENCY CHECK REPORT")
        print("=" * 80)

        stats = self.results["global_stats"]
        
        print(f"\nConfiguration:")
        print(f"  - Number of samples tested: {stats['total_samples_tested']}")
        print(f"  - Runs per sample: {stats['num_runs_per_sample']}")
        print(f"  - Random seed: {stats['random_seed_used']}")

        print(f"\n{'Consistency Results:'}")
        print(f"  - Samples with identical results: {stats['all_identical_samples']} ({stats['consistency_rate']:.2%})")
        print(f"  - Samples with differences: {stats['samples_with_differences']} ({stats['inconsistency_rate']:.2%})")

        if stats['samples_with_differences'] > 0:
            print(f"\n{'Inconsistency Details:'}")
            inc_stats = stats.get("inconsistency_statistics", {})
            
            if "max_differences_across_samples" in inc_stats:
                max_diffs = inc_stats["max_differences_across_samples"]
                print(f"  Max differences (absolute):")
                print(f"    - Range: {max_diffs['min']:.1f} - {max_diffs['max']:.1f}")
                print(f"    - Mean: {max_diffs['mean']:.2f} ± {max_diffs['std']:.2f}")
                print(f"    - Median: {max_diffs['median']:.1f}")

            if "max_difference_ratios" in inc_stats:
                ratios = inc_stats["max_difference_ratios"]
                print(f"  Max differences (ratio):")
                print(f"    - Range: {ratios['min']:.4f} - {ratios['max']:.4f}")
                print(f"    - Mean: {ratios['mean']:.6f} ± {ratios['std']:.6f}")
                print(f"    - Median: {ratios['median']:.6f}")

            # Show top inconsistent samples
            print(f"\n{'Top 10 most inconsistent samples:'}")
            sorted_inconsistent = sorted(
                self.results["inconsistent_samples"],
                key=lambda x: x["max_difference_ratio"],
                reverse=True
            )[:10]
            
            for i, sample in enumerate(sorted_inconsistent, 1):
                print(f"  {i}. {sample['sample_key']}")
                print(f"     Max diff: {sample['max_differences']}, Ratio: {sample['max_difference_ratio']:.6f}")

        print("\n" + "=" * 80)

        # Save detailed JSON report
        report_path = self.output_dir / "consistency_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed report saved to: {report_path}")

        # Generate visualizations
        if stats['samples_with_differences'] > 0:
            self.generate_visualizations()

    def generate_visualizations(self):
        """Generate visualization plots."""
        try:
            # Plot 1: Distribution of difference ratios
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            ratios = [s["max_difference_ratio"] for s in self.results["inconsistent_samples"]]
            plt.hist(ratios, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Max Difference Ratio')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Maximum Difference Ratios')
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            diffs = [s["max_differences"] for s in self.results["inconsistent_samples"]]
            plt.hist(diffs, bins=30, edgecolor='black', alpha=0.7)
            plt.xlabel('Max Differences (absolute)')
            plt.ylabel('Number of Samples')
            plt.title('Distribution of Maximum Differences')
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.output_dir / "difference_distributions.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {plot_path}")

        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")




def main():
    parser = argparse.ArgumentParser(
        description="Check quantization consistency by running multiple times"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=5,
        help="Number of times to quantize each sample"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to test (None = all)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (None = test randomness)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    input_dir = getattr(cfg, "input_dir", None)
    device = getattr(cfg, "device", "cuda")

    # Validate required parameters
    if not input_dir:
        raise ValueError("input_dir is required in config file")
    if not hasattr(cfg, "semantic_codec"):
        raise ValueError("semantic_codec is required in config file")
    if not hasattr(cfg.semantic_codec, "pretrained_path"):
        raise ValueError("semantic_codec.pretrained_path is required in config file")

    print("=" * 80)
    print("QUANTIZATION CONSISTENCY CHECKER")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Number of runs per sample: {args.num_runs}")
    print(f"Number of samples to test: {args.num_samples}")
    print(f"Random seed: {args.random_seed if args.random_seed is not None else 'None (testing randomness)'}")
    print(f"Device: {device}")
    print("=" * 80)

    # Initialize semantic codec
    semantic_codec = SemanticCodec(
        cfg=cfg.semantic_codec,
        device=device
    )

    # Initialize checker
    checker = QuantizationConsistencyChecker(
        input_dir=input_dir,
        semantic_codec=semantic_codec,
        num_runs=args.num_runs,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        random_seed=args.random_seed
    )

    # Run consistency check
    results = checker.check_consistency()

    # Generate report
    checker.generate_report()

    print("\nDone!")


if __name__ == "__main__":
    main()