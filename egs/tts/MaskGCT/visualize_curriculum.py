#!/usr/bin/env python3
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualize T2S Curriculum Learning Schedule

This script generates plots showing how curriculum parameters change over training.
"""

import matplotlib.pyplot as plt
import numpy as np


def get_curriculum_params(step, total_steps=100000, stage1_ratio=0.3, stage2_ratio=0.7):
    """Get curriculum parameters for a given training step"""
    stage1_steps = int(total_steps * stage1_ratio)
    stage2_steps = int(total_steps * stage2_ratio)

    if step < stage1_steps:
        # Stage 1
        return {
            "mask_ratio_range": (0.10, 0.30),
            "prompt_ratio_range": (0.40, 0.60),
            "cfg_dropout": 0.0,
            "stage": 1
        }
    elif step < stage2_steps:
        # Stage 2
        return {
            "mask_ratio_range": (0.20, 0.50),
            "prompt_ratio_range": (0.20, 0.50),
            "cfg_dropout": 0.05,
            "stage": 2
        }
    else:
        # Stage 3
        return {
            "mask_ratio_range": (0.30, 0.80),
            "prompt_ratio_range": (0.10, 0.40),
            "cfg_dropout": 0.15,
            "stage": 3
        }


def plot_curriculum_schedule(total_steps=100000, output_file="curriculum_schedule.png"):
    """Plot curriculum schedule over training"""
    steps = np.arange(0, total_steps, 100)

    mask_min = []
    mask_max = []
    prompt_min = []
    prompt_max = []
    cfg_dropout = []
    stages = []

    for step in steps:
        params = get_curriculum_params(step, total_steps)
        mask_min.append(params["mask_ratio_range"][0])
        mask_max.append(params["mask_ratio_range"][1])
        prompt_min.append(params["prompt_ratio_range"][0])
        prompt_max.append(params["prompt_ratio_range"][1])
        cfg_dropout.append(params["cfg_dropout"])
        stages.append(params["stage"])

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('T2S Curriculum Learning Schedule', fontsize=16, fontweight='bold')

    # Plot 1: Mask Ratio Range
    ax1 = axes[0, 0]
    ax1.fill_between(steps, mask_min, mask_max, alpha=0.3, color='blue', label='Mask Ratio Range')
    ax1.plot(steps, mask_min, 'b--', linewidth=2, label='Min Mask Ratio')
    ax1.plot(steps, mask_max, 'b-', linewidth=2, label='Max Mask Ratio')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Mask Ratio', fontsize=12)
    ax1.set_title('Mask Ratio Schedule', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, total_steps)
    ax1.set_ylim(0, 1.0)

    # Add stage boundaries
    ax1.axvline(x=total_steps*0.3, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax1.axvline(x=total_steps*0.7, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax1.text(total_steps*0.15, 0.95, 'Stage 1\nEasy', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax1.text(total_steps*0.5, 0.95, 'Stage 2\nMedium', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax1.text(total_steps*0.85, 0.95, 'Stage 3\nHard', ha='center', fontsize=11,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    # Plot 2: Prompt Ratio Range
    ax2 = axes[0, 1]
    ax2.fill_between(steps, prompt_min, prompt_max, alpha=0.3, color='green', label='Prompt Ratio Range')
    ax2.plot(steps, prompt_min, 'g--', linewidth=2, label='Min Prompt Ratio')
    ax2.plot(steps, prompt_max, 'g-', linewidth=2, label='Max Prompt Ratio')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Prompt Ratio', fontsize=12)
    ax2.set_title('Prompt Length Schedule', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, total_steps)
    ax2.set_ylim(0, 0.7)

    # Add stage boundaries
    ax2.axvline(x=total_steps*0.3, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax2.axvline(x=total_steps*0.7, color='red', linestyle=':', alpha=0.5, linewidth=2)

    # Plot 3: CFG Dropout
    ax3 = axes[1, 0]
    ax3.plot(steps, cfg_dropout, 'r-', linewidth=3, label='CFG Dropout Probability')
    ax3.fill_between(steps, 0, cfg_dropout, alpha=0.3, color='red')
    ax3.set_xlabel('Training Step', fontsize=12)
    ax3.set_ylabel('CFG Dropout Probability', fontsize=12)
    ax3.set_title('Classifier-Free Guidance Dropout', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, total_steps)
    ax3.set_ylim(0, 0.2)

    # Add stage boundaries
    ax3.axvline(x=total_steps*0.3, color='red', linestyle=':', alpha=0.5, linewidth=2)
    ax3.axvline(x=total_steps*0.7, color='red', linestyle=':', alpha=0.5, linewidth=2)

    # Plot 4: Training Stage
    ax4 = axes[1, 1]
    stage_colors = ['lightgreen', 'lightyellow', 'lightcoral']
    for i, stage in enumerate([1, 2, 3], 1):
        stage_mask = np.array(stages) == stage
        if np.any(stage_mask):
            stage_steps = steps[stage_mask]
            ax4.fill_between(stage_steps, 0, 1, alpha=0.5, color=stage_colors[i-1],
                           label=f'Stage {stage}')

    ax4.set_xlabel('Training Step', fontsize=12)
    ax4.set_ylabel('', fontsize=12)
    ax4.set_title('Training Stages', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=12, loc='center')
    ax4.set_xlim(0, total_steps)
    ax4.set_ylim(0, 1)
    ax4.set_yticks([])
    ax4.grid(True, alpha=0.3, axis='x')

    # Add annotations
    ax4.text(total_steps*0.15, 0.5, 'Stage 1: Foundation\n\n• Low mask ratio (10-30%)\n• Long prompts (40-60%)\n• No CFG dropout\n• Learn basic alignment',
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.text(total_steps*0.5, 0.5, 'Stage 2: Refinement\n\n• Medium mask (20-50%)\n• Medium prompts (20-50%)\n• 5% CFG dropout\n• Improve quality',
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax4.text(total_steps*0.85, 0.5, 'Stage 3: Mastery\n\n• High mask (30-80%)\n• Short prompts (10-40%)\n• 15% CFG dropout\n• Match original T2S',
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Curriculum schedule visualization saved to: {output_file}")

    # Also create a summary table
    print("\nCurriculum Schedule Summary:")
    print("=" * 80)
    print(f"{'Stage':<10} {'Steps':<20} {'Mask Ratio':<20} {'Prompt Ratio':<20} {'CFG Dropout':<15}")
    print("=" * 80)

    stages_info = [
        (1, f"0-{int(total_steps*0.3):,}", "10-30%", "40-60%", "0%"),
        (2, f"{int(total_steps*0.3):,}-{int(total_steps*0.7):,}", "20-50%", "20-50%", "5%"),
        (3, f"{int(total_steps*0.7):,}-{total_steps:,}", "30-80%", "10-40%", "15%"),
    ]

    for stage, steps_range, mask, prompt, cfg in stages_info:
        print(f"{stage:<10} {steps_range:<20} {mask:<20} {prompt:<20} {cfg:<15}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize T2S curriculum schedule")
    parser.add_argument("--total_steps", type=int, default=100000,
                       help="Total training steps")
    parser.add_argument("--output", type=str, default="curriculum_schedule.png",
                       help="Output file path")

    args = parser.parse_args()

    plot_curriculum_schedule(args.total_steps, args.output)
