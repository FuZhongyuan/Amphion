#!/usr/bin/env python3
"""
批量计算多个数据集的mel统计量
"""

import sys
from pathlib import Path
from compute_mel_statistics import compute_mel_statistics_online
import json


def main():
    # 数据集目录
    base_dir = Path("/mnt/storage/fuzhongyuan_space/fzy_project/Amphion/data/processed_maskgct")
    
    datasets = ["ljspeech", "libritts"]
    
    all_results = {}
    
    for dataset in datasets:
        processed_dir = base_dir / dataset
        
        if not processed_dir.exists():
            print(f"跳过不存在的目录: {processed_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            results = compute_mel_statistics_online(processed_dir)
            all_results[dataset] = results
            
            # 保存单个数据集的结果
            output_path = processed_dir / "mel_statistics.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"结果已保存到: {output_path}")
            
        except Exception as e:
            print(f"处理 {dataset} 时出错: {e}")
            continue
    
    # 保存汇总结果
    summary_path = base_dir / "mel_statistics_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("汇总结果")
    print(f"{'='*60}")
    
    for dataset, results in all_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"  样本数: {results['sample_count']}")
        print(f"  总帧数: {results['total_frames']}")
        print(f"  Mel维度: {results['mel_dim']}")
        print(f"  均值: {results['global_stats']['mean']:.4f}")
        print(f"  方差: {results['global_stats']['variance']:.4f}")
        print(f"  标准差: {results['global_stats']['std']:.4f}")
        print(f"\n  配置文件参数:")
        print(f'    "mel_mean": {results["global_stats"]["mean"]:.2f},')
        print(f'    "mel_var": {results["global_stats"]["variance"]:.2f},')
    
    print(f"\n汇总结果已保存到: {summary_path}")


if __name__ == "__main__":
    main()
