#!/usr/bin/env python3
"""
计算HDF5文件中mel-spectrogram的均值和方差统计
基于 preprocess_semantic_base.py 生成的HDF5文件格式
"""

import os
import json
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import argparse


def load_hdf5_index(processed_dir: Path):
    """加载HDF5索引文件"""
    index_path = processed_dir / "hdf5_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"HDF5索引文件不存在: {index_path}")
    
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    
    return index_data


def get_hdf5_path(processed_dir: Path, file_idx: int) -> Path:
    """获取HDF5文件路径"""
    return processed_dir / f"features_{file_idx:05d}.h5"


def compute_mel_statistics_online(processed_dir: Path, use_float16: bool = True):
    """
    使用在线算法（Welford's algorithm）计算mel-spectrogram的均值和方差
    避免一次性加载所有数据到内存
    """
    print(f"正在处理目录: {processed_dir}")
    
    # 加载HDF5索引
    index_data = load_hdf5_index(processed_dir)
    sample_index = index_data.get("sample_index", {})
    
    if len(sample_index) == 0:
        raise ValueError("HDF5索引为空，没有找到任何样本")
    
    print(f"找到 {len(sample_index)} 个样本")
    
    # Welford在线算法的变量
    count = 0  # 总的时间帧数
    mean = None  # 当前均值
    M2 = None  # 方差的累积量
    sample_count = 0  # 样本数量
    
    # 遍历所有样本
    for sample_key, sample_info in tqdm(sample_index.items(), desc="计算统计量"):
        file_idx = sample_info["file_idx"]
        group_name = sample_info["group_name"]
        
        # 打开HDF5文件
        hdf5_path = get_hdf5_path(processed_dir, file_idx)
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if group_name not in f:
                    print(f"\n警告: 样本 {group_name} 在文件 {hdf5_path} 中不存在")
                    continue
                
                sample_group = f[group_name]
                
                # 检查是否有mel_spectrogram数据
                if "mel_spectrogram" not in sample_group:
                    continue
                
                # 读取mel_spectrogram数据
                mel_data = sample_group["mel_spectrogram"][:]
                
                # 转换为float32进行计算（如果是float16）
                if mel_data.dtype == np.float16:
                    mel_data = mel_data.astype(np.float32)
                
                # mel_data shape: (1, T, mel_dim) 或 (T, mel_dim)
                if len(mel_data.shape) == 3:
                    mel_data = mel_data[0]  # 去掉batch维度
                
                # mel_data shape: (T, mel_dim)
                n_frames = mel_data.shape[0]
                
                # 初始化
                if mean is None:
                    mel_dim = mel_data.shape[1]
                    mean = np.zeros(mel_dim, dtype=np.float64)
                    M2 = np.zeros(mel_dim, dtype=np.float64)
                
                # Welford在线算法更新
                for frame in mel_data:
                    count += 1
                    delta = frame - mean
                    mean += delta / count
                    delta2 = frame - mean
                    M2 += delta * delta2
                
                sample_count += 1
                
        except Exception as e:
            print(f"\n错误: 处理文件 {hdf5_path} 样本 {group_name} 时出错: {e}")
            continue
    
    if count == 0:
        raise ValueError("没有找到任何mel_spectrogram数据")
    
    # 计算最终的方差
    variance = M2 / count
    
    # 计算标准差
    std = np.sqrt(variance)
    
    # 计算全局统计量（所有维度的平均）
    global_mean = np.mean(mean)
    global_var = np.mean(variance)
    global_std = np.mean(std)
    
    print(f"\n处理完成!")
    print(f"总样本数: {sample_count}")
    print(f"总时间帧数: {count}")
    print(f"Mel维度: {len(mean)}")
    print(f"\n全局统计量:")
    print(f"  均值 (mean): {global_mean:.4f}")
    print(f"  方差 (variance): {global_var:.4f}")
    print(f"  标准差 (std): {global_std:.4f}")
    
    # 返回统计结果
    results = {
        "sample_count": sample_count,
        "total_frames": int(count),
        "mel_dim": len(mean),
        "per_dimension_stats": {
            "mean": mean.tolist(),
            "variance": variance.tolist(),
            "std": std.tolist()
        },
        "global_stats": {
            "mean": float(global_mean),
            "variance": float(global_var),
            "std": float(global_std)
        }
    }
    
    return results


def compute_mel_statistics_batch(processed_dir: Path, batch_size: int = 100):
    """
    批量计算mel-spectrogram的统计量（备用方法）
    """
    print(f"正在处理目录: {processed_dir}")
    
    # 加载HDF5索引
    index_data = load_hdf5_index(processed_dir)
    sample_index = index_data.get("sample_index", {})
    
    if len(sample_index) == 0:
        raise ValueError("HDF5索引为空，没有找到任何样本")
    
    print(f"找到 {len(sample_index)} 个样本")
    
    # 收集所有mel数据
    all_mel_data = []
    sample_count = 0
    
    for sample_key, sample_info in tqdm(sample_index.items(), desc="加载数据"):
        file_idx = sample_info["file_idx"]
        group_name = sample_info["group_name"]
        
        hdf5_path = get_hdf5_path(processed_dir, file_idx)
        
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if group_name not in f:
                    continue
                
                sample_group = f[group_name]
                
                if "mel_spectrogram" not in sample_group:
                    continue
                
                mel_data = sample_group["mel_spectrogram"][:]
                
                if mel_data.dtype == np.float16:
                    mel_data = mel_data.astype(np.float32)
                
                if len(mel_data.shape) == 3:
                    mel_data = mel_data[0]
                
                all_mel_data.append(mel_data)
                sample_count += 1
                
        except Exception as e:
            print(f"\n错误: {e}")
            continue
    
    if len(all_mel_data) == 0:
        raise ValueError("没有找到任何mel_spectrogram数据")
    
    # 拼接所有数据
    print("拼接所有mel数据...")
    all_mel_data = np.concatenate(all_mel_data, axis=0)  # (total_frames, mel_dim)
    
    # 计算统计量
    print("计算统计量...")
    mean = np.mean(all_mel_data, axis=0)  # (mel_dim,)
    variance = np.var(all_mel_data, axis=0)  # (mel_dim,)
    std = np.std(all_mel_data, axis=0)  # (mel_dim,)
    
    global_mean = np.mean(mean)
    global_var = np.mean(variance)
    global_std = np.mean(std)
    
    print(f"\n处理完成!")
    print(f"总样本数: {sample_count}")
    print(f"总时间帧数: {all_mel_data.shape[0]}")
    print(f"Mel维度: {all_mel_data.shape[1]}")
    print(f"\n全局统计量:")
    print(f"  均值 (mean): {global_mean:.4f}")
    print(f"  方差 (variance): {global_var:.4f}")
    print(f"  标准差 (std): {global_std:.4f}")
    
    results = {
        "sample_count": sample_count,
        "total_frames": int(all_mel_data.shape[0]),
        "mel_dim": int(all_mel_data.shape[1]),
        "per_dimension_stats": {
            "mean": mean.tolist(),
            "variance": variance.tolist(),
            "std": std.tolist()
        },
        "global_stats": {
            "mean": float(global_mean),
            "variance": float(global_var),
            "std": float(global_std)
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="计算HDF5文件中mel-spectrogram的统计量")
    parser.add_argument(
        "--processed_dir",
        type=str,
        required=True,
        help="处理后的数据目录路径（包含HDF5文件和索引的目录）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON文件路径（默认为processed_dir/mel_statistics.json）"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["online", "batch"],
        default="online",
        help="计算方法: online（在线算法，省内存）或 batch（批量计算，速度快但需要更多内存）"
    )
    
    args = parser.parse_args()
    
    processed_dir = Path(args.processed_dir)
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"目录不存在: {processed_dir}")
    
    # 计算统计量
    if args.method == "online":
        results = compute_mel_statistics_online(processed_dir)
    else:
        results = compute_mel_statistics_batch(processed_dir)
    
    # 保存结果
    if args.output is None:
        output_path = processed_dir / "mel_statistics.json"
    else:
        output_path = Path(args.output)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n统计结果已保存到: {output_path}")
    print(f"\n可以在配置文件中使用以下值:")
    print(f'  "mel_mean": {results["global_stats"]["mean"]:.2f},')
    print(f'  "mel_var": {results["global_stats"]["variance"]:.2f},')


if __name__ == "__main__":
    main()
