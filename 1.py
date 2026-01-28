import h5py
import os

def view_h5_shapes(file_path):
    """
    查看HDF5文件中所有数据集的形状
    
    参数:
        file_path: HDF5文件路径
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件: {file_path}")
            print("=" * 60)
            
            def print_dataset_info(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"数据集: {name}")
                    print(f"  形状: {obj.shape}")
                    print(f"  数据类型: {obj.dtype}")
                    print(f"  维度: {len(obj.shape)}")
                    print("-" * 60)
            
            # 递归遍历所有数据集
            f.visititems(print_dataset_info)
            
    except Exception as e:
        print(f"读取文件时出错: {e}")

# 使用示例
if __name__ == "__main__":
    # 替换为你的HDF5文件路径
    h5_file = "data/processed_maskgct/libritts/features_00002.h5"
    view_h5_shapes(h5_file)