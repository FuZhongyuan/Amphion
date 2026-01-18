# HDF5 缓存格式迁移说明

## 概述

特征缓存格式已从 NPZ 迁移到 HDF5，支持更高效的存储和切片读取。

## 主要变更

### 1. 配置文件 (`preprocess_config.json`)

新增以下配置选项：

```json
{
  "preprocessing": {
    "use_float16": true,          // 是否使用 float16 存储浮点数据
    "samples_per_hdf5": 20000,    // 每个 HDF5 文件包含的样本数
    "num_save_workers": 4         // 异步保存的工作线程数
  }
}
```

- `use_float16`: 控制是否将浮点特征（如 hidden_states 和 mel_spectrogram）存储为 float16，减少存储空间
- `samples_per_hdf5`: 每 20,000 个样本创建一个新的 HDF5 文件，避免单文件过大
- `num_save_workers`: 异步保存的并发线程数

### 2. 预处理脚本 (`preprocess_semantic_base.py`)

#### 核心改动：

1. **HDF5Writer 类**：管理多文件 HDF5 写入
   - 自动分割文件：每 20k 样本创建新文件 `features_00001.h5`, `features_00002.h5`, ...
   - 索引管理：维护 `hdf5_index.json` 记录样本到文件的映射
   - 压缩存储：使用 gzip 压缩（level 4）
   - Float16 支持：根据配置自动转换浮点数据类型

2. **存储结构**：
   ```
   processed_dir/
   ├── features_00001.h5      # 样本 1-20000
   ├── features_00002.h5      # 样本 20001-40000
   ├── ...
   └── hdf5_index.json        # 索引文件
   ```

3. **HDF5 文件内部结构**：
   ```
   features_00001.h5
   ├── sample_key_1/
   │   ├── hidden_states      # shape: (T, 1024), dtype: float16/float32
   │   └── mel_spectrogram    # shape: (1, T, 80), dtype: float16/float32
   ├── sample_key_2/
   │   └── ...
   ```

4. **索引文件格式** (`hdf5_index.json`)：
   ```json
   {
     "sample_index": {
       "LJ001-0001": {
         "file_idx": 1,
         "group_name": "LJ001-0001"
       },
       "train-clean-100/123/456/123_456_789": {
         "file_idx": 2,
         "group_name": "train-clean-100/123/456/123_456_789"
       }
     },
     "current_file_idx": 3,
     "current_file_count": 5234,
     "samples_per_file": 20000,
     "use_float16": true
   }
   ```

### 3. Dataset 文件适配

#### `maskgct_ljspeech_dataset.py` 和 `maskgct_libritts_dataset.py`

新增功能：

1. **HDF5 索引加载**：启动时加载 `hdf5_index.json`
2. **HDF5 文件缓存**：缓存已打开的 HDF5 文件句柄，避免重复打开
3. **切片读取**：使用 `group["hidden_states"][:]` 按需读取数据
4. **自动清理**：`__del__` 方法自动关闭 HDF5 文件

核心方法改动：

```python
def load_cached_semantic_features(self, wav_path):
    # 1. 将 wav_path 转换为 sample_key
    # 2. 从 hdf5_index 查找文件位置
    # 3. 打开对应的 HDF5 文件
    # 4. 切片读取所需数据
    # 5. 返回 numpy 数组
```

## 数据格式对比

### NPZ 格式（旧）：
```
processed_dir/
├── LJ001-0001.npz
├── LJ001-0002.npz
├── ...
└── train-clean-100/123/456/123_456_789.npz
```

每个文件独立存储一个样本，IO 开销大。

### HDF5 格式（新）：
```
processed_dir/
├── features_00001.h5      # 包含 20,000 个样本
├── features_00002.h5
├── ...
└── hdf5_index.json        # 快速索引
```

多个样本打包存储，支持压缩和切片读取，IO 效率高。

## 优势

1. **存储效率**：
   - Float16 存储减少约 50% 空间
   - gzip 压缩进一步减少 30-50%
   - 减少文件系统 inode 占用

2. **读取性能**：
   - 减少文件打开/关闭次数
   - HDF5 文件句柄缓存
   - 支持高效的切片读取

3. **可扩展性**：
   - 单个 HDF5 文件最多 20k 样本，避免过大
   - 索引文件支持快速查找
   - 易于并行读取

## 使用方法

### 预处理
```bash
cd models/tts/maskgct/Preprocess
python preprocess_libritts_semantic.py
```

配置会自动从 `preprocess_config.json` 读取 `use_float16` 和 `samples_per_hdf5` 参数。

### 训练

在训练配置中设置：
```yaml
preprocess:
  use_semantic_cache: true
  processed_dir: "data/processed_maskgct/ljspeech"  # 或 libritts
```

Dataset 会自动检测并使用 HDF5 缓存。

## 注意事项

1. **不兼容旧格式**：HDF5 格式与 NPZ 格式不兼容，需要重新预处理
2. **Float16 精度**：如果需要完整精度，设置 `use_float16: false`
3. **文件句柄**：Dataset 会缓存 HDF5 文件句柄，确保在多进程环境下正确处理
4. **索引文件**：`hdf5_index.json` 是必需的，删除后需要重新预处理

## 性能建议

- 推荐使用 `use_float16: true` 以节省存储空间（对模型精度影响极小）
- `samples_per_hdf5: 20000` 是经验值，可根据实际情况调整
- 使用 SSD 存储 HDF5 文件以获得最佳读取性能
