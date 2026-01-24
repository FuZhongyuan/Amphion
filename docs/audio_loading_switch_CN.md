# 音频加载开关功能说明

## 概述

为了减少训练过程中的磁盘I/O开销，我们在MaskGCT的数据集中添加了音频加载开关功能。当训练器只需要使用预处理缓存的特征（如semantic features）时，可以通过配置关闭原始音频的加载，从而显著提高数据加载速度。

## 使用场景

以下训练场景**不需要**加载原始音频，建议设置 `load_audio: false`：

1. **Semantic Codec Trainer** (`semantic_codec_trainer.py`)
   - 只需要预缓存的semantic hidden states
   - 不需要原始wav数据

2. **T2S Trainer** (`t2s_trainer.py`, `t2s_curriculum_trainer.py`)
   - 只需要预缓存的semantic features或semantic tokens
   - 不需要原始wav数据（仅在缺少缓存时才会fallback到实时提取）

## 配置方法

在配置文件的 `preprocess` 部分添加 `load_audio` 参数：

### 示例1: Semantic Codec 训练配置

```json
{
    "preprocess": {
        "dataset_type": "libritts",
        "libritts_data_root": "data/LibriTTS",
        "libritts_cache_path": "data/LibriTTSCache",
        "processed_dir": "data/processed_maskgct/libritts",
        "use_semantic_cache": true,
        "load_semantic_features": true,
        "load_audio": false  // 关闭音频加载
    }
}
```

### 示例2: T2S 训练配置

```json
{
    "preprocess": {
        "dataset_type": "libritts",
        "processed_dir": "data/processed_maskgct/libritts",
        "use_semantic_cache": true,
        "load_phone": true,
        "load_audio": false  // 关闭音频加载
    }
}
```

### 示例3: 需要音频的场景（默认行为）

```json
{
    "preprocess": {
        "dataset_type": "libritts",
        "load_audio": true  // 或者不设置此参数（默认为true）
    }
}
```

## 重要提示

1. **必须有缓存特征**：设置 `load_audio: false` 时，必须确保：
   - 已经运行过特征预处理脚本
   - `use_semantic_cache: true` 已启用
   - `processed_dir` 指向正确的缓存目录
   - 缓存文件存在且完整

2. **缓存缺失的处理**：
   - 如果 `load_audio: false` 但缓存缺失，数据集会跳过该样本并记录警告
   - 建议在训练前确保所有样本都有对应的缓存

3. **性能提升**：
   - 对于大数据集，关闭音频加载可以显著减少磁盘I/O
   - 减少内存占用（不需要保存原始音频数据）
   - 加快数据加载速度

## 支持的数据集

该功能已在以下数据集类中实现：

- `LibriTTSDataset` (基类)
- `MaskgctLibriTTSDataset`
- `LJSpeechDataset` (基类)
- `MaskgctLJSpeechDataset`

## 实现细节

### 数据集初始化
```python
# 在数据集__init__中添加
self.load_audio = getattr(self.cfg.preprocess, "load_audio", True)
```

### __getitem__方法修改
```python
# Load audio only if enabled
speech = None
if self.load_audio:
    # 加载音频的代码
    speech, _ = self._load_audio_with_torchaudio(...)
    
# 处理音频（仅在加载时）
if self.load_audio and speech is not None:
    # 处理音频的代码
    speech = np.pad(speech, ...)
    single_feature.update({"wav": speech, ...})
```

### 缓存特征的Fallback
```python
if self.load_semantic_features:
    cached_features = self.load_cached_semantic_features(wav_path)
    if cached_features is not None:
        single_feature.update(cached_features)
    else:
        if not self.load_audio:
            # 缓存缺失且未加载音频，跳过样本
            logger.warning("Cache miss but load_audio=False. Skipping.")
            return self.__getitem__(random_index)
        # 否则实时提取特征
```

## 参考配置文件

完整配置示例请参考：
- `egs/tts/MaskGCT/semantic_codec_mini.json` - Semantic Codec训练配置（已设置 `load_audio: false`）

## 故障排查

### 问题1: 训练时提示 "Cache miss but load_audio=False"
**原因**：缓存文件不存在或路径配置错误

**解决方法**：
1. 检查 `processed_dir` 路径是否正确
2. 确认 `hdf5_index.json` 文件存在
3. 运行特征预处理脚本生成缓存

### 问题2: 训练速度没有提升
**原因**：可能有其他I/O瓶颈

**建议**：
1. 检查缓存文件是否在SSD上
2. 增加 `num_worker` 数量
3. 启用 `pin_memory` 和 `prefetch_factor`
