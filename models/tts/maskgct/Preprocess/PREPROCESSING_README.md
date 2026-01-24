# 语义特征预处理统一Pipeline

本目录包含用于预处理MaskGCT模型训练数据的统一pipeline。该pipeline支持从音频文件中提取语义特征(semantic features)和可选的mel频谱图(mel-spectrogram)。

## 架构设计

### 核心模块

- **`preprocess_semantic_base.py`**: 统一的预处理核心模块
  - `AudioLoader`: 音频加载和重采样
  - `SemanticFeatureExtractor`: 使用w2v-bert-2.0提取语义特征
  - `MelSpectrogramExtractor`: 提取mel频谱图
  - `DatasetPreprocessor`: 抽象基类,定义预处理接口
  - `LJSpeechPreprocessor`: LJSpeech数据集的具体实现
  - `LibriTTSPreprocessor`: LibriTTS数据集的具体实现
  - `create_preprocessor()`: 工厂函数,根据数据集类型创建对应的预处理器

### 数据集脚本

- **`preprocess_ljspeech_semantic.py`**: LJSpeech数据集预处理入口
- **`preprocess_libritts_semantic.py`**: LibriTTS数据集预处理入口

这两个脚本现在是轻量级的包装器,内部调用统一的预处理pipeline。

## 功能特性

### 1. 语义特征提取
- 使用facebook/w2v-bert-2.0模型
- 可配置输出层(默认第17层)
- GPU加速支持
- 自动批处理和内存管理

### 2. Mel频谱图提取(可选)
- 可配置的mel参数(采样率、hop size、窗口大小等)
- 与语义特征一起缓存,避免训练时重复计算
- 可通过配置文件开关控制

### 3. 数据集支持
- **LJSpeech**: 单说话人英语数据集
  - 目录结构: `{data_root}/wavs/*.wav`
  - 输出结构: `{processed_dir}/*.npz`

- **LibriTTS**: 多说话人英语数据集
  - 目录结构: `{data_root}/{subset}/{speaker_id}/{chapter_id}/*.wav`
  - 支持的子集: train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other
  - 输出结构: 保持原始目录层次结构

### 4. 其他特性
- 断点续传: 已处理的文件可跳过(通过`overwrite_existing`控制)
- 进度显示: 使用tqdm显示处理进度
- 错误处理: 单个文件失败不影响整体处理
- 元数据保存: 自动生成metadata.json记录处理信息

## 使用方法

### 1. 准备配置文件

创建一个JSON配置文件(参考`preprocess_config_example.json`):

```json
{
  "semantic_model": {
    "model_name": "facebook/w2v-bert-2.0",
    "output_layer": 17
  },
  "preprocessing": {
    "overwrite_existing": false,
    "extract_mel": true,
    "mel_config": {
      "preprocess": "AudioProcessor",
      "sample_rate": 24000,
      "hop_size": 256,
      "win_size": 1024,
      "n_fft": 1024,
      "n_mel": 100,
      "f_min": 0,
      "f_max": 12000
    }
  },
  "datasets": {
    "ljspeech": {
      "data_root": "/path/to/LJSpeech-1.1",
      "processed_dir": "/path/to/processed/ljspeech"
    },
    "libritts": {
      "data_root": "/path/to/LibriTTS",
      "processed_dir": "/path/to/processed/libritts"
    }
  }
}
```

### 2. 运行预处理

#### 处理LJSpeech数据集:
```bash
python preprocess_ljspeech_semantic.py --config preprocess_config.json --dataset ljspeech
```

#### 处理LibriTTS数据集:
```bash
python preprocess_libritts_semantic.py --config preprocess_config.json --dataset libritts
```

### 3. 输出格式

每个音频文件会生成一个对应的`.npz`文件,包含:
- `hidden_states`: 语义特征 (shape: [T, D], T为时间步数, D为特征维度)
- `mel_spectrogram` (可选): Mel频谱图 (shape: [1, T_mel, n_mel])

元数据文件`metadata.json`包含:
```json
{
  "dataset": "ljspeech",
  "total_files": 13100,
  "processed_files": 13100,
  "error_files": 0,
  "processed_dir": "/path/to/processed/ljspeech",
  "semantic_model": "facebook/w2v-bert-2.0",
  "extract_mel": true,
  "mel_config": {...}
}
```

## 配置参数说明

### semantic_model
- `model_name`: HuggingFace模型名称
- `output_layer`: 提取哪一层的hidden states (默认17)

### preprocessing
- `overwrite_existing`: 是否覆盖已存在的文件 (默认false)
- `extract_mel`: 是否提取mel频谱图 (默认false)
- `mel_config`: mel频谱图配置参数
  - `sample_rate`: 采样率
  - `hop_size`: 帧移
  - `win_size`: 窗口大小
  - `n_fft`: FFT点数
  - `n_mel`: mel滤波器组数量
  - `f_min`: 最小频率
  - `f_max`: 最大频率

### datasets
为每个数据集配置:
- `data_root`: 原始数据根目录
- `processed_dir`: 处理后数据保存目录

## 扩展新数据集

要支持新的数据集,只需在`preprocess_semantic_base.py`中添加新的预处理器类:

```python
class MyDatasetPreprocessor(DatasetPreprocessor):
    """Preprocessor for MyDataset."""
    
    def find_audio_files(self) -> List[Path]:
        """Find all audio files in MyDataset."""
        # 实现数据集特定的文件查找逻辑
        wav_files = []
        # ... 查找逻辑 ...
        return sorted(wav_files)
    
    def get_output_path(self, wav_file: Path) -> Path:
        """Get output path for MyDataset processed features."""
        # 实现输出路径映射逻辑
        relative_path = wav_file.relative_to(self.data_root)
        output_path = self.processed_dir / relative_path.with_suffix('.npz')
        return output_path
```

然后在`create_preprocessor()`函数中注册:

```python
def create_preprocessor(config: Dict[str, Any], dataset_name: str) -> DatasetPreprocessor:
    preprocessors = {
        "ljspeech": LJSpeechPreprocessor,
        "libritts": LibriTTSPreprocessor,
        "mydataset": MyDatasetPreprocessor,  # 添加新数据集
    }
    # ...
```

## 性能优化建议

1. **GPU加速**: 确保CUDA可用,语义特征提取会自动使用GPU
2. **批处理**: 当前实现是顺序处理,如需更高性能可考虑批处理
3. **存储**: 使用`np.savez_compressed`压缩存储,节省磁盘空间
4. **断点续传**: 设置`overwrite_existing=false`避免重复处理

## 常见问题

### Q: 如何只提取语义特征不提取mel频谱图?
A: 在配置文件中设置`"extract_mel": false`

### Q: 处理中断后如何继续?
A: 重新运行相同命令即可,已处理的文件会自动跳过(前提是`overwrite_existing=false`)

### Q: 如何验证处理结果?
A: 检查`metadata.json`中的统计信息,确认`processed_files`数量正确

### Q: 内存不足怎么办?
A: 当前实现是单文件处理,内存占用较小。如仍有问题,可减小batch size或使用更小的模型

## 依赖项

- torch
- numpy
- librosa
- transformers (SeamlessM4TFeatureExtractor, Wav2Vec2BertModel)
- tqdm
- pathlib (Python标准库)

## 版本历史

- **v2.0** (当前版本): 统一预处理pipeline,支持多数据集
  - 重构为面向对象设计
  - 统一接口,易于扩展
  - 完整的mel频谱图支持
  
- **v1.0**: 独立的数据集预处理脚本
  - 分离的LJSpeech和LibriTTS脚本
  - 基本的语义特征提取


## 1. 新增预处理脚本（New Preprocessing Script）

**新增文件：**
`models/tts/maskgct/Preprocess/preprocess_semantic_code.py`

**功能说明：**

* 从 HDF5 文件中加载预先提取的 `hidden_states`
* 使用 **RepCodec** 模型将其量化为离散的 `semantic_code`
* 将量化后的语义码保存为新的 HDF5 文件

---

## 2. 配置文件（Configuration Files）

**新增配置：**

* `models/tts/maskgct/Preprocess/preprocess_semantic_code_config.json`
  → 语义码预处理配置文件
* `egs/tts/MaskGCT/t2s_mini_with_semantic_code.json`
  → 示例 T2S 训练配置
* `egs/tts/MaskGCT/s2mel_dit_mini_with_semantic_code.json`
  → 示例 S2MelDiT 训练配置

---

## 3. 数据集加载器更新（Dataset Loaders Updated）

**修改文件：**

* `models/tts/base/maskgct_ljspeech_dataset.py`
* `models/tts/base/maskgct_libritts_dataset.py`

**更新内容：**

* 新增配置选项：

  * `load_semantic_code`
  * `semantic_code_dir`
* 新增方法：

  * `_load_semantic_code_index()`：加载 HDF5 索引
  * `_get_semantic_code_file()`：获取 HDF5 文件句柄
  * `load_cached_semantic_code()`：从 HDF5 加载语义码
* 更新：

  * `__getitem__`：在启用时加载语义码
  * collator：支持 `semantic_code` 的 padding

---

## 4. 新增配置项（New Configuration Options）

**训练配置文件中 `preprocess` 字段示例：**

```json
{
  "preprocess": {
    "load_semantic_code": true,
    "semantic_code_dir": "data/processed_maskgct/libritts_semantic_code"
  }
}
```

---

## 5. 使用流程（Usage Workflow）

### 步骤 1：预处理语义隐藏状态（如尚未完成）

```bash
./models/tts/maskgct/Preprocess/run_preprocessing.sh
```

### 步骤 2：将隐藏状态量化为语义码

```bash
./models/tts/maskgct/Preprocess/run_semantic_code_preprocessing.sh
```

### 步骤 3：使用预计算语义码进行训练

```bash
# 使用 load_semantic_code=true 的新配置
python bins/tts/maskgct_train.py \
  --config egs/tts/MaskGCT/t2s_mini_with_semantic_code.json
```

