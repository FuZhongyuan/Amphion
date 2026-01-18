# MaskGCT 推理脚本使用说明

## 修改内容

1. **支持本地checkpoint加载**：不再从Hugging Face下载，而是从本地目录加载预训练模型
2. **添加DEBUG模式**：在DEBUG模式下会打印详细的模型结构信息并保存到文件中
3. **命令行参数支持**：支持通过命令行参数配置推理选项
4. **Shell脚本封装**：提供易用的shell脚本封装

## 使用方法

### 方法1：直接使用Python脚本

```bash
python models/tts/maskgct/maskgct_inference.py \
  --cfg_path="./models/tts/maskgct/config/maskgct.json" \
  --ckpt_dir="./ckpts/MaskGCT-ckpt" \
  --prompt_wav_path="./models/tts/maskgct/wav/prompt.wav" \
  --prompt_text="We do not break. We never give in. We never back down." \
  --target_text="Hello world, this is a test of MaskGCT inference." \
  --save_path="output.wav" \
  --target_len=10.0 \
  --device="cuda:0" \
  --log_level="INFO" \
  --debug
```

### 方法2：使用Shell脚本（推荐）

```bash
# 使用默认参数（目录结构）
./models/tts/maskgct/run_maskgct_inference.sh

# 自定义参数
./models/tts/maskgct/run_maskgct_inference.sh \
  --prompt_text="自定义提示文本" \
  --target_text="自定义目标文本" \
  --save_path="my_output.wav" \
  --target_len=15.0 \
  --debug

# 单独指定每个checkpoint路径
./models/tts/maskgct/run_maskgct_inference.sh \
  --semantic_codec_ckpt="/path/to/semantic_codec/model.safetensors" \
  --codec_encoder_ckpt="/path/to/codec_encoder/model.safetensors" \
  --codec_decoder_ckpt="/path/to/codec_decoder/model_1.safetensors" \
  --t2s_model_ckpt="/path/to/t2s_model/model.safetensors" \
  --s2a_1layer_ckpt="/path/to/s2a_1layer/model.safetensors" \
  --s2a_full_ckpt="/path/to/s2a_full/model.safetensors" \
  --target_text="使用单独checkpoint路径的示例"
```

## 参数说明

### 必需参数

- `--ckpt_dir`: 本地checkpoint目录路径，默认为 `./ckpts/MaskGCT-ckpt`

### 可选参数

- `--cfg_path`: 配置文件路径，默认为 `./models/tts/maskgct/config/maskgct.json`

**Checkpoint 路径参数（可单独指定或使用目录结构）**
- `--semantic_codec_ckpt`: 语义编码器checkpoint路径
- `--codec_encoder_ckpt`: 声学编码器checkpoint路径
- `--codec_decoder_ckpt`: 声学解码器checkpoint路径
- `--t2s_model_ckpt`: 文本到语义模型checkpoint路径
- `--s2a_1layer_ckpt`: 语义到声学1层模型checkpoint路径
- `--s2a_full_ckpt`: 语义到声学完整模型checkpoint路径

**推理参数**
- `--prompt_wav_path`: 提示音频文件路径，默认为 `./models/tts/maskgct/wav/prompt.wav`
- `--prompt_text`: 提示文本
- `--target_text`: 要生成的文本
- `--save_path`: 输出音频文件路径，默认为 `generated_audio.wav`
- `--target_len`: 目标音频时长（秒），设置为0或负数则自动预测
- `--device`: 推理设备，默认为 `cuda:0`
- `--log_level`: 日志级别，可选 `DEBUG`, `INFO`, `WARNING`, `ERROR`，默认为 `INFO`
- `--debug`: 启用DEBUG模式，会打印模型结构信息并保存到 `model_info.txt`

## Checkpoint目录结构

确保checkpoint目录按以下结构组织：

```
ckpts/MaskGCT-ckpt/
├── semantic_codec/
│   └── model.safetensors
├── acoustic_codec/
│   ├── model.safetensors
│   └── model_1.safetensors
├── t2s_model/
│   └── model.safetensors
└── s2a_model/
    ├── s2a_model_1layer/
    │   └── model.safetensors
    └── s2a_model_full/
        └── model.safetensors
```

## DEBUG模式功能

启用 `--debug` 参数后，会：

1. 在控制台打印所有模型的详细结构信息
2. 将模型结构信息保存到 `model_info.txt` 文件中
3. 记录更详细的推理日志到 `maskgct_inference.log`

## 输出文件

- 生成的音频文件：指定路径（默认为 `generated_audio.wav`）
- 推理日志：`maskgct_inference.log`
- 模型信息（DEBUG模式）：`model_info.txt`

## 注意事项

1. 确保CUDA可用（如果使用GPU）
2. 确保所有checkpoint文件存在
3. 确保提示音频文件存在且格式正确
4. 推理时间可能较长，请耐心等待
