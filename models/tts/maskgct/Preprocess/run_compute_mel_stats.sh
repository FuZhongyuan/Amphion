#!/bin/bash

# 计算mel统计量的脚本

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

echo "=== 计算Mel-Spectrogram统计量 ==="
echo "工作目录: $work_dir"
echo ""

# 设置数据目录
DATA_DIR="$work_dir/data/processed_maskgct"

# 计算ljspeech统计量
echo "处理 LJSpeech 数据集..."
python $exp_dir/compute_mel_statistics.py \
    --processed_dir "$DATA_DIR/ljspeech" \
    --method online

echo ""
echo "----------------------------------------"
echo ""

# 计算libritts统计量  
echo "处理 LibriTTS 数据集..."
python $exp_dir/compute_mel_statistics.py \
    --processed_dir "$DATA_DIR/libritts" \
    --method online

echo ""
echo "=== 完成! ==="
echo ""
echo "统计结果保存在:"
echo "  - $DATA_DIR/ljspeech/mel_statistics.json"
echo "  - $DATA_DIR/libritts/mel_statistics.json"
