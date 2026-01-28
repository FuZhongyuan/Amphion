export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"
export CUDA_VISIBLE_DEVICES="5"

######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

# 测试随机性（不设置随机种子）
python models/tts/maskgct/Preprocess/check_quantization_consistency.py \
    --config models/tts/maskgct/Preprocess/preprocess_semantic_code_config.json \
    --num_runs 5 \
    --num_samples 100

# 测试确定性（设置随机种子）
python models/tts/maskgct/Preprocess/check_quantization_consistency.py \
    --config models/tts/maskgct/Preprocess/preprocess_semantic_code_config.json \
    --num_runs 5 \
    --num_samples 100 \
    --random_seed 42

# # 测试更多样本
# python models/tts/maskgct/Preprocess/check_quantization_consistency.py \
#     --config models/tts/maskgct/Preprocess/preprocess_semantic_code_config.json \
#     --num_runs 10 \
#     --num_samples 500 \
#     --output_dir ./consistency_results