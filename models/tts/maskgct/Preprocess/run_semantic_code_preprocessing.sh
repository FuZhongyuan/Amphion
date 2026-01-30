export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="data/cache/huggingface"


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $(dirname $exp_dir))))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

######## Set Experiment Configuration ###########
exp_config="$exp_dir/preprocess_semantic_code_config.json"

####### Preprocess Semantic Code ###########
CUDA_VISIBLE_DEVICES="0" python \
    "${work_dir}"/models/tts/maskgct/Preprocess/preprocess_semantic_code.py \
    --config=$exp_config
