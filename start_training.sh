#!/bin/bash

# 设置环境变量
DATASET_PATH="WikiTableQuestions/processed/processed_dataset.jsonl"
MODEL_NAME="gpt-3.5-turbo-1106"
AGENT_TYPE="TableRAG"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="output/wikitables_gpt3.5_tablerag_$TIMESTAMP"

# 检查数据集文件是否存在
if [ ! -f "$DATASET_PATH" ]; then
    echo "错误：数据集文件不存在：$DATASET_PATH"
    exit 1
fi

# 检查数据集文件是否为空
if [ ! -s "$DATASET_PATH" ]; then
    echo "错误：数据集文件为空：$DATASET_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建数据库
echo "开始构建数据库..."
python build_db.py \
    --dataset_path "$DATASET_PATH" \
    --max_encode_cell 10000 \
    --output_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "错误：数据库构建失败"
    exit 1
fi

# 运行实验
echo "开始运行实验..."
python run.py \
    --dataset_path "$DATASET_PATH" \
    --model_name "$MODEL_NAME" \
    --agent_type "$AGENT_TYPE" \
    --log_dir "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "错误：实验运行失败"
    exit 1
fi

echo "实验结束。"
echo "结果保存在: $OUTPUT_DIR" 