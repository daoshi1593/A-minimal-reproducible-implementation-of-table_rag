# Table RAG

这是一个用于教育和演示目的的最小可复现实现。它提供了理解Table RAG核心概念的基本功能，但不是一个生产就绪的系统。

## 功能特点

- 支持多种表格问答数据集（WikiTableQuestions, TabFact, WTQ, ARCADE, BIRD）
- 多种检索模式：BM25、向量嵌入、混合模式
- 支持多种嵌入模型：OpenAI、VertexAI、HuggingFace
- 灵活的代理类型：TableRAG、TableSampling、PyReAct等

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/table_rag.git
cd table_rag
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据集：
```bash
python convert_dataset.py
```

2. 构建向量数据库：
```bash
python build_db.py
```

3. 运行实验：
```bash
python run.py
```

或者使用提供的脚本：
```bash
# 基础参数
bash start_training.sh

# 高参数设置
bash start_training_high_params.sh
```

## 配置

主要配置参数：
- `dataset_path`: 数据集路径
- `model_name`: 使用的模型名称
- `agent_type`: 代理类型
- `retrieve_mode`: 检索模式
- `embed_model_name`: 嵌入模型名称

## 环境变量

需要设置以下环境变量：
- `OPENAI_API_KEY`: OpenAI API密钥
- `OPENAI_API_BASE`: OpenAI API基础URL（可选）

## 项目结构

```
table_rag/
├── agent/              # 代理实现
├── prompts/           # 提示模板
├── utils/             # 工具函数
├── WikiTableQuestions/ # 数据集
├── build_db.py        # 数据库构建脚本
├── run.py            # 主运行脚本
└── requirements.txt   # 依赖列表
```

## 许可证

Apache License 2.0

## 贡献

欢迎提交 Issue 和 Pull Request！

## 语言

[English](README_EN.md) | [中文](README.md)
