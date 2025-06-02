# Table RAG

A minimal reproducible implementation created for educational and demonstration purposes. It provides basic functionality to understand the core concepts of Table RAG but is not a production-ready system.

## Features

- Support for multiple table QA datasets (WikiTableQuestions, TabFact, WTQ, ARCADE, BIRD)
- Multiple retrieval modes: BM25, vector embedding, hybrid mode
- Support for various embedding models: OpenAI, VertexAI, HuggingFace
- Flexible agent types: TableRAG, TableSampling, PyReAct, etc.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/daoshi1593/A-minimal-reproducible-implementation-of-table_rag.git
cd table_rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare the dataset:
```bash
python convert_dataset.py
```

2. Build vector database:
```bash
python build_db.py
```

3. Run experiments:
```bash
python run.py
```

Or use the provided scripts:
```bash
# Basic parameters
bash start_training.sh

# High parameter settings
bash start_training_high_params.sh
```

## Configuration

Main configuration parameters:
- `dataset_path`: Dataset path
- `model_name`: Model name to use
- `agent_type`: Agent type
- `retrieve_mode`: Retrieval mode
- `embed_model_name`: Embedding model name

## Environment Variables

Set the following environment variables:
- `OPENAI_API_KEY`: OpenAI API key
- `OPENAI_API_BASE`: OpenAI API base URL (optional)

## Project Structure

```
table_rag/
├── agent/              # Agent implementations
├── prompts/           # Prompt templates
├── utils/             # Utility functions
├── WikiTableQuestions/ # Dataset
├── build_db.py        # Database building script
├── run.py            # Main running script
└── requirements.txt   # Dependencies list
```

## License

Apache License 2.0

## Contributing

Issues and Pull Requests are welcome!

## Language

[English](README_EN.md) | [中文](README.md) 