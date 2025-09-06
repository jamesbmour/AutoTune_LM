# AutoTune_LM

Automated fine-tuning data generation and training for LLMs from unstructured Markdown files.

## Overview

AutoTune_LM is a project that automates the process of generating fine-tuning data from unstructured Markdown files. It uses an LLM to structure the content and create question-answer pairs, then prepares the dataset and performs fine-tuning on a base LLM (e.g., GPT-2 or Llama via Hugging Face).

## Features

- **Automated Q&A Generation:** Uses an LLM (OpenAI or local via Ollama) to generate high-quality question-answer pairs from your documents.
- **Markdown Support:** Ingests unstructured Markdown files and intelligently chunks them for processing.
- **Flexible Configuration:** Uses YAML files for easy configuration of data preparation, prompts, and fine-tuning parameters.
- **Efficient Fine-Tuning:** Integrates with Unsloth for faster and more memory-efficient fine-tuning.
- **Modular and Extensible:** The project is designed to be modular and easy to extend.

## Pipeline

```
+-----------------------+
|   Raw Markdown Files  |
+-----------------------+
           |
           v
+-----------------------+
|  Markdown Parser      |
| (Semantic Chunking)   |
+-----------------------+
           |
           v
+-----------------------+
|  QA Generator (LLM)   |
| (Ollama/OpenAI)       |
+-----------------------+
           |
           v
+-----------------------+
|  Dataset Preparation  |
| (Dedupe, Filter, Split) |
+-----------------------+
           |
           v
+-----------------------+
|   Fine-Tuning         |
| (Hugging Face/Unsloth)|
+-----------------------+
           |
           v
+-----------------------+
|   Fine-Tuned LLM      |
+-----------------------+
```

## Getting Started

### Prerequisites

- Python 3.10+
- Pip and Uv

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/AutoTune_LM.git
   cd AutoTune_LM
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Create a `.env` file in the root of the project and add your API keys:
   ```
   OPENAI_API_KEY="your-openai-api-key"
   OLLAMA_HOST="http://localhost:11434"
   ```

### Usage

1. **Add your Markdown files** to the `data/raw` directory.

2. **Generate Q&A pairs:**
   ```bash
   python scripts/generate_qa.py --config configs/data_prep.yaml --input data/raw --output data/interim
   ```

3. **Prepare the dataset for fine-tuning:**
   ```bash
   python scripts/prepare_dataset.py --input data/interim --output data/processed
   ```

4. **Start fine-tuning:**
   ```bash
   python scripts/fine_tune.py --config configs/fine_tune.yaml --train_file data/processed/train.jsonl --val_file data/processed/val.jsonl
   ```

## Configuration

- `configs/data_prep.yaml`: Parameters for data preparation, such as chunk size and the LLM to use for generation.
- `configs/prompts.yaml`: Prompts for the LLM to structure content and generate Q&A pairs.
- `configs/fine_tune.yaml`: Parameters for fine-tuning, such as the base model, number of epochs, and batch size.