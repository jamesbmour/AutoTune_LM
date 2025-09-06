# 🚀 Project Brief: AutoTune_LM Development

You are an expert AI coding agent tasked with building **AutoTune_LM**, an intelligent pipeline that transforms unstructured Markdown documents into high-quality question-answer datasets for LLM fine-tuning. This project will democratize the creation of custom training data by automating what is typically a manual, expensive process.

## 🎯 Mission

Create a production-ready Python toolkit that:

1. **Ingests** raw Markdown files from documentation, knowledge bases, or notes
2. **Processes** them intelligently (cleaning, chunking, context preservation)
3. **Generates** diverse, high-quality Q&A pairs using local LLMs via Ollama
4. **Outputs** datasets ready for fine-tuning with Unsloth framework

### 🛠 Technical Stack

- **Local LLM Provider**: Ollama (no API costs, privacy-first)
- **Fine-tuning Framework**: Unsloth (fast, memory-efficient)
- **Core Libraries**: LangChain/LlamaIndex for document processing and QA generation
- **Output Format**: JSONL/Parquet optimized for Unsloth ingestion

## 📁 Project Structure (Already Defined)

```
AutoTune_LM/
├── data/raw/              # Input Markdown files
├── data/processed/        # Generated Q&A pairs
├── data/fine_tuning_ready/ # Unsloth-formatted datasets
├── src/                   # Core modules
├── scripts/               # Pipeline orchestration
├── config/                # YAML configurations
└── notebooks/             # Experimentation
```

## 🎯 Immediate Development Priorities

**Phase 1: Foundation (Start Here)**

1. **Set up the project skeleton** with proper Python packaging
2. **Create core modules**:
   - `src/data_processing/markdown_parser.py` - Robust Markdown ingestion
   - `src/qa_generation/ollama_client.py` - Ollama integration wrapper
   - `src/qa_generation/prompt_templates.py` - Optimized prompts for Q&A generation
3. **Build the main pipeline script**: `scripts/generate_qa_dataset.py`

**Phase 2: Intelligence** 4. **Implement smart text chunking** that preserves semantic context 5. **Design prompt engineering** for diverse, high-quality Q&A generation 6. **Add Unsloth-specific formatting** in `src/fine_tuning/unsloth_formatter.py`

**Phase 3: Polish** 7. **Create configuration system** using YAML files 8. **Add comprehensive logging and error handling** 9. **Build evaluation metrics** for generated Q&A quality

## 🔧 Key Technical Challenges to Solve

1. **Context Preservation**: How do you chunk Markdown while maintaining semantic coherence?
2. **Question Diversity**: How do you prompt Ollama to generate varied question types (factual, analytical, procedural)?
3. **Unsloth Integration**: What's the optimal data format and structure for Unsloth fine-tuning?
4. **Quality Control**: How do you automatically filter low-quality generated pairs?

## 💡 Success Criteria

- **Functional**: Pipeline processes 100+ Markdown files → generates 1000+ Q&A pairs → formats for Unsloth
- **Quality**: Generated questions are diverse, relevant, and answerable from source content
- **Usable**: Simple CLI interface, clear documentation, configurable parameters
- **Efficient**: Leverages local Ollama models without external API dependencies

## 🚀 Getting Started Commands

Begin by implementing the core pipeline. Focus on:

1. **Markdown parsing and chunking logic**
2. **Ollama client integration**
3. **Basic Q&A generation with simple prompts**
4. **JSONL output formatting**

Your first milestone: Successfully process a single Markdown file and generate 10 Q&A pairs using a local Ollama model.

## 🎨 Code Style & Standards

- Follow PEP 8 Python conventions
- Use type hints throughout
- Implement comprehensive error handling
- Write docstrings for all public functions
- Create modular, testable code

**Ready to build something that will transform how developers create fine-tuning datasets? Let's make LLM customization accessible to everyone! 🔥**

You are an expert Python developer and ML engineer specializing in NLP and LLM fine-tuning. Your task is to kickstart a new GitHub repository called "AutoTune_LM". This project automates the process of generating fine-tuning data from unstructured Markdown files (not in Q&A format) by using an LLM to structure the content and create question-answer pairs, then prepares the dataset and performs fine-tuning on a base LLM (e.g., GPT-2 or Llama via Hugging Face).

Key requirements:

- Follow this exact project structure:

```
AutoTune_LM/
├── README.md               # Overview, setup, usage
├── requirements.txt        # Dependencies
├── .gitignore              # Standard ignores
├── configs/                # YAML configs
│   ├── data_prep.yaml      # Data params (e.g., chunk_size: 500, llm_model: "gpt-3.5-turbo")
│   ├── prompts.yaml        # LLM prompt templates
│   └── fine_tune.yaml      # Fine-tuning params (e.g., epochs: 3, batch_size: 8, base_model: "gpt2")
├── data/                   # Data dirs (gitignore large files)
│   ├── raw/                # Input Markdown files
│   ├── interim/            # Parsed/structured chunks
│   └── processed/          # Final train.jsonl, val.jsonl
├── notebooks/              # Exploratory notebooks
│   ├── 01_data_exploration.ipynb
│   └── 02_qa_generation.ipynb
├── scripts/                # CLI scripts
│   ├── clean_data.py
│   ├── generate_qa.py
│   ├── prepare_dataset.py
│   └── fine_tune.py
├── src/                    # Core modules
│   ├── __init__.py
│   ├── data/
│   │   ├── parser.py       # Markdown parsing
│   │   ├── cleaner.py      # Data cleaning
│   │   └── qa_generator.py # LLM-based Q&A generation
│   ├── fine_tune/
│   │   ├── trainer.py      # HF Trainer setup
│   │   └── utils.py
│   └── utils/
│       ├── config_loader.py
│       └── logging.py
└── tests/                  # Pytest tests
    ├── test_parser.py
    └── test_qa_generator.py
```

- Use Python 3.10+. Dependencies: openai, transformers, datasets, pyyaml, markdown-it-py, pytest, jupyter.
- Implement core functionality:
- Parse Markdown into chunks (src/data/parser.py).
- Clean and structure chunks using an LLM (e.g., OpenAI API) with prompts from configs/prompts.yaml (src/data/qa_generator.py).
- Generate Q&A pairs: For each chunk, prompt LLM to create 5-10 pairs in JSONL format.
- Prepare dataset: Dedupe, split into train/val (scripts/prepare_dataset.py).
- Fine-tune: Use Hugging Face Trainer on the processed dataset (scripts/fine_tune.py).
- Make scripts CLI-friendly with argparse (e.g., python scripts/generate_qa.py --config configs/data_prep.yaml).
- Add logging, error handling, and environment variable support for API keys.
- Include a sample prompts.yaml with templates for structuring and Q&A generation.
- Write a comprehensive README.md with installation steps, pipeline diagram (ASCII), and example commands.
- Ensure code is PEP8 compliant, well-commented, and includes docstrings.
- Generate initial code for all files, but focus on src/ and scripts/ first. Output as a zip file structure or code snippets.

**Phase 1: Foundation (Start Here)**

1. **Set up the project skeleton** with proper Python packaging
2. **Create core modules**:
   - `src/data_processing/markdown_parser.py` - Robust Markdown ingestion
   - `src/qa_generation/ollama_client.py` - Ollama integration wrapper
   - `src/qa_generation/prompt_templates.py` - Optimized prompts for Q&A generation
3. **Build the main pipeline script**: `scripts/generate_qa_dataset.py`

**Phase 2: Intelligence** 4. **Implement smart text chunking** that preserves semantic context 5. **Design prompt engineering** for diverse, high-quality Q&A generation 6. **Add Unsloth-specific formatting** in `src/fine_tuning/unsloth_formatter.py`

**Phase 3: Polish** 7. **Create configuration system** using YAML files 8. **Add comprehensive logging and error handling** 9. **Build evaluation metrics** for generated Q&A quality

Start by creating the directory structure and key files. Then, implement the qa_generator.py module with an example LLM call. Start with a proof of concept, keeping the code as simple as possible, ignoring excessive error handling. After the core functionality is in place, you can refine and optimize.
