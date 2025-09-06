# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoTune_LM is a Python project for processing Markdown documents into structured Q&A datasets and fine-tuning language models. The project follows a data pipeline approach: raw Markdown → cleaned data → Q&A generation → dataset preparation → fine-tuning.

## Architecture

### Core Modules Structure
- **src/data/**: Data processing pipeline
  - `parser.py`: Markdown parsing functionality
  - `cleaner.py`: Data cleaning and normalization
  - `qa_generator.py`: LLM-based Q&A generation
- **src/fine_tune/**: Fine-tuning components
  - `trainer.py`: Custom Hugging Face Trainer setup
  - `utils.py`: Training utilities and metrics
- **src/utils/**: Shared utilities
  - `config_loader.py`: YAML configuration loading
  - `logging.py`: Centralized logging setup

### Directory Structure
- **scripts/**: Executable pipeline scripts
  - `clean_data.py`: Data cleaning script
  - `generate_qa.py`: Q&A generation script
  - `prepare_dataset.py`: Dataset preparation and splitting
  - `fine_tune.py`: Fine-tuning execution
  - `run_pipeline.sh`: End-to-end pipeline orchestration
- **data/**: Data storage (gitignored)
  - `raw/`: Input Markdown files
  - `interim/`: Intermediate processed data
  - `processed/`: Final Q&A datasets (train.jsonl, val.jsonl)
- **configs/**: Configuration files (YAML/JSON format)
- **notebooks/**: Jupyter notebooks for exploration and prototyping
- **tests/**: Unit tests (currently empty)

## Development Setup

This is a template project - most implementation files are currently empty placeholders. The project structure is designed but not yet implemented.

### Dependencies
No requirements.txt or pyproject.toml found - dependencies need to be established.

### Testing
No test framework is currently configured. Consider adding pytest for unit testing.

### Pipeline Execution
The main pipeline is intended to be run via `scripts/run_pipeline.sh`, which should chain:
1. Data cleaning (`clean_data.py`)
2. Q&A generation (`generate_qa.py`) 
3. Dataset preparation (`prepare_dataset.py`)
4. Fine-tuning (`fine_tune.py`)

## Key Implementation Notes

- This is a template/skeleton project - most Python files are empty
- Configuration system is designed around YAML files in configs/ directory
- Data pipeline follows raw → interim → processed flow
- Fine-tuning is designed around Hugging Face Transformers
- Jupyter notebooks are intended for experimentation and evaluation