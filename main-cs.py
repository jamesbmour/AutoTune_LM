#!/usr/bin/env python3
"""
Markdown to Q&A Fine-tuning Dataset Converter

This script converts unstructured Markdown documents into Q&A pairs for fine-tuning
language models using Pydantic AI for question generation, Ollama for inference,
and Unsloth for fine-tuning preparation.

Requirements:
    pip install pydantic-ai ollama unsloth datasets transformers trl pyyaml
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import yaml

import ollama
from pydantic import BaseModel, Field, validator
from pydantic_ai import Agent, RunContext
from datasets import Dataset
from transformers import AutoTokenizer
import markdown
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
@dataclass
class Config:
    """Configuration for the markdown to Q&A converter"""
    # Input/Output settings
    input_directory: str = "./markdown_docs"
    output_directory: str = "./qa_dataset"
    config_file: Optional[str] = None
    
    # Model settings
    ollama_model: str = "llama3.2:3b"
    backup_ollama_model: str = "llama3.1:8b"
    
    # Processing settings
    max_context_length: int = 2048
    chunk_overlap: int = 200
    num_questions_per_chunk: int = 3
    min_chunk_length: int = 100
    
    # Output format settings
    output_format: str = "unsloth"  # "unsloth", "alpaca", or "sharegpt"
    
    # File processing settings
    file_extensions: List[str] = None
    exclude_patterns: List[str] = None
    max_files_to_process: Optional[int] = None
    
    # Quality control settings
    min_question_length: int = 10
    min_answer_length: int = 20
    max_retries_per_chunk: int = 3
    
    # Fine-tuning settings
    default_base_model: str = "unsloth/llama-3.2-3b-instruct-bnb-4bit"
    training_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    
    def __post_init__(self):
        if self.file_extensions is None:
            self.file_extensions = ['.md', '.markdown', '.mdown', '.mkd']
        if self.exclude_patterns is None:
            self.exclude_patterns = ['README.md', 'CHANGELOG.md', 'LICENSE.md']
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Create config with defaults, then update with file data
            config = cls()
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    logger.warning(f"Unknown configuration key: {key}")
            
            config.config_file = config_path
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def to_yaml(self, output_path: str):
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        # Remove the config_file field to avoid circular reference
        config_dict.pop('config_file', None)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def validate(self):
        """Validate configuration settings"""
        errors = []
        
        if not Path(self.input_directory).exists():
            errors.append(f"Input directory does not exist: {self.input_directory}")
        
        if self.max_context_length < 100:
            errors.append("max_context_length must be at least 100")
        
        if self.num_questions_per_chunk < 1:
            errors.append("num_questions_per_chunk must be at least 1")
        
        if self.output_format not in ['unsloth', 'alpaca', 'sharegpt']:
            errors.append(f"Invalid output_format: {self.output_format}")
        
        if errors:
            raise ValueError("Configuration validation errors:\n" + "\n".join(f"- {error}" for error in errors))

# Pydantic models for structured output (unchanged from previous version)
class QuestionAnswer(BaseModel):
    """Single Q&A pair generated from content"""
    question: str = Field(description="A clear, specific question about the content")
    answer: str = Field(description="A comprehensive answer based on the content")
    context: str = Field(description="Relevant context snippet from the original content")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")
    
    @validator('question')
    def validate_question(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Question must be at least 10 characters long")
        return v.strip()
    
    @validator('answer')
    def validate_answer(cls, v):
        if len(v.strip()) < 20:
            raise ValueError("Answer must be at least 20 characters long")
        return v.strip()

class QACollection(BaseModel):
    """Collection of Q&A pairs from a content chunk"""
    qa_pairs: List[QuestionAnswer] = Field(description="List of question-answer pairs")
    chunk_summary: str = Field(description="Brief summary of the content chunk")
    source_file: Optional[str] = Field(description="Source file path", default=None)
    chunk_id: Optional[str] = Field(description="Unique chunk identifier", default=None)

@dataclass
class Dependencies:
    """Dependencies for the Pydantic AI agent"""
    config: Config
    content_chunk: str
    chunk_id: str
    source_file: str

class MarkdownScanner:
    """Scans directories recursively for markdown files"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def scan_directory(self, directory_path: Union[str, Path]) -> List[Path]:
        """Recursively scan directory for markdown files"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        logger.info(f"Scanning directory: {directory_path}")
        
        markdown_files = []
        
        # Recursively find all markdown files
        for ext in self.config.file_extensions:
            pattern = f"**/*{ext}"
            files = list(directory_path.glob(pattern))
            markdown_files.extend(files)
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in markdown_files:
            should_exclude = False
            for pattern in self.config.exclude_patterns:
                if pattern.lower() in file_path.name.lower():
                    should_exclude = True
                    break
            
            if not should_exclude:
                filtered_files.append(file_path)
            else:
                logger.debug(f"Excluding file: {file_path}")
        
        # Sort files for consistent processing order
        filtered_files.sort()
        
        # Limit number of files if specified
        if self.config.max_files_to_process:
            filtered_files = filtered_files[:self.config.max_files_to_process]
            logger.info(f"Limited to first {self.config.max_files_to_process} files")
        
        logger.info(f"Found {len(filtered_files)} markdown files to process")
        
        return filtered_files
    
    def get_file_stats(self, files: List[Path]) -> Dict[str, Any]:
        """Get statistics about the files to be processed"""
        total_size = 0
        file_sizes = []
        extensions = {}
        
        for file_path in files:
            try:
                size = file_path.stat().st_size
                total_size += size
                file_sizes.append(size)
                
                ext = file_path.suffix.lower()
                extensions[ext] = extensions.get(ext, 0) + 1
                
            except OSError:
                logger.warning(f"Could not get stats for file: {file_path}")
        
        return {
            'total_files': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'average_size_bytes': round(sum(file_sizes) / len(file_sizes)) if file_sizes else 0,
            'extensions': extensions,
            'largest_file_bytes': max(file_sizes) if file_sizes else 0,
            'smallest_file_bytes': min(file_sizes) if file_sizes else 0
        }

class MarkdownProcessor:
    """Processes markdown files and extracts content"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def load_markdown_file(self, file_path: Path) -> str:
        """Load and convert markdown file to text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            
            # Convert markdown to HTML then to text
            html = markdown.markdown(markdown_content, extensions=['extra', 'codehilite'])
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            
            # Clean up the text
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
            text = re.sub(r' +', ' ', text)  # Normalize spaces
            text = re.sub(r'\t+', ' ', text)  # Replace tabs with spaces
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
            return ""
    
    def chunk_content(self, content: str, source_file: str) -> List[Dict[str, Any]]:
        """Split content into overlapping chunks"""
        if len(content) < self.config.min_chunk_length:
            logger.warning(f"Content too short to chunk: {source_file}")
            return []
        
        chunks = []
        chunk_size = self.config.max_context_length
        overlap = self.config.chunk_overlap
        
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # If we're not at the end, try to break at a sentence or paragraph
            if end < len(content):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                    if content[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk_text = content[start:end].strip()
            if len(chunk_text) >= self.config.min_chunk_length:
                chunks.append({
                    'id': f"{Path(source_file).stem}_chunk_{chunk_id:04d}",
                    'content': chunk_text,
                    'start_pos': start,
                    'end_pos': end,
                    'source_file': source_file,
                    'word_count': len(chunk_text.split())
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - overlap
            if start <= 0 or start >= len(content):
                break
        
        logger.debug(f"Created {len(chunks)} chunks from {source_file}")
        return chunks

class QAGenerator:
    """Generates Q&A pairs using Pydantic AI and Ollama"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent = None
        self._ensure_ollama_model()
        self.agent = self._create_agent()
    
    def _ensure_ollama_model(self):
        """Ensure the Ollama model is available"""
        try:
            # Check if model exists
            models = ollama.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.config.ollama_model not in model_names:
                logger.info(f"Pulling Ollama model: {self.config.ollama_model}")
                try:
                    ollama.pull(self.config.ollama_model)
                    logger.info(f"Successfully pulled model: {self.config.ollama_model}")
                except Exception as e:
                    logger.warning(f"Failed to pull primary model {self.config.ollama_model}: {e}")
                    logger.info(f"Trying backup model: {self.config.backup_ollama_model}")
                    
                    if self.config.backup_ollama_model not in model_names:
                        ollama.pull(self.config.backup_ollama_model)
                    self.config.ollama_model = self.config.backup_ollama_model
            else:
                logger.info(f"Using existing Ollama model: {self.config.ollama_model}")
                
        except Exception as e:
            logger.error(f"Error with Ollama model: {e}")
            raise
    
    def _create_agent(self) -> Agent:
        """Create the Pydantic AI agent for Q&A generation"""
        
        # Configure Ollama model for Pydantic AI
        model_config = f"ollama:{self.config.ollama_model}"
        
        agent = Agent(
            model_config,
            deps_type=Dependencies,
            result_type=QACollection,
            system_prompt=(
                "You are an expert at creating educational Q&A pairs from text content. "
                "Your task is to generate diverse, high-quality questions and comprehensive answers "
                "that help people learn and understand the material. "
                "Focus on creating questions that test different types of understanding: "
                "factual recall, conceptual understanding, application, and analysis. "
                "Ensure all questions and answers meet the minimum length requirements."
            )
        )
        
        @agent.system_prompt
        def dynamic_system_prompt(ctx: RunContext[Dependencies]) -> str:
            return (
                f"Generate exactly {self.config.num_questions_per_chunk} question-answer pairs "
                f"from the provided content chunk (ID: {ctx.deps.chunk_id}) from file: {ctx.deps.source_file}. "
                f"Create questions of varying difficulty levels and types. "
                f"Questions must be at least {self.config.min_question_length} characters long. "
                f"Answers must be at least {self.config.min_answer_length} characters long. "
                f"Ensure answers are comprehensive and based solely on the provided content."
            )
        
        @agent.tool
        async def analyze_content_structure(
            ctx: RunContext[Dependencies]
        ) -> str:
            """Analyze the structure and key topics in the content chunk"""
            content = ctx.deps.content_chunk
            
            # Simple analysis - count sentences, identify potential topics
            sentences = len(re.findall(r'[.!?]+', content))
            words = len(content.split())
            
            # Extract potential key terms (simple approach)
            words_list = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{4,}\b', content)
            common_terms = {}
            for word in words_list:
                if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'they', 'have', 'will', 'been', 'were']:
                    common_terms[word.lower()] = common_terms.get(word.lower(), 0) + 1
            
            top_terms = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return (
                f"Content analysis for {ctx.deps.source_file}: "
                f"{sentences} sentences, {words} words. "
                f"Key terms: {', '.join([term for term, _ in top_terms[:5]])}. "
                f"Use this analysis to create diverse questions covering the main concepts."
            )
        
        return agent
    
    async def generate_qa_pairs(self, chunk_data: Dict[str, Any]) -> Optional[QACollection]:
        """Generate Q&A pairs for a content chunk with retry logic"""
        
        for attempt in range(self.config.max_retries_per_chunk):
            try:
                deps = Dependencies(
                    config=self.config,
                    content_chunk=chunk_data['content'],
                    chunk_id=chunk_data['id'],
                    source_file=chunk_data['source_file']
                )
                
                logger.debug(f"Generating Q&A pairs for chunk {chunk_data['id']} (attempt {attempt + 1})")
                
                result = await self.agent.run(
                    f"Create educational Q&A pairs from this content:\n\n{chunk_data['content'][:1500]}...",
                    deps=deps
                )
                
                # Validate the result
                qa_collection = result.data
                if qa_collection and len(qa_collection.qa_pairs) > 0:
                    # Add metadata
                    qa_collection.source_file = chunk_data['source_file']
                    qa_collection.chunk_id = chunk_data['id']
                    
                    logger.info(f"Successfully generated {len(qa_collection.qa_pairs)} Q&A pairs for chunk {chunk_data['id']}")
                    return qa_collection
                else:
                    logger.warning(f"No Q&A pairs generated for chunk {chunk_data['id']}")
                    
            except Exception as e:
                logger.error(f"Error generating Q&A for chunk {chunk_data['id']} (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries_per_chunk - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to generate Q&A pairs for chunk {chunk_data['id']} after {self.config.max_retries_per_chunk} attempts")
        return None

# DatasetFormatter, UnslothIntegration classes remain the same as previous version
# ... (keeping the same implementation for brevity)

class DatasetFormatter:
    """Formats Q&A pairs for different fine-tuning frameworks"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def format_for_unsloth(self, qa_collections: List[QACollection]) -> List[Dict[str, Any]]:
        """Format Q&A pairs for Unsloth fine-tuning"""
        formatted_data = []
        
        for collection in qa_collections:
            for qa_pair in collection.qa_pairs:
                formatted_data.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that provides accurate answers based on the given context."
                        },
                        {
                            "role": "user",
                            "content": qa_pair.question
                        },
                        {
                            "role": "assistant",
                            "content": qa_pair.answer
                        }
                    ],
                    "metadata": {
                        "difficulty": qa_pair.difficulty,
                        "context": qa_pair.context[:200] + "..." if len(qa_pair.context) > 200 else qa_pair.context,
                        "source_file": collection.source_file,
                        "chunk_id": collection.chunk_id
                    }
                })
        
        return formatted_data
    
    def format_for_alpaca(self, qa_collections: List[QACollection]) -> List[Dict[str, Any]]:
        """Format Q&A pairs in Alpaca format"""
        formatted_data = []
        
        for collection in qa_collections:
            for qa_pair in collection.qa_pairs:
                formatted_data.append({
                    "instruction": qa_pair.question,
                    "input": "",
                    "output": qa_pair.answer,
                    "difficulty": qa_pair.difficulty,
                    "context": qa_pair.context,
                    "source_file": collection.source_file,
                    "chunk_id": collection.chunk_id
                })
        
        return formatted_data
    
    def format_for_sharegpt(self, qa_collections: List[QACollection]) -> List[Dict[str, Any]]:
        """Format Q&A pairs in ShareGPT format"""
        formatted_data = []
        
        for collection in qa_collections:
            for qa_pair in collection.qa_pairs:
                formatted_data.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": qa_pair.question
                        },
                        {
                            "from": "gpt",
                            "value": qa_pair.answer
                        }
                    ],
                    "metadata": {
                        "difficulty": qa_pair.difficulty,
                        "context": qa_pair.context,
                        "source_file": collection.source_file,
                        "chunk_id": collection.chunk_id
                    }
                })
        
        return formatted_data

class UnslothIntegration:
    """Integration with Unsloth for fine-tuning preparation"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def prepare_dataset_for_unsloth(self, formatted_data: List[Dict[str, Any]]) -> Dataset:
        """Prepare dataset for Unsloth fine-tuning"""
        
        # Convert to Hugging Face Dataset
        dataset = Dataset.from_list(formatted_data)
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        
        return dataset
    
    def create_fine_tuning_script(self, dataset_path: str, model_name: Optional[str] = None) -> str:
        """Generate a complete fine-tuning script for Unsloth"""
        
        if model_name is None:
            model_name = self.config.default_base_model
        
        script = f'''#!/usr/bin/env python3
"""
Generated fine-tuning script using Unsloth
Dataset: {dataset_path}
Generated on: {datetime.now().isoformat()}
Configuration: {self.config.config_file if self.config.config_file else 'Default'}
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Configuration from config file
max_seq_length = {self.config.max_context_length}
model_name = "{model_name}"
dataset_path = "{dataset_path}"
num_train_epochs = {self.config.training_epochs}
learning_rate = {self.config.learning_rate}
per_device_train_batch_size = {self.config.batch_size}
gradient_accumulation_steps = {self.config.gradient_accumulation_steps}

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect
    load_in_4bit=True,
)

# Apply PEFT/LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
)

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

def format_messages(example):
    """Format messages for training"""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {{"text": text}}

# Format dataset
dataset = dataset.map(format_messages, remove_columns=dataset.column_names)

# Training configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=True,
    args=TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="./fine_tuned_model",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",  # Disable wandb/tensorboard
    ),
)

# Train the model
print("Starting fine-tuning...")
trainer_stats = trainer.train()

# Save the model
print("Saving model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Optionally save as merged model (uncomment if needed)
# print("Saving merged model...")
# model.save_pretrained_merged("./merged_model", tokenizer, save_method="merged_16bit")

print("Fine-tuning completed!")
print(f"Training stats: {{trainer_stats}}")
'''
        return script

class MarkdownToQAConverter:
    """Main class that orchestrates the conversion process"""
    
    def __init__(self, config: Config):
        self.config = config
        self.scanner = MarkdownScanner(config)
        self.processor = MarkdownProcessor(config)
        self.generator = QAGenerator(config)
        self.formatter = DatasetFormatter(config)
        self.unsloth = UnslothIntegration(config)
    
    async def convert_directory(self, input_directory: Optional[str] = None, output_directory: Optional[str] = None) -> Dict[str, Any]:
        """Convert all markdown files in directory to Q&A dataset"""
        
        # Use provided directories or fall back to config
        input_dir = Path(input_directory) if input_directory else Path(self.config.input_directory)
        output_dir = Path(output_directory) if output_directory else Path(self.config.output_directory)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the configuration used for this run
        config_backup_path = output_dir / "config_used.yaml"
        self.config.to_yaml(str(config_backup_path))
        logger.info(f"Saved configuration to: {config_backup_path}")
        
        # Scan for markdown files
        markdown_files = self.scanner.scan_directory(input_dir)
        
        if not markdown_files:
            logger.warning("No markdown files found to process")
            return {"error": "No markdown files found"}
        
        # Get file statistics
        file_stats = self.scanner.get_file_stats(markdown_files)
        logger.info(f"File statistics: {file_stats}")
        
        all_qa_collections = []
        processed_files = []
        failed_files = []
        
        # Process each markdown file
        total_files = len(markdown_files)
        for i, file_path in enumerate(markdown_files, 1):
            logger.info(f"Processing file {i}/{total_files}: {file_path}")
            
            try:
                # Load and chunk content
                content = self.processor.load_markdown_file(file_path)
                if not content:
                    logger.warning(f"Skipping empty file: {file_path}")
                    failed_files.append({
                        'file': str(file_path),
                        'error': 'Empty or unreadable content'
                    })
                    continue
                
                chunks = self.processor.chunk_content(content, str(file_path))
                if not chunks:
                    logger.warning(f"No valid chunks created from: {file_path}")
                    failed_files.append({
                        'file': str(file_path),
                        'error': 'No valid chunks created'
                    })
                    continue
                
                logger.info(f"Created {len(chunks)} chunks from {file_path}")
                
                # Generate Q&A pairs for each chunk
                file_qa_collections = []
                successful_chunks = 0
                
                for chunk in chunks:
                    qa_collection = await self.generator.generate_qa_pairs(chunk)
                    if qa_collection:
                        file_qa_collections.append(qa_collection)
                        all_qa_collections.append(qa_collection)
                        successful_chunks += 1
                    
                    # Add small delay to avoid overwhelming the model
                    await asyncio.sleep(0.1)
                
                processed_files.append({
                    'file': str(file_path),
                    'total_chunks': len(chunks),
                    'successful_chunks': successful_chunks,
                    'qa_collections': len(file_qa_collections),
                    'total_qa_pairs': sum(len(collection.qa_pairs) for collection in file_qa_collections),
                    'file_size_bytes': file_path.stat().st_size
                })
                
                logger.info(f"Successfully processed {successful_chunks}/{len(chunks)} chunks from {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                failed_files.append({
                    'file': str(file_path),
                    'error': str(e)
                })
        
        if not all_qa_collections:
            logger.error("No Q&A collections were generated successfully")
            return {"error": "No Q&A collections generated"}
        
        logger.info(f"Generated {len(all_qa_collections)} Q&A collections total")
        
        # Format the data
        if self.config.output_format == "unsloth":
            formatted_data = self.formatter.format_for_unsloth(all_qa_collections)
        elif self.config.output_format == "alpaca":
            formatted_data = self.formatter.format_for_alpaca(all_qa_collections)
        elif self.config.output_format == "sharegpt":
            formatted_data = self.formatter.format_for_sharegpt(all_qa_collections)
        else:
            raise ValueError(f"Unsupported output format: {self.config.output_format}")
        
        # Save the dataset
        dataset_path = output_dir / f"qa_dataset_{self.config.output_format}.json"
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved dataset to: {dataset_path}")
        
        # Create Hugging Face dataset
        hf_dataset = self.unsloth.prepare_dataset_for_unsloth(formatted_data)
        hf_dataset_path = output_dir / "hf_dataset"
        hf_dataset.save_to_disk(str(hf_dataset_path))
        
        # Generate fine-tuning script
        fine_tuning_script = self.unsloth.create_fine_tuning_script(str(dataset_path))
        script_path = output_dir / "fine_tune_with_unsloth.py"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(fine_tuning_script)
        
        # Generate comprehensive report
        total_qa_pairs = sum(len(collection.qa_pairs) for collection in all_qa_collections)
        
        # Calculate quality metrics
        difficulty_distribution = {}
        for collection in all_qa_collections:
            for qa_pair in collection.qa_pairs:
                difficulty = qa_pair.difficulty
                difficulty_distribution[difficulty] = difficulty_distribution.get(difficulty, 0) + 1
        
        report = {
            'summary': {
                'input_directory': str(input_dir),
                'output_directory': str(output_dir),
                'processing_timestamp': datetime.now().isoformat(),
                'configuration_file': self.config.config_file,
                'total_files_found': total_files,
                'successfully_processed_files': len(processed_files),
                'failed_files': len(failed_files),
                'total_chunks': sum(f['total_chunks'] for f in processed_files),
                'successful_chunks': sum(f['successful_chunks'] for f in processed_files),
                'total_qa_collections': len(all_qa_collections),
                'total_qa_pairs': total_qa_pairs,
                'output_format': self.config.output_format,
                'dataset_path': str(dataset_path),
                'hf_dataset_path': str(hf_dataset_path),
                'fine_tuning_script': str(script_path),
                'config_backup': str(config_backup_path)
            },
            'file_statistics': file_stats,
            'processed_files': processed_files,
            'failed_files': failed_files,
            'quality_metrics': {
                'difficulty_distribution': difficulty_distribution,
                'average_qa_pairs_per_collection': round(total_qa_pairs / len(all_qa_collections), 2) if all_qa_collections else 0,
                'average_qa_pairs_per_file': round(total_qa_pairs / len(processed_files), 2) if processed_files else 0
            },
            'configuration': {
                'ollama_model': self.config.ollama_model,
                'max_context_length': self.config.max_context_length,
                'num_questions_per_chunk': self.config.num_questions_per_chunk,
                'chunk_overlap': self.config.chunk_overlap,
                'file_extensions': self.config.file_extensions,
                'exclude_patterns': self.config.exclude_patterns
            }
        }
        
        report_path = output_dir / "conversion_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion completed! Report saved to: {report_path}")
        return report

def create_default_config(config_path: str):
    """Create a default configuration file"""
    config = Config()
    config.to_yaml(config_path)
    logger.info(f"Created default configuration file: {config_path}")

# CLI interface
async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Markdown documents to Q&A fine-tuning dataset")
    parser.add_argument("input_directory", nargs='?', help="Input directory containing markdown files")
    parser.add_argument("--config", "-c", help="Configuration file path (YAML)")
    parser.add_argument("--output-dir", "-o", help="Output directory (overrides config)")
    parser.add_argument("--create-config", help="Create default configuration file and exit")
    parser.add_argument("--dry-run", action="store_true", help="Scan files but don't process them")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        create_default_config(args.create_config)
        return
    
    # Load configuration
    if args.config:
        if not Path(args.config).exists():
            logger.error(f"Configuration file not found: {args.config}")
            return
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.input_directory:
        config.input_directory = args.input_directory
    if args.output_dir:
        config.output_directory = args.output_dir
    
    # Validate configuration
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Show configuration
    logger.info(f"Using configuration:")
    logger.info(f"  Input directory: {config.input_directory}")
    logger.info(f"  Output directory: {config.output_directory}")
    logger.info(f"  Ollama model: {config.ollama_model}")
    logger.info(f"  Output format: {config.output_format}")
    logger.info(f"  Questions per chunk: {config.num_questions_per_chunk}")
    
    # Dry run - just scan files
    if args.dry_run:
        scanner = MarkdownScanner(config)
        files = scanner.scan_directory(config.input_directory)
        stats = scanner.get_file_stats(files)
        
        print("\n" + "="*50)
        print("DRY RUN RESULTS")
        print("="*50)
        print(f"Files found: {stats['total_files']}")
        print(f"Total size: {stats['total_size_mb']} MB")
        print(f"Extensions: {stats['extensions']}")
        print("\nFiles to process:")
        for i, file_path in enumerate(files[:10], 1):  # Show first 10 files
            print(f"  {i}. {file_path}")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        return
    
    # Convert files
    converter = MarkdownToQAConverter(config)
    
    try:
        report = await converter.convert_directory()
        
        if "error" in report:
            logger.error(f"Conversion failed: {report['error']}")
            return
        
        print("\n" + "="*70)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"Input directory: {report['summary']['input_directory']}")
        print(f"Output directory: {report['summary']['output_directory']}")
        print(f"Processed files: {report['summary']['successfully_processed_files']}/{report['summary']['total_files_found']}")
        print(f"Total Q&A pairs: {report['summary']['total_qa_pairs']}")
        print(f"Dataset format: {report['summary']['output_format']}")
        print(f"Dataset saved to: {report['summary']['dataset_path']}")
        print(f"Fine-tuning script: {report['summary']['fine_tuning_script']}")
        print(f"Detailed report: {Path(report['summary']['output_directory']) / 'conversion_report.json'}")
        
        if report['summary']['failed_files'] > 0:
            print(f"\nWarning: {report['summary']['failed_files']} files failed to process")
        
        print(f"\nQuality metrics:")
        for difficulty, count in report['quality_metrics']['difficulty_distribution'].items():
            print(f"  {difficulty}: {count} questions")
        
        print("\nTo start fine-tuning, run:")
        print(f"python {report['summary']['fine_tuning_script']}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
