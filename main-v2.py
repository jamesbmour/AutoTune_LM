#!/usr/bin/env python3
"""
Markdown to Q&A Fine-tuning Dataset Converter

This script converts unstructured Markdown documents into Q&A pairs for fine-tuning
language models using Pydantic AI for question generation, Ollama for inference,
and Unsloth for fine-tuning preparation.

Requirements:
    pip install pydantic-ai ollama unsloth datasets transformers trl
"""

import asyncio
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from datetime import datetime

import ollama
from pydantic import BaseModel, Field
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
    ollama_model: str = "llama3.2:3b"  # Default Ollama model
    max_context_length: int = 2048     # Maximum context length for chunks
    chunk_overlap: int = 200           # Overlap between chunks
    num_questions_per_chunk: int = 3   # Questions to generate per chunk
    output_format: str = "unsloth"     # Output format: "unsloth", "alpaca", or "sharegpt"
    min_chunk_length: int = 100        # Minimum chunk length to process

# Pydantic models for structured output
class QuestionAnswer(BaseModel):
    """Single Q&A pair generated from content"""
    question: str = Field(description="A clear, specific question about the content")
    answer: str = Field(description="A comprehensive answer based on the content")
    context: str = Field(description="Relevant context snippet from the original content")
    difficulty: str = Field(description="Difficulty level: easy, medium, or hard")

class QACollection(BaseModel):
    """Collection of Q&A pairs from a content chunk"""
    qa_pairs: List[QuestionAnswer] = Field(description="List of question-answer pairs")
    chunk_summary: str = Field(description="Brief summary of the content chunk")

@dataclass
class Dependencies:
    """Dependencies for the Pydantic AI agent"""
    config: Config
    content_chunk: str
    chunk_id: str

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
            html = markdown.markdown(markdown_content)
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text()
            
            # Clean up the text
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
            text = re.sub(r' +', ' ', text)  # Normalize spaces
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error loading markdown file {file_path}: {e}")
            return ""
    
    def chunk_content(self, content: str) -> List[Dict[str, Any]]:
        """Split content into overlapping chunks"""
        if len(content) < self.config.min_chunk_length:
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
                    'id': f"chunk_{chunk_id:04d}",
                    'content': chunk_text,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - overlap
            if start <= 0 or start >= len(content):
                break
        
        return chunks

class QAGenerator:
    """Generates Q&A pairs using Pydantic AI and Ollama"""
    
    def __init__(self, config: Config):
        self.config = config
        self.agent = self._create_agent()
        self._ensure_ollama_model()
    
    def _ensure_ollama_model(self):
        """Ensure the Ollama model is available"""
        try:
            # Check if model exists
            models = ollama.list()
            model_names = [model['name'] for model in models['models']]
            
            if self.config.ollama_model not in model_names:
                logger.info(f"Pulling Ollama model: {self.config.ollama_model}")
                ollama.pull(self.config.ollama_model)
                logger.info(f"Successfully pulled model: {self.config.ollama_model}")
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
            output_type=QACollection,
            instructions=(
                "You are an expert at creating educational Q&A pairs from text content. "
                "Your task is to generate diverse, high-quality questions and comprehensive answers "
                "that help people learn and understand the material. "
                "Focus on creating questions that test different types of understanding: "
                "factual recall, conceptual understanding, application, and analysis."
            )
        )
        
        @agent.instructions
        async def dynamic_instructions(ctx: RunContext[Dependencies]) -> str:
            return (
                f"Generate exactly {self.config.num_questions_per_chunk} question-answer pairs "
                f"from the provided content chunk (ID: {ctx.deps.chunk_id}). "
                "Create questions of varying difficulty levels and types. "
                "Ensure answers are comprehensive and based solely on the provided content."
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
                if len(word) > 3:
                    common_terms[word] = common_terms.get(word, 0) + 1
            
            top_terms = sorted(common_terms.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return (
                f"Content analysis: {sentences} sentences, {words} words. "
                f"Key terms: {', '.join([term for term, _ in top_terms[:5]])}. "
                "Use this analysis to create diverse questions covering the main concepts."
            )
        
        return agent
    
    async def generate_qa_pairs(self, chunk_data: Dict[str, Any]) -> Optional[QACollection]:
        """Generate Q&A pairs for a content chunk"""
        try:
            deps = Dependencies(
                config=self.config,
                content_chunk=chunk_data['content'],
                chunk_id=chunk_data['id']
            )
            
            logger.info(f"Generating Q&A pairs for chunk {chunk_data['id']}")
            
            result = await self.agent.run(
                f"Create educational Q&A pairs from this content:\n\n{chunk_data['content'][:1000]}...",
                deps=deps
            )
            
            return result.output
            
        except Exception as e:
            logger.error(f"Error generating Q&A for chunk {chunk_data['id']}: {e}")
            return None

class DatasetFormatter:
    """Formats Q&A pairs for different fine-tuning frameworks"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def format_for_unsloth(self, qa_collections: List[QACollection]) -> List[Dict[str, Any]]:
        """Format Q&A pairs for Unsloth fine-tuning"""
        formatted_data = []
        
        for collection in qa_collections:
            for qa_pair in collection.qa_pairs:
                # Unsloth typically uses a conversation format
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
                        "context": qa_pair.context[:200] + "..." if len(qa_pair.context) > 200 else qa_pair.context
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
                    "context": qa_pair.context
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
                        "context": qa_pair.context
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
    
    def create_fine_tuning_script(self, dataset_path: str, model_name: str = "unsloth/llama-3.2-3b-instruct-bnb-4bit") -> str:
        """Generate a complete fine-tuning script for Unsloth"""
        
        script = f'''#!/usr/bin/env python3
"""
Generated fine-tuning script using Unsloth
Dataset: {dataset_path}
Generated on: {datetime.now().isoformat()}
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# Configuration
max_seq_length = {self.config.max_context_length}
model_name = "{model_name}"
dataset_path = "{dataset_path}"

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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=3,
        learning_rate=2e-4,
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
    ),
)

# Train the model
trainer_stats = trainer.train()

# Save the model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Optionally save as merged model
# model.save_pretrained_merged("./merged_model", tokenizer, save_method="merged_16bit")

print("Fine-tuning completed!")
print(f"Training stats: {{trainer_stats}}")
'''
        return script

class MarkdownToQAConverter:
    """Main class that orchestrates the conversion process"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = MarkdownProcessor(config)
        self.generator = QAGenerator(config)
        self.formatter = DatasetFormatter(config)
        self.unsloth = UnslothIntegration(config)
    
    async def convert_files(self, input_paths: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Convert markdown files to Q&A dataset"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_qa_collections = []
        processed_files = []
        
        # Process each markdown file
        for file_path in input_paths:
            logger.info(f"Processing file: {file_path}")
            
            # Load and chunk content
            content = self.processor.load_markdown_file(file_path)
            if not content:
                logger.warning(f"Skipping empty file: {file_path}")
                continue
            
            chunks = self.processor.chunk_content(content)
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            
            # Generate Q&A pairs for each chunk
            file_qa_collections = []
            for chunk in chunks:
                qa_collection = await self.generator.generate_qa_pairs(chunk)
                if qa_collection:
                    file_qa_collections.append(qa_collection)
                    all_qa_collections.append(qa_collection)
                
                # Add small delay to avoid overwhelming the model
                await asyncio.sleep(0.1)
            
            processed_files.append({
                'file': str(file_path),
                'chunks': len(chunks),
                'qa_collections': len(file_qa_collections)
            })
        
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
        
        # Generate summary report
        total_qa_pairs = sum(len(collection.qa_pairs) for collection in all_qa_collections)
        
        report = {
            'summary': {
                'processed_files': len(processed_files),
                'total_chunks': sum(f['chunks'] for f in processed_files),
                'total_qa_collections': len(all_qa_collections),
                'total_qa_pairs': total_qa_pairs,
                'output_format': self.config.output_format,
                'dataset_path': str(dataset_path),
                'hf_dataset_path': str(hf_dataset_path),
                'fine_tuning_script': str(script_path)
            },
            'files': processed_files,
            'config': {
                'ollama_model': self.config.ollama_model,
                'max_context_length': self.config.max_context_length,
                'num_questions_per_chunk': self.config.num_questions_per_chunk,
                'chunk_overlap': self.config.chunk_overlap
            }
        }
        
        report_path = output_dir / "conversion_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion completed! Report saved to: {report_path}")
        return report

# CLI interface
async def main():
    """Main function for CLI usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Markdown documents to Q&A fine-tuning dataset")
    parser.add_argument("input_files", nargs="+", help="Input markdown files")
    parser.add_argument("--output-dir", "-o", default="./qa_dataset", help="Output directory")
    parser.add_argument("--model", "-m", default="llama3.2:3b", help="Ollama model to use")
    parser.add_argument("--format", "-f", choices=["unsloth", "alpaca", "sharegpt"], 
                       default="unsloth", help="Output format")
    parser.add_argument("--questions-per-chunk", "-q", type=int, default=3, 
                       help="Number of questions per chunk")
    parser.add_argument("--max-context", type=int, default=2048, 
                       help="Maximum context length")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                       help="Chunk overlap size")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        ollama_model=args.model,
        max_context_length=args.max_context,
        chunk_overlap=args.chunk_overlap,
        num_questions_per_chunk=args.questions_per_chunk,
        output_format=args.format
    )
    
    # Convert files
    input_paths = [Path(f) for f in args.input_files]
    output_dir = Path(args.output_dir)
    
    converter = MarkdownToQAConverter(config)
    
    try:
        report = await converter.convert_files(input_paths, output_dir)
        
        print("\n" + "="*50)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Processed files: {report['summary']['processed_files']}")
        print(f"Total Q&A pairs: {report['summary']['total_qa_pairs']}")
        print(f"Dataset saved to: {report['summary']['dataset_path']}")
        print(f"Fine-tuning script: {report['summary']['fine_tuning_script']}")
        print("\nTo start fine-tuning, run:")
        print(f"python {report['summary']['fine_tuning_script']}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
