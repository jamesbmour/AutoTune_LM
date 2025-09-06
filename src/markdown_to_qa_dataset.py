"""
Markdown to Q&A Dataset Converter
Converts unstructured markdown documents into Q&A pairs for LLM fine-tuning with Unsloth.
Uses PydanticAI for structured generation and Ollama for local LLM inference.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Core dependencies
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import OllamaModel
import markdown
from bs4 import BeautifulSoup
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QAPair(BaseModel):
    """Structured Q&A pair model"""
    question: str = Field(description="A clear, specific question about the content")
    answer: str = Field(description="A comprehensive answer based on the content")
    context: str = Field(description="The source context from which the Q&A was derived")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")
    category: str = Field(description="Content category or topic area")


class QADataset(BaseModel):
    """Collection of Q&A pairs"""
    pairs: List[QAPair] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MarkdownProcessor:
    """Processes markdown files and extracts content"""
    
    def __init__(self):
        self.md = markdown.Markdown(extensions=['extra', 'codehilite'])
    
    def extract_content_from_markdown(self, file_path: Path) -> List[str]:
        """Extract content chunks from markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Convert markdown to HTML
            html = self.md.convert(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract different content types
            chunks = []
            
            # Extract headers and following content
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                chunk = heading.get_text().strip()
                
                # Get content until next heading
                current = heading.next_sibling
                content_parts = [chunk]
                
                while current and current.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if hasattr(current, 'get_text'):
                        text = current.get_text().strip()
                        if text:
                            content_parts.append(text)
                    elif isinstance(current, str):
                        text = current.strip()
                        if text:
                            content_parts.append(text)
                    current = current.next_sibling
                
                if len(content_parts) > 1:
                    chunks.append(' '.join(content_parts))
            
            # Extract code blocks
            for code_block in soup.find_all('code'):
                code_text = code_block.get_text().strip()
                if len(code_text) > 50:  # Only substantial code blocks
                    chunks.append(f"Code example: {code_text}")
            
            # Extract list items
            for ul in soup.find_all(['ul', 'ol']):
                list_text = ul.get_text().strip()
                if len(list_text) > 100:  # Only substantial lists
                    chunks.append(list_text)
            
            # Filter out very short chunks
            chunks = [chunk for chunk in chunks if len(chunk) > 50]
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []


class QAGenerator:
    """Generates Q&A pairs using PydanticAI and Ollama"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """Initialize with Ollama model"""
        self.model = OllamaModel(model_name)
        self.agent = Agent(
            model=self.model,
            result_type=List[QAPair],
            system_prompt="""
            You are an expert at creating high-quality question-answer pairs for LLM fine-tuning.
            
            Your task is to generate diverse, educational Q&A pairs from the given content.
            
            Guidelines:
            1. Create questions that test understanding, application, and analysis
            2. Vary question difficulty (easy, medium, hard)
            3. Include different question types: factual, conceptual, procedural
            4. Ensure answers are comprehensive but concise
            5. Use the exact content provided as context
            6. Generate 2-5 Q&A pairs per content chunk depending on its richness
            
            Question types to include:
            - What/Who/When/Where questions (factual)
            - How/Why questions (procedural/conceptual)
            - Compare/contrast questions
            - Application/example questions
            """
        )
    
    async def generate_qa_pairs(self, content_chunk: str, category: str = "general") -> List[QAPair]:
        """Generate Q&A pairs from a content chunk"""
        try:
            prompt = f"""
            Generate high-quality question-answer pairs from this content:
            
            Content: {content_chunk}
            Category: {category}
            
            Create 2-5 diverse Q&A pairs with varying difficulty levels.
            Ensure questions are specific and answers are informative.
            """
            
            result = await self.agent.run(prompt)
            
            # Ensure all pairs have the content as context and correct category
            for pair in result.data:
                pair.context = content_chunk
                pair.category = category
            
            return result.data
            
        except Exception as e:
            logger.error(f"Error generating Q&A pairs: {e}")
            return []


class UnslothFormatter:
    """Formats Q&A pairs for Unsloth fine-tuning"""
    
    @staticmethod
    def format_for_unsloth(qa_pairs: List[QAPair], format_type: str = "alpaca") -> List[Dict[str, str]]:
        """Format Q&A pairs for Unsloth training"""
        formatted_data = []
        
        for pair in qa_pairs:
            if format_type == "alpaca":
                formatted_data.append({
                    "instruction": pair.question,
                    "input": "",
                    "output": pair.answer
                })
            elif format_type == "chatml":
                formatted_data.append({
                    "messages": [
                        {"role": "user", "content": pair.question},
                        {"role": "assistant", "content": pair.answer}
                    ]
                })
            elif format_type == "sharegpt":
                formatted_data.append({
                    "conversations": [
                        {"from": "human", "value": pair.question},
                        {"from": "gpt", "value": pair.answer}
                    ]
                })
        
        return formatted_data
    
    @staticmethod
    def save_dataset(formatted_data: List[Dict], output_path: Path, format_type: str = "json"):
        """Save formatted dataset"""
        if format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data, f, indent=2, ensure_ascii=False)
        elif format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in formatted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format_type == "csv":
            df = pd.DataFrame(formatted_data)
            df.to_csv(output_path, index=False)


class MarkdownToQAConverter:
    """Main converter class"""
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.processor = MarkdownProcessor()
        self.generator = QAGenerator(model_name)
        self.formatter = UnslothFormatter()
    
    async def convert_directory(
        self, 
        input_dir: Path, 
        output_path: Path,
        dataset_format: str = "alpaca",
        file_format: str = "jsonl",
        max_files: Optional[int] = None
    ) -> QADataset:
        """Convert all markdown files in directory to Q&A dataset"""
        
        markdown_files = list(input_dir.glob("**/*.md"))
        if max_files:
            markdown_files = markdown_files[:max_files]
        
        logger.info(f"Found {len(markdown_files)} markdown files")
        
        all_qa_pairs = []
        processed_files = 0
        
        for file_path in markdown_files:
            logger.info(f"Processing: {file_path}")
            
            # Extract content chunks
            content_chunks = self.processor.extract_content_from_markdown(file_path)
            logger.info(f"Extracted {len(content_chunks)} content chunks")
            
            # Generate Q&A pairs for each chunk
            for chunk in content_chunks:
                category = file_path.stem  # Use filename as category
                qa_pairs = await self.generator.generate_qa_pairs(chunk, category)
                all_qa_pairs.extend(qa_pairs)
                
                # Small delay to avoid overwhelming the model
                await asyncio.sleep(0.1)
            
            processed_files += 1
            logger.info(f"Processed {processed_files}/{len(markdown_files)} files")
        
        logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs total")
        
        # Create dataset
        dataset = QADataset(
            pairs=all_qa_pairs,
            metadata={
                "created_at": datetime.now().isoformat(),
                "source_files": [str(f) for f in markdown_files],
                "total_pairs": len(all_qa_pairs),
                "model_used": self.generator.model.model_name if hasattr(self.generator.model, 'model_name') else "unknown"
            }
        )
        
        # Format for Unsloth
        formatted_data = self.formatter.format_for_unsloth(all_qa_pairs, dataset_format)
        
        # Save dataset
        self.formatter.save_dataset(formatted_data, output_path, file_format)
        logger.info(f"Saved dataset to {output_path}")
        
        return dataset
    
    async def convert_single_file(
        self, 
        input_file: Path, 
        output_path: Path,
        dataset_format: str = "alpaca",
        file_format: str = "jsonl"
    ) -> QADataset:
        """Convert single markdown file to Q&A dataset"""
        
        logger.info(f"Processing single file: {input_file}")
        
        # Extract content chunks
        content_chunks = self.processor.extract_content_from_markdown(input_file)
        logger.info(f"Extracted {len(content_chunks)} content chunks")
        
        all_qa_pairs = []
        
        # Generate Q&A pairs for each chunk
        for chunk in content_chunks:
            category = input_file.stem
            qa_pairs = await self.generator.generate_qa_pairs(chunk, category)
            all_qa_pairs.extend(qa_pairs)
            
            # Small delay
            await asyncio.sleep(0.1)
        
        logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs")
        
        # Create dataset
        dataset = QADataset(
            pairs=all_qa_pairs,
            metadata={
                "created_at": datetime.now().isoformat(),
                "source_file": str(input_file),
                "total_pairs": len(all_qa_pairs),
                "model_used": self.generator.model.model_name if hasattr(self.generator.model, 'model_name') else "unknown"
            }
        )
        
        # Format for Unsloth
        formatted_data = self.formatter.format_for_unsloth(all_qa_pairs, dataset_format)
        
        # Save dataset
        self.formatter.save_dataset(formatted_data, output_path, file_format)
        logger.info(f"Saved dataset to {output_path}")
        
        return dataset


# Example usage and CLI interface
async def main():
    """Main function with example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert markdown documents to Q&A fine-tuning dataset")
    parser.add_argument("input", type=str, help="Input markdown file or directory")
    parser.add_argument("output", type=str, help="Output dataset file path")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model name")
    parser.add_argument("--format", type=str, choices=["alpaca", "chatml", "sharegpt"], 
                       default="alpaca", help="Dataset format")
    parser.add_argument("--file-format", type=str, choices=["json", "jsonl", "csv"], 
                       default="jsonl", help="Output file format")
    parser.add_argument("--max-files", type=int, help="Maximum number of files to process")
    
    args = parser.parse_args()
    
    converter = MarkdownToQAConverter(args.model)
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_dir():
        dataset = await converter.convert_directory(
            input_path, 
            output_path,
            args.format,
            args.file_format,
            args.max_files
        )
    else:
        dataset = await converter.convert_single_file(
            input_path, 
            output_path,
            args.format,
            args.file_format
        )
    
    print(f"✅ Successfully created dataset with {len(dataset.pairs)} Q&A pairs")
    print(f"💾 Saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())