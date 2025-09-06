"""
Configuration-driven Markdown to Q&A Dataset Converter
Refactored version that runs entirely off a config file
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import time
import hashlib
from dataclasses import asdict

# Core dependencies
from pydantic import BaseModel, Field
from pydantic_ai import Agent
# from pydantic_ai.models import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

import markdown
from bs4 import BeautifulSoup
import pandas as pd
import yaml
from tqdm.asyncio import tqdm

# Local imports
from config_manager import Config, ConfigManager, load_config


class QAPair(BaseModel):
    """Structured Q&A pair model"""
    question: str = Field(description="A clear, specific question about the content")
    answer: str = Field(description="A comprehensive answer based on the content")
    context: str = Field(description="The source context from which the Q&A was derived")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")
    category: str = Field(description="Content category or topic area")
    question_type: str = Field(description="Type of question: factual, conceptual, etc.")
    source_file: str = Field(description="Source file path")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class QADataset(BaseModel):
    """Collection of Q&A pairs with metadata"""
    pairs: List[QAPair] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConfigurableMarkdownProcessor:
    """Processes markdown files based on configuration"""
    
    def __init__(self, config: Config):
        self.config = config
        self.md = markdown.Markdown(extensions=['extra', 'codehilite', 'tables'])
    
    def extract_content_from_markdown(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract content chunks from markdown file based on configuration"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            html = self.md.convert(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            chunks = []
            extract_types = self.config.processing.extract_types
            
            # Extract headers with content
            if "headers_with_content" in extract_types:
                chunks.extend(self._extract_headers_with_content(soup))
            
            # Extract code blocks
            if "code_blocks" in extract_types:
                chunks.extend(self._extract_code_blocks(soup))
            
            # Extract lists
            if "lists" in extract_types:
                chunks.extend(self._extract_lists(soup))
            
            # Extract tables
            if "tables" in extract_types:
                chunks.extend(self._extract_tables(soup))
            
            # Extract blockquotes
            if "blockquotes" in extract_types:
                chunks.extend(self._extract_blockquotes(soup))
            
            # Filter chunks by size
            filtered_chunks = []
            for chunk in chunks:
                content_len = len(chunk['content'])
                if (self.config.processing.min_chunk_size <= content_len <= 
                    self.config.processing.max_chunk_size):
                    filtered_chunks.append(chunk)
            
            print(f"Extracted {len(filtered_chunks)} valid chunks from {file_path}")
            return filtered_chunks
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    def _extract_headers_with_content(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract headers with their following content"""
        chunks = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            chunk_content = heading.get_text().strip()
            
            current = heading.next_sibling
            content_parts = [chunk_content]
            
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
                chunks.append({
                    'content': ' '.join(content_parts),
                    'type': 'header_section',
                    'level': heading.name
                })
        
        return chunks
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract code blocks"""
        chunks = []
        for code_block in soup.find_all(['code', 'pre']):
            code_text = code_block.get_text().strip()
            if len(code_text) >= self.config.processing.min_code_block_size:
                chunks.append({
                    'content': f"Code example: {code_text}",
                    'type': 'code_block',
                    'language': code_block.get('class', [''])[0] if code_block.get('class') else ''
                })
        
        return chunks
    
    def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract list content"""
        chunks = []
        for list_elem in soup.find_all(['ul', 'ol']):
            list_text = list_elem.get_text().strip()
            if len(list_text) > 100:
                chunks.append({
                    'content': list_text,
                    'type': 'list',
                    'list_type': 'ordered' if list_elem.name == 'ol' else 'unordered'
                })
        
        return chunks
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract table content"""
        chunks = []
        for table in soup.find_all('table'):
            table_text = table.get_text().strip()
            if len(table_text) > 50:
                chunks.append({
                    'content': f"Table data: {table_text}",
                    'type': 'table'
                })
        
        return chunks
    
    def _extract_blockquotes(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract blockquote content"""
        chunks = []
        for quote in soup.find_all('blockquote'):
            quote_text = quote.get_text().strip()
            if len(quote_text) > 50:
                chunks.append({
                    'content': f"Quote: {quote_text}",
                    'type': 'blockquote'
                })
        
        return chunks


class ConfigurableQAGenerator:
    """Generates Q&A pairs using configuration-driven prompts"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = OllamaProvider(
            base_url=config.model.base_url,
        )
        
        self.agent = Agent(
            model=self.model,
            result_type=List[QAPair],
            system_prompt=config.prompts.system_prompt
        )
        
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_pairs_generated': 0
        }
    
    async def generate_qa_pairs(
        self, 
        content_chunk: Dict[str, Any], 
        source_file: Path,
        category: str = "general"
    ) -> List[QAPair]:
        """Generate Q&A pairs from a content chunk with retry logic"""
        
        for attempt in range(self.config.performance.retry.max_attempts):
            try:
                self.stats['total_requests'] += 1
                
                # Build dynamic prompt from template
                prompt = self._build_generation_prompt(content_chunk, category)
                
                # Generate Q&A pairs
                result = await self.agent.run(prompt)
                
                # Post-process and validate pairs
                validated_pairs = self._validate_and_enhance_pairs(
                    result.data, content_chunk, source_file, category
                )
                
                self.stats['successful_requests'] += 1
                self.stats['total_pairs_generated'] += len(validated_pairs)
                
                print(f"Generated {len(validated_pairs)} pairs for {source_file}")
                return validated_pairs
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {source_file}: {e}")
                if attempt < self.config.performance.retry.max_attempts - 1:
                    delay = (self.config.performance.retry.delay * 
                            (2 ** attempt if self.config.performance.retry.exponential_backoff else 1))
                    await asyncio.sleep(delay)
                else:
                    self.stats['failed_requests'] += 1
                    print(f"All attempts failed for {source_file}: {e}")
        
        return []
    
    def _build_generation_prompt(self, content_chunk: Dict[str, Any], category: str) -> str:
        """Build generation prompt from template"""
        template = self.config.prompts.generation_prompt_template
        
        return template.format(
            content=content_chunk['content'],
            category=category,
            min_pairs=self.config.qa_generation.pairs_per_chunk.min,
            max_pairs=self.config.qa_generation.pairs_per_chunk.max,
            difficulty_levels=', '.join(self.config.qa_generation.difficulty_levels),
            question_types=', '.join(self.config.qa_generation.question_types),
            min_q_len=self.config.qa_generation.quality_filters.min_question_length,
            max_q_len=self.config.qa_generation.quality_filters.max_question_length,
            min_a_len=self.config.qa_generation.quality_filters.min_answer_length,
            max_a_len=self.config.qa_generation.quality_filters.max_answer_length
        )
    
    def _validate_and_enhance_pairs(
        self,
        pairs: List[QAPair],
        content_chunk: Dict[str, Any],
        source_file: Path,
        category: str
    ) -> List[QAPair]:
        """Validate and enhance generated Q&A pairs"""
        
        validated_pairs = []
        filters = self.config.qa_generation.quality_filters
        
        for pair in pairs:
            # Basic length validation
            if (len(pair.question) < filters.min_question_length or 
                len(pair.question) > filters.max_question_length or
                len(pair.answer) < filters.min_answer_length or
                len(pair.answer) > filters.max_answer_length):
                continue
            
            # Enhance with metadata
            pair.context = content_chunk['content']
            pair.category = category
            pair.source_file = str(source_file)
            pair.created_at = datetime.now().isoformat()
            
            # Assign question type if not set
            if not hasattr(pair, 'question_type') or not pair.question_type:
                pair.question_type = self._infer_question_type(pair.question)
            # Validate difficulty level
            if pair.difficulty not in self.config.qa_generation.difficulty_levels:
                pair.difficulty = "medium"  # Default
            validated_pairs.append(pair)
        
        return validated_pairs
    
    def _infer_question_type(self, question: str) -> str:
        """Infer question type from question text"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'who', 'when', 'where']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'why']):
            return 'procedural' if 'how' in question_lower else 'conceptual'
        elif any(word in question_lower for word in ['compare', 'contrast', 'difference']):
            return 'comparative'
        elif any(word in question_lower for word in ['example', 'demonstrate', 'show']):
            return 'application'
        elif any(word in question_lower for word in ['analyze', 'evaluate', 'assess']):
            return 'analytical'
        else:
            return 'general'
    
    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.stats.copy()


class ConfigurableUnslothFormatter:
    """Formats Q&A pairs for different training formats based on configuration"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def format_for_unsloth(self, qa_pairs: List[QAPair]) -> List[Dict[str, Any]]:
        """Format Q&A pairs for Unsloth training"""
        
        formatted_data = []
        format_type = self.config.qa_generation.format
        
        for pair in qa_pairs:
            if format_type == "alpaca":
                formatted_item = {
                    "instruction": pair.question,
                    "input": "",
                    "output": pair.answer
                }
            elif format_type == "chatml":
                formatted_item = {
                    "messages": [
                        {"role": "user", "content": pair.question},
                        {"role": "assistant", "content": pair.answer}
                    ]
                }
            elif format_type == "sharegpt":
                formatted_item = {
                    "conversations": [
                        {"from": "human", "value": pair.question},
                        {"from": "gpt", "value": pair.answer}
                    ]
                }
            else:
                formatted_item = {
                    "question": pair.question,
                    "answer": pair.answer
                }
            
            # Add metadata if configured
            if self.config.export.include_metadata:
                metadata = {}
                for field in self.config.export.metadata_fields:
                    if hasattr(pair, field):
                        metadata[field] = getattr(pair, field)
                
                if format_type in ["alpaca", "chatml", "sharegpt"]:
                    formatted_item["metadata"] = metadata
                else:
                    formatted_item.update(metadata)
            
            formatted_data.append(formatted_item)
        
        return formatted_data
    
    def save_dataset(self, formatted_data: List[Dict], output_path: Path) -> None:
        """Save formatted dataset based on configuration"""
        
        file_format = self.config.output.file_format
        
        # Create backup if file exists
        if output_path.exists() and self.config.output.backup_existing:
            backup_path = output_path.with_suffix(f'.backup_{int(time.time())}{output_path.suffix}')
            output_path.rename(backup_path)
            print(f"Created backup: {backup_path}")
        
        if file_format == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    formatted_data, 
                    f, 
                    indent=2 if self.config.export.pretty_print else None,
                    ensure_ascii=False
                )
        elif file_format == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in formatted_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif file_format == "csv":
            df = pd.DataFrame(formatted_data)
            df.to_csv(output_path, index=False)
        # Notify user of saved dataset
        print(f"Dataset saved to: {output_path}")


class ConfigurableMarkdownToQAConverter:
    """Main converter class driven by configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        # Validate configuration
        self.config_manager.validate_paths(self.config)
        # Initialize components
        self.processor = ConfigurableMarkdownProcessor(self.config)
        self.generator = ConfigurableQAGenerator(self.config)
        self.formatter = ConfigurableUnslothFormatter(self.config)
        print("Converter initialized with configuration")
    
    async def convert(self) -> QADataset:
        """Main conversion method"""
        
        input_path = Path(self.config.input.path)
        
        if input_path.is_file():
            return await self._convert_single_file(input_path)
        elif input_path.is_dir():
            return await self._convert_directory(input_path)
        else:
            raise ValueError(f"Input path is neither file nor directory: {input_path}")
    
    async def _convert_directory(self, input_dir: Path) -> QADataset:
        """Convert all markdown files in directory"""
        # Find all markdown files
        markdown_files = []
        for ext in self.config.input.extensions:
            if self.config.input.recursive:
                markdown_files.extend(input_dir.glob(f"**/*{ext}"))
            else:
                markdown_files.extend(input_dir.glob(f"*{ext}"))
        # Limit files if configured
        if self.config.input.max_files:
            markdown_files = markdown_files[:self.config.input.max_files]
        print(f"Found {len(markdown_files)} markdown files to process")
        # Process in batches for memory management
        all_qa_pairs = []
        batch_size = self.config.performance.memory.batch_size
        for i in range(0, len(markdown_files), batch_size):
            batch_files = markdown_files[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(markdown_files) + batch_size - 1)//batch_size}")
            batch_pairs = await self._process_file_batch(batch_files)
            all_qa_pairs.extend(batch_pairs)
            # Clear cache if configured
            if self.config.performance.memory.clear_cache:
                import gc
                gc.collect()
        # Create and save dataset
        dataset = await self._create_and_save_dataset(all_qa_pairs, markdown_files)
        return dataset
    
    async def _convert_single_file(self, input_file: Path) -> QADataset:
        """Convert single markdown file"""
        print(f"Processing single file: {input_file}")
        qa_pairs = await self._process_single_file(input_file)
        dataset = await self._create_and_save_dataset(qa_pairs, [input_file])
        return dataset
    
    async def _process_file_batch(self, files: List[Path]) -> List[QAPair]:
        """Process a batch of files with concurrency control"""
        
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_tasks)
        
        async def process_with_semaphore(file_path: Path) -> List[QAPair]:
            async with semaphore:
                return await self._process_single_file(file_path)
        
        # Process files concurrently
        tasks = [process_with_semaphore(file_path) for file_path in files]
        
        if self.config.logging.level == "INFO":
            results = await tqdm.gather(*tasks, desc="Processing files")
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_pairs = []
        for result in results:
            if isinstance(result, list):
                all_pairs.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Error processing file: {result}")
        
        return all_pairs
    
    async def _process_single_file(self, file_path: Path) -> List[QAPair]:
        """Process a single markdown file"""
        
        try:
            # Extract content chunks
            content_chunks = self.processor.extract_content_from_markdown(file_path)
            
            if not content_chunks:
                print(f"No content extracted from {file_path}")
                return []
            
            # Generate Q&A pairs for each chunk
            all_pairs = []
            category = file_path.stem
            
            for chunk in content_chunks:
                pairs = await self.generator.generate_qa_pairs(chunk, file_path, category)
                all_pairs.extend(pairs)
                
                # API delay
                await asyncio.sleep(self.config.performance.api_delay)
            
            return all_pairs
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
    
    async def _create_and_save_dataset(self, qa_pairs: List[QAPair], source_files: List[Path]) -> QADataset:
        """Create dataset and save it"""
        print(f"Generated {len(qa_pairs)} total Q&A pairs")
        # Apply validation if configured
        if self.config.validation.enabled:
            qa_pairs = self._validate_dataset(qa_pairs)
            print(f"After validation: {len(qa_pairs)} Q&A pairs")
        # Create dataset with metadata
        dataset = QADataset(
            pairs=qa_pairs,
            metadata={
                "created_at": datetime.now().isoformat(),
                "source_files": [str(f) for f in source_files],
                "total_pairs": len(qa_pairs),
                "model_used": self.config.model.name,
                "config_checksum": self._get_config_checksum(),
                "generation_stats": self.generator.get_stats()
            }
        )
        # Format and save dataset
        formatted_data = self.formatter.format_for_unsloth(qa_pairs)
        output_path = Path(self.config.output.path)
        self.formatter.save_dataset(formatted_data, output_path)
        # Generate statistics if configured
        if self.config.export.generate_stats:
            await self._generate_statistics(dataset)
        return dataset
    
    def _validate_dataset(self, qa_pairs: List[QAPair]) -> List[QAPair]:
        """Apply validation rules to dataset"""
        
        validated_pairs = []
        
        # Check for duplicates if configured
        if self.config.validation.check_duplicates:
            seen_questions = set()
            for pair in qa_pairs:
                question_hash = hashlib.md5(pair.question.lower().encode()).hexdigest()
                if question_hash not in seen_questions:
                    seen_questions.add(question_hash)
                    validated_pairs.append(pair)
                else:
                    self.logger.debug(f"Removed duplicate question: {pair.question[:50]}...")
        else:
            validated_pairs = qa_pairs
        
        return validated_pairs
    
    def _get_config_checksum(self) -> str:
        """Get checksum of current configuration"""
        config_str = json.dumps(self.config.dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    async def _generate_statistics(self, dataset: QADataset) -> None:
        """Generate and save dataset statistics"""
        
        stats = {
            "total_pairs": len(dataset.pairs),
            "difficulty_distribution": {},
            "question_type_distribution": {},
            "category_distribution": {},
            "source_file_distribution": {},
            "average_question_length": 0,
            "average_answer_length": 0,
            "generation_stats": dataset.metadata.get("generation_stats", {}),
            "created_at": dataset.metadata.get("created_at"),
            "model_used": dataset.metadata.get("model_used")
        }
        
        if dataset.pairs:
            # Calculate distributions
            for pair in dataset.pairs:
                # Difficulty
                stats["difficulty_distribution"][pair.difficulty] = (
                    stats["difficulty_distribution"].get(pair.difficulty, 0) + 1
                )
                
                # Question type
                stats["question_type_distribution"][pair.question_type] = (
                    stats["question_type_distribution"].get(pair.question_type, 0) + 1
                )
                
                # Category
                stats["category_distribution"][pair.category] = (
                    stats["category_distribution"].get(pair.category, 0) + 1
                )
                
                # Source file
                stats["source_file_distribution"][pair.source_file] = (
                    stats["source_file_distribution"].get(pair.source_file, 0) + 1
                )
            
            # Calculate averages
            stats["average_question_length"] = sum(len(pair.question) for pair in dataset.pairs) / len(dataset.pairs)
            stats["average_answer_length"] = sum(len(pair.answer) for pair in dataset.pairs) / len(dataset.pairs)
        
        # Save statistics
        stats_path = Path(self.config.export.stats_file)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Statistics saved to: {stats_path}")


# CLI Interface
async def main():
    """Main CLI function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration-driven Markdown to Q&A Dataset Converter")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--create-config", action="store_true",
                       help="Create a sample configuration file")
    parser.add_argument("--validate-config", action="store_true",
                       help="Validate configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        manager = ConfigManager()
        manager.create_default_config(args.config)
        print(f"✅ Sample configuration created: {args.config}")
        return
    
    if args.validate_config:
        try:
            config = load_config(args.config)
            print("✅ Configuration is valid")
            print(f"Input: {config.input.path}")
            print(f"Output: {config.output.path}")
            print(f"Model: {config.model.name}")
            return
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return
    
    # Run conversion
    try:
        converter = ConfigurableMarkdownToQAConverter(args.config)
        dataset = await converter.convert()
        
        print("\n🎉 Conversion completed successfully!")
        print(f"📊 Generated {len(dataset.pairs)} Q&A pairs")
        print(f"💾 Dataset saved to: {converter.config.output.path}")
        
        # Show some statistics
        if dataset.metadata.get("generation_stats"):
            stats = dataset.metadata["generation_stats"]
            print(f"📈 Success rate: {stats['successful_requests']}/{stats['total_requests']} requests")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)