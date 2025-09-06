"""
Configuration management for the Markdown to Q&A Dataset Converter
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
import logging


class InputConfig(BaseModel):
    path: str
    extensions: List[str] = [".md", ".markdown"]
    max_files: Optional[int] = None
    recursive: bool = True


class OutputConfig(BaseModel):
    path: str
    file_format: str = "jsonl"
    backup_existing: bool = True
    
    @validator('file_format')
    def validate_file_format(cls, v):
        if v not in ['json', 'jsonl', 'csv']:
            raise ValueError('file_format must be one of: json, jsonl, csv')
        return v


class ModelConfig(BaseModel):
    name: str = "llama3.2:3b"
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    temperature: float = 0.7
    max_tokens: int = 2048


class ProcessingConfig(BaseModel):
    min_chunk_size: int = 50
    max_chunk_size: int = 4000
    extract_types: List[str] = [
        "headers_with_content", "code_blocks", "lists", "tables", "blockquotes"
    ]
    min_code_block_size: int = 50
    skip_empty: bool = True


class QAPairsRange(BaseModel):
    min: int = 1
    max: int = 5


class QualityFilters(BaseModel):
    min_question_length: int = 10
    min_answer_length: int = 20
    max_question_length: int = 200
    max_answer_length: int = 1000


class QAGenerationConfig(BaseModel):
    format: str = "alpaca"
    pairs_per_chunk: QAPairsRange = QAPairsRange()
    question_types: List[str] = [
        "factual", "conceptual", "procedural", "comparative", "application", "analytical"
    ]
    difficulty_levels: List[str] = ["easy", "medium", "hard"]
    quality_filters: QualityFilters = QualityFilters()
    
    @validator('format')
    def validate_format(cls, v):
        if v not in ['alpaca', 'chatml', 'sharegpt']:
            raise ValueError('format must be one of: alpaca, chatml, sharegpt')
        return v


class PromptsConfig(BaseModel):
    system_prompt: str
    generation_prompt_template: str


class RetryConfig(BaseModel):
    max_attempts: int = 3
    delay: float = 1.0
    exponential_backoff: bool = True


class MemoryConfig(BaseModel):
    batch_size: int = 10
    clear_cache: bool = True


class PerformanceConfig(BaseModel):
    max_concurrent_tasks: int = 3
    api_delay: float = 0.1
    retry: RetryConfig = RetryConfig()
    memory: MemoryConfig = MemoryConfig()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "./logs/converter.log"
    console: bool = True


class ValidationConfig(BaseModel):
    enabled: bool = True
    min_quality_score: float = 0.6
    check_duplicates: bool = True
    duplicate_threshold: float = 0.8


class ExportConfig(BaseModel):
    include_metadata: bool = True
    metadata_fields: List[str] = [
        "source_file", "category", "difficulty", "question_type", "created_at", "model_used"
    ]
    pretty_print: bool = True
    generate_stats: bool = True
    stats_file: str = "./datasets/qa_dataset_stats.json"


class LoRAConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ]


class TrainingConfig(BaseModel):
    max_seq_length: int = 2048
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    output_dir: str = "./fine_tuned_model"
    eval_steps: int = 25
    save_steps: int = 50
    save_total_limit: int = 2


class UnslothConfig(BaseModel):
    model_name: str = "unsloth/llama-3.2-3b-bnb-4bit"
    lora: LoRAConfig = LoRAConfig()
    training: TrainingConfig = TrainingConfig()


class Config(BaseModel):
    input: InputConfig
    output: OutputConfig
    model: ModelConfig = ModelConfig()
    processing: ProcessingConfig = ProcessingConfig()
    qa_generation: QAGenerationConfig = QAGenerationConfig()
    prompts: PromptsConfig
    logging: LoggingConfig = LoggingConfig()
    performance: PerformanceConfig = PerformanceConfig()
    validation: ValidationConfig = ValidationConfig()
    export: ExportConfig = ExportConfig()
    unsloth: UnslothConfig = UnslothConfig()


class ConfigManager:
    """Manages configuration loading and validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self.config: Optional[Config] = None
    
    def load_config(self) -> Config:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            self.config = Config(**config_data)
            return self.config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def save_config(self, config: Config, path: Optional[str] = None) -> None:
        """Save configuration to YAML file"""
        save_path = Path(path) if path else self.config_path
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.dict(), f, default_flow_style=False, indent=2)
    
    def create_default_config(self, path: Optional[str] = None) -> Config:
        """Create a default configuration file"""
        default_config = Config(
            input=InputConfig(path="./docs"),
            output=OutputConfig(path="./datasets/qa_dataset.jsonl"),
            prompts=PromptsConfig(
                system_prompt="You are an expert at creating Q&A pairs for LLM training.",
                generation_prompt_template="Generate Q&A pairs from: {content}"
            )
        )
        
        save_path = Path(path) if path else self.config_path
        self.save_config(default_config, str(save_path))
        return default_config
    
    def get_config(self) -> Config:
        """Get loaded configuration or load if not already loaded"""
        if self.config is None:
            self.config = self.load_config()
        return self.config
    
    def validate_paths(self, config: Config) -> None:
        """Validate that required paths exist"""
        input_path = Path(config.input.path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
        output_path = Path(config.output.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if config.logging.file:
            log_path = Path(config.logging.file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self, config: Config) -> logging.Logger:
        """Setup logging based on configuration"""
        logger = logging.getLogger("markdown_qa_converter")
        logger.setLevel(getattr(logging, config.logging.level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        formatter = logging.Formatter(config.logging.format)
        
        # Console handler
        if config.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if config.logging.file:
            file_handler = logging.FileHandler(config.logging.file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger


# Utility functions for configuration
def load_config(config_path: str = "config.yaml") -> Config:
    """Quick function to load configuration"""
    manager = ConfigManager(config_path)
    return manager.load_config()


def create_sample_config(output_path: str = "config_sample.yaml") -> None:
    """Create a sample configuration file"""
    manager = ConfigManager()
    manager.create_default_config(output_path)
    print(f"Sample configuration created at: {output_path}")


if __name__ == "__main__":
    # Create sample configuration
    create_sample_config()
    
    # Test loading configuration
    try:
        config = load_config("config_sample.yaml")
        print("✅ Configuration loaded successfully")
        print(f"Input path: {config.input.path}")
        print(f"Output path: {config.output.path}")
        print(f"Model: {config.model.name}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")