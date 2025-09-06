"""
Training script that uses the generated dataset with Unsloth
Configured based on the same config file
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from config_manager import load_config, Config


class UnslothTrainer:
    """Unsloth training pipeline using configuration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        
        if not UNSLOTH_AVAILABLE:
            raise ImportError("Unsloth not available. Install with: pip install unsloth")
    
    def load_dataset(self) -> Dataset:
        """Load the generated Q&A dataset"""
        dataset_path = Path(self.config.output.path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        # Load based on format
        if dataset_path.suffix == ".jsonl":
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif dataset_path.suffix == ".json":
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
        
        return Dataset.from_list(data)
    
    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer with Unsloth optimizations"""
        unsloth_config = self.config.unsloth
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=unsloth_config.model_name,
            max_seq_length=unsloth_config.training.max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=True,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=unsloth_config.lora.r,
            target_modules=unsloth_config.lora.target_modules,
            lora_alpha=unsloth_config.lora.lora_alpha,
            lora_dropout=unsloth_config.lora.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        return model, tokenizer
    
    def format_dataset_for_training(self, dataset: Dataset, tokenizer) -> Dataset:
        """Format dataset for training based on configuration"""
        
        def formatting_prompts_func(examples):
            texts = []
            format_type = self.config.qa_generation.format
            
            if format_type == "alpaca":
                instructions = examples["instruction"]
                inputs = examples.get("input", [""] * len(instructions))
                outputs = examples["output"]
                
                for instruction, input_text, output in zip(instructions, inputs, outputs):
                    if input_text and input_text.strip():
                        text = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}{tokenizer.eos_token}"""
                    else:
                        text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}{tokenizer.eos_token}"""
                    texts.append(text)
            
            elif format_type == "chatml":
                messages_list = examples["messages"]
                for messages in messages_list:
                    text = ""
                    for message in messages:
                        if message["role"] == "user":
                            text += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
                        elif message["role"] == "assistant":
                            text += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"
                    text += tokenizer.eos_token
                    texts.append(text)
            
            return {"text": texts}
        
        return dataset.map(formatting_prompts_func, batched=True)
    
    def train(self) -> tuple:
        """Complete training pipeline"""
        
        print("🚀 Starting Unsloth fine-tuning pipeline...")
        
        # Load dataset
        print("📊 Loading Q&A dataset...")
        dataset = self.load_dataset()
        print(f"Dataset size: {len(dataset)} examples")
        
        # Setup model
        print("🤖 Setting up model and tokenizer...")
        model, tokenizer = self.setup_model_and_tokenizer()
        
        # Format dataset
        print("📝 Formatting dataset for training...")
        formatted_dataset = self.format_dataset_for_training(dataset, tokenizer)
        
        # Split dataset
        train_test_split = formatted_dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        eval_dataset = train_test_split["test"]
        
        print(f"Training examples: {len(train_dataset)}")
        print(f"Evaluation examples: {len(eval_dataset)}")
        
        # Setup trainer
        training_config = self.config.unsloth.training
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=training_config.max_seq_length,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=training_config.per_device_train_batch_size,
                gradient_accumulation_steps=training_config.gradient_accumulation_steps,
                warmup_steps=training_config.warmup_steps,
                max_steps=training_config.max_steps,
                learning_rate=training_config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=training_config.weight_decay,
                lr_scheduler_type=training_config.lr_scheduler_type,
                seed=3407,
                output_dir=training_config.output_dir,
                evaluation_strategy="steps",
                eval_steps=training_config.eval_steps,
                save_steps=training_config.save_steps,
                save_total_limit=training_config.save_total_limit,
                load_best_model_at_end=True,
                report_to=None,
            ),
        )
        
        # Show GPU info
        if torch.cuda.is_available():
            gpu_stats = torch.cuda.get_device_properties(0)
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
            
            print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
            print(f"GPU memory before training = {start_gpu_memory} GB.")
        
        # Train
        print("🎯 Starting training...")
        trainer_stats = trainer.train()
        
        # Save model
        output_dir = Path(training_config.output_dir)
        print(f"💾 Saving model to {output_dir}...")
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save merged model
        merged_dir = output_dir.parent / f"{output_dir.name}_merged"
        print(f"📤 Saving merged model to {merged_dir}...")
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        
        print("✅ Training completed successfully!")
        
        return model, tokenizer


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model with Unsloth using generated dataset")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    try:
        trainer = UnslothTrainer(args.config)
        model, tokenizer = trainer.train()
        
        # Test the model
        print("\n🧪 Testing trained model...")
        FastLanguageModel.for_inference(model)
        
        test_instruction = "What is Python programming?"
        inputs = tokenizer(
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{test_instruction}

### Response:
""", return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Test Question: {test_instruction}")
        print(f"Model Response:\n{response.split('### Response:')[-1].strip()}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
