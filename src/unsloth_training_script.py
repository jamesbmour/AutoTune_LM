"""
Complete Unsloth training script using the generated Q&A dataset
"""

import json
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

def load_qa_dataset(file_path: str):
    """Load the generated Q&A dataset"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    return Dataset.from_list(data)

def setup_model_and_tokenizer(model_name: str = "unsloth/llama-3.2-3b-bnb-4bit"):
    """Setup the model and tokenizer with Unsloth optimizations"""
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,  # None for auto detection
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # Rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    return model, tokenizer

def format_dataset(dataset, tokenizer):
    """Format the dataset for training"""
    
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples["output"]
        texts = []
        
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Alpaca format
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
        
        return {"text": texts}
    
    return dataset.map(formatting_prompts_func, batched=True)

def train_model(dataset_path: str, output_dir: str = "./fine_tuned_model"):
    """Complete training pipeline"""
    
    print("🚀 Starting Unsloth fine-tuning pipeline...")
    
    # 1. Load dataset
    print("📊 Loading Q&A dataset...")
    dataset = load_qa_dataset(dataset_path)
    print(f"Dataset size: {len(dataset)} examples")
    
    # 2. Setup model and tokenizer
    print("🤖 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer()
    
    # 3. Format dataset
    print("📝 Formatting dataset...")
    formatted_dataset = format_dataset(dataset, tokenizer)
    
    # 4. Split dataset
    train_test_split = formatted_dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # 5. Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,  # Adjust based on your dataset size
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=25,
            save_steps=50,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to=None,  # Disable wandb logging
        ),
    )
    
    # 6. Show model info
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"GPU memory before training = {start_gpu_memory} GB.")
    
    # 7. Train the model
    print("🎯 Starting training...")
    trainer_stats = trainer.train()
    
    # 8. Show memory stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    
    # 9. Save the model
    print("💾 Saving fine-tuned model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 10. Optional: Save to Hugging Face format
    print("📤 Saving in Hugging Face format...")
    model.save_pretrained_merged(f"{output_dir}_merged", tokenizer, save_method="merged_16bit")
    
    print("✅ Fine-tuning completed!")
    return model, tokenizer

def test_model(model, tokenizer, test_instruction: str = "What is Python?"):
    """Test the fine-tuned model"""
    
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    
    inputs = tokenizer(
        f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{test_instruction}

### Response:
""", 
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Test Question: {test_instruction}")
    print(f"Model Response:\n{response.split('### Response:')[-1].strip()}")

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train_model("./datasets/qa_training_dataset.jsonl")
    
    # Test the model
    test_model(model, tokenizer, "Explain how to use PydanticAI for structured generation")