"""
Example usage of the Markdown to Q&A Dataset Converter
"""

import asyncio
from pathlib import Path
from markdown_to_qa_dataset import MarkdownToQAConverter

async def example_usage():
    """Example of how to use the converter"""
    
    # Initialize converter with your preferred Ollama model
    converter = MarkdownToQAConverter(model_name="llama3.2:3b")
    
    # Example 1: Convert a single markdown file
    print("Converting single file...")
    single_file_dataset = await converter.convert_single_file(
        input_file=Path("docs/README.md"),
        output_path=Path("datasets/readme_qa_dataset.jsonl"),
        dataset_format="alpaca",
        file_format="jsonl"
    )
    
    # Example 2: Convert entire directory
    print("Converting directory...")
    directory_dataset = await converter.convert_directory(
        input_dir=Path("docs/"),
        output_path=Path("datasets/full_qa_dataset.jsonl"),
        dataset_format="alpaca",
        file_format="jsonl",
        max_files=10  # Limit for testing
    )
    
    print(f"Single file dataset: {len(single_file_dataset.pairs)} pairs")
    print(f"Directory dataset: {len(directory_dataset.pairs)} pairs")

# Example of using the dataset with Unsloth
def unsloth_fine_tuning_example():
    """Example of how to use the generated dataset with Unsloth"""
    
    try:
        from unsloth import FastLanguageModel
        import torch
        from datasets import Dataset
        import json
        
        # Load the generated dataset
        with open("datasets/full_qa_dataset.jsonl", 'r') as f:
            qa_data = [json.loads(line) for line in f]
        
        # Convert to Hugging Face dataset format
        dataset = Dataset.from_list(qa_data)
        
        # Initialize Unsloth model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/llama-3.2-3b-bnb-4bit",
            max_seq_length=2048,
            dtype=None,  # Auto detection
            load_in_4bit=True,
        )
        
        # Add LoRA adapters
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
            use_rslora=False,
            loftq_config=None,
        )
        
        # Format the dataset for training
        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                texts.append(text)
            
            return {"text": texts}
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        print("Dataset ready for Unsloth fine-tuning!")
        print(f"Dataset size: {len(dataset)}")
        
        # You can now use this dataset with Unsloth's SFTTrainer
        # trainer = SFTTrainer(...)
        
    except ImportError:
        print("Unsloth not installed. Install with:")
        print("pip install unsloth[cu121]==2024.11.12  # for CUDA 12.1")
        print("or")
        print("pip install unsloth[cu118]==2024.11.12  # for CUDA 11.8")

if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())
    
    # Show Unsloth integration example
    unsloth_fine_tuning_example()