"""
This script converts unstructured Markdown documents into a structured
Question & Answer dataset for fine-tuning LLMs with Unsloth.

It uses LangChain with an Ollama model and Pydantic for structured
data extraction, and outputs a JSONL file in the Alpaca format.
All settings are managed through the 'config.yaml' file.

Usage:
1. Ensure Ollama is running and you have a model like 'llama3.2' pulled.
   `ollama pull llama3.2`

2. Configure your settings in the 'config.yaml' file.

3. Place your Markdown files in the specified input directory.

4. Run the script:
   `python create_dataset.py`
"""
import os
import glob
import json
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable

# --- Pydantic Models for Structured Output ---
# These models define the exact structure we want the LLM to generate.
# LangChain's .with_structured_output() will use these.

class QuestionAnswer(BaseModel):
    """A single question and its corresponding answer."""
    question: str = Field(..., description="A specific, clear question derived from the text.")
    answer: str = Field(..., description="A detailed, comprehensive answer to the question based *only* on the provided text.")

class QADocument(BaseModel):
    """A collection of question-answer pairs extracted from a document."""
    qa_pairs: List[QuestionAnswer] = Field(..., description="A list of question and answer pairs from the document.")

# --- Helper Functions ---

def load_config(config_path: str = 'config.yaml') -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please create a 'config.yaml' file based on the example in the README.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)


# --- Core Functions ---

def generate_qa_from_markdown(structured_client: Runnable, content: str) -> Optional[QADocument]:
    """
    Uses a structured-output-enabled LangChain Ollama model to generate Q&A pairs.

    Args:
        structured_client: The LangChain client with structured output enabled.
        content: The Markdown content as a string.

    Returns:
        A QADocument object containing the generated Q&A pairs, or None on failure.
    """
    system_prompt = (
        "You are an expert data scientist creating a fine-tuning dataset. "
        "Your task is to analyze the provided Markdown document and generate multiple, "
        "high-quality question-and-answer pairs that cover the key information within the text. "
        "The answers must be grounded in the provided text only. Do not invent information."
    )
    
    user_prompt = f"Here is the Markdown document:\n\n---\n\n{content}\n\n---"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    try:
        print("   Generating Q&A pairs... (This might take a moment)")
        # The .invoke method is called on the client that already knows the output structure.
        response = structured_client.invoke(messages)
        print(f"   Successfully generated {len(response.qa_pairs)} pairs.")
        return response
    except Exception as e:
        print(f"   An error occurred while generating Q&A: {e}")
        return None

def main():
    """Main function to run the data generation process."""
    config = load_config()
    
    ollama_config = config.get('ollama_settings', {})
    paths_config = config.get('file_paths', {})

    model_name = ollama_config.get('model', 'llama3.2')
    ollama_host = ollama_config.get('host', 'http://127.0.0.1:11434')
    input_dir = paths_config.get('input_directory', './data/raw')
    output_file = paths_config.get('output_file', 'dataset.jsonl')

    print(f"Starting dataset generation with model '{model_name}'...")

    # 1. Instantiate the LangChain model and bind it to the Pydantic schema.
    
    llm = ChatOllama(model=model_name, base_url=ollama_host)
    structured_llm = llm.with_structured_output(QADocument)
    # 2. Find all Markdown files
    markdown_files = glob.glob(os.path.join(input_dir, "*.md"))
    if not markdown_files:
        print(f"Error: No Markdown (.md) files found in '{input_dir}'.")
        return

    print(f"Found {len(markdown_files)} Markdown files to process.")

    # 3. Process each file and write to JSONL
    total_pairs = 0
    with open(output_file, "w") as f:
        for file_path in markdown_files:
            print(f"\nProcessing file: {os.path.basename(file_path)}")
            with open(file_path, "r", encoding="utf-8") as md_file:
                content = md_file.read()

            if not content.strip():
                print("   Skipping empty file.")
                continue

            # Pass the structured LLM client to the generation function
            qa_document = generate_qa_from_markdown(structured_llm, content)
            
            if qa_document and qa_document.qa_pairs:
                # Convert to Alpaca format for Unsloth compatibility
                for pair in qa_document.qa_pairs:
                    alpaca_record = {
                        "instruction": pair.question,
                        "input": "", # Input is empty as the context is self-contained in the instruction/output
                        "output": pair.answer
                    }
                    f.write(json.dumps(alpaca_record) + "\n")
                total_pairs += len(qa_document.qa_pairs)

    print("\n-------------------------------------------------")
    print("Dataset generation complete!")
    print(f"Total question-answer pairs generated: {total_pairs}")
    print(f"Output saved to: {output_file}")
    print("-------------------------------------------------")
    print("\nNext step: Use this `.jsonl` file to fine-tune your model with Unsloth!")

if __name__ == "__main__":
    main()

