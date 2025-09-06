import argparse
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing.markdown_parser import MarkdownParser
from qa_generation.qa_generator import QAGenerator
from utils.config_loader import ConfigLoader
from qa_generation.prompt_templates import PromptTemplates

def main():
    """
    Main function to generate Q&A pairs from Markdown files.
    """
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from Markdown files.")
    parser.add_argument("--config", type=str, required=True, help="Path to the data preparation config file.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input directory with Markdown files.")
    parser.add_argument("--output", type=str, required=True, help="Path to the output directory for the generated Q&A pairs.")
    args = parser.parse_args()

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load_config(args.config)

    # Load prompts
    prompts = PromptTemplates(prompts_file='configs/prompts.yaml')

    # Initialize components
    md_parser = MarkdownParser()
    qa_generator = QAGenerator(config, prompts)

    # Check if model is available
    if not qa_generator.llm_client.has_model(config['llm_model']):
        print(f"Model '{config['llm_model']}' not found in Ollama. Please pull the model first.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Process each file
    for filename in os.listdir(args.input):
        if filename.endswith(".md"):
            input_path = os.path.join(args.input, filename)
            print(f"Processing {input_path}...")

            # Parse markdown
            chunks = md_parser.parse(input_path)

            # Generate Q&A pairs
            qa_pairs = qa_generator.generate_qa_pairs(chunks)

            # Save Q&A pairs
            output_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}.jsonl")
            with open(output_path, 'w') as f:
                for pair in qa_pairs:
                    f.write(f"{json.dumps(pair)}\n")
            print(f"Saved Q&A pairs to {output_path}")

if __name__ == "__main__":
    main()
