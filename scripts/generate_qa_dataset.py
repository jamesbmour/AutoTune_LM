# scripts/generate_qa_dataset.py
"""Main pipeline script to orchestrate Markdown parsing, cleaning, and Q&A generation."""

import argparse
from pathlib import Path
import yaml
from src.data_processing.markdown_parser import parse_markdown
from src.qa_generation.qa_generator import generate_qa_pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate Q&A dataset from Markdown files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input Markdown file or directory.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output JSONL file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/prompts.yaml",
        help="Path to prompts config.",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Parse Markdown into chunks (proof of concept: single file)
    input_path = Path(args.input)
    chunks = parse_markdown(input_path.read_text())

    # Generate Q&A for each chunk
    all_pairs = []
    for chunk in chunks:
        pairs = generate_qa_pairs(chunk, config)
        all_pairs.extend(pairs)

    # Write to JSONL (simple output)
    output_path = Path(args.output)
    with output_path.open("w") as f:
        for pair in all_pairs:
            f.write(f"{pair}\n")

    print(f"Generated {len(all_pairs)} Q&A pairs at {output_path}")


if __name__ == "__main__":
    main()
