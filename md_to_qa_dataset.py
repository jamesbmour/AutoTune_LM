
import os
import glob
import json
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
import time

# Optional rich progress bar support
try:
    from rich.progress import (
        Progress,
        TimeElapsedColumn,
        TimeRemainingColumn,
        BarColumn,
        SpinnerColumn,
        MofNCompleteColumn,
        TaskProgressColumn,
        TextColumn,
    )
    from rich.console import Console
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback when rich isn't installed
    _RICH_AVAILABLE = False

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

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration file.

    Priority order for path resolution:
    1. Explicit path argument
    2. AUTO_TUNE_LM_CONFIG environment variable (if provided)
    3. Default 'config.yaml' in project root
    """
    env_override = os.environ.get("AUTO_TUNE_LM_CONFIG")
    if env_override and os.path.exists(env_override):
        config_path = env_override

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_markdown_files(input_dir: str, extensions, exclude_patterns, max_files=None):
    files = []
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            if any(name.endswith(ext) for ext in extensions):
                if name in exclude_patterns:
                    continue
                files.append(os.path.join(root, name))
    files.sort()
    if max_files is not None:
        return files[:max_files]
    return files


def main():
    """Main function to run the data generation process using config.yaml."""
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # Extract settings with safe defaults
    ollama_cfg = config.get("ollama_settings", {})
    file_paths_cfg = config.get("file_paths", {})
    file_proc_cfg = config.get("file_processing", {})

    model_name = ollama_cfg.get("model", "llama3.2")
    host = ollama_cfg.get("host", "http://127.0.0.1:11434")

    input_dir = file_paths_cfg.get("input_directory", "./data/raw")
    output_file = file_paths_cfg.get("output_file", "./data/processed/dataset.jsonl")

    extensions = file_proc_cfg.get("file_extensions", [".md"]) or [".md"]
    exclude_patterns = file_proc_cfg.get("exclude_patterns", [])
    max_files = file_proc_cfg.get("max_files_to_process")

    # Normalize paths relative to repo root (this script's parent directory)
    repo_root = Path(__file__).resolve().parent
    input_dir_path = (repo_root / input_dir).resolve()
    output_file_path = (repo_root / output_file).resolve()
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loaded configuration:")
    print(f"  Model: {model_name}")
    print(f"  Host: {host}")
    print(f"  Input directory: {input_dir_path}")
    print(f"  Output file: {output_file_path}")
    print(f"  Extensions: {extensions}")
    print(f"  Exclude patterns: {exclude_patterns or 'None'}")
    print(f"  Max files: {max_files or 'All'}")

    # Instantiate model
    try:
        llm = ChatOllama(model=model_name, base_url=host)
        structured_llm = llm.with_structured_output(QADocument)
    except Exception as e:
        print("Error: Could not initialize the LangChain Ollama client.")
        print("Please ensure 'langchain_ollama' is installed and check your connection settings.")
        print(f"Details: {e}")
        return

    if not input_dir_path.exists():
        print(f"Error: Input directory does not exist: {input_dir_path}")
        return

    markdown_files = collect_markdown_files(str(input_dir_path), extensions, exclude_patterns, max_files)
    if not markdown_files:
        print(f"Error: No Markdown files found in '{input_dir_path}'.")
        return

    print(f"Found {len(markdown_files)} Markdown files to process.")

    total_pairs = 0
    start_time = time.perf_counter()

    # Setup rich progress if available, otherwise fallback to simple prints
    if _RICH_AVAILABLE:
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("[cyan]{task.fields[files_rate]:.2f} files/min"),
            TextColumn("[magenta]{task.fields[pairs_rate]:.1f} pairs/min"),
            console=console,
            transient=False,
        )

        with progress, open(output_file_path, "w", encoding="utf-8") as f:
            task_id = progress.add_task(
                "Processing files", total=len(markdown_files), files_rate=0.0, pairs_rate=0.0
            )
            for idx, file_path in enumerate(markdown_files):
                file_start = time.perf_counter()
                filename = os.path.basename(file_path)
                progress.console.log(f"[bold]Reading:[/bold] {filename}")
                try:
                    with open(file_path, "r", encoding="utf-8") as md_file:
                        content = md_file.read()
                except Exception as e:
                    progress.console.log(f"[red]Skipping {filename} (read error): {e}[/red]")
                    progress.update(task_id, advance=1)
                    continue

                if not content.strip():
                    progress.console.log(f"[yellow]Skipping empty file: {filename}[/yellow]")
                    progress.update(task_id, advance=1)
                    continue

                qa_document = generate_qa_from_markdown(structured_llm, content)
                if qa_document and qa_document.qa_pairs:
                    for pair in qa_document.qa_pairs:
                        alpaca_record = {
                            "instruction": pair.question,
                            "input": "",
                            "output": pair.answer,
                        }
                        f.write(json.dumps(alpaca_record) + "\n")
                    total_pairs += len(qa_document.qa_pairs)
                    progress.console.log(
                        f"Generated {len(qa_document.qa_pairs)} pairs from {filename}"
                    )
                else:
                    progress.console.log(
                        f"[yellow]No pairs generated for {filename}[/yellow]"
                    )

                elapsed = time.perf_counter() - start_time
                files_done = idx + 1
                files_rate = files_done / (elapsed / 60) if elapsed > 0 else 0.0
                pairs_rate = total_pairs / (elapsed / 60) if elapsed > 0 else 0.0
                progress.update(
                    task_id,
                    advance=1,
                    files_rate=files_rate,
                    pairs_rate=pairs_rate,
                    description=f"Processing files ({files_done}/{len(markdown_files)})",
                )
    else:
        print("rich not installed; using simple progress logging. Install 'rich' for a nicer progress bar.")
        with open(output_file_path, "w", encoding="utf-8") as f:
            for idx, file_path in enumerate(markdown_files, start=1):
                filename = os.path.basename(file_path)
                print(f"\n[{idx}/{len(markdown_files)}] Processing file: {filename}")
                try:
                    with open(file_path, "r", encoding="utf-8") as md_file:
                        content = md_file.read()
                except Exception as e:
                    print(f"   Skipping file (read error): {e}")
                    continue

                if not content.strip():
                    print("   Skipping empty file.")
                    continue

                qa_document = generate_qa_from_markdown(structured_llm, content)
                if qa_document and qa_document.qa_pairs:
                    for pair in qa_document.qa_pairs:
                        alpaca_record = {
                            "instruction": pair.question,
                            "input": "",
                            "output": pair.answer,
                        }
                        f.write(json.dumps(alpaca_record) + "\n")
                    total_pairs += len(qa_document.qa_pairs)
                elapsed = time.perf_counter() - start_time
                files_rate = idx / (elapsed / 60) if elapsed > 0 else 0.0
                print(f"   Files/min: {files_rate:.2f} | Total pairs: {total_pairs}")

    print("\n-------------------------------------------------")
    print("Dataset generation complete!")
    print(f"Total question-answer pairs generated: {total_pairs}")
    print(f"Output saved to: {output_file_path}")
    print("-------------------------------------------------")
    print("\nNext step: Use this `.jsonl` file to fine-tune your model with Unsloth!")

if __name__ == "__main__":
    main()

