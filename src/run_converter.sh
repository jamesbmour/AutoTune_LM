#!/bin/bash

# Script to run the markdown to Q&A converter

echo "🚀 Starting Markdown to Q&A Dataset Conversion"

# Make sure Ollama is running
echo "📡 Checking Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama service..."
    ollama serve &
    sleep 5
fi

# Pull the model if not already available
echo "📥 Ensuring model is available..."
ollama pull llama3.2:3b

# Create output directory
mkdir -p datasets

# Convert markdown files to Q&A dataset
echo "🔄 Converting markdown files to Q&A dataset..."
python markdown_to_qa_dataset.py \
    ./docs \
    ./datasets/qa_training_dataset.jsonl \
    --model llama3.2:3b \
    --format alpaca \
    --file-format jsonl \
    --max-files 20

echo "✅ Conversion completed!"
echo "📁 Dataset saved to: ./datasets/qa_training_dataset.jsonl"

# Optional: Show some statistics
echo "📊 Dataset Statistics:"
python -c "
import json
with open('./datasets/qa_training_dataset.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
print(f'Total Q&A pairs: {len(data)}')
print(f'Sample question: {data[0][\"instruction\"] if data else \"No data\"}')
"

echo "🎯 Ready for Unsloth fine-tuning!"