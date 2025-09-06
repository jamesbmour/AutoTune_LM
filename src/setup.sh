#!/bin/bash

# Setup script for the Configuration-driven Markdown to Q&A Converter

echo "🚀 Setting up Markdown to Q&A Dataset Converter"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p {docs,datasets,logs,configs}

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Start Ollama service if not running
echo "🔄 Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 5
fi

# Pull default model
echo "📥 Pulling default model..."
ollama pull llama3.2:3b

# Create sample configuration
echo "⚙️  Creating sample configuration..."
python config_manager.py

# Create sample markdown files for testing
echo "📝 Creating sample markdown files..."
cat > docs/sample1.md << 'EOF'
# Python Programming Guide

Python is a high-level, interpreted programming language known for its simplicity and readability.

## Variables and Data Types

In Python, variables are created when you assign a value to them:

```python
name = "Alice"
age = 30
is_student = False