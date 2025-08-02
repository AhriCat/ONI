#!/bin/bash
# Installation script for Oni

set -e

# Determine project root directory and move there so relative paths work
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."

echo "Installing Oni AGI System..."

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv oni_env
source oni_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# Install Oni in development mode
echo "Installing Oni..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p data models logs cache knowledge_base

# Set permissions
chmod +x scripts/*.sh

echo "Installation complete!"
echo "To activate the environment, run: source oni_env/bin/activate"
echo "To start Oni, run: python oni_core.py"