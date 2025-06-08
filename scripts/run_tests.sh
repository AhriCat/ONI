#!/bin/bash
# Test runner script

echo "Running Oni tests..."

# Activate virtual environment if it exists
if [ -d "oni_env" ]; then
    source oni_env/bin/activate
fi

# Run tests
python -m pytest tests/ -v --tb=short

# Run linting
echo "Running code quality checks..."
flake8 modules/ oni_core.py --max-line-length=100 --ignore=E203,W503

echo "Tests complete!"