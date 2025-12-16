#!/bin/bash

# Quick setup script for the project

set -e

echo "=========================================="
echo "Household Electricity Forecasting - Setup"
echo "=========================================="

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV is not installed. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "Please restart your terminal or run: source ~/.bashrc"
    exit 1
fi

echo "UV is installed"

# Create virtual environment
echo "Creating virtual environment..."
uv venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your household_power_consumption.txt to data/raw/"
echo "2. Run: python scripts/01_clean_and_split.py"
echo "3. Run: python scripts/02_h2o_automl.py"
echo "4. Train models: python scripts/train_*.py"
echo ""
echo "For detailed instructions, see README.md"

