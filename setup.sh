#!/bin/bash

# Install system-level dependencies
echo "Installing system dependencies..."
sudo apt update && sudo apt install -y libgl1

# Create and activate virtual environment
echo "Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
