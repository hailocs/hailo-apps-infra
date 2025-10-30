#!/bin/bash
# Complete setup script for VLM app

set -e  # Exit on error

echo "Starting VLM app setup..."

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "vlm_env" ]; then
    python3 -m venv vlm_env --system-site-packages
fi

# Activate virtual environment
echo "Activating virtual environment..."
source vlm_env/bin/activate

# Install Python requirements
echo "Installing Python requirements..."
pip install -r requirements.txt --quiet

echo "Downloading VLM HEF file (will take a while)..."
curl -O "https://dev-public.hailo.ai/v5.1.0/blob/Qwen2-VL-2B-Instruct.hef"

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source vlm_env/bin/activate"
echo "To run the application, run: python3 app.py --prompts prompt.json"
echo "To stop the application, run: Ctrl+C"
echo "To deactivate the virtual environment, run: deactivate"