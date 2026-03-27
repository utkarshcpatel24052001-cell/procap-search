#!/bin/bash

# Check Python version
PYTHON_VERSION=$(python3 --version)
echo "Current Python version: $PYTHON_VERSION"

# Create a virtual environment
python3 -m venv venv

echo "Virtual environment created. To activate, use: source venv/bin/activate"

echo "Upgrading pip..."
source venv/bin/activate
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! You can now run your project inside the virtual environment."

echo "To deactivate the virtual environment, simply run: deactivate"