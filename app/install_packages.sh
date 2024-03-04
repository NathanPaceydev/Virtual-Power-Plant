#!/bin/bash

# Activate your virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing requests..."
pip install requests

echo "Installing numpy..."
pip install numpy

echo "Installing matplotlib..."
pip install matplotlib

echo "Installing mplcursors..."
pip install mplcursors

echo "All required packages have been installed."
