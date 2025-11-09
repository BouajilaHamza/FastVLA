#!/bin/bash

# Exit on error
set -e

# Create and activate conda environment
conda create -n fastvla python=3.10 -y
conda activate fastvla

# Install PyTorch with CUDA 11.8 (compatible with T4 GPUs)
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install requirements
pip install -r requirements.txt

# Install flash-attn (requires specific build)
pip install flash-attn --no-build-isolation

# Install xformers with CUDA support
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118

# Install pre-commit hooks
pre-commit install

echo "\nEnvironment setup complete! Activate with:"
echo "conda activate fastvla"
