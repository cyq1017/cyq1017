#!/bin/bash
set -e

echo "=== BLIP2 Product Category Recognition - Setup ==="

# HuggingFace mirror for China (uncomment if needed)
# export HF_ENDPOINT=https://hf-mirror.com

# Detect environment: AutoDL/cloud (base conda) vs local
USE_CONDA_ENV=true
if [ -n "$AUTODL_HOME" ] || [ -d "/root/autodl-tmp" ]; then
    echo "[INFO] AutoDL environment detected, using base environment."
    USE_CONDA_ENV=false
fi

if [ "$USE_CONDA_ENV" = true ]; then
    # Local: create dedicated conda env
    if ! conda info --envs | grep -q "blip2-product"; then
        echo "[1/4] Creating conda environment..."
        conda create -n blip2-product python=3.10 -y
    else
        echo "[1/4] Conda environment already exists."
    fi
    RUN_PREFIX="conda run -n blip2-product"
else
    RUN_PREFIX=""
fi

echo "[2/4] Installing dependencies..."
$RUN_PREFIX pip install -r requirements.txt

echo "[3/4] Pre-downloading BLIP2 model (this may take a while)..."
$RUN_PREFIX python -c "
from transformers import Blip2Processor, Blip2ForConditionalGeneration
print('Downloading processor...')
Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
print('Downloading model...')
Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-opt-2.7b')
print('Model download complete.')
"

echo "[4/4] Creating directories..."
mkdir -p data/images embeddings indices results

echo ""
echo "=== Setup complete! ==="
if [ "$USE_CONDA_ENV" = true ]; then
    echo "Activate environment: conda activate blip2-product"
fi
echo "Run full pipeline:    make all"
