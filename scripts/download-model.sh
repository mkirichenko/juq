#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-model}"
mkdir -p "$MODEL_DIR"

echo "Downloading all-MiniLM-L6-v2 ONNX model..."

# Tokenizer
if [ ! -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "  Downloading tokenizer.json..."
    curl -L -o "$MODEL_DIR/tokenizer.json" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
else
    echo "  tokenizer.json already exists, skipping."
fi

# ONNX model
if [ ! -f "$MODEL_DIR/model.onnx" ]; then
    echo "  Downloading model.onnx (~90MB)..."
    curl -L -o "$MODEL_DIR/model.onnx" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
else
    echo "  model.onnx already exists, skipping."
fi

echo "Done. Model files are in $MODEL_DIR/"
ls -lh "$MODEL_DIR"
