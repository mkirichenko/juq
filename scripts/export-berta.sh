#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-model}/berta"
mkdir -p "$MODEL_DIR"

echo "=== Exporting sergeyzh/BERTA to ONNX ==="

# Check if already exported
if [ -f "$MODEL_DIR/model.onnx" ] && [ -f "$MODEL_DIR/tokenizer.json" ]; then
    echo "BERTA ONNX files already exist in $MODEL_DIR, skipping."
    echo "To re-export, delete $MODEL_DIR/model.onnx first."
    ls -lh "$MODEL_DIR"
    exit 0
fi

# Install dependencies (in a venv to keep things clean)
VENV_DIR="$MODEL_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet transformers torch onnx onnxruntime optimum

echo "Exporting model to ONNX via optimum..."
python3 - "$MODEL_DIR" <<'PYEOF'
import sys
output_dir = sys.argv[1]

from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

print("  Loading model and exporting via optimum...")
model = ORTModelForFeatureExtraction.from_pretrained("sergeyzh/BERTA", export=True)
tokenizer = AutoTokenizer.from_pretrained("sergeyzh/BERTA")

print("  Saving to", output_dir)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("  Export complete.")
PYEOF

# Clean up venv
echo "Cleaning up venv..."
rm -rf "$VENV_DIR"

echo ""
echo "Done. BERTA ONNX files are in $MODEL_DIR/"
ls -lh "$MODEL_DIR"
