#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-model}"

download_file() {
    local dir="$1" file="$2" url="$3"
    if [ ! -f "$dir/$file" ]; then
        echo "  Downloading $file..."
        curl -L -o "$dir/$file" "$url"
    else
        echo "  $file already exists, skipping."
    fi
}

# --- all-MiniLM-L6-v2 ---
echo "=== all-MiniLM-L6-v2 (minilm) ==="
MINILM_DIR="$BASE_DIR/minilm"
mkdir -p "$MINILM_DIR"
download_file "$MINILM_DIR" "tokenizer.json" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
download_file "$MINILM_DIR" "model.onnx" \
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
echo "Done: $(du -sh "$MINILM_DIR" | cut -f1)"
echo

# --- multilingual-e5-small ---
echo "=== multilingual-e5-small (e5-small) ==="
E5_DIR="$BASE_DIR/e5-small"
mkdir -p "$E5_DIR"
download_file "$E5_DIR" "tokenizer.json" \
    "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/tokenizer.json"
download_file "$E5_DIR" "model.onnx" \
    "https://huggingface.co/intfloat/multilingual-e5-small/resolve/main/onnx/model.onnx"
echo "Done: $(du -sh "$E5_DIR" | cut -f1)"
echo

echo "All models downloaded to $BASE_DIR/"
ls -lhR "$BASE_DIR"
