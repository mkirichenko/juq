#!/usr/bin/env bash
# Benchmark commands for the Java search engine.
#
# Prereqs:
#   - JDK 25 (project compiles to class file v69). On macOS via brew:
#       brew install openjdk@25
#       export JAVA_HOME=/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home
#   - mvn package + dependency:build-classpath -Dmdep.outputFile=target/cp.txt
#
# Usage:
#   ./scripts/bench.sh build      # mvn package + classpath
#   ./scripts/bench.sh en-onnx    # ONNX backend, EN docs (minilm)
#   ./scripts/bench.sh ru-onnx    # ONNX backend, RU docs (berta multilingual)
#   ./scripts/bench.sh en-pure    # pure-java backend, EN docs (minilm)
#   ./scripts/bench.sh ru-pure    # pure-java backend, RU docs (berta float)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${JAVA_HOME:=/opt/homebrew/opt/openjdk@25/libexec/openjdk.jdk/Contents/Home}"
JAVA="$JAVA_HOME/bin/java"

JAR="target/juq-0.1.0-SNAPSHOT.jar"
CP_FILE="target/cp.txt"

build() {
  mvn -q -DskipTests package
  mvn -q dependency:build-classpath -Dmdep.outputFile="$CP_FILE"
}

run() {
  local backend="$1" model_name="$2" data="$3"
  shift 3
  local CP="$JAR:$(cat "$CP_FILE")"
  "$JAVA" --add-modules=jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
    -cp "$CP" dev.juq.Main \
    --backend "$backend" \
    --model-name "$model_name" \
    --data "$data" \
    --benchmark "$@"
}

case "${1:-}" in
  build)    build ;;
  en-onnx)  run onnx      minilm data/documents.json ;;
  ru-onnx)  run onnx      berta  data/documents-ru.json ;;
  ru-int8)  run onnx      berta  data/documents-ru.json --model-file model_int8.onnx ;;
  en-pure)  run pure-java minilm data/documents.json ;;
  # pure-java uses float model.onnx; we quantize to int8 at load time. The
  # pre-quantized model_int8.onnx uses MatMulInteger + per-row scales that
  # our minimal ONNX loader doesn't fully decode.
  ru-pure)  run pure-java berta  data/documents-ru.json ;;
  ru-pure-tiny)  run pure-java berta  data/documents-tiny-ru.json ;;
  *)
    echo "Usage: $0 {build|en-onnx|ru-onnx|en-pure|ru-pure}" >&2
    exit 1
    ;;
esac
