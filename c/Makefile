CC ?= gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDFLAGS = -lm

# ONNX Runtime configuration (adjust path as needed)
ONNXRUNTIME_DIR ?= /tmp/onnxruntime-linux-x64-1.21.0
ONNX_CFLAGS = -DUSE_ONNX -I$(ONNXRUNTIME_DIR)/include
ONNX_LDFLAGS = -L$(ONNXRUNTIME_DIR)/lib -lonnxruntime -Wl,-rpath,$(ONNXRUNTIME_DIR)/lib

SRC_DIR = src
BUILD_DIR = build

TARGET_GGUF = $(BUILD_DIR)/juq
TARGET_ONNX = $(BUILD_DIR)/juq-onnx

.PHONY: all onnx clean

# Default: GGUF-only build (no external dependencies)
all: $(TARGET_GGUF)

$(TARGET_GGUF): $(BUILD_DIR)/main.o $(BUILD_DIR)/gguf.o $(BUILD_DIR)/bert.o \
                $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/simd_ops.o $(BUILD_DIR)/cjson.o
	$(CC) $^ -o $@ $(LDFLAGS)

# ONNX build: includes ONNX Runtime backend for int8 quantized models
onnx: $(TARGET_ONNX)

$(TARGET_ONNX): $(BUILD_DIR)/main-onnx.o $(BUILD_DIR)/gguf.o $(BUILD_DIR)/bert.o \
                $(BUILD_DIR)/tokenizer.o $(BUILD_DIR)/simd_ops.o $(BUILD_DIR)/cjson.o \
                $(BUILD_DIR)/onnx_backend.o
	$(CC) $^ -o $@ $(LDFLAGS) $(ONNX_LDFLAGS)

# ONNX-aware main and backend objects
$(BUILD_DIR)/main-onnx.o: $(SRC_DIR)/main.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ONNX_CFLAGS) -c $< -o $@

$(BUILD_DIR)/onnx_backend.o: $(SRC_DIR)/onnx_backend.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(ONNX_CFLAGS) -c $< -o $@

# Generic rule for core objects
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# Header dependencies
$(BUILD_DIR)/main.o: $(SRC_DIR)/bert.h $(SRC_DIR)/cjson.h $(SRC_DIR)/simd_ops.h
$(BUILD_DIR)/main-onnx.o: $(SRC_DIR)/bert.h $(SRC_DIR)/cjson.h $(SRC_DIR)/simd_ops.h $(SRC_DIR)/onnx_backend.h
$(BUILD_DIR)/bert.o: $(SRC_DIR)/bert.h $(SRC_DIR)/gguf.h $(SRC_DIR)/tokenizer.h $(SRC_DIR)/simd_ops.h
$(BUILD_DIR)/gguf.o: $(SRC_DIR)/gguf.h
$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/tokenizer.h $(SRC_DIR)/cjson.h
$(BUILD_DIR)/simd_ops.o: $(SRC_DIR)/simd_ops.h
$(BUILD_DIR)/cjson.o: $(SRC_DIR)/cjson.h
$(BUILD_DIR)/onnx_backend.o: $(SRC_DIR)/onnx_backend.h $(SRC_DIR)/bert.h $(SRC_DIR)/tokenizer.h $(SRC_DIR)/simd_ops.h
