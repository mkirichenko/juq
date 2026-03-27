CC ?= gcc
CFLAGS = -O3 -march=native -Wall -Wextra -std=c11 -D_GNU_SOURCE
LDFLAGS = -lm

# Enable AVX2/FMA if supported (auto-detected via -march=native)
# For explicit control, uncomment:
# CFLAGS += -mavx2 -mfma -mf16c

SRC_DIR = src
BUILD_DIR = build

SOURCES = $(SRC_DIR)/main.c \
          $(SRC_DIR)/gguf.c \
          $(SRC_DIR)/bert.c \
          $(SRC_DIR)/tokenizer.c \
          $(SRC_DIR)/simd_ops.c \
          $(SRC_DIR)/cjson.c

OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))
TARGET = $(BUILD_DIR)/juq

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# Header dependencies
$(BUILD_DIR)/main.o: $(SRC_DIR)/bert.h $(SRC_DIR)/cjson.h $(SRC_DIR)/simd_ops.h
$(BUILD_DIR)/bert.o: $(SRC_DIR)/bert.h $(SRC_DIR)/gguf.h $(SRC_DIR)/tokenizer.h $(SRC_DIR)/simd_ops.h
$(BUILD_DIR)/gguf.o: $(SRC_DIR)/gguf.h
$(BUILD_DIR)/tokenizer.o: $(SRC_DIR)/tokenizer.h $(SRC_DIR)/cjson.h
$(BUILD_DIR)/simd_ops.o: $(SRC_DIR)/simd_ops.h
$(BUILD_DIR)/cjson.o: $(SRC_DIR)/cjson.h
