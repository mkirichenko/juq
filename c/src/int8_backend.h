#ifndef INT8_BACKEND_H
#define INT8_BACKEND_H

#include "bert.h"

/*
 * Int8 quantized BERT inference backend.
 * Loads weights from ONNX model files, quantizes float weights to int8
 * at load time, and runs inference with SIMD-accelerated int8 matmul.
 *
 * This is our own hand-written int8 runtime (not delegated to ONNX Runtime),
 * enabling direct comparison of:
 *   - GGUF quantization (Q4/Q5/Q8) with our BERT engine
 *   - ONNX Runtime's int8 inference
 *   - Our own int8 BERT engine (this backend)
 */

/* Load embedder using int8 quantized BERT engine.
 * model_dir should contain an .onnx file and tokenizer.json.
 * model_file can be NULL to default to "model.onnx". */
bert_embedder_t *int8_embedder_load(const char *model_dir,
                                     const char *model_file,
                                     const char *query_prefix,
                                     const char *doc_prefix);

#endif
