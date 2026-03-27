#ifndef ONNX_BACKEND_H
#define ONNX_BACKEND_H

#include "bert.h"

/*
 * ONNX Runtime backend for BERT embedding.
 * Uses the ONNX Runtime C API for int8-quantized (or full-precision) models.
 * Provides the same bert_embedder_t interface as the GGUF backend.
 */

/* Load embedder using ONNX Runtime.
 * model_dir should contain model.onnx (or specified model_file) and tokenizer.json.
 * model_file can be NULL to default to "model.onnx". */
bert_embedder_t *onnx_embedder_load(const char *model_dir,
                                     const char *model_file,
                                     const char *query_prefix,
                                     const char *doc_prefix);

#endif
