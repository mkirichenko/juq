#ifndef ONNX_MODEL_H
#define ONNX_MODEL_H

#include <stdint.h>
#include <stddef.h>

/*
 * Minimal ONNX protobuf parser.
 * Extracts initializer tensors (weights) from an ONNX model file.
 */

/* ONNX tensor data types */
#define ONNX_FLOAT    1
#define ONNX_UINT8    2
#define ONNX_INT8     3
#define ONNX_INT32    6
#define ONNX_INT64    7
#define ONNX_FLOAT16  10

/* Parsed tensor from ONNX initializer */
typedef struct {
    char *name;
    int data_type;       /* ONNX_FLOAT, ONNX_INT8, etc. */
    int64_t *dims;
    int n_dims;
    void *data;          /* raw tensor data */
    size_t data_size;    /* size of data in bytes */
    size_t n_elements;   /* total number of elements */
} onnx_tensor_t;

/* Parsed ONNX model */
typedef struct {
    onnx_tensor_t *tensors;
    int n_tensors;
    int capacity;
} onnx_model_t;

/* Parse ONNX file, extracting all initializer tensors */
onnx_model_t *onnx_model_load(const char *path);
void onnx_model_free(onnx_model_t *model);

/* Find tensor by name, returns NULL if not found */
const onnx_tensor_t *onnx_model_get(const onnx_model_t *model, const char *name);

/* Get float data from a tensor (dequantizes int8 if needed using provided scale/zp) */
float *onnx_tensor_to_float(const onnx_tensor_t *tensor);

#endif
