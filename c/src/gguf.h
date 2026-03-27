#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stddef.h>

/* GGML quantization types */
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
} ggml_type_t;

/* Block sizes for quantized types */
#define GGML_QK 32  /* Block size for all Q types */

/* Quantization block structures */
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0;                    /* 18 bytes */
typedef struct { uint16_t d; uint16_t m; uint8_t qs[16]; } block_q4_1;        /* 20 bytes */
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } block_q5_0;     /* 22 bytes */
typedef struct { uint16_t d; uint16_t m; uint8_t qh[4]; uint8_t qs[16]; } block_q5_1; /* 24 bytes */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0;                      /* 34 bytes */

/* GGUF metadata value types */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type_t;

/* Tensor info */
typedef struct {
    char *name;
    uint32_t n_dims;
    uint64_t dims[4];
    ggml_type_t type;
    uint64_t offset;   /* offset from data section start */
} gguf_tensor_info_t;

/* Metadata KV pair */
typedef struct {
    char *key;
    gguf_value_type_t type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        uint64_t u64;
        int64_t  i64;
        double   f64;
        int      boolean;
        char    *string;
    };
} gguf_kv_t;

/* Loaded tensor (dequantized to f32) */
typedef struct {
    char *name;
    float *data;
    int n_dims;
    int dims[4];
    size_t n_elements;
} gguf_tensor_t;

/* GGUF file context */
typedef struct {
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    gguf_kv_t *kv;
    gguf_tensor_info_t *tensor_infos;
    uint8_t *data_start;  /* pointer to mmap'd or loaded tensor data */
    uint8_t *file_data;   /* full file buffer */
    size_t file_size;
} gguf_ctx_t;

/* Load and parse GGUF file */
gguf_ctx_t *gguf_load(const char *path);
void gguf_free(gguf_ctx_t *ctx);

/* Metadata access */
uint32_t gguf_get_u32(const gguf_ctx_t *ctx, const char *key);
float gguf_get_f32(const gguf_ctx_t *ctx, const char *key);

/* Dequantize a tensor to f32 */
gguf_tensor_t *gguf_get_tensor(const gguf_ctx_t *ctx, const char *name);
void gguf_tensor_free(gguf_tensor_t *t);

/* Low-level dequantize functions */
void dequantize_q4_0(const void *src, float *dst, size_t n);
void dequantize_q4_1(const void *src, float *dst, size_t n);
void dequantize_q5_0(const void *src, float *dst, size_t n);
void dequantize_q5_1(const void *src, float *dst, size_t n);
void dequantize_q8_0(const void *src, float *dst, size_t n);
void dequantize_f16(const void *src, float *dst, size_t n);

/* f16 <-> f32 conversion */
float f16_to_f32(uint16_t h);

#endif
