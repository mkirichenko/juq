#include "gguf.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define GGUF_MAGIC 0x46475547  /* "GGUF" */
#define ALIGNMENT  32

/* ---- f16 conversion ---- */

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t bits = sign;
            float f;
            memcpy(&f, &bits, 4);
            return f;
        }
        /* Denormalized */
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | ((uint32_t)mant << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }

    exp = exp + (127 - 15);
    uint32_t bits = sign | (exp << 23) | ((uint32_t)mant << 13);
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* ---- Dequantization kernels ---- */

void dequantize_q4_0(const void *src, float *dst, size_t n) {
    const block_q4_0 *blocks = (const block_q4_0 *)src;
    size_t nb = n / GGML_QK;
    for (size_t i = 0; i < nb; i++) {
        float d = f16_to_f32(blocks[i].d);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blocks[i].qs[j];
            dst[i * GGML_QK + j]      = ((int)(byte & 0xF) - 8) * d;
            dst[i * GGML_QK + j + 16] = ((int)(byte >> 4) - 8) * d;
        }
    }
}

void dequantize_q4_1(const void *src, float *dst, size_t n) {
    const block_q4_1 *blocks = (const block_q4_1 *)src;
    size_t nb = n / GGML_QK;
    for (size_t i = 0; i < nb; i++) {
        float d = f16_to_f32(blocks[i].d);
        float m = f16_to_f32(blocks[i].m);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blocks[i].qs[j];
            dst[i * GGML_QK + j]      = (byte & 0xF) * d + m;
            dst[i * GGML_QK + j + 16] = (byte >> 4) * d + m;
        }
    }
}

void dequantize_q5_0(const void *src, float *dst, size_t n) {
    const block_q5_0 *blocks = (const block_q5_0 *)src;
    size_t nb = n / GGML_QK;
    for (size_t i = 0; i < nb; i++) {
        float d = f16_to_f32(blocks[i].d);
        uint32_t qh;
        memcpy(&qh, blocks[i].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blocks[i].qs[j];
            int lo = (byte & 0xF) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)  | (((qh >> (j + 16)) & 1) << 4);
            dst[i * GGML_QK + j]      = (lo - 16) * d;
            dst[i * GGML_QK + j + 16] = (hi - 16) * d;
        }
    }
}

void dequantize_q5_1(const void *src, float *dst, size_t n) {
    const block_q5_1 *blocks = (const block_q5_1 *)src;
    size_t nb = n / GGML_QK;
    for (size_t i = 0; i < nb; i++) {
        float d = f16_to_f32(blocks[i].d);
        float m = f16_to_f32(blocks[i].m);
        uint32_t qh;
        memcpy(&qh, blocks[i].qh, 4);
        for (int j = 0; j < 16; j++) {
            uint8_t byte = blocks[i].qs[j];
            int lo = (byte & 0xF) | (((qh >> j) & 1) << 4);
            int hi = (byte >> 4)  | (((qh >> (j + 16)) & 1) << 4);
            dst[i * GGML_QK + j]      = lo * d + m;
            dst[i * GGML_QK + j + 16] = hi * d + m;
        }
    }
}

void dequantize_q8_0(const void *src, float *dst, size_t n) {
    const block_q8_0 *blocks = (const block_q8_0 *)src;
    size_t nb = n / GGML_QK;
    for (size_t i = 0; i < nb; i++) {
        float d = f16_to_f32(blocks[i].d);
        for (int j = 0; j < GGML_QK; j++) {
            dst[i * GGML_QK + j] = blocks[i].qs[j] * d;
        }
    }
}

void dequantize_f16(const void *src, float *dst, size_t n) {
    const uint16_t *s = (const uint16_t *)src;
    for (size_t i = 0; i < n; i++) {
        dst[i] = f16_to_f32(s[i]);
    }
}

/* ---- GGUF Reader ---- */

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t size;
} reader_t;

static uint8_t read_u8(reader_t *r) {
    uint8_t v = r->data[r->pos];
    r->pos += 1;
    return v;
}

static uint16_t read_u16(reader_t *r) {
    uint16_t v;
    memcpy(&v, r->data + r->pos, 2);
    r->pos += 2;
    return v;
}

static uint32_t read_u32(reader_t *r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(reader_t *r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static double read_f64(reader_t *r) {
    double v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static char *read_string(reader_t *r) {
    uint64_t len = read_u64(r);
    char *s = malloc(len + 1);
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

static void skip_value(reader_t *r, gguf_value_type_t type);

static void read_kv_value(reader_t *r, gguf_kv_t *kv) {
    switch (kv->type) {
        case GGUF_TYPE_UINT8:   kv->u8 = read_u8(r); break;
        case GGUF_TYPE_INT8:    kv->i8 = (int8_t)read_u8(r); break;
        case GGUF_TYPE_UINT16:  kv->u16 = read_u16(r); break;
        case GGUF_TYPE_INT16:   kv->i16 = (int16_t)read_u16(r); break;
        case GGUF_TYPE_UINT32:  kv->u32 = read_u32(r); break;
        case GGUF_TYPE_INT32:   kv->i32 = (int32_t)read_u32(r); break;
        case GGUF_TYPE_FLOAT32: kv->f32 = read_f32(r); break;
        case GGUF_TYPE_BOOL:    kv->boolean = read_u8(r); break;
        case GGUF_TYPE_STRING:  kv->string = read_string(r); break;
        case GGUF_TYPE_UINT64:  kv->u64 = read_u64(r); break;
        case GGUF_TYPE_INT64:   kv->i64 = (int64_t)read_u64(r); break;
        case GGUF_TYPE_FLOAT64: kv->f64 = read_f64(r); break;
        case GGUF_TYPE_ARRAY: {
            /* For arrays, skip all elements (we don't store them) */
            uint32_t elem_type = read_u32(r);
            uint64_t count = read_u64(r);
            for (uint64_t i = 0; i < count; i++) {
                skip_value(r, (gguf_value_type_t)elem_type);
            }
            break;
        }
    }
}

static void skip_value(reader_t *r, gguf_value_type_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: r->pos += 1; break;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: r->pos += 2; break;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: r->pos += 4; break;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: r->pos += 8; break;
        case GGUF_TYPE_STRING: { uint64_t len = read_u64(r); r->pos += len; break; }
        case GGUF_TYPE_ARRAY: {
            uint32_t et = read_u32(r);
            uint64_t count = read_u64(r);
            for (uint64_t i = 0; i < count; i++) skip_value(r, (gguf_value_type_t)et);
            break;
        }
    }
}

gguf_ctx_t *gguf_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }

    fseek(f, 0, SEEK_END);
    size_t fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = malloc(fsize);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, fsize, f) != fsize) { free(buf); fclose(f); return NULL; }
    fclose(f);

    reader_t r = { .data = buf, .pos = 0, .size = fsize };

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
        free(buf);
        return NULL;
    }

    gguf_ctx_t *ctx = calloc(1, sizeof(gguf_ctx_t));
    ctx->file_data = buf;
    ctx->file_size = fsize;
    ctx->version = read_u32(&r);
    ctx->n_tensors = read_u64(&r);
    ctx->n_kv = read_u64(&r);

    /* Read metadata */
    ctx->kv = calloc(ctx->n_kv, sizeof(gguf_kv_t));
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        ctx->kv[i].key = read_string(&r);
        ctx->kv[i].type = (gguf_value_type_t)read_u32(&r);
        read_kv_value(&r, &ctx->kv[i]);
    }

    /* Read tensor infos */
    ctx->tensor_infos = calloc(ctx->n_tensors, sizeof(gguf_tensor_info_t));
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        ctx->tensor_infos[i].name = read_string(&r);
        ctx->tensor_infos[i].n_dims = read_u32(&r);
        for (uint32_t d = 0; d < ctx->tensor_infos[i].n_dims; d++) {
            ctx->tensor_infos[i].dims[d] = read_u64(&r);
        }
        ctx->tensor_infos[i].type = (ggml_type_t)read_u32(&r);
        ctx->tensor_infos[i].offset = read_u64(&r);
    }

    /* Align to ALIGNMENT boundary for tensor data */
    size_t data_offset = r.pos;
    data_offset = (data_offset + ALIGNMENT - 1) & ~(size_t)(ALIGNMENT - 1);
    ctx->data_start = buf + data_offset;

    return ctx;
}

void gguf_free(gguf_ctx_t *ctx) {
    if (!ctx) return;
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        free(ctx->kv[i].key);
        if (ctx->kv[i].type == GGUF_TYPE_STRING) free(ctx->kv[i].string);
    }
    free(ctx->kv);
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        free(ctx->tensor_infos[i].name);
    }
    free(ctx->tensor_infos);
    free(ctx->file_data);
    free(ctx);
}

uint32_t gguf_get_u32(const gguf_ctx_t *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) {
            switch (ctx->kv[i].type) {
                case GGUF_TYPE_UINT32: return ctx->kv[i].u32;
                case GGUF_TYPE_UINT16: return ctx->kv[i].u16;
                case GGUF_TYPE_UINT8:  return ctx->kv[i].u8;
                case GGUF_TYPE_INT32:  return (uint32_t)ctx->kv[i].i32;
                default:
                    fprintf(stderr, "gguf_get_u32: unexpected type for %s\n", key);
                    return 0;
            }
        }
    }
    fprintf(stderr, "gguf_get_u32: key not found: %s\n", key);
    return 0;
}

float gguf_get_f32(const gguf_ctx_t *ctx, const char *key) {
    for (uint64_t i = 0; i < ctx->n_kv; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0) {
            switch (ctx->kv[i].type) {
                case GGUF_TYPE_FLOAT32: return ctx->kv[i].f32;
                case GGUF_TYPE_FLOAT64: return (float)ctx->kv[i].f64;
                default:
                    fprintf(stderr, "gguf_get_f32: unexpected type for %s\n", key);
                    return 0.0f;
            }
        }
    }
    fprintf(stderr, "gguf_get_f32: key not found: %s\n", key);
    return 0.0f;
}

gguf_tensor_t *gguf_get_tensor(const gguf_ctx_t *ctx, const char *name) {
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensor_infos[i].name, name) != 0) continue;

        gguf_tensor_info_t *info = &ctx->tensor_infos[i];
        size_t n_elements = 1;
        for (uint32_t d = 0; d < info->n_dims; d++) {
            n_elements *= info->dims[d];
        }

        gguf_tensor_t *t = calloc(1, sizeof(gguf_tensor_t));
        t->name = strdup(name);
        t->n_dims = info->n_dims;
        for (uint32_t d = 0; d < info->n_dims; d++) {
            t->dims[d] = (int)info->dims[d];
        }
        t->n_elements = n_elements;
        t->data = malloc(n_elements * sizeof(float));

        const uint8_t *src = ctx->data_start + info->offset;

        switch (info->type) {
            case GGML_TYPE_F32:
                memcpy(t->data, src, n_elements * sizeof(float));
                break;
            case GGML_TYPE_F16:
                dequantize_f16(src, t->data, n_elements);
                break;
            case GGML_TYPE_Q4_0:
                dequantize_q4_0(src, t->data, n_elements);
                break;
            case GGML_TYPE_Q4_1:
                dequantize_q4_1(src, t->data, n_elements);
                break;
            case GGML_TYPE_Q5_0:
                dequantize_q5_0(src, t->data, n_elements);
                break;
            case GGML_TYPE_Q5_1:
                dequantize_q5_1(src, t->data, n_elements);
                break;
            case GGML_TYPE_Q8_0:
                dequantize_q8_0(src, t->data, n_elements);
                break;
            default:
                fprintf(stderr, "Unsupported tensor type %d for %s\n", info->type, name);
                free(t->data);
                free(t->name);
                free(t);
                return NULL;
        }
        return t;
    }
    fprintf(stderr, "Tensor not found: %s\n", name);
    return NULL;
}

void gguf_tensor_free(gguf_tensor_t *t) {
    if (!t) return;
    free(t->name);
    free(t->data);
    free(t);
}
