#include "onnx_model.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/*
 * Minimal protobuf reader for ONNX format.
 * Only parses what we need: ModelProto -> GraphProto -> initializer (TensorProto[]).
 */

/* Protobuf wire types */
#define PB_VARINT  0
#define PB_64BIT   1
#define PB_BYTES   2
#define PB_32BIT   5

typedef struct {
    const uint8_t *data;
    size_t pos;
    size_t end;
} pb_reader_t;

static uint64_t pb_read_varint(pb_reader_t *r) {
    uint64_t val = 0;
    int shift = 0;
    while (r->pos < r->end) {
        uint8_t byte = r->data[r->pos++];
        val |= (uint64_t)(byte & 0x7F) << shift;
        if (!(byte & 0x80)) break;
        shift += 7;
    }
    return val;
}

static uint32_t __attribute__((unused)) pb_read_fixed32(pb_reader_t *r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t __attribute__((unused)) pb_read_fixed64(pb_reader_t *r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static void pb_read_bytes(pb_reader_t *r, const uint8_t **out, size_t *len) {
    *len = (size_t)pb_read_varint(r);
    *out = r->data + r->pos;
    r->pos += *len;
}

static void pb_skip(pb_reader_t *r, int wire_type) {
    switch (wire_type) {
        case PB_VARINT: pb_read_varint(r); break;
        case PB_64BIT:  r->pos += 8; break;
        case PB_32BIT:  r->pos += 4; break;
        case PB_BYTES: {
            size_t len = (size_t)pb_read_varint(r);
            r->pos += len;
            break;
        }
    }
}

/* ---- TensorProto parser ---- */
/* Field numbers from onnx.proto3:
 *   1: dims (repeated int64)
 *   2: data_type (int32)
 *   4: float_data (repeated float, packed)
 *   5: int32_data (repeated int32, packed)
 *   7: int64_data (repeated int64, packed)
 *   8: name (string)
 *   13: raw_data (bytes)
 */

static void parse_tensor_proto(pb_reader_t *r, onnx_model_t *model) {
    onnx_tensor_t t = {0};

    /* Temp storage for dims (grow as needed) */
    int dims_cap = 8;
    t.dims = malloc(dims_cap * sizeof(int64_t));

    /* First pass: parse all fields */
    const uint8_t *raw_data = NULL;
    size_t raw_data_len = 0;
    const uint8_t *float_data = NULL;
    size_t float_data_len = 0;
    const uint8_t *int32_data = NULL;
    size_t int32_data_len = 0;
    const uint8_t *int64_data = NULL;
    size_t int64_data_len = 0;

    while (r->pos < r->end) {
        uint64_t tag = pb_read_varint(r);
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);

        switch (field) {
            case 1: /* dims */
                if (wire == PB_BYTES) {
                    /* packed repeated int64 */
                    size_t len = (size_t)pb_read_varint(r);
                    size_t end = r->pos + len;
                    while (r->pos < end) {
                        if (t.n_dims >= dims_cap) {
                            dims_cap *= 2;
                            t.dims = realloc(t.dims, dims_cap * sizeof(int64_t));
                        }
                        t.dims[t.n_dims++] = (int64_t)pb_read_varint(r);
                    }
                } else {
                    /* individual varint */
                    if (t.n_dims >= dims_cap) {
                        dims_cap *= 2;
                        t.dims = realloc(t.dims, dims_cap * sizeof(int64_t));
                    }
                    t.dims[t.n_dims++] = (int64_t)pb_read_varint(r);
                }
                break;
            case 2: /* data_type */
                t.data_type = (int)pb_read_varint(r);
                break;
            case 4: /* float_data (packed fixed32) */
                if (wire == PB_BYTES) {
                    pb_read_bytes(r, &float_data, &float_data_len);
                } else {
                    pb_skip(r, wire);
                }
                break;
            case 5: /* int32_data (packed varint) */
                if (wire == PB_BYTES) {
                    pb_read_bytes(r, &int32_data, &int32_data_len);
                } else {
                    pb_skip(r, wire);
                }
                break;
            case 7: /* int64_data (packed varint) */
                if (wire == PB_BYTES) {
                    pb_read_bytes(r, &int64_data, &int64_data_len);
                } else {
                    pb_skip(r, wire);
                }
                break;
            case 8: /* name */ {
                size_t len = (size_t)pb_read_varint(r);
                t.name = malloc(len + 1);
                memcpy(t.name, r->data + r->pos, len);
                t.name[len] = '\0';
                r->pos += len;
                break;
            }
            case 13: /* raw_data */
                pb_read_bytes(r, &raw_data, &raw_data_len);
                break;
            default:
                pb_skip(r, wire);
                break;
        }
    }

    /* Compute n_elements */
    t.n_elements = 1;
    for (int i = 0; i < t.n_dims; i++) {
        t.n_elements *= (size_t)t.dims[i];
    }

    /* Copy tensor data */
    if (raw_data && raw_data_len > 0) {
        t.data = malloc(raw_data_len);
        memcpy(t.data, raw_data, raw_data_len);
        t.data_size = raw_data_len;
    } else if (float_data && float_data_len > 0) {
        t.data = malloc(float_data_len);
        memcpy(t.data, float_data, float_data_len);
        t.data_size = float_data_len;
    } else if (int32_data && int32_data_len > 0) {
        /* Packed varint int32 -> need to decode */
        int32_t *buf = malloc(t.n_elements * sizeof(int32_t));
        pb_reader_t dr = { .data = int32_data, .pos = 0, .end = int32_data_len };
        size_t idx = 0;
        while (dr.pos < dr.end && idx < t.n_elements) {
            uint64_t v = pb_read_varint(&dr);
            buf[idx++] = (int32_t)(uint32_t)v;
        }
        t.data = buf;
        t.data_size = idx * sizeof(int32_t);
    } else if (int64_data && int64_data_len > 0) {
        int64_t *buf = malloc(t.n_elements * sizeof(int64_t));
        pb_reader_t dr = { .data = int64_data, .pos = 0, .end = int64_data_len };
        size_t idx = 0;
        while (dr.pos < dr.end && idx < t.n_elements) {
            buf[idx++] = (int64_t)pb_read_varint(&dr);
        }
        t.data = buf;
        t.data_size = idx * sizeof(int64_t);
    }

    /* Only store if we have a name and data */
    if (t.name && t.data) {
        if (model->n_tensors >= model->capacity) {
            model->capacity = model->capacity ? model->capacity * 2 : 128;
            model->tensors = realloc(model->tensors, model->capacity * sizeof(onnx_tensor_t));
        }
        model->tensors[model->n_tensors++] = t;
    } else {
        free(t.name);
        free(t.dims);
        free(t.data);
    }
}

/* ---- GraphProto parser ---- */
/* Field numbers:
 *   1: node (repeated NodeProto) - skip
 *   5: initializer (repeated TensorProto) - parse
 */

static void parse_graph_proto(pb_reader_t *r, onnx_model_t *model) {
    while (r->pos < r->end) {
        uint64_t tag = pb_read_varint(r);
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);

        if (field == 5 && wire == PB_BYTES) {
            /* initializer: TensorProto */
            size_t len = (size_t)pb_read_varint(r);
            pb_reader_t sub = { .data = r->data, .pos = r->pos, .end = r->pos + len };
            r->pos += len;
            parse_tensor_proto(&sub, model);
        } else {
            pb_skip(r, wire);
        }
    }
}

/* ---- ModelProto parser ---- */
/* Field numbers:
 *   7: graph (GraphProto)
 */

static void parse_model_proto(pb_reader_t *r, onnx_model_t *model) {
    while (r->pos < r->end) {
        uint64_t tag = pb_read_varint(r);
        int field = (int)(tag >> 3);
        int wire = (int)(tag & 7);

        if (field == 7 && wire == PB_BYTES) {
            /* graph: GraphProto */
            size_t len = (size_t)pb_read_varint(r);
            pb_reader_t sub = { .data = r->data, .pos = r->pos, .end = r->pos + len };
            r->pos += len;
            parse_graph_proto(&sub, model);
        } else {
            pb_skip(r, wire);
        }
    }
}

/* ---- Public API ---- */

onnx_model_t *onnx_model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open ONNX model: %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    size_t fsize = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t *buf = malloc(fsize);
    if (fread(buf, 1, fsize, f) != fsize) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    onnx_model_t *model = calloc(1, sizeof(onnx_model_t));
    pb_reader_t r = { .data = buf, .pos = 0, .end = fsize };
    parse_model_proto(&r, model);
    free(buf);

    printf("  Parsed ONNX: %d initializer tensors\n", model->n_tensors);
    return model;
}

void onnx_model_free(onnx_model_t *model) {
    if (!model) return;
    for (int i = 0; i < model->n_tensors; i++) {
        free(model->tensors[i].name);
        free(model->tensors[i].dims);
        free(model->tensors[i].data);
    }
    free(model->tensors);
    free(model);
}

const onnx_tensor_t *onnx_model_get(const onnx_model_t *model, const char *name) {
    for (int i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}

float *onnx_tensor_to_float(const onnx_tensor_t *tensor) {
    if (!tensor) return NULL;

    float *out = malloc(tensor->n_elements * sizeof(float));

    switch (tensor->data_type) {
        case ONNX_FLOAT:
            memcpy(out, tensor->data, tensor->n_elements * sizeof(float));
            break;
        case ONNX_INT8: {
            const int8_t *src = (const int8_t *)tensor->data;
            for (size_t i = 0; i < tensor->n_elements; i++) {
                out[i] = (float)src[i];
            }
            break;
        }
        case ONNX_UINT8: {
            const uint8_t *src = (const uint8_t *)tensor->data;
            for (size_t i = 0; i < tensor->n_elements; i++) {
                out[i] = (float)src[i];
            }
            break;
        }
        case ONNX_INT32: {
            const int32_t *src = (const int32_t *)tensor->data;
            for (size_t i = 0; i < tensor->n_elements; i++) {
                out[i] = (float)src[i];
            }
            break;
        }
        case ONNX_FLOAT16: {
            /* Reuse f16_to_f32 from gguf.h */
            extern float f16_to_f32(uint16_t h);
            const uint16_t *src = (const uint16_t *)tensor->data;
            for (size_t i = 0; i < tensor->n_elements; i++) {
                out[i] = f16_to_f32(src[i]);
            }
            break;
        }
        default:
            fprintf(stderr, "Unsupported ONNX data type: %d\n", tensor->data_type);
            memset(out, 0, tensor->n_elements * sizeof(float));
            break;
    }
    return out;
}
