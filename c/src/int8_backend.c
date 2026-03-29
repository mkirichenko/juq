#include "int8_backend.h"
#include "onnx_model.h"
#include "simd_ops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/*
 * Int8 quantized linear layer.
 * Weights are stored as int8 with per-tensor symmetric quantization.
 * Bias is stored as float.
 * Inference: quantize input -> int8 matmul -> dequantize output -> add bias.
 */
typedef struct {
    int8_t *weight;        /* [out_features x in_features] int8 */
    float *bias;           /* [out_features] float */
    float weight_scale;    /* quantization scale for weights */
    int in_features;
    int out_features;
} int8_linear_t;

/* BERT layer with int8 linear projections */
typedef struct {
    int8_linear_t query;
    int8_linear_t key;
    int8_linear_t value;
    int8_linear_t attn_output;
    float *attn_output_norm_gamma;
    float *attn_output_norm_beta;
    int8_linear_t ffn_up;
    int8_linear_t ffn_down;
    float *layer_output_norm_gamma;
    float *layer_output_norm_beta;
    int num_heads;
    int head_dim;
} int8_bert_layer_t;

/* Int8 BERT model */
typedef struct {
    /* Embeddings (kept as float - lookup tables don't benefit from int8) */
    float *word_embeddings;      /* [vocab_size x hidden] */
    float *position_embeddings;  /* [max_pos x hidden] */
    float *token_type_embeddings;/* [2 x hidden] */
    float *emb_norm_gamma;
    float *emb_norm_beta;

    /* Transformer layers */
    int8_bert_layer_t *layers;
    int num_layers;

    /* Config */
    int hidden_size;
    int num_heads;
    int head_dim;
    float layer_norm_eps;
} int8_bert_model_t;

/* ---- Helper: get float tensor data from ONNX ---- */

static float *get_float_tensor(const onnx_model_t *onnx, const char *name, size_t *out_n) {
    const onnx_tensor_t *t = onnx_model_get(onnx, name);
    if (!t) {
        fprintf(stderr, "Tensor not found: %s\n", name);
        return NULL;
    }
    if (out_n) *out_n = t->n_elements;
    return onnx_tensor_to_float(t);
}

/* ---- Helper: load int8 linear layer ---- */

static int8_linear_t load_int8_linear(const onnx_model_t *onnx,
                                       const char *w_name, const char *b_name,
                                       int in_feat, int out_feat) {
    int8_linear_t l = { .in_features = in_feat, .out_features = out_feat };

    const onnx_tensor_t *wt = onnx_model_get(onnx, w_name);
    if (!wt) {
        fprintf(stderr, "Weight tensor not found: %s\n", w_name);
        exit(1);
    }

    size_t n_weights = (size_t)out_feat * in_feat;

    if (wt->data_type == ONNX_INT8) {
        /* Already int8 - use directly */
        l.weight = malloc(n_weights);
        /* ONNX stores as [out, in] row-major, same as we need */
        memcpy(l.weight, wt->data, n_weights);

        /* Look for scale tensor */
        char scale_name[256];
        snprintf(scale_name, sizeof(scale_name), "%s_scale", w_name);
        const onnx_tensor_t *st = onnx_model_get(onnx, scale_name);
        if (st && st->data_type == ONNX_FLOAT) {
            memcpy(&l.weight_scale, st->data, sizeof(float));
        } else {
            l.weight_scale = 1.0f;
        }
    } else {
        /* Float weights - quantize to int8 at load time */
        float *float_w = onnx_tensor_to_float(wt);

        /* ONNX BERT weights are often [in_features, out_features] and need transpose
         * Check dims to determine layout */
        int need_transpose = 0;
        if (wt->n_dims == 2) {
            if (wt->dims[0] == in_feat && wt->dims[1] == out_feat) {
                need_transpose = 1;  /* [in, out] -> need [out, in] */
            }
        }

        float *ordered_w;
        if (need_transpose) {
            ordered_w = malloc(n_weights * sizeof(float));
            for (int i = 0; i < out_feat; i++) {
                for (int j = 0; j < in_feat; j++) {
                    ordered_w[i * in_feat + j] = float_w[j * out_feat + i];
                }
            }
            free(float_w);
        } else {
            ordered_w = float_w;
        }

        l.weight = malloc(n_weights);
        l.weight_scale = vec_quantize_symmetric(ordered_w, l.weight, n_weights);
        free(ordered_w);
    }

    /* Load bias (always float) */
    l.bias = get_float_tensor(onnx, b_name, NULL);
    if (!l.bias) {
        fprintf(stderr, "Bias tensor not found: %s\n", b_name);
        exit(1);
    }

    return l;
}

static void free_int8_linear(int8_linear_t *l) {
    free(l->weight);
    free(l->bias);
}

/* ---- Int8 linear forward ---- */

static float *int8_linear_forward(const int8_linear_t *l, const float *input, int seq_len) {
    int in = l->in_features;
    int out = l->out_features;
    float *output = malloc((size_t)seq_len * out * sizeof(float));

    /* Temp buffer for quantized input */
    int8_t *input_q = malloc(in);
    int32_t *acc = malloc(out * sizeof(int32_t));

    for (int s = 0; s < seq_len; s++) {
        const float *x = input + s * in;
        float *y = output + s * out;

        /* Dynamically quantize this input row to int8 */
        float input_scale = vec_quantize_symmetric(x, input_q, in);

        /* Int8 matmul: acc[i] = sum_j(weight[i,j] * input_q[j]) */
        mat_vec_mul_i8(l->weight, input_q, acc, out, in);

        /* Dequantize and add bias:
         * y[i] = acc[i] * input_scale * weight_scale + bias[i] */
        float combined_scale = input_scale * l->weight_scale;
        for (int i = 0; i < out; i++) {
            y[i] = (float)acc[i] * combined_scale + l->bias[i];
        }
    }

    free(input_q);
    free(acc);
    return output;
}

/* ---- Int8 BERT forward pass ---- */

static float *int8_bert_forward(const int8_bert_model_t *m,
                                 const uint32_t *input_ids,
                                 const uint32_t *type_ids, int seq_len) {
    int H = m->hidden_size;

    /* 1. Embeddings (float) */
    float *embeddings = malloc((size_t)seq_len * H * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        const float *word_emb = m->word_embeddings + (size_t)input_ids[s] * H;
        const float *pos_emb = m->position_embeddings + (size_t)s * H;
        const float *type_emb = m->token_type_embeddings + (size_t)type_ids[s] * H;
        float *out = embeddings + s * H;
        for (int i = 0; i < H; i++) {
            out[i] = word_emb[i] + pos_emb[i] + type_emb[i];
        }
    }

    float *normed = malloc((size_t)seq_len * H * sizeof(float));
    layer_norm(embeddings, m->emb_norm_gamma, m->emb_norm_beta,
               normed, seq_len, H, m->layer_norm_eps);
    free(embeddings);

    float *x = normed;

    /* 2. Transformer layers (int8 linear ops) */
    for (int li = 0; li < m->num_layers; li++) {
        const int8_bert_layer_t *layer = &m->layers[li];
        int heads = layer->num_heads;
        int hd = layer->head_dim;

        /* Self-attention Q, K, V with int8 matmul */
        float *Q = int8_linear_forward(&layer->query, x, seq_len);
        float *K = int8_linear_forward(&layer->key, x, seq_len);
        float *V = int8_linear_forward(&layer->value, x, seq_len);

        /* Attention (float - scores are small tensors) */
        float scale = 1.0f / sqrtf((float)hd);
        float *attn_out = calloc((size_t)seq_len * H, sizeof(float));

        for (int h = 0; h < heads; h++) {
            float *scores = malloc((size_t)seq_len * seq_len * sizeof(float));
            for (int i = 0; i < seq_len; i++) {
                const float *qi = Q + i * H + h * hd;
                for (int j = 0; j < seq_len; j++) {
                    const float *kj = K + j * H + h * hd;
                    scores[i * seq_len + j] = vec_dot(qi, kj, hd) * scale;
                }
            }
            for (int i = 0; i < seq_len; i++) {
                vec_softmax(scores + i * seq_len, seq_len);
            }
            for (int i = 0; i < seq_len; i++) {
                float *out = attn_out + i * H + h * hd;
                for (int j = 0; j < seq_len; j++) {
                    float w = scores[i * seq_len + j];
                    const float *vj = V + j * H + h * hd;
                    for (int d = 0; d < hd; d++) {
                        out[d] += w * vj[d];
                    }
                }
            }
            free(scores);
        }
        free(Q); free(K); free(V);

        /* Attention output projection (int8) */
        float *attn_proj = int8_linear_forward(&layer->attn_output, attn_out, seq_len);
        free(attn_out);

        /* Residual + LayerNorm */
        float *residual1 = malloc((size_t)seq_len * H * sizeof(float));
        for (int i = 0; i < seq_len * H; i++) residual1[i] = x[i] + attn_proj[i];
        free(attn_proj);

        float *normed1 = malloc((size_t)seq_len * H * sizeof(float));
        layer_norm(residual1, layer->attn_output_norm_gamma,
                   layer->attn_output_norm_beta, normed1, seq_len, H, m->layer_norm_eps);
        free(residual1);

        /* FFN (int8) */
        float *ffn_up = int8_linear_forward(&layer->ffn_up, normed1, seq_len);
        int intermediate = layer->ffn_up.out_features;
        vec_gelu(ffn_up, (size_t)seq_len * intermediate);

        float *ffn_down = int8_linear_forward(&layer->ffn_down, ffn_up, seq_len);
        free(ffn_up);

        /* Residual + LayerNorm */
        float *residual2 = malloc((size_t)seq_len * H * sizeof(float));
        for (int i = 0; i < seq_len * H; i++) residual2[i] = normed1[i] + ffn_down[i];
        free(normed1); free(ffn_down);

        float *normed2 = malloc((size_t)seq_len * H * sizeof(float));
        layer_norm(residual2, layer->layer_output_norm_gamma,
                   layer->layer_output_norm_beta, normed2, seq_len, H, m->layer_norm_eps);
        free(residual2);

        free(x);
        x = normed2;
    }
    return x;
}

/* ---- ONNX tensor name patterns ---- */

/*
 * HuggingFace BERT ONNX models use these naming patterns:
 *
 * Variant A (with model prefix):
 *   bert.embeddings.word_embeddings.weight
 *   bert.encoder.layer.{i}.attention.self.query.weight
 *
 * Variant B (without prefix):
 *   embeddings.word_embeddings.weight
 *   encoder.layer.{i}.attention.self.query.weight
 *
 * Variant C (ONNX export flattened):
 *   /embeddings/word_embeddings/Gather
 *   (initializer names may differ)
 *
 * We try multiple patterns and use whichever exists.
 */

/* Detect the model prefix (e.g. "bert." or "" or "model.") */
static const char *detect_prefix(const onnx_model_t *onnx) {
    static const char *prefixes[] = { "", "bert.", "model.", NULL };
    for (int i = 0; prefixes[i]; i++) {
        char name[256];
        snprintf(name, sizeof(name), "%sembeddings.word_embeddings.weight", prefixes[i]);
        if (onnx_model_get(onnx, name)) return prefixes[i];
    }
    /* Try to find any tensor with "word_embeddings" */
    for (int i = 0; i < onnx->n_tensors; i++) {
        if (strstr(onnx->tensors[i].name, "word_embeddings")) {
            /* Extract prefix */
            const char *pos = strstr(onnx->tensors[i].name, "embeddings.word_embeddings");
            if (pos) {
                size_t plen = (size_t)(pos - onnx->tensors[i].name);
                static char detected[64];
                if (plen < sizeof(detected)) {
                    memcpy(detected, onnx->tensors[i].name, plen);
                    detected[plen] = '\0';
                    return detected;
                }
            }
        }
    }
    return "";
}

/* ---- Load int8 BERT from ONNX ---- */

static int8_bert_model_t *int8_bert_load(const onnx_model_t *onnx) {
    int8_bert_model_t *m = calloc(1, sizeof(int8_bert_model_t));

    const char *prefix = detect_prefix(onnx);
    printf("  ONNX tensor prefix: \"%s\"\n", prefix);

    char name_buf[256];
    #define PFX(fmt, ...) (snprintf(name_buf, sizeof(name_buf), "%s" fmt, prefix, ##__VA_ARGS__), name_buf)

    /* Embeddings */
    m->word_embeddings = get_float_tensor(onnx, PFX("embeddings.word_embeddings.weight"), NULL);
    m->position_embeddings = get_float_tensor(onnx, PFX("embeddings.position_embeddings.weight"), NULL);
    m->token_type_embeddings = get_float_tensor(onnx, PFX("embeddings.token_type_embeddings.weight"), NULL);
    m->emb_norm_gamma = get_float_tensor(onnx, PFX("embeddings.LayerNorm.weight"), NULL);
    m->emb_norm_beta = get_float_tensor(onnx, PFX("embeddings.LayerNorm.bias"), NULL);

    if (!m->word_embeddings || !m->position_embeddings || !m->token_type_embeddings ||
        !m->emb_norm_gamma || !m->emb_norm_beta) {
        fprintf(stderr, "Failed to load embedding tensors\n");
        /* List available tensors for debugging */
        fprintf(stderr, "Available tensors:\n");
        for (int i = 0; i < onnx->n_tensors && i < 20; i++) {
            fprintf(stderr, "  [%d] %s (type=%d, dims=", i, onnx->tensors[i].name, onnx->tensors[i].data_type);
            for (int d = 0; d < onnx->tensors[i].n_dims; d++) {
                fprintf(stderr, "%s%ld", d ? "x" : "", (long)onnx->tensors[i].dims[d]);
            }
            fprintf(stderr, ")\n");
        }
        if (onnx->n_tensors > 20) fprintf(stderr, "  ... and %d more\n", onnx->n_tensors - 20);
        exit(1);
    }

    /* Detect hidden_size from word_embeddings dims */
    const onnx_tensor_t *we = onnx_model_get(onnx, PFX("embeddings.word_embeddings.weight"));
    m->hidden_size = (int)we->dims[we->n_dims - 1];

    /* Detect num_layers by counting existing layer tensors */
    m->num_layers = 0;
    for (int i = 0; i < 100; i++) {
        if (onnx_model_get(onnx, PFX("encoder.layer.%d.attention.self.query.weight", i))) {
            m->num_layers = i + 1;
        } else {
            break;
        }
    }

    if (m->num_layers == 0) {
        fprintf(stderr, "No transformer layers found\n");
        exit(1);
    }

    /* Detect num_heads from query weight shape */
    /* BERT query weight is [hidden, hidden], heads = hidden / 64 typically */
    /* We'll try common head dims */
    int head_dims[] = {64, 32, 48, 96, 128};
    m->num_heads = m->hidden_size / 64;  /* default */
    for (int i = 0; i < 5; i++) {
        if (m->hidden_size % head_dims[i] == 0) {
            m->num_heads = m->hidden_size / head_dims[i];
            break;
        }
    }
    m->head_dim = m->hidden_size / m->num_heads;
    m->layer_norm_eps = 1e-12f;

    printf("  Int8 BERT config: layers=%d, hidden=%d, heads=%d, head_dim=%d\n",
           m->num_layers, m->hidden_size, m->num_heads, m->head_dim);

    /* Load transformer layers */
    m->layers = calloc(m->num_layers, sizeof(int8_bert_layer_t));

    for (int i = 0; i < m->num_layers; i++) {
        int8_bert_layer_t *layer = &m->layers[i];
        layer->num_heads = m->num_heads;
        layer->head_dim = m->head_dim;

        int H = m->hidden_size;

        layer->query = load_int8_linear(onnx,
            PFX("encoder.layer.%d.attention.self.query.weight", i),
            PFX("encoder.layer.%d.attention.self.query.bias", i), H, H);

        layer->key = load_int8_linear(onnx,
            PFX("encoder.layer.%d.attention.self.key.weight", i),
            PFX("encoder.layer.%d.attention.self.key.bias", i), H, H);

        layer->value = load_int8_linear(onnx,
            PFX("encoder.layer.%d.attention.self.value.weight", i),
            PFX("encoder.layer.%d.attention.self.value.bias", i), H, H);

        layer->attn_output = load_int8_linear(onnx,
            PFX("encoder.layer.%d.attention.output.dense.weight", i),
            PFX("encoder.layer.%d.attention.output.dense.bias", i), H, H);

        layer->attn_output_norm_gamma = get_float_tensor(onnx,
            PFX("encoder.layer.%d.attention.output.LayerNorm.weight", i), NULL);
        layer->attn_output_norm_beta = get_float_tensor(onnx,
            PFX("encoder.layer.%d.attention.output.LayerNorm.bias", i), NULL);

        /* Detect intermediate size from ffn weight dims */
        const onnx_tensor_t *ffn_w = onnx_model_get(onnx,
            PFX("encoder.layer.%d.intermediate.dense.weight", i));
        int intermediate = ffn_w ? (int)ffn_w->dims[ffn_w->n_dims - 1] : H * 4;
        if (ffn_w && ffn_w->n_dims == 2) {
            /* [in, out] or [out, in] - the larger dim is intermediate */
            int d0 = (int)ffn_w->dims[0], d1 = (int)ffn_w->dims[1];
            intermediate = (d0 > d1) ? d0 : d1;
        }

        layer->ffn_up = load_int8_linear(onnx,
            PFX("encoder.layer.%d.intermediate.dense.weight", i),
            PFX("encoder.layer.%d.intermediate.dense.bias", i), H, intermediate);

        layer->ffn_down = load_int8_linear(onnx,
            PFX("encoder.layer.%d.output.dense.weight", i),
            PFX("encoder.layer.%d.output.dense.bias", i), intermediate, H);

        layer->layer_output_norm_gamma = get_float_tensor(onnx,
            PFX("encoder.layer.%d.output.LayerNorm.weight", i), NULL);
        layer->layer_output_norm_beta = get_float_tensor(onnx,
            PFX("encoder.layer.%d.output.LayerNorm.bias", i), NULL);
    }

    #undef PFX
    return m;
}

static void int8_bert_free(int8_bert_model_t *m) {
    if (!m) return;
    free(m->word_embeddings);
    free(m->position_embeddings);
    free(m->token_type_embeddings);
    free(m->emb_norm_gamma);
    free(m->emb_norm_beta);
    for (int i = 0; i < m->num_layers; i++) {
        int8_bert_layer_t *l = &m->layers[i];
        free_int8_linear(&l->query);
        free_int8_linear(&l->key);
        free_int8_linear(&l->value);
        free_int8_linear(&l->attn_output);
        free(l->attn_output_norm_gamma);
        free(l->attn_output_norm_beta);
        free_int8_linear(&l->ffn_up);
        free_int8_linear(&l->ffn_down);
        free(l->layer_output_norm_gamma);
        free(l->layer_output_norm_beta);
    }
    free(m->layers);
    free(m);
}

/* ---- Mean pooling ---- */

static float *int8_mean_pool(const float *hidden, const uint32_t *mask,
                              int seq_len, int hidden_size) {
    float *pooled = calloc(hidden_size, sizeof(float));
    float count = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        if (mask[s]) {
            const float *row = hidden + s * hidden_size;
            for (int i = 0; i < hidden_size; i++) pooled[i] += row[i];
            count += 1.0f;
        }
    }
    if (count > 0.0f) {
        float inv = 1.0f / count;
        for (int i = 0; i < hidden_size; i++) pooled[i] *= inv;
    }
    return pooled;
}

/* ---- Backend interface ---- */

static float *int8_embed_fn(bert_embedder_t *emb, const char *text) {
    int8_bert_model_t *model = (int8_bert_model_t *)emb->backend_data;
    token_encoding_t *enc = tokenizer_encode(emb->tokenizer, text);
    if (!enc) return NULL;

    float *hidden = int8_bert_forward(model, enc->ids, enc->type_ids, (int)enc->length);
    float *pooled = int8_mean_pool(hidden, enc->attention_mask,
                                    (int)enc->length, emb->dims);
    free(hidden);
    token_encoding_free(enc);

    vec_l2_normalize(pooled, emb->dims);
    return pooled;
}

static void int8_free_fn(bert_embedder_t *emb) {
    int8_bert_free((int8_bert_model_t *)emb->backend_data);
}

/* ---- Public loader ---- */

bert_embedder_t *int8_embedder_load(const char *model_dir,
                                     const char *model_file,
                                     const char *query_prefix,
                                     const char *doc_prefix) {
    if (!model_file) model_file = "model.onnx";

    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s", model_dir, model_file);

    printf("  Loading ONNX (int8 backend): %s\n", model_path);

    onnx_model_t *onnx = onnx_model_load(model_path);
    if (!onnx) return NULL;

    int8_bert_model_t *model = int8_bert_load(onnx);
    onnx_model_free(onnx);

    /* Load tokenizer */
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    tokenizer_t *tokenizer = tokenizer_load(tok_path);
    if (!tokenizer) {
        int8_bert_free(model);
        return NULL;
    }

    bert_embedder_t *emb = calloc(1, sizeof(bert_embedder_t));
    emb->backend_data = model;
    emb->tokenizer = tokenizer;
    emb->query_prefix = strdup(query_prefix);
    emb->doc_prefix = strdup(doc_prefix);
    emb->dims = model->hidden_size;
    emb->embed_fn = int8_embed_fn;
    emb->free_fn = int8_free_fn;

    return emb;
}
