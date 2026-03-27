#include "bert.h"
#include "simd_ops.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <dirent.h>

/* ---- Linear layer (pre-dequantized weights stored in row-major) ---- */

typedef struct {
    float *weight;  /* [out_features x in_features] row-major */
    float *bias;    /* [out_features] */
    int in_features;
    int out_features;
} linear_t;

/* ---- BERT layer ---- */

typedef struct {
    linear_t query;
    linear_t key;
    linear_t value;
    linear_t attn_output;
    float *attn_output_norm_gamma;
    float *attn_output_norm_beta;
    linear_t ffn_up;
    linear_t ffn_down;
    float *layer_output_norm_gamma;
    float *layer_output_norm_beta;
    int num_heads;
    int head_dim;
} bert_layer_t;

struct bert_model {
    /* Embeddings */
    float *word_embeddings;      /* [vocab_size x hidden] */
    float *position_embeddings;  /* [max_pos x hidden] */
    float *token_type_embeddings;/* [2 x hidden] */
    float *emb_norm_gamma;
    float *emb_norm_beta;
    int vocab_size;
    int max_positions;

    /* Transformer layers */
    bert_layer_t *layers;
    int num_layers;

    /* Config */
    int hidden_size;
    int num_heads;
    int head_dim;
    float layer_norm_eps;
};

/* ---- Helpers to load tensors ---- */

static float *load_tensor_data(gguf_ctx_t *ctx, const char *name, size_t *out_n) {
    gguf_tensor_t *t = gguf_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "Failed to load tensor: %s\n", name);
        exit(1);
    }
    float *data = t->data;
    if (out_n) *out_n = t->n_elements;
    t->data = NULL;  /* prevent double-free */
    gguf_tensor_free(t);
    return data;
}

static linear_t load_linear(gguf_ctx_t *ctx, const char *w_name, const char *b_name,
                             int in_feat, int out_feat) {
    linear_t l;
    l.in_features = in_feat;
    l.out_features = out_feat;

    size_t n_w;
    float *raw_weight = load_tensor_data(ctx, w_name, &n_w);

    /* GGUF stores weights as [out_features x in_features] already (row-major) */
    /* But we need transposed for our matmul: input [seq x in] * W^T [in x out] = [seq x out] */
    /* Store as [out x in] for mat_vec_mul / direct access */
    l.weight = raw_weight;

    l.bias = load_tensor_data(ctx, b_name, NULL);
    return l;
}

static void free_linear(linear_t *l) {
    free(l->weight);
    free(l->bias);
}

/* ---- Model loading ---- */

bert_model_t *bert_model_load(gguf_ctx_t *ctx) {
    bert_model_t *m = calloc(1, sizeof(bert_model_t));

    m->num_layers = (int)gguf_get_u32(ctx, "bert.block_count");
    m->hidden_size = (int)gguf_get_u32(ctx, "bert.embedding_length");
    m->num_heads = (int)gguf_get_u32(ctx, "bert.attention.head_count");
    m->layer_norm_eps = gguf_get_f32(ctx, "bert.attention.layer_norm_epsilon");
    m->head_dim = m->hidden_size / m->num_heads;

    printf("  BERT config: layers=%d, hidden=%d, heads=%d, head_dim=%d, eps=%g\n",
           m->num_layers, m->hidden_size, m->num_heads, m->head_dim, m->layer_norm_eps);

    /* Embeddings */
    m->word_embeddings = load_tensor_data(ctx, "token_embd.weight", NULL);
    m->position_embeddings = load_tensor_data(ctx, "position_embd.weight", NULL);
    m->token_type_embeddings = load_tensor_data(ctx, "token_types.weight", NULL);
    m->emb_norm_gamma = load_tensor_data(ctx, "token_embd_norm.weight", NULL);
    m->emb_norm_beta = load_tensor_data(ctx, "token_embd_norm.bias", NULL);

    /* Compute vocab_size and max_positions from tensor dims */
    for (uint64_t i = 0; i < ctx->n_tensors; i++) {
        if (strcmp(ctx->tensor_infos[i].name, "token_embd.weight") == 0) {
            m->vocab_size = (int)ctx->tensor_infos[i].dims[1];
        }
        if (strcmp(ctx->tensor_infos[i].name, "position_embd.weight") == 0) {
            m->max_positions = (int)ctx->tensor_infos[i].dims[1];
        }
    }

    /* Transformer layers */
    m->layers = calloc(m->num_layers, sizeof(bert_layer_t));
    for (int i = 0; i < m->num_layers; i++) {
        char name[128];
        bert_layer_t *layer = &m->layers[i];
        layer->num_heads = m->num_heads;
        layer->head_dim = m->head_dim;

        #define LAYER_TENSOR(fmt) (snprintf(name, sizeof(name), fmt, i), name)

        layer->query = load_linear(ctx,
            LAYER_TENSOR("blk.%d.attn_q.weight"),
            LAYER_TENSOR("blk.%d.attn_q.bias"),
            m->hidden_size, m->hidden_size);

        layer->key = load_linear(ctx,
            LAYER_TENSOR("blk.%d.attn_k.weight"),
            LAYER_TENSOR("blk.%d.attn_k.bias"),
            m->hidden_size, m->hidden_size);

        layer->value = load_linear(ctx,
            LAYER_TENSOR("blk.%d.attn_v.weight"),
            LAYER_TENSOR("blk.%d.attn_v.bias"),
            m->hidden_size, m->hidden_size);

        layer->attn_output = load_linear(ctx,
            LAYER_TENSOR("blk.%d.attn_output.weight"),
            LAYER_TENSOR("blk.%d.attn_output.bias"),
            m->hidden_size, m->hidden_size);

        layer->attn_output_norm_gamma = load_tensor_data(ctx,
            LAYER_TENSOR("blk.%d.attn_output_norm.weight"), NULL);
        layer->attn_output_norm_beta = load_tensor_data(ctx,
            LAYER_TENSOR("blk.%d.attn_output_norm.bias"), NULL);

        /* FFN: up projects hidden -> intermediate, down projects back */
        /* Get intermediate size from the tensor dimensions */
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", i);
        int intermediate_size = m->hidden_size; /* default */
        for (uint64_t t = 0; t < ctx->n_tensors; t++) {
            if (strcmp(ctx->tensor_infos[t].name, name) == 0) {
                intermediate_size = (int)ctx->tensor_infos[t].dims[1];
                break;
            }
        }

        layer->ffn_up = load_linear(ctx,
            LAYER_TENSOR("blk.%d.ffn_up.weight"),
            LAYER_TENSOR("blk.%d.ffn_up.bias"),
            m->hidden_size, intermediate_size);

        layer->ffn_down = load_linear(ctx,
            LAYER_TENSOR("blk.%d.ffn_down.weight"),
            LAYER_TENSOR("blk.%d.ffn_down.bias"),
            intermediate_size, m->hidden_size);

        layer->layer_output_norm_gamma = load_tensor_data(ctx,
            LAYER_TENSOR("blk.%d.layer_output_norm.weight"), NULL);
        layer->layer_output_norm_beta = load_tensor_data(ctx,
            LAYER_TENSOR("blk.%d.layer_output_norm.bias"), NULL);

        #undef LAYER_TENSOR
    }

    return m;
}

void bert_model_free(bert_model_t *m) {
    if (!m) return;
    free(m->word_embeddings);
    free(m->position_embeddings);
    free(m->token_type_embeddings);
    free(m->emb_norm_gamma);
    free(m->emb_norm_beta);
    for (int i = 0; i < m->num_layers; i++) {
        bert_layer_t *l = &m->layers[i];
        free_linear(&l->query);
        free_linear(&l->key);
        free_linear(&l->value);
        free_linear(&l->attn_output);
        free(l->attn_output_norm_gamma);
        free(l->attn_output_norm_beta);
        free_linear(&l->ffn_up);
        free_linear(&l->ffn_down);
        free(l->layer_output_norm_gamma);
        free(l->layer_output_norm_beta);
    }
    free(m->layers);
    free(m);
}

int bert_hidden_size(const bert_model_t *model) {
    return model->hidden_size;
}

/* ---- Linear forward: input[seq x in] -> output[seq x out] ---- */

static float *linear_forward(const linear_t *l, const float *input, int seq_len) {
    int out = l->out_features;
    int in = l->in_features;
    float *output = malloc((size_t)seq_len * out * sizeof(float));

    /* For each position, compute output = input * W^T + bias
     * W is stored as [out x in], so we do mat_vec_mul for each position */
    for (int s = 0; s < seq_len; s++) {
        mat_vec_mul(l->weight, input + s * in, output + s * out, out, in);
    }
    broadcast_add(output, l->bias, seq_len, out);
    return output;
}

/* ---- BERT forward pass ---- */

float *bert_forward(const bert_model_t *m, const uint32_t *input_ids,
                    const uint32_t *type_ids, int seq_len) {
    int H = m->hidden_size;

    /* 1. Embeddings: word + position + token_type */
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

    /* Layer norm on embeddings */
    float *normed = malloc((size_t)seq_len * H * sizeof(float));
    layer_norm(embeddings, m->emb_norm_gamma, m->emb_norm_beta,
               normed, seq_len, H, m->layer_norm_eps);
    free(embeddings);

    float *x = normed;  /* current hidden states [seq_len x H] */

    /* 2. Transformer layers */
    for (int li = 0; li < m->num_layers; li++) {
        const bert_layer_t *layer = &m->layers[li];
        int heads = layer->num_heads;
        int hd = layer->head_dim;

        /* Self-attention: Q, K, V projections */
        float *Q = linear_forward(&layer->query, x, seq_len);   /* [seq x H] */
        float *K = linear_forward(&layer->key, x, seq_len);
        float *V = linear_forward(&layer->value, x, seq_len);

        /* Compute attention per head */
        float scale = 1.0f / sqrtf((float)hd);
        float *attn_out = calloc((size_t)seq_len * H, sizeof(float));

        for (int h = 0; h < heads; h++) {
            /* Extract head slices and compute scores */
            /* Q_h[s] = Q[s * H + h * hd ... + hd] */
            /* scores[i][j] = sum_d Q_h[i][d] * K_h[j][d] * scale */

            /* Compute attention scores for this head */
            float *scores = malloc((size_t)seq_len * seq_len * sizeof(float));
            for (int i = 0; i < seq_len; i++) {
                const float *qi = Q + i * H + h * hd;
                for (int j = 0; j < seq_len; j++) {
                    const float *kj = K + j * H + h * hd;
                    scores[i * seq_len + j] = vec_dot(qi, kj, hd) * scale;
                }
            }

            /* Softmax per row */
            for (int i = 0; i < seq_len; i++) {
                vec_softmax(scores + i * seq_len, seq_len);
            }

            /* Weighted sum of values */
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

        free(Q);
        free(K);
        free(V);

        /* Attention output projection */
        float *attn_proj = linear_forward(&layer->attn_output, attn_out, seq_len);
        free(attn_out);

        /* Residual + LayerNorm */
        float *residual1 = malloc((size_t)seq_len * H * sizeof(float));
        for (int i = 0; i < seq_len * H; i++) {
            residual1[i] = x[i] + attn_proj[i];
        }
        free(attn_proj);

        float *normed1 = malloc((size_t)seq_len * H * sizeof(float));
        layer_norm(residual1, layer->attn_output_norm_gamma,
                   layer->attn_output_norm_beta,
                   normed1, seq_len, H, m->layer_norm_eps);
        free(residual1);

        /* Feed-forward network */
        float *ffn_up = linear_forward(&layer->ffn_up, normed1, seq_len);
        int intermediate = layer->ffn_up.out_features;
        vec_gelu(ffn_up, (size_t)seq_len * intermediate);

        float *ffn_down = linear_forward(&layer->ffn_down, ffn_up, seq_len);
        free(ffn_up);

        /* Residual + LayerNorm */
        float *residual2 = malloc((size_t)seq_len * H * sizeof(float));
        for (int i = 0; i < seq_len * H; i++) {
            residual2[i] = normed1[i] + ffn_down[i];
        }
        free(normed1);
        free(ffn_down);

        float *normed2 = malloc((size_t)seq_len * H * sizeof(float));
        layer_norm(residual2, layer->layer_output_norm_gamma,
                   layer->layer_output_norm_beta,
                   normed2, seq_len, H, m->layer_norm_eps);
        free(residual2);

        free(x);
        x = normed2;
    }

    return x;  /* [seq_len x H] */
}

/* ---- Mean pooling ---- */

static float *mean_pool(const float *hidden, const uint32_t *attention_mask,
                         int seq_len, int hidden_size) {
    float *pooled = calloc(hidden_size, sizeof(float));
    float count = 0.0f;

    for (int s = 0; s < seq_len; s++) {
        if (attention_mask[s]) {
            const float *row = hidden + s * hidden_size;
            for (int i = 0; i < hidden_size; i++) {
                pooled[i] += row[i];
            }
            count += 1.0f;
        }
    }

    if (count > 0.0f) {
        float inv = 1.0f / count;
        for (int i = 0; i < hidden_size; i++) {
            pooled[i] *= inv;
        }
    }
    return pooled;
}

/* ---- Embedder ---- */

bert_embedder_t *bert_embedder_load(const char *model_dir,
                                     const char *query_prefix,
                                     const char *doc_prefix) {
    /* Find .gguf file in directory */
    char gguf_path[512] = {0};
    DIR *dir = opendir(model_dir);
    if (!dir) {
        fprintf(stderr, "Cannot open model directory: %s\n", model_dir);
        return NULL;
    }
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        size_t len = strlen(entry->d_name);
        if (len > 5 && strcmp(entry->d_name + len - 5, ".gguf") == 0) {
            snprintf(gguf_path, sizeof(gguf_path), "%s/%s", model_dir, entry->d_name);
            break;
        }
    }
    closedir(dir);

    if (!gguf_path[0]) {
        fprintf(stderr, "No .gguf file found in %s\n", model_dir);
        return NULL;
    }

    printf("  Loading GGUF: %s\n", gguf_path);
    gguf_ctx_t *ctx = gguf_load(gguf_path);
    if (!ctx) return NULL;

    bert_model_t *model = bert_model_load(ctx);
    gguf_free(ctx);

    /* Load tokenizer */
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    tokenizer_t *tokenizer = tokenizer_load(tok_path);
    if (!tokenizer) {
        bert_model_free(model);
        return NULL;
    }

    bert_embedder_t *emb = calloc(1, sizeof(bert_embedder_t));
    emb->model = model;
    emb->tokenizer = tokenizer;
    emb->query_prefix = strdup(query_prefix);
    emb->doc_prefix = strdup(doc_prefix);

    /* Probe dimensions */
    float *probe = bert_embed(emb, "probe");
    emb->dims = bert_hidden_size(model);
    free(probe);

    return emb;
}

void bert_embedder_free(bert_embedder_t *emb) {
    if (!emb) return;
    bert_model_free(emb->model);
    tokenizer_free(emb->tokenizer);
    free(emb->query_prefix);
    free(emb->doc_prefix);
    free(emb);
}

float *bert_embed(const bert_embedder_t *emb, const char *text) {
    token_encoding_t *enc = tokenizer_encode(emb->tokenizer, text);
    if (!enc) return NULL;

    float *hidden = bert_forward(emb->model, enc->ids, enc->type_ids,
                                  (int)enc->length);

    float *pooled = mean_pool(hidden, enc->attention_mask,
                               (int)enc->length, emb->dims);
    free(hidden);
    token_encoding_free(enc);

    vec_l2_normalize(pooled, emb->dims);
    return pooled;
}

float *bert_embed_query(const bert_embedder_t *emb, const char *query) {
    size_t plen = strlen(emb->query_prefix);
    size_t qlen = strlen(query);
    char *text = malloc(plen + qlen + 1);
    memcpy(text, emb->query_prefix, plen);
    memcpy(text + plen, query, qlen + 1);
    float *result = bert_embed(emb, text);
    free(text);
    return result;
}

float *bert_embed_document(const bert_embedder_t *emb, const char *doc) {
    size_t plen = strlen(emb->doc_prefix);
    size_t dlen = strlen(doc);
    char *text = malloc(plen + dlen + 1);
    memcpy(text, emb->doc_prefix, plen);
    memcpy(text + plen, doc, dlen + 1);
    float *result = bert_embed(emb, text);
    free(text);
    return result;
}
