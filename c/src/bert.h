#ifndef BERT_H
#define BERT_H

#include "gguf.h"
#include "tokenizer.h"

/* BERT model loaded from GGUF with pre-dequantized weights */
typedef struct bert_model bert_model_t;

/* High-level embedder wrapping model + tokenizer */
typedef struct {
    bert_model_t *model;
    tokenizer_t *tokenizer;
    char *query_prefix;
    char *doc_prefix;
    int dims;
} bert_embedder_t;

/* Load BERT model from GGUF file */
bert_model_t *bert_model_load(gguf_ctx_t *ctx);
void bert_model_free(bert_model_t *model);

/* Forward pass: input_ids[seq_len], type_ids[seq_len] -> output[seq_len x hidden] */
float *bert_forward(const bert_model_t *model, const uint32_t *input_ids,
                    const uint32_t *type_ids, int seq_len);

/* Get hidden size */
int bert_hidden_size(const bert_model_t *model);

/* Load embedder (model + tokenizer from directory) */
bert_embedder_t *bert_embedder_load(const char *model_dir,
                                     const char *query_prefix,
                                     const char *doc_prefix);
void bert_embedder_free(bert_embedder_t *emb);

/* Embed text -> normalized vector of size emb->dims, caller must free */
float *bert_embed(const bert_embedder_t *emb, const char *text);
float *bert_embed_query(const bert_embedder_t *emb, const char *query);
float *bert_embed_document(const bert_embedder_t *emb, const char *doc);

#endif
