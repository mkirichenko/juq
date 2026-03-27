#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct tokenizer tokenizer_t;

/* Tokenization output */
typedef struct {
    uint32_t *ids;
    uint32_t *type_ids;
    uint32_t *attention_mask;
    size_t length;
} token_encoding_t;

/* Load tokenizer from HuggingFace tokenizer.json */
tokenizer_t *tokenizer_load(const char *path);
void tokenizer_free(tokenizer_t *tok);

/* Encode text with [CLS] ... [SEP] wrapping */
token_encoding_t *tokenizer_encode(const tokenizer_t *tok, const char *text);
void token_encoding_free(token_encoding_t *enc);

#endif
