#include "tokenizer.h"
#include "cjson.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* Hash table for vocab lookup */
#define VOCAB_HASH_SIZE 65521

typedef struct vocab_entry {
    char *token;
    uint32_t id;
    struct vocab_entry *next;
} vocab_entry_t;

struct tokenizer {
    vocab_entry_t *hash_table[VOCAB_HASH_SIZE];
    uint32_t cls_id;
    uint32_t sep_id;
    uint32_t unk_id;
    uint32_t pad_id;
    char *continuing_prefix;  /* "##" for WordPiece */
    int max_input_chars_per_word;
    int vocab_size;
};

static uint32_t hash_string(const char *s) {
    uint32_t h = 5381;
    for (; *s; s++) h = ((h << 5) + h) ^ (unsigned char)*s;
    return h % VOCAB_HASH_SIZE;
}

static void vocab_insert(tokenizer_t *tok, const char *token, uint32_t id) {
    uint32_t h = hash_string(token);
    vocab_entry_t *entry = malloc(sizeof(vocab_entry_t));
    entry->token = strdup(token);
    entry->id = id;
    entry->next = tok->hash_table[h];
    tok->hash_table[h] = entry;
}

static int vocab_lookup(const tokenizer_t *tok, const char *token, uint32_t *id) {
    uint32_t h = hash_string(token);
    for (vocab_entry_t *e = tok->hash_table[h]; e; e = e->next) {
        if (strcmp(e->token, token) == 0) {
            *id = e->id;
            return 1;
        }
    }
    return 0;
}

/* Read entire file into string */
static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc(size + 1);
    if (fread(buf, 1, size, f) != (size_t)size) { free(buf); fclose(f); return NULL; }
    buf[size] = '\0';
    fclose(f);
    return buf;
}

tokenizer_t *tokenizer_load(const char *path) {
    char *text = read_file(path);
    if (!text) { fprintf(stderr, "Cannot read tokenizer: %s\n", path); return NULL; }

    json_value_t *root = json_parse(text);
    free(text);
    if (!root) { fprintf(stderr, "Failed to parse tokenizer JSON\n"); return NULL; }

    tokenizer_t *tok = calloc(1, sizeof(tokenizer_t));
    tok->max_input_chars_per_word = 100;
    tok->continuing_prefix = strdup("##");

    /* Parse model.vocab */
    json_value_t *model = json_object_get(root, "model");
    if (!model) { fprintf(stderr, "No 'model' in tokenizer.json\n"); goto fail; }

    json_value_t *vocab = json_object_get(model, "vocab");
    if (!vocab || vocab->type != JSON_OBJECT) {
        fprintf(stderr, "No 'model.vocab' in tokenizer.json\n");
        goto fail;
    }

    /* Load vocabulary */
    int count = 0;
    for (json_member_t *m = vocab->object; m; m = m->next) {
        if (m->value->type == JSON_NUMBER) {
            vocab_insert(tok, m->key, (uint32_t)m->value->number);
            count++;
        }
    }
    tok->vocab_size = count;

    /* Parse special tokens from added_tokens */
    tok->cls_id = 101;  /* defaults for BERT */
    tok->sep_id = 102;
    tok->unk_id = 100;
    tok->pad_id = 0;

    json_value_t *added = json_object_get(root, "added_tokens");
    if (added && added->type == JSON_ARRAY) {
        for (json_element_t *el = added->array; el; el = el->next) {
            json_value_t *item = el->value;
            const char *content = json_string_value(json_object_get(item, "content"));
            json_value_t *id_val = json_object_get(item, "id");
            if (!content || !id_val) continue;
            uint32_t id = (uint32_t)json_number_value(id_val);
            if (strcmp(content, "[CLS]") == 0) tok->cls_id = id;
            else if (strcmp(content, "[SEP]") == 0) tok->sep_id = id;
            else if (strcmp(content, "[UNK]") == 0) tok->unk_id = id;
            else if (strcmp(content, "[PAD]") == 0) tok->pad_id = id;
        }
    }

    /* Check for continuing_subword_prefix */
    json_value_t *prefix = json_object_get(model, "continuing_subword_prefix");
    if (prefix && prefix->type == JSON_STRING) {
        free(tok->continuing_prefix);
        tok->continuing_prefix = strdup(prefix->string);
    }

    json_value_t *max_chars = json_object_get(model, "max_input_chars_per_word");
    if (max_chars && max_chars->type == JSON_NUMBER) {
        tok->max_input_chars_per_word = (int)max_chars->number;
    }

    printf("  Tokenizer: %d vocab entries, prefix=\"%s\"\n",
           tok->vocab_size, tok->continuing_prefix);

    json_free(root);
    return tok;

fail:
    json_free(root);
    tokenizer_free(tok);
    return NULL;
}

void tokenizer_free(tokenizer_t *tok) {
    if (!tok) return;
    for (int i = 0; i < VOCAB_HASH_SIZE; i++) {
        vocab_entry_t *e = tok->hash_table[i];
        while (e) {
            vocab_entry_t *next = e->next;
            free(e->token);
            free(e);
            e = next;
        }
    }
    free(tok->continuing_prefix);
    free(tok);
}

/* ---- Pre-tokenization ---- */

/* Split text into words (whitespace + punctuation boundaries) */
typedef struct {
    char **words;
    int count;
    int capacity;
} word_list_t;

static void wl_push(word_list_t *wl, const char *start, size_t len) {
    if (len == 0) return;
    if (wl->count >= wl->capacity) {
        wl->capacity = wl->capacity ? wl->capacity * 2 : 64;
        wl->words = realloc(wl->words, wl->capacity * sizeof(char *));
    }
    char *w = malloc(len + 1);
    memcpy(w, start, len);
    w[len] = '\0';
    wl->words[wl->count++] = w;
}

static void wl_free(word_list_t *wl) {
    for (int i = 0; i < wl->count; i++) free(wl->words[i]);
    free(wl->words);
}

static int is_punct(unsigned char c) {
    return ispunct(c) || c == 0;
}

static word_list_t pre_tokenize(const char *text) {
    word_list_t wl = {0};
    size_t len = strlen(text);
    size_t i = 0;

    while (i < len) {
        /* Skip whitespace */
        while (i < len && isspace((unsigned char)text[i])) i++;
        if (i >= len) break;

        if (is_punct((unsigned char)text[i])) {
            /* Each punctuation is its own token */
            wl_push(&wl, &text[i], 1);
            i++;
        } else {
            /* Collect word characters */
            size_t start = i;
            while (i < len && !isspace((unsigned char)text[i]) &&
                   !is_punct((unsigned char)text[i])) {
                i++;
            }
            wl_push(&wl, &text[start], i - start);
        }
    }
    return wl;
}

/* ---- WordPiece tokenization ---- */

typedef struct {
    uint32_t *ids;
    int count;
    int capacity;
} id_list_t;

static void il_push(id_list_t *il, uint32_t id) {
    if (il->count >= il->capacity) {
        il->capacity = il->capacity ? il->capacity * 2 : 128;
        il->ids = realloc(il->ids, il->capacity * sizeof(uint32_t));
    }
    il->ids[il->count++] = id;
}

/* Convert a UTF-8 string to lowercase in-place (ASCII only for simplicity) */
static void to_lower(char *s) {
    for (; *s; s++) *s = tolower((unsigned char)*s);
}

/* Return number of bytes in the UTF-8 character starting at s */
static int utf8_char_len(const unsigned char *s) {
    if (*s < 0x80) return 1;
    if ((*s & 0xE0) == 0xC0) return 2;
    if ((*s & 0xF0) == 0xE0) return 3;
    if ((*s & 0xF8) == 0xF0) return 4;
    return 1;
}

static void wordpiece_tokenize(const tokenizer_t *tok, const char *word,
                                id_list_t *out) {
    size_t wlen = strlen(word);
    if ((int)wlen > tok->max_input_chars_per_word) {
        il_push(out, tok->unk_id);
        return;
    }

    /* Make lowercase copy */
    char *lower = strdup(word);
    to_lower(lower);
    wlen = strlen(lower);

    size_t start = 0;
    int is_first = 1;

    size_t prefix_len = strlen(tok->continuing_prefix);
    /* Temporary buffer for subword candidate */
    char *candidate = malloc(wlen + prefix_len + 1);

    while (start < wlen) {
        size_t end = wlen;
        int found = 0;

        while (end > start) {
            /* Build candidate */
            if (is_first) {
                size_t slen = end - start;
                memcpy(candidate, lower + start, slen);
                candidate[slen] = '\0';
            } else {
                memcpy(candidate, tok->continuing_prefix, prefix_len);
                size_t slen = end - start;
                memcpy(candidate + prefix_len, lower + start, slen);
                candidate[prefix_len + slen] = '\0';
            }

            uint32_t id;
            if (vocab_lookup(tok, candidate, &id)) {
                il_push(out, id);
                found = 1;
                break;
            }

            /* Shrink by one UTF-8 character from the end */
            if (end == start + 1) break;
            /* Move end back by one UTF-8 char */
            end--;
            while (end > start && (lower[end] & 0xC0) == 0x80) end--;
        }

        if (!found) {
            il_push(out, tok->unk_id);
            /* Skip one UTF-8 char */
            start += utf8_char_len((const unsigned char *)lower + start);
            is_first = 0;
            continue;
        }

        start = end;
        is_first = 0;
    }

    free(candidate);
    free(lower);
}

token_encoding_t *tokenizer_encode(const tokenizer_t *tok, const char *text) {
    word_list_t words = pre_tokenize(text);

    id_list_t ids = {0};

    /* [CLS] */
    il_push(&ids, tok->cls_id);

    /* WordPiece each word */
    for (int i = 0; i < words.count; i++) {
        wordpiece_tokenize(tok, words.words[i], &ids);
    }

    /* [SEP] */
    il_push(&ids, tok->sep_id);

    wl_free(&words);

    token_encoding_t *enc = calloc(1, sizeof(token_encoding_t));
    enc->length = ids.count;
    enc->ids = ids.ids;

    /* All type_ids = 0, all attention_mask = 1 */
    enc->type_ids = calloc(ids.count, sizeof(uint32_t));
    enc->attention_mask = malloc(ids.count * sizeof(uint32_t));
    for (size_t i = 0; i < enc->length; i++) {
        enc->attention_mask[i] = 1;
    }

    return enc;
}

void token_encoding_free(token_encoding_t *enc) {
    if (!enc) return;
    free(enc->ids);
    free(enc->type_ids);
    free(enc->attention_mask);
    free(enc);
}
