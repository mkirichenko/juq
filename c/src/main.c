#include "bert.h"
#include "cjson.h"
#include "simd_ops.h"
#ifdef USE_ONNX
#include "onnx_backend.h"
#endif
#ifdef USE_INT8
#include "int8_backend.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---- Document types ---- */

typedef struct {
    char *id;
    char *phrase1;
    char *phrase2;
} document_t;

typedef struct {
    int doc_index;
    float score;
} search_result_t;

/* ---- Load documents from JSON ---- */

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

static document_t *load_documents(const char *path, int *out_count) {
    char *text = read_file(path);
    if (!text) { fprintf(stderr, "Cannot read: %s\n", path); return NULL; }

    json_value_t *root = json_parse(text);
    free(text);
    if (!root || root->type != JSON_ARRAY) {
        fprintf(stderr, "Invalid documents JSON\n");
        json_free(root);
        return NULL;
    }

    int n = (int)json_array_length(root);
    document_t *docs = calloc(n, sizeof(document_t));

    for (int i = 0; i < n; i++) {
        json_value_t *item = json_array_get(root, i);
        const char *id = json_string_value(json_object_get(item, "id"));
        const char *p1 = json_string_value(json_object_get(item, "phrase1"));
        const char *p2 = json_string_value(json_object_get(item, "phrase2"));
        docs[i].id = strdup(id ? id : "");
        docs[i].phrase1 = strdup(p1 ? p1 : "");
        docs[i].phrase2 = strdup(p2 ? p2 : "");
    }

    *out_count = n;
    json_free(root);
    return docs;
}

static void free_documents(document_t *docs, int n) {
    for (int i = 0; i < n; i++) {
        free(docs[i].id);
        free(docs[i].phrase1);
        free(docs[i].phrase2);
    }
    free(docs);
}

/* ---- Brute-force vector index ---- */

typedef struct {
    int *doc_ids;
    float **vectors;
    int count;
    int capacity;
    int dims;
} brute_force_index_t;

static brute_force_index_t *bfi_new(int dims) {
    brute_force_index_t *idx = calloc(1, sizeof(brute_force_index_t));
    idx->dims = dims;
    idx->capacity = 256;
    idx->doc_ids = malloc(idx->capacity * sizeof(int));
    idx->vectors = malloc(idx->capacity * sizeof(float *));
    return idx;
}

static void bfi_add(brute_force_index_t *idx, int doc_id, float *vector) {
    if (idx->count >= idx->capacity) {
        idx->capacity *= 2;
        idx->doc_ids = realloc(idx->doc_ids, idx->capacity * sizeof(int));
        idx->vectors = realloc(idx->vectors, idx->capacity * sizeof(float *));
    }
    idx->doc_ids[idx->count] = doc_id;
    idx->vectors[idx->count] = vector;
    idx->count++;
}

/* Min-heap for top-K search */
typedef struct { float score; int id; } heap_entry_t;

static void heap_sift_up(heap_entry_t *heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[i].score < heap[parent].score) {
            heap_entry_t tmp = heap[i];
            heap[i] = heap[parent];
            heap[parent] = tmp;
            i = parent;
        } else break;
    }
}

static void heap_sift_down(heap_entry_t *heap, int n, int i) {
    while (1) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        if (left < n && heap[left].score < heap[smallest].score) smallest = left;
        if (right < n && heap[right].score < heap[smallest].score) smallest = right;
        if (smallest != i) {
            heap_entry_t tmp = heap[i];
            heap[i] = heap[smallest];
            heap[smallest] = tmp;
            i = smallest;
        } else break;
    }
}

static search_result_t *bfi_search(const brute_force_index_t *idx,
                                    const float *query, int top_k,
                                    int *out_count) {
    heap_entry_t *heap = malloc(top_k * sizeof(heap_entry_t));
    int heap_size = 0;

    for (int i = 0; i < idx->count; i++) {
        float score = vec_dot(query, idx->vectors[i], idx->dims);
        if (heap_size < top_k) {
            heap[heap_size] = (heap_entry_t){ .score = score, .id = idx->doc_ids[i] };
            heap_sift_up(heap, heap_size);
            heap_size++;
        } else if (score > heap[0].score) {
            heap[0] = (heap_entry_t){ .score = score, .id = idx->doc_ids[i] };
            heap_sift_down(heap, heap_size, 0);
        }
    }

    /* Extract sorted results */
    search_result_t *results = malloc(heap_size * sizeof(search_result_t));
    int n = heap_size;
    for (int i = n - 1; i >= 0; i--) {
        results[i] = (search_result_t){ .doc_index = heap[0].id, .score = heap[0].score };
        heap[0] = heap[--heap_size];
        if (heap_size > 0) heap_sift_down(heap, heap_size, 0);
    }

    free(heap);
    *out_count = n;
    return results;
}

static void bfi_free(brute_force_index_t *idx) {
    if (!idx) return;
    for (int i = 0; i < idx->count; i++) free(idx->vectors[i]);
    free(idx->doc_ids);
    free(idx->vectors);
    free(idx);
}

/* ---- Phrase strategy ---- */

typedef enum {
    STRATEGY_CONCATENATE,
    STRATEGY_AVERAGE,
    STRATEGY_MAX_SIM,
} phrase_strategy_t;

static phrase_strategy_t parse_strategy(const char *s) {
    if (strcasecmp(s, "concatenate") == 0) return STRATEGY_CONCATENATE;
    if (strcasecmp(s, "average") == 0) return STRATEGY_AVERAGE;
    if (strcasecmp(s, "max_sim") == 0 || strcasecmp(s, "maxsim") == 0) return STRATEGY_MAX_SIM;
    fprintf(stderr, "Unknown strategy: %s (use: concatenate, average, max_sim)\n", s);
    exit(1);
}

static const char *strategy_name(phrase_strategy_t s) {
    switch (s) {
        case STRATEGY_CONCATENATE: return "Concatenate";
        case STRATEGY_AVERAGE: return "Average";
        case STRATEGY_MAX_SIM: return "MaxSim";
    }
    return "?";
}

/* ---- Search engine ---- */

typedef struct {
    bert_embedder_t *embedder;
    brute_force_index_t *index;
    phrase_strategy_t strategy;
    int *id_to_doc;   /* maps internal id -> doc index */
    int next_id;
    int id_capacity;
} search_engine_t;

static search_engine_t *engine_new(bert_embedder_t *embedder, phrase_strategy_t strategy) {
    search_engine_t *e = calloc(1, sizeof(search_engine_t));
    e->embedder = embedder;
    e->index = bfi_new(embedder->dims);
    e->strategy = strategy;
    e->id_capacity = 256;
    e->id_to_doc = malloc(e->id_capacity * sizeof(int));
    return e;
}

static void engine_register_id(search_engine_t *e, int doc_idx) {
    if (e->next_id >= e->id_capacity) {
        e->id_capacity *= 2;
        e->id_to_doc = realloc(e->id_to_doc, e->id_capacity * sizeof(int));
    }
    e->id_to_doc[e->next_id] = doc_idx;
}

static int engine_index_documents(search_engine_t *e, const document_t *docs, int n) {
    for (int i = 0; i < n; i++) {
        int id = e->next_id++;
        engine_register_id(e, i);

        switch (e->strategy) {
            case STRATEGY_CONCATENATE: {
                size_t len = strlen(docs[i].phrase1) + strlen(docs[i].phrase2) + 3;
                char *text = malloc(len);
                snprintf(text, len, "%s. %s", docs[i].phrase1, docs[i].phrase2);
                float *vec = bert_embed_document(e->embedder, text);
                free(text);
                if (!vec) return -1;
                bfi_add(e->index, id, vec);
                break;
            }
            case STRATEGY_AVERAGE: {
                float *v1 = bert_embed_document(e->embedder, docs[i].phrase1);
                float *v2 = bert_embed_document(e->embedder, docs[i].phrase2);
                if (!v1 || !v2) { free(v1); free(v2); return -1; }
                int dims = e->embedder->dims;
                float *avg = malloc(dims * sizeof(float));
                for (int d = 0; d < dims; d++) {
                    avg[d] = (v1[d] + v2[d]) * 0.5f;
                }
                vec_l2_normalize(avg, dims);
                free(v1);
                free(v2);
                bfi_add(e->index, id, avg);
                break;
            }
            case STRATEGY_MAX_SIM: {
                float *v1 = bert_embed_document(e->embedder, docs[i].phrase1);
                float *v2 = bert_embed_document(e->embedder, docs[i].phrase2);
                if (!v1 || !v2) { free(v1); free(v2); return -1; }
                bfi_add(e->index, id, v1);
                /* Second vector gets a new id mapping to same doc */
                int id2 = e->next_id++;
                engine_register_id(e, i);
                bfi_add(e->index, id2, v2);
                break;
            }
        }
    }
    return 0;
}

static search_result_t *engine_search(const search_engine_t *e, const char *query,
                                       int top_k, int *out_count) {
    float *query_vec = bert_embed_query(e->embedder, query);
    if (!query_vec) { *out_count = 0; return NULL; }

    if (e->strategy == STRATEGY_MAX_SIM) {
        int raw_count;
        search_result_t *raw = bfi_search(e->index, query_vec, top_k * 2, &raw_count);
        free(query_vec);

        /* Deduplicate: keep best score per document */
        typedef struct { int doc_idx; float score; } best_t;
        best_t *best = calloc(raw_count, sizeof(best_t));
        int n_best = 0;

        for (int i = 0; i < raw_count; i++) {
            int doc_idx = e->id_to_doc[raw[i].doc_index];
            int found = 0;
            for (int j = 0; j < n_best; j++) {
                if (best[j].doc_idx == doc_idx) {
                    if (raw[i].score > best[j].score) best[j].score = raw[i].score;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                best[n_best++] = (best_t){ .doc_idx = doc_idx, .score = raw[i].score };
            }
        }
        free(raw);

        /* Sort by score descending */
        for (int i = 0; i < n_best - 1; i++) {
            for (int j = i + 1; j < n_best; j++) {
                if (best[j].score > best[i].score) {
                    best_t tmp = best[i]; best[i] = best[j]; best[j] = tmp;
                }
            }
        }

        int result_count = n_best < top_k ? n_best : top_k;
        search_result_t *results = malloc(result_count * sizeof(search_result_t));
        for (int i = 0; i < result_count; i++) {
            results[i] = (search_result_t){ .doc_index = best[i].doc_idx, .score = best[i].score };
        }
        free(best);
        *out_count = result_count;
        return results;
    }

    int raw_count;
    search_result_t *raw = bfi_search(e->index, query_vec, top_k, &raw_count);
    free(query_vec);

    /* Map internal ids to doc indices */
    for (int i = 0; i < raw_count; i++) {
        raw[i].doc_index = e->id_to_doc[raw[i].doc_index];
    }
    *out_count = raw_count;
    return raw;
}

static void engine_free(search_engine_t *e) {
    if (!e) return;
    bfi_free(e->index);
    free(e->id_to_doc);
    free(e);
}

/* ---- Time helpers ---- */

static double time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* ---- Benchmark ---- */

static const char *SAMPLE_QUERIES[] = {
    "machine learning algorithms",
    "climate change impact",
    "software engineering best practices",
    "healthy cooking recipes",
    "space exploration missions",
    "financial market analysis",
    "artificial intelligence ethics",
    "renewable energy sources",
    "modern web development",
    "quantum computing applications",
};
#define NUM_QUERIES 10

static void run_benchmark(bert_embedder_t *embedder, const document_t *docs, int n_docs) {
    printf("\n=== BENCHMARK ===\n\n");

    phrase_strategy_t strategies[] = { STRATEGY_CONCATENATE, STRATEGY_AVERAGE, STRATEGY_MAX_SIM };

    for (int si = 0; si < 3; si++) {
        phrase_strategy_t strategy = strategies[si];
        printf("--- Strategy: %s ---\n", strategy_name(strategy));

        search_engine_t *engine = engine_new(embedder, strategy);

        double t0 = time_ms();
        engine_index_documents(engine, docs, n_docs);
        double index_ms = time_ms() - t0;
        printf("  Indexing: %d docs in %.0f ms (%.1f ms/doc)\n",
               n_docs, index_ms, index_ms / n_docs);
        printf("  Index vectors: %d\n", engine->index->count);

        /* Warmup */
        for (int i = 0; i < 3; i++) {
            int rc;
            search_result_t *r = engine_search(engine, SAMPLE_QUERIES[i % NUM_QUERIES], 5, &rc);
            free(r);
        }

        /* Timed search */
        double latencies[NUM_QUERIES];
        for (int i = 0; i < NUM_QUERIES; i++) {
            double t1 = time_ms();
            int rc;
            search_result_t *r = engine_search(engine, SAMPLE_QUERIES[i], 5, &rc);
            latencies[i] = time_ms() - t1;
            free(r);
        }

        /* Sort latencies */
        for (int i = 0; i < NUM_QUERIES - 1; i++) {
            for (int j = i + 1; j < NUM_QUERIES; j++) {
                if (latencies[j] < latencies[i]) {
                    double tmp = latencies[i]; latencies[i] = latencies[j]; latencies[j] = tmp;
                }
            }
        }

        printf("  Search latency (ms): min=%.0f, p50=%.0f, p95=%.0f, max=%.0f\n",
               latencies[0], latencies[NUM_QUERIES / 2],
               latencies[(int)(NUM_QUERIES * 0.95)], latencies[NUM_QUERIES - 1]);

        /* Sample result */
        int rc;
        search_result_t *sample = engine_search(engine, SAMPLE_QUERIES[0], 3, &rc);
        printf("  Sample query: \"%s\"\n", SAMPLE_QUERIES[0]);
        for (int i = 0; i < rc; i++) {
            printf("    %d. [%.4f] %s\n", i + 1, sample[i].score, docs[sample[i].doc_index].id);
        }
        free(sample);
        printf("\n");

        engine_free(engine);
    }
}

/* ---- CLI ---- */

static void usage(void) {
    fprintf(stderr, "Usage: juq [options]\n");
    fprintf(stderr, "  --data <path>           Path to documents.json\n");
    fprintf(stderr, "  --model <path>          Model directory\n");
    fprintf(stderr, "  --backend <name>        gguf, onnx, or int8 (default: gguf)\n");
    fprintf(stderr, "  --model-file <name>     ONNX model filename (default: model.onnx)\n");
    fprintf(stderr, "  --query <text>          Search query\n");
    fprintf(stderr, "  --top-k <n>             Number of results (default: 5)\n");
    fprintf(stderr, "  --strategy <name>       concatenate|average|max_sim\n");
    fprintf(stderr, "  --benchmark             Run benchmark suite\n");
    fprintf(stderr, "  --query-prefix <str>    Query prefix (default: \"search_query: \")\n");
    fprintf(stderr, "  --doc-prefix <str>      Document prefix (default: \"search_document: \")\n");
    exit(1);
}

int main(int argc, char **argv) {
    const char *data_path = "data/documents.json";
    const char *model_dir = "model/berta";
    const char *query = NULL;
    const char *backend = "gguf";
    const char *model_file __attribute__((unused)) = NULL;
    int top_k = 5;
    phrase_strategy_t strategy = STRATEGY_CONCATENATE;
    int benchmark = 0;
    const char *query_prefix = "search_query: ";
    const char *doc_prefix = "search_document: ";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) { data_path = argv[++i]; }
        else if ((strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "--model-dir") == 0) && i + 1 < argc) { model_dir = argv[++i]; }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) { backend = argv[++i]; }
        else if (strcmp(argv[i], "--model-file") == 0 && i + 1 < argc) { model_file = argv[++i]; }
        else if (strcmp(argv[i], "--query") == 0 && i + 1 < argc) { query = argv[++i]; }
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) { top_k = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--strategy") == 0 && i + 1 < argc) { strategy = parse_strategy(argv[++i]); }
        else if (strcmp(argv[i], "--benchmark") == 0) { benchmark = 1; }
        else if (strcmp(argv[i], "--query-prefix") == 0 && i + 1 < argc) { query_prefix = argv[++i]; }
        else if (strcmp(argv[i], "--doc-prefix") == 0 && i + 1 < argc) { doc_prefix = argv[++i]; }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); usage(); }
    }

    if (!benchmark && !query) usage();

    double t0 = time_ms();
    printf("Loading model from %s (backend: %s) ...\n", model_dir, backend);

    bert_embedder_t *embedder = NULL;
    if (strcasecmp(backend, "onnx") == 0) {
#ifdef USE_ONNX
        embedder = onnx_embedder_load(model_dir, model_file, query_prefix, doc_prefix);
#else
        fprintf(stderr, "ONNX backend not compiled. Rebuild with: make onnx\n");
        return 1;
#endif
    } else if (strcasecmp(backend, "int8") == 0) {
#ifdef USE_INT8
        embedder = int8_embedder_load(model_dir, model_file, query_prefix, doc_prefix);
#else
        fprintf(stderr, "Int8 backend not compiled. Rebuild with: make int8\n");
        return 1;
#endif
    } else if (strcasecmp(backend, "gguf") == 0) {
        embedder = bert_embedder_load(model_dir, query_prefix, doc_prefix);
    } else {
        fprintf(stderr, "Unknown backend: %s (use: gguf, onnx, int8)\n", backend);
        return 1;
    }
    if (!embedder) { fprintf(stderr, "Failed to load model\n"); return 1; }
    printf("Model loaded in %.0f ms (dimensions: %d)\n", time_ms() - t0, embedder->dims);

    int n_docs;
    document_t *docs = load_documents(data_path, &n_docs);
    if (!docs) { bert_embedder_free(embedder); return 1; }
    printf("Loaded %d documents\n", n_docs);

    if (benchmark) {
        run_benchmark(embedder, docs, n_docs);
    } else {
        search_engine_t *engine = engine_new(embedder, strategy);

        double t1 = time_ms();
        engine_index_documents(engine, docs, n_docs);
        printf("Indexed %d documents in %.0f ms (strategy: %s)\n",
               n_docs, time_ms() - t1, strategy_name(strategy));

        double t2 = time_ms();
        int result_count;
        search_result_t *results = engine_search(engine, query, top_k, &result_count);
        printf("\nQuery: \"%s\" (took %.0f ms)\n", query, time_ms() - t2);

        /* Print separator */
        for (int i = 0; i < 60; i++) putchar('-');
        putchar('\n');

        for (int i = 0; i < result_count; i++) {
            int idx = results[i].doc_index;
            printf("%d. [%.4f] %s\n", i + 1, results[i].score, docs[idx].id);
            printf("   phrase1: %s\n", docs[idx].phrase1);
            printf("   phrase2: %s\n", docs[idx].phrase2);
        }

        free(results);
        engine_free(engine);
    }

    free_documents(docs, n_docs);
    bert_embedder_free(embedder);
    return 0;
}
