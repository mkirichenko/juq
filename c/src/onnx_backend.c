#include "onnx_backend.h"
#include "simd_ops.h"
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dirent.h>

/* ONNX Runtime backend data */
typedef struct {
    const OrtApi *api;
    OrtEnv *env;
    OrtSession *session;
    OrtSessionOptions *session_opts;
    OrtMemoryInfo *memory_info;
    OrtAllocator *allocator;
    int has_token_type_ids;
    char *hidden_state_output_name;
    int hidden_state_output_index;
} onnx_backend_t;

/* ---- Error handling ---- */

static void check_ort_status(const OrtApi *api, OrtStatus *status, const char *msg) {
    if (status != NULL) {
        const char *err = api->GetErrorMessage(status);
        fprintf(stderr, "ONNX Runtime error (%s): %s\n", msg, err);
        api->ReleaseStatus(status);
        exit(1);
    }
}

/* ---- Detect model inputs/outputs ---- */

static int has_input_name(const OrtApi *api, OrtSession *session,
                           OrtAllocator *allocator, const char *target) {
    size_t num_inputs;
    check_ort_status(api, api->SessionGetInputCount(session, &num_inputs), "GetInputCount");

    for (size_t i = 0; i < num_inputs; i++) {
        char *name;
        check_ort_status(api, api->SessionGetInputName(session, i, allocator, &name), "GetInputName");
        int match = (strcmp(name, target) == 0);
        check_ort_status(api, api->AllocatorFree(allocator, name), "AllocatorFree");
        if (match) return 1;
    }
    return 0;
}

/* Detect hidden state output (the 3D output [batch, seq, hidden]) by running a probe */
static void detect_output(onnx_backend_t *backend, bert_embedder_t *emb) {
    const OrtApi *api = backend->api;

    /* Create minimal input: batch=1, seq=3 (e.g. [CLS] probe [SEP]) */
    int64_t input_shape[2] = {1, 3};
    int64_t input_ids_data[3] = {101, 2000, 102};  /* typical BERT tokens */
    int64_t attn_mask_data[3] = {1, 1, 1};
    int64_t type_ids_data[3] = {0, 0, 0};

    /* Build input tensors */
    OrtValue *input_ids_tensor = NULL;
    OrtValue *attn_mask_tensor = NULL;
    OrtValue *type_ids_tensor = NULL;

    check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
        backend->memory_info, input_ids_data, sizeof(input_ids_data),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_ids_tensor), "CreateTensor");

    check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
        backend->memory_info, attn_mask_data, sizeof(attn_mask_data),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &attn_mask_tensor), "CreateTensor");

    const char *input_names[3];
    OrtValue *input_tensors[3];
    size_t num_inputs = 2;

    input_names[0] = "input_ids";
    input_names[1] = "attention_mask";
    input_tensors[0] = input_ids_tensor;
    input_tensors[1] = attn_mask_tensor;

    if (backend->has_token_type_ids) {
        check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
            backend->memory_info, type_ids_data, sizeof(type_ids_data),
            input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &type_ids_tensor), "CreateTensor");
        input_names[2] = "token_type_ids";
        input_tensors[2] = type_ids_tensor;
        num_inputs = 3;
    }

    /* Get all output names */
    size_t num_outputs;
    check_ort_status(api, api->SessionGetOutputCount(backend->session, &num_outputs), "GetOutputCount");

    char **output_names = malloc(num_outputs * sizeof(char *));
    for (size_t i = 0; i < num_outputs; i++) {
        check_ort_status(api, api->SessionGetOutputName(
            backend->session, i, backend->allocator, &output_names[i]), "GetOutputName");
    }

    /* Run inference */
    OrtValue **output_tensors = calloc(num_outputs, sizeof(OrtValue *));
    check_ort_status(api, api->Run(backend->session, NULL,
        input_names, (const OrtValue *const *)input_tensors, num_inputs,
        (const char *const *)output_names, num_outputs, output_tensors), "Run");

    /* Find the 3D output [batch, seq, hidden] */
    backend->hidden_state_output_index = 0;
    int dims_found = 0;

    for (size_t i = 0; i < num_outputs; i++) {
        OrtTensorTypeAndShapeInfo *info;
        check_ort_status(api, api->GetTensorTypeAndShape(output_tensors[i], &info), "GetInfo");

        size_t ndims;
        check_ort_status(api, api->GetDimensionsCount(info, &ndims), "GetDimsCount");

        if (ndims == 3) {
            int64_t dims[3];
            check_ort_status(api, api->GetDimensions(info, dims, 3), "GetDims");

            backend->hidden_state_output_index = (int)i;
            backend->hidden_state_output_name = strdup(output_names[i]);
            emb->dims = (int)dims[2];  /* hidden size */
            dims_found = 1;
            api->ReleaseTensorTypeAndShapeInfo(info);
            break;
        }
        api->ReleaseTensorTypeAndShapeInfo(info);
    }

    if (!dims_found) {
        /* Fallback: use first output */
        OrtTensorTypeAndShapeInfo *info;
        check_ort_status(api, api->GetTensorTypeAndShape(output_tensors[0], &info), "GetInfo");
        size_t ndims;
        check_ort_status(api, api->GetDimensionsCount(info, &ndims), "GetDimsCount");
        int64_t *dims = malloc(ndims * sizeof(int64_t));
        check_ort_status(api, api->GetDimensions(info, dims, ndims), "GetDims");
        emb->dims = (int)dims[ndims - 1];
        free(dims);
        api->ReleaseTensorTypeAndShapeInfo(info);
        backend->hidden_state_output_index = 0;
        backend->hidden_state_output_name = strdup(output_names[0]);
    }

    /* Cleanup */
    for (size_t i = 0; i < num_outputs; i++) {
        api->ReleaseValue(output_tensors[i]);
        check_ort_status(api, api->AllocatorFree(backend->allocator, output_names[i]), "Free");
    }
    free(output_tensors);
    free(output_names);
    api->ReleaseValue(input_ids_tensor);
    api->ReleaseValue(attn_mask_tensor);
    if (type_ids_tensor) api->ReleaseValue(type_ids_tensor);
}

/* ---- ONNX embed function ---- */

static float *onnx_embed_fn(bert_embedder_t *emb, const char *text) {
    onnx_backend_t *backend = (onnx_backend_t *)emb->backend_data;
    const OrtApi *api = backend->api;

    /* Tokenize */
    token_encoding_t *enc = tokenizer_encode(emb->tokenizer, text);
    if (!enc) return NULL;

    int seq_len = (int)enc->length;
    int64_t input_shape[2] = {1, seq_len};

    /* Convert uint32_t ids to int64_t for ONNX */
    int64_t *ids64 = malloc(seq_len * sizeof(int64_t));
    int64_t *mask64 = malloc(seq_len * sizeof(int64_t));
    int64_t *type64 = malloc(seq_len * sizeof(int64_t));
    for (int i = 0; i < seq_len; i++) {
        ids64[i] = enc->ids[i];
        mask64[i] = enc->attention_mask[i];
        type64[i] = enc->type_ids[i];
    }

    /* Create tensors */
    OrtValue *input_ids_tensor = NULL;
    OrtValue *attn_mask_tensor = NULL;
    OrtValue *type_ids_tensor = NULL;

    check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
        backend->memory_info, ids64, seq_len * sizeof(int64_t),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_ids_tensor), "CreateTensor");

    check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
        backend->memory_info, mask64, seq_len * sizeof(int64_t),
        input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &attn_mask_tensor), "CreateTensor");

    const char *input_names[3];
    OrtValue *input_tensors[3];
    size_t num_inputs = 2;
    input_names[0] = "input_ids";
    input_names[1] = "attention_mask";
    input_tensors[0] = input_ids_tensor;
    input_tensors[1] = attn_mask_tensor;

    if (backend->has_token_type_ids) {
        check_ort_status(api, api->CreateTensorWithDataAsOrtValue(
            backend->memory_info, type64, seq_len * sizeof(int64_t),
            input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &type_ids_tensor), "CreateTensor");
        input_names[2] = "token_type_ids";
        input_tensors[2] = type_ids_tensor;
        num_inputs = 3;
    }

    /* Run inference */
    const char *output_names[1] = { backend->hidden_state_output_name };
    OrtValue *output_tensor = NULL;

    check_ort_status(api, api->Run(backend->session, NULL,
        input_names, (const OrtValue *const *)input_tensors, num_inputs,
        output_names, 1, &output_tensor), "Run");

    /* Extract output: [1, seq_len, hidden] */
    float *output_data;
    check_ort_status(api, api->GetTensorMutableData(output_tensor, (void **)&output_data), "GetData");

    int hidden = emb->dims;

    /* Mean pooling */
    float *pooled = calloc(hidden, sizeof(float));
    float count = 0.0f;
    for (int s = 0; s < seq_len; s++) {
        if (enc->attention_mask[s]) {
            const float *row = output_data + s * hidden;
            for (int i = 0; i < hidden; i++) {
                pooled[i] += row[i];
            }
            count += 1.0f;
        }
    }
    if (count > 0.0f) {
        float inv = 1.0f / count;
        for (int i = 0; i < hidden; i++) pooled[i] *= inv;
    }

    /* L2 normalize */
    vec_l2_normalize(pooled, hidden);

    /* Cleanup */
    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_ids_tensor);
    api->ReleaseValue(attn_mask_tensor);
    if (type_ids_tensor) api->ReleaseValue(type_ids_tensor);
    free(ids64);
    free(mask64);
    free(type64);
    token_encoding_free(enc);

    return pooled;
}

static void onnx_free_fn(bert_embedder_t *emb) {
    onnx_backend_t *backend = (onnx_backend_t *)emb->backend_data;
    if (!backend) return;
    backend->api->ReleaseSession(backend->session);
    backend->api->ReleaseSessionOptions(backend->session_opts);
    backend->api->ReleaseMemoryInfo(backend->memory_info);
    backend->api->ReleaseEnv(backend->env);
    free(backend->hidden_state_output_name);
    free(backend);
}

/* ---- Public loader ---- */

bert_embedder_t *onnx_embedder_load(const char *model_dir,
                                     const char *model_file,
                                     const char *query_prefix,
                                     const char *doc_prefix) {
    if (!model_file) model_file = "model.onnx";

    /* Build model path */
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/%s", model_dir, model_file);

    printf("  Loading ONNX: %s\n", model_path);

    /* Init ONNX Runtime */
    const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        fprintf(stderr, "Failed to get ONNX Runtime API\n");
        return NULL;
    }

    onnx_backend_t *backend = calloc(1, sizeof(onnx_backend_t));
    backend->api = api;

    check_ort_status(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "juq", &backend->env), "CreateEnv");
    check_ort_status(api, api->CreateSessionOptions(&backend->session_opts), "CreateSessionOptions");

    /* Use all available CPU threads for intra-op parallelism */
    check_ort_status(api, api->SetIntraOpNumThreads(backend->session_opts, 0), "SetThreads");

    /* Enable graph optimizations for int8 quantized models */
    check_ort_status(api, api->SetSessionGraphOptimizationLevel(
        backend->session_opts, ORT_ENABLE_ALL), "SetOptLevel");

    check_ort_status(api, api->CreateSession(
        backend->env, model_path, backend->session_opts, &backend->session), "CreateSession");

    check_ort_status(api, api->CreateCpuMemoryInfo(
        OrtArenaAllocator, OrtMemTypeDefault, &backend->memory_info), "CreateMemInfo");

    check_ort_status(api, api->GetAllocatorWithDefaultOptions(&backend->allocator), "GetAllocator");

    /* Check if model has token_type_ids input */
    backend->has_token_type_ids = has_input_name(
        api, backend->session, backend->allocator, "token_type_ids");

    /* Load tokenizer */
    char tok_path[512];
    snprintf(tok_path, sizeof(tok_path), "%s/tokenizer.json", model_dir);
    tokenizer_t *tokenizer = tokenizer_load(tok_path);
    if (!tokenizer) {
        onnx_backend_t *b = backend;
        b->api->ReleaseSession(b->session);
        b->api->ReleaseSessionOptions(b->session_opts);
        b->api->ReleaseMemoryInfo(b->memory_info);
        b->api->ReleaseEnv(b->env);
        free(b);
        return NULL;
    }

    bert_embedder_t *emb = calloc(1, sizeof(bert_embedder_t));
    emb->backend_data = backend;
    emb->tokenizer = tokenizer;
    emb->query_prefix = strdup(query_prefix);
    emb->doc_prefix = strdup(doc_prefix);
    emb->embed_fn = onnx_embed_fn;
    emb->free_fn = onnx_free_fn;

    /* Probe to detect output name and dimensions */
    detect_output(backend, emb);

    printf("  ONNX model: hidden=%d, token_type_ids=%s, output=\"%s\"\n",
           emb->dims, backend->has_token_type_ids ? "yes" : "no",
           backend->hidden_state_output_name);

    return emb;
}
