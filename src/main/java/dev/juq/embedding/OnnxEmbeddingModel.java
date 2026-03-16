package dev.juq.embedding;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class OnnxEmbeddingModel implements EmbeddingModel {

    private final OrtEnvironment env;
    private final OrtSession session;
    private final HuggingFaceTokenizer tokenizer;
    private final int dimensions;
    private final boolean hasTokenTypeIds;
    private final String queryPrefix;
    private final String documentPrefix;

    public OnnxEmbeddingModel(Path modelDir) throws OrtException, IOException {
        this(modelDir, "", "");
    }

    public OnnxEmbeddingModel(Path modelDir, String queryPrefix, String documentPrefix) throws OrtException, IOException {
        this.env = OrtEnvironment.getEnvironment();
        this.session = env.createSession(modelDir.resolve("model.onnx").toString());
        this.tokenizer = HuggingFaceTokenizer.newInstance(modelDir.resolve("tokenizer.json"));
        this.queryPrefix = queryPrefix;
        this.documentPrefix = documentPrefix;

        Set<String> inputNames = session.getInputNames();
        this.hasTokenTypeIds = inputNames.contains("token_type_ids");

        // Detect dimensions by running a probe embedding
        this.dimensions = detectDimensions();
    }

    private int detectDimensions() {
        try {
            Encoding[] encodings = tokenizer.batchEncode(new String[]{"probe"});
            Map<String, OnnxTensor> inputs = buildTensors(encodings, encodings[0].getIds().length);
            try (OrtSession.Result result = session.run(inputs)) {
                float[][][] output = (float[][][]) result.get(0).getValue();
                return output[0][0].length;
            } finally {
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException("Failed to detect model dimensions", e);
        }
    }

    @Override
    public float[] embed(String text) {
        return embedBatch(List.of(text))[0];
    }

    @Override
    public float[][] embedBatch(List<String> texts) {
        try {
            Encoding[] encodings = tokenizer.batchEncode(texts.toArray(new String[0]));

            int batchSize = encodings.length;
            int maxLen = 0;
            for (Encoding enc : encodings) {
                maxLen = Math.max(maxLen, (int) enc.getIds().length);
            }

            Map<String, OnnxTensor> inputs = buildTensors(encodings, maxLen);

            long[][] attentionMask = new long[batchSize][maxLen];
            for (int i = 0; i < batchSize; i++) {
                long[] mask = encodings[i].getAttentionMask();
                System.arraycopy(mask, 0, attentionMask[i], 0, mask.length);
            }

            try (OrtSession.Result result = session.run(inputs)) {
                float[][][] output = (float[][][]) result.get(0).getValue();

                float[][] embeddings = new float[batchSize][];
                for (int i = 0; i < batchSize; i++) {
                    embeddings[i] = meanPool(output[i], attentionMask[i], maxLen);
                    l2Normalize(embeddings[i]);
                }
                return embeddings;
            } finally {
                for (OnnxTensor tensor : inputs.values()) {
                    tensor.close();
                }
            }
        } catch (OrtException e) {
            throw new RuntimeException("Embedding failed", e);
        }
    }

    private Map<String, OnnxTensor> buildTensors(Encoding[] encodings, int maxLen) throws OrtException {
        int batchSize = encodings.length;
        long[][] inputIds = new long[batchSize][maxLen];
        long[][] attentionMask = new long[batchSize][maxLen];

        for (int i = 0; i < batchSize; i++) {
            long[] ids = encodings[i].getIds();
            long[] mask = encodings[i].getAttentionMask();
            System.arraycopy(ids, 0, inputIds[i], 0, ids.length);
            System.arraycopy(mask, 0, attentionMask[i], 0, mask.length);
        }

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", OnnxTensor.createTensor(env, inputIds));
        inputs.put("attention_mask", OnnxTensor.createTensor(env, attentionMask));

        if (hasTokenTypeIds) {
            long[][] tokenTypeIds = new long[batchSize][maxLen];
            for (int i = 0; i < batchSize; i++) {
                long[] types = encodings[i].getTypeIds();
                System.arraycopy(types, 0, tokenTypeIds[i], 0, types.length);
            }
            inputs.put("token_type_ids", OnnxTensor.createTensor(env, tokenTypeIds));
        }

        return inputs;
    }

    private float[] meanPool(float[][] tokenEmbeddings, long[] attentionMask, int seqLen) {
        float[] pooled = new float[dimensions];
        float maskSum = 0;

        for (int t = 0; t < seqLen; t++) {
            if (attentionMask[t] == 1) {
                maskSum++;
                for (int d = 0; d < dimensions; d++) {
                    pooled[d] += tokenEmbeddings[t][d];
                }
            }
        }

        if (maskSum > 0) {
            for (int d = 0; d < dimensions; d++) {
                pooled[d] /= maskSum;
            }
        }

        return pooled;
    }

    private void l2Normalize(float[] vector) {
        float norm = 0;
        for (float v : vector) {
            norm += v * v;
        }
        norm = (float) Math.sqrt(norm);
        if (norm > 0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public String queryPrefix() {
        return queryPrefix;
    }

    @Override
    public String documentPrefix() {
        return documentPrefix;
    }

    @Override
    public void close() throws Exception {
        session.close();
        tokenizer.close();
    }
}
