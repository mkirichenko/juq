package dev.juq.embedding;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

class PureJavaEmbeddingModelTest {

    private static final Path MODEL_DIR = Path.of("model", "minilm");

    static boolean modelExists() {
        return Files.exists(MODEL_DIR.resolve("model.onnx"))
            && Files.exists(MODEL_DIR.resolve("tokenizer.json"));
    }

    @Test
    @EnabledIf("modelExists")
    void testDimensions() throws IOException {
        try (PureJavaEmbeddingModel model = new PureJavaEmbeddingModel(MODEL_DIR)) {
            assertEquals(384, model.dimensions(), "MiniLM should have 384 dimensions");
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEmbedProducesUnitVector() throws IOException {
        try (PureJavaEmbeddingModel model = new PureJavaEmbeddingModel(MODEL_DIR)) {
            float[] emb = model.embed("hello world");

            assertEquals(384, emb.length);

            // Check L2 norm is approximately 1.0
            float norm = 0;
            for (float v : emb) norm += v * v;
            norm = (float) Math.sqrt(norm);
            assertEquals(1.0f, norm, 0.01f, "Embedding should be L2-normalized");
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testSimilarTextsSimilarEmbeddings() throws IOException {
        try (PureJavaEmbeddingModel model = new PureJavaEmbeddingModel(MODEL_DIR)) {
            float[] emb1 = model.embed("the cat sat on the mat");
            float[] emb2 = model.embed("a cat was sitting on a mat");
            float[] emb3 = model.embed("quantum mechanics and particle physics");

            float simSimilar = cosineSim(emb1, emb2);
            float simDifferent = cosineSim(emb1, emb3);

            // Similar sentences should have higher similarity than dissimilar ones
            assertTrue(simSimilar > simDifferent,
                "Similar texts should have higher cosine similarity: " +
                simSimilar + " vs " + simDifferent);
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testDifferentTextsProduceDifferentEmbeddings() throws IOException {
        try (PureJavaEmbeddingModel model = new PureJavaEmbeddingModel(MODEL_DIR)) {
            float[] emb1 = model.embed("artificial intelligence");
            float[] emb2 = model.embed("cooking recipes");

            // Should not be identical
            boolean different = false;
            for (int i = 0; i < emb1.length; i++) {
                if (Math.abs(emb1[i] - emb2[i]) > 1e-6f) {
                    different = true;
                    break;
                }
            }
            assertTrue(different, "Different texts should produce different embeddings");
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testBatchEmbed() throws IOException {
        try (PureJavaEmbeddingModel model = new PureJavaEmbeddingModel(MODEL_DIR)) {
            float[][] batch = model.embedBatch(java.util.List.of("hello", "world"));

            assertEquals(2, batch.length);
            assertEquals(384, batch[0].length);
            assertEquals(384, batch[1].length);
        }
    }

    private static float cosineSim(float[] a, float[] b) {
        float dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }
}
