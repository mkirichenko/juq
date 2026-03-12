package dev.juq.embedding;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests that require the ONNX model to be downloaded.
 * Run ./scripts/download-model.sh first.
 */
class OnnxEmbeddingModelTest {

    private static final Path MODEL_DIR = Path.of("model");

    static boolean modelExists() {
        return Files.exists(MODEL_DIR.resolve("model.onnx"))
            && Files.exists(MODEL_DIR.resolve("tokenizer.json"));
    }

    @Test
    @EnabledIf("modelExists")
    void testEmbedProducesCorrectDimensions() throws Exception {
        try (OnnxEmbeddingModel model = new OnnxEmbeddingModel(MODEL_DIR)) {
            float[] embedding = model.embed("Hello world");
            assertEquals(384, embedding.length);
            assertEquals(384, model.dimensions());
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEmbeddingIsNormalized() throws Exception {
        try (OnnxEmbeddingModel model = new OnnxEmbeddingModel(MODEL_DIR)) {
            float[] embedding = model.embed("Test sentence for normalization");
            float norm = 0;
            for (float v : embedding) norm += v * v;
            assertEquals(1.0f, (float) Math.sqrt(norm), 0.01f);
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testSimilarTextsHaveHigherSimilarity() throws Exception {
        try (OnnxEmbeddingModel model = new OnnxEmbeddingModel(MODEL_DIR)) {
            float[] cat = model.embed("The cat sat on the mat");
            float[] dog = model.embed("The dog sat on the rug");
            float[] car = model.embed("Quantum physics explains particle behavior");

            float catDog = dotProduct(cat, dog);
            float catCar = dotProduct(cat, car);

            assertTrue(catDog > catCar,
                "cat-dog similarity (%.4f) should be higher than cat-physics (%.4f)"
                    .formatted(catDog, catCar));
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testBatchEmbedding() throws Exception {
        try (OnnxEmbeddingModel model = new OnnxEmbeddingModel(MODEL_DIR)) {
            float[][] embeddings = model.embedBatch(List.of("Hello", "World", "Test"));
            assertEquals(3, embeddings.length);
            for (float[] emb : embeddings) {
                assertEquals(384, emb.length);
            }
        }
    }

    private static float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}
