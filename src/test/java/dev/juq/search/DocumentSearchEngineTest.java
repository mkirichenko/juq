package dev.juq.search;

import dev.juq.embedding.EmbeddingModel;
import dev.juq.index.BruteForceIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class DocumentSearchEngineTest {

    /**
     * A fake embedding model that returns a deterministic vector based on text hash.
     * Used to test search engine logic without requiring the ONNX model.
     */
    static class FakeEmbeddingModel implements EmbeddingModel {
        private static final int DIMS = 8;

        @Override
        public float[] embed(String text) {
            float[] vec = new float[DIMS];
            int hash = text.hashCode();
            for (int i = 0; i < DIMS; i++) {
                vec[i] = ((hash >> (i * 4)) & 0xF) - 8.0f;
            }
            // L2 normalize
            float norm = 0;
            for (float v : vec) norm += v * v;
            norm = (float) Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < DIMS; i++) vec[i] /= norm;
            }
            return vec;
        }

        @Override
        public float[][] embedBatch(List<String> texts) {
            return texts.stream().map(this::embed).toArray(float[][]::new);
        }

        @Override
        public int dimensions() { return DIMS; }

        @Override
        public void close() {}
    }

    private List<Document> sampleDocs() {
        return List.of(
            new Document("1", "machine learning", "neural networks", Map.of()),
            new Document("2", "cooking recipes", "healthy food", Map.of()),
            new Document("3", "deep learning", "AI models", Map.of())
        );
    }

    @Test
    void testConcatenateStrategy() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("machine learning", 2);
        assertEquals(2, results.size());
        assertNotNull(results.get(0).document());
        assertTrue(results.get(0).score() >= results.get(1).score());
    }

    @Test
    void testAverageStrategy() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.AVERAGE);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("test query", 3);
        assertEquals(3, results.size());
        assertEquals(3, engine.documentCount());
    }

    @Test
    void testMaxSimStrategy() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.MAX_SIM);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("neural networks", 3);
        assertEquals(3, results.size());
        // Verify deduplication — should not have more results than documents
        long uniqueDocs = results.stream().map(r -> r.document().id()).distinct().count();
        assertEquals(results.size(), uniqueDocs);
    }

    @Test
    void testSearchResultsContainDocumentData() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("test", 1);
        assertEquals(1, results.size());
        assertNotNull(results.get(0).document().id());
        assertNotNull(results.get(0).document().phrase1());
        assertNotNull(results.get(0).document().phrase2());
    }
}
