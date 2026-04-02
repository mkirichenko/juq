package dev.juq.search;

import dev.juq.embedding.EmbeddingModel;
import dev.juq.index.BruteForceIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

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
            new Document("1", "machine learning", "neural networks", Map.of(), Set.of("tech", "ai")),
            new Document("2", "cooking recipes", "healthy food", Map.of(), Set.of("lifestyle", "health")),
            new Document("3", "deep learning", "AI models", Map.of(), Set.of("tech", "ai", "premium"))
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

    // --- Tag filtering tests ---

    @Test
    void testFilterByTags_matchingDocs() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        // Only tech-tagged docs should appear
        List<SearchResult> results = engine.search("test", 10, Set.of("tech"));
        assertEquals(2, results.size());
        assertTrue(results.stream().allMatch(r -> r.document().tags().contains("tech")));
    }

    @Test
    void testFilterByTags_noMatch() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("test", 10, Set.of("nonexistent"));
        assertEquals(0, results.size());
    }

    @Test
    void testFilterByTags_multipleTagsAreUnion() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        // "tech" matches docs 1,3 and "health" matches doc 2 — union is all 3
        List<SearchResult> results = engine.search("test", 10, Set.of("tech", "health"));
        assertEquals(3, results.size());
    }

    @Test
    void testFilterByTags_nullMeansNoFilter() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("test", 10, null);
        assertEquals(3, results.size());
    }

    @Test
    void testFilterByTags_worksWithMaxSimStrategy() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.MAX_SIM);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results = engine.search("test", 10, Set.of("lifestyle"));
        assertEquals(1, results.size());
        assertEquals("2", results.get(0).document().id());
    }

    // --- Popularity ranking tests ---

    @Test
    void testPopularityBoostReranksResults() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        // Get baseline order without popularity
        List<SearchResult> baseline = engine.search("test", 3);
        String firstWithout = baseline.get(0).document().id();

        // Set huge popularity for a different doc to force reorder
        String boostTarget = baseline.get(baseline.size() - 1).document().id();
        engine.setPopularityScores(Map.of(boostTarget, 1000.0f), 1.0f, 1.0f);

        List<SearchResult> boosted = engine.search("test", 3);
        assertEquals(boostTarget, boosted.get(0).document().id(),
            "Document with high popularity should be ranked first");
    }

    @Test
    void testPopularityBoostDoesNotAffectWithoutScores() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        List<SearchResult> results1 = engine.search("test", 3);
        // Don't set any popularity scores — should return same scores
        List<SearchResult> results2 = engine.search("test", 3);

        assertEquals(results1.size(), results2.size());
        for (int i = 0; i < results1.size(); i++) {
            assertEquals(results1.get(i).document().id(), results2.get(i).document().id());
            assertEquals(results1.get(i).score(), results2.get(i).score(), 0.0001f);
        }
    }

    @Test
    void testSaturationFunction() {
        // At pivot, saturation = 0.5
        assertEquals(0.5f, DocumentSearchEngine.saturation(10.0f, 10.0f), 0.001f);
        // Zero value = zero boost
        assertEquals(0.0f, DocumentSearchEngine.saturation(0.0f, 10.0f), 0.001f);
        // Very large value approaches 1.0
        assertTrue(DocumentSearchEngine.saturation(10000.0f, 10.0f) > 0.99f);
        // Negative value = zero
        assertEquals(0.0f, DocumentSearchEngine.saturation(-5.0f, 10.0f), 0.001f);
    }

    @Test
    void testFilterAndPopularityCombined() {
        FakeEmbeddingModel model = new FakeEmbeddingModel();
        DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), PhraseStrategy.CONCATENATE);
        engine.indexDocuments(sampleDocs());

        // Boost doc "2" (lifestyle/health) but filter to "tech" only
        engine.setPopularityScores(Map.of("2", 1000.0f), 1.0f, 1.0f);

        List<SearchResult> results = engine.search("test", 10, Set.of("tech"));
        // Doc "2" should NOT appear — filtered out despite high popularity
        assertTrue(results.stream().noneMatch(r -> r.document().id().equals("2")));
        assertEquals(2, results.size());
    }
}
