package dev.juq.search;

import dev.juq.embedding.EmbeddingModel;
import dev.juq.index.VectorIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;

import java.util.*;

public class DocumentSearchEngine {

    private final EmbeddingModel model;
    private final VectorIndex index;
    private final PhraseStrategy strategy;
    private final Map<Integer, Document> documents = new HashMap<>();
    private int nextId = 0;

    public DocumentSearchEngine(EmbeddingModel model, VectorIndex index, PhraseStrategy strategy) {
        this.model = model;
        this.index = index;
        this.strategy = strategy;
    }

    public void indexDocuments(List<Document> docs) {
        for (Document doc : docs) {
            int docId = nextId++;
            documents.put(docId, doc);

            switch (strategy) {
                case CONCATENATE -> {
                    String combined = doc.phrase1() + ". " + doc.phrase2();
                    float[] vector = model.embed(combined);
                    index.add(docId, vector);
                }
                case AVERAGE -> {
                    float[] v1 = model.embed(doc.phrase1());
                    float[] v2 = model.embed(doc.phrase2());
                    float[] avg = average(v1, v2);
                    l2Normalize(avg);
                    index.add(docId, avg);
                }
                case MAX_SIM -> {
                    float[] v1 = model.embed(doc.phrase1());
                    float[] v2 = model.embed(doc.phrase2());
                    index.add(docId, v1);
                    index.add(docId, v2);
                }
            }
        }
    }

    public List<SearchResult> search(String query, int topK) {
        float[] queryVector = model.embed(query);

        if (strategy == PhraseStrategy.MAX_SIM) {
            // Fetch more results since there are 2 entries per doc, then deduplicate
            List<Map.Entry<Integer, Float>> raw = index.search(queryVector, topK * 2);
            Map<Integer, Float> bestScores = new LinkedHashMap<>();
            for (Map.Entry<Integer, Float> entry : raw) {
                bestScores.merge(entry.getKey(), entry.getValue(), Math::max);
            }
            return bestScores.entrySet().stream()
                .sorted((a, b) -> Float.compare(b.getValue(), a.getValue()))
                .limit(topK)
                .map(e -> new SearchResult(documents.get(e.getKey()), e.getValue()))
                .toList();
        }

        List<Map.Entry<Integer, Float>> raw = index.search(queryVector, topK);
        return raw.stream()
            .map(e -> new SearchResult(documents.get(e.getKey()), e.getValue()))
            .toList();
    }

    public int documentCount() {
        return documents.size();
    }

    private static float[] average(float[] a, float[] b) {
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            result[i] = (a[i] + b[i]) / 2.0f;
        }
        return result;
    }

    private static void l2Normalize(float[] vector) {
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
}
