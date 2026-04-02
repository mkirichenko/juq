package dev.juq.search;

import dev.juq.embedding.EmbeddingModel;
import dev.juq.index.VectorIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;

import java.util.*;
import java.util.function.IntPredicate;

public class DocumentSearchEngine {

    private final EmbeddingModel model;
    private final VectorIndex index;
    private final PhraseStrategy strategy;
    private final Map<Integer, Document> documents = new HashMap<>();
    private int nextId = 0;

    /** Popularity scores by document id (from external tracking system). */
    private Map<String, Float> popularityScores = Map.of();
    private float popularityWeight = 0.2f;
    private float popularityPivot = 10.0f;

    public DocumentSearchEngine(EmbeddingModel model, VectorIndex index, PhraseStrategy strategy) {
        this.model = model;
        this.index = index;
        this.strategy = strategy;
    }

    /**
     * Set popularity scores from an external tracking system.
     * Works like Elasticsearch's rank_feature — higher values boost documents
     * in search results using a saturation function: score += weight * (value / (value + pivot)).
     *
     * @param scores     map of document id to popularity value (e.g. view count, click count)
     * @param weight     how much popularity affects the final score (default 0.2)
     * @param pivot      the value at which the saturation function returns 0.5 (default 10.0)
     */
    public void setPopularityScores(Map<String, Float> scores, float weight, float pivot) {
        this.popularityScores = scores;
        this.popularityWeight = weight;
        this.popularityPivot = pivot;
    }

    public void setPopularityScores(Map<String, Float> scores) {
        setPopularityScores(scores, 0.2f, 10.0f);
    }

    public void indexDocuments(List<Document> docs) {
        for (Document doc : docs) {
            int docId = nextId++;
            documents.put(docId, doc);

            String prefix = model.documentPrefix();
            switch (strategy) {
                case CONCATENATE -> {
                    String combined = prefix + doc.phrase1() + ". " + doc.phrase2();
                    float[] vector = model.embed(combined);
                    index.add(docId, vector);
                }
                case AVERAGE -> {
                    float[] v1 = model.embed(prefix + doc.phrase1());
                    float[] v2 = model.embed(prefix + doc.phrase2());
                    float[] avg = average(v1, v2);
                    l2Normalize(avg);
                    index.add(docId, avg);
                }
                case MAX_SIM -> {
                    float[] v1 = model.embed(prefix + doc.phrase1());
                    float[] v2 = model.embed(prefix + doc.phrase2());
                    index.add(docId, v1);
                    index.add(docId, v2);
                }
            }
        }
    }

    /**
     * Search without tag filtering.
     */
    public List<SearchResult> search(String query, int topK) {
        return search(query, topK, null);
    }

    /**
     * Search with tag-based access filtering and popularity ranking.
     *
     * @param query       the search query text
     * @param topK        maximum number of results to return
     * @param allowedTags if non-null, only documents having at least one tag in common
     *                    with this set are returned (like a user's JWT-derived permissions)
     */
    public List<SearchResult> search(String query, int topK, Set<String> allowedTags) {
        float[] queryVector = model.embed(model.queryPrefix() + query);

        IntPredicate tagFilter = buildTagFilter(allowedTags);

        List<Map.Entry<Integer, Float>> raw;
        if (strategy == PhraseStrategy.MAX_SIM) {
            raw = index.search(queryVector, topK * 2, tagFilter);
            Map<Integer, Float> bestScores = new LinkedHashMap<>();
            for (Map.Entry<Integer, Float> entry : raw) {
                bestScores.merge(entry.getKey(), entry.getValue(), Math::max);
            }
            raw = bestScores.entrySet().stream()
                .sorted((a, b) -> Float.compare(b.getValue(), a.getValue()))
                .limit(topK)
                .toList();
        } else {
            raw = index.search(queryVector, topK, tagFilter);
        }

        if (popularityScores.isEmpty()) {
            return raw.stream()
                .map(e -> new SearchResult(documents.get(e.getKey()), e.getValue()))
                .toList();
        }

        // Re-rank with popularity boost using saturation function
        return raw.stream()
            .map(e -> {
                Document doc = documents.get(e.getKey());
                float vectorScore = e.getValue();
                float popularity = popularityScores.getOrDefault(doc.id(), 0.0f);
                float boost = popularityWeight * saturation(popularity, popularityPivot);
                return new SearchResult(doc, vectorScore + boost);
            })
            .sorted(Comparator.reverseOrder())
            .limit(topK)
            .toList();
    }

    /**
     * Saturation function (same as Elasticsearch rank_feature saturation).
     * Returns a value in [0, 1) that grows quickly for small inputs and
     * saturates for large ones. At value == pivot, returns 0.5.
     */
    static float saturation(float value, float pivot) {
        if (value <= 0) return 0;
        return value / (value + pivot);
    }

    public int documentCount() {
        return documents.size();
    }

    private IntPredicate buildTagFilter(Set<String> allowedTags) {
        if (allowedTags == null || allowedTags.isEmpty()) return null;
        return docId -> {
            Document doc = documents.get(docId);
            if (doc == null) return false;
            Set<String> docTags = doc.tags();
            if (docTags.isEmpty()) return false;
            for (String tag : docTags) {
                if (allowedTags.contains(tag)) return true;
            }
            return false;
        };
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
