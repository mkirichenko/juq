package dev.juq.bench;

import dev.juq.embedding.EmbeddingModel;
import dev.juq.index.BruteForceIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;
import dev.juq.search.DocumentSearchEngine;
import dev.juq.search.PhraseStrategy;

import java.util.Arrays;
import java.util.List;

public class Benchmark {

    private static final String[] SAMPLE_QUERIES = {
        "machine learning algorithms",
        "climate change impact",
        "software engineering best practices",
        "healthy cooking recipes",
        "space exploration missions",
        "financial market analysis",
        "artificial intelligence ethics",
        "renewable energy sources",
        "modern web development",
        "quantum computing applications"
    };

    private static final int WARMUP_QUERIES = 3;
    private static final int TIMED_QUERIES = SAMPLE_QUERIES.length;

    public static void run(EmbeddingModel model, List<Document> docs) {
        System.out.println("\n=== BENCHMARK ===\n");

        for (PhraseStrategy strategy : PhraseStrategy.values()) {
            System.out.printf("--- Strategy: %s ---%n", strategy);

            BruteForceIndex index = new BruteForceIndex();
            DocumentSearchEngine engine = new DocumentSearchEngine(model, index, strategy);

            // Index
            long t0 = System.currentTimeMillis();
            engine.indexDocuments(docs);
            long indexMs = System.currentTimeMillis() - t0;
            System.out.printf("  Indexing: %d docs in %d ms (%.1f ms/doc)%n",
                docs.size(), indexMs, (double) indexMs / docs.size());
            System.out.printf("  Index vectors: %d%n", index.size());

            // Warmup
            for (int i = 0; i < WARMUP_QUERIES; i++) {
                engine.search(SAMPLE_QUERIES[i % SAMPLE_QUERIES.length], 5);
            }

            // Timed search
            long[] latencies = new long[TIMED_QUERIES];
            for (int i = 0; i < TIMED_QUERIES; i++) {
                long t1 = System.nanoTime();
                engine.search(SAMPLE_QUERIES[i], 5);
                latencies[i] = (System.nanoTime() - t1) / 1_000_000;
            }

            Arrays.sort(latencies);
            System.out.printf("  Search latency (ms): min=%d, p50=%d, p95=%d, max=%d%n",
                latencies[0],
                latencies[latencies.length / 2],
                latencies[(int) (latencies.length * 0.95)],
                latencies[latencies.length - 1]);

            // Show sample result
            List<SearchResult> sample = engine.search(SAMPLE_QUERIES[0], 3);
            System.out.printf("  Sample query: \"%s\"%n", SAMPLE_QUERIES[0]);
            for (int i = 0; i < sample.size(); i++) {
                System.out.printf("    %d. [%.4f] %s%n", i + 1, sample.get(i).score(), sample.get(i).document().id());
            }
            System.out.println();
        }
    }
}
