package dev.juq.bench;

import dev.juq.embedding.EmbeddingModel;
import dev.juq.index.BruteForceIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;
import dev.juq.search.DocumentSearchEngine;
import dev.juq.search.PhraseStrategy;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class Benchmark {

    private static final String[] SAMPLE_QUERIES_EN = {
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
    private static final String[] SAMPLE_QUERIES_RU = {
        "алгоритмы машинного обучения",
        "последствия изменения климата",
        "лучшие практики разработки программного обеспечения",
        "рецепты здорового питания",
        "миссии по исследованию космоса",
        "анализ финансовых рынков",
        "этика искусственного интеллекта",
        "возобновляемые источники энергии",
        "современная веб-разработка",
        "применение квантовых вычислений"
    };

    private static final int WARMUP_QUERIES = 3;

    public static void run(EmbeddingModel model, List<Document> docs, String dataPath) {
        String[] queries = pickQueries(dataPath);
        int timedQueries = queries.length;
        System.out.println("\n=== BENCHMARK ===\n");
        System.out.printf("Query language: %s%n", queries == SAMPLE_QUERIES_RU ? "ru" : "en");

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
                engine.search(queries[i % queries.length], 5);
            }

            // Timed search
            long[] latencies = new long[timedQueries];
            for (int i = 0; i < timedQueries; i++) {
                long t1 = System.nanoTime();
                engine.search(queries[i], 5);
                latencies[i] = (System.nanoTime() - t1) / 1_000_000;
            }

            Arrays.sort(latencies);
            System.out.printf("  Search latency (ms): min=%d, p50=%d, p95=%d, max=%d%n",
                latencies[0],
                latencies[latencies.length / 2],
                latencies[(int) (latencies.length * 0.95)],
                latencies[latencies.length - 1]);

            // Show sample result
            List<SearchResult> sample = engine.search(queries[0], 3);
            System.out.printf("  Sample query: \"%s\"%n", queries[0]);
            for (int i = 0; i < sample.size(); i++) {
                System.out.printf("    %d. [%.4f] %s%n", i + 1, sample.get(i).score(), sample.get(i).document().id());
            }
            System.out.println();
        }
    }

    private static String[] pickQueries(String dataPath) {
        return dataPath.endsWith("-ru.json") ? SAMPLE_QUERIES_RU : SAMPLE_QUERIES_EN;
    }
}
