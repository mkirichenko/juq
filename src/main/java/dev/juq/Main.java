package dev.juq;

import dev.juq.bench.Benchmark;
import dev.juq.data.JsonDocumentLoader;
import dev.juq.embedding.EmbeddingModel;
import dev.juq.embedding.OnnxEmbeddingModel;
import dev.juq.index.BruteForceIndex;
import dev.juq.model.Document;
import dev.juq.model.SearchResult;
import dev.juq.search.DocumentSearchEngine;
import dev.juq.search.PhraseStrategy;

import java.nio.file.Path;
import java.util.List;

public class Main {

    public static void main(String[] args) throws Exception {
        String dataPath = "data/documents.json";
        String modelPath = "model";
        String modelName = "minilm";
        String modelFile = "model.onnx";
        String query = null;
        int topK = 5;
        PhraseStrategy strategy = PhraseStrategy.CONCATENATE;
        boolean benchmark = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--data" -> dataPath = args[++i];
                case "--model" -> modelPath = args[++i];
                case "--model-name" -> modelName = args[++i].toLowerCase();
                case "--model-file" -> modelFile = args[++i];
                case "--query" -> query = args[++i];
                case "--top-k" -> topK = Integer.parseInt(args[++i]);
                case "--strategy" -> strategy = PhraseStrategy.valueOf(args[++i].toUpperCase());
                case "--benchmark" -> benchmark = true;
                default -> {
                    System.err.println("Unknown option: " + args[i]);
                    printUsage();
                    System.exit(1);
                }
            }
        }

        if (!benchmark && query == null) {
            printUsage();
            System.exit(1);
        }

        Path modelDir = Path.of(modelPath, modelName);
        System.out.printf("Loading model '%s' from %s ...%n", modelName, modelDir);
        long t0 = System.currentTimeMillis();

        try (EmbeddingModel model = createModel(modelName, modelDir, modelFile)) {
            long modelLoadMs = System.currentTimeMillis() - t0;
            System.out.printf("Model loaded in %d ms (dimensions: %d)%n", modelLoadMs, model.dimensions());

            List<Document> docs = JsonDocumentLoader.load(Path.of(dataPath));
            System.out.printf("Loaded %d documents%n", docs.size());

            if (benchmark) {
                Benchmark.run(model, docs);
            } else {
                DocumentSearchEngine engine = new DocumentSearchEngine(model, new BruteForceIndex(), strategy);

                long t1 = System.currentTimeMillis();
                engine.indexDocuments(docs);
                long indexMs = System.currentTimeMillis() - t1;
                System.out.printf("Indexed %d documents in %d ms (strategy: %s)%n",
                    docs.size(), indexMs, strategy);

                long t2 = System.currentTimeMillis();
                List<SearchResult> results = engine.search(query, topK);
                long searchMs = System.currentTimeMillis() - t2;

                System.out.printf("%nQuery: \"%s\" (took %d ms)%n", query, searchMs);
                System.out.println("─".repeat(60));
                for (int i = 0; i < results.size(); i++) {
                    SearchResult r = results.get(i);
                    System.out.printf("%d. [%.4f] %s%n", i + 1, r.score(), r.document().id());
                    System.out.printf("   phrase1: %s%n", r.document().phrase1());
                    System.out.printf("   phrase2: %s%n", r.document().phrase2());
                }
            }
        }
    }

    private static EmbeddingModel createModel(String modelName, Path modelDir, String modelFile) throws Exception {
        return switch (modelName) {
            case "minilm" -> new OnnxEmbeddingModel(modelDir, modelFile, "", "");
            case "e5-small" -> new OnnxEmbeddingModel(modelDir, modelFile, "query: ", "passage: ");
            case "berta" -> new OnnxEmbeddingModel(modelDir, modelFile, "search_query: ", "search_document: ");
            default -> throw new IllegalArgumentException(
                "Unknown model: " + modelName + ". Available: minilm, e5-small, berta");
        };
    }

    private static void printUsage() {
        System.err.println("""
            Usage: juq [options]
              --data <path>         Path to documents.json (default: data/documents.json)
              --model <path>        Base model directory (default: model/)
              --model-name <name>   Model to use: minilm, e5-small, berta (default: minilm)
              --model-file <name>   ONNX model filename (default: model.onnx)
              --query <text>        Search query
              --top-k <n>           Number of results (default: 5)
              --strategy <name>     CONCATENATE|AVERAGE|MAX_SIM (default: CONCATENATE)
              --benchmark           Run benchmark suite
            """);
    }
}
