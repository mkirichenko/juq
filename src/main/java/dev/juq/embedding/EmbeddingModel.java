package dev.juq.embedding;

import java.util.List;

public interface EmbeddingModel extends AutoCloseable {

    float[] embed(String text);

    float[][] embedBatch(List<String> texts);

    int dimensions();
}
