package dev.juq.index;

import java.util.List;
import java.util.Map;
import java.util.function.IntPredicate;

public interface VectorIndex {

    void add(int docId, float[] vector);

    List<Map.Entry<Integer, Float>> search(float[] queryVector, int topK);

    /**
     * Search with a pre-filter. Only entries where filter.test(docId) returns true
     * are considered as candidates. This avoids scanning results that will be discarded.
     */
    List<Map.Entry<Integer, Float>> search(float[] queryVector, int topK, IntPredicate filter);

    int size();
}
