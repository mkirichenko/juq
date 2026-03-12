package dev.juq.index;

import java.util.List;
import java.util.Map;

public interface VectorIndex {

    void add(int docId, float[] vector);

    List<Map.Entry<Integer, Float>> search(float[] queryVector, int topK);

    int size();
}
