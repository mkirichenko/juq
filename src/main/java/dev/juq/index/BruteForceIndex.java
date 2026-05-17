package dev.juq.index;

import dev.juq.inference.MathOps;

import java.util.*;
import java.util.function.IntPredicate;

public class BruteForceIndex implements VectorIndex {

    private final List<Integer> docIds = new ArrayList<>();
    private final List<float[]> vectors = new ArrayList<>();

    @Override
    public void add(int docId, float[] vector) {
        docIds.add(docId);
        vectors.add(vector);
    }

    @Override
    public List<Map.Entry<Integer, Float>> search(float[] queryVector, int topK) {
        return search(queryVector, topK, null);
    }

    @Override
    public List<Map.Entry<Integer, Float>> search(float[] queryVector, int topK, IntPredicate filter) {
        PriorityQueue<Map.Entry<Integer, Float>> heap = new PriorityQueue<>(
            topK, Comparator.comparingDouble(Map.Entry::getValue)
        );

        for (int i = 0; i < vectors.size(); i++) {
            int docId = docIds.get(i);
            if (filter != null && !filter.test(docId)) continue;

            float[] v = vectors.get(i);
            float score = MathOps.dotProduct(queryVector, 0, v, 0, v.length);
            if (heap.size() < topK) {
                heap.offer(Map.entry(docId, score));
            } else if (score > heap.peek().getValue()) {
                heap.poll();
                heap.offer(Map.entry(docId, score));
            }
        }

        List<Map.Entry<Integer, Float>> results = new ArrayList<>(heap);
        results.sort((a, b) -> Float.compare(b.getValue(), a.getValue()));
        return results;
    }

    @Override
    public int size() {
        return vectors.size();
    }
}
