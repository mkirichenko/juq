package dev.juq.index;

import java.util.*;

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
        PriorityQueue<Map.Entry<Integer, Float>> heap = new PriorityQueue<>(
            topK, Comparator.comparingDouble(Map.Entry::getValue)
        );

        for (int i = 0; i < vectors.size(); i++) {
            float score = dotProduct(queryVector, vectors.get(i));
            if (heap.size() < topK) {
                heap.offer(Map.entry(docIds.get(i), score));
            } else if (score > heap.peek().getValue()) {
                heap.poll();
                heap.offer(Map.entry(docIds.get(i), score));
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

    private static float dotProduct(float[] a, float[] b) {
        float sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }
}
