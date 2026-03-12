package dev.juq.index;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BruteForceIndexTest {

    @Test
    void testAddAndSize() {
        BruteForceIndex index = new BruteForceIndex();
        assertEquals(0, index.size());

        index.add(0, new float[]{1.0f, 0.0f, 0.0f});
        index.add(1, new float[]{0.0f, 1.0f, 0.0f});
        assertEquals(2, index.size());
    }

    @Test
    void testSearchReturnsMostSimilar() {
        BruteForceIndex index = new BruteForceIndex();
        // Normalized vectors
        index.add(0, new float[]{1.0f, 0.0f, 0.0f});
        index.add(1, new float[]{0.0f, 1.0f, 0.0f});
        index.add(2, new float[]{0.707f, 0.707f, 0.0f}); // 45 degrees between 0 and 1

        // Query close to doc 0
        List<Map.Entry<Integer, Float>> results = index.search(new float[]{0.9f, 0.1f, 0.0f}, 2);

        assertEquals(2, results.size());
        assertEquals(0, results.get(0).getKey()); // most similar
        assertTrue(results.get(0).getValue() > results.get(1).getValue());
    }

    @Test
    void testTopKLimitsResults() {
        BruteForceIndex index = new BruteForceIndex();
        for (int i = 0; i < 10; i++) {
            float[] v = new float[3];
            v[i % 3] = 1.0f;
            index.add(i, v);
        }

        List<Map.Entry<Integer, Float>> results = index.search(new float[]{1.0f, 0.0f, 0.0f}, 3);
        assertEquals(3, results.size());
    }

    @Test
    void testResultsOrderedByScoreDescending() {
        BruteForceIndex index = new BruteForceIndex();
        index.add(0, new float[]{1.0f, 0.0f});
        index.add(1, new float[]{0.5f, 0.866f});
        index.add(2, new float[]{0.0f, 1.0f});

        List<Map.Entry<Integer, Float>> results = index.search(new float[]{1.0f, 0.0f}, 3);

        for (int i = 0; i < results.size() - 1; i++) {
            assertTrue(results.get(i).getValue() >= results.get(i + 1).getValue());
        }
    }
}
