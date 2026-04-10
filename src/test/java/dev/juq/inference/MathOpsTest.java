package dev.juq.inference;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MathOpsTest {

    @Test
    void testDotProduct() {
        float[] a = {1.0f, 2.0f, 3.0f};
        float[] b = {4.0f, 5.0f, 6.0f};
        // 1*4 + 2*5 + 3*6 = 32
        assertEquals(32.0f, MathOps.dotProduct(a, 0, b, 0, 3), 1e-6f);
    }

    @Test
    void testDotProductWithOffset() {
        float[] a = {0.0f, 1.0f, 2.0f, 3.0f};
        float[] b = {0.0f, 4.0f, 5.0f, 6.0f};
        assertEquals(32.0f, MathOps.dotProduct(a, 1, b, 1, 3), 1e-6f);
    }

    @Test
    void testL2Normalize() {
        float[] v = {3.0f, 4.0f};
        MathOps.l2Normalize(v);
        assertEquals(0.6f, v[0], 1e-6f);
        assertEquals(0.8f, v[1], 1e-6f);

        // Check norm is 1
        float norm = (float) Math.sqrt(v[0] * v[0] + v[1] * v[1]);
        assertEquals(1.0f, norm, 1e-6f);
    }

    @Test
    void testL2NormalizeZeroVector() {
        float[] v = {0.0f, 0.0f, 0.0f};
        MathOps.l2Normalize(v);
        assertEquals(0.0f, v[0]);
        assertEquals(0.0f, v[1]);
        assertEquals(0.0f, v[2]);
    }

    @Test
    void testSoftmax() {
        float[] v = {1.0f, 2.0f, 3.0f};
        MathOps.softmax(v, 0, 3);

        // Sum should be 1
        float sum = v[0] + v[1] + v[2];
        assertEquals(1.0f, sum, 1e-6f);

        // Should be monotonically increasing
        assertTrue(v[0] < v[1]);
        assertTrue(v[1] < v[2]);
    }

    @Test
    void testSoftmaxWithOffset() {
        float[] v = {999.0f, 1.0f, 2.0f, 3.0f, 999.0f};
        MathOps.softmax(v, 1, 3);

        // Only elements 1..3 should sum to 1
        float sum = v[1] + v[2] + v[3];
        assertEquals(1.0f, sum, 1e-6f);
        // Elements outside range should be untouched
        assertEquals(999.0f, v[0]);
        assertEquals(999.0f, v[4]);
    }

    @Test
    void testGelu() {
        float[] v = {0.0f, 1.0f, -1.0f};
        MathOps.gelu(v, 0, 3);

        // gelu(0) ≈ 0
        assertEquals(0.0f, v[0], 1e-4f);
        // gelu(1) ≈ 0.8412
        assertEquals(0.8412f, v[1], 0.01f);
        // gelu(-1) ≈ -0.1588
        assertEquals(-0.1588f, v[2], 0.01f);
    }

    @Test
    void testLayerNorm() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f}; // 2 positions x 2 hidden
        float[] gamma = {1.0f, 1.0f};
        float[] beta = {0.0f, 0.0f};
        float eps = 1e-5f;

        float[] output = MathOps.layerNorm(input, gamma, beta, 2, 2, eps);

        // Each position normalized: mean=1.5, var=0.25 -> [-1, 1] * gamma + beta
        assertEquals(-1.0f, output[0], 0.01f);
        assertEquals(1.0f, output[1], 0.01f);
        assertEquals(-1.0f, output[2], 0.01f);
        assertEquals(1.0f, output[3], 0.01f);
    }

    @Test
    void testLayerNormWithGammaBeta() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] gamma = {2.0f, 0.5f};
        float[] beta = {1.0f, -1.0f};
        float eps = 1e-5f;

        float[] output = MathOps.layerNorm(input, gamma, beta, 2, 2, eps);

        // Position 0: normalized = [-1, 1], then gamma*normalized + beta
        assertEquals(-2.0f + 1.0f, output[0], 0.01f); // 2*(-1) + 1 = -1
        assertEquals(0.5f - 1.0f, output[1], 0.01f);   // 0.5*(1) + (-1) = -0.5
    }

    @Test
    void testMeanPool() {
        // 3 tokens, hidden=2
        float[] hidden = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        int[] mask = {1, 1, 0}; // only first 2 tokens active

        float[] pooled = MathOps.meanPool(hidden, mask, 3, 2);

        assertEquals(2.0f, pooled[0], 1e-6f); // (1+3)/2
        assertEquals(3.0f, pooled[1], 1e-6f); // (2+4)/2
    }

    @Test
    void testSymmetricQuantize() {
        float[] input = {-1.0f, 0.0f, 0.5f, 1.0f};
        byte[] output = new byte[4];
        float scale = MathOps.symmetricQuantize(input, 0, 4, output);

        // absMax = 1.0, scale = 1/127
        assertEquals(1.0f / 127.0f, scale, 1e-6f);
        assertEquals(-127, output[0]);
        assertEquals(0, output[1]);
        assertEquals(64, output[2]); // round(0.5 * 127) = 64
        assertEquals(127, output[3]);
    }

    @Test
    void testDotProductI8() {
        byte[] a = {1, 2, 3};
        byte[] b = {4, 5, 6};
        assertEquals(32, MathOps.dotProductI8(a, 0, b, 0, 3));
    }

    @Test
    void testMatVecMulI8() {
        // 2x3 matrix * 3-vector
        byte[] mat = {1, 0, 0, 0, 1, 0}; // identity-like
        byte[] vec = {10, 20, 30};
        int[] out = new int[2];

        MathOps.matVecMulI8(mat, vec, out, 2, 3);

        assertEquals(10, out[0]); // 1*10 + 0*20 + 0*30
        assertEquals(20, out[1]); // 0*10 + 1*20 + 0*30
    }
}
