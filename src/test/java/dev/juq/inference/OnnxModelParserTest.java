package dev.juq.inference;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

class OnnxModelParserTest {

    private static final Path MODEL_PATH = Path.of("model", "minilm", "model.onnx");

    static boolean modelExists() {
        return Files.exists(MODEL_PATH);
    }

    @Test
    @EnabledIf("modelExists")
    void testLoadModel() throws IOException {
        OnnxModelParser parser = OnnxModelParser.load(MODEL_PATH);

        // MiniLM-L6 should have many tensors (embeddings + 6 layers * ~16 tensors each)
        assertTrue(parser.tensorCount() > 50,
            "Expected >50 tensors, got " + parser.tensorCount());
    }

    @Test
    @EnabledIf("modelExists")
    void testWordEmbeddingsShape() throws IOException {
        OnnxModelParser parser = OnnxModelParser.load(MODEL_PATH);

        // Try common prefixes
        OnnxModelParser.TensorData we = parser.getTensor("embeddings.word_embeddings.weight");
        if (we == null) we = parser.getTensor("bert.embeddings.word_embeddings.weight");

        assertNotNull(we, "word_embeddings tensor should exist");
        assertEquals(2, we.dims.length, "word_embeddings should be 2D");
        assertTrue(we.dims[0] > 10000, "vocab size should be >10k");
        assertEquals(384, we.dims[1], "MiniLM hidden size should be 384");
    }

    @Test
    @EnabledIf("modelExists")
    void testTensorDataConversion() throws IOException {
        OnnxModelParser parser = OnnxModelParser.load(MODEL_PATH);

        // Find any float tensor
        for (OnnxModelParser.TensorData t : parser.getAllTensors().values()) {
            if (t.dataType == OnnxModelParser.ONNX_FLOAT && t.numElements < 1000) {
                float[] data = t.toFloat();
                assertEquals(t.numElements, data.length);
                // At least some values should be non-zero
                boolean hasNonZero = false;
                for (float v : data) {
                    if (v != 0) { hasNonZero = true; break; }
                }
                assertTrue(hasNonZero, "Tensor data should contain non-zero values");
                return;
            }
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testLayerTensorsExist() throws IOException {
        OnnxModelParser parser = OnnxModelParser.load(MODEL_PATH);

        // Detect prefix
        String prefix = "";
        if (parser.getTensor("bert.embeddings.word_embeddings.weight") != null) {
            prefix = "bert.";
        }

        // ONNX exports preserve bias names but typically rename matmul weights
        // to anonymous initializers. Check the bias for layer existence and the
        // graph walker for the corresponding weight tensor.
        for (int i = 0; i < 6; i++) {
            String biasName = prefix + "encoder.layer." + i + ".attention.self.query.bias";
            assertNotNull(parser.getTensor(biasName),
                "Layer " + i + " query bias should exist: " + biasName);

            String resolved = parser.resolveWeightFromBias(biasName, 384, 384);
            assertNotNull(resolved,
                "Layer " + i + " query weight should be resolvable from bias: " + biasName);
            assertNotNull(parser.getTensor(resolved),
                "Resolved weight tensor should exist: " + resolved);
        }

        // Layer 6 should NOT exist
        assertNull(parser.getTensor(prefix + "encoder.layer.6.attention.self.query.bias"));
    }
}
