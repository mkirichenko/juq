package dev.juq.inference;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Int8 quantized BERT model with hand-written inference.
 * Port of c/src/int8_backend.c int8_bert_model_t + int8_bert_forward().
 */
public class BertModel implements AutoCloseable {

    // Embeddings (kept as float — lookup tables don't benefit from int8)
    private final float[] wordEmbeddings;      // [vocabSize * hidden]
    private final float[] positionEmbeddings;   // [maxPos * hidden]
    private final float[] tokenTypeEmbeddings;  // [2 * hidden]
    private final float[] embNormGamma;
    private final float[] embNormBeta;

    // Transformer layers
    private final BertLayer[] layers;

    // Config
    private final int hiddenSize;
    private final int numHeads;
    private final int headDim;
    private final float layerNormEps;

    private static class BertLayer {
        Int8Linear query, key, value, attnOutput;
        float[] attnOutputNormGamma, attnOutputNormBeta;
        Int8Linear ffnUp, ffnDown;
        float[] layerOutputNormGamma, layerOutputNormBeta;
        int numHeads, headDim;
    }

    private BertModel(float[] wordEmbeddings, float[] positionEmbeddings,
                      float[] tokenTypeEmbeddings, float[] embNormGamma, float[] embNormBeta,
                      BertLayer[] layers, int hiddenSize, int numHeads, int headDim) {
        this.wordEmbeddings = wordEmbeddings;
        this.positionEmbeddings = positionEmbeddings;
        this.tokenTypeEmbeddings = tokenTypeEmbeddings;
        this.embNormGamma = embNormGamma;
        this.embNormBeta = embNormBeta;
        this.layers = layers;
        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.layerNormEps = 1e-12f;
    }

    public int hiddenSize() { return hiddenSize; }

    /**
     * Load BERT model from ONNX file.
     * Auto-detects tensor naming prefix and model configuration.
     */
    public static BertModel load(Path onnxPath) throws IOException {
        OnnxModelParser onnx = OnnxModelParser.load(onnxPath);
        return load(onnx);
    }

    static BertModel load(OnnxModelParser onnx) {
        String prefix = detectPrefix(onnx);

        // Load embeddings
        float[] wordEmb = getFloatTensor(onnx, prefix + "embeddings.word_embeddings.weight");
        float[] posEmb = getFloatTensor(onnx, prefix + "embeddings.position_embeddings.weight");
        float[] typeEmb = getFloatTensor(onnx, prefix + "embeddings.token_type_embeddings.weight");
        float[] embGamma = getFloatTensor(onnx, prefix + "embeddings.LayerNorm.weight");
        float[] embBeta = getFloatTensor(onnx, prefix + "embeddings.LayerNorm.bias");

        // Detect hidden_size from word_embeddings dims
        OnnxModelParser.TensorData weTensor = onnx.getTensor(prefix + "embeddings.word_embeddings.weight");
        int hiddenSize = (int) weTensor.dims[weTensor.dims.length - 1];

        // Detect num_layers
        int numLayers = 0;
        for (int i = 0; i < 100; i++) {
            if (onnx.getTensor(prefix + "encoder.layer." + i + ".attention.self.query.weight") != null) {
                numLayers = i + 1;
            } else {
                break;
            }
        }
        if (numLayers == 0) {
            throw new IllegalStateException("No transformer layers found in ONNX model");
        }

        // Detect num_heads
        int numHeads = hiddenSize / 64; // default
        int[] headDims = {64, 32, 48, 96, 128};
        for (int hd : headDims) {
            if (hiddenSize % hd == 0) {
                numHeads = hiddenSize / hd;
                break;
            }
        }
        int headDim = hiddenSize / numHeads;

        // Load layers
        BertLayer[] layers = new BertLayer[numLayers];
        for (int i = 0; i < numLayers; i++) {
            String lp = prefix + "encoder.layer." + i + ".";
            BertLayer layer = new BertLayer();
            layer.numHeads = numHeads;
            layer.headDim = headDim;

            layer.query = Int8Linear.fromOnnx(onnx,
                lp + "attention.self.query.weight", lp + "attention.self.query.bias",
                hiddenSize, hiddenSize);
            layer.key = Int8Linear.fromOnnx(onnx,
                lp + "attention.self.key.weight", lp + "attention.self.key.bias",
                hiddenSize, hiddenSize);
            layer.value = Int8Linear.fromOnnx(onnx,
                lp + "attention.self.value.weight", lp + "attention.self.value.bias",
                hiddenSize, hiddenSize);
            layer.attnOutput = Int8Linear.fromOnnx(onnx,
                lp + "attention.output.dense.weight", lp + "attention.output.dense.bias",
                hiddenSize, hiddenSize);

            layer.attnOutputNormGamma = getFloatTensor(onnx, lp + "attention.output.LayerNorm.weight");
            layer.attnOutputNormBeta = getFloatTensor(onnx, lp + "attention.output.LayerNorm.bias");

            // Detect intermediate size from ffn weight dims
            OnnxModelParser.TensorData ffnW = onnx.getTensor(lp + "intermediate.dense.weight");
            int intermediate = hiddenSize * 4;
            if (ffnW != null && ffnW.dims.length == 2) {
                int d0 = (int) ffnW.dims[0], d1 = (int) ffnW.dims[1];
                intermediate = Math.max(d0, d1);
            }

            layer.ffnUp = Int8Linear.fromOnnx(onnx,
                lp + "intermediate.dense.weight", lp + "intermediate.dense.bias",
                hiddenSize, intermediate);
            layer.ffnDown = Int8Linear.fromOnnx(onnx,
                lp + "output.dense.weight", lp + "output.dense.bias",
                intermediate, hiddenSize);

            layer.layerOutputNormGamma = getFloatTensor(onnx, lp + "output.LayerNorm.weight");
            layer.layerOutputNormBeta = getFloatTensor(onnx, lp + "output.LayerNorm.bias");

            layers[i] = layer;
        }

        System.out.printf("  Int8 BERT config: layers=%d, hidden=%d, heads=%d, head_dim=%d%n",
            numLayers, hiddenSize, numHeads, headDim);

        return new BertModel(wordEmb, posEmb, typeEmb, embGamma, embBeta,
            layers, hiddenSize, numHeads, headDim);
    }

    /**
     * Run BERT forward pass. Returns last hidden state as [seqLen * hiddenSize] flat array.
     */
    public float[] forward(int[] inputIds, int[] typeIds, int seqLen) {
        int H = hiddenSize;

        // 1. Embeddings (float): word + position + token_type
        float[] embeddings = new float[seqLen * H];
        for (int s = 0; s < seqLen; s++) {
            int wordOff = inputIds[s] * H;
            int posOff = s * H;
            int typeOff = typeIds[s] * H;
            int outOff = s * H;
            for (int i = 0; i < H; i++) {
                embeddings[outOff + i] = wordEmbeddings[wordOff + i]
                    + positionEmbeddings[posOff + i]
                    + tokenTypeEmbeddings[typeOff + i];
            }
        }

        // LayerNorm on embeddings
        float[] x = MathOps.layerNorm(embeddings, embNormGamma, embNormBeta,
            seqLen, H, layerNormEps);

        // 2. Transformer layers
        for (BertLayer layer : layers) {
            x = transformerLayer(layer, x, seqLen);
        }

        return x;
    }

    private float[] transformerLayer(BertLayer layer, float[] x, int seqLen) {
        int H = hiddenSize;
        int heads = layer.numHeads;
        int hd = layer.headDim;

        // Self-attention Q, K, V with int8 matmul
        float[] Q = layer.query.forward(x, seqLen);
        float[] K = layer.key.forward(x, seqLen);
        float[] V = layer.value.forward(x, seqLen);

        // Multi-head attention
        float scale = (float) (1.0 / Math.sqrt(hd));
        float[] attnOut = new float[seqLen * H];

        float[] scores = new float[seqLen * seqLen];
        for (int h = 0; h < heads; h++) {
            // Compute attention scores for this head
            for (int i = 0; i < seqLen; i++) {
                int qOff = i * H + h * hd;
                for (int j = 0; j < seqLen; j++) {
                    int kOff = j * H + h * hd;
                    scores[i * seqLen + j] = MathOps.dotProduct(Q, qOff, K, kOff, hd) * scale;
                }
            }

            // Softmax per row
            for (int i = 0; i < seqLen; i++) {
                MathOps.softmax(scores, i * seqLen, seqLen);
            }

            // Weighted sum of V
            for (int i = 0; i < seqLen; i++) {
                int outOff = i * H + h * hd;
                for (int j = 0; j < seqLen; j++) {
                    float w = scores[i * seqLen + j];
                    int vOff = j * H + h * hd;
                    for (int d = 0; d < hd; d++) {
                        attnOut[outOff + d] += w * V[vOff + d];
                    }
                }
            }
        }

        // Attention output projection
        float[] attnProj = layer.attnOutput.forward(attnOut, seqLen);

        // Residual + LayerNorm
        float[] residual1 = new float[seqLen * H];
        for (int i = 0; i < seqLen * H; i++) {
            residual1[i] = x[i] + attnProj[i];
        }
        float[] normed1 = MathOps.layerNorm(residual1, layer.attnOutputNormGamma,
            layer.attnOutputNormBeta, seqLen, H, layerNormEps);

        // FFN
        float[] ffnUp = layer.ffnUp.forward(normed1, seqLen);
        int intermediate = layer.ffnUp.outFeatures;
        MathOps.gelu(ffnUp, 0, seqLen * intermediate);

        float[] ffnDown = layer.ffnDown.forward(ffnUp, seqLen);

        // Residual + LayerNorm
        float[] residual2 = new float[seqLen * H];
        for (int i = 0; i < seqLen * H; i++) {
            residual2[i] = normed1[i] + ffnDown[i];
        }
        return MathOps.layerNorm(residual2, layer.layerOutputNormGamma,
            layer.layerOutputNormBeta, seqLen, H, layerNormEps);
    }

    // --- Helpers ---

    /**
     * Detect tensor naming prefix: "", "bert.", or "model."
     */
    private static String detectPrefix(OnnxModelParser onnx) {
        String[] prefixes = {"", "bert.", "model."};
        for (String p : prefixes) {
            if (onnx.getTensor(p + "embeddings.word_embeddings.weight") != null) {
                return p;
            }
        }
        // Try to find any tensor with "word_embeddings"
        for (String name : onnx.getAllTensors().keySet()) {
            int idx = name.indexOf("embeddings.word_embeddings");
            if (idx >= 0) {
                return name.substring(0, idx);
            }
        }
        return "";
    }

    private static float[] getFloatTensor(OnnxModelParser onnx, String name) {
        OnnxModelParser.TensorData t = onnx.getTensor(name);
        if (t == null) {
            throw new IllegalStateException("Tensor not found: " + name);
        }
        return t.toFloat();
    }

    @Override
    public void close() {
        // No native resources
    }
}
