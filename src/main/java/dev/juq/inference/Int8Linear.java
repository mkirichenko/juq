package dev.juq.inference;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Int8 quantized linear layer.
 * Weights stored as int8 with per-tensor symmetric quantization.
 * Bias stored as float.
 * Port of int8_linear_t from c/src/int8_backend.c.
 */
public class Int8Linear {

    final byte[] weight;     // [outFeatures x inFeatures] int8
    final float[] bias;      // [outFeatures]
    final float weightScale; // quantization scale
    final int inFeatures;
    final int outFeatures;

    public Int8Linear(byte[] weight, float[] bias, float weightScale, int inFeatures, int outFeatures) {
        this.weight = weight;
        this.bias = bias;
        this.weightScale = weightScale;
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }

    /**
     * Load an Int8Linear layer from ONNX tensors.
     * Handles both pre-quantized INT8 weights and float weights (quantized at load time).
     * Detects and transposes [in, out] layout to [out, in].
     */
    public static Int8Linear fromOnnx(OnnxModelParser parser, String weightName, String biasName,
                                       int inFeatures, int outFeatures) {
        OnnxModelParser.TensorData wt = parser.getTensor(weightName);
        if (wt == null) {
            String resolved = parser.resolveWeightFromBias(biasName, inFeatures, outFeatures);
            if (resolved != null) {
                wt = parser.getTensor(resolved);
            }
            if (wt == null) {
                throw new IllegalArgumentException(
                    "Weight tensor not found: " + weightName
                    + " (also could not resolve via bias " + biasName + ")");
            }
        }

        int nWeights = outFeatures * inFeatures;
        byte[] weight;
        float weightScale;

        if (wt.dataType == OnnxModelParser.ONNX_INT8) {
            // Already int8 - use directly
            weight = wt.toInt8();

            // Look for scale tensor
            OnnxModelParser.TensorData st = parser.getTensor(weightName + "_scale");
            if (st != null && st.dataType == OnnxModelParser.ONNX_FLOAT) {
                weightScale = ByteBuffer.wrap(st.rawData).order(ByteOrder.LITTLE_ENDIAN).getFloat();
            } else {
                weightScale = 1.0f;
            }
        } else {
            // Float weights - quantize to int8 at load time
            float[] floatW = wt.toFloat();

            // Check if we need transpose: ONNX BERT may store as [in, out]
            boolean needTranspose = false;
            if (wt.dims.length == 2) {
                if (wt.dims[0] == inFeatures && wt.dims[1] == outFeatures) {
                    needTranspose = true;
                }
            }

            float[] ordered;
            if (needTranspose) {
                ordered = new float[nWeights];
                for (int i = 0; i < outFeatures; i++) {
                    for (int j = 0; j < inFeatures; j++) {
                        ordered[i * inFeatures + j] = floatW[j * outFeatures + i];
                    }
                }
            } else {
                ordered = floatW;
            }

            // Quantize
            weight = new byte[nWeights];
            weightScale = MathOps.symmetricQuantize(ordered, 0, nWeights, weight);
        }

        // Load bias (always float)
        OnnxModelParser.TensorData bt = parser.getTensor(biasName);
        if (bt == null) {
            throw new IllegalArgumentException("Bias tensor not found: " + biasName);
        }
        float[] bias = bt.toFloat();

        return new Int8Linear(weight, bias, weightScale, inFeatures, outFeatures);
    }

    /**
     * Forward pass: int8 matmul with dynamic input quantization.
     * Input is [seqLen * inFeatures] flat array.
     * Returns [seqLen * outFeatures] flat array.
     */
    public float[] forward(float[] input, int seqLen) {
        float[] output = new float[seqLen * outFeatures];
        byte[] inputQ = new byte[inFeatures];
        int[] acc = new int[outFeatures];

        for (int s = 0; s < seqLen; s++) {
            int inOff = s * inFeatures;
            int outOff = s * outFeatures;

            // Dynamically quantize this input row
            float inputScale = MathOps.symmetricQuantize(input, inOff, inFeatures, inputQ);

            // Int8 mat-vec: acc[i] = sum_j(weight[i,j] * inputQ[j])
            MathOps.matVecMulI8(weight, inputQ, acc, outFeatures, inFeatures);

            // Dequantize and add bias
            float combinedScale = inputScale * weightScale;
            for (int i = 0; i < outFeatures; i++) {
                output[outOff + i] = acc[i] * combinedScale + bias[i];
            }
        }

        return output;
    }
}
