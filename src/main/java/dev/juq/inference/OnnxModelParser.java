package dev.juq.inference;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Minimal ONNX protobuf parser that extracts initializer tensors.
 * Port of c/src/onnx_model.c. No protobuf library dependency.
 *
 * Only parses what we need: ModelProto → GraphProto → initializer (TensorProto[]).
 */
public class OnnxModelParser {

    // ONNX data type constants
    public static final int ONNX_FLOAT = 1;
    public static final int ONNX_UINT8 = 2;
    public static final int ONNX_INT8 = 3;
    public static final int ONNX_INT32 = 6;
    public static final int ONNX_INT64 = 7;
    public static final int ONNX_FLOAT16 = 10;

    // Protobuf wire types
    private static final int PB_VARINT = 0;
    private static final int PB_64BIT = 1;
    private static final int PB_BYTES = 2;
    private static final int PB_32BIT = 5;

    /**
     * Parsed tensor data from an ONNX initializer.
     */
    public static class TensorData {
        public final String name;
        public final int dataType;
        public final long[] dims;
        public final long numElements;
        public final byte[] rawData; // raw bytes, interpreted according to dataType

        TensorData(String name, int dataType, long[] dims, long numElements, byte[] rawData) {
            this.name = name;
            this.dataType = dataType;
            this.dims = dims;
            this.numElements = numElements;
            this.rawData = rawData;
        }

        /** Convert tensor data to float array regardless of original type. */
        public float[] toFloat() {
            float[] out = new float[(int) numElements];
            ByteBuffer buf = ByteBuffer.wrap(rawData).order(ByteOrder.LITTLE_ENDIAN);

            switch (dataType) {
                case ONNX_FLOAT -> {
                    for (int i = 0; i < numElements; i++) {
                        out[i] = buf.getFloat(i * 4);
                    }
                }
                case ONNX_INT8 -> {
                    for (int i = 0; i < numElements; i++) {
                        out[i] = rawData[i];
                    }
                }
                case ONNX_UINT8 -> {
                    for (int i = 0; i < numElements; i++) {
                        out[i] = rawData[i] & 0xFF;
                    }
                }
                case ONNX_INT32 -> {
                    for (int i = 0; i < numElements; i++) {
                        out[i] = buf.getInt(i * 4);
                    }
                }
                case ONNX_INT64 -> {
                    for (int i = 0; i < numElements; i++) {
                        out[i] = buf.getLong(i * 8);
                    }
                }
                case ONNX_FLOAT16 -> {
                    for (int i = 0; i < numElements; i++) {
                        out[i] = fp16ToFloat(buf.getShort(i * 2) & 0xFFFF);
                    }
                }
                default -> throw new IllegalArgumentException("Unsupported ONNX data type: " + dataType);
            }
            return out;
        }

        /** Get raw data as byte array (for INT8 tensors). */
        public byte[] toInt8() {
            if (dataType == ONNX_INT8) {
                return rawData.clone();
            }
            throw new IllegalStateException("Tensor is not INT8, type=" + dataType);
        }
    }

    private final Map<String, TensorData> tensors = new HashMap<>();
    private final List<NodeInfo> nodes = new ArrayList<>();
    private final Map<String, NodeInfo> producerByOutput = new HashMap<>();

    /** Lightweight view of an ONNX graph node — just what we need for weight resolution. */
    private static class NodeInfo {
        final String opType;
        final List<String> inputs;
        final List<String> outputs;

        NodeInfo(String opType, List<String> inputs, List<String> outputs) {
            this.opType = opType;
            this.inputs = inputs;
            this.outputs = outputs;
        }
    }

    private OnnxModelParser() {}

    /** Load and parse an ONNX model file. */
    public static OnnxModelParser load(Path path) throws IOException {
        byte[] data = Files.readAllBytes(path);
        OnnxModelParser parser = new OnnxModelParser();
        PbReader r = new PbReader(data, 0, data.length);
        parser.parseModelProto(r);
        return parser;
    }

    public TensorData getTensor(String name) {
        return tensors.get(name);
    }

    /**
     * Resolve a logical float tensor: if {@code name} exists, return it as float;
     * otherwise look for an int8 quantized counterpart {@code name + "_quantized"}
     * (with {@code "_scale"} and optional {@code "_zero_point"}) and dequantize.
     * Returns {@code null} if neither pattern is present.
     *
     * <p>Used for ONNX exports produced by quantization tools that replace float
     * embedding/weight tables with their int8 + scale + zero-point equivalents.
     */
    public float[] getFloatOrDequantized(String name) {
        TensorData direct = tensors.get(name);
        if (direct != null) return direct.toFloat();

        TensorData q = tensors.get(name + "_quantized");
        TensorData scale = tensors.get(name + "_scale");
        if (q == null || scale == null) return null;

        float[] scaleF = scale.toFloat();
        // scale may be a scalar (per-tensor) or per-row (per-channel). Per-row is broadcast
        // along the last dim of the quantized tensor.
        int rows = (int) (scaleF.length > 1 ? q.dims[0] : 1);
        int cols = (int) (q.numElements / Math.max(rows, 1));

        TensorData zp = tensors.get(name + "_zero_point");
        int[] zpI;
        if (zp == null) {
            zpI = new int[Math.max(scaleF.length, 1)];
        } else if (zp.numElements == 1) {
            zpI = new int[]{ zeroPointAt(zp, 0) };
        } else {
            zpI = new int[(int) zp.numElements];
            for (int i = 0; i < zpI.length; i++) zpI[i] = zeroPointAt(zp, i);
        }

        float[] out = new float[(int) q.numElements];
        if (scaleF.length == 1) {
            float s = scaleF[0];
            int z = zpI[0];
            for (int i = 0; i < out.length; i++) {
                out[i] = (qValueAt(q, i) - z) * s;
            }
        } else {
            for (int r = 0; r < rows; r++) {
                float s = scaleF[r];
                int z = zpI.length == 1 ? zpI[0] : zpI[r];
                int base = r * cols;
                for (int c = 0; c < cols; c++) {
                    out[base + c] = (qValueAt(q, base + c) - z) * s;
                }
            }
        }
        return out;
    }

    private static int qValueAt(TensorData t, int i) {
        return switch (t.dataType) {
            case ONNX_INT8 -> t.rawData[i];
            case ONNX_UINT8 -> t.rawData[i] & 0xFF;
            default -> throw new IllegalStateException(
                "Unsupported quantized dtype: " + t.dataType + " for " + t.name);
        };
    }

    private static int zeroPointAt(TensorData t, int i) {
        return qValueAt(t, i);
    }

    public Map<String, TensorData> getAllTensors() {
        return tensors;
    }

    public int tensorCount() {
        return tensors.size();
    }

    /**
     * Resolve the weight initializer feeding a named bias.
     *
     * Modern ONNX exports preserve PyTorch parameter names for biases
     * ({@code encoder.layer.0.attention.self.query.bias}) but give MatMul weight
     * initializers anonymous names ({@code onnx::MatMul_2027}). We find the Add node
     * consuming the bias, walk backward through the graph, and pick the initializer
     * matching {@code inFeatures * outFeatures} elements (or the largest one we find).
     *
     * @return the resolved tensor name, or {@code null} if no candidate found.
     */
    public String resolveWeightFromBias(String biasName, int inFeatures, int outFeatures) {
        long expected = (long) inFeatures * outFeatures;
        for (NodeInfo n : nodes) {
            if (!"Add".equals(n.opType)) continue;
            if (!n.inputs.contains(biasName)) continue;

            Deque<String> queue = new ArrayDeque<>();
            for (String in : n.inputs) {
                if (!in.equals(biasName)) queue.add(in);
            }
            Set<String> visited = new HashSet<>();
            String fallback = null;
            long fallbackSize = -1;
            while (!queue.isEmpty()) {
                String t = queue.poll();
                if (t.isEmpty() || !visited.add(t)) continue;

                TensorData td = tensors.get(t);
                if (td != null) {
                    if (t.equals(biasName)) continue;
                    if (td.numElements == expected) return t;
                    if (td.numElements > fallbackSize) {
                        fallbackSize = td.numElements;
                        fallback = t;
                    }
                    continue;
                }
                NodeInfo prod = producerByOutput.get(t);
                if (prod != null) queue.addAll(prod.inputs);
            }
            if (fallback != null) return fallback;
        }
        return null;
    }

    // --- Protobuf reader ---

    private static class PbReader {
        final byte[] data;
        int pos;
        final int end;

        PbReader(byte[] data, int pos, int end) {
            this.data = data;
            this.pos = pos;
            this.end = end;
        }

        long readVarint() {
            long val = 0;
            int shift = 0;
            while (pos < end) {
                int b = data[pos++] & 0xFF;
                val |= (long) (b & 0x7F) << shift;
                if ((b & 0x80) == 0) break;
                shift += 7;
            }
            return val;
        }

        byte[] readBytes() {
            int len = (int) readVarint();
            byte[] result = new byte[len];
            System.arraycopy(data, pos, result, 0, len);
            pos += len;
            return result;
        }

        /** Read length-delimited and return a sub-reader. */
        PbReader readSubMessage() {
            int len = (int) readVarint();
            PbReader sub = new PbReader(data, pos, pos + len);
            pos += len;
            return sub;
        }

        void skip(int wireType) {
            switch (wireType) {
                case PB_VARINT -> readVarint();
                case PB_64BIT -> pos += 8;
                case PB_32BIT -> pos += 4;
                case PB_BYTES -> {
                    int len = (int) readVarint();
                    pos += len;
                }
            }
        }
    }

    // --- Protobuf message parsers ---

    /**
     * ModelProto: field 7 = graph (GraphProto)
     */
    private void parseModelProto(PbReader r) {
        while (r.pos < r.end) {
            long tag = r.readVarint();
            int field = (int) (tag >> 3);
            int wire = (int) (tag & 7);

            if (field == 7 && wire == PB_BYTES) {
                parseGraphProto(r.readSubMessage());
            } else {
                r.skip(wire);
            }
        }
    }

    /**
     * GraphProto: field 1 = node (repeated NodeProto), field 5 = initializer (repeated TensorProto)
     */
    private void parseGraphProto(PbReader r) {
        while (r.pos < r.end) {
            long tag = r.readVarint();
            int field = (int) (tag >> 3);
            int wire = (int) (tag & 7);

            if (field == 1 && wire == PB_BYTES) {
                parseNodeProto(r.readSubMessage());
            } else if (field == 5 && wire == PB_BYTES) {
                parseTensorProto(r.readSubMessage());
            } else {
                r.skip(wire);
            }
        }
    }

    /**
     * NodeProto fields:
     *   1: input (repeated string)
     *   2: output (repeated string)
     *   3: name (string)
     *   4: op_type (string)
     *   6: attribute (repeated AttributeProto) — skipped
     */
    private void parseNodeProto(PbReader r) {
        List<String> inputs = new ArrayList<>();
        List<String> outputs = new ArrayList<>();
        String opType = "";

        while (r.pos < r.end) {
            long tag = r.readVarint();
            int field = (int) (tag >> 3);
            int wire = (int) (tag & 7);

            switch (field) {
                case 1 -> {
                    if (wire == PB_BYTES) {
                        int len = (int) r.readVarint();
                        inputs.add(new String(r.data, r.pos, len, java.nio.charset.StandardCharsets.UTF_8));
                        r.pos += len;
                    } else {
                        r.skip(wire);
                    }
                }
                case 2 -> {
                    if (wire == PB_BYTES) {
                        int len = (int) r.readVarint();
                        outputs.add(new String(r.data, r.pos, len, java.nio.charset.StandardCharsets.UTF_8));
                        r.pos += len;
                    } else {
                        r.skip(wire);
                    }
                }
                case 4 -> {
                    if (wire == PB_BYTES) {
                        int len = (int) r.readVarint();
                        opType = new String(r.data, r.pos, len, java.nio.charset.StandardCharsets.UTF_8);
                        r.pos += len;
                    } else {
                        r.skip(wire);
                    }
                }
                default -> r.skip(wire);
            }
        }

        NodeInfo info = new NodeInfo(opType, inputs, outputs);
        nodes.add(info);
        for (String out : outputs) producerByOutput.put(out, info);
    }

    /**
     * TensorProto fields:
     *   1: dims (repeated int64)
     *   2: data_type (int32)
     *   4: float_data (repeated float, packed)
     *   5: int32_data (repeated int32, packed)
     *   7: int64_data (repeated int64, packed)
     *   8: name (string)
     *   9: raw_data (bytes)
     */
    private void parseTensorProto(PbReader r) {
        List<Long> dims = new ArrayList<>();
        int dataType = 0;
        String name = null;
        byte[] rawData = null;
        byte[] floatData = null;
        byte[] int32Data = null;
        byte[] int64Data = null;

        while (r.pos < r.end) {
            long tag = r.readVarint();
            int field = (int) (tag >> 3);
            int wire = (int) (tag & 7);

            switch (field) {
                case 1 -> { // dims
                    if (wire == PB_BYTES) {
                        // packed repeated int64
                        PbReader sub = r.readSubMessage();
                        while (sub.pos < sub.end) {
                            dims.add(sub.readVarint());
                        }
                    } else {
                        dims.add(r.readVarint());
                    }
                }
                case 2 -> dataType = (int) r.readVarint(); // data_type
                case 4 -> { // float_data (packed fixed32)
                    if (wire == PB_BYTES) {
                        floatData = r.readBytes();
                    } else {
                        r.skip(wire);
                    }
                }
                case 5 -> { // int32_data (packed varint)
                    if (wire == PB_BYTES) {
                        int32Data = r.readBytes();
                    } else {
                        r.skip(wire);
                    }
                }
                case 7 -> { // int64_data (packed varint)
                    if (wire == PB_BYTES) {
                        int64Data = r.readBytes();
                    } else {
                        r.skip(wire);
                    }
                }
                case 8 -> { // name
                    int len = (int) r.readVarint();
                    name = new String(r.data, r.pos, len, java.nio.charset.StandardCharsets.UTF_8);
                    r.pos += len;
                }
                case 9 -> rawData = r.readBytes(); // raw_data
                default -> r.skip(wire);
            }
        }

        // Compute number of elements
        long numElements = 1;
        for (long d : dims) numElements *= d;

        // Resolve data: prefer raw_data, else decode packed fields
        byte[] data = null;
        if (rawData != null && rawData.length > 0) {
            data = rawData;
        } else if (floatData != null && floatData.length > 0) {
            data = floatData;
        } else if (int32Data != null && int32Data.length > 0) {
            // Packed varint int32 → decode to raw bytes
            data = decodePackedInt32(int32Data, (int) numElements);
        } else if (int64Data != null && int64Data.length > 0) {
            data = decodePackedInt64(int64Data, (int) numElements);
        }

        if (name != null && data != null) {
            long[] dimsArray = dims.stream().mapToLong(Long::longValue).toArray();
            tensors.put(name, new TensorData(name, dataType, dimsArray, numElements, data));
        }
    }

    /** Decode packed varint int32 values into raw little-endian int32 bytes. */
    private static byte[] decodePackedInt32(byte[] packed, int numElements) {
        ByteBuffer out = ByteBuffer.allocate(numElements * 4).order(ByteOrder.LITTLE_ENDIAN);
        PbReader r = new PbReader(packed, 0, packed.length);
        int idx = 0;
        while (r.pos < r.end && idx < numElements) {
            out.putInt(idx * 4, (int) r.readVarint());
            idx++;
        }
        return out.array();
    }

    /** Decode packed varint int64 values into raw little-endian int64 bytes. */
    private static byte[] decodePackedInt64(byte[] packed, int numElements) {
        ByteBuffer out = ByteBuffer.allocate(numElements * 8).order(ByteOrder.LITTLE_ENDIAN);
        PbReader r = new PbReader(packed, 0, packed.length);
        int idx = 0;
        while (r.pos < r.end && idx < numElements) {
            out.putLong(idx * 8, r.readVarint());
            idx++;
        }
        return out.array();
    }

    /** Convert IEEE 754 half-precision (fp16) to float. */
    private static float fp16ToFloat(int h) {
        int sign = (h >> 15) & 1;
        int exp = (h >> 10) & 0x1F;
        int mantissa = h & 0x3FF;

        if (exp == 0) {
            // Subnormal or zero
            return (float) ((sign == 0 ? 1 : -1) * Math.scalb(mantissa, -24));
        } else if (exp == 31) {
            // Inf or NaN
            return mantissa == 0 ?
                (sign == 0 ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY) :
                Float.NaN;
        }
        // Normal
        return (float) ((sign == 0 ? 1 : -1) * Math.scalb(mantissa + 1024, exp - 25));
    }
}
