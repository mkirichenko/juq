package dev.juq.inference;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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

    public Map<String, TensorData> getAllTensors() {
        return tensors;
    }

    public int tensorCount() {
        return tensors.size();
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
     * GraphProto: field 5 = initializer (repeated TensorProto)
     */
    private void parseGraphProto(PbReader r) {
        while (r.pos < r.end) {
            long tag = r.readVarint();
            int field = (int) (tag >> 3);
            int wire = (int) (tag & 7);

            if (field == 5 && wire == PB_BYTES) {
                parseTensorProto(r.readSubMessage());
            } else {
                r.skip(wire);
            }
        }
    }

    /**
     * TensorProto fields:
     *   1: dims (repeated int64)
     *   2: data_type (int32)
     *   4: float_data (repeated float, packed)
     *   5: int32_data (repeated int32, packed)
     *   7: int64_data (repeated int64, packed)
     *   8: name (string)
     *  13: raw_data (bytes)
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
                case 13 -> rawData = r.readBytes(); // raw_data
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
