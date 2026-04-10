package dev.juq.embedding;

import dev.juq.inference.BertModel;
import dev.juq.inference.MathOps;
import dev.juq.tokenizer.WordPieceTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

/**
 * Pure Java embedding model using hand-written int8 BERT inference.
 * No JNI dependencies — fully virtual-thread compatible.
 */
public class PureJavaEmbeddingModel implements EmbeddingModel {

    private final WordPieceTokenizer tokenizer;
    private final BertModel model;
    private final int dimensions;
    private final String queryPrefix;
    private final String documentPrefix;

    public PureJavaEmbeddingModel(Path modelDir) throws IOException {
        this(modelDir, "", "");
    }

    public PureJavaEmbeddingModel(Path modelDir, String queryPrefix, String documentPrefix) throws IOException {
        this(modelDir, "model.onnx", queryPrefix, documentPrefix);
    }

    public PureJavaEmbeddingModel(Path modelDir, String modelFileName,
                                   String queryPrefix, String documentPrefix) throws IOException {
        this.queryPrefix = queryPrefix;
        this.documentPrefix = documentPrefix;
        this.tokenizer = new WordPieceTokenizer(modelDir.resolve("tokenizer.json"));
        this.model = BertModel.load(modelDir.resolve(modelFileName));
        this.dimensions = model.hiddenSize();
    }

    @Override
    public float[] embed(String text) {
        WordPieceTokenizer.Encoding enc = tokenizer.encode(text);

        float[] hidden = model.forward(enc.ids(), enc.typeIds(), enc.ids().length);
        float[] pooled = MathOps.meanPool(hidden, enc.attentionMask(),
            enc.ids().length, dimensions);
        MathOps.l2Normalize(pooled);

        return pooled;
    }

    @Override
    public float[][] embedBatch(List<String> texts) {
        float[][] result = new float[texts.size()][];
        for (int i = 0; i < texts.size(); i++) {
            result[i] = embed(texts.get(i));
        }
        return result;
    }

    @Override
    public int dimensions() {
        return dimensions;
    }

    @Override
    public String queryPrefix() {
        return queryPrefix;
    }

    @Override
    public String documentPrefix() {
        return documentPrefix;
    }

    @Override
    public void close() {
        tokenizer.close();
        model.close();
    }
}
