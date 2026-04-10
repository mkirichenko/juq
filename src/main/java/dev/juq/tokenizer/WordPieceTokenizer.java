package dev.juq.tokenizer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure Java WordPiece tokenizer compatible with HuggingFace tokenizer.json format.
 * Port of c/src/tokenizer.c.
 */
public class WordPieceTokenizer implements AutoCloseable {

    public record Encoding(int[] ids, int[] attentionMask, int[] typeIds) {}

    private final Map<String, Integer> vocab;
    private int clsId = 101;
    private int sepId = 102;
    private int unkId = 100;
    private int padId = 0;
    private String continuingPrefix = "##";
    private int maxInputCharsPerWord = 100;
    private boolean doLowerCase = true;

    public WordPieceTokenizer(Path tokenizerJsonPath) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(tokenizerJsonPath.toFile());

        // Load vocabulary from model.vocab
        JsonNode modelNode = root.get("model");
        if (modelNode == null) {
            throw new IOException("No 'model' in tokenizer.json");
        }

        JsonNode vocabNode = modelNode.get("vocab");
        if (vocabNode == null || !vocabNode.isObject()) {
            throw new IOException("No 'model.vocab' in tokenizer.json");
        }

        this.vocab = new HashMap<>(vocabNode.size() * 2);
        var fields = vocabNode.fields();
        while (fields.hasNext()) {
            var entry = fields.next();
            vocab.put(entry.getKey(), entry.getValue().asInt());
        }

        // Parse added_tokens for special token IDs
        JsonNode addedTokens = root.get("added_tokens");
        if (addedTokens != null && addedTokens.isArray()) {
            for (JsonNode token : addedTokens) {
                String content = token.has("content") ? token.get("content").asText() : null;
                if (content == null || !token.has("id")) continue;
                int id = token.get("id").asInt();
                switch (content) {
                    case "[CLS]" -> clsId = id;
                    case "[SEP]" -> sepId = id;
                    case "[UNK]" -> unkId = id;
                    case "[PAD]" -> padId = id;
                }
            }
        }

        // Parse continuing_subword_prefix
        JsonNode prefixNode = modelNode.get("continuing_subword_prefix");
        if (prefixNode != null && prefixNode.isTextual()) {
            continuingPrefix = prefixNode.asText();
        }

        // Parse max_input_chars_per_word
        JsonNode maxCharsNode = modelNode.get("max_input_chars_per_word");
        if (maxCharsNode != null && maxCharsNode.isNumber()) {
            maxInputCharsPerWord = maxCharsNode.asInt();
        }

        // Detect lowercasing from normalizer config
        JsonNode normalizer = root.get("normalizer");
        if (normalizer != null) {
            JsonNode lowercase = normalizer.get("lowercase");
            if (lowercase != null) {
                doLowerCase = lowercase.asBoolean(true);
            }
        }
    }

    public Encoding encode(String text) {
        List<String> words = preTokenize(text);
        List<Integer> ids = new ArrayList<>();

        // [CLS]
        ids.add(clsId);

        // WordPiece each word
        for (String word : words) {
            wordPieceTokenize(word, ids);
        }

        // [SEP]
        ids.add(sepId);

        int len = ids.size();
        int[] idArray = new int[len];
        int[] attentionMask = new int[len];
        int[] typeIds = new int[len];

        for (int i = 0; i < len; i++) {
            idArray[i] = ids.get(i);
            attentionMask[i] = 1;
            // typeIds stays 0
        }

        return new Encoding(idArray, attentionMask, typeIds);
    }

    // --- Pre-tokenization: split on whitespace, each punctuation char is its own token ---

    List<String> preTokenize(String text) {
        List<String> words = new ArrayList<>();
        int len = text.length();
        int i = 0;

        while (i < len) {
            // Skip whitespace
            while (i < len && Character.isWhitespace(text.charAt(i))) i++;
            if (i >= len) break;

            char c = text.charAt(i);
            if (isPunctuation(c)) {
                words.add(String.valueOf(c));
                i++;
            } else {
                int start = i;
                while (i < len && !Character.isWhitespace(text.charAt(i)) && !isPunctuation(text.charAt(i))) {
                    i++;
                }
                words.add(text.substring(start, i));
            }
        }
        return words;
    }

    private static boolean isPunctuation(char c) {
        // Match BERT's definition: Unicode punctuation or ASCII punctuation
        int type = Character.getType(c);
        if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
            (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
            return true;
        }
        return type == Character.DASH_PUNCTUATION ||
               type == Character.START_PUNCTUATION ||
               type == Character.END_PUNCTUATION ||
               type == Character.CONNECTOR_PUNCTUATION ||
               type == Character.OTHER_PUNCTUATION ||
               type == Character.INITIAL_QUOTE_PUNCTUATION ||
               type == Character.FINAL_QUOTE_PUNCTUATION;
    }

    // --- WordPiece algorithm ---

    private void wordPieceTokenize(String word, List<Integer> out) {
        if (word.length() > maxInputCharsPerWord) {
            out.add(unkId);
            return;
        }

        String lower = doLowerCase ? word.toLowerCase() : word;
        int wlen = lower.length();
        int start = 0;
        boolean isFirst = true;

        while (start < wlen) {
            int end = wlen;
            boolean found = false;

            while (end > start) {
                String candidate;
                if (isFirst) {
                    candidate = lower.substring(start, end);
                } else {
                    candidate = continuingPrefix + lower.substring(start, end);
                }

                Integer id = vocab.get(candidate);
                if (id != null) {
                    out.add(id);
                    found = true;
                    break;
                }

                end--;
            }

            if (!found) {
                out.add(unkId);
                // Skip one character
                start++;
                isFirst = false;
                continue;
            }

            start = end;
            isFirst = false;
        }
    }

    public int vocabSize() {
        return vocab.size();
    }

    public int getClsId() { return clsId; }
    public int getSepId() { return sepId; }
    public int getUnkId() { return unkId; }
    public int getPadId() { return padId; }

    @Override
    public void close() {
        // No native resources to free
    }
}
