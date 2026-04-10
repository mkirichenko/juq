package dev.juq.tokenizer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

class WordPieceTokenizerTest {

    private static final Path MODEL_DIR = Path.of("model", "minilm");

    static boolean modelExists() {
        return Files.exists(MODEL_DIR.resolve("tokenizer.json"));
    }

    @Test
    @EnabledIf("modelExists")
    void testLoadTokenizer() throws IOException {
        try (WordPieceTokenizer tok = new WordPieceTokenizer(MODEL_DIR.resolve("tokenizer.json"))) {
            assertTrue(tok.vocabSize() > 0, "Vocab should not be empty");
            assertTrue(tok.vocabSize() > 10000, "BERT vocab typically has >30k entries");
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEncodeSimpleText() throws IOException {
        try (WordPieceTokenizer tok = new WordPieceTokenizer(MODEL_DIR.resolve("tokenizer.json"))) {
            WordPieceTokenizer.Encoding enc = tok.encode("hello world");

            // Should start with [CLS] and end with [SEP]
            assertEquals(tok.getClsId(), enc.ids()[0]);
            assertEquals(tok.getSepId(), enc.ids()[enc.ids().length - 1]);

            // Should have at least [CLS] + some tokens + [SEP]
            assertTrue(enc.ids().length >= 3);

            // Attention mask should all be 1
            for (int m : enc.attentionMask()) {
                assertEquals(1, m);
            }

            // Type IDs should all be 0
            for (int t : enc.typeIds()) {
                assertEquals(0, t);
            }

            // Length consistency
            assertEquals(enc.ids().length, enc.attentionMask().length);
            assertEquals(enc.ids().length, enc.typeIds().length);
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEncodePunctuation() throws IOException {
        try (WordPieceTokenizer tok = new WordPieceTokenizer(MODEL_DIR.resolve("tokenizer.json"))) {
            WordPieceTokenizer.Encoding enc = tok.encode("Hello, world!");

            // Should have separate tokens for comma and exclamation
            assertTrue(enc.ids().length >= 5); // [CLS] hello , world ! [SEP]
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEncodeEmptyText() throws IOException {
        try (WordPieceTokenizer tok = new WordPieceTokenizer(MODEL_DIR.resolve("tokenizer.json"))) {
            WordPieceTokenizer.Encoding enc = tok.encode("");

            // Should still have [CLS] [SEP]
            assertEquals(2, enc.ids().length);
            assertEquals(tok.getClsId(), enc.ids()[0]);
            assertEquals(tok.getSepId(), enc.ids()[1]);
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEncodeMultipleSpaces() throws IOException {
        try (WordPieceTokenizer tok = new WordPieceTokenizer(MODEL_DIR.resolve("tokenizer.json"))) {
            WordPieceTokenizer.Encoding single = tok.encode("hello world");
            WordPieceTokenizer.Encoding multi = tok.encode("hello   world");

            // Multiple spaces should produce same tokens
            assertArrayEquals(single.ids(), multi.ids());
        }
    }

    @Test
    @EnabledIf("modelExists")
    void testEncodeNumbers() throws IOException {
        try (WordPieceTokenizer tok = new WordPieceTokenizer(MODEL_DIR.resolve("tokenizer.json"))) {
            WordPieceTokenizer.Encoding enc = tok.encode("test 42 example");

            // Should produce valid tokens
            assertTrue(enc.ids().length >= 4); // [CLS] test 42 example [SEP]
            for (int id : enc.ids()) {
                assertTrue(id >= 0);
            }
        }
    }
}
