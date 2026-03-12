package dev.juq.data;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.juq.model.Document;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class JsonDocumentLoader {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    public static List<Document> load(Path path) throws IOException {
        return MAPPER.readValue(path.toFile(), new TypeReference<>() {});
    }
}
