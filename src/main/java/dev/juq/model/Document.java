package dev.juq.model;

import java.util.Map;
import java.util.Set;

public record Document(
    String id,
    String phrase1,
    String phrase2,
    Map<String, Object> metadata,
    Set<String> tags
) {
    public Document {
        if (tags == null) tags = Set.of();
    }
}
