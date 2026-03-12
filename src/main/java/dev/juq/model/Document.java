package dev.juq.model;

import java.util.Map;

public record Document(
    String id,
    String phrase1,
    String phrase2,
    Map<String, Object> metadata
) {}
