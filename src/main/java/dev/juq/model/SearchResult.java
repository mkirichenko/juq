package dev.juq.model;

public record SearchResult(
    Document document,
    float score
) implements Comparable<SearchResult> {

    @Override
    public int compareTo(SearchResult other) {
        return Float.compare(this.score, other.score);
    }
}
