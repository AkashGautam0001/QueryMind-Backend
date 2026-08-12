from app.retrieval.context import RerankedChunk


def has_relevant_context(chunks: list[RerankedChunk], threshold: float) -> bool:
    """True if at least one chunk clears the relevance bar. This is the
    primary defense against the single biggest hallucination source in
    RAG: forcing the model to answer from weak or irrelevant context. The
    check runs before any LLM call — refusing to answer costs nothing,
    while generating from bad context costs a wrong, confidently-worded
    answer."""
    if not chunks:
        return False
    return any(chunk.rerank_score >= threshold for chunk in chunks)
