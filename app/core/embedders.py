from app.config import get_settings
from app.ingestion.embeddings.openai_provider import OpenAIEmbeddingProvider
from app.ingestion.embeddings.sparse import FastEmbedBM25Provider

_dense_embedder : OpenAIEmbeddingProvider | None = None
_sparse_embedder : FastEmbedBM25Provider | None = None

def get_dense_embedder() -> OpenAIEmbeddingProvider:
    global _dense_embedder
    if _dense_embedder is None:
        settings = get_settings()
        _dense_embedder = OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            batch_size=settings.embedding_batch_size,
        )
    return _dense_embedder

def get_sparse_embedder() -> FastEmbedBM25Provider:
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = FastEmbedBM25Provider()
    return _sparse_embedder

async def close_embedder() -> None:
    global _dense_embedder, _sparse_embedder
    if _dense_embedder is not None:
        await _dense_embedder.close()
        _dense_embedder = None
    _sparse_embedder = None