import asyncio

from fastembed import SparseTextEmbedding

from app.ingestion.embeddings.base import SparseEmbeddingProvider, SparseVector

class FastEmbedBM25Provider(SparseEmbeddingProvider):
    model_name = "Qdrant/bm25"

    def __init__(self):
        self._model : SparseTextEmbedding | None = None

    def _get_model(self) -> SparseTextEmbedding:
        if self._model is None:
            self._model = SparseTextEmbedding(model_name=self.model_name)
        return self._model

    def _embed_batch_sync(self, texts: list[str]) -> list[SparseVector]:
        model = self._get_model()
        embeddings = list(model.embed(texts))
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in embeddings
        ]

    async def embed_batch(self, texts : list[str]) -> list[SparseVector]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_batch_sync, texts)
        