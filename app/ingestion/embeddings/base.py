from dataclasses import dataclass

@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

class EmbeddingProvider:
    dimensions: int
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

class SparseEmbeddingProvider:
    async def embed_batch(self, texts: list[str]) -> list[SparseVector]:
        raise NotImplementedError