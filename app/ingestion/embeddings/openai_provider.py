from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.ingestion.embeddings.base import EmbeddingProvider

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self, 
        api_key: str, 
        model: str = "text-embedding-3-large",
        dimensions: int = 3072,
        batch_size: int = 100,
    ):
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError))
    )
    async def _embed_one_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self.model,
            inputs=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]

    async def embed_batch(self, texts : list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts, self.batch_size)):
            batch = texts[i : i + self.batch_size]
            results.extend(await self._embed_one_batch(batch))
        return results

    async def close(self) -> None:
        await self._client.close()
        

        