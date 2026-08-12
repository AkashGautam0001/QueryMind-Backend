import asyncio

from app.reranking.base import Reranker


class CrossEncoderReranker(Reranker):
    """Scores (query, passage) pairs with bidirectional attention — slower
    than the embedding-based first-stage retrieval, but far more accurate,
    since it actually reads the query and passage together instead of
    comparing two separately-computed vectors. This is why it's a second
    stage over a small candidate pool, not the only retrieval mechanism.

    Note on scores: cross-encoder outputs are relevance scores for sorting,
    not calibrated probabilities, unless the specific model documents
    otherwise. Don't threshold on an absolute value without checking your
    model's calibration story."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            # Imported here, not at module level — this dependency (and the
            # model weights themselves) shouldn't be required just to
            # import this module, and the actual download only needs to
            # happen once, lazily, on first real use.
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []

        def _predict() -> list[float]:
            model = self._load_model()
            pairs = [(query, candidate) for candidate in candidates]
            return model.predict(pairs).tolist()

        # CrossEncoder.predict is a synchronous, CPU/GPU-bound torch forward
        # pass — running it directly would block the event loop for the
        # duration of inference.
        return await asyncio.to_thread(_predict)
