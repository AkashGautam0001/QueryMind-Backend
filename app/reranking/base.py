class Reranker:
    async def rerank(self, query: str, candidates: list[str]) -> list[float]:
        """Returns one relevance score per candidate, same order as input.
        Scores are for ranking only — see CrossEncoderReranker for the
        calibration caveat."""
        raise NotImplementedError
