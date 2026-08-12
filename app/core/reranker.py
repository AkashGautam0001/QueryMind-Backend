from app.config import get_settings
from app.reranking.cross_encoder import CrossEncoderReranker

_reranker: CrossEncoderReranker | None = None


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        settings = get_settings()
        _reranker = CrossEncoderReranker(model_name=settings.rerank_model_name)
    return _reranker
