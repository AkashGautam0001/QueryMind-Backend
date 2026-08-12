import json
import math
import time

from app.config import get_settings
from app.core.clients import get_redis_client
from app.core.logging import get_logger

logger = get_logger(__name__)

_PREFIX = "semcache"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _scope_key(document_id: str | None) -> str:
    return document_id or "global"


async def cache_lookup(
    query_embedding: list[float],
    document_id: str | None = None,
) -> dict | None:
    """Returns {"answer": str, "citations": list} if a cached hit is found,
    or None if the query should proceed through the full pipeline."""
    settings = get_settings()
    if settings.semantic_cache_similarity_threshold >= 1.0:
        return None  # cache effectively disabled

    try:
        redis = get_redis_client()
        scope = _scope_key(document_id)
        keys = await redis.keys(f"{_PREFIX}:{scope}:*")

        for key in keys:
            raw = await redis.get(key)
            if not raw:
                continue
            entry = json.loads(raw)
            similarity = _cosine(query_embedding, entry["embedding"])
            if similarity >= settings.semantic_cache_similarity_threshold:
                logger.info(
                    "semantic_cache_hit",
                    similarity=round(similarity, 4),
                    scope=scope,
                )
                return {"answer": entry["answer"], "citations": entry["citations"]}
    except Exception as exc:
        logger.warning("semantic_cache_lookup_failed", error=str(exc))

    return None


async def cache_store(
    query_embedding: list[float],
    answer: str,
    citations: list,
    document_id: str | None = None,
) -> None:
    """Stores a query result in the semantic cache."""
    settings = get_settings()
    if settings.semantic_cache_similarity_threshold >= 1.0:
        return  # cache disabled

    try:
        redis = get_redis_client()
        scope = _scope_key(document_id)
        key = f"{_PREFIX}:{scope}:{int(time.time() * 1000)}"
        entry = json.dumps(
            {"embedding": query_embedding, "answer": answer, "citations": citations}
        )
        await redis.set(key, entry, ex=settings.semantic_cache_ttl)
        logger.info("semantic_cache_stored", scope=scope)
    except Exception as exc:
        logger.warning("semantic_cache_store_failed", error=str(exc))
