from arq import ArqRedis, create_pool
from arq.connections import RedisSettings
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.config import get_settings

settings = get_settings()

_qdrant_client: AsyncQdrantClient | None = None
_redis_client: Redis | None = None
_arq_pool: ArqRedis | None = None

def get_qdrant_client() -> AsyncQdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
    return _qdrant_client

def get_redis_client() -> Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )
    return _redis_client

async def check_qdrant_connection() -> bool:
    try:
        await get_qdrant_client().get_collection()
        return True
    except Exception:
        return False

async def check_redis_connection() -> bool:
    try:
        return await get_redis_client().ping()
    except Exception:
        return False

async def get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(
            RedisSettings(host=settings.redis_host, port=settings.redis_port)
        )
    return _arq_pool

async def close_clients() -> None:
    global _qdrant_client, _redis_client, _arq_pool
    if _qdrant_client is not None:
        await _qdrant_client.close()
        _qdrant_client = None
    if _redis_client is not None:
        await _redis_client.aclose()
        _redis_client = None
    if _arq_pool is not None:
        await _arq_pool.close()
        _arq_pool = None