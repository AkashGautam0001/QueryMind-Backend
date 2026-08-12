from fastapi import APIRouter, Response

from app.core.clients import check_qdrant_connection, check_redis_connection
from app.core.storage import check_storage_connection
from app.db.session import check_db_connection

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness() -> dict:
    """Is the process up? No dependency checks — if this is slow or down,
    the container itself is the problem, not Postgres/Qdrant/Redis."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness(response: Response) -> dict:
    """Can this instance actually serve requests right now? Checks every
    hard dependency with a real round trip. Used for load-balancer routing
    and deploy gating, not for fast liveness checks."""
    checks = {
        "postgres": await check_db_connection(),
        "qdrant": await check_qdrant_connection(),
        "redis": await check_redis_connection(),
        "storage": await check_storage_connection(),
    }
    all_healthy = all(checks.values())
    if not all_healthy:
        response.status_code = 503
    return {
        "status": "ready" if all_healthy else "degraded",
        "checks": checks,
    }
