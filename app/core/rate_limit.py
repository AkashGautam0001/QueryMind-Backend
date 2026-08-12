import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.core.clients import get_redis_client

_EXEMPT_PATHS = {"/health/live", "/health/ready", "/"}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window rate limit per client IP, stored in Redis with a 60s
    TTL. Misses gracefully — if Redis is unreachable, the request is
    allowed through rather than blocked, so a Redis hiccup doesn't take
    down the API. Configure via RATE_LIMIT_PER_MINUTE."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        settings = get_settings()
        client_ip = request.client.host if request.client else "unknown"
        window = int(time.time() // 60)
        key = f"rl:{client_ip}:{window}"

        try:
            redis = get_redis_client()
            count = await redis.incr(key)
            if count == 1:
                await redis.expire(key, 70)  # slightly over 60s for safety
            if count > settings.rate_limit_per_minute:
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit exceeded — max {settings.rate_limit_per_minute} requests/min"},
                    headers={"Retry-After": "60"},
                )
        except Exception:
            # Redis unavailable — fail open so a cache outage doesn't DoS the API.
            pass

        return await call_next(request)
