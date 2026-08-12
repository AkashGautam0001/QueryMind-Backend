from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_PUBLIC_PATHS = {"/health/live", "/health/ready", "/"}

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        settings = get_settings()
        if not settings.api_key or request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key", "")
        if provided != settings.api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)