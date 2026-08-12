from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, context, documents, eval, health, retrieve
from app.config import get_settings
from app.core.auth import APIKeyMiddleware
from app.core.clients import close_clients
from app.core.embedders import close_embedders
from app.core.llm import close_llm_provider
from app.core.logging import configure_logging, get_logger
from app.core.rate_limit import RateLimitMiddleware
from app.core.storage import ensure_bucket_exists
from app.core.telemetry import setup_tracing
from app.vectorstore.qdrant_setup import ensure_collection_exists

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("starting_up", environment=settings.environment)
    await ensure_collection_exists()
    await ensure_bucket_exists()
    yield
    logger.info("shutting_down")
    await close_clients()
    await close_embedders()
    await close_llm_provider()


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan,
)

setup_tracing(app)

# Middleware — order matters: CORS first so preflight requests get the
# right headers before auth or rate limiting can reject them.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(RateLimitMiddleware)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(retrieve.router)
app.include_router(context.router)
app.include_router(chat.router)
app.include_router(eval.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an unexpected error returns a clean JSON 500 instead of
    leaking a stack trace to the client. The real trace still goes to logs."""
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/")
async def root() -> dict:
    return {"name": settings.app_name, "environment": settings.environment}
