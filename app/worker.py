import uuid

from arq.connections import RedisSettings

from app.config import get_settings
from app.core.embedders import close_embedders
from app.core.logging import configure_logging, get_logger
from app.evaluation.runner import run_evaluation
from app.ingestion.pipeline import run_ingestion

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


async def process_document(ctx: dict, document_id: str, job_id: str) -> None:
    await run_ingestion(document_id=uuid.UUID(document_id), job_id=uuid.UUID(job_id))


async def run_eval(ctx: dict, run_id: str) -> None:
    """Eval runs are long (one LLM call per question × 4 metrics) and should
    never run in the API process — enqueued to the worker just like ingestion."""
    await run_evaluation(run_id=uuid.UUID(run_id))


async def on_startup(ctx: dict) -> None:
    logger.info("worker_starting_up")


async def on_shutdown(ctx: dict) -> None:
    logger.info("worker_shutting_down")
    await close_embedders()


class WorkerSettings:
    functions = [process_document, run_eval]
    redis_settings = RedisSettings(host=settings.redis_host, port=settings.redis_port)
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = 5
    job_timeout = 600
