import hashlib
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.clients import get_arq_pool
from app.core.logging import get_logger
from app.core.storage import upload_bytes
from app.db.session import get_db
from app.ingestion.chunking import STRATEGY_NAMES
from app.models.document import Document, DocumentStatus, SourceType
from app.models.ingestion_job import IngestionJob
from app.schemas.document import (
    DocumentListResponse,
    DocumentResponse,
    IngestionAcceptedResponse,
    IngestURLRequest,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

_EXTENSION_TO_SOURCE_TYPE = {
    ".pdf": SourceType.PDF,
    ".docx": SourceType.DOCX,
}


def _detect_source_type(filename: str) -> SourceType:
    suffix = Path(filename).suffix.lower()
    source_type = _EXTENSION_TO_SOURCE_TYPE.get(suffix)
    if source_type is None:
        supported = ", ".join(_EXTENSION_TO_SOURCE_TYPE)
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {supported}",
        )
    return source_type


def _validate_strategy(strategy: str) -> None:
    if strategy not in STRATEGY_NAMES:
        raise HTTPException(
            status_code=422,
            detail=f"chunking_strategy must be one of {STRATEGY_NAMES}",
        )


async def _find_existing(session: AsyncSession, content_hash: str) -> Document | None:
    result = await session.execute(select(Document).where(Document.content_hash == content_hash))
    return result.scalar_one_or_none()


async def _create_job_and_enqueue(session: AsyncSession, document: Document) -> IngestionJob:
    job = IngestionJob(document_id=document.id, chunking_strategy=document.chunking_strategy)
    session.add(job)
    await session.flush()

    pool = await get_arq_pool()
    await pool.enqueue_job("process_document", str(document.id), str(job.id))

    return job


@router.post("", status_code=202, response_model=IngestionAcceptedResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunking_strategy: str = Form(default=None),
    force: bool = Form(default=False),
    session: AsyncSession = Depends(get_db),
) -> IngestionAcceptedResponse:
    settings = get_settings()
    strategy = chunking_strategy or settings.default_chunking_strategy
    _validate_strategy(strategy)

    source_type = _detect_source_type(file.filename)

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f}MB, exceeds the {settings.max_upload_size_mb}MB limit.",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    content_hash = hashlib.sha256(content).hexdigest()

    existing = await _find_existing(session, content_hash)
    if existing is not None and not force:
        return IngestionAcceptedResponse(
            document_id=existing.id,
            job_id=None,
            status=existing.status,
            duplicate=True,
        )

    storage_key = f"documents/{content_hash}/{file.filename}"
    await upload_bytes(storage_key, content)

    document = Document(
        filename=file.filename,
        source_type=source_type,
        content_hash=content_hash,
        storage_key=storage_key,
        chunking_strategy=strategy,
        file_size_bytes=len(content),
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    await session.flush()

    job = await _create_job_and_enqueue(session, document)
    await session.commit()

    logger.info("document_upload_accepted", document_id=str(document.id), job_id=str(job.id))
    return IngestionAcceptedResponse(document_id=document.id, job_id=job.id, status=document.status)


@router.post("/url", status_code=202, response_model=IngestionAcceptedResponse)
async def ingest_url(
    payload: IngestURLRequest,
    session: AsyncSession = Depends(get_db),
) -> IngestionAcceptedResponse:
    _validate_strategy(payload.chunking_strategy)
    url_str = str(payload.url)
    content_hash = hashlib.sha256(url_str.encode("utf-8")).hexdigest()

    existing = await _find_existing(session, content_hash)
    if existing is not None and not payload.force:
        return IngestionAcceptedResponse(
            document_id=existing.id,
            job_id=None,
            status=existing.status,
            duplicate=True,
        )

    document = Document(
        filename=url_str,
        source_type=SourceType.URL,
        source_url=url_str,
        content_hash=content_hash,
        chunking_strategy=payload.chunking_strategy,
        status=DocumentStatus.PENDING,
    )
    session.add(document)
    await session.flush()

    job = await _create_job_and_enqueue(session, document)
    await session.commit()

    logger.info("url_ingestion_accepted", document_id=str(document.id), job_id=str(job.id))
    return IngestionAcceptedResponse(document_id=document.id, job_id=job.id, status=document.status)


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    total = await session.scalar(select(func.count()).select_from(Document))
    result = await session.execute(
        select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    )
    documents = list(result.scalars().all())
    return DocumentListResponse(documents=documents, total=total or 0, limit=limit, offset=offset)
