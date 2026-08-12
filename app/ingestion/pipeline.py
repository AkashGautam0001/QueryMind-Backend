import uuid
from datetime import UTC, datetime

from app.core.embedders import get_dense_embedder, get_sparse_embedder
from app.core.logging import get_logger
from app.core.storage import download_bytes, upload_bytes
from app.db.session import AsyncSessionLocal
from app.ingestion.chunking import ChunkData, get_chunker
from app.ingestion.parsers.base import ParsedDocument
from app.ingestion.parsers.docx import DocxParser
from app.ingestion.parsers.pdf import PDFParser
from app.ingestion.parsers.url import extract_from_html, fetch_url
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus, SourceType
from app.models.ingestion_job import IngestionJob, IngestionJobStatus
from app.vectorstore.writer import ChunkPoint, delete_chunks_for_document, upsert_chunks

logger = get_logger(__name__)

_PARSERS = {
    SourceType.PDF: PDFParser(),
    SourceType.DOCX: DocxParser(),
}


class IngestionError(Exception):
    """Raised for failures with a message that's safe to show end users —
    as opposed to letting a raw exception's message leak into the API."""


async def _parse_document(document: Document) -> ParsedDocument:
    if document.source_type == SourceType.URL:
        html = await fetch_url(document.source_url)
        # Snapshot raw HTML before extraction — if the live page changes or
        # disappears later, we still have what was actually indexed.
        snapshot_key = f"documents/{document.content_hash}/snapshot.html"
        await upload_bytes(snapshot_key, html)
        return extract_from_html(html, document.source_url)

    parser = _PARSERS.get(document.source_type)
    if parser is None:
        raise IngestionError(f"No parser registered for source type {document.source_type}")
    raw_bytes = await download_bytes(document.storage_key)
    return parser.parse(raw_bytes)


def _build_chunk_rows(
    document_id: uuid.UUID, chunk_data_list: list[ChunkData]
) -> list[Chunk]:
    # Generate every chunk's id up front so children can resolve their
    # parent_chunk_id regardless of insertion order, without depending on
    # SQLAlchemy's flush-time default timing.
    chunk_ids = {data.chunk_index: uuid.uuid4() for data in chunk_data_list}

    rows = []
    for data in chunk_data_list:
        parent_id = chunk_ids[data.parent_index] if data.parent_index is not None else None
        rows.append(
            Chunk(
                id=chunk_ids[data.chunk_index],
                document_id=document_id,
                parent_chunk_id=parent_id,
                chunk_index=data.chunk_index,
                content=data.content,
                page_number=data.page_number,
                token_count=data.token_count,
                is_parent=data.is_parent,
            )
        )
    return rows


async def run_ingestion(document_id: uuid.UUID, job_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as session:
        document = await session.get(Document, document_id)
        job = await session.get(IngestionJob, job_id)
        if document is None or job is None:
            logger.error(
                "ingestion_job_missing_records",
                document_id=str(document_id),
                job_id=str(job_id),
            )
            return
        document.status = DocumentStatus.PROCESSING
        job.status = IngestionJobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        await session.commit()

    try:
        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)

            parsed = await _parse_document(document)

            dense_embedder = get_dense_embedder()
            sparse_embedder = get_sparse_embedder()

            chunker = get_chunker(document.chunking_strategy, embedder=dense_embedder)
            chunk_data_list = await chunker.chunk(parsed)

            if not chunk_data_list:
                raise IngestionError("Chunking produced zero chunks from this document.")

            chunk_rows = _build_chunk_rows(document.id, chunk_data_list)
            for row in chunk_rows:
                session.add(row)
            await session.flush()  # populates each row's qdrant_point_id default

            # Only non-parent chunks are embedded and searched — parents
            # exist purely for context expansion at generation time.
            searchable_pairs = [
                (data, row)
                for data, row in zip(chunk_data_list, chunk_rows, strict=True)
                if not data.is_parent
            ]
            texts = [data.content for data, _ in searchable_pairs]
            dense_vectors = await dense_embedder.embed_batch(texts)
            sparse_vectors = await sparse_embedder.embed_batch(texts)

            points = [
                ChunkPoint(
                    point_id=row.qdrant_point_id,
                    dense_vector=dense_vectors[i],
                    sparse_vector=sparse_vectors[i],
                    payload={
                        "document_id": str(document.id),
                        "chunk_id": str(row.id),
                        "parent_chunk_id": (
                            str(row.parent_chunk_id) if row.parent_chunk_id else None
                        ),
                        "content": row.content,
                        "page_number": row.page_number,
                        "filename": document.filename,
                    },
                )
                for i, (_, row) in enumerate(searchable_pairs)
            ]

            # Replace rather than append, so re-ingesting a document can't
            # leave stale vectors from a previous version behind.
            await delete_chunks_for_document(document.id)
            await upsert_chunks(points)

            document.status = DocumentStatus.COMPLETED
            document.chunk_count = len(searchable_pairs)
            document.error_message = None

            job = await session.get(IngestionJob, job_id)
            job.status = IngestionJobStatus.COMPLETED
            job.completed_at = datetime.now(UTC)

            await session.commit()
            logger.info(
                "ingestion_completed",
                document_id=str(document_id),
                chunk_count=len(searchable_pairs),
                total_chunks_with_parents=len(chunk_rows),
            )

    except Exception as exc:
        logger.error("ingestion_failed", document_id=str(document_id), error=str(exc))
        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            job = await session.get(IngestionJob, job_id)
            if document is not None:
                document.status = DocumentStatus.FAILED
                document.error_message = str(exc)
            if job is not None:
                job.status = IngestionJobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.now(UTC)
            await session.commit()
