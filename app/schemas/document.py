import uuid
from datetime import datetime

from pydantic import BaseModel, Field, HttpUrl

from app.models.document import DocumentStatus, SourceType


class IngestURLRequest(BaseModel):
    url: HttpUrl
    chunking_strategy: str = Field(default="parent_child")
    force: bool = Field(
        default=False,
        description="Re-ingest even if this URL was already ingested before.",
    )


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    source_type: SourceType
    source_url: str | None
    status: DocumentStatus
    chunking_strategy: str
    chunk_count: int
    file_size_bytes: int | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
    limit: int
    offset: int


class IngestionAcceptedResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID | None
    status: DocumentStatus
    duplicate: bool = Field(
        default=False,
        description="True if this content was already ingested — no new job was enqueued.",
    )
