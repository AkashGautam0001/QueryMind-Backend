import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import BigInteger, Integer, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.ingestion_job import IngestionJob

class SourceType(enum.StrEnum):
    PDF="pdf"
    DOCX="docx"
    URL="url"

class DocumentStatus(enum.StrEnum):
    PENDING="pending"
    PROCESSIONG="processing"
    COMPLETED="completed"
    FAILED="failed"

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512))
    source_type: Mapped[SourceType] = mapped_column(
        SAEnum(SourceType, name="source_type", values_callable=lambda obj: [e.value for e in obj])
    )
    source_url: Mapped[str | None] = mapped_column(String(2028), nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(
            DocumentStatus,
            name="document_Status",
            values_callable=lambda obj: [e.value for e in obj],
        ),
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value
    )
    chunking_strategy: Mapped[str] = mapped_column(String(50))
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    
    jobs: Mapped[list["IngestionJob"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )