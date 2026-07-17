import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.document import Document

class IngestionJobStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPELEED = "completed"
    FAILED = "failed"

class IngestionJob(Base, TimestampMixin):
    __tablename__ = "ingesion_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id : Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index =True
    )
    status: Mapped[IngestionJobStatus] = mapped_column(
        SAEnum(
            IngestionJobStatus,
            name="ingestion_job_status",
            value_callable = lambda obj: [e.value for e in obj]
        ),
        default=IngestionJobStatus.PENDING,
        server_default=IngestionJobStatus.PENDING.value
    )
    chunking_strategy: Mapped[str] = mapped_column(String(50))
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="jobs")