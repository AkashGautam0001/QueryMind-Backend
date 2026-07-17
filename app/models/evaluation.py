import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, Text
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.eval_question import EvalQuestion


class EvalRunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EvalRun(Base, TimestampMixin):
    __tablename__ = "eval_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status: Mapped[EvalRunStatus] = mapped_column(
        SAEnum(EvalRunStatus, name="eval_run_status", values_callable=lambda obj: [e.value for e in obj]),
        default=EvalRunStatus.PENDING,
        server_default=EvalRunStatus.PENDING.value,
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    question_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    mean_faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_answer_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_context_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_context_recall: Mapped[float | None] = mapped_column(Float, nullable=True)

    results: Mapped[list["EvalResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalResult(Base, TimestampMixin):
    __tablename__ = "eval_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("eval_questions.id", ondelete="CASCADE"), index=True
    )
    generated_answer: Mapped[str] = mapped_column(Text)
    retrieved_context: Mapped[list] = mapped_column(JSONB)
    faithfulness: Mapped[float] = mapped_column(Float)
    answer_relevancy: Mapped[float] = mapped_column(Float)
    context_precision: Mapped[float] = mapped_column(Float)
    context_recall: Mapped[float] = mapped_column(Float)
    # Raw judge reasoning kept for debugging surprising scores
    judge_reasoning: Mapped[dict] = mapped_column(JSONB)

    run: Mapped["EvalRun"] = relationship(back_populates="results")
    question: Mapped["EvalQuestion"] = relationship(back_populates="results")
