import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.models.evaluation import EvalRunStatus


class EvalQuestionCreate(BaseModel):
    question: str = Field(min_length=5, max_length=2000)
    expected_answer: str = Field(min_length=5, max_length=5000)
    document_id: uuid.UUID | None = None


class EvalQuestionResponse(BaseModel):
    id: uuid.UUID
    question: str
    expected_answer: str
    document_id: uuid.UUID | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalRunResponse(BaseModel):
    id: uuid.UUID
    status: EvalRunStatus
    question_count: int
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
    mean_faithfulness: float | None
    mean_answer_relevancy: float | None
    mean_context_precision: float | None
    mean_context_recall: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class EvalResultResponse(BaseModel):
    id: uuid.UUID
    question_id: uuid.UUID
    generated_answer: str
    retrieved_context: list
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    judge_reasoning: dict

    model_config = {"from_attributes": True}


class EvalRunDetailResponse(EvalRunResponse):
    results: list[EvalResultResponse]
