import uuid

from pydantic import BaseModel, Field


class AssembleContextRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    document_id: uuid.UUID | None = None
    candidate_pool_size: int | None = Field(default=None, ge=1, le=100)
    final_top_n: int | None = Field(default=None, ge=1, le=20)
    max_tokens: int | None = Field(default=None, ge=100, le=20000)


class RerankedChunkResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    fusion_score: float
    rerank_score: float
    child_content: str
    content: str


class CitationResponse(BaseModel):
    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None


class AssembleContextResponse(BaseModel):
    query: str
    context_text: str
    citations: list[CitationResponse]
    total_tokens: int
    chunks_included: int
    chunks_dropped: int
    reranked_chunks: list[RerankedChunkResponse]
