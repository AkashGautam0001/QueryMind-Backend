import uuid

from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    document_id: uuid.UUID | None = Field(
        default=None, description="Restrict search to a single document."
    )
    expand_to_parent: bool = Field(
        default=False,
        description="Also fetch each result's full parent-chunk content from Postgres.",
    )


class RetrievedChunkResponse(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    score: float
    parent_chunk_id: uuid.UUID | None
    parent_content: str | None = None


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedChunkResponse]
