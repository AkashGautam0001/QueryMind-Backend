import uuid

from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    conversation_id: uuid.UUID | None = None
    document_id: uuid.UUID | None = None