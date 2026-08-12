import uuid
from datetime import datetime

from pydantic import BaseModel

from app.models.message import MessageRole


class MessageResponse(BaseModel):
    id: uuid.UUID
    role: MessageRole
    content: str
    citations: list[dict] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    messages: list[MessageResponse]
