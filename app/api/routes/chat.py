import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.logging import get_logger
from app.db.session import AsyncSessionLocal, get_db
from app.generation.events import CitationEvent, DoneEvent, NoContextEvent, TokenEvent
from app.generation.pipeline import run_chat
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.chat import ChatRequest
from app.schemas.conversation import ConversationResponse, MessageResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_SENTINEL = object()

# asyncio.create_task() only holds a weak reference to the task internally;
# without an external strong reference, the task can be garbage-collected
# mid-execution before it completes. This set exists purely to hold that
# reference until the task finishes, then discards it via add_done_callback.
_background_tasks: set[asyncio.Task] = set()


def _frame_event(event) -> dict | None:
    if isinstance(event, TokenEvent):
        return {"event": "token", "data": json.dumps({"text": event.text})}
    if isinstance(event, NoContextEvent):
        return {"event": "no_context", "data": json.dumps({"message": event.message})}
    if isinstance(event, CitationEvent):
        return {"event": "citations", "data": json.dumps({"citations": event.citations})}
    if isinstance(event, DoneEvent):
        return {
            "event": "done",
            "data": json.dumps(
                {"message_id": event.message_id, "conversation_id": event.conversation_id}
            ),
        }
    return None


async def _run_chat_into_queue(payload: ChatRequest, queue: asyncio.Queue) -> None:
    """Runs as an independent asyncio.Task, not awaited directly by the
    SSE generator. Opens its own Postgres session rather than reusing a
    request-scoped one — FastAPI closes a `Depends(get_db)` session when
    the response cycle ends, which for a dropped SSE connection happens
    while this task may still be running; sharing that session would mean
    operating on an already-closed connection.

    If the HTTP client disconnects, sse-starlette stops iterating the
    generator below — but it has no way to cancel a task it never
    awaited, so generation and the final DB commit in run_chat complete
    regardless. The frontend can reconnect and re-fetch the conversation
    (GET /chat/{conversation_id}) even if it missed the live token
    stream."""
    try:
        async with AsyncSessionLocal() as session:
            async for event in run_chat(
                query=payload.query,
                session=session,
                conversation_id=payload.conversation_id,
                document_id=payload.document_id,
            ):
                await queue.put(event)
    except Exception:
        logger.error("chat_background_task_failed", exc_info=True)
    finally:
        await queue.put(_SENTINEL)


async def _event_stream(payload: ChatRequest):
    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(_run_chat_into_queue(payload, queue))
    # Deliberately not awaited here — creating the task (and never awaiting
    # it directly) is what decouples it from this generator's own
    # cancellation. If the client disconnects, sse-starlette stops
    # iterating this generator, but the task keeps running independently
    # in the event loop until it finishes and persists to Postgres.
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        framed = _frame_event(item)
        if framed is not None:
            yield framed


@router.post("/stream")
async def chat_stream(payload: ChatRequest):
    return EventSourceResponse(_event_stream(payload))


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: uuid.UUID, session: AsyncSession = Depends(get_db)
) -> Conversation:
    """Lets the frontend reload full conversation history — including the
    case where an SSE connection dropped mid-stream: the answer was still
    generated and persisted server-side (see _run_chat_into_queue), so the
    client can always recover it here even if it missed the live tokens."""
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = list(result.scalars().all())

    return ConversationResponse(
        id=conversation.id,
        created_at=conversation.created_at,
        messages=[MessageResponse.model_validate(m, from_attributes=True) for m in messages],
    )
