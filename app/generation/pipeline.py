import uuid
from collections.abc import AsyncIterator

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.llm import get_llm_provider
from app.core.logging import get_logger
from app.core.semantic_cache import cache_lookup, cache_store
from app.generation.events import CitationEvent, DoneEvent, NoContextEvent, TokenEvent
from app.generation.prompt import NO_RELEVANT_CONTEXT_MESSAGE, build_system_prompt
from app.generation.relevance import has_relevant_context
from app.generation.verification import verify_citations
from app.models.conversation import Conversation
from app.models.message import Message, MessageRole
from app.retrieval.context import Citation, assemble_context, retrieve_and_rerank
from app.retrieval.hybrid import embed_query

logger = get_logger(__name__)


class _CitationWithContent:
    """verify_citations needs each citation's actual source text to check
    overlap; the Citation dataclass from context assembly doesn't carry
    content (by design — it's the lightweight public-facing shape). This
    pairs the two for the verification step only."""

    def __init__(self, citation: Citation, content: str):
        self.chunk_id = citation.chunk_id
        self.document_id = citation.document_id
        self.filename = citation.filename
        self.page_number = citation.page_number
        self.content = content


def _citation_to_dict(v) -> dict:
    return {
        "index": v.index,
        "chunk_id": str(v.chunk_id),
        "document_id": str(v.document_id),
        "filename": v.filename,
        "page_number": v.page_number,
    }


async def _load_history(session: AsyncSession, conversation_id: uuid.UUID) -> list[dict]:
    settings = get_settings()
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = list(result.scalars().all())
    recent = messages[-settings.conversation_history_max_messages :]
    return [{"role": m.role.value, "content": m.content} for m in recent]


async def run_chat(
    query: str,
    session: AsyncSession,
    conversation_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
) -> AsyncIterator[TokenEvent | CitationEvent | DoneEvent | NoContextEvent]:
    settings = get_settings()

    conversation = await session.get(Conversation, conversation_id) if conversation_id else None
    history = await _load_history(session, conversation.id) if conversation is not None else []
    if conversation is None:
        conversation = Conversation()
        session.add(conversation)
        await session.flush()

    user_message = Message(conversation_id=conversation.id, role=MessageRole.USER, content=query)
    session.add(user_message)
    await session.flush()
    await session.commit()

    # Semantic cache check — embed the query once and reuse the vector for
    # both the cache lookup and (on a miss) the hybrid search.
    query_dense, _ = await embed_query(query)
    doc_id_str = str(document_id) if document_id else None
    cached = await cache_lookup(query_dense, document_id=doc_id_str)
    if cached:
        logger.info("semantic_cache_hit_served", query=query)
        for token in cached["answer"].split(" "):
            yield TokenEvent(text=token + " ")
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=cached["answer"],
            citations=cached["citations"],
        )
        session.add(assistant_message)
        await session.commit()
        yield CitationEvent(citations=cached["citations"])
        yield DoneEvent(
            message_id=str(assistant_message.id),
            conversation_id=str(conversation.id),
        )
        return

    reranked = await retrieve_and_rerank(query, session=session, document_id=document_id)

    if not has_relevant_context(reranked, settings.relevance_score_threshold):
        logger.info("chat_no_relevant_context", query=query, candidate_count=len(reranked))
        assistant_message = Message(
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content=NO_RELEVANT_CONTEXT_MESSAGE,
            citations=None,
        )
        session.add(assistant_message)
        await session.commit()
        yield NoContextEvent(message=NO_RELEVANT_CONTEXT_MESSAGE)
        yield DoneEvent(message_id=str(assistant_message.id), conversation_id=str(conversation.id))
        return

    assembled = assemble_context(reranked)
    system_prompt = build_system_prompt(assembled.context_text)
    chat_messages = [*history, {"role": "user", "content": query}]

    provider = get_llm_provider()
    answer_parts: list[str] = []
    async for delta in provider.stream(
        system_prompt=system_prompt,
        messages=chat_messages,
        max_tokens=settings.generation_max_tokens,
        temperature=settings.generation_temperature,
    ):
        answer_parts.append(delta)
        yield TokenEvent(text=delta)

    full_answer = "".join(answer_parts)

    content_by_chunk_id = {rc.chunk_id: rc.content for rc in reranked}
    enriched_by_index = {
        c.index: _CitationWithContent(c, content_by_chunk_id.get(c.chunk_id, ""))
        for c in assembled.citations
    }

    cleaned_text, verified = verify_citations(full_answer, enriched_by_index)
    verified_citations = [v for v in verified if v.verified]

    assistant_message = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content=cleaned_text,
        citations=[_citation_to_dict(v) for v in verified_citations],
    )
    session.add(assistant_message)
    await session.commit()

    # Fire-and-forget cache store — a failure here should never surface to
    # the user, so we don't await with error propagation.
    citation_dicts = [_citation_to_dict(v) for v in verified_citations]
    await cache_store(query_dense, cleaned_text, citation_dicts, document_id=doc_id_str)

    logger.info(
        "chat_generation_completed",
        conversation_id=str(conversation.id),
        citation_count_total=len(verified),
        citation_count_verified=len(verified_citations),
    )

    yield CitationEvent(citations=citation_dicts)
    yield DoneEvent(message_id=str(assistant_message.id), conversation_id=str(conversation.id))
