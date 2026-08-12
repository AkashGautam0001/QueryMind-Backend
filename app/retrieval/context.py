import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.reranker import get_reranker
from app.ingestion.chunking.tokenizer import count_tokens
from app.retrieval.expansion import expand_to_parents
from app.retrieval.hybrid import hybrid_search


@dataclass
class RerankedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None
    fusion_score: float
    rerank_score: float
    # The precise child snippet that actually matched and was reranked —
    # kept separate from `content` so callers can see what the reranker
    # actually scored versus what ends up in the final prompt.
    child_content: str
    # What goes into the assembled prompt: the expanded parent content if
    # one exists, otherwise the child content itself.
    content: str


@dataclass
class Citation:
    index: int
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page_number: int | None


@dataclass
class AssembledContext:
    context_text: str
    citations: list[Citation] = field(default_factory=list)
    total_tokens: int = 0
    chunks_included: int = 0
    chunks_dropped: int = 0


async def retrieve_and_rerank(
    query: str,
    session: AsyncSession,
    document_id: uuid.UUID | None = None,
    candidate_pool_size: int | None = None,
    final_top_n: int | None = None,
) -> list[RerankedChunk]:
    """Two-stage retrieval: a wide, cheap recall pass (hybrid fusion) over
    `candidate_pool_size` candidates, then a narrow, expensive precision
    pass (cross-encoder) that picks the best `final_top_n`. Reranking
    scores the short child snippets, not parent-expanded text — keeps the
    cross-encoder within its effective sequence length and keeps the
    signal focused on what actually matched. Parent expansion only runs
    on the survivors, since it's a DB round trip you don't want to pay for
    candidates that get discarded anyway."""
    settings = get_settings()
    candidate_pool_size = candidate_pool_size or settings.rerank_candidate_pool_size
    final_top_n = final_top_n or settings.rerank_final_top_n

    candidates = await hybrid_search(
        query, top_k=candidate_pool_size, document_id=document_id
    )
    if not candidates:
        return []

    reranker = get_reranker()
    scores = await reranker.rerank(query, [c.content for c in candidates])

    ranked = sorted(
        zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    survivors = ranked[:final_top_n]

    parent_content_by_id = await expand_to_parents(session, [c for c, _ in survivors])

    return [
        RerankedChunk(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            filename=c.filename,
            page_number=c.page_number,
            fusion_score=c.score,
            rerank_score=score,
            child_content=c.content,
            content=parent_content_by_id.get(c.parent_chunk_id, c.content),
        )
        for c, score in survivors
    ]


def assemble_context(
    chunks: list[RerankedChunk], max_tokens: int | None = None
) -> AssembledContext:
    """Greedily fills the token budget in rank order. A chunk that
    wouldn't fit is skipped (not a hard stop) so a smaller, lower-ranked
    chunk later in the list still gets a chance — better budget
    utilization than bailing out on the first chunk that doesn't fit.
    Citation numbers are sequential among chunks actually included, so a
    skipped chunk never leaves a gap like [1], [3] in the visible
    context."""
    settings = get_settings()
    max_tokens = max_tokens if max_tokens is not None else settings.context_max_tokens

    parts: list[str] = []
    citations: list[Citation] = []
    total_tokens = 0
    dropped = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk.content)
        if total_tokens + chunk_tokens > max_tokens:
            dropped += 1
            continue

        citation_index = len(citations) + 1
        page_info = f", page {chunk.page_number}" if chunk.page_number else ""
        parts.append(f"[{citation_index}] (source: {chunk.filename}{page_info})\n{chunk.content}")
        citations.append(
            Citation(
                index=citation_index,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
            )
        )
        total_tokens += chunk_tokens

    return AssembledContext(
        context_text="\n\n".join(parts),
        citations=citations,
        total_tokens=total_tokens,
        chunks_included=len(citations),
        chunks_dropped=dropped,
    )
