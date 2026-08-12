from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.retrieval.context import assemble_context, retrieve_and_rerank
from app.schemas.context import (
    AssembleContextRequest,
    AssembleContextResponse,
    CitationResponse,
    RerankedChunkResponse,
)

logger = get_logger(__name__)
router = APIRouter(prefix="/context", tags=["context"])


@router.post("", response_model=AssembleContextResponse)
async def build_context(
    payload: AssembleContextRequest,
    session: AsyncSession = Depends(get_db),
) -> AssembleContextResponse:
    reranked = await retrieve_and_rerank(
        query=payload.query,
        session=session,
        document_id=payload.document_id,
        candidate_pool_size=payload.candidate_pool_size,
        final_top_n=payload.final_top_n,
    )

    assembled = assemble_context(reranked, max_tokens=payload.max_tokens)

    logger.info(
        "context_assembled",
        query=payload.query,
        candidates_reranked=len(reranked),
        chunks_included=assembled.chunks_included,
        chunks_dropped=assembled.chunks_dropped,
    )

    return AssembleContextResponse(
        query=payload.query,
        context_text=assembled.context_text,
        citations=[
            CitationResponse.model_validate(c, from_attributes=True) for c in assembled.citations
        ],
        total_tokens=assembled.total_tokens,
        chunks_included=assembled.chunks_included,
        chunks_dropped=assembled.chunks_dropped,
        reranked_chunks=[
            RerankedChunkResponse.model_validate(c, from_attributes=True) for c in reranked
        ],
    )
