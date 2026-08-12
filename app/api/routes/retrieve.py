from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import get_db
from app.retrieval.expansion import expand_to_parents
from app.retrieval.hybrid import hybrid_search
from app.schemas.retrieval import RetrievedChunkResponse, RetrieveRequest, RetrieveResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("", response_model=RetrieveResponse)
async def retrieve(
    payload: RetrieveRequest,
    session: AsyncSession = Depends(get_db),
) -> RetrieveResponse:
    results = await hybrid_search(
        query=payload.query,
        top_k=payload.top_k,
        document_id=payload.document_id,
    )

    parent_content_by_id: dict = {}
    if payload.expand_to_parent:
        parent_content_by_id = await expand_to_parents(session, results)

    logger.info("retrieval_query", query=payload.query, result_count=len(results))

    return RetrieveResponse(
        query=payload.query,
        results=[
            RetrievedChunkResponse(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                filename=r.filename,
                content=r.content,
                page_number=r.page_number,
                score=r.score,
                parent_chunk_id=r.parent_chunk_id,
                parent_content=(
                    parent_content_by_id.get(r.parent_chunk_id) if r.parent_chunk_id else None
                ),
            )
            for r in results
        ],
    )
