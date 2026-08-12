import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chunk import Chunk
from app.retrieval.hybrid import RetrievedChunk


async def expand_to_parents(
    session: AsyncSession, results: list[RetrievedChunk]
) -> dict[uuid.UUID, str]:
    """Returns {parent_chunk_id: parent_content} for every retrieved chunk
    that has a parent — single batched query, not one lookup per result."""
    parent_ids = {r.parent_chunk_id for r in results if r.parent_chunk_id is not None}
    if not parent_ids:
        return {}
    rows = await session.execute(select(Chunk).where(Chunk.id.in_(parent_ids)))
    return {row.id: row.content for row in rows.scalars().all()}
