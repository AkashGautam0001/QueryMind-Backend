import uuid
from dataclasses import dataclass

from qdrant_client import models

from app.core.clients import get_qdrant_client
from app.ingestion.embeddings.base import SparseVector
from app.vectorstore.qdrant_setup import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@dataclass
class ChunkPoint:
    point_id: uuid.UUID
    dense_vector: list[float]
    sparse_vector: SparseVector
    payload: dict


async def upsert_chunks(points: list[ChunkPoint]) -> None:
    if not points:
        return
    client = get_qdrant_client()
    qdrant_points = [
        models.PointStruct(
            id=str(p.point_id),
            vector={
                DENSE_VECTOR_NAME: p.dense_vector,
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=p.sparse_vector.indices,
                    values=p.sparse_vector.values,
                ),
            },
            payload=p.payload,
        )
        for p in points
    ]
    await client.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)


async def delete_chunks_for_document(document_id: uuid.UUID) -> None:
    """Used when re-ingesting a document — old vectors must be removed
    before new ones are written, or stale chunks would still surface in
    retrieval alongside the fresh ones."""
    client = get_qdrant_client()
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=str(document_id)),
                    )
                ]
            )
        ),
    )
