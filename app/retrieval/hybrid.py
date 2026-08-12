import uuid
from dataclasses import dataclass

from qdrant_client import models

from app.config import get_settings
from app.core.clients import get_qdrant_client
from app.core.embedders import get_dense_embedder, get_sparse_embedder
from app.ingestion.embeddings.base import SparseVector
from app.vectorstore.qdrant_setup import COLLECTION_NAME, DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    content: str
    page_number: int | None
    parent_chunk_id: uuid.UUID | None
    score: float


async def embed_query(query: str) -> tuple[list[float], SparseVector]:
    """Query and chunks must be embedded with the same models, or the
    vector spaces simply don't line up — this is why retrieval reuses the
    exact same embedder singletons ingestion uses, not a separate copy."""
    dense_embedder = get_dense_embedder()
    sparse_embedder = get_sparse_embedder()
    dense_vectors = await dense_embedder.embed_batch([query])
    sparse_vectors = await sparse_embedder.embed_batch([query])
    return dense_vectors[0], sparse_vectors[0]


def _build_document_filter(document_id: uuid.UUID | None) -> models.Filter | None:
    if document_id is None:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=str(document_id)),
            )
        ]
    )


async def hybrid_search(
    query: str,
    top_k: int = 10,
    prefetch_limit: int | None = None,
    document_id: uuid.UUID | None = None,
) -> list[RetrievedChunk]:
    if not query or not query.strip():
        return []

    settings = get_settings()
    prefetch_limit = prefetch_limit or settings.retrieval_prefetch_limit
    dense_vector, sparse_vector = await embed_query(query)
    query_filter = _build_document_filter(document_id)

    client = get_qdrant_client()
    response = await client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vector.indices, values=sparse_vector.values
                ),
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        # RRF combines the two prefetch rankings by reciprocal rank, not by
        # the prefetch stages' raw scores — which is exactly why a result
        # that's merely decent on both dense and sparse can outrank one
        # that's excellent on only one. Note the fused score itself is a
        # rank-based number, not a similarity — it isn't meaningful to
        # compare across different queries or use as an absolute threshold.
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    results = []
    for point in response.points:
        payload = point.payload or {}
        parent_chunk_id = payload.get("parent_chunk_id")
        results.append(
            RetrievedChunk(
                chunk_id=uuid.UUID(payload["chunk_id"]),
                document_id=uuid.UUID(payload["document_id"]),
                filename=payload.get("filename", ""),
                content=payload.get("content", ""),
                page_number=payload.get("page_number"),
                parent_chunk_id=uuid.UUID(parent_chunk_id) if parent_chunk_id else None,
                score=point.score,
            )
        )
    return results
