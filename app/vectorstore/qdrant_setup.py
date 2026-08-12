from qdrant_client import models

from app.config import get_settings
from app.core.clients import get_qdrant_client
from app.core.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "document_chunks"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


async def ensure_collection_exists() -> None:
    """Creates the chunks collection on first boot. Safe to call on every
    startup — it's a no-op once the collection exists, which is what lets
    this live in the app lifespan rather than a separate one-off script."""
    client = get_qdrant_client()
    settings = get_settings()

    existing = await client.get_collections()
    if COLLECTION_NAME in {c.name for c in existing.collections}:
        logger.info("qdrant_collection_already_exists", collection=COLLECTION_NAME)
        return

    await client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=settings.embedding_dimensions,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # modifier=IDF tells Qdrant to apply IDF weighting computed from
            # the collection's own term statistics at query time — this is
            # what makes the sparse side behave like real BM25 rather than
            # raw term frequency.
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    logger.info("qdrant_collection_created", collection=COLLECTION_NAME)
