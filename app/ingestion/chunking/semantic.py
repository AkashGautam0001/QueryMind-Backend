import numpy as np
import pysbd

from app.ingestion.chunking.base import ChunkData, ChunkingStrategy
from app.ingestion.chunking.tokenizer import count_tokens
from app.ingestion.embeddings.base import EmbeddingProvider
from app.ingestion.parsers.base import ParsedDocument, ParsedPage


def _cosine_distance(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a_arr, b_arr) / denom)


class SemanticChunker(ChunkingStrategy):
    """Embeds individual sentences and breaks a new chunk wherever the
    semantic distance between consecutive sentences spikes — i.e. wherever
    the topic actually shifts — rather than at an arbitrary token count.
    Costs one extra embedding call per sentence at ingestion time, which is
    the trade-off for chunk boundaries that track meaning instead of length.
    """

    name = "semantic"

    def __init__(
        self,
        embedder: EmbeddingProvider,
        max_chunk_tokens: int = 500,
        breakpoint_percentile: float = 90.0,
    ):
        self.embedder = embedder
        self.max_chunk_tokens = max_chunk_tokens
        self.breakpoint_percentile = breakpoint_percentile
        self._segmenter = pysbd.Segmenter(language="en", clean=False)

    async def chunk(self, document: ParsedDocument) -> list[ChunkData]:
        pages = document.pages or [ParsedPage(page_number=None, text=document.text)]

        sentences: list[str] = []
        sentence_pages: list[int | None] = []
        for page in pages:
            if not page.text.strip():
                continue
            for sentence in self._segmenter.segment(page.text):
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
                    sentence_pages.append(page.page_number)

        if not sentences:
            return []
        if len(sentences) == 1:
            return [
                ChunkData(
                    content=sentences[0],
                    chunk_index=0,
                    page_number=sentence_pages[0],
                    token_count=count_tokens(sentences[0]),
                )
            ]

        embeddings = await self.embedder.embed_batch(sentences)
        distances = [
            _cosine_distance(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
        ]
        breakpoint_threshold = float(np.percentile(distances, self.breakpoint_percentile))

        chunks: list[ChunkData] = []
        current_sentences: list[str] = []
        current_tokens = 0
        current_page = sentence_pages[0]
        idx = 0

        def flush() -> None:
            nonlocal current_sentences, current_tokens, idx
            if not current_sentences:
                return
            content = " ".join(current_sentences)
            chunks.append(
                ChunkData(
                    content=content,
                    chunk_index=idx,
                    page_number=current_page,
                    token_count=current_tokens,
                )
            )
            idx += 1
            current_sentences = []
            current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = count_tokens(sentence)
            would_exceed_budget = current_tokens + sentence_tokens > self.max_chunk_tokens
            is_semantic_breakpoint = i > 0 and distances[i - 1] >= breakpoint_threshold

            if current_sentences and (would_exceed_budget or is_semantic_breakpoint):
                flush()
                current_page = sentence_pages[i]

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        flush()
        return chunks
