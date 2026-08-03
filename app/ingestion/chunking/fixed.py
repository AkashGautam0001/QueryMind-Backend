from app.ingestion.chunking.base import ChunkData, ChunkingStrategy
from app.ingestion.chunking.tokenizer import decode, encode
from app.ingestion.parsers.base import ParsedDocument, ParsedPage

class FixedSizeChunker(ChunkingStrategy):
    name = "fixed"

    def __init__(self, chunk_tokens: int = 400, overlap_tokens: int = 50):
        if overlap_tokens >= chunk_tokens :
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens

    async def chunk(self, document: ParsedDocument) -> list[ChunkData]:
        pages = document.pages or [ParsedPage(page_number=None, text=document.text)]
        chunks: list[ChunkData] = []
        idx = 0

        for page in pages:
            if not page.text.strip():
                continue
            tokens = encode(page.text)
            start = 0
            while start < len(tokens):
                end = min(start + self.chunk_tokens, len(tokens))
                window = tokens[start : end]
                chunks.append(
                    ChunkData(
                        content=decode(window),
                        chunk_index=idx,
                        page_number=page.page_number,
                        token_count=len(window)

                    )
                )
                idx += 1
                if end == len(tokens):
                    break
                start = end - self.overlap_tokens
        return chunks

        