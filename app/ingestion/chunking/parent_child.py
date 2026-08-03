from app.ingestion.chunking.base import ChunkData, ChunkingStrategy
from app.ingestion.chunking.tokenizer import decode, encode
from app.ingestion.parsers.base import ParsedDocument, ParsedPage

class ParentChildChunker(ChunkingStrategy):
    name = "parent_child"

    def __init__(
        self,
        parent_tokens: int = 1200,
        child_tokens: int = 300,
        child_overlap_tokens: int = 40,
    ):
        if child_overlap_tokens >= child_tokens:
            raise ValueError("child_overlap_tokens must be smaller than child_tokens")
        if child_tokens >= parent_tokens:
            raise ValueError("child_tokens must be smaller than parent_tokens")
        self.parent_tokens = parent_tokens
        self.child_tokens = child_tokens
        self.child_overlap_tokens = child_overlap_tokens

    async def chunk(self, document: ParsedDocument) -> list[ChunkData]:
        pages = document.pages or [ParsedPage(page_number=None, text=document.text)]
        chunks: list[ChunkData] = []
        idx = 0

        for page in pages:
            if not page.text.strip():
                continue
            tokens = encode(page.text)
            p_start = 0
            while p_start < len(tokens):
                p_end = min(p_start + self.parent_tokens, len(tokens))
                parent_window = tokens[p_start: p_end]
                parent_index = idx
                chunks.append(
                    ChunkData(
                        content=decode(parent_window),
                        chunk_index=parent_index,
                        page_number=page.page_number,
                        token_count=len(parent_window),
                        is_parent=True
                    )
                )
                idx += 1

                c_start = 0
                while c_start < len(parent_window):
                    c_end = min(c_start + self.child_tokens, len(parent_window))
                    child_window = parent_window[c_start: c_end]
                    chunks.append(
                        ChunkData(
                            content=decode(child_window),
                            chunk_index=idx,
                            page_number=page.page_number,
                            token_count=len(child_window),
                            parent_index=parent_index,
                        )
                    )
                    idx += 1
                    if c_end == len(parent_window):
                        break
                    c_start = c_end - self.child_overlap_tokens

                p_start = p_end

        return chunks

        