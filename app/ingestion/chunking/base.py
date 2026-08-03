from dataclasses import dataclass

from app.ingestion.parsers.base import ParsedDocument

@dataclass
class ChunkData:
    content : str
    chunk_index : int
    page_number : int | None
    token_count : int
    is_parent: bool = False
    parent_index : int | None = None

class ChunkingStrategy:
    name : str = "Base"
    async def chunk(self, document: ParsedDocument) -> list[ChunkData]:
        raise NotImplementedError