from dataclasses import dataclass,field

class EmptyDocumentError(Exception):
    """Raised when a document yields no extractable text — most commonly a
    scanned/image-only PDF that would need OCR, which this pipeline doesn't
    do yet. Ingestion should fail loudly here rather than silently indexing
    nothing and leaving the user wondering why retrieval finds no results."""

@dataclass
class ParsedPage:
    page_number: int | None
    text: str

@dataclass
class ParsedDocument:
    text: str
    pages: list[ParsedPage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

class DocumentParser:
    def parse(self, content: bytes) -> ParsedDocument:
        raise NotImplementedError