import httpx
import trafilatura

from app.ingestion.parsers.base import EmptyDocumentError, ParsedDocument, ParsedPage

_USER_AGENT = "Mozilla/5.0 (compatible; RAGIngestBot/1.0; +https://example.com/bot)"

async def fetch_url(url: str, timeout: float = 20.0) -> bytes:
    """Fetch raw HTML for a URL. Separated from extraction so the raw bytes
    can be snapshotted to object storage before any parsing happens —
    useful for audit/reproducibility if the source page changes later."""

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent":_USER_AGENT})
        response.raise_for_status()
        return response.content

def extract_from_html(html: bytes, url: str) -> ParsedDocument:
    extracted = trafilatura.extract(
        html,
        url=url,
        include_tables=False,
        favor_recall=True
    )

    if not extracted or not extracted.strip():
        raise EmptyDocumentError(f"Could not extract readable article content from {url}")

    return ParsedDocument(
        text=extracted,
        pages=[ParsedPage(page_number=None, text=extracted)],
        metadata={"url":url}
    )
    