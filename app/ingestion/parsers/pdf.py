import fitz

from app.ingestion.parsers.base import (
    DocumentParser, EmptyDocumentError, ParsedDocument, ParsedPage
)

class PDFParser(DocumentParser):
    def parse(self, content : bytes) -> ParsedDocument:
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"Could not open file as a PDF: {exc}") from exc

        try:
            pages = [
                ParsedPage(
                    page_number=i+1, 
                    text = page.get_text().strip()) for i, page in enumerate(doc)
            ]
        finally:
            doc.close()

        full_text = "\n\n".join(p.text for p in pages if p.text)
        if not full_text.strip():
            raise EmptyDocumentError(
                "No extractable text found in this PDF. It may be a scanned."
                "or image-only document - OCR isn't supported yet."
            )

        return ParsedDocument(text=full_text, pages=pages, metadata={"page_count": len(pages)})
        