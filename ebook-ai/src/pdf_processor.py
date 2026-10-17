"""PDF ingestion and text extraction."""

from __future__ import annotations

from typing import TypedDict

import fitz


class PageContent(TypedDict):
    page_number: int
    text: str


class ExtractedPdf(TypedDict):
    pages: list[PageContent]


def clean_whitespace(text: str) -> str:
    """Strip lines, collapse internal spaces, and trim excess blank lines."""
    if not text:
        return ""

    lines = [" ".join(line.split()) for line in text.splitlines()]

    collapsed: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and prev_blank:
            continue
        collapsed.append(line)
        prev_blank = is_blank

    return "\n".join(collapsed).strip()


def extract_pdf(pdf_bytes: bytes) -> ExtractedPdf:
    """Extract text from each page of a PDF, with 1-based page numbers."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pages: list[PageContent] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            raw_text = page.get_text()
            pages.append(
                {
                    "page_number": index + 1,
                    "text": clean_whitespace(raw_text),
                }
            )
        return {"pages": pages}
    finally:
        doc.close()
