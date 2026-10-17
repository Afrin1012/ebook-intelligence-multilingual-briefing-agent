"""Document chunking for retrieval."""

from __future__ import annotations

import re
from typing import TypedDict

from src.pdf_processor import ExtractedPdf, PageContent

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


class Chunk(TypedDict):
    chunk_id: int
    text: str
    page_number: int
    chapter_number: int | str
    chapter_title: str


def _clean_chapter_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", title or "").strip(" :;,-")
    if not cleaned:
        return "Introduction"
    return cleaned


def _roman_to_int(value: str) -> int:
    roman_map = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }
    total = 0
    for index, char in enumerate(value.upper()):
        current = roman_map.get(char, 0)
        next_value = roman_map.get(value[index + 1].upper(), 0) if index + 1 < len(value) else 0
        total += current * (-1 if current < next_value else 1)
    return total


def _normalize_chapter_number(raw_number: str) -> int | str:
    if not raw_number:
        return "Unknown"
    value = raw_number.strip()
    if value.isdigit():
        return int(value)
    if value.isalpha() and any(char.isalpha() for char in value.upper()):
        return _roman_to_int(value)
    return value


def _looks_like_subtopic(title: str) -> bool:
    clean_title = (title or "").strip()
    if not clean_title:
        return True
    lowered = clean_title.lower()
    if "?" in lowered:
        return True
    if re.match(r"^(what|why|how|when|where|who|which|types?|examples?|definitions?|features?|characteristics?|classification|categories?|components?)\b", lowered):
        return True
    return False


def _match_explicit_chapter_heading(line: str) -> tuple[int | str, str] | None:
    if not line or len(line) > 150:
        return None

    patterns = [
        r"^(?:CHAPTER|CHAP)\s+(?P<number>[IVXLCDM]+|\d+)\s*(?:[:\-–—]\s*|\s+)?(?P<title>.*)$",
        r"^(?:LECTURE|LEC)\s*[- ]*(?P<number>[IVXLCDM]+|\d+)\s*(?:[:\-–—]\s*|\s+)?(?P<title>.*)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if not match:
            continue

        chapter_number = _normalize_chapter_number(match.group("number"))
        title = _clean_chapter_title(match.group("title"))
        if chapter_number == "Unknown":
            return None
        return chapter_number, title

    return None


def _match_major_numbered_heading(line: str) -> tuple[int | str, str] | None:
    if not line or len(line) > 120:
        return None

    match = re.match(
        r"^(?P<number>[IVXLCDM]+|\d+)\s*(?:[.\):\-–—]\s*|\s+)(?P<title>.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    title = _clean_chapter_title(match.group("title"))
    if not title or title.endswith("?") or _looks_like_subtopic(title):
        return None

    return _normalize_chapter_number(match.group("number")), title


def _detect_page_header(page_text: str, allow_major_numbered: bool = True) -> tuple[int | str, str] | None:
    if not page_text:
        return None

    lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
    for index, line in enumerate(lines):
        if not line:
            continue
        explicit = _match_explicit_chapter_heading(line)
        if explicit is not None:
            return explicit
        if allow_major_numbered and index <= 3:
            major = _match_major_numbered_heading(line)
            if major is not None:
                return major
    return None


def detect_chapter_for_page(page_text: str) -> tuple[int | str, str]:
    """Detect the dominant chapter heading for a single page when one is present."""
    header = _detect_page_header(page_text, allow_major_numbered=True)
    if header is None:
        return "Unknown", "Introduction"
    return header


def _build_chapter_ranges(pages: list[PageContent], allow_major_numbered: bool) -> list[dict]:
    chapters: list[dict] = []
    current_chapter: dict | None = None

    for page in pages:
        header = _detect_page_header(page["text"], allow_major_numbered=allow_major_numbered)
        if header is None:
            if current_chapter is not None:
                current_chapter["end_page"] = page["page_number"]
            continue

        chapter_number, chapter_title = header
        if current_chapter is None:
            current_chapter = {
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "start_page": page["page_number"],
                "end_page": page["page_number"],
            }
            continue

        if current_chapter["chapter_number"] == chapter_number and current_chapter["chapter_title"] == chapter_title:
            current_chapter["end_page"] = page["page_number"]
            continue

        chapters.append(current_chapter)
        current_chapter = {
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
            "start_page": page["page_number"],
            "end_page": page["page_number"],
        }

    if current_chapter is not None:
        chapters.append(current_chapter)

    normalized: list[dict] = []
    for chapter in chapters:
        if chapter["start_page"] <= chapter["end_page"]:
            normalized.append(
                {
                    "chapter_number": chapter["chapter_number"],
                    "chapter_title": chapter["chapter_title"],
                    "start_page": chapter["start_page"],
                    "end_page": chapter["end_page"],
                }
            )

    return normalized


def detect_document_chapters(pages: list[PageContent]) -> list[dict]:
    """Return a normalized chapter list with number, title, and page range."""
    if not pages:
        return []

    explicit = _build_chapter_ranges(pages, allow_major_numbered=False)
    if explicit:
        return explicit

    major = _build_chapter_ranges(pages, allow_major_numbered=True)
    if not major:
        return []

    return major


def assign_chapters_to_pages(pages: list[PageContent]) -> list[dict]:
    """Attach chapter metadata to each page using the page text."""
    result: list[dict] = []
    chapter_ranges = detect_document_chapters(pages)
    page_to_chapter = {}
    for chapter in chapter_ranges:
        for page_number in range(chapter["start_page"], chapter["end_page"] + 1):
            page_to_chapter[page_number] = {
                "chapter_number": chapter["chapter_number"],
                "chapter_title": chapter["chapter_title"],
            }

    for page in pages:
        page_number = page["page_number"]
        if page_number in page_to_chapter:
            chapter_number = page_to_chapter[page_number]["chapter_number"]
            chapter_title = page_to_chapter[page_number]["chapter_title"]
        else:
            chapter_number, chapter_title = detect_chapter_for_page(page["text"])
        result.append(
            {
                "page_number": page_number,
                "text": page["text"],
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
            }
        )
    return result


def _chunk_page_text(
    text: str,
    page_number: int,
    chunk_size: int,
    overlap: int,
    next_id: int,
    chapter_number: int | str = "Unknown",
    chapter_title: str = "Introduction",
) -> tuple[list[Chunk], int]:
    if not text:
        return [], next_id

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(
            {
                "chunk_id": next_id,
                "text": text[start:end],
                "page_number": page_number,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
            }
        )
        next_id += 1
        if end >= length:
            break
        start += chunk_size - overlap

    return chunks, next_id


def chunk_pages(
    pages: list[PageContent],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split page-level text into overlapping chunks with stable metadata."""
    all_chunks: list[Chunk] = []
    next_id = 1

    for page in pages:
        chapter_number, chapter_title = detect_chapter_for_page(page["text"])
        page_chunks, next_id = _chunk_page_text(
            page["text"],
            page["page_number"],
            chunk_size,
            overlap,
            next_id,
            chapter_number,
            chapter_title,
        )
        all_chunks.extend(page_chunks)

    return all_chunks


def chunk_extracted_pdf(
    extracted: ExtractedPdf,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk all pages from an extracted PDF document."""
    return chunk_pages(extracted["pages"], chunk_size, overlap)


def print_detected_chapters(pages: list[PageContent]) -> None:
    """Display chapter names and page ranges for quick verification."""
    chapters = detect_document_chapters(pages)
    if not chapters:
        print("No reliable chapter structure detected.")
        return

    for chapter in chapters:
        label = f"{chapter['chapter_number']} / {chapter['chapter_title']}"
        print(f"{label} | pages {chapter['start_page']}-{chapter['end_page']}")


if __name__ == "__main__":
    sample_pages = [
        {"page_number": 1, "text": "CHAPTER 1: Introduction\nThis chapter introduces the database."},
        {"page_number": 2, "text": "Chapter 1 Introduction\nMore detail about the chapter."},
        {"page_number": 3, "text": "The next section starts here without a chapter header."},
        {"page_number": 4, "text": "CHAPTER 2: Data Models\nA model describes the structure."},
    ]
    print_detected_chapters(sample_pages)
