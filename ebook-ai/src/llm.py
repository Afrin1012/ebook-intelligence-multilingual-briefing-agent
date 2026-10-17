"""LLM client and generation helpers."""

from __future__ import annotations

import os
import re
import time
from typing import TypedDict

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.prompts import (
    GROUNDED_QA_SYSTEM_PROMPT,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    build_grounded_user_message,
)
from src.retriever import RetrievalResult

DEFAULT_LLM_MODEL = "gemini-3.5-flash"


class SourceCitation(TypedDict):
    chunk_id: int
    page_number: int
    chapter: str | None


class GroundedAnswer(TypedDict):
    question: str
    answer: str
    sources: list[SourceCitation]
    sufficient_evidence: bool


def _get_client() -> genai.Client:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment to use Ask the Book."
        )
    return genai.Client(api_key=api_key)


def _clean_answer_text(answer: str) -> str:
    cleaned = (answer or "").strip()
    if not cleaned:
        return ""

    patterns = [
        r"\s*\(\s*(?:pages?|pp?\.?|source(?:s)?)(?:\s*[:#-])?\s*[^)]*\)\s*",
        r"\s*(?:pages?|pp?\.?|source(?:s)?)(?:\s*[:#-])?\s*\d+(?:\s*[-–—]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-–—]\s*\d+)?)*(?:\s*\.|\s|$)",
    ]

    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" .;:-")
    return cleaned


def _extract_text(response: object) -> str:
    text = getattr(response, "text", "")
    if isinstance(text, str) and text.strip():
        return _clean_answer_text(text)

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""

    parts = getattr(candidates[0], "content", None)
    if parts is None:
        return ""

    content_parts = getattr(parts, "parts", None) or []
    collected: list[str] = []
    for part in content_parts:
        part_text = getattr(part, "text", None)
        if isinstance(part_text, str):
            collected.append(part_text)
    return _clean_answer_text("".join(collected))


def _evidence_to_sources(evidence: list[RetrievalResult]) -> list[SourceCitation]:
    seen: set[int] = set()
    sources: list[SourceCitation] = []
    for item in evidence:
        chunk_id = item["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        sources.append(
            {
                "chunk_id": chunk_id,
                "page_number": item["page_number"],
                "chapter": item.get("chapter"),
            }
        )
    return sources


def answer_from_evidence(
    question: str,
    evidence: list[RetrievalResult],
    model: str | None = None,
) -> GroundedAnswer:
    """
    Generate an answer using only the provided reranked evidence chunks.
    """
    cleaned_question = question.strip()
    if not cleaned_question or not evidence:
        return {
            "question": cleaned_question,
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "sources": [],
            "sufficient_evidence": False,
        }

    client = _get_client()
    chosen_model = model or os.getenv("GEMINI_MODEL", DEFAULT_LLM_MODEL)
    prompt = build_grounded_user_message(cleaned_question, evidence)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=chosen_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=GROUNDED_QA_SYSTEM_PROMPT,
                    temperature=0,
                    max_output_tokens=512,
                    response_mime_type="text/plain",
                ),
            )
            answer = _extract_text(response)
            insufficient = not answer or answer == INSUFFICIENT_EVIDENCE_MESSAGE

            if insufficient:
                return {
                    "question": cleaned_question,
                    "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
                    "sources": [],
                    "sufficient_evidence": False,
                }

            return {
                "question": cleaned_question,
                "answer": answer,
                "sources": _evidence_to_sources(evidence),
                "sufficient_evidence": True,
            }
        except Exception as exc:
            last_error = exc
            message = str(exc)
            lower_message = message.lower()
            if (
                "503" in message
                or "unavailable" in lower_message
                or "high demand" in lower_message
            ) and attempt < 2:
                time.sleep(1)
                continue
            if (
                "503" in message
                or "unavailable" in lower_message
                or "high demand" in lower_message
            ) and attempt == 2:
                return {
                    "question": cleaned_question,
                    "answer": "Gemini is temporarily unavailable due to high demand. Please try again in a moment.",
                    "sources": [],
                    "sufficient_evidence": False,
                }
            raise

    if last_error is not None:
        raise last_error

    return {
        "question": cleaned_question,
        "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "sources": [],
        "sufficient_evidence": False,
    }
