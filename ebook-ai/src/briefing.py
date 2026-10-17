"""Chapter briefing generation using the existing Gemini configuration."""

from __future__ import annotations

import os
import time

from google.genai import types

from src.llm import DEFAULT_LLM_MODEL, _extract_text, _get_client

SUPPORTED_BRIEFING_LANGUAGES = (
    "English",
    "Hindi",
    "Odia",
    "Bengali",
    "Telugu",
    "Tamil",
    "Marathi",
    "Gujarati",
    "Kannada",
    "Malayalam",
)


def _chapter_label(chapter_number: int | str, chapter_title: str) -> str:
    if chapter_number == "Unknown":
        return "Unknown / Introduction"
    return f"Chapter {chapter_number} — {chapter_title}"


def _build_briefing_prompt(
    chapter_title: str,
    chapter_number: int | str,
    chapter_content: str,
    language: str = "English",
) -> str:
    chapter_label = _chapter_label(chapter_number, chapter_title)
    return (
        f"Chapter to brief: {chapter_label}\n\n"
        "Create a concise, student-facing chapter briefing for exam revision. Summarize and synthesize the chapter's "
        "main ideas rather than reproducing its definitions or paraphrasing it sentence by sentence. Use ONLY the "
        "provided chapter content below; do not use outside knowledge or invent facts, explanations, examples, dates, "
        "names, or page numbers. Do not mix in information from other chapters.\n\n"
        f"Write the entire briefing in {language}, using its natural writing system. Translate all four section "
        "headings naturally into that language, while keeping these section roles and this order: Overview, Key Concepts, "
        "Important Points, Key Takeaway. Analyze the original chapter content directly; do not translate it before "
        "analysis. Preserve important technical terms, formulas, names, and terminology from the source.\n\n"
        "Chapter content:\n"
        f"{chapter_content.strip()}\n\n"
        "Write exactly four Markdown sections, in the specified order, and nothing else. Make the briefing "
        "significantly shorter than the source, aiming for about 10–20% of its useful information depending on chapter "
        "length. Avoid filler, repetition, dictionary-style definition lists, copied paragraphs, and minor details.\n\n"
        "## Overview\n"
        "Write one concise paragraph describing what the chapter is about and what the reader should understand.\n\n"
        "## Key Concepts\n"
        "Select only the most important concepts. Aim for 4–7; combine related concepts where useful and explain each "
        "briefly in context instead of copying or translating every definition. Use concise bullets.\n\n"
        "## Important Points\n"
        "Select 4–7 meaningful ideas, relationships, processes, comparisons, or facts that matter most. Avoid repeating "
        "the source sentence by sentence. Use concise bullets. Use fewer only when the chapter does not support more.\n\n"
        "## Key Takeaway\n"
        "Write one short concluding paragraph that expresses the chapter's central understanding, not another definition."
    )


def generate_chapter_briefing(
    chapter_title: str,
    chapter_number: int | str,
    chapter_content: str,
    language: str = "English",
) -> str:
    """Generate a chapter briefing in the selected language from chapter-only text."""
    content = (chapter_content or "").strip()
    if not content:
        return "No chapter content was found for the selected chapter."
    if language not in SUPPORTED_BRIEFING_LANGUAGES:
        raise ValueError(f"Unsupported briefing language: {language}")

    client = _get_client()
    chosen_model = os.getenv("GEMINI_MODEL", DEFAULT_LLM_MODEL)
    prompt = _build_briefing_prompt(chapter_title, chapter_number, content, language)

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=chosen_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=1024,
                    response_mime_type="text/plain",
                ),
            )
            answer = _extract_text(response)
            if not answer:
                return "I could not generate a briefing for the selected chapter from the provided chapter content."
            return answer.strip()
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
            raise

    if last_error is not None:
        raise last_error

    return "I could not generate a briefing for the selected chapter from the provided chapter content."
