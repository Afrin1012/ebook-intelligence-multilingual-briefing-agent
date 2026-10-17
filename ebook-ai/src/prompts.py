"""Prompt templates for the briefing agent."""

from __future__ import annotations

from src.retriever import RetrievalResult

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find sufficient evidence for this question in the uploaded book."
)

GROUNDED_QA_SYSTEM_PROMPT = f"""You are a grounded question-answering assistant for an uploaded e-book.

Rules:
- Answer only from the provided e-book evidence.
- Do not use outside knowledge.
- Do not invent facts.
- Do not invent page numbers.
- Do not add information that is not supported by the evidence.
- Answer the user's exact question directly.
- Do not repeat the entire retrieved passage.
- Keep answers concise and directly relevant to the question.
- If the user asks for a definition, give only the definition and closely relevant explanation.
- If the user asks about advantages, answer the advantages only.
- If the user asks for a comparison, answer the comparison only.
- Include the page number(s) from the evidence that support your answer.
- If the evidence is insufficient to answer the question, reply with exactly this sentence and nothing else:
  "{INSUFFICIENT_EVIDENCE_MESSAGE}"
"""


def format_evidence_for_prompt(evidence: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for index, item in enumerate(evidence, start=1):
        chapter_line = (
            f" | chapter: {item['chapter']}" if item.get("chapter") else ""
        )
        blocks.append(
            f"[Evidence {index} | page {item['page_number']} | "
            f"chunk_id {item['chunk_id']}{chapter_line}]\n{item['text']}"
        )
    return "\n\n".join(blocks)


def build_grounded_user_message(
    question: str,
    evidence: list[RetrievalResult],
) -> str:
    return (
        f"User question:\n{question.strip()}\n\n"
        f"E-book evidence (the only source you may use):\n"
        f"{format_evidence_for_prompt(evidence)}\n\n"
        "Write a concise answer that cites the supporting page number(s)."
    )
