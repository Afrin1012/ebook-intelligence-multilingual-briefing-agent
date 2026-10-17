import hashlib
import re

import streamlit as st

from src.briefing import SUPPORTED_BRIEFING_LANGUAGES, generate_chapter_briefing
from src.chunker import chunk_extracted_pdf, detect_document_chapters
from src.llm import answer_from_evidence
from src.pdf_processor import extract_pdf
from src.retriever import build_hybrid_index, evidence_for_llm, hybrid_search

GEMINI_QUOTA_MESSAGE = "⚠️ Gemini API quota has been reached. Please try again later or check your Gemini API quota."


def _is_gemini_quota_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None) or getattr(error, "code", None)
    message = str(error).lower()
    return (
        status_code in (429, "429")
        or "resource_exhausted" in message
        or ("quota" in message and ("429" in message or "exceed" in message or "limit" in message))
    )


def _normalize_briefing_markdown(raw_briefing: str) -> str:
    text = raw_briefing.strip()
    text = re.sub(
        r"(?m)^([ \t]*)\((Overview|Key Concepts|Important Points|Key Takeaway)\):\s*\\\*\\\*\s*\\\*\s*(?=\*\*)",
        r"\1## \2\n\n- ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^([ \t]*\([^()\n]{1,80}\):)\s*\\\*\\\*\s*\\\*\s*(?=\*\*)",
        r"### \1\n\n- ",
        text,
    )
    text = re.sub(r"\\\*\\\*", "**", text)
    text = re.sub(r"\\\*([^*\n]+?)\\\*", r"*\1*", text)
    text = re.sub(r"(?m)^([ \t]*)\\\*(?=[ \t]+)", r"\1- ", text)
    text = re.sub(r"(?m)^([ \t]*[-*+])[ \t]+", r"\1 ", text)
    return text


def _format_briefing_for_display(raw_briefing: str, language: str = "English") -> str:
    """Format Gemini output for readable Streamlit sections without altering the content."""
    if not raw_briefing:
        return raw_briefing
    text = _normalize_briefing_markdown(raw_briefing)
    if language != "English":
        return text

    headings = ["Overview", "Key Concepts", "Important Points", "Key Takeaway"]
    rendered_sections: list[str] = []

    for heading in headings:
        pattern = rf"##\s*{re.escape(heading)}\s*(.*?)(?=##\s*(?:Overview|Key Concepts|Important Points|Key Takeaway)|$)"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            continue

        section_body = match.group(1).strip()
        if not section_body:
            section_body = ""

        if heading in {"Key Concepts", "Important Points"}:
            section_body = re.sub(r"\s*\*\s*", "\n- ", section_body)
            section_body = re.sub(r"\s*-\s*", "\n- ", section_body)
            section_body = section_body.strip()
            if section_body and not section_body.startswith("- "):
                section_body = "- " + section_body

        rendered_sections.append(f"## {heading}\n{section_body}".strip())

    return "\n\n".join(rendered_sections)


def _navigate_whole_book_batch(document_key: str, direction: int) -> None:
    batch_indexes = st.session_state["whole_book_batch_indexes"]
    batch_indexes[document_key] = max(0, batch_indexes.get(document_key, 0) + direction)
    if direction > 0:
        st.session_state["whole_book_pending_generation"].add(document_key)


st.set_page_config(
    page_title="E-Book Intelligence & Briefing Agent",
    page_icon="📚",
    layout="wide",
)

st.title("E-Book Intelligence & Multilingual Briefing Agent")
st.caption("Upload a PDF to extract text page by page.")

uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded is not None:
    pdf_bytes = uploaded.getvalue()
    document_key = hashlib.sha256(pdf_bytes).hexdigest()
    extracted = extract_pdf(pdf_bytes)
    pages = extracted["pages"]
    page_count = len(pages)

    detected_chapters = detect_document_chapters(pages)
    for chapter in detected_chapters:
        chapter["label"] = f"{chapter['chapter_number']} — {chapter['chapter_title']}"
        chapter["pages"] = list(range(chapter["start_page"], chapter["end_page"] + 1))

    st.session_state["detected_chapters"] = detected_chapters

    st.subheader("Upload summary")
    st.write(f"**File name:** {uploaded.name}")
    st.write(f"**Number of pages:** {page_count}")

    st.markdown("---")
    st.subheader("📚 Book Briefing")

    briefing_choice = st.radio(
        "Do you want a briefing?",
        ["No", "Yes"],
        index=0,
        key="book_briefing_choice",
    )

    if briefing_choice == "Yes":
        selected_language = st.selectbox(
            "Briefing Language",
            SUPPORTED_BRIEFING_LANGUAGES,
            index=0,
            key="briefing_language",
        )

        st.write("What would you like to brief?")
        briefing_scope = st.radio(
            "",
            ["Whole Book", "Specific Chapter"],
            index=0,
            key="book_briefing_scope",
            label_visibility="collapsed",
        )

        if briefing_scope == "Specific Chapter":
            detected_chapters = st.session_state.get("detected_chapters", [])

            if not detected_chapters:
                st.write("No chapters detected in this PDF.")
            else:
                chapter_options = [chapter["label"] for chapter in detected_chapters]
                if "briefing_selected_label" not in st.session_state:
                    st.session_state["briefing_selected_label"] = chapter_options[0]
                if st.session_state["briefing_selected_label"] not in chapter_options:
                    st.session_state["briefing_selected_label"] = chapter_options[0]

                selected_chapter_label = st.selectbox(
                    "Select Chapter",
                    chapter_options,
                    index=chapter_options.index(st.session_state["briefing_selected_label"]),
                    key="briefing_selected_label",
                )

                if st.button("Generate Briefing", key="briefing_generate_specific"):
                    selected_chapter = next(
                        chapter for chapter in detected_chapters if chapter["label"] == selected_chapter_label
                    )
                    chapter_content = "\n\n".join(
                        page["text"]
                        for page in extracted["pages"]
                        if selected_chapter["start_page"] <= page["page_number"] <= selected_chapter["end_page"]
                    )

                    if not chapter_content.strip():
                        st.warning(
                            "No usable extracted content was found for this selected chapter, so a briefing cannot be generated."
                        )
                    else:
                        if "specific_chapter_briefings" not in st.session_state:
                            st.session_state["specific_chapter_briefings"] = {}
                        chapter_cache_key = (
                            document_key,
                            selected_chapter["chapter_number"],
                            selected_chapter["chapter_title"],
                            selected_chapter["start_page"],
                            selected_chapter["end_page"],
                            selected_language,
                        )
                        if chapter_cache_key not in st.session_state["specific_chapter_briefings"]:
                            try:
                                st.session_state["specific_chapter_briefings"][chapter_cache_key] = (
                                    generate_chapter_briefing(
                                        selected_chapter["chapter_title"],
                                        selected_chapter["chapter_number"],
                                        chapter_content,
                                        language=selected_language,
                                    )
                                )
                            except Exception as exc:
                                if _is_gemini_quota_error(exc):
                                    st.error(GEMINI_QUOTA_MESSAGE)
                                else:
                                    st.error("I couldn't generate the chapter briefing right now. Please try again.")

                selected_chapter = next(
                    chapter for chapter in detected_chapters if chapter["label"] == selected_chapter_label
                )
                chapter_cache_key = (
                    document_key,
                    selected_chapter["chapter_number"],
                    selected_chapter["chapter_title"],
                    selected_chapter["start_page"],
                    selected_chapter["end_page"],
                    selected_language,
                )
                chapter_briefing = st.session_state.get("specific_chapter_briefings", {}).get(chapter_cache_key)
                if chapter_briefing:
                    st.subheader("Briefing")
                    st.markdown(_format_briefing_for_display(chapter_briefing, selected_language))
        else:
            if not detected_chapters:
                st.write("No chapters detected in this PDF.")
            else:
                if "whole_book_briefings" not in st.session_state:
                    st.session_state["whole_book_briefings"] = {}
                if "whole_book_batch_indexes" not in st.session_state:
                    st.session_state["whole_book_batch_indexes"] = {}
                if "whole_book_started" not in st.session_state:
                    st.session_state["whole_book_started"] = set()
                if "whole_book_pending_generation" not in st.session_state:
                    st.session_state["whole_book_pending_generation"] = set()

                document_briefings = st.session_state["whole_book_briefings"].setdefault(document_key, {})
                batch_indexes = st.session_state["whole_book_batch_indexes"]
                started_documents = st.session_state["whole_book_started"]
                pending_generations = st.session_state["whole_book_pending_generation"]

                generate_clicked = st.button("Generate Briefing", key="briefing_generate_whole")
                current_batch = batch_indexes.get(document_key, 0)
                started = document_key in started_documents

                if len(detected_chapters) > 5 and (started or generate_clicked):
                    previous_column, next_column = st.columns(2)
                    with previous_column:
                        st.button(
                            "Previous 5",
                            key="briefing_previous_five",
                            disabled=current_batch == 0,
                            on_click=_navigate_whole_book_batch,
                            args=(document_key, -1),
                        )
                    if (current_batch + 1) * 5 < len(detected_chapters):
                        with next_column:
                            st.button(
                                "Next 5",
                                key="briefing_next_five",
                                on_click=_navigate_whole_book_batch,
                                args=(document_key, 1),
                            )

                should_generate_batch = document_key in pending_generations
                if should_generate_batch:
                    pending_generations.discard(document_key)

                if generate_clicked:
                    current_batch = 0
                    batch_indexes[document_key] = current_batch
                    started_documents.add(document_key)
                    should_generate_batch = True

                current_batch = batch_indexes.get(document_key, current_batch)
                batch_start = current_batch * 5
                batch_chapters = detected_chapters[batch_start : batch_start + 5]
                quota_failed_chapters: set[tuple] = set()
                quota_failure_index: int | None = None

                if should_generate_batch:
                    for chapter_index, chapter in enumerate(batch_chapters):
                        chapter_key = (
                            chapter["chapter_number"],
                            chapter["chapter_title"],
                            chapter["start_page"],
                            chapter["end_page"],
                            selected_language,
                        )
                        cached_result = document_briefings.get(chapter_key, {})
                        if "briefing" in cached_result:
                            continue

                        chapter_content = "\n\n".join(
                            page["text"]
                            for page in pages
                            if chapter["start_page"] <= page["page_number"] <= chapter["end_page"]
                        )
                        if not chapter_content.strip():
                            document_briefings[chapter_key] = {
                                "error": "No usable extracted content was found for this chapter."
                            }
                            continue

                        try:
                            document_briefings[chapter_key] = {
                                "briefing": generate_chapter_briefing(
                                    chapter["chapter_title"],
                                    chapter["chapter_number"],
                                    chapter_content,
                                    language=selected_language,
                                )
                            }
                        except Exception as exc:
                            if _is_gemini_quota_error(exc):
                                quota_failed_chapters.add(chapter_key)
                                quota_failure_index = chapter_index
                                break
                            document_briefings[chapter_key] = {
                                "error": "Briefing generation failed for this chapter. Please try again."
                            }

                if document_key in started_documents:
                    for chapter_index, chapter in enumerate(batch_chapters):
                        st.markdown(f"### Chapter {chapter['chapter_number']} — {chapter['chapter_title']}")
                        chapter_key = (
                            chapter["chapter_number"],
                            chapter["chapter_title"],
                            chapter["start_page"],
                            chapter["end_page"],
                            selected_language,
                        )
                        chapter_result = document_briefings.get(chapter_key, {})
                        if "briefing" in chapter_result:
                            st.markdown(_format_briefing_for_display(chapter_result["briefing"], selected_language))
                        elif chapter_key in quota_failed_chapters:
                            st.error(GEMINI_QUOTA_MESSAGE)
                        elif "error" in chapter_result:
                            st.error(chapter_result["error"])
                        elif quota_failure_index is not None and chapter_index > quota_failure_index:
                            st.info("Not generated because the Gemini API quota was reached.")

    st.markdown("---")
    st.subheader("Ask the Book")

    if "book_question" not in st.session_state:
        st.session_state["book_question"] = ""

    with st.form("ask_book_form"):
        question = st.text_input(
            "Question",
            key="book_question",
            placeholder="Ask a question about the uploaded book...",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        try:
            chunks = chunk_extracted_pdf(extracted)
            hybrid_index = build_hybrid_index(chunks)
            search_response = hybrid_search(hybrid_index, question, retrieve_k=10, top_k=5)
            evidence = evidence_for_llm(search_response)

            if not evidence:
                answer = "I could not find sufficient evidence for this question in the uploaded book."
                sources = []
            else:
                grounded = answer_from_evidence(question, evidence)
                answer = grounded["answer"]
                sources = grounded["sources"]

            st.subheader("Answer")
            st.write(answer)

            if sources:
                unique_pages = sorted({source["page_number"] for source in sources})
                st.subheader("Sources / Pages")
                st.write(", ".join(str(page) for page in unique_pages))
        except Exception as exc:
            st.error(f"Ask the Book failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
