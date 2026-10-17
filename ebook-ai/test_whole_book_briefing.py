import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import fitz
from streamlit.testing.v1 import AppTest
from src.briefing import SUPPORTED_BRIEFING_LANGUAGES, _build_briefing_prompt, generate_chapter_briefing


class WholeBookBriefingTests(unittest.TestCase):
    def _make_pdf(self, chapter_count: int) -> bytes:
        document = fitz.open()
        for chapter_number in range(1, chapter_count + 1):
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"LEC-{chapter_number}: DBMS Chapter {chapter_number}",
                fontsize=16,
            )
            page.insert_text(
                (72, 110),
                f"Content belonging only to chapter {chapter_number}.",
                fontsize=12,
            )
        pdf_bytes = document.tobytes()
        document.close()
        return pdf_bytes

    def test_prompt_requests_supported_language_and_preserves_source_content(self):
        chapter_content = "Original English source: primary keys identify records."
        for language in SUPPORTED_BRIEFING_LANGUAGES:
            with self.subTest(language=language):
                prompt = _build_briefing_prompt("Database Design", 1, chapter_content, language)
                self.assertIn(f"entire briefing in {language}", prompt)
                self.assertIn(chapter_content, prompt)
                self.assertIn("Overview, Key Concepts, Important Points, Key Takeaway", prompt)

        client = MagicMock()
        client.models.generate_content.return_value = SimpleNamespace(text="हिंदी सारांश")
        with patch("src.briefing._get_client", return_value=client):
            result = generate_chapter_briefing("Database Design", 1, chapter_content, language="Hindi")

        sent_prompt = client.models.generate_content.call_args.kwargs["contents"]
        self.assertEqual(result, "हिंदी सारांश")
        self.assertIn(chapter_content, sent_prompt)
        self.assertIn("entire briefing in Hindi", sent_prompt)

    def test_escaped_hindi_markdown_renders_cleanly_in_both_briefing_scopes(self):
        source_markdown = (
            "(Key Concepts):\\*\\* \\* **डेटा (Data):** "
            "यह कच्चे, असंगठित तथ्यों का संग्रह है।\n\\* **सूचना:** व्यवस्थित डेटा।"
        )

        with patch("src.briefing.generate_chapter_briefing", return_value=source_markdown) as generate_briefing:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(1), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)
            app.selectbox(key="briefing_language").set_value("Hindi").run(timeout=30)
            app.radio[1].set_value("Specific Chapter").run(timeout=30)
            app.button(key="briefing_generate_specific").click().run(timeout=30)

            specific_display = "\n".join(item.value for item in app.markdown)
            self.assertIn("## Key Concepts", specific_display)
            self.assertIn("- **डेटा (Data):** यह कच्चे, असंगठित तथ्यों का संग्रह है।", specific_display)
            self.assertIn("- **सूचना:** व्यवस्थित डेटा।", specific_display)
            self.assertNotIn(r"\*", specific_display)

            app.radio[1].set_value("Whole Book").run(timeout=30)
            app.button(key="briefing_generate_whole").click().run(timeout=30)
            whole_book_display = "\n".join(item.value for item in app.markdown)
            self.assertIn("## Key Concepts", whole_book_display)
            self.assertIn("- **डेटा (Data):** यह कच्चे, असंगठित तथ्यों का संग्रह है।", whole_book_display)
            self.assertNotIn(r"\*", whole_book_display)
            self.assertEqual(generate_briefing.call_count, 2)

        cached_specific = next(iter(app.session_state["specific_chapter_briefings"].values()))
        self.assertEqual(cached_specific, source_markdown)

    def test_whole_book_generates_batches_and_uses_cache_on_previous(self):
        calls = []

        def generate_briefing(title, number, content, language="English"):
            calls.append((title, number, content, language))
            return f"## Overview\nBriefing for {title}."

        with patch("src.briefing.generate_chapter_briefing", side_effect=generate_briefing):
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(10), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)

            app.button(key="briefing_generate_whole").click().run(timeout=30)
            self.assertEqual(len(calls), 5)
            self.assertEqual([call[1] for call in calls], [1, 2, 3, 4, 5])
            self.assertEqual({call[3] for call in calls}, {"English"})
            first_batch_text = "\n".join(element.value for element in app.markdown)
            self.assertIn("Chapter 1 — DBMS Chapter 1", first_batch_text)
            self.assertIn("Chapter 5 — DBMS Chapter 5", first_batch_text)
            self.assertNotIn("Chapter 6 — DBMS Chapter 6", first_batch_text)

            app.radio[1].set_value("Specific Chapter").run(timeout=30)
            app.radio[1].set_value("Whole Book").run(timeout=30)
            self.assertEqual(len(calls), 5)
            self.assertIn(
                "Chapter 1 — DBMS Chapter 1",
                "\n".join(element.value for element in app.markdown),
            )

            app.button(key="briefing_next_five").click().run(timeout=30)
            self.assertEqual(len(calls), 10)
            second_batch_text = "\n".join(element.value for element in app.markdown)
            self.assertIn("Chapter 6 — DBMS Chapter 6", second_batch_text)
            self.assertIn("Chapter 10 — DBMS Chapter 10", second_batch_text)
            self.assertNotIn("briefing_next_five", [button.key for button in app.button])

            app.button(key="briefing_previous_five").click().run(timeout=30)
            self.assertEqual(len(calls), 10)
            previous_batch_text = "\n".join(element.value for element in app.markdown)
            self.assertIn("Chapter 1 — DBMS Chapter 1", previous_batch_text)
            self.assertIn("Chapter 5 — DBMS Chapter 5", previous_batch_text)
            self.assertNotIn("Chapter 6 — DBMS Chapter 6", previous_batch_text)

    def test_short_book_and_generation_errors_are_handled(self):
        calls = []

        def generate_briefing(title, number, content, language="English"):
            calls.append(number)
            if number == 3:
                raise RuntimeError("Gemini request failed")
            return f"## Overview\nBriefing for {title}."

        with patch("src.briefing.generate_chapter_briefing", side_effect=generate_briefing):
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(5), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)
            app.button(key="briefing_generate_whole").click().run(timeout=30)

        self.assertEqual(calls, [1, 2, 3, 4, 5])
        self.assertNotIn("briefing_next_five", [button.key for button in app.button])
        self.assertTrue(any("Briefing generation failed for this chapter" in error.value for error in app.error))
        rendered_text = "\n".join(element.value for element in app.markdown)
        self.assertIn("Chapter 4 — DBMS Chapter 4", rendered_text)
        self.assertIn("Chapter 5 — DBMS Chapter 5", rendered_text)

    def test_language_selection_persists_across_briefing_scopes_without_generation(self):
        with patch("src.briefing.generate_chapter_briefing") as generate_briefing:
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(2), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)

            self.assertEqual(app.selectbox(key="briefing_language").value, "English")
            app.selectbox(key="briefing_language").set_value("Odia").run(timeout=30)
            self.assertEqual(app.session_state["briefing_language"], "Odia")

            app.radio[1].set_value("Specific Chapter").run(timeout=30)
            self.assertEqual(app.selectbox(key="briefing_language").value, "Odia")
            app.radio[1].set_value("Whole Book").run(timeout=30)
            self.assertEqual(app.selectbox(key="briefing_language").value, "Odia")
            generate_briefing.assert_not_called()

    def test_specific_chapter_results_are_cached_separately_by_language(self):
        calls = []

        def generate_briefing(title, number, content, language="English"):
            calls.append((number, content, language))
            if language == "English":
                return f"## Overview\nEnglish briefing for {title}."
            return f"## अवलोकन\n{language} briefing for {title}."

        with patch("src.briefing.generate_chapter_briefing", side_effect=generate_briefing):
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(2), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)
            app.radio[1].set_value("Specific Chapter").run(timeout=30)

            app.button(key="briefing_generate_specific").click().run(timeout=30)
            self.assertEqual(calls[-1][2], "English")
            self.assertIn("English briefing", "\n".join(element.value for element in app.markdown))

            app.selectbox(key="briefing_language").set_value("Hindi").run(timeout=30)
            self.assertNotIn("English briefing", "\n".join(element.value for element in app.markdown))
            app.button(key="briefing_generate_specific").click().run(timeout=30)
            self.assertEqual(calls[-1][2], "Hindi")
            self.assertIn("LEC-1: DBMS Chapter 1", calls[-1][1])
            self.assertIn("Content belonging only to chapter 1.", calls[-1][1])
            self.assertIn("Hindi briefing", "\n".join(element.value for element in app.markdown))

            app.selectbox(key="briefing_language").set_value("Odia").run(timeout=30)
            app.button(key="briefing_generate_specific").click().run(timeout=30)
            self.assertEqual(calls[-1][2], "Odia")
            self.assertEqual(len(calls), 3)

            app.selectbox(key="briefing_language").set_value("English").run(timeout=30)
            self.assertIn("English briefing", "\n".join(element.value for element in app.markdown))
            self.assertEqual(len(calls), 3)

    def test_whole_book_briefings_are_cached_by_language(self):
        calls = []

        def generate_briefing(title, number, content, language="English"):
            calls.append((number, language))
            if language == "English":
                return f"## Overview\nEnglish briefing for {title}."
            return f"## अवलोकन\n{language} briefing for {title}."

        with patch("src.briefing.generate_chapter_briefing", side_effect=generate_briefing):
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(2), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)
            app.button(key="briefing_generate_whole").click().run(timeout=30)
            self.assertEqual(calls, [(1, "English"), (2, "English")])

            app.selectbox(key="briefing_language").set_value("Hindi").run(timeout=30)
            self.assertNotIn("English briefing", "\n".join(element.value for element in app.markdown))
            app.button(key="briefing_generate_whole").click().run(timeout=30)
            self.assertEqual(calls[-2:], [(1, "Hindi"), (2, "Hindi")])

            app.selectbox(key="briefing_language").set_value("English").run(timeout=30)
            self.assertIn("English briefing", "\n".join(element.value for element in app.markdown))
            self.assertEqual(len(calls), 4)

        rendered_text = "\n".join(element.value for element in app.markdown)
        self.assertIn("English briefing", rendered_text)

    def test_specific_quota_error_is_not_cached_or_used_to_replace_success(self):
        calls = []

        def generate_briefing(title, number, content, language="English"):
            calls.append(language)
            if language == "Hindi":
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
            return "## Overview\nA successful English briefing."

        with patch("src.briefing.generate_chapter_briefing", side_effect=generate_briefing):
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(1), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)
            app.radio[1].set_value("Specific Chapter").run(timeout=30)
            app.button(key="briefing_generate_specific").click().run(timeout=30)

            app.selectbox(key="briefing_language").set_value("Hindi").run(timeout=30)
            app.button(key="briefing_generate_specific").click().run(timeout=30)
            error_text = "\n".join(error.value for error in app.error)
            self.assertIn("Gemini API quota has been reached", error_text)
            cached_languages = {
                cache_key[-1] for cache_key in app.session_state["specific_chapter_briefings"]
            }
            self.assertEqual(cached_languages, {"English"})

            app.selectbox(key="briefing_language").set_value("English").run(timeout=30)
            self.assertIn("A successful English briefing", "\n".join(item.value for item in app.markdown))
            self.assertEqual(calls, ["English", "Hindi"])

    def test_whole_book_quota_error_preserves_successes_and_is_not_cached(self):
        def generate_briefing(title, number, content, language="English"):
            if number == 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
            return f"## Overview\nSuccessful briefing for chapter {number}."

        with patch("src.briefing.generate_chapter_briefing", side_effect=generate_briefing):
            app = AppTest.from_file("app.py").run(timeout=30)
            app.file_uploader[0].upload("dbms.pdf", self._make_pdf(6), "application/pdf").run(timeout=30)
            app.radio[0].set_value("Yes").run(timeout=30)
            app.button(key="briefing_generate_whole").click().run(timeout=30)

        errors = "\n".join(error.value for error in app.error)
        self.assertIn("Gemini API quota has been reached", errors)
        self.assertIn("Successful briefing for chapter 1", "\n".join(item.value for item in app.markdown))
        self.assertIn("Successful briefing for chapter 2", "\n".join(item.value for item in app.markdown))
        cached_briefings = next(iter(app.session_state["whole_book_briefings"].values()))
        self.assertEqual(len(cached_briefings), 2)
        self.assertTrue(all("briefing" in value for value in cached_briefings.values()))


if __name__ == "__main__":
    unittest.main()
