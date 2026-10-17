import unittest

from src.chunker import detect_document_chapters


class DetectDocumentChaptersTests(unittest.TestCase):
    def test_detects_lecture_headings_and_ranges(self):
        pages = [
            {"page_number": 1, "text": "LEC-1: Introduction to DBMS\nThis chapter introduces data."},
            {"page_number": 2, "text": "1. What is Data?\n2. Types of Data?\n3. What is Information?"},
            {"page_number": 3, "text": "LEC-2: DBMS Architecture\nSystem components follow."},
            {"page_number": 4, "text": "1. Database Design\n2. Storage Management"},
            {"page_number": 5, "text": "LEC-3: Entity-Relationship Model\nEntities and relationships."},
        ]

        chapters = detect_document_chapters(pages)

        self.assertEqual(
            chapters,
            [
                {"chapter_number": 1, "chapter_title": "Introduction to DBMS", "start_page": 1, "end_page": 2},
                {"chapter_number": 2, "chapter_title": "DBMS Architecture", "start_page": 3, "end_page": 4},
                {"chapter_number": 3, "chapter_title": "Entity-Relationship Model", "start_page": 5, "end_page": 5},
            ],
        )

    def test_numbered_major_headings_are_supported(self):
        pages = [
            {"page_number": 1, "text": "1. Introduction\nOverview of the domain."},
            {"page_number": 2, "text": "1. What is Data?\n2. Types of Data?"},
            {"page_number": 3, "text": "2. Process Management\nManaging tasks and execution."},
        ]

        chapters = detect_document_chapters(pages)

        self.assertEqual(
            chapters,
            [
                {"chapter_number": 1, "chapter_title": "Introduction", "start_page": 1, "end_page": 2},
                {"chapter_number": 2, "chapter_title": "Process Management", "start_page": 3, "end_page": 3},
            ],
        )

    def test_invalid_numbered_subtopics_do_not_create_chapters(self):
        pages = [
            {"page_number": 1, "text": "1. What is Data?\n2. Types of Data?\n3. What is Information?"},
            {"page_number": 2, "text": "This is just a normal section."},
        ]

        self.assertEqual(detect_document_chapters(pages), [])


if __name__ == "__main__":
    unittest.main()
