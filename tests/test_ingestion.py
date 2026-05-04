"""
Tests for document ingestion pipeline.

Run: pytest tests/test_ingestion.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import tempfile

from ingestion.pdf_extractor import PDFExtractor, CSVLoader
from models.schemas import DocumentChunk


class TestPDFExtractor:
    def setup_method(self):
        self.extractor = PDFExtractor(ocr_fallback=False)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.extractor.extract("/nonexistent/file.pdf")

    def test_clean_text(self):
        raw = "  Hello   World  \n\n\n\nTest  "
        cleaned = self.extractor._clean_text(raw)
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned

    def test_detect_sections_item_pattern(self):
        text = (
            "Item 1. Business\n"
            "We are a technology company.\n\n"
            "Item 7. Management's Discussion\n"
            "Revenue increased by 15%."
        )
        sections = self.extractor._detect_sections(text)
        assert len(sections) == 2
        assert "Item 1" in sections[0][0]
        assert "Item 7" in sections[1][0]

    def test_detect_sections_part_pattern(self):
        text = (
            "PART I\n"
            "Some content here.\n\n"
            "PART II\n"
            "More content."
        )
        sections = self.extractor._detect_sections(text)
        assert len(sections) == 2
        assert "PART I" in sections[0][0]

    def test_detect_sections_no_pattern(self):
        text = "Just some plain text without any section headings."
        sections = self.extractor._detect_sections(text)
        assert len(sections) == 0

    def test_table_to_markdown(self):
        table = {
            "headers": ["Year", "Revenue", "Net Income"],
            "rows": [
                {"Year": "2023", "Revenue": "$1,000", "Net Income": "$200"},
                {"Year": "2024", "Revenue": "$1,500", "Net Income": "$350"},
            ],
        }
        md = self.extractor._table_to_markdown(table)
        assert "| Year | Revenue | Net Income |" in md
        assert "$1,000" in md
        assert "---" in md

    def test_make_chunk(self):
        chunk = self.extractor._make_chunk(
            document_name="test.pdf",
            section_heading="Revenue",
            page_number=5,
            text="Revenue was $1.5 billion in FY2024.",
            tables=[],
        )
        assert isinstance(chunk, DocumentChunk)
        assert chunk.document_name == "test.pdf"
        assert chunk.section_heading == "Revenue"
        assert chunk.page_number == 5
        assert chunk.char_count > 0
        assert chunk.token_estimate > 0
        assert len(chunk.chunk_id) == 12

    def test_make_chunk_with_tables(self):
        table = {
            "headers": ["Metric", "Value"],
            "rows": [{"Metric": "Revenue", "Value": "$1B"}],
        }
        chunk = self.extractor._make_chunk(
            document_name="test.pdf",
            section_heading="Financials",
            page_number=10,
            text="Financial summary.",
            tables=[table],
        )
        assert "[TABLE]" in chunk.text
        assert "Revenue" in chunk.text


class TestCSVLoader:
    def test_load_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Year,Revenue,Net Income\n")
            f.write("2022,1000000,200000\n")
            f.write("2023,1200000,250000\n")
            f.write("2024,1500000,350000\n")
            path = f.name

        df = CSVLoader.load(path)
        assert len(df) == 3
        assert "revenue" in df.columns
        assert "net_income" in df.columns

    def test_to_chunks(self):
        import pandas as pd

        df = pd.DataFrame({
            "year": [2022, 2023, 2024],
            "revenue": [1000000, 1200000, 1500000],
        })
        chunks = CSVLoader.to_chunks(df, "financials.csv")
        assert len(chunks) == 1
        assert chunks[0].document_name == "financials.csv"
        assert chunks[0].section_heading == "Tabular Data"
        assert len(chunks[0].tables) == 1
        assert chunks[0].tables[0]["row_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
