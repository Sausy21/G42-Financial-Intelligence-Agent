"""
Phase 1: Document Ingestion Pipeline

PDF text + table extraction using pdfplumber.
Handles multi-column layouts, footnotes, and scanned pages (OCR fallback with pytesseract).
Outputs clean markdown per document section.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import pdfplumber
import pandas as pd

from models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text and tables from financial PDFs using pdfplumber."""

    def __init__(self, ocr_fallback: bool = True):
        self.ocr_fallback = ocr_fallback

    def extract(self, pdf_path: str | Path, display_name: str | None = None) -> list[DocumentChunk]:
        """
        Extract all content from a PDF, returning structured chunks.

        Args:
            pdf_path:     Path to the PDF file (may be a temp path).
            display_name: User-facing filename to embed in all citations.
                          Defaults to pdf_path.name if not provided.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_name = display_name or pdf_path.name
        logger.info(f"Extracting from: {doc_name}")
        chunks: list[DocumentChunk] = []

        with pdfplumber.open(pdf_path) as pdf:
            current_section = "Introduction"
            current_text_parts: list[str] = []
            current_tables: list[dict] = []
            current_page = 1

            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                if not text.strip() and self.ocr_fallback:
                    text = self._ocr_page(page)

                text = self._clean_text(text)
                sections = self._detect_sections(text)

                if sections:
                    for heading, content in sections:
                        if current_text_parts:
                            chunks.append(self._make_chunk(
                                document_name=doc_name,
                                section_heading=current_section,
                                page_number=current_page,
                                text="\n".join(current_text_parts),
                                tables=current_tables,
                            ))
                        current_section = heading
                        current_text_parts = [content] if content else []
                        current_tables = []
                        current_page = page_num
                else:
                    current_text_parts.append(text)

                page_tables = self._extract_tables(page)
                current_tables.extend(page_tables)

            if current_text_parts:
                chunks.append(self._make_chunk(
                    document_name=doc_name,
                    section_heading=current_section,
                    page_number=current_page,
                    text="\n".join(current_text_parts),
                    tables=current_tables,
                ))

        logger.info(f"Extracted {len(chunks)} chunks from {doc_name}")
        return chunks

    def _extract_tables(self, page) -> list[dict]:
        """Extract tables from a page and convert to list of dicts."""
        tables = []
        try:
            raw_tables = page.extract_tables()
            for raw in raw_tables:
                if not raw or len(raw) < 2:
                    continue
                # First row as headers
                headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(raw[0])]
                rows = []
                for row in raw[1:]:
                    cleaned = [str(cell).strip() if cell else "" for cell in row]
                    rows.append(dict(zip(headers, cleaned)))
                tables.append({
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                })
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")
        return tables

    def _detect_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Detect section headings across financial document formats:
        - SEC 10-K: "Item 1. Business", "PART I"
        - European URD / Annual Report: "5.1 Consolidated Statement of Income",
          "Chapter 5 – Financial Statements", "1.4 – Consolidation principles"
        - All-caps headings used in both formats
        """
        sections = []

        # ── Pattern 1: SEC "Item N. Title" ──────────────────────────────────
        item_pattern = re.compile(
            r'^(Item\s+\d+[A-Z]?\.\s+.+?)$',
            re.MULTILINE | re.IGNORECASE,
        )
        matches = list(item_pattern.finditer(text))
        if matches:
            for i, match in enumerate(matches):
                heading = match.group(1).strip()
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((heading, text[start:end].strip()))
            return sections

        # ── Pattern 2: Numbered section "5.1 Title" / "1.4 – Title" ────────
        # Handles: "5.1 Consolidated statement of income"
        #          "1.4 – Consolidation principles"
        #          "Chapter 5 – Consolidated financial statements"
        numbered_pattern = re.compile(
            r'^((?:Chapter\s+)?\d+(?:\.\d+)*\s*[\-–—]?\s+[A-Z][^\n]{3,60})$',
            re.MULTILINE,
        )
        matches = list(numbered_pattern.finditer(text))
        if matches:
            for i, match in enumerate(matches):
                heading = match.group(1).strip()
                # Clean up the heading: normalise dashes and extra spaces
                heading = re.sub(r'\s*[\-–—]\s*', ' – ', heading)
                heading = re.sub(r'\s+', ' ', heading)
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((heading, text[start:end].strip()))
            return sections

        # ── Pattern 3: "PART I", "PART II" ──────────────────────────────────
        part_pattern = re.compile(r'^(PART\s+[IVX]+)', re.MULTILINE)
        matches = list(part_pattern.finditer(text))
        if matches:
            for i, match in enumerate(matches):
                heading = match.group(1).strip()
                start = match.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                sections.append((heading, text[start:end].strip()))
            return sections

        return []

    def _ocr_page(self, page) -> str:
        """OCR a scanned page using pytesseract."""
        try:
            import pytesseract
            image = page.to_image(resolution=300).original
            return pytesseract.image_to_string(image)
        except ImportError:
            logger.warning("pytesseract not installed — skipping OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: normalize whitespace, remove artifacts."""
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def _make_chunk(
        self,
        document_name: str,
        section_heading: str,
        page_number: int,
        text: str,
        tables: list[dict],
    ) -> DocumentChunk:
        """Create a DocumentChunk with computed metadata."""
        chunk_id = hashlib.sha256(
            f"{document_name}:{section_heading}:{page_number}".encode()
        ).hexdigest()[:12]

        # Append table data as markdown to the text
        full_text = text
        for table in tables:
            if table.get("rows"):
                md_table = self._table_to_markdown(table)
                full_text += f"\n\n[TABLE]\n{md_table}\n[/TABLE]"

        return DocumentChunk(
            chunk_id=chunk_id,
            document_name=document_name,
            section_heading=section_heading,
            page_number=page_number,
            text=full_text,
            tables=tables,
            char_count=len(full_text),
            token_estimate=len(full_text) // 4,  # rough estimate
        )

    def _table_to_markdown(self, table: dict) -> str:
        """Convert a table dict to markdown format."""
        headers = table.get("headers", [])
        rows = table.get("rows", [])
        if not headers or not rows:
            return ""

        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            vals = [row.get(h, "") for h in headers]
            lines.append("| " + " | ".join(vals) + " |")

        return "\n".join(lines)


class CSVLoader:
    """Load structured financial data from CSV files."""

    @staticmethod
    def load(csv_path: str | Path) -> pd.DataFrame:
        """Load a CSV file and clean column names."""
        df = pd.read_csv(csv_path)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        return df

    @staticmethod
    def to_chunks(df: pd.DataFrame, source_name: str) -> list[DocumentChunk]:
        """Convert a DataFrame into document chunks for RAG indexing."""
        chunks = []
        chunk_id = hashlib.sha256(source_name.encode()).hexdigest()[:12]

        # Create a single chunk with the full CSV as a markdown table
        md = df.to_markdown(index=False)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            document_name=source_name,
            section_heading="Tabular Data",
            text=md,
            tables=[{
                "headers": list(df.columns),
                "rows": df.to_dict("records"),
                "row_count": len(df),
            }],
            char_count=len(md),
            token_estimate=len(md) // 4,
        ))

        return chunks
