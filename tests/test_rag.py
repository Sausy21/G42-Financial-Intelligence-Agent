"""
Tests for the RAG layer: chunker, BM25 index, and hybrid retrieval.

These tests use in-memory components and don't require API keys.
Run: pytest tests/test_rag.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np

from rag.retriever import SectionChunker, BM25Index, VectorStore
from models.schemas import DocumentChunk


def make_chunk(chunk_id: str, text: str, section: str = "Test") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_name="test.pdf",
        section_heading=section,
        page_number=1,
        text=text,
        char_count=len(text),
        token_estimate=len(text) // 4,
    )


# ── SectionChunker Tests ─────────────────────────────────────────────

class TestSectionChunker:
    def setup_method(self):
        self.chunker = SectionChunker(max_chunk_chars=200, overlap_chars=50)

    def test_small_chunks_unchanged(self):
        chunks = [make_chunk("1", "Short text.", "Section A")]
        result = self.chunker.chunk(chunks)
        assert len(result) == 1
        assert result[0].text == "Short text."

    def test_large_chunk_split(self):
        long_text = "\n\n".join([f"Paragraph {i}. " + "x" * 50 for i in range(10)])
        chunks = [make_chunk("1", long_text, "Long Section")]
        result = self.chunker.chunk(chunks)
        assert len(result) > 1
        for chunk in result:
            assert chunk.char_count <= self.chunker.max_chunk_chars + 100  # some tolerance

    def test_section_heading_preserved(self):
        long_text = "\n\n".join(["A" * 100 for _ in range(5)])
        chunks = [make_chunk("1", long_text, "Revenue Discussion")]
        result = self.chunker.chunk(chunks)
        assert all("Revenue Discussion" in c.section_heading for c in result)

    def test_multiple_chunks_input(self):
        chunks = [
            make_chunk("1", "Short A.", "Section A"),
            make_chunk("2", "Short B.", "Section B"),
        ]
        result = self.chunker.chunk(chunks)
        assert len(result) == 2


# ── BM25 Index Tests ─────────────────────────────────────────────────

class TestBM25Index:
    def setup_method(self):
        self.index = BM25Index()
        self.chunks = [
            make_chunk("1", "Revenue increased by 15% driven by data center growth", "Revenue"),
            make_chunk("2", "Net income was $5.2 billion in fiscal year 2024", "Income"),
            make_chunk("3", "Total debt decreased to $8.4 billion from $10.1 billion", "Debt"),
            make_chunk("4", "Operating cash flow was $28.9 billion for the year", "Cash Flow"),
            make_chunk("5", "The company repurchased $25 billion of common stock", "Buybacks"),
        ]
        self.index.add(self.chunks)

    def test_search_returns_results(self):
        results = self.index.search("revenue growth", k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_relevance(self):
        results = self.index.search("revenue", k=1)
        assert len(results) == 1
        assert "revenue" in results[0][0].text.lower()

    def test_search_debt(self):
        results = self.index.search("debt decreased", k=1)
        assert len(results) == 1
        assert "debt" in results[0][0].text.lower()

    def test_search_no_match(self):
        results = self.index.search("quantum computing blockchain", k=3)
        # BM25 may still return results but with low scores
        if results:
            assert results[0][1] < 1.0

    def test_empty_index(self):
        empty = BM25Index()
        results = empty.search("anything", k=3)
        assert len(results) == 0

    def test_tokenize(self):
        tokens = BM25Index._tokenize("Hello, World! Test 123.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "123" in tokens


# ── Vector Store Tests ────────────────────────────────────────────────

class TestVectorStore:
    def setup_method(self):
        self.store = VectorStore()

    def test_add_and_search(self):
        chunks = [
            make_chunk("1", "Revenue was $10 billion", "Revenue"),
            make_chunk("2", "Expenses totaled $7 billion", "Expenses"),
        ]
        # Create simple embeddings (2D for testing)
        embeddings = np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
        ], dtype=np.float32)

        self.store.add(chunks, embeddings)

        # Query closer to first chunk
        query = np.array([0.9, 0.1, 0.5], dtype=np.float32)
        results = self.store.search(query, k=1)

        assert len(results) == 1
        assert results[0][0].chunk_id == "1"

    def test_empty_store(self):
        results = self.store.search(np.array([1.0, 0.0], dtype=np.float32), k=3)
        assert len(results) == 0

    def test_k_larger_than_store(self):
        chunks = [make_chunk("1", "Only one chunk")]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
        self.store.add(chunks, embeddings)

        results = self.store.search(np.array([1.0, 0.0], dtype=np.float32), k=10)
        assert len(results) == 1


# ── Integration: Hybrid Retrieval Logic ──────────────────────────────

class TestHybridFusion:
    """Test the reciprocal rank fusion logic without API calls."""

    def test_rrf_merges_correctly(self):
        from rag.retriever import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense_weight = 0.6
        retriever.sparse_weight = 0.4

        chunk_a = make_chunk("a", "Revenue growth")
        chunk_b = make_chunk("b", "Debt analysis")
        chunk_c = make_chunk("c", "Cash flow")

        dense = [(chunk_a, 0.95), (chunk_b, 0.80)]
        sparse = [(chunk_b, 5.2), (chunk_c, 3.1)]

        fused = retriever._reciprocal_rank_fusion(dense, sparse, k=3)

        # chunk_b appears in both lists — should rank highly
        ids = [c.chunk_id for c, _ in fused]
        assert "b" in ids
        assert len(fused) <= 3

    def test_rrf_handles_empty_lists(self):
        from rag.retriever import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.dense_weight = 0.6
        retriever.sparse_weight = 0.4

        fused = retriever._reciprocal_rank_fusion([], [], k=5)
        assert len(fused) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
