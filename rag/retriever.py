"""
Phase 2: RAG Layer

Section-based chunking (not fixed tokens), sentence-transformers embeddings,
hybrid BM25 + dense retrieval with Reciprocal Rank Fusion.

Vector backend auto-selected at runtime:
  - PGVECTOR_URL set  →  PgVectorStore  (persistent, survives restarts)
  - PGVECTOR_URL unset →  VectorStore   (in-memory FAISS, default)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from models.schemas import DocumentChunk, Citation

logger = logging.getLogger(__name__)


# ── Chunker ────────────────────────────────────────────────────────────

class SectionChunker:
    """
    Chunk documents by section structure, not fixed token windows.

    Financial documents have natural section boundaries (Item 1, Item 7, etc.)
    that carry semantic meaning. Chunking by section preserves context better
    than arbitrary 512-token windows.
    """

    def __init__(self, max_chunk_chars: int = 4000, overlap_chars: int = 200):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    def chunk(self, doc_chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        result = []
        for chunk in doc_chunks:
            if chunk.char_count <= self.max_chunk_chars:
                result.append(chunk)
            else:
                result.extend(self._split_section(chunk))
        logger.info(f"Chunking: {len(doc_chunks)} sections → {len(result)} chunks")
        return result

    def _split_section(self, chunk: DocumentChunk) -> list[DocumentChunk]:
        paragraphs = chunk.text.split("\n\n")
        sub_chunks = []
        current_parts: list[str] = []
        current_len = 0
        part_idx = 0

        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len > self.max_chunk_chars and current_parts:
                sub_chunks.append(DocumentChunk(
                    chunk_id=f"{chunk.chunk_id}_{part_idx}",
                    document_name=chunk.document_name,
                    section_heading=f"{chunk.section_heading} (part {part_idx + 1})",
                    page_number=chunk.page_number,
                    text="\n\n".join(current_parts),
                    tables=chunk.tables if part_idx == 0 else [],
                    char_count=current_len,
                    token_estimate=current_len // 4,
                ))
                current_parts = [current_parts[-1]] if current_parts else []
                current_len = len(current_parts[0]) if current_parts else 0
                part_idx += 1
            current_parts.append(para)
            current_len += para_len

        if current_parts:
            text = "\n\n".join(current_parts)
            sub_chunks.append(DocumentChunk(
                chunk_id=f"{chunk.chunk_id}_{part_idx}",
                document_name=chunk.document_name,
                section_heading=f"{chunk.section_heading} (part {part_idx + 1})",
                page_number=chunk.page_number,
                text=text,
                tables=[],
                char_count=len(text),
                token_estimate=len(text) // 4,
            ))
        return sub_chunks


# ── Embedder ───────────────────────────────────────────────────────────

class Embedder:
    """sentence-transformers local embeddings — free, no API key."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        return self.model.encode(
            texts, show_progress_bar=len(texts) > 50,
            batch_size=64, convert_to_numpy=True,
        ).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


# ── FAISS Vector Store (default, in-memory) ────────────────────────────

class VectorStore:
    """In-memory FAISS vector store. Used when PGVECTOR_URL is not set."""

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._index = None

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray):
        import faiss
        self.chunks.extend(chunks)
        self.embeddings = (
            embeddings if self.embeddings is None
            else np.vstack([self.embeddings, embeddings])
        )
        dim = self.embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(self.embeddings)
        self._index.add(self.embeddings)
        logger.info(f"FAISS: {len(self.chunks)} chunks indexed")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[DocumentChunk, float]]:
        import faiss
        if self._index is None or not self.chunks:
            return []
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = self._index.search(query, min(k, len(self.chunks)))
        return [
            (self.chunks[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if 0 <= idx < len(self.chunks)
        ]


# ── pgvector Store (persistent, activated by PGVECTOR_URL) ────────────

class PgVectorStore:
    """
    PostgreSQL + pgvector persistent vector store.

    Activated automatically when PGVECTOR_URL is set in environment
    or Streamlit secrets. Survives app restarts and is shared across
    all user sessions — documents only need to be indexed once.

    Table schema (auto-created on first use):
        financial_chunks (
            chunk_id        TEXT PRIMARY KEY,
            document_name   TEXT,
            section_heading TEXT,
            page_number     INT,
            text            TEXT,
            tables_json     TEXT,
            embedding       vector(384)   -- all-MiniLM-L6-v2 dimension
        )
    """

    TABLE = "financial_chunks"
    DIM   = 384  # all-MiniLM-L6-v2

    def __init__(self, url: str):
        self.url = url
        self._conn = None
        self._ensure_table()

    # ── Connection ──────────────────────────────────────────────────

    @property
    def conn(self):
        """Lazy connection with automatic reconnect."""
        import psycopg2
        try:
            if self._conn is None or self._conn.closed:
                self._conn = psycopg2.connect(self.url)
                self._conn.autocommit = False
        except Exception as e:
            logger.error(f"pgvector connection failed: {e}")
            raise
        return self._conn

    def _ensure_table(self):
        """Create the chunks table and vector index if they don't exist."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE} (
                        chunk_id        TEXT PRIMARY KEY,
                        document_name   TEXT NOT NULL,
                        section_heading TEXT,
                        page_number     INT,
                        text            TEXT,
                        tables_json     TEXT DEFAULT '[]',
                        embedding       vector({self.DIM})
                    );
                """)
                # IVFFlat index for fast approximate nearest-neighbour search
                cur.execute(f"""
                    CREATE INDEX IF NOT EXISTS {self.TABLE}_embedding_idx
                    ON {self.TABLE}
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
            self.conn.commit()
            logger.info("pgvector table ready")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"pgvector table setup failed: {e}")
            raise

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _to_pg_vector(arr: np.ndarray) -> str:
        """Convert numpy array to pgvector literal '[x,y,z,...]'."""
        return "[" + ",".join(f"{v:.6f}" for v in arr.tolist()) + "]"

    @staticmethod
    def _chunk_from_row(row: tuple) -> DocumentChunk:
        chunk_id, doc_name, section, page, text, tables_json = row[:6]
        try:
            tables = json.loads(tables_json or "[]")
        except json.JSONDecodeError:
            tables = []
        return DocumentChunk(
            chunk_id=chunk_id,
            document_name=doc_name,
            section_heading=section or "",
            page_number=page,
            text=text or "",
            tables=tables,
            char_count=len(text or ""),
            token_estimate=len(text or "") // 4,
        )

    # ── Public API (mirrors VectorStore) ──────────────────────────

    def add(self, chunks: list[DocumentChunk], embeddings: np.ndarray):
        """Upsert chunks and their embeddings into PostgreSQL."""
        try:
            with self.conn.cursor() as cur:
                for chunk, emb in zip(chunks, embeddings):
                    cur.execute(f"""
                        INSERT INTO {self.TABLE}
                            (chunk_id, document_name, section_heading,
                             page_number, text, tables_json, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            document_name   = EXCLUDED.document_name,
                            section_heading = EXCLUDED.section_heading,
                            page_number     = EXCLUDED.page_number,
                            text            = EXCLUDED.text,
                            tables_json     = EXCLUDED.tables_json,
                            embedding       = EXCLUDED.embedding;
                    """, (
                        chunk.chunk_id,
                        chunk.document_name,
                        chunk.section_heading,
                        chunk.page_number,
                        chunk.text,
                        json.dumps(chunk.tables),
                        self._to_pg_vector(emb),
                    ))
            self.conn.commit()
            logger.info(f"pgvector: upserted {len(chunks)} chunks")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"pgvector add failed: {e}")
            raise

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """Cosine similarity search via pgvector <=> operator."""
        try:
            vec = self._to_pg_vector(query_embedding)
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT chunk_id, document_name, section_heading,
                           page_number, text, tables_json,
                           1 - (embedding <=> %s::vector) AS score
                    FROM {self.TABLE}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """, (vec, vec, k))
                rows = cur.fetchall()
            return [
                (self._chunk_from_row(row), float(row[6]))
                for row in rows
            ]
        except Exception as e:
            logger.error(f"pgvector search failed: {e}")
            return []

    def remove_document(self, document_name: str):
        """Delete all chunks for a given document."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.TABLE} WHERE document_name = %s;",
                    (document_name,),
                )
                deleted = cur.rowcount
            self.conn.commit()
            logger.info(f"pgvector: deleted {deleted} chunks for '{document_name}'")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"pgvector remove failed: {e}")
            raise

    def clear(self):
        """Delete all chunks from the table."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.TABLE};")
            self.conn.commit()
            logger.info("pgvector: table cleared")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"pgvector clear failed: {e}")
            raise

    def list_documents(self) -> list[str]:
        """Return a list of distinct document names in the store."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    f"SELECT DISTINCT document_name FROM {self.TABLE} ORDER BY 1;"
                )
                return [row[0] for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"pgvector list_documents failed: {e}")
            return []

    @property
    def chunks(self) -> list[DocumentChunk]:
        """Load all chunks from the DB (used by remove_document rebuild path)."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(f"""
                    SELECT chunk_id, document_name, section_heading,
                           page_number, text, tables_json
                    FROM {self.TABLE};
                """)
                return [self._chunk_from_row(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"pgvector chunks load failed: {e}")
            return []


# ── Backend factory ────────────────────────────────────────────────────

def _get_pgvector_url() -> Optional[str]:
    """Read PGVECTOR_URL from Streamlit secrets or environment."""
    url = None
    try:
        import streamlit as st
        url = st.secrets.get("PGVECTOR_URL")
    except Exception:
        pass
    return url or os.getenv("PGVECTOR_URL")


def make_vector_store() -> VectorStore | PgVectorStore:
    """
    Return the appropriate vector store backend:
      PGVECTOR_URL set  →  PgVectorStore (persistent)
      PGVECTOR_URL unset →  VectorStore  (in-memory FAISS)
    """
    url = _get_pgvector_url()
    if url:
        try:
            store = PgVectorStore(url)
            logger.info("Using pgvector backend (persistent)")
            return store
        except Exception as e:
            logger.warning(f"pgvector unavailable ({e}) — falling back to FAISS")
    logger.info("Using FAISS backend (in-memory)")
    return VectorStore()


# ── BM25 Index ─────────────────────────────────────────────────────────

class BM25Index:
    """BM25 sparse retrieval index for keyword matching."""

    def __init__(self):
        self.chunks: list[DocumentChunk] = []
        self._index: Optional[BM25Okapi] = None

    def add(self, chunks: list[DocumentChunk]):
        self.chunks.extend(chunks)
        tokenized = [self._tokenize(c.text) for c in self.chunks]
        self._index = BM25Okapi(tokenized)
        logger.info(f"BM25: {len(self.chunks)} chunks indexed")

    def search(self, query: str, k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if self._index is None or not self.chunks:
            return []
        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [
            (self.chunks[idx], float(scores[idx]))
            for idx in top_indices if scores[idx] > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()


# ── Hybrid Retriever ───────────────────────────────────────────────────

class HybridRetriever:
    """
    BM25 + dense vector retrieval with Reciprocal Rank Fusion.

    Vector backend is chosen automatically:
      PGVECTOR_URL set  →  PgVectorStore (persistent across restarts)
      PGVECTOR_URL unset →  VectorStore  (in-memory FAISS)
    """

    def __init__(
        self,
        embedder = None,
        dense_weight = 0.6,
        sparse_weight = 0.4,
    ):
        self._embedder = embedder
        self.vector_store = make_vector_store()
        self.bm25_index = BM25Index()
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self._query_cache: dict[str, np.ndarray] = {}

    @property
    def backend(self) -> str:
        return "pgvector" if isinstance(self.vector_store, PgVectorStore) else "faiss"
    
    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def index_chunks(self, chunks: list[DocumentChunk]):
        if not chunks:
            return
        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        self.vector_store.add(chunks, embeddings)
        self.bm25_index.add(chunks)
        self._query_cache.clear()
        logger.info(f"Indexed {len(chunks)} chunks via {self.backend}")

    def remove_document(self, document_name: str):
        if isinstance(self.vector_store, PgVectorStore):
            # pgvector: direct SQL delete — no rebuild needed
            self.vector_store.remove_document(document_name)
            # Rebuild BM25 from remaining chunks
            remaining = [c for c in self.bm25_index.chunks if c.document_name != document_name]
            self.bm25_index = BM25Index()
            if remaining:
                self.bm25_index.add(remaining)
        else:
            # FAISS: filter and rebuild
            remaining = [c for c in self.vector_store.chunks if c.document_name != document_name]
            self.vector_store = VectorStore()
            self.bm25_index = BM25Index()
            if remaining:
                embeddings = self.embedder.embed_texts([c.text for c in remaining])
                self.vector_store.add(remaining, embeddings)
                self.bm25_index.add(remaining)

        self._query_cache.clear()
        logger.info(f"Removed '{document_name}'")

    def clear(self):
        self.vector_store.clear() if isinstance(self.vector_store, PgVectorStore) \
            else setattr(self, 'vector_store', VectorStore())
        self.bm25_index = BM25Index()
        self._query_cache.clear()
        logger.info("Cleared all indexed documents")

    def retrieve(self, query: str, k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if query not in self._query_cache:
            self._query_cache[query] = self.embedder.embed_query(query)
        query_embedding = self._query_cache[query]
        dense  = self.vector_store.search(query_embedding, k=k * 2)
        sparse = self.bm25_index.search(query, k=k * 2)
        return self._reciprocal_rank_fusion(dense, sparse, k=k)

    def retrieve_with_citations(self, query: str, k: int = 5) -> tuple[list[str], list[Citation]]:
        results = self.retrieve(query, k=k)
        contexts, citations = [], []
        for chunk, score in results:
            contexts.append(
                f"[Source: {chunk.document_name} | Section: {chunk.section_heading} | "
                f"Page: {chunk.page_number}]\n{chunk.text}"
            )
            citations.append(Citation(
                source_document=chunk.document_name,
                section=chunk.section_heading,
                page=chunk.page_number,
                text_excerpt=chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                confidence=min(score, 1.0),
            ))
        return contexts, citations

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[DocumentChunk, float]],
        sparse_results: list[tuple[DocumentChunk, float]],
        k: int = 5,
        rrf_k: int = 60,
    ) -> list[tuple[DocumentChunk, float]]:
        scores: dict[str, float] = {}
        chunk_map: dict[str, DocumentChunk] = {}

        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0) + self.dense_weight / (rrf_k + rank + 1)

        for rank, (chunk, _) in enumerate(sparse_results):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0) + self.sparse_weight / (rrf_k + rank + 1)

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:k]
        return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]

