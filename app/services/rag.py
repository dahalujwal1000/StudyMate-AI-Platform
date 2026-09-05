"""Lightweight RAG: chunk retrieval via TF-IDF cosine similarity (stdlib only).

Swap-in friendly: replace `search()` with Chroma/FAISS embeddings later —
callers only depend on the `ChunkHit` dataclass.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "of", "to", "in", "on", "for",
    "with", "as", "by", "at", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "from", "into", "your", "you", "we", "they",
    "he", "she", "his", "her", "not", "no", "can", "will", "would", "should", "could",
    "do", "does", "did", "have", "has", "had", "what", "which", "who", "when", "where",
    "why", "how", "about", "there", "their", "them", "than", "also", "so", "such",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOP]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and p.strip() and len(p.strip()) > 20]


def chunk_text(text: str, target: int = 1100, overlap_sentences: int = 1) -> list[str]:
    sentences = split_sentences(text)
    if not sentences:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for s in sentences:
        buf.append(s)
        size += len(s)
        if size >= target:
            chunks.append(" ".join(buf))
            buf = buf[-overlap_sentences:]
            size = sum(len(x) for x in buf)
    if buf:
        tail = " ".join(buf)
        if chunks and len(tail) < 120:
            chunks[-1] += " " + tail
        else:
            chunks.append(tail)
    return chunks


@dataclass
class ChunkHit:
    document_id: int | None
    chunk_id: int
    idx: int
    label: str
    content: str
    score: float


class TfidfIndex:
    """In-memory TF-IDF index over a user's document chunks."""

    def __init__(self) -> None:
        self._docs: dict[int, list[str]] = {}  # chunk_id -> tokens
        self._meta: dict[int, ChunkHit] = {}
        self._idf: dict[str, float] = {}
        self._tf: dict[int, Counter[str]] = {}

    def rebuild(self, rows: list[tuple[int, int, str, str, str]]) -> None:
        """rows: (chunk_id, idx, label, content, ...)"""
        self._docs.clear()
        self._meta.clear()
        self._tf.clear()
        for chunk_id, idx, label, content in rows:
            tokens = tokenize(content)
            if not tokens:
                continue
            self._docs[chunk_id] = tokens
            self._meta[chunk_id] = ChunkHit(None, chunk_id, idx, label, content, 0.0)
            self._tf[chunk_id] = Counter(tokens)
        n = max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for tokens in self._docs.values():
            df.update(set(tokens))
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def search(self, query: str, k: int = 5, document_id: int | None = None) -> list[ChunkHit]:
        if not self._docs:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        q_counter = Counter(q_tokens)
        hits: list[ChunkHit] = []
        for chunk_id, tokens in self._docs.items():
            if document_id is not None and self._meta[chunk_id].document_id != document_id:
                continue
            norm = math.sqrt(sum(v * v for v in self._tf[chunk_id].values()))
            score = 0.0
            for t, qv in q_counter.items():
                tfv = self._tf[chunk_id].get(t, 0)
                if not tfv:
                    continue
                score += (tfv / norm) * self._idf.get(t, 1.0) * qv
            if score > 0:
                meta = self._meta[chunk_id]
                hit = ChunkHit(meta.document_id, chunk_id, meta.idx, meta.label,
                               meta.content, round(score, 6))
                hits.append(hit)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


_INDEX: TfidfIndex | None = None


def get_index() -> TfidfIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = TfidfIndex()
    return _INDEX


def rebuild_index(db) -> None:
    """Rebuild the global index from all chunks in the DB (called after uploads)."""
    from ..models import Chunk

    rows = [(c.id, c.idx, c.label, c.content) for c in db.query(Chunk).all()]
    index = get_index()
    index.rebuild(rows)
    # store document ownership for filtering
    for chunk in db.query(Chunk).all():
        if chunk.id in index._meta:
            index._meta[chunk.id].document_id = chunk.document_id  # type: ignore[attr-defined]
