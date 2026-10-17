"""Vector / keyword retrieval over document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.chunker import Chunk
from src.embeddings import embed_query, embed_texts
from src.reranker import DEFAULT_MIN_RERANK_SCORE, cross_encoder_scores

DEFAULT_TOP_K = 10
DEFAULT_HYBRID_FINAL_K = 5
# Cosine similarity (inner product on normalized vectors). Tune if needed.
DEFAULT_EVIDENCE_THRESHOLD = 0.35


class ChunkRecord(TypedDict):
    chunk_id: int
    page_number: int
    chapter: str | None
    text: str


class RetrievalResult(TypedDict):
    chunk_id: int
    page_number: int
    chapter: str | None
    text: str
    score: float


class SearchResponse(TypedDict):
    query: str
    results: list[RetrievalResult]
    sufficient_evidence: bool


@dataclass
class SemanticIndex:
    """FAISS index plus aligned chunk metadata for semantic search."""

    index: faiss.Index
    metadata: list[ChunkRecord]

    @property
    def size(self) -> int:
        return len(self.metadata)


@dataclass
class HybridIndex:
    """FAISS semantic index plus BM25 corpus over the same chunk metadata."""

    semantic: SemanticIndex
    bm25: BM25Okapi | None

    @property
    def metadata(self) -> list[ChunkRecord]:
        return self.semantic.metadata

    @property
    def size(self) -> int:
        return self.semantic.size


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _chunks_to_records(chunks: list[Chunk]) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for chunk in chunks:
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "chapter": chunk.get("chapter"),  # not extracted yet
                "text": chunk["text"],
            }
        )
    return records


def build_semantic_index(chunks: list[Chunk]) -> SemanticIndex:
    """Embed all chunks and build a FAISS inner-product index."""
    records = _chunks_to_records(chunks)
    if not records:
        empty_index = faiss.IndexFlatIP(384)
        return SemanticIndex(index=empty_index, metadata=[])

    texts = [record["text"] for record in records]
    vectors = embed_texts(texts)
    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)

    return SemanticIndex(index=index, metadata=records)


def build_hybrid_index(chunks: list[Chunk]) -> HybridIndex:
    """Build FAISS and BM25 indexes over the same chunk metadata."""
    semantic = build_semantic_index(chunks)
    if not semantic.metadata:
        return HybridIndex(semantic=semantic, bm25=None)

    tokenized_corpus = [_tokenize(record["text"]) for record in semantic.metadata]
    bm25 = BM25Okapi(tokenized_corpus)
    return HybridIndex(semantic=semantic, bm25=bm25)


def _faiss_candidate_indices(
    semantic_index: SemanticIndex,
    query: str,
    top_k: int,
) -> list[int]:
    if semantic_index.size == 0 or not query.strip():
        return []

    query_vector = embed_query(query.strip())
    query_matrix = np.asarray([query_vector], dtype=np.float32)
    k = min(top_k, semantic_index.size)
    _, indices = semantic_index.index.search(query_matrix, k)

    ordered: list[int] = []
    seen: set[int] = set()
    for idx in indices[0]:
        if idx < 0 or idx in seen:
            continue
        seen.add(int(idx))
        ordered.append(int(idx))
    return ordered


def _bm25_candidate_indices(
    hybrid_index: HybridIndex,
    query: str,
    top_k: int,
) -> list[int]:
    if hybrid_index.bm25 is None or hybrid_index.size == 0 or not query.strip():
        return []

    query_tokens = _tokenize(query.strip())
    if not query_tokens:
        return []

    scores = hybrid_index.bm25.get_scores(query_tokens)
    ranked = np.argsort(scores)[::-1]

    ordered: list[int] = []
    seen: set[int] = set()
    for idx in ranked:
        idx = int(idx)
        if scores[idx] <= 0:
            break
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
        if len(ordered) >= top_k:
            break
    return ordered


def _merge_candidate_indices(
    faiss_indices: list[int],
    bm25_indices: list[int],
) -> list[int]:
    merged: list[int] = []
    seen: set[int] = set()
    for idx in faiss_indices + bm25_indices:
        if idx in seen:
            continue
        seen.add(idx)
        merged.append(idx)
    return merged


def semantic_search(
    semantic_index: SemanticIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    evidence_threshold: float = DEFAULT_EVIDENCE_THRESHOLD,
) -> SearchResponse:
    """
    Return up to top_k chunks ranked by cosine similarity.
    If no result meets the evidence threshold, results are empty.
    """
    empty: SearchResponse = {
        "query": query,
        "results": [],
        "sufficient_evidence": False,
    }

    if not query.strip() or semantic_index.size == 0:
        return empty

    query_vector = embed_query(query.strip())
    query_matrix = np.asarray([query_vector], dtype=np.float32)

    k = min(top_k, semantic_index.size)
    scores, indices = semantic_index.index.search(query_matrix, k)

    results: list[RetrievalResult] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        similarity = float(score)
        if similarity < evidence_threshold:
            continue
        meta = semantic_index.metadata[idx]
        results.append(
            {
                "chunk_id": meta["chunk_id"],
                "page_number": meta["page_number"],
                "chapter": meta["chapter"],
                "text": meta["text"],
                "score": similarity,
            }
        )

    return {
        "query": query,
        "results": results,
        "sufficient_evidence": len(results) > 0,
    }


def hybrid_search(
    hybrid_index: HybridIndex,
    query: str,
    retrieve_k: int = DEFAULT_TOP_K,
    top_k: int = DEFAULT_HYBRID_FINAL_K,
    min_rerank_score: float = DEFAULT_MIN_RERANK_SCORE,
) -> SearchResponse:
    """FAISS + BM25 candidate retrieval, CrossEncoder rerank, top-k evidence."""
    empty: SearchResponse = {
        "query": query,
        "results": [],
        "sufficient_evidence": False,
    }

    if not query.strip() or hybrid_index.size == 0:
        return empty

    faiss_indices = _faiss_candidate_indices(hybrid_index.semantic, query, retrieve_k)
    bm25_indices = _bm25_candidate_indices(hybrid_index, query, retrieve_k)
    candidate_indices = _merge_candidate_indices(faiss_indices, bm25_indices)

    if not candidate_indices:
        return empty

    candidates = [hybrid_index.metadata[idx] for idx in candidate_indices]
    passages = [record["text"] for record in candidates]
    scores = cross_encoder_scores(query, passages)

    scored: list[RetrievalResult] = []
    for record, score in zip(candidates, scores):
        if score < min_rerank_score:
            continue
        scored.append(
            {
                "chunk_id": record["chunk_id"],
                "page_number": record["page_number"],
                "chapter": record["chapter"],
                "text": record["text"],
                "score": score,
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    results = scored[:top_k]

    return {
        "query": query,
        "results": results,
        "sufficient_evidence": len(results) > 0,
    }


def evidence_for_llm(search_response: SearchResponse) -> list[RetrievalResult]:
    """Evidence payload for a future LLM layer (verbatim chunk text only)."""
    if not search_response["sufficient_evidence"]:
        return []
    return list(search_response["results"])
