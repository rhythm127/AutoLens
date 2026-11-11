"""
hybrid_retriever.py — Combines ChromaDB (Dense) + BM25 (Sparse) using Reciprocal Rank Fusion.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Plus

from app.rag.indexer import RetrievedRecord, VectorIndexService


def _tokenize(text: str) -> list[str]:
    """Simple lowercase word/alphanumeric tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self, vector_service: VectorIndexService, rrf_k: int = 60) -> None:
        self.vector_service = vector_service
        self.rrf_k = rrf_k  # Standard RRF constant to smooth rank differences

    def search(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[RetrievedRecord]:
        """
        Run both Dense (Vector) and Sparse (BM25) search, then fuse rankings via RRF.
        """
        # 1. Fetch Dense Vector Results from ChromaDB
        dense_results = self.vector_service.query(
            query_text=query_text,
            n_results=n_results * 2,  # Fetch a wider pool for fusion
            where=where,
        )

        # 2. Fetch candidate pool for BM25 keyword search (filtered by same vehicle metadata)
        corpus_records = self.vector_service.get(where=where)
        if not corpus_records:
            return []

        # Build BM25Plus index on the fly (BM25Plus guarantees strictly positive IDF
        # even on tiny 1- or 2-chunk corpora where standard BM25Okapi IDF hits ln(1)=0)
        tokenized_corpus = [_tokenize(r["text"]) for r in corpus_records]
        bm25 = BM25Plus(tokenized_corpus)
        tokenized_query = _tokenize(query_text)
        query_token_set = set(tokenized_query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Only keep sparse matches that share at least one actual token with the query
        scored_sparse = [
            (record, float(score))
            for record, tokens, score in zip(corpus_records, tokenized_corpus, bm25_scores)
            if query_token_set.intersection(tokens) and score > 0
        ]
        scored_sparse.sort(key=lambda item: item[1], reverse=True)
        sparse_results = [record for record, _ in scored_sparse[: n_results * 2]]

        # 3. Reciprocal Rank Fusion (RRF)
        # Formula: RRF_Score(doc) = 1 / (k + rank_dense) + 1 / (k + rank_sparse)
        rrf_scores: dict[str, float] = {}
        record_map: dict[str, RetrievedRecord] = {}

        for rank, record in enumerate(dense_results, start=1):
            doc_id = record["id"]
            record_map[doc_id] = record
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, record in enumerate(sparse_results, start=1):
            doc_id = record["id"]
            if doc_id not in record_map:
                record_map[doc_id] = record
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort by final fused RRF score descending
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda d_id: rrf_scores[d_id], reverse=True
        )

        # Return top n_results records with their fused score stored in metadata
        final_results: list[RetrievedRecord] = []
        for doc_id in sorted_ids[:n_results]:
            rec = dict(record_map[doc_id])
            rec["metadata"] = dict(rec["metadata"])
            rec["metadata"]["rrf_score"] = round(rrf_scores[doc_id], 5)
            final_results.append(rec)  # type: ignore

        return final_results