"""Hybrid dense (FAISS) + sparse (BM25) search with Reciprocal Rank Fusion.

RRF is used instead of naive score-averaging because dense cosine
similarity and BM25 scores live on incompatible scales; RRF only needs
each list's RANK, making fusion scale-free and robust.

    RRF(segment) = sum over each ranker r of  1 / (rrf_k + rank_r(segment))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.phase3_indexing.bm25_indexer import BM25Corpus
from src.phase3_indexing.embedder import embed_texts
from src.phase3_indexing.faiss_indexer import search_faiss_index
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


@dataclass
class RetrievedSegment:
    segment_id: int
    fused_score: float
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    sparse_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    sparse_results: list[tuple[int, float]],
    rrf_k: int,
) -> dict[int, dict]:
    """Fuse two ranked (segment_id, score) lists into per-segment fused stats."""
    fused: dict[int, dict] = {}

    for rank, (seg_id, score) in enumerate(dense_results, start=1):
        entry = fused.setdefault(seg_id, {"rrf": 0.0})
        entry["rrf"] += 1.0 / (rrf_k + rank)
        entry["dense_score"] = score
        entry["dense_rank"] = rank

    for rank, (seg_id, score) in enumerate(sparse_results, start=1):
        entry = fused.setdefault(seg_id, {"rrf": 0.0})
        entry["rrf"] += 1.0 / (rrf_k + rank)
        entry["sparse_score"] = score
        entry["sparse_rank"] = rank

    return fused


def hybrid_search(
    query: str,
    faiss_index,
    bm25_corpus: BM25Corpus,
    metadata_df: pd.DataFrame,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    final_top_k: Optional[int] = None,
    min_relevance_score: Optional[float] = None,
) -> list[RetrievedSegment]:
    """Run FAISS + BM25 retrieval for `query`, fuse with RRF, attach metadata.

    Returns segments sorted by fused_score descending, truncated to final_top_k
    and filtered by min_relevance_score.
    """
    cfg = load_config()
    p4 = cfg["phase4"]
    dense_top_k = dense_top_k or p4["dense_top_k"]
    sparse_top_k = sparse_top_k or p4["sparse_top_k"]
    rrf_k = rrf_k or p4["rrf_k"]
    final_top_k = final_top_k or p4["final_top_k"]
    min_relevance_score = (
        min_relevance_score if min_relevance_score is not None else p4["min_relevance_score"]
    )

    query_vector = embed_texts([query])[0]
    dense_results = search_faiss_index(faiss_index, np.asarray(query_vector), top_k=dense_top_k)
    sparse_results = bm25_corpus.search(query, top_k=sparse_top_k)

    fused = reciprocal_rank_fusion(dense_results, sparse_results, rrf_k=rrf_k)

    # Normalize fused RRF scores to [0, 1] for interpretable min_relevance_score filtering.
    max_possible = (1.0 / (rrf_k + 1)) * 2  # both rankers rank it #1
    metadata_lookup = metadata_df.set_index("segment_id").to_dict(orient="index")

    scored: list[RetrievedSegment] = []
    for seg_id, stats in fused.items():
        normalized_score = stats["rrf"] / max_possible if max_possible > 0 else 0.0
        if normalized_score < min_relevance_score:
            continue
        scored.append(
            RetrievedSegment(
                segment_id=seg_id,
                fused_score=normalized_score,
                dense_score=stats.get("dense_score"),
                dense_rank=stats.get("dense_rank"),
                sparse_score=stats.get("sparse_score"),
                sparse_rank=stats.get("sparse_rank"),
                metadata=metadata_lookup.get(seg_id, {}),
            )
        )

    scored.sort(key=lambda r: r.fused_score, reverse=True)
    top_results = scored[:final_top_k]

    logger.info(
        "hybrid_search(%r): dense=%d sparse=%d fused=%d -> returning top %d",
        query[:60],
        len(dense_results),
        len(sparse_results),
        len(scored),
        len(top_results),
    )
    return top_results
