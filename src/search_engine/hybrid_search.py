"""Hybrid search wrapper for Qdrant.

This module acts as a clean adapter between the Qdrant native hybrid 
search (which handles Dense BGE-M3 + Visual SigLIP + Sparse BM25 + RRF internally)
and the rest of the pipeline which expects `RetrievedSegment` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.phase3_indexing.Qdrant_index import search_qdrant_hybrid
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


def hybrid_search(
    query: str,
    faiss_index=None,     # Kept for backward compatibility with CLI, ignored
    bm25_corpus=None,     # Kept for backward compatibility with CLI, ignored
    metadata_df: Optional[pd.DataFrame] = None,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    visual_top_k: Optional[int] = None,
    rrf_k: Optional[int] = None,         # Ignored by Qdrant (native RRF used)
    final_top_k: Optional[int] = None,
    min_relevance_score: Optional[float] = None,
    video_id: Optional[str] = None,
) -> list[RetrievedSegment]:
    """Run Qdrant Hybrid Search and format the output as RetrievedSegment."""
    cfg = load_config()
    p4 = cfg["phase4"]
    final_top_k = final_top_k or p4["final_top_k"]
    min_relevance_score = (
        min_relevance_score if min_relevance_score is not None else p4.get("min_relevance_score", 0.0)
    )

    # RRF score safety: Qdrant's RRF score is sum of 1/(60+rank) <= ~0.049.
    # If a legacy cosine threshold (> 0.05) was passed in, reset it to 0.0 so it doesn't drop all candidates.
    if min_relevance_score > 0.05:
        logger.warning(
            "min_relevance_score=%s is too high for RRF (max RRF score is ~0.049); adjusting to 0.0.",
            min_relevance_score,
        )
        min_relevance_score = 0.0

    raw_results = search_qdrant_hybrid(
        query=query,
        top_k=final_top_k,
        video_id=video_id,
        dense_top_k=dense_top_k,
        sparse_top_k=sparse_top_k,
        visual_top_k=visual_top_k,
    )

    metadata_lookup = {}
    if metadata_df is not None:
        metadata_lookup = metadata_df.set_index("segment_id").to_dict(orient="index")

    scored: list[RetrievedSegment] = []
    for item in raw_results:
        seg_id = item[0]
        score = float(item[1])
        payload = item[2] if len(item) > 2 else {}

        if score < min_relevance_score:
            continue

        meta = metadata_lookup.get(seg_id, payload)
        if not meta and payload:
            meta = payload

        scored.append(
            RetrievedSegment(
                segment_id=seg_id,
                fused_score=score,
                metadata=meta,
            )
        )

    scored.sort(key=lambda r: r.fused_score, reverse=True)
    logger.info(
        "qdrant_hybrid_search(%r): returned top %d segments (filter video_id=%s)",
        query[:60],
        len(scored),
        video_id or "all",
    )
    return scored
