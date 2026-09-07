"""Hybrid search wrapper for Qdrant.

This module now acts as a thin adapter between the Qdrant native hybrid 
search (which handles Dense + Sparse + RRF internally) and the rest of 
the pipeline which expects `RetrievedSegment` objects.
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
    metadata_df: pd.DataFrame = None,
    dense_top_k: Optional[int] = None,   # Ignored by Qdrant (handled by limit/prefetch)
    sparse_top_k: Optional[int] = None,  # Ignored by Qdrant 
    rrf_k: Optional[int] = None,         # Ignored by Qdrant (native RRF used)
    final_top_k: Optional[int] = None,
    min_relevance_score: Optional[float] = None,
) -> list[RetrievedSegment]:
    """Run Qdrant Hybrid Search and format the output as RetrievedSegment."""
    
    cfg = load_config()
    p4 = cfg["phase4"]
    final_top_k = final_top_k or p4["final_top_k"]
    min_relevance_score = (
        min_relevance_score if min_relevance_score is not None else p4["min_relevance_score"]
    )

    # 1. Gọi thẳng hàm của Qdrant (bỏ qua faiss và bm25)
    raw_results = search_qdrant_hybrid(query=query, top_k=final_top_k)
    
    # 2. Tạo lookup dictionary để ghép metadata
    if metadata_df is not None:
        metadata_lookup = metadata_df.set_index("segment_id").to_dict(orient="index")
    else:
        metadata_lookup = {}

    # 3. Đóng gói kết quả
    scored: list[RetrievedSegment] = []
    for seg_id, score in raw_results:
        # Qdrant's RRF score is not normalized to [0,1], but we still apply the threshold
        if score < min_relevance_score:
            continue
            
        scored.append(
            RetrievedSegment(
                segment_id=seg_id,
                fused_score=float(score),
                metadata=metadata_lookup.get(seg_id, {}),
            )
        )

    # Sort again just to be safe, though Qdrant returns them sorted
    scored.sort(key=lambda r: r.fused_score, reverse=True)

    logger.info(
        "qdrant_hybrid_search(%r): returned top %d segments",
        query[:60],
        len(scored),
    )
    return scored
