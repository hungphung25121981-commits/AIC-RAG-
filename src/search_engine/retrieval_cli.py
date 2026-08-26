"""High-level retrieval pipeline used by the `query --s/--q` CLI mode.

hybrid_search -> (optional single-video filter) -> (optional --rerank)
-> (optional exact --select-frame) -> ResultRow list.

Kept separate from hybrid_search.py / generator.py so this CLI-specific
plumbing (video filtering, frame resolution, printing, CSV rows)
doesn't leak into the core multi-hop RAG library used by
`answer_question` (the plain `query <question>` mode).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.search_engine.csv_export import ResultRow
from src.search_engine.frame_selector import select_best_frame
from src.search_engine.hybrid_search import RetrievedSegment, hybrid_search
from src.search_engine.reranker import rerank_segments
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


def search_and_rank(
    query: str,
    faiss_index,
    bm25_corpus,
    metadata_df: pd.DataFrame,
    top_n: int,
    rerank: bool = False,
    video_id: Optional[str] = None,
    engine: Optional[QwenVLEngine] = None,
) -> list[RetrievedSegment]:
    """Run hybrid_search for `query`, optionally restricted to one `video_id`,
    optionally re-ranked with the VLM, truncated to `top_n` results.

    There's no per-video FAISS/BM25 sub-index, so when `video_id` is
    given we over-fetch from the fused corpus-wide search first, then
    filter down to that video and re-truncate.
    """
    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)

    search_top_n = top_n if not video_id else max(top_n * 10, cfg["phase4"]["dense_top_k"], 50)

    # When filtering to a single video_id, `final_top_k` alone isn't enough --
    # the underlying dense/sparse candidate pools (dense_top_k/sparse_top_k)
    # must ALSO be widened, otherwise they stay capped at the small config
    # default (e.g. 20/20) and a target video that isn't already in the
    # global top-20 gets silently filtered down to zero results.
    search_kwargs = dict(
        query=query,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        final_top_k=search_top_n,
    )
    if video_id:
        search_kwargs["dense_top_k"] = search_top_n
        search_kwargs["sparse_top_k"] = search_top_n

    segments = hybrid_search(**search_kwargs)

    if video_id:
        segments = [s for s in segments if s.metadata.get("video_id") == video_id]
        if not segments:
            logger.warning("Không có đoạn (segment) nào khớp video_id=%s cho truy vấn %r.", video_id, query[:60])

    if rerank and segments:
        segments = rerank_segments(query, segments, engine=engine)
    else:
        segments = sorted(segments, key=lambda s: s.fused_score, reverse=True)

    return segments[:top_n]


def segments_to_result_rows(
    query: str,
    segments: list[RetrievedSegment],
    select_frame: bool = False,
    keyframes_dir: Optional[str] = None,
    keyframe_map_csv: Optional[str] = None,
    engine: Optional[QwenVLEngine] = None,
) -> list[ResultRow]:
    """Resolve each segment down to one concrete frame_id + timestamp and build CSV rows."""
    rows: list[ResultRow] = []
    for rank, seg in enumerate(segments, start=1):
        meta = seg.metadata
        frame_choice = select_best_frame(
            query,
            seg,
            keyframes_dir=keyframes_dir,
            keyframe_map_csv=keyframe_map_csv,
            engine=engine,
            use_vlm=select_frame,
        )
        if frame_choice is None:
            frame_id, timestamp = "", float(meta.get("start_time", 0.0))
        else:
            frame_id, timestamp = frame_choice.frame_id, frame_choice.timestamp_sec

        rows.append(
            ResultRow(
                rank=rank,
                frame_id=frame_id,
                video_id=meta.get("video_id", ""),
                segment_id=seg.segment_id,
                timestamp_sec=timestamp,
                score=seg.fused_score,
                video_title=meta.get("video_title", ""),
                jump_link=meta.get("jump_link", ""),
            )
        )
    return rows


def print_result_rows(rows: list[ResultRow], console_limit: int = 20) -> None:
    if not rows:
        print("(Không có kết quả nào.)")
        return
    print(f"{'rank':<5}{'frame_id':<28}{'video_id':<16}{'t(s)':<10}{'score':<10}")
    for row in rows[:console_limit]:
        print(f"{row.rank:<5}{row.frame_id:<28}{row.video_id:<16}{row.timestamp_sec:<10.1f}{row.score:<10.4f}")
    if len(rows) > console_limit:
        print(f"... còn {len(rows) - console_limit} dòng nữa -- xem đầy đủ trong --out-csv.")
