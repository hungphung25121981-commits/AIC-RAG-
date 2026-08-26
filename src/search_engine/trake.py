"""TRAKE track: locate N sequential sub-moments ("khoanh khac") of ONE
event inside a SINGLE video, each described by a short stage query.

Example: an athlete's high jump has 4 stages -- (1) giam nhay [takeoff],
(2) bay qua xa [clearing the bar], (3) tiep dat [landing],
(4) dung day [standing up] -- and the task is to return the frame_id
that best represents each stage, IN THE SAME VIDEO, in order.

Two-stage retrieve-and-align (matches how this task is normally solved):

  Stage A -- coarse VIDEO localization: figure out WHICH video the
    whole event happens in. Either given directly via --video-id, or
    found by running one hybrid_search over the FULL corpus with the
    overall event description (or all stage texts joined) and taking
    the top-ranked segment's video_id.

  Stage B -- per-stage localization + temporal alignment: for each
    stage description, in order, search restricted to that one video,
    take the best-scoring candidate segments, and resolve the exact
    frame_id inside each with the VLM frame-picker
    (frame_selector.select_best_frame). A monotonic-time constraint is
    then enforced: since the stages happen in real time order within
    the same video, each stage's chosen timestamp should be >= the
    previous stage's. Among the top few candidate segments for a
    stage, the highest-scoring one that respects this ordering is
    picked; if NONE respect it, the single best-scoring candidate is
    kept anyway and a warning is logged (never silently drops a stage).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.search_engine.frame_selector import select_best_frame
from src.search_engine.hybrid_search import RetrievedSegment, hybrid_search
from src.utils_common import ensure_dir, get_logger, load_config

logger = get_logger(__name__)

# How many top-scoring same-video candidates to check per stage when
# looking for one that respects the monotonic-time ordering. Kept small
# because each check costs one VLM frame-selection call.
_MAX_CANDIDATES_PER_STAGE = 5


@dataclass
class TrakeStageResult:
    stage_index: int
    stage_query: str
    video_id: str
    segment_id: int
    frame_id: str
    timestamp_sec: float
    score: float
    ok: bool = True  # False if no usable segment/frame was found at all for this stage


def localize_target_video(
    query: str,
    faiss_index,
    bm25_corpus,
    metadata_df: pd.DataFrame,
    search_top_n: int = 50,
) -> Optional[str]:
    """Stage A: coarse search across the WHOLE corpus, return the top segment's video_id."""
    segments = hybrid_search(
        query=query,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        final_top_k=search_top_n,
    )
    if not segments:
        return None
    return segments[0].metadata.get("video_id")


def _search_within_video(
    query: str,
    video_id: str,
    faiss_index,
    bm25_corpus,
    metadata_df: pd.DataFrame,
    search_top_n: int,
) -> list[RetrievedSegment]:
    """Stage B search: over-fetch from the corpus-wide fused search, then filter to one video_id.

    (There's no per-video FAISS/BM25 sub-index to search directly.)
    """
    # dense_top_k/sparse_top_k must be widened too (not just final_top_k), otherwise
    # the underlying candidate pool stays capped at the small config default and the
    # target video's segments may never surface before the post-filter. See the
    # matching fix in retrieval_cli.py::search_and_rank for the same issue.
    segments = hybrid_search(
        query=query,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        final_top_k=search_top_n,
        dense_top_k=search_top_n,
        sparse_top_k=search_top_n,
    )
    matched = [s for s in segments if s.metadata.get("video_id") == video_id]
    return sorted(matched, key=lambda s: s.fused_score, reverse=True)


def run_trake(
    stages: list[str],
    faiss_index,
    bm25_corpus,
    metadata_df: pd.DataFrame,
    video_id: Optional[str] = None,
    overall_query: Optional[str] = None,
    keyframes_dir: Optional[str] = None,
    keyframe_map_csv: Optional[str] = None,
    engine: Optional[QwenVLEngine] = None,
    search_top_n: Optional[int] = None,
) -> list[TrakeStageResult]:
    if not stages:
        raise ValueError("TRAKE cần it nhat 1 --stages.")

    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)
    search_top_n = search_top_n or cfg["phase4"].get("trake_search_top_n", 50)

    if not video_id:
        localization_query = overall_query or " ; ".join(stages)
        video_id = localize_target_video(localization_query, faiss_index, bm25_corpus, metadata_df, search_top_n)
        if not video_id:
            raise RuntimeError(
                "Stage A khong xac dinh duoc video muc tieu (khong tim thay segment nao). "
                "Hay truyen --video-id truc tiep neu ban da biet la video nao."
            )
        logger.info("TRAKE Stage A: localized video_id=%s", video_id)
    else:
        logger.info("TRAKE Stage A: dung video_id=%s do nguoi dung chi dinh (bo qua tim kiem toan corpus).", video_id)

    results: list[TrakeStageResult] = []
    last_timestamp: Optional[float] = None

    for i, stage_query in enumerate(stages, start=1):
        candidates = _search_within_video(
            stage_query, video_id, faiss_index, bm25_corpus, metadata_df, search_top_n
        )
        if not candidates:
            logger.warning("TRAKE stage %d (%r): khong tim thay segment nao trong video_id=%s.", i, stage_query, video_id)
            results.append(
                TrakeStageResult(
                    stage_index=i, stage_query=stage_query, video_id=video_id,
                    segment_id=-1, frame_id="", timestamp_sec=-1.0, score=0.0, ok=False,
                )
            )
            continue

        chosen_seg, chosen_frame = None, None
        fallback_seg, fallback_frame = None, None

        for cand in candidates[:_MAX_CANDIDATES_PER_STAGE]:
            frame_choice = select_best_frame(
                stage_query, cand,
                keyframes_dir=keyframes_dir, keyframe_map_csv=keyframe_map_csv,
                engine=engine, use_vlm=True,
            )
            if frame_choice is None:
                continue
            if fallback_seg is None:  # keep the best-scoring candidate as a safety net
                fallback_seg, fallback_frame = cand, frame_choice
            if last_timestamp is None or frame_choice.timestamp_sec >= last_timestamp:
                chosen_seg, chosen_frame = cand, frame_choice
                break

        if chosen_seg is None:
            chosen_seg, chosen_frame = fallback_seg, fallback_frame
            if chosen_seg is not None and last_timestamp is not None:
                logger.warning(
                    "TRAKE stage %d (%r): khong co ung vien nao co t >= stage truoc (t=%.1fs); "
                    "van giu ung vien tot nhat (t=%.1fs) -- thu tu thoi gian co the bi lech.",
                    i, stage_query, last_timestamp, chosen_frame.timestamp_sec,
                )

        if chosen_seg is None or chosen_frame is None:
            logger.warning("TRAKE stage %d (%r): khong xac dinh duoc frame nao.", i, stage_query)
            results.append(
                TrakeStageResult(
                    stage_index=i, stage_query=stage_query, video_id=video_id,
                    segment_id=-1, frame_id="", timestamp_sec=-1.0, score=0.0, ok=False,
                )
            )
            continue

        last_timestamp = chosen_frame.timestamp_sec
        results.append(
            TrakeStageResult(
                stage_index=i,
                stage_query=stage_query,
                video_id=video_id,
                segment_id=chosen_seg.segment_id,
                frame_id=chosen_frame.frame_id,
                timestamp_sec=chosen_frame.timestamp_sec,
                score=chosen_seg.fused_score,
                ok=True,
            )
        )

    return results


def write_trake_csv(results: list[TrakeStageResult], out_csv: str | Path) -> Path:
    out_path = Path(out_csv)
    ensure_dir(out_path.parent)
    df = pd.DataFrame([r.__dict__ for r in results])
    df.to_csv(out_path, index=False)
    logger.info("Wrote TRAKE result (%d stage(s)) to %s", len(results), out_path)
    return out_path
