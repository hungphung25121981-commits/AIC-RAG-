"""Keyframe -> segment grouping and keyframe_map.csv generation.

Combines PySceneDetect boundaries with a target segment window
(15-30s, default 20s from settings.yaml) so that:
  - segments never straddle a hard scene cut where avoidable
  - segments stay within the configured min/max duration
  - every kept keyframe (post-SSIM-filter) is assigned a contiguous
    integer segment_id per video, matching the FAISS row requirement
    downstream (segment_id 0..N-1 contiguous, enforced in Phase 3).

Output schema (keyframe_map.csv):
    frame_id (str), video_id (str), timestamp_sec (float), segment_id (int)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.phase1_extraction.scene_detect import detect_scenes
from src.phase1_extraction.ssim_filter import filter_near_duplicate_frames, save_frame_candidates
from src.utils_common import ensure_dir, get_logger, load_config

logger = get_logger(__name__)


def _windows_from_scenes(
    scene_boundaries: list[tuple[float, float]],
    window_sec: float,
    min_sec: float,
    max_sec: float,
) -> list[tuple[float, float]]:
    """Split each scene into one or more segment windows within [min_sec, max_sec]."""
    windows: list[tuple[float, float]] = []
    for start, end in scene_boundaries:
        duration = end - start
        if duration <= max_sec:
            windows.append((start, end))
            continue
        n_chunks = max(round(duration / window_sec), 1)
        chunk_len = duration / n_chunks
        chunk_len = min(max(chunk_len, min_sec), max_sec)
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + chunk_len, end)
            windows.append((cursor, chunk_end))
            cursor = chunk_end
    return windows


def build_keyframe_map_for_video(
    video_path: str | Path,
    video_id: str,
    keyframes_output_dir: str | Path,
    sample_fps: float = 2.0,
) -> pd.DataFrame:
    """Run SSIM filtering + scene detection + windowing for one video.

    Returns a per-video DataFrame with columns:
        frame_id, video_id, timestamp_sec, segment_id
    (segment_id is contiguous 0..N-1 *within this video*; the caller /
    metadata_builder is responsible for making it globally contiguous
    across the whole corpus before FAISS indexing.)
    """
    cfg = load_config()
    p1 = cfg["phase1"]

    logger.info("Phase1: processing video_id=%s (%s)", video_id, video_path)

    kept_frames = filter_near_duplicate_frames(
        video_path, ssim_threshold=p1["ssim_threshold"], sample_fps=sample_fps
    )
    if not kept_frames:
        logger.warning("No frames survived SSIM filtering for %s", video_id)
        return pd.DataFrame(columns=["frame_id", "video_id", "timestamp_sec", "segment_id"])

    frame_records = save_frame_candidates(kept_frames, video_id, keyframes_output_dir)

    scene_boundaries = detect_scenes(
        video_path,
        threshold=p1["scenedetect_threshold"],
        min_scene_len_sec=p1["min_scene_len_sec"],
    )
    windows = _windows_from_scenes(
        scene_boundaries,
        window_sec=p1["segment_window_sec"],
        min_sec=p1["segment_window_min_sec"],
        max_sec=p1["segment_window_max_sec"],
    )

    rows = []
    for rec in frame_records:
        ts = rec["timestamp_sec"]
        segment_id = _assign_window_id(ts, windows)
        rows.append(
            {
                "frame_id": rec["frame_id"],
                "video_id": video_id,
                "timestamp_sec": ts,
                "segment_id": segment_id,
            }
        )

    df = pd.DataFrame(rows).sort_values(["segment_id", "timestamp_sec"]).reset_index(drop=True)
    df = _cap_frames_per_segment(df, p1["max_keyframes_per_segment"])
    df = _renumber_segments_contiguously(df)
    logger.info(
        "video_id=%s: %d keyframes across %d segments", video_id, len(df), df["segment_id"].nunique()
    )
    return df


def _assign_window_id(timestamp_sec: float, windows: list[tuple[float, float]]) -> int:
    for idx, (start, end) in enumerate(windows):
        if start <= timestamp_sec < end:
            return idx
    return max(len(windows) - 1, 0)


def _cap_frames_per_segment(df: pd.DataFrame, max_per_segment: int) -> pd.DataFrame:
    """Evenly subsample if a segment has more keyframes than allowed."""
    parts = []
    for _, group in df.groupby("segment_id", sort=True):
        if len(group) > max_per_segment:
            step = len(group) / max_per_segment
            indices = [int(i * step) for i in range(max_per_segment)]
            group = group.iloc[indices]
        parts.append(group)
    return pd.concat(parts).reset_index(drop=True)


def _renumber_segments_contiguously(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure segment_id is contiguous 0..N-1 within this video after any drops."""
    unique_ids = sorted(df["segment_id"].unique())
    remap = {old: new for new, old in enumerate(unique_ids)}
    df = df.copy()
    df["segment_id"] = df["segment_id"].map(remap)
    return df


def build_corpus_keyframe_map(
    video_specs: list[dict],
    output_csv: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Run build_keyframe_map_for_video across many videos and write keyframe_map.csv.

    video_specs: list of {"video_path": ..., "video_id": ..., "keyframes_output_dir": ...}
    NOTE: segment_id here stays PER-VIDEO contiguous; metadata_builder.py
    is responsible for creating the GLOBAL contiguous segment_id used by
    FAISS (video_id, local_segment_id) -> global segment_id.
    """
    cfg = load_config()
    output_csv = output_csv or cfg["paths"]["keyframe_map_csv"]

    all_dfs = []
    for spec in video_specs:
        df = build_keyframe_map_for_video(
            video_path=spec["video_path"],
            video_id=spec["video_id"],
            keyframes_output_dir=spec.get(
                "keyframes_output_dir", cfg["paths"]["keyframes_dir"]
            ),
        )
        all_dfs.append(df)

    corpus_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame(
        columns=["frame_id", "video_id", "timestamp_sec", "segment_id"]
    )

    out_path = ensure_dir(Path(str(output_csv)).parent) / Path(str(output_csv)).name
    corpus_df.to_csv(out_path, index=False)
    logger.info("Wrote keyframe_map.csv with %d rows to %s", len(corpus_df), out_path)
    return corpus_df
