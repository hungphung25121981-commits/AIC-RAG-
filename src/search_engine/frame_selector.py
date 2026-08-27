"""Segment -> exact frame_id resolution.

hybrid_search + rerank operate at SEGMENT granularity (each segment
spans several keyframes / a 15-30s window). CSV export and TRAKE need
one concrete `frame_id`, so this module picks, among a segment's own
keyframes, the ONE frame that best matches the query -- then resolves
that frame_id's exact timestamp from keyframe_map.csv (the only place
PER-FRAME timestamps are kept; metadata.parquet only stores
segment-level start_time/end_time).

Three selection modes (`use_vlm` + `fast_mode` below):
  - "middle": free, no model call at all -- just the temporal midpoint
    keyframe of the segment. Used as the last-resort fallback if every
    other mode fails to load/parse.
  - "siglip" (fast path DEFAULT when use_vlm=False, e.g. bulk
    --s/--top-n rows without --select-frame): one cheap SigLIP
    image-text similarity pass across the segment's few keyframes --
    much smarter than "middle" for near-zero extra latency (no
    generation, just embeddings). See phase2_captioning/visual_embedder.py.
    Falls back to "middle" automatically if SigLIP fails to load.
  - "vlm" (--select-frame, and always for --q/--qa/trake): one extra
    Qwen2.5-VL generation call per segment for the most precise pick.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.search_engine.hybrid_search import RetrievedSegment
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


FRAME_PICK_PROMPT_TEMPLATE = """These {n} images are sequential keyframes (Frame 1..Frame {n})
from one short video segment.

QUERY: "{query}"

Which single frame best represents/matches the query? Respond with
ONLY the frame number (an integer from 1 to {n}), nothing else."""


@dataclass
class FrameChoice:
    frame_id: str
    timestamp_sec: float
    frame_index_in_segment: int


@lru_cache(maxsize=4)
def _load_frame_timestamp_lookup(keyframe_map_csv: str) -> dict:
    """frame_id -> timestamp_sec, read once from keyframe_map.csv and cached."""
    path = Path(keyframe_map_csv)
    if not path.exists():
        logger.warning(
            "keyframe_map.csv not found at %s; frame timestamps will fall back to segment start_time.",
            path,
        )
        return {}
    df = pd.read_csv(path)
    return dict(zip(df["frame_id"], df["timestamp_sec"]))


def _resolve_keyframe_paths(
    segment: RetrievedSegment, keyframes_dir: str | Path
) -> tuple[list[str], list[str]]:
    k_list = segment.metadata.get("keyframes_list")
    
    # Kiểm tra an toàn thay vì dùng chữ "or"
    if k_list is None:
        frame_ids = []
    else:
        frame_ids = list(k_list)
    frame_paths = [str(Path(keyframes_dir) / f"{fid}.jpg") for fid in frame_ids]
    return frame_paths, frame_ids


def _select_via_siglip(query: str, frame_paths: list[str], frame_ids: list[str]) -> Optional[int]:
    """Try SigLIP embedding similarity; returns None on any failure so the
    caller can fall back to the middle frame (never raises)."""
    try:
        from src.phase2_captioning.visual_embedder import best_matching_image_index

        idx = best_matching_image_index(query, frame_paths)
        return idx
    except Exception as exc:  # noqa: BLE001 - SigLIP is a "nice to have" fast path, never fatal
        logger.warning("SigLIP frame selection failed (%s); using middle frame.", exc)
        return None


def select_best_frame(
    query: str,
    segment: RetrievedSegment,
    keyframes_dir: Optional[str | Path] = None,
    keyframe_map_csv: Optional[str | Path] = None,
    engine: Optional[QwenVLEngine] = None,
    use_vlm: bool = True,
    fast_mode: Optional[str] = None,
) -> Optional[FrameChoice]:
    """Pick the single best-matching frame_id inside `segment` for `query`.

    When `use_vlm=True`, asks Qwen2.5-VL directly (most precise, one
    generation call). When `use_vlm=False`, uses `fast_mode` instead
    (defaults to `phase2.frame_select_fast_mode` in settings.yaml,
    normally "siglip"): a cheap SigLIP embedding pass, falling back to
    the plain middle-keyframe heuristic if SigLIP can't be used or
    `fast_mode="middle"` is requested explicitly. Returns None only if
    the segment has no keyframes at all.
    """
    cfg = load_config()
    keyframes_dir = keyframes_dir or cfg["paths"]["keyframes_dir"]
    keyframe_map_csv = str(keyframe_map_csv or cfg["paths"]["keyframe_map_csv"])
    ts_lookup = _load_frame_timestamp_lookup(keyframe_map_csv)

    frame_paths, frame_ids = _resolve_keyframe_paths(segment, keyframes_dir)
    if not frame_ids:
        return None

    chosen_index = len(frame_ids) // 2  # default: middle frame (last-resort fallback)

    if use_vlm and len(frame_ids) > 1:
        engine = engine or get_qwen_engine(cfg)
        prompt = FRAME_PICK_PROMPT_TEMPLATE.format(n=len(frame_ids), query=query)
        try:
            raw = engine.generate(
                text_prompt=prompt, image_paths=frame_paths, max_new_tokens=10, temperature=0.0
            )
            match = re.search(r"\d+", raw)
            if match:
                idx = int(match.group(0)) - 1
                if 0 <= idx < len(frame_ids):
                    chosen_index = idx
        except Exception as exc:  # noqa: BLE001 - never let frame selection kill the run
            logger.warning("Frame selection VLM call failed (%s); using middle frame.", exc)

    elif not use_vlm and len(frame_ids) > 1:
        mode = fast_mode or cfg["phase2"].get("frame_select_fast_mode", "siglip")
        if mode == "siglip":
            idx = _select_via_siglip(query, frame_paths, frame_ids)
            if idx is not None:
                chosen_index = idx
        # mode == "middle" (or SigLIP failed above): chosen_index already defaults to the middle frame.

    frame_id = frame_ids[chosen_index]
    timestamp = float(ts_lookup.get(frame_id, segment.metadata.get("start_time", 0.0)))
    return FrameChoice(frame_id=frame_id, timestamp_sec=timestamp, frame_index_in_segment=chosen_index)
