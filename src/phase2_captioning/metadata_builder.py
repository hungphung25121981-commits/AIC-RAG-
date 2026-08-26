"""Builds metadata.parquet from keyframe_map.csv + video_info.json + OCR.

ARCHITECTURE NOTE: Phase 2 is OCR-only now -- there is no VLM/LLM call
in this file anymore. `visual_caption` is built directly from the same
de-duplicated OCR text as `ocr_screen_text` (see
`ocr_engine.py::build_segment_caption`), instead of a Qwen2.5-VL
captioning pass per segment. This is by far the biggest Phase 2 speedup:
no GPU model needs to be loaded to run `caption`/`pipeline` at all. The
VLM (`src/search_engine/vlm_engine.py`, now Qwen2.5-VL-3B) is
loaded only later, in Phase 4, for the `--s/--q/--qa` search modes and
`--rerank`.

Handles:
  - dynamic / missing video_info.json keys via .get()
  - fallback jump_link construction (watch_url based, else raw timestamp)
  - GLOBAL contiguous segment_id required by FAISS (0..N-1 across the
    whole corpus, not just per-video)
  - composite full_text_for_embedding string used by Phase 3 embedder
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.phase2_captioning.ocr_engine import build_segment_caption
from src.utils_common import (
    ensure_dir,
    free_gpu_memory,
    get_logger,
    load_config,
    maybe_free_memory,
    safe_load_video_info,
)

logger = get_logger(__name__)


def _build_jump_link(video_info: dict[str, Any], video_id: str, start_time: float) -> str:
    """Dynamically construct a clickable timestamp link.

    Priority:
      1. watch_url + "&t={start}s"  (YouTube-style)
      2. watch_url + "#t={start}s"  (generic web video fallback, if watch_url
         already has query params it still appends safely below)
      3. "{video_id}#t={start}s"    (local file / no URL available)
    """
    cfg = load_config()
    fallback_fmt = cfg["phase4"]["timestamp_link_fallback_format"]

    watch_url = video_info.get("watch_url") or video_info.get("url")
    start_int = int(start_time)

    if watch_url:
        separator = "&" if "?" in watch_url else "?"
        # Use &t= only if there's already a query string (YouTube convention);
        # otherwise use #t= to avoid inventing an invalid query on a bare URL.
        if "?" in watch_url:
            return f"{watch_url}{separator}t={start_int}s"
        return f"{watch_url}#t={start_int}s"

    return fallback_fmt.format(video_id=video_id, start_time=start_int)


def _make_full_text_for_embedding(video_title: str, visual_caption: str, ocr_text: str) -> str:
    return f"TITLE: {video_title} | VISUAL: {visual_caption} | OCR: {ocr_text}"


def build_metadata_for_video(
    keyframe_map_df: pd.DataFrame,
    video_id: str,
    video_info_path: str | Path,
    keyframes_dir: str | Path,
    global_segment_id_offset: int = 0,
) -> pd.DataFrame:
    """Build metadata rows for one video's segments.

    `global_segment_id_offset` lets the caller keep segment_id contiguous
    across the ENTIRE corpus (required for FAISS row alignment), since
    keyframe_map.csv only stores per-video-local segment ids.
    """
    cfg = load_config()
    video_info = safe_load_video_info(video_info_path)
    video_title = video_info.get("title", "Unknown Title")

    video_df = keyframe_map_df[keyframe_map_df["video_id"] == video_id]
    if video_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    local_segment_ids = sorted(video_df["segment_id"].unique())
    for step_index, local_seg_id in enumerate(local_segment_ids, start=1):
        seg_frames = video_df[video_df["segment_id"] == local_seg_id].sort_values("timestamp_sec")
        keyframe_ids = seg_frames["frame_id"].tolist()
        keyframe_paths = [str(Path(keyframes_dir) / f"{fid}.jpg") for fid in keyframe_ids]

        start_time = float(seg_frames["timestamp_sec"].min())
        end_time = float(seg_frames["timestamp_sec"].max())

        try:
            ocr_text, visual_caption = build_segment_caption(keyframe_paths)
        except Exception as exc:  # noqa: BLE001 - keep pipeline alive on a single bad segment
            logger.error("OCR caption building failed for %s seg %d: %s", video_id, local_seg_id, exc)
            ocr_text, visual_caption = "", ""

        global_segment_id = global_segment_id_offset + step_index - 1
        jump_link = _build_jump_link(video_info, video_id, start_time)
        full_text = _make_full_text_for_embedding(video_title, visual_caption, ocr_text)

        rows.append(
            {
                "segment_id": global_segment_id,
                "video_id": video_id,
                "video_title": video_title,
                "start_time": start_time,
                "end_time": end_time,
                "jump_link": jump_link,
                "keyframes_list": keyframe_ids,
                "ocr_screen_text": ocr_text,
                "visual_caption": visual_caption,
                "full_text_for_embedding": full_text,
            }
        )

        maybe_free_memory(step_index, cfg)

    return pd.DataFrame(rows)


def build_corpus_metadata(
    keyframe_map_csv: Optional[str | Path] = None,
    video_info_dir: Optional[str | Path] = None,
    keyframes_dir: Optional[str | Path] = None,
    output_parquet: Optional[str | Path] = None,
) -> pd.DataFrame:
    """End-to-end Phase 2 driver: keyframe_map.csv -> metadata.parquet.

    Assumes one <video_id>.json file per video inside video_info_dir.
    """
    cfg = load_config()
    keyframe_map_csv = keyframe_map_csv or cfg["paths"]["keyframe_map_csv"]
    video_info_dir = Path(video_info_dir or cfg["paths"]["video_info_dir"])
    keyframes_dir = keyframes_dir or cfg["paths"]["keyframes_dir"]
    output_parquet = output_parquet or cfg["paths"]["metadata_parquet"]

    keyframe_map_df = pd.read_csv(keyframe_map_csv)
    video_ids = keyframe_map_df["video_id"].unique().tolist()
    logger.info("Building metadata.parquet for %d videos (OCR-only captioning, no VLM)...", len(video_ids))

    all_dfs = []
    running_offset = 0
    for video_id in video_ids:
        video_info_path = video_info_dir / f"{video_id}.json"
        video_df = build_metadata_for_video(
            keyframe_map_df=keyframe_map_df,
            video_id=video_id,
            video_info_path=video_info_path,
            keyframes_dir=keyframes_dir,
            global_segment_id_offset=running_offset,
        )
        if not video_df.empty:
            all_dfs.append(video_df)
            running_offset += len(video_df)

        gc.collect()
        free_gpu_memory()

    corpus_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    # Final safety check: segment_id MUST be contiguous 0..N-1 for FAISS row alignment.
    assert list(corpus_df["segment_id"]) == list(range(len(corpus_df))), (
        "segment_id is not contiguous 0..N-1 -- FAISS row alignment will break downstream."
    )

    out_path = ensure_dir(Path(str(output_parquet)).parent) / Path(str(output_parquet)).name
    corpus_df.to_parquet(out_path, index=False)
    logger.info("Wrote metadata.parquet with %d segments to %s", len(corpus_df), out_path)
    return corpus_df
