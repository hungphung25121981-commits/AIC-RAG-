"""SSIM-based near-duplicate frame filtering.

Given a directory of raw candidate frames (or a live cv2.VideoCapture
stream), keep only frames that are structurally different enough from
the last *kept* frame, per `phase1.ssim_threshold` in settings.yaml.
This trims redundant frames (e.g. a presenter talking with a mostly
static slide) before the more expensive scene-detection / VLM stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


@dataclass
class FrameCandidate:
    frame_index: int
    timestamp_sec: float
    image: np.ndarray


def _to_gray_resized(image: np.ndarray, max_side: int = 320) -> np.ndarray:
    """Downscale + grayscale for cheap SSIM comparison."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale != 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Structural similarity between two frames in [-1, 1] (1.0 == identical)."""
    gray_a = _to_gray_resized(frame_a)
    gray_b = _to_gray_resized(frame_b)
    if gray_a.shape != gray_b.shape:
        gray_b = cv2.resize(gray_b, (gray_a.shape[1], gray_a.shape[0]))
    score, _ = ssim(gray_a, gray_b, full=True)
    return float(score)


def iter_video_frames(
    video_path: str | Path, sample_fps: float = 2.0
) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_sec, frame) sampled at `sample_fps`."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(int(round(native_fps / sample_fps)), 1)

    frame_index = 0
    read_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if read_index % step == 0:
                timestamp_sec = read_index / native_fps
                yield frame_index, timestamp_sec, frame
                frame_index += 1
            read_index += 1
    finally:
        cap.release()


def filter_near_duplicate_frames(
    video_path: str | Path,
    ssim_threshold: Optional[float] = None,
    sample_fps: float = 2.0,
) -> list[FrameCandidate]:
    """Return the subset of sampled frames that pass SSIM deduplication.

    A frame is KEPT if its SSIM similarity to the last kept frame is
    BELOW `ssim_threshold` (i.e. it's different enough to be informative).
    The very first frame is always kept as the anchor.
    """
    cfg = load_config()
    threshold = ssim_threshold if ssim_threshold is not None else cfg["phase1"]["ssim_threshold"]

    kept: list[FrameCandidate] = []
    last_kept_gray_frame: Optional[np.ndarray] = None

    for frame_index, timestamp_sec, frame in iter_video_frames(video_path, sample_fps=sample_fps):
        if last_kept_gray_frame is None:
            kept.append(FrameCandidate(frame_index, timestamp_sec, frame))
            last_kept_gray_frame = frame
            continue

        similarity = compute_ssim(last_kept_gray_frame, frame)
        if similarity < threshold:
            kept.append(FrameCandidate(frame_index, timestamp_sec, frame))
            last_kept_gray_frame = frame

    logger.info(
        "SSIM filter: kept %d/%d sampled frames (threshold=%.2f) for %s",
        len(kept),
        frame_index + 1 if 'frame_index' in dir() else len(kept),
        threshold,
        Path(video_path).name,
    )
    return kept


def save_frame_candidates(
    frames: list[FrameCandidate],
    video_id: str,
    output_dir: str | Path,
    jpeg_quality: Optional[int] = None,
    resize_max_side: Optional[int] = None,
) -> list[dict]:
    """Persist kept frames as JPEGs and return their metadata records."""
    cfg = load_config()
    quality = jpeg_quality if jpeg_quality is not None else cfg["phase1"]["jpeg_quality"]
    max_side = resize_max_side if resize_max_side is not None else cfg["phase1"]["frame_resize_max_side"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for fc in frames:
        image = fc.image
        h, w = image.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        frame_id = f"{video_id}_f{fc.frame_index:06d}"
        out_path = out_dir / f"{frame_id}.jpg"
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])

        records.append(
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "timestamp_sec": round(fc.timestamp_sec, 3),
                "path": str(out_path),
            }
        )
    logger.info("Saved %d keyframes for video_id=%s to %s", len(records), video_id, out_dir)
    return records
