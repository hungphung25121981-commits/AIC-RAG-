"""PySceneDetect wrapper.

Detects hard scene cuts / content changes so that keyframes can be
grouped into semantically coherent segments (rather than fixed-size
windows alone). Falls back gracefully to fixed-window segmentation if
PySceneDetect finds zero scenes (e.g. a single continuous screen
recording with no hard cuts).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


def detect_scenes(
    video_path: str | Path,
    threshold: Optional[float] = None,
    min_scene_len_sec: Optional[float] = None,
) -> list[tuple[float, float]]:
    """Return a list of (start_sec, end_sec) scene boundaries.

    Uses ContentDetector (HSV histogram delta) which is robust for both
    camera footage and screen recordings / slide decks.
    """
    cfg = load_config()
    p1 = cfg["phase1"]
    threshold = threshold if threshold is not None else p1["scenedetect_threshold"]
    min_scene_len_sec = (
        min_scene_len_sec if min_scene_len_sec is not None else p1["min_scene_len_sec"]
    )

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    min_scene_len_frames = max(int(min_scene_len_sec * video.frame_rate), 1)
    scene_manager.add_detector(
        ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames)
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        duration = video.duration.get_seconds() if video.duration else 0.0
        logger.warning(
            "PySceneDetect found 0 scenes for %s; falling back to whole-video single scene.",
            Path(video_path).name,
        )
        return [(0.0, duration)]

    boundaries = [
        (start.get_seconds(), end.get_seconds()) for start, end in scene_list
    ]
    logger.info("Detected %d scenes in %s", len(boundaries), Path(video_path).name)
    return boundaries


def assign_scene_id(timestamp_sec: float, scene_boundaries: list[tuple[float, float]]) -> int:
    """Map a timestamp to its containing scene index (clamped to last scene)."""
    for idx, (start, end) in enumerate(scene_boundaries):
        if start <= timestamp_sec < end:
            return idx
    return max(len(scene_boundaries) - 1, 0)
