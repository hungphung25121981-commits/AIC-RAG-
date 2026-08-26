"""OCR wrapper for on-screen text / table / UI-label extraction.

Since the pipeline has no Whisper transcript, OCR text is a primary
signal (news banners, code, terminal output, table values, slide text).
This module extracts and de-duplicates OCR text across the keyframes
belonging to a single segment.

ARCHITECTURE NOTE -- this module is now ALSO the Phase 2 caption
builder. The old design generated `visual_caption` with a VLM
(Qwen2.5-VL) forward pass per segment; that call has been removed
from Phase 2 entirely (it was by far the slowest step in indexing).
Instead, `build_caption_from_ocr()` below turns the same de-duplicated
OCR text this module already extracts into the `visual_caption` string
written to metadata.parquet -- no model call, no GPU, just formatting.
The VLM is still used elsewhere in the pipeline (Phase 4 search/QA/
rerank -- see search_engine/vlm_engine.py), just never here.

Engine choice: RapidOCR (ONNXRuntime-based) instead of PaddleOCR.

  - PaddleOCR requires `paddlepaddle-gpu`, a large compiled package whose
    prebuilt wheels are pinned to a specific numpy ABI. On Kaggle's stock
    image (numpy 2.0.2 as of this writing) that mismatch throws either an
    install failure or a `numpy.dtype size changed, may indicate binary
    incompatibility` error at import time -- the exact symptom that
    prompted this rewrite.
  - RapidOCR ships ONNX models run through `onnxruntime`, which has no
    numpy-C-API coupling, so it isn't sensitive to which numpy ABI the
    host image shipped. It's also a much lighter install (no CUDA-matched
    paddle wheel to get right) and, per the environment check, is the
    engine that actually imports and initializes successfully here.

`ocr_engine` in config/settings.yaml selects the backend ("rapidocr" by
default). PaddleOCR support is kept as an optional path for environments
where it's known to work (e.g. a local machine with numpy<2 pinned), but
RapidOCR is what you should run on Kaggle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.utils_common import get_logger, load_config

logger = get_logger(__name__)

_OCR_SINGLETON = None  # lazy-loaded OCR engine instance (expensive to init)
_OCR_BACKEND: Optional[str] = None  # which backend the singleton above is


@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: list[tuple[float, float]]  # 4 (x, y) corner points


def _get_rapidocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    logger.info("Initializing RapidOCR (onnxruntime backend)...")
    return RapidOCR()


def _get_paddleocr_engine(p2: dict):
    from paddleocr import PaddleOCR

    logger.info("Initializing PaddleOCR (lang=%s)...", p2["ocr_lang"])

    # PaddleOCR's constructor kwargs have changed across major versions
    # (e.g. `show_log` was removed entirely, `use_angle_cls` renamed to
    # `use_textline_orientation` in PaddleOCR 3.x) -- and it validates
    # kwargs internally with an explicit "Unknown argument: X" error
    # rather than a plain Python TypeError, so a static signature check
    # isn't reliable either. Instead, try progressively smaller kwarg
    # sets and use whichever one the installed version actually accepts.
    attempts = [
        {
            "use_angle_cls": p2["ocr_use_angle_cls"],
            "lang": p2["ocr_lang"],
            "show_log": False,
        },  # PaddleOCR <= 2.x
        {
            "use_textline_orientation": p2["ocr_use_angle_cls"],
            "lang": p2["ocr_lang"],
        },  # PaddleOCR 3.x (renamed arg, show_log removed)
        {"lang": p2["ocr_lang"]},  # minimal fallback
        {},  # last resort: pure defaults
    ]

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            engine = PaddleOCR(**kwargs)
            logger.info("PaddleOCR initialized with kwargs=%s", kwargs)
            return engine
        except Exception as exc:  # noqa: BLE001 - PaddleOCR's kwarg-validation exception
                                    # type isn't guaranteed stable across versions, so we
                                    # catch broadly here and fall through to the next
                                    # candidate kwarg set rather than guessing the type.
            last_error = exc
            logger.debug("PaddleOCR init failed with kwargs=%s (%s); trying next fallback.", kwargs, exc)
    raise RuntimeError(f"Could not initialize PaddleOCR with any known kwarg set: {last_error}")


def _get_ocr_engine():
    global _OCR_SINGLETON, _OCR_BACKEND

    cfg = load_config()
    p2 = cfg["phase2"]
    backend = p2.get("ocr_engine", "rapidocr")

    if _OCR_SINGLETON is not None and _OCR_BACKEND == backend:
        return _OCR_SINGLETON, backend

    if backend == "rapidocr":
        _OCR_SINGLETON = _get_rapidocr_engine()
    elif backend == "paddleocr":
        _OCR_SINGLETON = _get_paddleocr_engine(p2)
    else:
        raise ValueError(f"Unknown ocr_engine '{backend}' in config (expected 'rapidocr' or 'paddleocr')")

    _OCR_BACKEND = backend
    return _OCR_SINGLETON, backend


def run_ocr_on_image(image_path: str | Path, min_confidence: Optional[float] = None) -> list[OCRBox]:
    """Run OCR on a single keyframe image, filtered by confidence."""
    cfg = load_config()
    min_conf = min_confidence if min_confidence is not None else cfg["phase2"]["ocr_min_confidence"]

    engine, backend = _get_ocr_engine()
    boxes: list[OCRBox] = []

    if backend == "rapidocr":
        # RapidOCR returns (result, elapse) where result is either None (no text
        # found) or a list of [bbox, text, score]; bbox is 4 (x, y) corner points,
        # matching PaddleOCR's box format, so OCRBox / downstream IoU dedupe code
        # is unchanged.
        result, _elapse = engine(str(image_path))
        if not result:
            return boxes
        for bbox, text, score in result:
            if score >= min_conf and text.strip():
                boxes.append(OCRBox(text=text.strip(), confidence=float(score), bbox=bbox))
        return boxes

    # backend == "paddleocr"
    result = engine.ocr(str(image_path), cls=True)
    if not result or result[0] is None:
        return boxes
    for line in result[0]:
        bbox, (text, confidence) = line
        if confidence >= min_conf and text.strip():
            boxes.append(OCRBox(text=text.strip(), confidence=float(confidence), bbox=bbox))
    return boxes


def _iou(box_a: list[tuple[float, float]], box_b: list[tuple[float, float]]) -> float:
    """Approximate IoU between two quadrilateral boxes via their axis-aligned bounds."""

    def bounds(box):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return min(xs), min(ys), max(xs), max(ys)

    ax1, ay1, ax2, ay2 = bounds(box_a)
    bx1, by1, bx2, by2 = bounds(box_b)

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / float(area_a + area_b - inter_area)


def extract_segment_ocr_text(
    keyframe_paths: list[str | Path],
    dedupe_iou: Optional[float] = None,
    min_confidence: Optional[float] = None,
) -> str:
    """Run OCR across all keyframes in a segment and return de-duplicated joined text.

    Text boxes that spatially overlap (IoU >= dedupe_iou) AND share very
    similar text across consecutive frames are merged, since a static
    slide/table usually repeats across several frames in one segment.
    """
    cfg = load_config()
    dedupe_iou = dedupe_iou if dedupe_iou is not None else cfg["phase2"]["ocr_dedupe_iou"]

    seen_texts: list[str] = []
    kept_boxes: list[OCRBox] = []

    for path in keyframe_paths:
        try:
            boxes = run_ocr_on_image(path, min_confidence=min_confidence)
        except Exception as exc:  # noqa: BLE001 - OCR engine errors shouldn't kill the pipeline
            logger.warning("OCR failed on %s: %s", path, exc)
            continue

        for box in boxes:
            is_duplicate = False
            for kept in kept_boxes:
                if box.text.lower() == kept.text.lower() and _iou(box.bbox, kept.bbox) >= dedupe_iou:
                    is_duplicate = True
                    break
            if not is_duplicate and box.text.lower() not in [t.lower() for t in seen_texts]:
                kept_boxes.append(box)
                seen_texts.append(box.text)

    return " | ".join(seen_texts)


# ----------------------------------------------------------------------
# Phase 2 caption building -- OCR-only replacement for the old VLM captioner
# ----------------------------------------------------------------------
def build_caption_from_ocr(
    ocr_text: str,
    max_chars: Optional[int] = None,
    max_lines: Optional[int] = None,
    empty_fallback: Optional[str] = None,
) -> str:
    """Turn a segment's de-duplicated `" | "`-joined OCR text into the
    `visual_caption` string, with NO model call.

    This is the direct replacement for the old `vlm_captioner.caption_segment()`
    call: instead of asking a VLM to describe the keyframes in prose, the
    segment's own on-screen text (already extracted by
    `extract_segment_ocr_text`) IS the caption -- reordered into a light,
    readable sentence rather than the raw ` | `-delimited dedupe format used
    for the separate `ocr_screen_text` column.

    Deliberately simple string formatting (lowercase noise trimming, line
    cap, char cap) -- no NLP/model dependency, so Phase 2 captioning is now
    pure CPU work and doesn't need the VLM loaded at all.
    """
    cfg = load_config()
    p2 = cfg["phase2"]
    max_chars = max_chars if max_chars is not None else p2.get("caption_max_chars", 600)
    max_lines = max_lines if max_lines is not None else p2.get("caption_max_lines", 12)
    empty_fallback = (
        empty_fallback if empty_fallback is not None else p2.get(
            "caption_empty_fallback", "No on-screen text detected in this segment."
        )
    )

    if not ocr_text or not ocr_text.strip():
        return empty_fallback

    lines = [line.strip() for line in ocr_text.split(" | ") if line.strip()]
    lines = lines[:max_lines]

    caption = "On-screen text detected: " + "; ".join(lines) + "."
    if len(caption) > max_chars:
        caption = caption[: max_chars - 1].rstrip() + "…"
    return caption


def build_segment_caption(
    keyframe_paths: list[str | Path],
    dedupe_iou: Optional[float] = None,
    min_confidence: Optional[float] = None,
) -> tuple[str, str]:
    """Convenience wrapper: run OCR once for a segment and return BOTH
    `(ocr_screen_text, visual_caption)` derived from it, so callers (e.g.
    `metadata_builder.py`) don't need to run OCR twice for the two columns.
    """
    ocr_text = extract_segment_ocr_text(
        keyframe_paths, dedupe_iou=dedupe_iou, min_confidence=min_confidence
    )
    caption = build_caption_from_ocr(ocr_text)
    return ocr_text, caption
