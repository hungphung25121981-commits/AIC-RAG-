"""OCR wrapper for on-screen text / table / UI-label extraction using Surya OCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

from PIL import Image
# --- Surya v1 API (pure PyTorch, no inference server / no Docker needed). ---
# DO NOT switch to `from surya.inference import SuryaInferenceManager` (Surya >=0.20,
# "Surya 2"): that API needs SuryaInferenceManager to spawn a vllm server in Docker
# (GPU) or a llama.cpp server (CPU/Mac) on first call, which Kaggle kernels cannot do
# (no dockerd / no privileged containers available there). See requirements.txt.
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)

# Module-level singletons so weights are loaded once per process, shared across
# every call (both the CLI's `--select-frame`/`query` paths and Phase 2 batch OCR).
_FOUNDATION_PREDICTOR = None
_RECOGNITION_PREDICTOR = None
_DETECTION_PREDICTOR = None

@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: list[tuple[float, float]]  # 4 (x, y) corner points

def _get_ocr_predictor() -> tuple[Any, Any]:
    """Lazily load and return (recognition_predictor, detection_predictor).

    Pure-torch Surya v1 predictors -- no inference server, no Docker required,
    safe to run inside a Kaggle notebook kernel.
    """
    global _FOUNDATION_PREDICTOR, _RECOGNITION_PREDICTOR, _DETECTION_PREDICTOR

    if _RECOGNITION_PREDICTOR is not None and _DETECTION_PREDICTOR is not None:
        return _RECOGNITION_PREDICTOR, _DETECTION_PREDICTOR

    logger.info("Loading Surya v1 OCR models (FoundationPredictor + RecognitionPredictor + DetectionPredictor)...")
    _FOUNDATION_PREDICTOR = FoundationPredictor()
    _RECOGNITION_PREDICTOR = RecognitionPredictor(_FOUNDATION_PREDICTOR)
    _DETECTION_PREDICTOR = DetectionPredictor()
    return _RECOGNITION_PREDICTOR, _DETECTION_PREDICTOR

def run_ocr_on_image(
    image_path: str | Path,
    min_confidence: Optional[float] = None,
    predictor: Optional[tuple[Any, Any]] = None,
) -> list[OCRBox]:
    """Run Surya OCR on a single keyframe image, filtered by confidence.

    `predictor`, if given, must be the `(recognition_predictor, detection_predictor)`
    tuple returned by `_get_ocr_predictor()` -- Surya v1's RecognitionPredictor needs a
    DetectionPredictor passed in explicitly (it no longer bundles one, and there's no
    "langs" hint anymore -- v1 detection/recognition are language-agnostic).
    """
    cfg = load_config()
    min_conf = min_confidence if min_confidence is not None else cfg["phase2"]["ocr_min_confidence"]

    # Dùng predictor truyền vào từ main.py (đã load 1 lần), nếu không có thì lazy-load singleton
    recognition_predictor, detection_predictor = predictor if predictor else _get_ocr_predictor()
    boxes: list[OCRBox] = []

    try:
        image = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return boxes

    # Surya v1: pure-torch call, no inference server involved.
    predictions = recognition_predictor([image], det_predictor=detection_predictor)

    if not predictions:
        return boxes

    # v1 schema: OCRResult.text_lines -> List[TextLine] (.text / .confidence / .polygon)
    result_items = getattr(predictions[0], "text_lines", None) or []

    if not result_items:
        return boxes

    for item in result_items:
        score = getattr(item, "confidence", 1.0) 
        text = getattr(item, "text", "").strip()
        polygon = getattr(item, "polygon", [])
        
        if score >= min_conf and text:
            boxes.append(OCRBox(
                text=text, 
                confidence=float(score), 
                bbox=polygon
            ))

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
    predictor: Optional[Any] = None,
) -> str:
    """Run OCR across all keyframes in a segment and return de-duplicated joined text."""
    cfg = load_config()
    dedupe_iou = dedupe_iou if dedupe_iou is not None else cfg["phase2"]["ocr_dedupe_iou"]

    seen_texts: list[str] = []
    kept_boxes: list[OCRBox] = []

    for path in keyframe_paths:
        try:
            boxes = run_ocr_on_image(path, min_confidence=min_confidence, predictor=predictor)
        except Exception as exc:
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

def build_caption_from_ocr(
    ocr_text: str,
    max_chars: Optional[int] = None,
    max_lines: Optional[int] = None,
    empty_fallback: Optional[str] = None,
) -> str:
    """Turn a segment's de-duplicated `" | "`-joined OCR text into the visual_caption string."""
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
    predictor: Optional[Any] = None,
) -> tuple[str, str]:
    """Convenience wrapper: run OCR once for a segment and return BOTH `(ocr_screen_text, visual_caption)`."""
    ocr_text = extract_segment_ocr_text(
        keyframe_paths, dedupe_iou=dedupe_iou, min_confidence=min_confidence, predictor=predictor
    )
    caption = build_caption_from_ocr(ocr_text)
    return ocr_text, caption