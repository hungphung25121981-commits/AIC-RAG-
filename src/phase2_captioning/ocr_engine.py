"""OCR wrapper for on-screen text / table / UI-label extraction using Surya OCR.

This module replaces RapidOCR/PaddleOCR with Surya OCR. Surya provides superior
layout detection and reading order parsing, which is critical for complex RAG
documents.

The engine uses two parallel models (Detection and Recognition) initialized lazily
via `_get_ocr_engine()` to conserve VRAM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

from src.utils_common import get_logger, load_config

logger = get_logger(__name__)

# Lazy-loaded Surya engine instances
_SURYA_DET_MODEL = None
_SURYA_DET_PROCESSOR = None
_SURYA_REC_MODEL = None
_SURYA_REC_PROCESSOR = None


@dataclass
class OCRBox:
    text: str
    confidence: float
    bbox: list[tuple[float, float]]  # 4 (x, y) corner points


def _get_ocr_engine():
    """Lazily load and return the Surya Detection and Recognition pipelines."""
    global _SURYA_DET_MODEL, _SURYA_DET_PROCESSOR, _SURYA_REC_MODEL, _SURYA_REC_PROCESSOR

    if _SURYA_DET_MODEL is not None:
        return _SURYA_DET_MODEL, _SURYA_DET_PROCESSOR, _SURYA_REC_MODEL, _SURYA_REC_PROCESSOR

    logger.info("Initializing Surya OCR Detection Model...")
    _SURYA_DET_PROCESSOR = load_det_processor()
    _SURYA_DET_MODEL = load_det_model()

    logger.info("Initializing Surya OCR Recognition Model...")
    _SURYA_REC_PROCESSOR = load_rec_processor()
    _SURYA_REC_MODEL = load_rec_model()

    return _SURYA_DET_MODEL, _SURYA_DET_PROCESSOR, _SURYA_REC_MODEL, _SURYA_REC_PROCESSOR


def run_ocr_on_image(image_path: str | Path, min_confidence: Optional[float] = None) -> list[OCRBox]:
    """Run Surya OCR on a single keyframe image, filtered by confidence."""
    cfg = load_config()
    min_conf = min_confidence if min_confidence is not None else cfg["phase2"]["ocr_min_confidence"]
    langs = [cfg["phase2"].get("ocr_lang", "vi")]

    det_model, det_processor, rec_model, rec_processor = _get_ocr_engine()
    boxes: list[OCRBox] = []

    try:
        image = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        return boxes

    # Surya batch process (1 image per batch here)
    predictions = run_ocr(
        [image], 
        [langs], 
        det_model, 
        det_processor, 
        rec_model, 
        rec_processor
    )
    
    if not predictions or not predictions[0].text_lines:
        return boxes

    # Extract bounding box and text
    for line in predictions[0].text_lines:
        # Surya returns confidence per line if available, otherwise default to 1.0 for valid text
        score = getattr(line, "confidence", 1.0) 
        if score >= min_conf and line.text.strip():
            boxes.append(OCRBox(
                text=line.text.strip(), 
                confidence=float(score), 
                bbox=line.polygon
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
) -> str:
    """Run OCR across all keyframes in a segment and return de-duplicated joined text."""
    cfg = load_config()
    dedupe_iou = dedupe_iou if dedupe_iou is not None else cfg["phase2"]["ocr_dedupe_iou"]

    seen_texts: list[str] = []
    kept_boxes: list[OCRBox] = []

    for path in keyframe_paths:
        try:
            boxes = run_ocr_on_image(path, min_confidence=min_confidence)
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
) -> tuple[str, str]:
    """Convenience wrapper: run OCR once for a segment and return BOTH `(ocr_screen_text, visual_caption)`."""
    ocr_text = extract_segment_ocr_text(
        keyframe_paths, dedupe_iou=dedupe_iou, min_confidence=min_confidence
    )
    caption = build_caption_from_ocr(ocr_text)
    return ocr_text, caption