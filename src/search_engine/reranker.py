"""VLM-based re-ranking of fused hybrid_search results (the `--rerank` CLI flag).

RRF fusion (hybrid_search.py) is a cheap, scale-free first-pass ranker,
fusing dense text, visual embeddings, and sparse tokens. This
module asks the Qwen2.5-VL engine to score each candidate 0-10 against
the query, then re-sorts by that score.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.search_engine.hybrid_search import RetrievedSegment
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


RERANK_PROMPT_TEMPLATE = """You are a strict video-retrieval relevance scorer.

QUERY: "{query}"

CANDIDATE SEGMENTS:
{segments_block}

For EACH candidate, score how well it matches the QUERY on a 0-10 scale
(10 = perfect direct match, 0 = irrelevant). Respond with ONLY a JSON
array of numbers, one per candidate, in the exact same order. Example: [8, 2, 5]"""


def _format_segments_for_rerank(segments: list[RetrievedSegment]) -> str:
    lines = []
    for i, seg in enumerate(segments):
        meta = seg.metadata
        visual_text = meta.get("visual_caption", "")
        if "No on-screen text detected" in visual_text:
            visual_text = "(visual content indexed via video embeddings)"
        ocr_text = meta.get("ocr_screen_text", "")

        lines.append(
            f"[{i}] video={meta.get('video_title', 'Unknown')} "
            f"t={meta.get('start_time', '?')}-{meta.get('end_time', '?')}s | "
            f"VISUAL: {visual_text[:300]} | "
            f"OCR: {ocr_text[:300]}"
        )
    return "\n".join(lines)


def _extract_json_array(raw_text: str) -> list:
    """Robustly pull the first JSON array out of a model response."""
    match = re.search(r"\[.*?\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output: {raw_text!r}")
    return json.loads(match.group(0))


def _parse_rerank_scores(raw: str, expected_count: int) -> list[float]:
    """Parse scores from VLM output with robust fallbacks, preventing index confusion."""
    scores: list[float] = []

    # 1. Clean JSON array attempt
    try:
        parsed = _extract_json_array(raw)
        if isinstance(parsed, list):
            scores = [float(s) for s in parsed if isinstance(s, (int, float, str))]
    except Exception:
        pass

    # 2. Bracket array numbers fallback: e.g. "Scores: [8, 2, 5]"
    if not scores:
        array_match = re.search(r"\[([\s\d.,-]+)\]", raw)
        if array_match:
            try:
                num_strings = re.findall(r"[-+]?\d*\.?\d+", array_match.group(1))
                scores = [float(s) for s in num_strings]
            except Exception:
                scores = []

    # 3. Candidate-indexed matching: "Candidate [0]: 8, Candidate [1]: 2"
    if not scores or len(scores) != expected_count:
        indexed_scores = []
        for i in range(expected_count):
            pattern = rf"(?:\[{i}\]|candidate\s*{i}|item\s*{i})[^\d\n]*?(\d+(?:\.\d+)?)"
            m = re.search(pattern, raw, re.IGNORECASE)
            if m:
                indexed_scores.append(float(m.group(1)))
        if len(indexed_scores) == expected_count:
            scores = indexed_scores

    # 4. If all else fails, use raw scores or default order
    if not scores:
        try:
            numbers = re.findall(r"[-+]?\d*\.?\d+", raw)
            scores = [float(s) for s in numbers]
        except Exception:
            scores = []

    # Pad or truncate to match expected count
    if len(scores) > expected_count:
        scores = scores[:expected_count]
    elif len(scores) < expected_count:
        scores.extend([0.0] * (expected_count - len(scores)))

    # Clamp scores to [0.0, 10.0]
    return [max(0.0, min(10.0, float(s))) for s in scores]


def rerank_segments(
    query: str,
    segments: list[RetrievedSegment],
    engine: Optional[QwenVLEngine] = None,
    batch_size: Optional[int] = None,
) -> list[RetrievedSegment]:
    """Re-score `segments` against `query` with the VLM and re-sort."""
    if not segments:
        return segments

    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)
    batch_size = batch_size or cfg["phase4"].get("rerank_batch_size", 15)

    scored: list[tuple[RetrievedSegment, float]] = []
    for start in range(0, len(segments), batch_size):
        batch = segments[start : start + batch_size]
        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query, segments_block=_format_segments_for_rerank(batch)
        )
        try:
            raw = engine.generate(text_prompt=prompt, image_paths=None, max_new_tokens=150, temperature=0.0)
            print(f"\n[DEBUG RERANK] Qwen Output:\n{raw}", flush=True)
            scores = _parse_rerank_scores(raw, len(batch))
        except Exception as exc:
            logger.warning("Rerank batch scoring failed (%s); keeping original fused rank order.", exc)
            scores = [float(len(batch) - i) for i in range(len(batch))]

        scored.extend(zip(batch, scores))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked: list[RetrievedSegment] = []
    for seg, score in scored:
        seg.fused_score = float(score)
        reranked.append(seg)

    logger.info("Reranked %d segment(s) for query %r.", len(reranked), query[:60])
    return reranked
