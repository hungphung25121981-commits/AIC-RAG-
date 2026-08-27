"""VLM-based re-ranking of fused hybrid_search results (the `--rerank` CLI flag).

RRF fusion (hybrid_search.py) is a cheap, scale-free first-pass ranker,
but it only knows about BM25/dense RANK POSITIONS -- it has no idea
whether a segment actually, semantically satisfies the query. This
module asks the same Qwen2.5-VL engine used everywhere else in the
pipeline to score each candidate 0-10 against the query (text-only,
against OCR + visual_caption -- no images, so --rerank stays reasonably
fast even over a `--top-n` of a few dozen candidates), then re-sorts by
that score.

Called explicitly (opt-in via --rerank) because it costs one extra VLM
call per `rerank_batch_size` candidates, on top of the free RRF fusion.
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
array of numbers, one per candidate, in the same order. Example: [8, 2, 5]"""


def _format_segments_for_rerank(segments: list[RetrievedSegment]) -> str:
    lines = []
    for i, seg in enumerate(segments):
        meta = seg.metadata
        lines.append(
            f"[{i}] video={meta.get('video_title', 'Unknown')} "
            f"t={meta.get('start_time', '?')}-{meta.get('end_time', '?')}s | "
            f"VISUAL: {meta.get('visual_caption', '')[:300]} | "
            f"OCR: {meta.get('ocr_screen_text', '')[:300]}"
        )
    return "\n".join(lines)


def _extract_json_array(raw_text: str) -> list:
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output: {raw_text!r}")
    return json.loads(match.group(0))


def rerank_segments(
    query: str,
    segments: list[RetrievedSegment],
    engine: Optional[QwenVLEngine] = None,
    batch_size: Optional[int] = None,
) -> list[RetrievedSegment]:
    """Re-score `segments` against `query` with the VLM (text-only) and re-sort.

    Runs in batches of `batch_size` candidates per VLM call so a long
    --top-n candidate list doesn't blow past the prompt's practical
    length. Returns a NEW list sorted by rerank score descending;
    `fused_score` is overwritten in place with the 0-10 rerank score
    (this becomes the "score" column downstream in the CSV export). On
    any parse failure for a batch, that batch's ORIGINAL fused-rank
    relative order is preserved (never crashes the whole rerank).
    """
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
        raw = engine.generate(text_prompt=prompt, image_paths=None, max_new_tokens=150, temperature=0.0)
        
        # DÙNG FLUSH=TRUE ĐỂ ÉP KAGGLE PHẢI IN RA NGAY LẬP TỨC
        print(f"\n=====================================")
        print(f"[DEBUG RERANK] Qwen trả lời:\n{raw}")
        print(f"=====================================\n", flush=True)
        
        # (Các dòng code bên dưới của bạn giữ nguyên)
        try:
            raw_scores = _extract_json_array(raw)
            scores = [float(s) for s in raw_scores]
            if len(scores) != len(batch):
                raise ValueError(f"expected {len(batch)} scores, got {len(scores)}")
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning(
                "Rerank parse failed for batch starting at %d (%s); keeping fused order for this batch.",
                start,
                exc,
            )
            scores = [float(len(batch) - i) for i in range(len(batch))]

        scored.extend(zip(batch, scores))

    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked: list[RetrievedSegment] = []
    for seg, score in scored:
        seg.fused_score = float(score)
        reranked.append(seg)

    logger.info("Reranked %d segment(s) for query %r.", len(reranked), query[:60])
    return reranked
