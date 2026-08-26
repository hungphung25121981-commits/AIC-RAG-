"""Multi-hop sub-query decomposition & RAG relevance evaluation.

Uses the SAME Qwen2.5-VL-3B-Instruct engine (text-only calls, no
images) to:
  1. Decompose a complex user question into up to `max_subqueries`
     atomic retrievable sub-questions.
  2. Judge whether the segments retrieved for each sub-query actually
     answer it, looping additional hybrid_search calls if not (bounded
     by max_subqueries to avoid infinite loops).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.search_engine.hybrid_search import RetrievedSegment, hybrid_search
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


DECOMPOSITION_PROMPT_TEMPLATE = """You are a query planning assistant for a video RAG system.
The video corpus contains ONLY: on-screen OCR text, visual scene descriptions,
and video metadata (titles/descriptions) -- there is NO audio transcript.

Break the following user question into at most {max_subqueries} atomic,
independently-searchable sub-questions. If the question is already simple
and single-hop, return exactly ONE sub-question (the original question,
possibly cleaned up).

USER QUESTION: "{question}"

Respond with ONLY a JSON array of strings, no other text. Example:
["sub-question 1", "sub-question 2"]"""


RELEVANCE_JUDGE_PROMPT_TEMPLATE = """You are a strict RAG relevance judge.

SUB-QUESTION: "{sub_question}"

RETRIEVED SEGMENTS:
{segments_block}

For each segment, does it contain information that helps answer the
sub-question? Respond with ONLY a JSON array of 0/1 integers, one per
segment, in the same order. Example: [1, 0, 1]"""


@dataclass
class SubQueryResult:
    sub_question: str
    retrieved_segments: list[RetrievedSegment] = field(default_factory=list)
    relevant_segment_ids: list[int] = field(default_factory=list)


def _extract_json_array(raw_text: str) -> list:
    """Robustly pull the first JSON array out of a model response."""
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output: {raw_text!r}")
    return json.loads(match.group(0))


def decompose_query(
    question: str,
    engine: Optional[QwenVLEngine] = None,
    max_subqueries: Optional[int] = None,
) -> list[str]:
    """Decompose a user question into sub-questions using Qwen2.5-VL (text-only)."""
    cfg = load_config()
    p4 = cfg["phase4"]
    if not p4.get("enable_multihop", True):
        return [question]

    max_subqueries = max_subqueries or p4["max_subqueries"]
    engine = engine or get_qwen_engine(cfg)

    prompt = DECOMPOSITION_PROMPT_TEMPLATE.format(question=question, max_subqueries=max_subqueries)
    raw = engine.generate(text_prompt=prompt, image_paths=None, max_new_tokens=200, temperature=0.1)

    try:
        sub_questions = _extract_json_array(raw)
        sub_questions = [str(q).strip() for q in sub_questions if str(q).strip()]
        if not sub_questions:
            raise ValueError("empty list")
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Sub-query decomposition parse failed (%s); falling back to original question.", exc)
        sub_questions = [question]

    return sub_questions[:max_subqueries]


def _format_segments_for_judge(segments: list[RetrievedSegment]) -> str:
    lines = []
    for i, seg in enumerate(segments):
        meta = seg.metadata
        lines.append(
            f"[{i}] video={meta.get('video_title', 'Unknown')} "
            f"t={meta.get('start_time', '?')}-{meta.get('end_time', '?')}s | "
            f"VISUAL: {meta.get('visual_caption', '')[:200]} | "
            f"OCR: {meta.get('ocr_screen_text', '')[:200]}"
        )
    return "\n".join(lines)


def judge_segment_relevance(
    sub_question: str,
    segments: list[RetrievedSegment],
    engine: Optional[QwenVLEngine] = None,
) -> list[int]:
    """Return the segment_ids Qwen judges as actually relevant to the sub-question."""
    if not segments:
        return []

    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)

    prompt = RELEVANCE_JUDGE_PROMPT_TEMPLATE.format(
        sub_question=sub_question, segments_block=_format_segments_for_judge(segments)
    )
    raw = engine.generate(text_prompt=prompt, image_paths=None, max_new_tokens=150, temperature=0.0)

    try:
        flags = _extract_json_array(raw)
        flags = [int(bool(f)) for f in flags]
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("Relevance judge parse failed (%s); keeping all retrieved segments.", exc)
        flags = [1] * len(segments)

    if len(flags) != len(segments):
        logger.warning(
            "Relevance judge returned %d flags for %d segments; padding/truncating.",
            len(flags),
            len(segments),
        )
        flags = (flags + [1] * len(segments))[: len(segments)]

    return [seg.segment_id for seg, flag in zip(segments, flags) if flag == 1]


def run_multihop_retrieval(
    question: str,
    faiss_index,
    bm25_corpus,
    metadata_df: pd.DataFrame,
    engine: Optional[QwenVLEngine] = None,
    search_fn: Optional[Callable] = None,
) -> list[SubQueryResult]:
    """Full multi-hop pipeline: decompose -> hybrid_search per sub-query -> judge relevance."""
    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)
    search_fn = search_fn or hybrid_search

    sub_questions = decompose_query(question, engine=engine)
    logger.info("Decomposed question into %d sub-question(s): %s", len(sub_questions), sub_questions)

    results: list[SubQueryResult] = []
    for sub_q in sub_questions:
        retrieved = search_fn(
            query=sub_q,
            faiss_index=faiss_index,
            bm25_corpus=bm25_corpus,
            metadata_df=metadata_df,
        )
        relevant_ids = judge_segment_relevance(sub_q, retrieved, engine=engine)
        results.append(
            SubQueryResult(sub_question=sub_q, retrieved_segments=retrieved, relevant_segment_ids=relevant_ids)
        )

    return results
