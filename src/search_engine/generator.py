"""Answer synthesis: Qwen2.5-VL-3B produces the final grounded answer,
citing exact timestamp jump_links pulled from metadata.parquet.

This is the Phase 4 "Timestamp Link Synthesizer" -- it never invents
timestamps; every [Source N] citation is mapped back to a real
jump_link from the retrieved segments so the UI/notebook can render
clickable links.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.search_engine.hybrid_search import RetrievedSegment
from src.search_engine.multihop_evaluator import SubQueryResult, run_multihop_retrieval
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


ANSWER_SYSTEM_PROMPT = """You are a video-RAG answer synthesizer. Answer the
user's question using ONLY the evidence in the CONTEXT SEGMENTS below.
Each segment came from OCR text and visual scene descriptions of a video
(there is no audio transcript, so do not claim anything was "said").

Rules:
- Cite evidence inline using [Source N] tags matching the segment numbers below.
- If the context does not contain enough information, say so explicitly --
  do not fabricate facts, table values, or timestamps.
- Be concise and directly answer the question first, then support with detail.

CONTEXT SEGMENTS:
{context_block}

USER QUESTION: {question}

ANSWER:"""


@dataclass
class Citation:
    source_index: int
    video_id: str
    video_title: str
    start_time: float
    end_time: float
    jump_link: str


@dataclass
class RAGAnswer:
    question: str
    answer_text: str
    citations: list[Citation] = field(default_factory=list)
    sub_query_results: list[SubQueryResult] = field(default_factory=list)


def _dedupe_segments(sub_query_results: list[SubQueryResult]) -> list[RetrievedSegment]:
    """Collect the union of judged-relevant segments across all sub-queries, de-duplicated,
    sorted by fused_score descending."""
    seen: dict[int, RetrievedSegment] = {}
    for result in sub_query_results:
        relevant_ids = set(result.relevant_segment_ids)
        for seg in result.retrieved_segments:
            if seg.segment_id in relevant_ids and seg.segment_id not in seen:
                seen[seg.segment_id] = seg
    return sorted(seen.values(), key=lambda s: s.fused_score, reverse=True)


def _build_context_block(segments: list[RetrievedSegment]) -> tuple[str, list[Citation]]:
    lines = []
    citations = []
    for i, seg in enumerate(segments, start=1):
        meta = seg.metadata
        lines.append(
            f"[Source {i}] video=\"{meta.get('video_title', 'Unknown')}\" "
            f"time={meta.get('start_time', '?')}s-{meta.get('end_time', '?')}s\n"
            f"  VISUAL: {meta.get('visual_caption', '')}\n"
            f"  OCR: {meta.get('ocr_screen_text', '')}"
        )
        citations.append(
            Citation(
                source_index=i,
                video_id=meta.get("video_id", "unknown"),
                video_title=meta.get("video_title", "Unknown Title"),
                start_time=meta.get("start_time", 0.0),
                end_time=meta.get("end_time", 0.0),
                jump_link=meta.get("jump_link", ""),
            )
        )
    return "\n\n".join(lines), citations


def synthesize_answer(
    question: str,
    segments: list[RetrievedSegment],
    engine: Optional[QwenVLEngine] = None,
) -> RAGAnswer:
    """Generate the final grounded answer text for a fixed list of retrieved segments."""
    cfg = load_config()
    p4 = cfg["phase4"]
    engine = engine or get_qwen_engine(cfg)

    if not segments:
        return RAGAnswer(
            question=question,
            answer_text=(
                "I could not find any sufficiently relevant on-screen text or visual "
                "content in the indexed videos to answer this question."
            ),
            citations=[],
        )

    context_block, citations = _build_context_block(segments)
    prompt = ANSWER_SYSTEM_PROMPT.format(context_block=context_block, question=question)

    answer_text = engine.generate(
        text_prompt=prompt,
        image_paths=None,
        max_new_tokens=p4["generator_max_new_tokens"],
        temperature=p4["generator_temperature"],
    )

    return RAGAnswer(question=question, answer_text=answer_text, citations=citations)


def answer_question(
    question: str,
    faiss_index,
    bm25_corpus,
    metadata_df: pd.DataFrame,
    engine: Optional[QwenVLEngine] = None,
) -> RAGAnswer:
    """Full Phase 4 pipeline: multi-hop retrieval -> dedupe -> synthesize grounded answer."""
    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)

    sub_query_results = run_multihop_retrieval(
        question=question,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        engine=engine,
    )

    fused_segments = _dedupe_segments(sub_query_results)
    top_segments = fused_segments[: cfg["phase4"]["final_top_k"]]

    answer = synthesize_answer(question, top_segments, engine=engine)
    answer.sub_query_results = sub_query_results

    logger.info(
        "Answered %r using %d segments across %d sub-query(ies).",
        question[:60],
        len(top_segments),
        len(sub_query_results),
    )
    return answer


def format_answer_with_links(answer: RAGAnswer) -> str:
    """Render the answer text plus a clickable-link reference list (Markdown-friendly)."""
    lines = [answer.answer_text, "", "**Sources:**"]
    for c in answer.citations:
        lines.append(
            f"- [Source {c.source_index}] {c.video_title} "
            f"({c.start_time:.1f}s-{c.end_time:.1f}s): {c.jump_link}"
        )
    return "\n".join(lines)
