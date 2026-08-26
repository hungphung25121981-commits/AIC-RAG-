"""Direct visual question-answering from concrete frame image(s).

`generator.py`'s `synthesize_answer` answers from TEXT captions only
(image_paths=None) -- good for "what error message appeared" style
questions where the OCR/caption text already has the answer. But
counting-style questions ("bao nhieu nguoi dang an?" -> "2") need the
model to actually LOOK at the frame, so this module sends the real
image(s) to the same Qwen2.5-VL engine instead of just its caption.

Used by `main.py query --q` (search then answer from the resolved
top-1 frame) and `query --qa` (answer from a frame_id already recorded
in an existing --out-csv, no search).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.search_engine.vlm_engine import QwenVLEngine, get_qwen_engine
from src.utils_common import get_logger, load_config

logger = get_logger(__name__)


QA_VISUAL_PROMPT_TEMPLATE = """Answer the question using ONLY what is visible in the
provided image(s). Be concise and direct.
- If the question asks "how many" / "bao nhieu", answer with just the
  number, followed by a short justification.
- If the answer cannot be determined from the image(s), say so
  explicitly instead of guessing.

QUESTION: {question}

ANSWER:"""


def answer_question_from_frame(
    question: str,
    frame_paths: list[str | Path],
    engine: Optional[QwenVLEngine] = None,
) -> str:
    """Ask the VLM `question` directly against one or more concrete frame images."""
    cfg = load_config()
    engine = engine or get_qwen_engine(cfg)

    prompt = QA_VISUAL_PROMPT_TEMPLATE.format(question=question)
    answer = engine.generate(
        text_prompt=prompt,
        image_paths=[str(p) for p in frame_paths],
        max_new_tokens=cfg["phase4"]["generator_max_new_tokens"],
        temperature=0.0,
    )
    answer = answer.strip()
    logger.info("Visual QA %r over %d frame(s) -> %r", question[:60], len(frame_paths), answer[:80])
    return answer
