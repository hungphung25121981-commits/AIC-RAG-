"""rank_bm25 sparse lexical indexer + serializer.

BM25 complements dense (BGE-M3) retrieval by catching exact-token
matches that embeddings can miss -- e.g. a specific file name, error
code, ticker symbol, or terminal command visible via OCR.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from rank_bm25 import BM25Okapi

from src.utils_common import ensure_dir, get_logger, load_config

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def simple_lower_split_tokenizer(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer. Deliberately simple/fast for BM25
    over short segment texts; swap for a stemmer/lemmatizer if recall on
    morphological variants becomes an issue.

    Uses `[^\\W_]+` (Unicode word chars, minus underscore) instead of the
    ASCII-only `[a-z0-9]+` -- the latter silently drops any accented/non-Latin
    character (e.g. Vietnamese diacritics, or any other non-English on-screen
    text captured by OCR/captions), which can shred words into meaningless
    fragments and quietly tank BM25 recall on non-ASCII content.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class BM25Corpus:
    tokenized_corpus: list[list[str]]
    segment_ids: list[int]
    bm25: Optional[BM25Okapi] = field(default=None, repr=False)

    def build(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=k1, b=b)

    def search(self, query: str, top_k: int, tokenizer=simple_lower_split_tokenizer) -> list[tuple[int, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call .build() or load from disk first.")
        tokenized_query = tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(zip(self.segment_ids, scores), key=lambda x: x[1], reverse=True)
        return [(seg_id, float(score)) for seg_id, score in ranked[:top_k] if score > 0]


def build_bm25_corpus(
    metadata_df: pd.DataFrame,
    text_column: str = "full_text_for_embedding",
    k1: Optional[float] = None,
    b: Optional[float] = None,
) -> BM25Corpus:
    cfg = load_config()
    p3 = cfg["phase3"]
    k1 = k1 if k1 is not None else p3["bm25_k1"]
    b = b if b is not None else p3["bm25_b"]

    texts = metadata_df[text_column].fillna("").tolist()
    segment_ids = metadata_df["segment_id"].tolist()
    tokenized_corpus = [simple_lower_split_tokenizer(t) for t in texts]

    corpus = BM25Corpus(tokenized_corpus=tokenized_corpus, segment_ids=segment_ids)
    corpus.build(k1=k1, b=b)
    logger.info("Built BM25 index over %d segments (k1=%.2f, b=%.2f).", len(texts), k1, b)
    return corpus


def save_bm25_corpus(corpus: BM25Corpus, output_path: Optional[str | Path] = None) -> Path:
    cfg = load_config()
    output_path = Path(output_path or cfg["paths"]["bm25_corpus_path"])
    ensure_dir(output_path.parent)
    with open(output_path, "wb") as f:
        pickle.dump(corpus, f)
    logger.info("Saved BM25 corpus pickle to %s", output_path)
    return output_path


def load_bm25_corpus(input_path: Optional[str | Path] = None) -> BM25Corpus:
    cfg = load_config()
    input_path = Path(input_path or cfg["paths"]["bm25_corpus_path"])
    if not input_path.exists():
        raise FileNotFoundError(f"BM25 corpus not found at {input_path}. Run Phase 3 first.")
    with open(input_path, "rb") as f:
        corpus: BM25Corpus = pickle.load(f)
    logger.info("Loaded BM25 corpus with %d segments from %s", len(corpus.segment_ids), input_path)
    return corpus
