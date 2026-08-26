"""BGE-M3 dense embedding generator.

Produces float32 numpy embeddings for `full_text_for_embedding`, ready
for L2-normalization + FAISS IndexFlatIP (== cosine similarity search).
"""

from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import pandas as pd

from src.utils_common import free_gpu_memory, get_logger, load_config

logger = get_logger(__name__)

_EMBEDDER_SINGLETON = None


def _get_embedder():
    global _EMBEDDER_SINGLETON
    if _EMBEDDER_SINGLETON is None:
        from FlagEmbedding import BGEM3FlagModel

        cfg = load_config()
        model_id = cfg["phase3"]["embedding_model_id"]
        logger.info("Loading dense embedding model: %s", model_id)
        _EMBEDDER_SINGLETON = BGEM3FlagModel(model_id, use_fp16=True)
    return _EMBEDDER_SINGLETON


def embed_texts(
    texts: list[str],
    batch_size: Optional[int] = None,
    max_length: Optional[int] = None,
) -> np.ndarray:
    """Embed a list of strings -> (N, embedding_dim) float32 array."""
    cfg = load_config()
    p3 = cfg["phase3"]
    batch_size = batch_size or p3["embedding_batch_size"]
    max_length = max_length or p3["embedding_max_length"]

    model = _get_embedder()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    dense_vecs = np.asarray(output["dense_vecs"], dtype=np.float32)

    expected_dim = p3["embedding_dim"]
    if dense_vecs.shape[1] != expected_dim:
        logger.warning(
            "Embedding dim mismatch: got %d, config expects %d. Update settings.yaml.",
            dense_vecs.shape[1],
            expected_dim,
        )
    return dense_vecs


def embed_metadata_dataframe(
    metadata_df: pd.DataFrame,
    text_column: str = "full_text_for_embedding",
) -> np.ndarray:
    """Embed the full_text_for_embedding column of metadata.parquet, in segment_id order.

    IMPORTANT: caller must ensure `metadata_df` is sorted by segment_id
    ascending and contiguous, since row index i here becomes FAISS
    internal row i in faiss_indexer.py.
    """
    assert list(metadata_df["segment_id"]) == list(range(len(metadata_df))), (
        "metadata_df must have contiguous segment_id 0..N-1 sorted ascending "
        "before embedding, to keep FAISS row alignment correct."
    )
    texts = metadata_df[text_column].fillna("").tolist()
    logger.info("Embedding %d segment texts with BGE-M3...", len(texts))
    vectors = embed_texts(texts)

    gc.collect()
    free_gpu_memory()
    return vectors


def unload_embedder() -> None:
    global _EMBEDDER_SINGLETON
    _EMBEDDER_SINGLETON = None
    gc.collect()
    free_gpu_memory()
    logger.info("BGE-M3 embedder unloaded.")
