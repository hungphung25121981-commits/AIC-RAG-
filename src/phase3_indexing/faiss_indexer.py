"""FAISS IndexFlatIP builder + L2 normalization + segment_id mapping.

Design decisions:
  - IndexFlatIP on L2-normalized vectors == exact cosine similarity search.
  - We wrap IndexFlatIP in IndexIDMap2 so that FAISS internal ids are
    EXPLICITLY set to metadata.parquet's `segment_id` (not implicit row
    order), which is safer if segments are ever removed/re-indexed later.
  - GPU index (faiss-gpu) is used for build+search when a CUDA device is
    available on Kaggle T4; otherwise falls back to CPU transparently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from src.utils_common import ensure_dir, get_logger, load_config

logger = get_logger(__name__)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows in-place-safe (returns a copy), required for IP == cosine sim."""
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    faiss.normalize_L2(vectors)
    return vectors


def _to_gpu_if_available(index: "faiss.Index") -> "faiss.Index":
    try:
        n_gpus = faiss.get_num_gpus()
    except AttributeError:
        n_gpus = 0
    if n_gpus > 0:
        logger.info("Moving FAISS index to GPU (found %d GPU(s)).", n_gpus)
        gpu_resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(gpu_resources, 0, index)
    logger.info("No GPU detected for FAISS; using CPU index.")
    return index


def build_faiss_index(
    vectors: np.ndarray,
    segment_ids: list[int],
    embedding_dim: Optional[int] = None,
    use_gpu: bool = True,
) -> "faiss.Index":
    """Build an IndexFlatIP wrapped in IndexIDMap2, keyed by explicit segment_id.

    `vectors` must already be row-aligned with `segment_ids` (same order).
    """
    cfg = load_config()
    embedding_dim = embedding_dim or cfg["phase3"]["embedding_dim"]

    if vectors.shape[1] != embedding_dim:
        raise ValueError(
            f"Vector dim {vectors.shape[1]} != configured embedding_dim {embedding_dim}"
        )
    if len(segment_ids) != vectors.shape[0]:
        raise ValueError("segment_ids length must match number of vector rows.")

    normalized = normalize_vectors(vectors)

    base_index = faiss.IndexFlatIP(embedding_dim)
    id_index = faiss.IndexIDMap2(base_index)
    id_index.add_with_ids(normalized, np.array(segment_ids, dtype=np.int64))

    logger.info("Built FAISS IndexFlatIP with %d vectors (dim=%d).", id_index.ntotal, embedding_dim)

    if use_gpu:
        id_index = _to_gpu_if_available(id_index)
    return id_index


def save_faiss_index(index: "faiss.Index", output_path: Optional[str | Path] = None) -> Path:
    """Persist the FAISS index to disk (always saved as CPU index for portability)."""
    cfg = load_config()
    output_path = Path(output_path or cfg["paths"]["faiss_index_path"])
    ensure_dir(output_path.parent)

   cpu_index = faiss.index_gpu_to_cpu(index) if hasattr(faiss, "index_gpu_to_cpu") and _is_gpu_index(index) else index
    
    # Khôi phục đúng kiểu dữ liệu (vd: IndexIDMap2) trước khi lưu
    cpu_index = faiss.downcast_index(cpu_index)
    
    faiss.write_index(cpu_index, str(output_path))
    logger.info("Saved FAISS index to %s", output_path)
    return output_path


def _is_gpu_index(index: "faiss.Index") -> bool:
    return "Gpu" in type(index).__name__


def load_faiss_index(index_path: Optional[str | Path] = None, use_gpu: bool = True) -> "faiss.Index":
    cfg = load_config()
    index_path = Path(index_path or cfg["paths"]["faiss_index_path"])
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}. Run Phase 3 first.")

    index = faiss.read_index(str(index_path))
    if use_gpu:
        index = _to_gpu_if_available(index)
    logger.info("Loaded FAISS index from %s (%d vectors).", index_path, index.ntotal)
    return index


def search_faiss_index(
    index: "faiss.Index", query_vector: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Search a single query vector; returns [(segment_id, similarity_score), ...] sorted desc."""
    query = normalize_vectors(query_vector.reshape(1, -1))
    scores, ids = index.search(query, top_k)

    results = [
        (int(seg_id), float(score))
        for seg_id, score in zip(ids[0], scores[0])
        if seg_id != -1
    ]
    return results


def build_and_save_index_from_metadata(
    metadata_df: pd.DataFrame,
    vectors: np.ndarray,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Convenience end-to-end: metadata_df (for segment_id order) + precomputed vectors -> saved index."""
    segment_ids = metadata_df["segment_id"].tolist()
    index = build_faiss_index(vectors, segment_ids, use_gpu=True)
    return save_faiss_index(index, output_path)
