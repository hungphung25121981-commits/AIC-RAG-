"""Qdrant Hybrid Indexer (replaces FAISS and rank_bm25).

Design decisions:
  - We use Qdrant to store BOTH dense vectors (BGE-M3) and sparse vectors (BM25)
    in a single payload, allowing native Reciprocal Rank Fusion (RRF).
  - Qdrant runs locally via file storage (no Docker required), keeping RAM footprint
    minimal compared to Milvus.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams

from src.utils_common import ensure_dir, get_logger, load_config
# Import BGE-M3 embedder for dense vectors (and the shared model instance for sparse too)
from src.phase3_indexing.embedder import embed_texts, get_shared_bge_m3_model

logger = get_logger(__name__)

# Qdrant Collection Name
COLLECTION_NAME = "video_rag_collection"

def _get_qdrant_client(db_path: Optional[str | Path] = None) -> QdrantClient:
    """Initialize a local Qdrant client."""
    cfg = load_config()
    db_path = Path(db_path or cfg["paths"].get("qdrant_db_path", "data/index/qdrant_storage"))
    ensure_dir(db_path)
    
    # Run locally without background service
    return QdrantClient(path=str(db_path))

def _init_qdrant_collection(client: QdrantClient, embedding_dim: int) -> None:
    """Create a new collection with both Dense and Sparse vector support."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing Qdrant collection: {COLLECTION_NAME}")

    # Cấu hình đa Vector: Dense (BGE-M3) và Sparse (BM25)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(
                size=embedding_dim, 
                distance=Distance.COSINE # Tương đương IndexFlatIP + L2 norm
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )
    logger.info(f"Created Qdrant Hybrid collection (Dense dim={embedding_dim} + Sparse)")


def _extract_sparse_vectors(texts: list[str], batch_size: Optional[int] = None) -> list[models.SparseVector]:
    """
    Generate sparse vectors for Qdrant, batched.

    In a real production system, you'd use a dedicated Sparse Encoder
    (like SPLADE or BGE-M3's sparse output). Here we use BGE-M3's sparse
    token weights to directly replace rank_bm25.

    IMPORTANT: reuses the SAME cached BGE-M3 instance as the dense embedder
    (`embedder.get_shared_bge_m3_model()`) instead of constructing a brand-new
    ~2.2GB BGEM3FlagModel per call -- the previous per-text instantiation made
    indexing (and every single query at search time) reload the full checkpoint
    from disk/HF cache on every row, which is catastrophically slow and can OOM
    a T4 under repeated loads.
    """
    if not texts:
        return []
    cfg = load_config()
    batch_size = batch_size or cfg["phase3"]["embedding_batch_size"]

    model = get_shared_bge_m3_model()
    output = model.encode(
        texts,
        batch_size=batch_size,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    sparse_vectors: list[models.SparseVector] = []
    for sparse_dict in output["lexical_weights"]:
        indices = [int(k) for k in sparse_dict.keys()]
        values = [float(v) for v in sparse_dict.values()]
        sparse_vectors.append(models.SparseVector(indices=indices, values=values))
    return sparse_vectors


def _extract_sparse_vector(text: str) -> models.SparseVector:
    """Single-text convenience wrapper around `_extract_sparse_vectors` (e.g. for a query)."""
    return _extract_sparse_vectors([text])[0]


def build_and_save_index_from_metadata(
    metadata_df: pd.DataFrame,
    dense_vectors: np.ndarray,
    db_path: Optional[str | Path] = None,
) -> None:
    """End-to-end: Push metadata + dense vectors + sparse vectors into Qdrant."""
    cfg = load_config()
    embedding_dim = cfg["phase3"]["embedding_dim"]
    
    if dense_vectors.shape[1] != embedding_dim:
        raise ValueError(f"Vector dim {dense_vectors.shape[1]} != config {embedding_dim}")

    client = _get_qdrant_client(db_path)
    _init_qdrant_collection(client, embedding_dim)

    texts = metadata_df["full_text_for_embedding"].fillna("").tolist()
    segment_ids = metadata_df["segment_id"].tolist()

    logger.info("Generating Sparse vectors and packaging Qdrant Points (batched)...")
    UPLOAD_BATCH = 100
    for start in range(0, len(texts), UPLOAD_BATCH):
        batch_texts = texts[start : start + UPLOAD_BATCH]
        batch_ids = segment_ids[start : start + UPLOAD_BATCH]
        batch_dense = dense_vectors[start : start + UPLOAD_BATCH]

        # One BGE-M3 forward pass for the whole batch's sparse vectors, not one per row.
        batch_sparse = _extract_sparse_vectors(batch_texts)

        points = [
            models.PointStruct(
                id=int(seg_id),
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": sparse_vec,
                },
                # Lưu đính kèm metadata để truy xuất nhanh mà không cần tra ngược Parquet
                payload={"segment_id": int(seg_id), "text": text},
            )
            for seg_id, text, dense_vec, sparse_vec in zip(batch_ids, batch_texts, batch_dense, batch_sparse)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    logger.info(f"Successfully indexed {len(segment_ids)} segments into Qdrant.")
    
    # Clean up
    gc.collect()


def search_qdrant_hybrid(
    query: str, 
    top_k: int = 6, 
    db_path: Optional[str | Path] = None
) -> list[tuple[int, float]]:
    """Perform native Hybrid Search (Dense + Sparse) using Qdrant."""
    client = _get_qdrant_client(db_path)
    
    # 1. Nhúng câu truy vấn thành Dense và Sparse
    dense_query = embed_texts([query])[0]
    sparse_query = _extract_sparse_vector(query)
    
    # 2. Bắn Query vào Qdrant để tự động gộp kết quả bằng RRF
    prefetch = [
        models.Prefetch(
            query=dense_query.tolist(),
            using="dense",
            limit=20,
        ),
        models.Prefetch(
            query=sparse_query,
            using="sparse",
            limit=20,
        ),
    ]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )
    
    # Trả về format [(segment_id, score), ...] để khớp với pipeline cũ
    return [(hit.id, hit.score) for hit in results.points]