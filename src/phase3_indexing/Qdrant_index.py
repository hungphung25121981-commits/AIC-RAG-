"""Qdrant Hybrid Indexer (Dense BGE-M3 + Visual SigLIP + Sparse BM25/BGE-M3).

Design decisions:
  - We use Qdrant to store:
      1. "dense": BGE-M3 text vectors (1024-dim)
      2. "visual": SigLIP-2 mean-pooled keyframe vectors (1152-dim)
      3. "sparse": BGE-M3 lexical weights
  - Performs native Reciprocal Rank Fusion (RRF) across all available modalities.
  - Supports native metadata payload filtering (e.g. video_id).
  - Qdrant runs locally via file storage (no Docker required).
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


def _init_qdrant_collection(
    client: QdrantClient,
    embedding_dim: int,
    visual_dim: Optional[int] = None,
) -> None:
    """Create a new collection with Dense, Visual (optional), and Sparse vector support."""
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        logger.info(f"Deleted existing Qdrant collection: {COLLECTION_NAME}")

    vectors_config = {
        "dense": VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        )
    }
    if visual_dim:
        vectors_config["visual"] = VectorParams(
            size=visual_dim,
            distance=Distance.COSINE,
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=vectors_config,
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        },
    )
    logger.info(
        f"Created Qdrant Hybrid collection (Dense={embedding_dim}, "
        f"Visual={visual_dim or 'none'}, Sparse=BM25/BGE-M3)"
    )


def _extract_sparse_vectors(texts: list[str], batch_size: Optional[int] = None) -> list[models.SparseVector]:
    """Generate sparse vectors for Qdrant, batched."""
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
    visual_vectors: Optional[np.ndarray] = None,
    db_path: Optional[str | Path] = None,
) -> None:
    """End-to-end: Push metadata + dense vectors + visual vectors + sparse vectors into Qdrant."""
    cfg = load_config()
    dense_dim = cfg["phase3"]["embedding_dim"]
    visual_dim = cfg["phase2"].get("siglip_embedding_dim", 1152) if visual_vectors is not None else None
    
    if dense_vectors.shape[1] != dense_dim:
        raise ValueError(f"Dense vector dim {dense_vectors.shape[1]} != config {dense_dim}")

    if visual_vectors is not None and visual_vectors.shape[1] != visual_dim:
        raise ValueError(f"Visual vector dim {visual_vectors.shape[1]} != config {visual_dim}")

    client = _get_qdrant_client(db_path)
    _init_qdrant_collection(client, dense_dim, visual_dim)

    texts = metadata_df["full_text_for_embedding"].fillna("").tolist()
    segment_ids = metadata_df["segment_id"].tolist()
    records = metadata_df.to_dict(orient="records")

    logger.info("Generating Sparse vectors and packaging Qdrant Points (batched)...")
    UPLOAD_BATCH = 100
    for start in range(0, len(texts), UPLOAD_BATCH):
        batch_texts = texts[start : start + UPLOAD_BATCH]
        batch_records = records[start : start + UPLOAD_BATCH]
        batch_dense = dense_vectors[start : start + UPLOAD_BATCH]
        batch_visual = visual_vectors[start : start + UPLOAD_BATCH] if visual_vectors is not None else None

        # One BGE-M3 forward pass for the whole batch's sparse vectors
        batch_sparse = _extract_sparse_vectors(batch_texts)

        points = []
        for i, (rec, text, dense_vec, sparse_vec) in enumerate(
            zip(batch_records, batch_texts, batch_dense, batch_sparse)
        ):
            seg_id = int(rec["segment_id"])
            vector_dict = {
                "dense": dense_vec.tolist(),
                "sparse": sparse_vec,
            }
            if batch_visual is not None:
                vector_dict["visual"] = batch_visual[i].tolist()

            k_list = rec.get("keyframes_list", [])
            if isinstance(k_list, np.ndarray):
                k_list = k_list.tolist()
            elif not isinstance(k_list, list):
                k_list = []

            payload = {
                "segment_id": seg_id,
                "video_id": str(rec.get("video_id", "")),
                "video_title": str(rec.get("video_title", "")),
                "start_time": float(rec.get("start_time", 0.0)),
                "end_time": float(rec.get("end_time", 0.0)),
                "jump_link": str(rec.get("jump_link", "")),
                "keyframes_list": k_list,
                "ocr_screen_text": str(rec.get("ocr_screen_text", "")),
                "visual_caption": str(rec.get("visual_caption", "")),
                "text": text,
            }

            points.append(
                models.PointStruct(
                    id=seg_id,
                    vector=vector_dict,
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    logger.info(f"Successfully indexed {len(segment_ids)} segments into Qdrant.")
    gc.collect()


def search_qdrant_hybrid(
    query: str, 
    top_k: int = 6, 
    video_id: Optional[str] = None,
    dense_top_k: Optional[int] = None,
    sparse_top_k: Optional[int] = None,
    visual_top_k: Optional[int] = None,
    db_path: Optional[str | Path] = None,
) -> list[tuple[int, float, dict]]:
    """Perform native Hybrid Search (Dense Text + Visual SigLIP + Sparse BM25) using Qdrant."""
    cfg = load_config()
    p4 = cfg["phase4"]
    client = _get_qdrant_client(db_path)

    dense_limit = dense_top_k or p4.get("dense_top_k", 30)
    sparse_limit = sparse_top_k or p4.get("sparse_top_k", 30)
    visual_limit = visual_top_k or p4.get("visual_top_k", 30)

    # 1. Native Qdrant Payload Filter if video_id is specified
    query_filter = None
    if video_id:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ]
        )

    # 2. Check available vector names in the collection
    has_visual = False
    try:
        col_info = client.get_collection(COLLECTION_NAME)
        vec_params = getattr(col_info.config.params, "vectors", None)
        if isinstance(vec_params, dict):
            has_visual = "visual" in vec_params
    except Exception as exc:
        logger.debug("Could not inspect collection vector configuration: %s", exc)

    # 3. Dense (BGE-M3) and Sparse (BGE-M3) query embeddings
    dense_query = embed_texts([query])[0]
    sparse_query = _extract_sparse_vector(query)

    prefetch = [
        models.Prefetch(
            query=dense_query.tolist(),
            using="dense",
            limit=dense_limit,
            filter=query_filter,
        ),
        models.Prefetch(
            query=sparse_query,
            using="sparse",
            limit=sparse_limit,
            filter=query_filter,
        ),
    ]

    # 4. Multimodal SigLIP query embedding (Text -> Image matching)
    if has_visual:
        try:
            from src.phase2_captioning.visual_embedder import embed_texts as embed_siglip_texts
            siglip_text_vec = embed_siglip_texts([query])[0]
            prefetch.append(
                models.Prefetch(
                    query=siglip_text_vec.tolist(),
                    using="visual",
                    limit=visual_limit,
                    filter=query_filter,
                )
            )
        except Exception as exc:
            logger.warning("SigLIP visual query embedding failed (%s); proceeding with text-only prefetch.", exc)

    # 5. Query points with RRF Fusion
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        query_filter=query_filter,
        limit=top_k,
    )
    
    # Returns [(segment_id, score, payload), ...]
    return [(hit.id, hit.score, hit.payload or {}) for hit in results.points]
