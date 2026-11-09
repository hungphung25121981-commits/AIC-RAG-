"""SigLIP visual-text similarity encoder.

This fills the slot the original design reserved for CLIP-ViT-B/32
("visual-similarity re-rank" in config/settings.yaml), using SigLIP
instead. SigLIP's sigmoid loss (vs. CLIP's softmax contrastive loss)
gives consistently better image-text retrieval quality at the same
model size, and its HF API is nearly a drop-in replacement -- but there
ARE real differences that matter for correctness, called out below.

Used by `search_engine/frame_selector.py`'s fast (non-VLM) frame-pick
path: given a segment's few keyframes and a text query, pick the frame
whose SigLIP image embedding is most similar to the query's SigLIP text
embedding -- a single cheap forward pass instead of a Qwen2.5-VL
generation call per segment.

--------------------------------------------------------------------
Porting notes (CLIP -> SigLIP): version, environment, tensor shapes
--------------------------------------------------------------------
1. API classes: `AutoModel` / `AutoProcessor` are used instead of the
   CLIP-specific `CLIPModel` / `CLIPProcessor` so this module works
   unmodified whether `siglip_model_id` points at a "siglip" or a
   newer "siglip2" checkpoint -- both resolve through the same Auto*
   classes to `SiglipModel`/`Siglip2Model`.

2. Version requirement: `SiglipModel` has been in `transformers` since
   4.37, which the repo's pinned `transformers==4.46.3` already
   satisfies -- no requirements.txt bump needed for base SigLIP. If
   you switch `siglip_model_id` to a "siglip2-*" / NaFlex checkpoint,
   bump the pin to `transformers>=4.49` first (Siglip2 support landed
   later); this module does not gate on that, so an incompatible pin
   will surface as an ordinary `from_pretrained` ImportError/KeyError,
   not a silent wrong-shape bug.

3. Tokenizer padding: SigLIP was TRAINED with fixed-length padding
   (`padding="max_length"`, 64 tokens for every official
   google/siglip-* checkpoint), unlike CLIP's dynamic `padding=True`.
   Using CLIP-style dynamic padding here would silently degrade
   similarity quality rather than error out, so `embed_texts()` always
   passes `padding="max_length"` + the configured
   `siglip_text_max_length`.

4. Embedding normalization: like `CLIPModel`, `SiglipModel`'s
   `get_image_features()` / `get_text_features()` convenience methods
   return the RAW projected features -- normalization happens only
   inside the model's full `forward()`, not in these getters. This
   module L2-normalizes both sides explicitly before the dot product,
   using the same convention as the FAISS `IndexFlatIP` path in
   `phase3_indexing/faiss_indexer.py` (unit-norm vectors -> dot product
   == cosine similarity).

5. Tensor-dimension check: `get_image_features()` -> (N, projection_dim)
   and `get_text_features()` -> (M, projection_dim) MUST share the same
   trailing dimension for the dot product in
   `best_matching_image_index()` to be valid. This is verified twice:
   once at model-load time against the model's own
   `config.projection_dim` (logged as a warning if it disagrees with
   `siglip_embedding_dim` in settings.yaml), and once at call time in
   `best_matching_image_index()` (raises `ValueError` rather than
   letting numpy silently broadcast/fail on a shape mismatch).
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import numpy as np

from src.utils_common import free_gpu_memory, get_logger, load_config

logger = get_logger(__name__)

_SIGLIP_SINGLETON: Optional[tuple] = None  # (model, processor, device) lazy-loaded


def _get_siglip():
    global _SIGLIP_SINGLETON
    if _SIGLIP_SINGLETON is None:
        import torch
        from transformers import AutoModel, AutoProcessor

        cfg = load_config()
        p2 = cfg["phase2"]
        model_id = p2["siglip_model_id"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype_name = p2.get("siglip_compute_dtype", "float16")
        dtype = getattr(torch, dtype_name, torch.float32) if device == "cuda" else torch.float32

        logger.info("Loading SigLIP visual-similarity encoder: %s (device=%s, dtype=%s)", model_id, device, dtype)

        model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)

        configured_dim = p2.get("siglip_embedding_dim")
        actual_dim = getattr(model.config, "projection_dim", None)
        if configured_dim and actual_dim and configured_dim != actual_dim:
            logger.warning(
                "siglip_embedding_dim=%s in settings.yaml but %s reports projection_dim=%s; "
                "update settings.yaml to match.",
                configured_dim,
                model_id,
                actual_dim,
            )

        _SIGLIP_SINGLETON = (model, processor, device)
    return _SIGLIP_SINGLETON


def _l2_normalize(tensor):
    """Row-wise L2 normalize. SigLIP's get_*_features() do NOT normalize internally."""
    norm = tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return tensor / norm


def embed_images(image_paths: list[str | Path]) -> np.ndarray:
    """Embed a batch of images -> (N, projection_dim) float32, L2-normalized."""
    if not image_paths:
        return np.empty((0, 0), dtype=np.float32)

    import torch
    from PIL import Image

    model, processor, device = _get_siglip()
    images = [Image.open(str(p)).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)  
        
        # BÓC LÕI TENSOR RA:
        if hasattr(image_features, 'image_embeds'):
            image_features = image_features.image_embeds
        elif hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
            
    image_features = _l2_normalize(image_features.float())
    return image_features.cpu().numpy().astype(np.float32)
def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of text queries -> (M, projection_dim) float32, L2-normalized."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    import torch

    model, processor, device = _get_siglip()
    cfg = load_config()
    max_length = cfg["phase2"].get("siglip_text_max_length", 64)

    inputs = processor(
        text=texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)  
        
        # BÓC LÕI TENSOR RA:
        if hasattr(text_features, 'text_embeds'):
            text_features = text_features.text_embeds
        elif hasattr(text_features, 'pooler_output'):
            text_features = text_features.pooler_output

    text_features = _l2_normalize(text_features.float())
    return text_features.cpu().numpy().astype(np.float32)


def best_matching_image_index(query: str, image_paths: list[str | Path]) -> Optional[int]:
    """Return the index into `image_paths` most similar to `query`, or None if empty.

    Both embeddings are unit-norm (see `_l2_normalize`), so their dot
    product IS the cosine similarity -- same convention used for the
    FAISS `IndexFlatIP` corpus search in phase3.
    """
    if not image_paths:
        return None

    text_vecs = embed_texts([query])       # (1, dim)
    image_vecs = embed_images(image_paths)  # (N, dim)

    if text_vecs.shape[-1] != image_vecs.shape[-1]:
        raise ValueError(
            f"SigLIP text/image embedding dim mismatch: text={text_vecs.shape[-1]} "
            f"vs image={image_vecs.shape[-1]}. Check `siglip_model_id` in settings.yaml "
            f"(text and image towers must share one checkpoint's projection space)."
        )

    similarities = image_vecs @ text_vecs[0]  # (N,) cosine similarities in [-1, 1]
    return int(np.argmax(similarities))


def embed_segment_visuals(
    segment_keyframe_paths: list[list[str | Path]],
    batch_size: int = 32,
) -> np.ndarray:
    """Embed keyframes for a list of segments -> (num_segments, projection_dim) float32 array.

    For each segment, embeds its keyframes and computes the L2-normalized mean vector.
    Batches image loading and GPU inference for maximum efficiency.
    """
    if not segment_keyframe_paths:
        return np.empty((0, 0), dtype=np.float32)

    import torch
    from PIL import Image

    model, processor, device = _get_siglip()
    cfg = load_config()
    projection_dim = cfg["phase2"].get("siglip_embedding_dim", 1152)

    # Flatten all existing image paths while tracking segment indices
    flat_paths = []
    seg_mapping = []  # maps flat_index -> segment_index
    for seg_idx, paths in enumerate(segment_keyframe_paths):
        for p in paths:
            p_obj = Path(p)
            if p_obj.exists():
                flat_paths.append(str(p_obj))
                seg_mapping.append(seg_idx)

    num_segments = len(segment_keyframe_paths)
    if not flat_paths:
        logger.warning("No existing keyframe images found on disk to embed; returning zero visual vectors.")
        return np.zeros((num_segments, projection_dim), dtype=np.float32)

    logger.info("Embedding %d keyframes across %d segments with SigLIP...", len(flat_paths), num_segments)

    # Embed all images in batches
    all_image_embeds = []
    for start in range(0, len(flat_paths), batch_size):
        batch_paths = flat_paths[start : start + batch_size]
        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt").to(device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                if hasattr(image_features, 'image_embeds'):
                    image_features = image_features.image_embeds
                elif hasattr(image_features, 'pooler_output'):
                    image_features = image_features.pooler_output
            image_features = _l2_normalize(image_features.float())
            all_image_embeds.append(image_features.cpu().numpy().astype(np.float32))
        except Exception as exc:
            logger.error("Error embedding image batch [%d:%d]: %s", start, start + len(batch_paths), exc)
            all_image_embeds.append(np.zeros((len(batch_paths), projection_dim), dtype=np.float32))

    flat_embeds = np.concatenate(all_image_embeds, axis=0) if all_image_embeds else np.empty((0, projection_dim))

    # Aggregate by segment (mean-pooling + L2 normalize)
    segment_vectors = np.zeros((num_segments, projection_dim), dtype=np.float32)
    segment_counts = np.zeros(num_segments, dtype=np.int32)

    for embed, seg_idx in zip(flat_embeds, seg_mapping):
        segment_vectors[seg_idx] += embed
        segment_counts[seg_idx] += 1

    for seg_idx in range(num_segments):
        if segment_counts[seg_idx] > 0:
            vec = segment_vectors[seg_idx] / segment_counts[seg_idx]
            norm = np.linalg.norm(vec)
            segment_vectors[seg_idx] = vec / max(norm, 1e-12)

    return segment_vectors


def unload_siglip() -> None:
    """Explicitly release the SigLIP model from VRAM (mirrors QwenVLEngine.unload())."""
    global _SIGLIP_SINGLETON
    _SIGLIP_SINGLETON = None
    gc.collect()
    free_gpu_memory()
    logger.info("SigLIP encoder unloaded.")
