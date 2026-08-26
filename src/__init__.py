"""video-visual-rag: Video-Native Multimodal Hybrid RAG System.

Pipeline covering:
  Phase 1 - Keyframe extraction (OpenCV SSIM + PySceneDetect)
  Phase 2 - OCR extraction & OCR-based caption building (no VLM/LLM call --
            visual_caption is built directly from OCR text)
  Phase 3 - Hybrid indexing (FAISS dense + BM25 sparse, bge-m3 embeddings)
  Phase 4 - Orchestration (RRF fusion, multi-hop sub-query decomposition,
            timestamp-grounded answer synthesis, --rerank) via a single
            Qwen2.5-VL-3B-Instruct model (4-bit NF4, optional QLoRA),
            used ONLY for the --s/--q/--qa search modes and --rerank
"""

__version__ = "0.1.0"
