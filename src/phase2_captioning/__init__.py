"""Phase 2: OCR extraction + OCR-based caption building, metadata.parquet assembly.

Pure OCR/CPU work -- no VLM/LLM call happens in this package at all.
`visual_caption` is built directly from OCR text (see
`ocr_engine.py::build_caption_from_ocr`). The Qwen2.5-VL-3B engine now
lives in `src/search_engine/vlm_engine.py` and is loaded only by Phase 4
(--s/--q/--qa search modes and --rerank) -- this package has no
dependency on it whatsoever.
"""
