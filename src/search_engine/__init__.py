"""Phase 4: Hybrid search (FAISS + BM25 + RRF), multi-hop reasoning, and
timestamp-grounded answer generation.

`vlm_engine.py` (Qwen2.5-VL-3B-Instruct, 4-bit NF4, optional QLoRA) lives
in this package because it is used ONLY here -- for the --s/--q/--qa
search modes and --rerank. This is the only phase that ever loads the VLM.
"""
