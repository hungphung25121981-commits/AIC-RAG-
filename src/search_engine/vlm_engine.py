"""Qwen2.5-VL-3B-Instruct runner (4-bit NF4 bitsandbytes quantization,
with optional QLoRA adapter support via `peft`).

FILE LOCATION NOTE: this module lives under `src/search_engine/` (not
`src/phase2_captioning/`) precisely because it is a Phase 4 module --
Phase 2 has zero dependency on it and never imports it.

ARCHITECTURE NOTE (downsized from 8B/7B, Phase-2 role removed): this
engine used to also run Phase 2 segment captioning. It no longer does --
Phase 2 now builds `visual_caption` straight from OCR (see
`phase2_captioning/ocr_engine.py::build_caption_from_ocr`), with no
model call at all. This module is now used ONLY in Phase 4, for:

  1. The three CLI search modes in `main.py query`:
       --s   search-only            (text-only relevance, via reranker.py)
       --q   search + answer        (visual_qa.py, real image call)
       --qa  answer-from-existing-CSV (visual_qa.py, real image call)
  2. `--rerank` (search_engine/reranker.py) -- text-only relevance scoring
     of fused candidates.
  3. Supporting text-only calls used by those modes: sub-query
     decomposition / relevance judging (multihop_evaluator.py), answer
     synthesis (generator.py), and VLM frame-picking (frame_selector.py).

Because the model is only ever needed at query time now (not during the
Phase 2 `caption` step), it was downsized from the original 7B/8B-class
checkpoint to **Qwen2.5-VL-3B-Instruct** -- noticeably faster/lighter
for these shorter, more frequent Phase 4 calls, optionally sharpened
for the pipeline's own search/rerank/QA prompt style with a QLoRA
adapter (`phase2.vlm_use_qlora` + `phase2.vlm_lora_adapter_path` in
config/settings.yaml) trained on top of the same 4-bit NF4 base weights.

The class is intentionally a lazy singleton loader (`get_qwen_engine()`)
so every Phase 4 module reuses the exact same in-memory model/tokenizer
without reloading a second time.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

from src.utils_common import free_gpu_memory, get_logger, load_config

logger = get_logger(__name__)

_ENGINE_SINGLETON: Optional["QwenVLEngine"] = None


def _load_images(image_paths: list[str | Path]):
    """Load local keyframe images as PIL Images for the Qwen2.5-VL processor.

    Minimal, dependency-free stand-in for `qwen_vl_utils.process_vision_info`'s
    image path -- see the comment in `QwenVLEngine.generate` for why we don't
    import that package. Only local file paths are needed here (Phase 2 always
    passes on-disk keyframe paths), so no URL/base64 handling is required.
    """
    from PIL import Image

    return [Image.open(str(p)).convert("RGB") for p in image_paths]


class QwenVLEngine:
    """Thin wrapper around Qwen2.5-VL-3B-Instruct loaded in 4-bit NF4,
    with an optional QLoRA adapter merged in at load time.

    Used exclusively by Phase 4 (search/QA/rerank) -- see module
    docstring above. Never called from Phase 2 caption building.
    """

    def __init__(self, config: Optional[dict] = None):
        self.cfg = config or load_config()
        p2 = self.cfg["phase2"]
        self.model_id: str = p2["vlm_model_id"]
        self.max_new_tokens: int = p2["vlm_max_new_tokens"]
        self.temperature: float = p2["vlm_temperature"]
        self.top_p: float = p2["vlm_top_p"]
        self.use_qlora: bool = bool(p2.get("vlm_use_qlora", False))
        self.lora_adapter_path: Optional[str] = p2.get("vlm_lora_adapter_path")
        self.model = None
        self.processor = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        compute_dtype = getattr(torch, self.cfg["phase2"]["vlm_compute_dtype"], torch.bfloat16)

        logger.info("Loading %s in 4-bit NF4 (compute_dtype=%s)...", self.model_id, compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map=self.cfg["runtime"]["device_map"],
            torch_dtype=compute_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        if self.use_qlora:
            self._attach_qlora_adapter()

        self.model.eval()
        self._loaded = True
        logger.info("Qwen2.5-VL-3B-Instruct loaded and ready (qlora=%s).", self.use_qlora)

    def _attach_qlora_adapter(self) -> None:
        """Merge a trained PEFT/LoRA adapter onto the 4-bit base model.

        If `vlm_lora_adapter_path` isn't set (no adapter trained yet), this
        prepares the model for QLoRA TRAINING instead (k-bit training prep +
        a fresh `LoraConfig` from settings.yaml) rather than inference, so
        the same config flag covers both "train a new adapter" and "run an
        already-trained adapter" without a separate code path.
        """
        try:
            from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
        except ImportError:
            logger.warning(
                "phase2.vlm_use_qlora=true but `peft` isn't installed; "
                "falling back to the plain base model. `pip install peft` to enable QLoRA."
            )
            return

        p2 = self.cfg["phase2"]
        if self.lora_adapter_path:
            logger.info("Attaching trained QLoRA adapter from %s...", self.lora_adapter_path)
            self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)
        else:
            logger.info("No vlm_lora_adapter_path set -- preparing base model for QLoRA training.")
            self.model = prepare_model_for_kbit_training(self.model)
            lora_config = LoraConfig(
                r=p2.get("vlm_lora_r", 16),
                lora_alpha=p2.get("vlm_lora_alpha", 32),
                lora_dropout=p2.get("vlm_lora_dropout", 0.05),
                target_modules=p2.get("vlm_lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)

    def unload(self) -> None:
        """Explicitly release the model from VRAM (e.g. between Kaggle sessions)."""
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        free_gpu_memory()
        logger.info("Qwen2.5-VL-3B-Instruct unloaded.")

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------
    def generate(
        self,
        text_prompt: str,
        image_paths: Optional[list[str | Path]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Run one Qwen2.5-VL generation call, with 0..N images + a text prompt."""
        self.load()
        import torch

        image_paths = image_paths or []
        content = [{"type": "image", "image": str(p)} for p in image_paths]
        content.append({"type": "text", "text": text_prompt})

        messages = [{"role": "user", "content": content}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # NOTE: this pipeline only ever feeds Qwen2.5-VL still keyframes (never raw
        # video), so we load images directly with PIL instead of pulling in the
        # `qwen-vl-utils` package's `process_vision_info`. That package's video
        # code path unconditionally imports decord/av at module import time, and
        # those wheels are compiled against an older numpy C-ABI -- on Kaggle's
        # numpy 2.0.2 base image that raises
        # `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
        # before we ever get to use the (unused) video functionality. The
        # HF Qwen2.5-VL image processor already does its own min/max-pixel smart
        # resize internally, so nothing is lost by skipping qwen-vl-utils here.
        image_inputs = _load_images(image_paths)

# Kiểm tra nếu danh sách ảnh có dữ liệu thì mới truyền tham số images
        if images and len(images) > 0:
            inputs = self.processor(
                text=text_prompt, # Thay bằng đúng tên biến text trong code cũ của bạn
                images=images,
                return_tensors="pt"
                # ... (giữ nguyên các tham số khác nếu code cũ có) ...
            )
        else:
            # Nếu không có ảnh, bỏ HẲN tham số images đi
            inputs = self.processor(
                text=text_prompt, # Thay bằng đúng tên biến text trong code cũ của bạn
                return_tensors="pt"
                # ... (giữ nguyên các tham số khác nếu code cũ có) ...
            ).to(self.model.device)

        effective_temperature = temperature if temperature is not None else self.temperature
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            do_sample=effective_temperature > 0,
        )
        # transformers warns/errors on sampling-only kwargs (temperature, top_p) being set
        # while do_sample=False (greedy decoding ignores them) -- only include them when
        # actually sampling, and always resolve 0.0 to greedy instead of silently falling
        # back to the config default (the original bug this fixes).
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = effective_temperature
            gen_kwargs["top_p"] = self.top_p

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]

        del inputs, generated_ids, trimmed_ids
        return output_text.strip()


def get_qwen_engine(config: Optional[dict] = None) -> QwenVLEngine:
    """Return the process-wide singleton QwenVLEngine (lazy-loaded on first `.generate`).

    Only ever needed from Phase 4 (search_engine/*.py) -- Phase 2 caption
    building no longer imports or calls this at all.
    """
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is None:
        _ENGINE_SINGLETON = QwenVLEngine(config=config)
    return _ENGINE_SINGLETON
