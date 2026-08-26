"""Shared utilities: config loading, logging, memory management.

Every module in src/ imports from here instead of re-implementing
config parsing / GPU cleanup, so behavior stays consistent and the
single source of truth remains config/settings.yaml.
"""

from __future__ import annotations

import gc
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """Load and cache config/settings.yaml.

    Cached with lru_cache since it is read-only for the lifetime of a
    process; call ``load_config.cache_clear()`` in tests if needed.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from settings.yaml relative to repo root."""
    p = Path(relative_path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger; safe to call repeatedly (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        cfg = None
        try:
            cfg = load_config()
        except FileNotFoundError:
            pass
        log_level = level or (cfg["runtime"]["log_level"] if cfg else "INFO")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        logger.propagate = False
    return logger


def free_gpu_memory() -> None:
    """Best-effort GPU + host memory cleanup. Call every N segments in loops."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def maybe_free_memory(step_index: int, cfg: dict[str, Any]) -> None:
    """Trigger free_gpu_memory() on the cadence defined in settings.yaml."""
    runtime_cfg = cfg.get("runtime", {})
    n_empty = runtime_cfg.get("empty_cache_every_n_segments", 10)
    n_gc = runtime_cfg.get("gc_collect_every_n_segments", 10)
    if n_empty and step_index % n_empty == 0:
        free_gpu_memory()
    elif n_gc and step_index % n_gc == 0:
        gc.collect()


def safe_load_video_info(video_info_path: str | Path) -> dict[str, Any]:
    """Load a video_info.json with fully dynamic / missing-key tolerance.

    Returns an empty dict (never raises) if the file is missing or malformed,
    so downstream code can always use `.get(key, default)` safely.
    """
    path = Path(video_info_path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def format_dynamic_metadata_block(video_info: dict[str, Any], max_chars: int = 900) -> str:
    """Turn an arbitrary video_info.json dict into a readable prompt block.

    Only known-useful keys are prioritized but ANY key present is included,
    since sources vary (YouTube exports, internal screen-recording tools,
    lecture-capture systems, etc. all use different schemas).
    """
    if not video_info:
        return "(no metadata available)"

    priority_keys = ["title", "description", "keywords", "channel", "publish_date", "watch_url"]
    lines: list[str] = []

    for key in priority_keys:
        if key in video_info and video_info[key] not in (None, "", []):
            lines.append(f"{key}: {_stringify(video_info[key])}")

    for key, value in video_info.items():
        if key in priority_keys or value in (None, "", []):
            continue
        lines.append(f"{key}: {_stringify(value)}")

    block = "\n".join(lines)
    return block[:max_chars]


def _stringify(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value[:15])
    return str(value)


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(str(path)) if not Path(path).is_absolute() else Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
