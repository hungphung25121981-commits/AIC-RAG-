"""CSV export/import for ranked search results (frame-level rows).

Used by `main.py query --s/--q` (write) and `main.py query --qa` (read
back, WITHOUT re-running search -- see read_results_csv).

Schema (columns, in this order):
    rank, frame_id, video_id, segment_id, timestamp_sec, score,
    video_title, jump_link, answer

`answer` is populated on the rank==1 row by --q, or written back
in-place by --qa; every other row's `answer` cell stays empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils_common import ensure_dir, get_logger

logger = get_logger(__name__)

CSV_COLUMNS = [
    "rank",
    "frame_id",
    "video_id",
    "segment_id",
    "timestamp_sec",
    "score",
    "video_title",
    "jump_link",
    "answer",
]


@dataclass
class ResultRow:
    rank: int
    frame_id: str
    video_id: str
    segment_id: int
    timestamp_sec: float
    score: float
    video_title: str = ""
    jump_link: str = ""
    answer: str = ""


def write_results_csv(rows: list[ResultRow], out_csv: str | Path) -> Path:
    out_path = Path(out_csv)
    ensure_dir(out_path.parent)
    df = pd.DataFrame([r.__dict__ for r in rows], columns=CSV_COLUMNS)
    df.to_csv(out_path, index=False)
    logger.info("Wrote %d row(s) to %s", len(rows), out_path)
    return out_path


def read_results_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load a previously-written results CSV.

    Raises a clear, actionable error if the file doesn't exist or is
    empty -- callers such as `--qa` must NOT silently fall back to
    re-running search when this is missing; the whole point of `--qa`
    is to read a fixed top-1 without paying for a fresh search.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"--out-csv '{path}' không tồn tại. Hãy chạy `query --s ...` hoặc `query --q ...` "
            f"với cùng --out-csv đó trước, để tạo file kết quả này -- rồi mới dùng `--qa` để "
            f"đọc lại top-1 từ đó (KHÔNG tự động chạy lại tìm kiếm)."
        )
    # keep_default_na=False: an empty `answer`/`jump_link` cell must stay "" (str), not become
    # NaN -- otherwise an all-empty column gets inferred as float64 and a later string write
    # (update_answer_in_csv) raises `TypeError: Invalid value '...' for dtype 'float64'`.
    df = pd.read_csv(path, keep_default_na=False)
    if df.empty:
        raise ValueError(f"CSV '{path}' rỗng (0 dòng), không có kết quả nào để lấy top-1.")
    return df


def get_rank1_row(df: pd.DataFrame) -> pd.Series:
    """Return the row with rank == 1 (or the first row, if there's no rank column)."""
    if "rank" in df.columns:
        df = df.sort_values("rank")
    return df.iloc[0]


def update_answer_in_csv(csv_path: str | Path, row_index, answer: str) -> None:
    """Write `answer` back into the CSV at `row_index` (in-place update)."""
    path = Path(csv_path)
    df = pd.read_csv(path, keep_default_na=False)  # see note in read_results_csv
    if "answer" not in df.columns:
        df["answer"] = ""
    df["answer"] = df["answer"].astype(object)
    df.loc[row_index, "answer"] = answer
    df.to_csv(path, index=False)
    logger.info("Updated 'answer' column at row %s in %s", row_index, path)
