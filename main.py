"""video-visual-rag :: main.py

Single CLI entrypoint for every pipeline task. Each phase is its own
subcommand so you can re-run just the piece you're iterating on
(e.g. re-run `query` repeatedly while tuning `generator.py`, without
re-extracting keyframes or rebuilding the index every time).

Usage
-----
    python main.py extract   --videos-dir data/raw_videos --video-info-dir data/video_info
    python main.py caption   --keyframe-map data/keyframe_map.csv
    python main.py index     --metadata data/metadata.parquet
    python main.py query     "What error message appeared in the terminal?"
    python main.py pipeline  --videos-dir data/raw_videos --video-info-dir data/video_info
    python main.py shell     # interactive query loop against an existing index

    # --- Phase 4, CLI search / QA / TRAKE modes (the only place the VLM,
    #     Qwen2.5-VL-3B-Instruct, is loaded -- Phase 2 `caption` above is
    #     pure OCR, no model call) ---
    # --s : search-only. Ranks frames, prints top rows, writes full ranked CSV.
    python main.py query --s "nguoi dan ong mac ao xanh" --out-csv out.csv --top-n 50 --rerank

    # --q : search + answer. Runs --s internally, takes rank-1, answers from that frame.
    python main.py query --q "co bao nhieu nguoi dang an?" --out-csv out.csv --rerank

    # --qa : answer-from-existing-CSV ONLY. Never re-runs search; errors out if
    #        --out-csv doesn't already exist (must be produced by --s/--q first).
    python main.py query --qa "co bao nhieu nguoi dang an?" --out-csv out.csv

    # trake : locate N sequential sub-moments of one event inside a single video.
    python main.py trake --stages "giam nhay" "bay qua xa" "tiep dat" "dung day" \
        --query "van dong vien nhay xa" --out-csv trake.csv
    # --wandb APIKEY_WWANDB (After running, W&B will print a link with an ID of the run EX: wandb: Run url = .../runs/xyz123 -> ID is xyz123)
    # --wandb-run-id ID ( IT will not create a new project but operate the old run of the ID)
Run `python main.py <command> --help` for per-command options. All
defaults are pulled from config/settings.yaml; CLI flags override them
for a single run only (the yaml file itself is never modified).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.utils_common import free_gpu_memory, get_logger, load_config

logger = get_logger("main")


# ----------------------------------------------------------------------
# Phase 1: extract
# ----------------------------------------------------------------------
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v")


def _find_video_files(videos_dir: str, recursive: bool = True) -> list[str]:
    """Find video files under `videos_dir`, case-insensitively and (by default)
    recursively -- Kaggle datasets are frequently nested a few folders deep
    rather than having videos directly at the top level.

    Globs ONCE with "*" (not once per extension/case-variant) and filters by
    `.suffix.lower()` -- this is the only approach that covers every possible
    case-variant (.mp4/.Mp4/.mP4/.MP4/...) without hardcoding each one, which
    matters because manually listing patterns (e.g. only "*.mp4"+"*.MP4")
    silently misses mixed-case names like "clip.Mp4".
    """
    videos_dir = Path(videos_dir)
    glob_fn = videos_dir.rglob if recursive else videos_dir.glob

    return sorted(
        str(p) for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def cmd_extract(args: argparse.Namespace) -> None:
    from src.phase1_extraction.mapper import build_corpus_keyframe_map

    videos_dir = args.videos_dir
    if not Path(videos_dir).exists():
        logger.error("--videos-dir does not exist: %s", videos_dir)
        sys.exit(1)

    video_paths = _find_video_files(videos_dir, recursive=not args.no_recursive)
    if not video_paths:
        logger.error(
            "No video files (%s) found under %s (recursive=%s). "
            "Run `find %s -type f | head -30` to inspect what's actually there.",
            ", ".join(VIDEO_EXTENSIONS),
            videos_dir,
            not args.no_recursive,
            videos_dir,
        )
        sys.exit(1)

    video_specs = [
        {
            "video_path": vp,
            "video_id": Path(vp).stem,
            "keyframes_output_dir": args.keyframes_dir,
        }
        for vp in video_paths
    ]
    logger.info("Phase 1: extracting keyframes for %d video(s)...", len(video_specs))
    for spec in video_specs:
        logger.info("  found: %s (video_id=%s)", spec["video_path"], spec["video_id"])
    build_corpus_keyframe_map(video_specs, output_csv=args.output_csv)
    logger.info("Phase 1 complete.")


# ----------------------------------------------------------------------
# Phase 2: caption
# ----------------------------------------------------------------------
def cmd_caption(args: argparse.Namespace) -> None:
    from src.phase2_captioning.metadata_builder import build_corpus_metadata

    logger.info("Phase 2: running OCR (visual_caption is built directly from OCR text, no VLM call)...")
    build_corpus_metadata(
        keyframe_map_csv=args.keyframe_map,
        video_info_dir=args.video_info_dir,
        keyframes_dir=args.keyframes_dir,
        output_parquet=args.output_parquet,
    )
    free_gpu_memory()
    logger.info("Phase 2 complete.")


# ----------------------------------------------------------------------
# Phase 3: index
# ----------------------------------------------------------------------
def cmd_index(args: argparse.Namespace) -> None:
    import pandas as pd

    from src.phase3_indexing.bm25_indexer import build_bm25_corpus, save_bm25_corpus
    from src.phase3_indexing.embedder import embed_metadata_dataframe, unload_embedder
    from src.phase3_indexing.faiss_indexer import build_and_save_index_from_metadata

    logger.info("Phase 3: building hybrid index from %s...", args.metadata)
    metadata_df = pd.read_parquet(args.metadata)

    vectors = embed_metadata_dataframe(metadata_df)
    build_and_save_index_from_metadata(metadata_df, vectors, output_path=args.faiss_output)
    unload_embedder()

    bm25_corpus = build_bm25_corpus(metadata_df)
    save_bm25_corpus(bm25_corpus, output_path=args.bm25_output)
    logger.info("Phase 3 complete.")


# ----------------------------------------------------------------------
# Phase 4: query (single question, one-shot)
# ----------------------------------------------------------------------
def _load_query_dependencies(args: argparse.Namespace):
    import pandas as pd

    from src.search_engine.vlm_engine import get_qwen_engine
    from src.phase3_indexing.bm25_indexer import load_bm25_corpus
    from src.phase3_indexing.faiss_indexer import load_faiss_index

    metadata_df = pd.read_parquet(args.metadata)
    faiss_index = load_faiss_index(index_path=args.faiss_index)
    bm25_corpus = load_bm25_corpus(input_path=args.bm25_index)
    engine = get_qwen_engine()
    return metadata_df, faiss_index, bm25_corpus, engine


def _run_search_pipeline(args: argparse.Namespace, query_text: str, select_frame: Optional[bool] = None):
    """Shared by --s and --q: hybrid_search -> optional rerank -> resolve frame_id -> rows.

    Returns (rows, engine) so --q can reuse the already-loaded engine.
    `select_frame` overrides args.select_frame when the caller (--q) needs
    an accurate frame regardless of the flag's default.
    """
    from src.search_engine.retrieval_cli import print_result_rows, search_and_rank, segments_to_result_rows

    metadata_df, faiss_index, bm25_corpus, engine = _load_query_dependencies(args)

    segments = search_and_rank(
        query=query_text,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        top_n=args.top_n,
        rerank=args.rerank,
        video_id=args.video_id,
        engine=engine,
    )
    rows = segments_to_result_rows(
        query_text,
        segments,
        select_frame=args.select_frame if select_frame is None else select_frame,
        engine=engine,
    )
    print_result_rows(rows)

    if args.out_csv:
        from src.search_engine.csv_export import write_results_csv

        write_results_csv(rows, args.out_csv)

    return rows, engine


def cmd_query_search(args: argparse.Namespace) -> None:
    """`query --s TEXT`: search/ranking only, no answer generation."""
    logger.info("Phase 4 [--s search]: %r (top_n=%d, rerank=%s)", args.s, args.top_n, args.rerank)
    _run_search_pipeline(args, args.s)


def cmd_query_qa_auto(args: argparse.Namespace) -> None:
    """`query --q TEXT`: run --s internally, take rank-1, answer TEXT from that frame."""
    from pathlib import Path

    from src.search_engine.csv_export import write_results_csv
    from src.search_engine.visual_qa import answer_question_from_frame

    logger.info("Phase 4 [--q search+answer]: %r (top_n=%d, rerank=%s)", args.q, args.top_n, args.rerank)

    # QA needs an accurate frame for its rank-1 row, regardless of --select-frame.
    rows, engine = _run_search_pipeline(args, args.q, select_frame=True)
    if not rows or not rows[0].frame_id:
        print("\nKhong tim thay ket qua nao de tra loi.")
        return

    top1 = rows[0]
    cfg = load_config()
    frame_path = Path(cfg["paths"]["keyframes_dir"]) / f"{top1.frame_id}.jpg"

    answer = answer_question_from_frame(args.q, [frame_path], engine=engine)
    top1.answer = answer

    print(f"\nAnswer: {answer}")
    print(f"(from rank-1: frame_id={top1.frame_id}, video_id={top1.video_id}, t={top1.timestamp_sec:.1f}s)")

    if args.out_csv:
        write_results_csv(rows, args.out_csv)  # rewrite so the 'answer' column is filled in


def cmd_query_qa_from_csv(args: argparse.Namespace) -> None:
    """`query --qa TEXT --out-csv FILE`: answer using an EXISTING CSV's rank-1 row.

    Never re-runs search -- if --out-csv is missing, this stops with a
    clear error instead of silently falling back to `--s`/`--q`.
    """
    from pathlib import Path

    from src.search_engine.csv_export import get_rank1_row, read_results_csv, update_answer_in_csv
    from src.search_engine.visual_qa import answer_question_from_frame

    if not args.out_csv:
        raise SystemExit(
            "--qa yeu cau phai co --out-csv tro toi file CSV da duoc tao truoc do bang `--s` hoac `--q`."
        )

    df = read_results_csv(args.out_csv)  # raises FileNotFoundError/ValueError with a clear message
    row = get_rank1_row(df)
    frame_id = str(row["frame_id"])
    if not frame_id or frame_id == "nan":
        raise SystemExit(f"Dong top-1 trong '{args.out_csv}' khong co frame_id hop le.")

    cfg = load_config()
    engine = get_qwen_engine_lazy(cfg)
    frame_path = Path(cfg["paths"]["keyframes_dir"]) / f"{frame_id}.jpg"

    logger.info("Phase 4 [--qa from CSV]: %r  (reusing top-1 of %s, frame_id=%s)", args.qa, args.out_csv, frame_id)
    answer = answer_question_from_frame(args.qa, [frame_path], engine=engine)

    print(f"Answer: {answer}")
    print(f"(from existing CSV top-1: frame_id={frame_id}, video_id={row.get('video_id', '')})")

    update_answer_in_csv(args.out_csv, row_index=row.name, answer=answer)


def get_qwen_engine_lazy(cfg):
    from src.search_engine.vlm_engine import get_qwen_engine

    return get_qwen_engine(cfg)


def cmd_query(args: argparse.Namespace) -> None:
    """Dispatch `query` to one of 4 modes, by priority: --qa > --q > --s > legacy positional question."""
    if args.qa is not None:
        cmd_query_qa_from_csv(args)
        return
    if args.q is not None:
        cmd_query_qa_auto(args)
        return
    if args.s is not None:
        cmd_query_search(args)
        return

    if not args.question:
        raise SystemExit(
            "main.py query can mot trong: <question> (RAG day du), --s TEXT (tim kiem/xep hang), "
            "--q TEXT (tim + tra loi), hoac --qa TEXT --out-csv FILE (tra loi tu CSV co san, "
            "KHONG tim lai)."
        )

    from src.search_engine.generator import answer_question, format_answer_with_links

    metadata_df, faiss_index, bm25_corpus, engine = _load_query_dependencies(args)

    answer = answer_question(
        question=args.question,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        engine=engine,
    )
    print(format_answer_with_links(answer))


# ----------------------------------------------------------------------
# Phase 4: trake (locate N sequential sub-moments of one event, 1 video)
# ----------------------------------------------------------------------
def cmd_trake(args: argparse.Namespace) -> None:
    from src.search_engine.trake import run_trake, write_trake_csv

    logger.info("TRAKE: %d stage(s) %s (video_id=%s)", len(args.stages), args.stages, args.video_id)

    metadata_df, faiss_index, bm25_corpus, engine = _load_query_dependencies(args)

    results = run_trake(
        stages=args.stages,
        faiss_index=faiss_index,
        bm25_corpus=bm25_corpus,
        metadata_df=metadata_df,
        video_id=args.video_id,
        overall_query=args.query,
        engine=engine,
        search_top_n=args.search_top_n,
    )

    print(f"{'stage':<7}{'video_id':<16}{'frame_id':<28}{'t(s)':<10}{'score':<8}{'ok':<5}  stage_query")
    for r in results:
        print(
            f"{r.stage_index:<7}{r.video_id:<16}{r.frame_id:<28}{r.timestamp_sec:<10.1f}"
            f"{r.score:<8.3f}{str(r.ok):<5}  {r.stage_query}"
        )

    if args.out_csv:
        write_trake_csv(results, args.out_csv)


# ----------------------------------------------------------------------
# Phase 4: shell (interactive REPL — loads model/index once, ask many questions)
# ----------------------------------------------------------------------
def cmd_shell(args: argparse.Namespace) -> None:
    from src.search_engine.generator import answer_question, format_answer_with_links

    metadata_df, faiss_index, bm25_corpus, engine = _load_query_dependencies(args)

    print("video-visual-rag interactive query shell. Type 'exit' or Ctrl+D to quit.\n")
    while True:
        try:
            question = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = answer_question(
            question=question,
            faiss_index=faiss_index,
            bm25_corpus=bm25_corpus,
            metadata_df=metadata_df,
            engine=engine,
        )
        print()
        print(format_answer_with_links(answer))
        print()


# ----------------------------------------------------------------------
# Full pipeline: extract -> caption -> index (query is run separately)
# ----------------------------------------------------------------------
def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_extract(args)
    cmd_caption(args)
    cmd_index(args)
    logger.info("Full pipeline (extract -> caption -> index) complete. Use `python main.py query` next.")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    paths = cfg["paths"]
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--wandb", type=str, default=None, metavar="API_KEY",
        help="Cung cap W&B API Key de log ket qua qua trinh chay."
    )
    base_parser.add_argument(
        "--wandb-project", type=str, default="video-visual-rag",
        help="Ten Project tren W&B (mac dinh: video-visual-rag)."
    )
    base_parser.add_argument(
        "--wandb-run-id", type=str, default=None, metavar="RUN_ID",
        help="ID cua Run cu de chay tiep neu bi loi. Bo trong de tao moi."
    )
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="video-visual-rag CLI: run each pipeline phase independently, or query an existing index.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- extract (Phase 1) ---
    p_extract = subparsers.add_parser("extract", parents=[base_parser], help="Phase 1: keyframe extraction (SSIM + scene detect).")
    p_extract.add_argument("--videos-dir", default=paths["raw_videos_dir"])
    p_extract.add_argument("--keyframes-dir", default=paths["keyframes_dir"])
    p_extract.add_argument("--output-csv", default=paths["keyframe_map_csv"])
    p_extract.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look for videos directly inside --videos-dir (default: search subfolders too, "
        "since Kaggle datasets are often nested).",
    )
    p_extract.set_defaults(func=cmd_extract)

    # --- caption (Phase 2) ---
    p_caption = subparsers.add_parser("caption", parents=[base_parser], help="Phase 2: OCR-only captioning (no VLM) -> metadata.parquet.")
    p_caption.add_argument("--keyframe-map", default=paths["keyframe_map_csv"])
    p_caption.add_argument("--video-info-dir", default=paths["video_info_dir"])
    p_caption.add_argument("--keyframes-dir", default=paths["keyframes_dir"])
    p_caption.add_argument("--output-parquet", default=paths["metadata_parquet"])
    p_caption.set_defaults(func=cmd_caption)

    # --- index (Phase 3) ---
    p_index = subparsers.add_parser("index", parents=[base_parser], help="Phase 3: build FAISS + BM25 hybrid index.")
    p_index.add_argument("--metadata", default=paths["metadata_parquet"])
    p_index.add_argument("--faiss-output", default=paths["faiss_index_path"])
    p_index.add_argument("--bm25-output", default=paths["bm25_corpus_path"])
    p_index.set_defaults(func=cmd_index)

    # --- query (Phase 4, one-shot) ---
    p_query = subparsers.add_parser(
        "query", parents=[base_parser],
        help="Phase 4: search (--s), search+answer (--q), answer-from-CSV (--qa), "
        "or legacy full-RAG (plain question).",
    )
    p_query.add_argument(
        "question", type=str, nargs="?", default=None,
        help="Legacy mode: full multi-hop RAG answer over the whole corpus. "
        "Ignored if --s/--q/--qa is given.",
    )
    _query_mode_group = p_query.add_mutually_exclusive_group()
    _query_mode_group.add_argument(
        "--s", dest="s", type=str, default=None, metavar="TEXT",
        help="Search mode: rank matching frames for TEXT (no answer generation).",
    )
    _query_mode_group.add_argument(
        "--q", dest="q", type=str, default=None, metavar="TEXT",
        help="Search+answer mode: run --s internally, take rank-1, answer TEXT from that frame's image.",
    )
    _query_mode_group.add_argument(
        "--qa", dest="qa", type=str, default=None, metavar="TEXT",
        help="Answer-from-CSV mode: answer TEXT using an EXISTING --out-csv's rank-1 row. "
        "Does NOT re-run search; errors out if --out-csv doesn't already exist.",
    )
    p_query.add_argument(
        "--out-csv", dest="out_csv", type=str, default=None,
        help="Write ranked results to this CSV (columns: rank, frame_id, video_id, segment_id, "
        "timestamp_sec, score, video_title, jump_link, answer). Required for --qa.",
    )
    p_query.add_argument(
        "--top-n", dest="top_n", type=int, default=int(cfg["phase4"].get("cli_top_n_default", 20)),
        help="How many ranked rows to return for --s/--q.",
    )
    p_query.add_argument(
        "--rerank", action="store_true",
        help="Re-score the fused candidates with one extra VLM pass before ranking (slower, more accurate).",
    )
    p_query.add_argument(
        "--video-id", dest="video_id", type=str, default=None,
        help="Restrict search to a single video_id.",
    )
    p_query.add_argument(
        "--select-frame", dest="select_frame", action="store_true",
        help="Use one extra VLM call per row to pick the exact best-matching frame_id inside each "
        "segment (slower, more precise). --q/--qa always do this for their single rank-1 row "
        "regardless of this flag; without it, --s falls back to the segment's middle frame.",
    )
    p_query.add_argument("--metadata", default=paths["metadata_parquet"])
    p_query.add_argument("--faiss-index", default=paths["faiss_index_path"])
    p_query.add_argument("--bm25-index", default=paths["bm25_corpus_path"])
    p_query.set_defaults(func=cmd_query)

    # --- trake (Phase 4, sequential sub-moment localization within one video) ---
    p_trake = subparsers.add_parser(
        "trake", parents=[base_parser], 
        help="TRAKE: locate N ordered sub-moments of one event inside a single video "
        "(e.g. a jump's takeoff / clearing the bar / landing / standing up).",
    )
    p_trake.add_argument(
        "--stages", nargs="+", required=True, metavar="TEXT",
        help='Ordered stage descriptions, e.g. --stages "giam nhay" "bay qua xa" "tiep dat" "dung day"',
    )
    p_trake.add_argument(
        "--query", type=str, default=None,
        help="Overall event description, used ONLY to auto-localize the target video "
        "(Stage A) when --video-id is not given. Defaults to all --stages joined.",
    )
    p_trake.add_argument(
        "--video-id", dest="video_id", type=str, default=None,
        help="Restrict directly to this video_id, skipping Stage-A auto-localization.",
    )
    p_trake.add_argument("--out-csv", dest="out_csv", type=str, default=None)
    p_trake.add_argument(
        "--search-top-n", dest="search_top_n", type=int,
        default=int(cfg["phase4"].get("trake_search_top_n", 50)),
        help="Candidate pool size per stage/localization search, before filtering to one video.",
    )
    p_trake.add_argument("--metadata", default=paths["metadata_parquet"])
    p_trake.add_argument("--faiss-index", default=paths["faiss_index_path"])
    p_trake.add_argument("--bm25-index", default=paths["bm25_corpus_path"])
    p_trake.set_defaults(func=cmd_trake)

    # --- shell (Phase 4, interactive) ---
    p_shell = subparsers.add_parser("shell", parents=[base_parser], help="Interactive REPL: load model/index once, ask many questions.")
    p_shell.add_argument("--metadata", default=paths["metadata_parquet"])
    p_shell.add_argument("--faiss-index", default=paths["faiss_index_path"])
    p_shell.add_argument("--bm25-index", default=paths["bm25_corpus_path"])
    p_shell.set_defaults(func=cmd_shell)

    # --- pipeline (all of Phase 1-3 in one go) ---
    p_pipeline = subparsers.add_parser("pipeline", parents=[base_parser], help="Run extract -> caption -> index sequentially.")
    p_pipeline.add_argument("--videos-dir", default=paths["raw_videos_dir"])
    p_pipeline.add_argument("--video-info-dir", default=paths["video_info_dir"])
    p_pipeline.add_argument("--keyframes-dir", default=paths["keyframes_dir"])
    p_pipeline.add_argument("--output-csv", default=paths["keyframe_map_csv"])
    p_pipeline.add_argument("--no-recursive", action="store_true")
    p_pipeline.add_argument("--keyframe-map", default=paths["keyframe_map_csv"])
    p_pipeline.add_argument("--output-parquet", default=paths["metadata_parquet"])
    p_pipeline.add_argument("--metadata", default=paths["metadata_parquet"])
    p_pipeline.add_argument("--faiss-output", default=paths["faiss_index_path"])
    p_pipeline.add_argument("--bm25-output", default=paths["bm25_corpus_path"])
    p_pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "wandb") and args.wandb:
        import wandb
        print("\n ---Đang thiết lập Weights & Biases...---")
        
        # Cấp quyền bằng key người dùng nhập
        os.environ["WANDB_API_KEY"] = args.wandb.strip()
        wandb.login()
        
        init_kwargs = {
            "project": args.wandb_project,
            "name": f"run_{args.command}"
        }
        
        # Logic ghi đè/chạy tiếp
        if hasattr(args, "wandb_run_id") and args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
            print(f" ---Đang kết nối lại với Run cũ (ID: {args.wandb_run_id}) để ghi tiếp...---")
        else:
            print("---Đang tạo một Run mới trên W&B...---")
            
        wandb.init(**init_kwargs)
    
    args.func(args)


if __name__ == "__main__":
    main()
