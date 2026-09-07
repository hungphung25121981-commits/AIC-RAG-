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

"""video-visual-rag :: main.py

Single CLI entrypoint for every pipeline task. Each phase is its own
subcommand so you can re-run just the piece you're iterating on.
"""

from __future__ import annotations

import argparse
import os
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
        logger.error("No video files found under %s", videos_dir)
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
# Phase 3: index (Qdrant Hybrid)
# ----------------------------------------------------------------------
def cmd_index(args: argparse.Namespace) -> None:
    import pandas as pd
    from src.phase3_indexing.embedder import embed_metadata_dataframe, unload_embedder
    from src.phase3_indexing.Qdrant_index import build_and_save_index_from_metadata

    logger.info("Phase 3: building Qdrant hybrid index from %s...", args.metadata)
    metadata_df = pd.read_parquet(args.metadata)

    vectors = embed_metadata_dataframe(metadata_df)
    build_and_save_index_from_metadata(metadata_df, vectors)
    unload_embedder()

    logger.info("Phase 3 complete.")


# ----------------------------------------------------------------------
# Phase 4: query
# ----------------------------------------------------------------------
def _load_query_dependencies(args: argparse.Namespace):
    import pandas as pd
    from src.search_engine.vlm_engine import get_qwen_engine

    metadata_df = pd.read_parquet(args.metadata)
    engine = get_qwen_engine()
    # Qdrant handles DB connection internally, no need to load/pass FAISS or BM25
    return metadata_df, engine


def _run_search_pipeline(args: argparse.Namespace, query_text: str, select_frame: Optional[bool] = None):
    from src.search_engine.retrieval_cli import print_result_rows, search_and_rank, segments_to_result_rows

    metadata_df, engine = _load_query_dependencies(args)

    segments = search_and_rank(
        query=query_text,
        faiss_index=None,    # Forward None to satisfy older signatures
        bm25_corpus=None,
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
    logger.info("Phase 4 [--s search]: %r (top_n=%d, rerank=%s)", args.s, args.top_n, args.rerank)
    _run_search_pipeline(args, args.s)


def cmd_query_qa_auto(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.search_engine.csv_export import write_results_csv
    from src.search_engine.visual_qa import answer_question_from_frame

    logger.info("Phase 4 [--q search+answer]: %r (top_n=%d, rerank=%s)", args.q, args.top_n, args.rerank)
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
        write_results_csv(rows, args.out_csv)


def cmd_query_qa_from_csv(args: argparse.Namespace) -> None:
    from pathlib import Path
    from src.search_engine.csv_export import get_rank1_row, read_results_csv, update_answer_in_csv
    from src.search_engine.visual_qa import answer_question_from_frame

    if not args.out_csv:
        raise SystemExit("--qa yeu cau phai co --out-csv")

    df = read_results_csv(args.out_csv)
    row = get_rank1_row(df)
    frame_id = str(row["frame_id"])
    
    cfg = load_config()
    engine = get_qwen_engine_lazy(cfg)
    frame_path = Path(cfg["paths"]["keyframes_dir"]) / f"{frame_id}.jpg"

    logger.info("Phase 4 [--qa from CSV]: %r", args.qa)
    answer = answer_question_from_frame(args.qa, [frame_path], engine=engine)

    print(f"Answer: {answer}")
    update_answer_in_csv(args.out_csv, row_index=row.name, answer=answer)


def get_qwen_engine_lazy(cfg):
    from src.search_engine.vlm_engine import get_qwen_engine
    return get_qwen_engine(cfg)


def cmd_query(args: argparse.Namespace) -> None:
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
        raise SystemExit("Cần truyền vào câu hỏi hoặc mode (--s/--q/--qa).")

    from src.search_engine.generator import answer_question, format_answer_with_links

    metadata_df, engine = _load_query_dependencies(args)
    answer = answer_question(
        question=args.question,
        faiss_index=None,
        bm25_corpus=None,
        metadata_df=metadata_df,
        engine=engine,
    )
    print(format_answer_with_links(answer))


# ----------------------------------------------------------------------
# Phase 4: trake
# ----------------------------------------------------------------------
def cmd_trake(args: argparse.Namespace) -> None:
    from src.search_engine.trake import run_trake, write_trake_csv

    logger.info("TRAKE: %d stage(s)", len(args.stages))
    metadata_df, engine = _load_query_dependencies(args)

    results = run_trake(
        stages=args.stages,
        faiss_index=None,
        bm25_corpus=None,
        metadata_df=metadata_df,
        video_id=args.video_id,
        overall_query=args.query,
        engine=engine,
        search_top_n=args.search_top_n,
    )

    print(f"{'stage':<7}{'video_id':<16}{'frame_id':<28}{'t(s)':<10}{'score':<8}{'ok':<5}  stage_query")
    for r in results:
        print(f"{r.stage_index:<7}{r.video_id:<16}{r.frame_id:<28}{r.timestamp_sec:<10.1f}{r.score:<8.3f}{str(r.ok):<5}  {r.stage_query}")

    if args.out_csv:
        write_trake_csv(results, args.out_csv)


# ----------------------------------------------------------------------
# Phase 4: shell
# ----------------------------------------------------------------------
def cmd_shell(args: argparse.Namespace) -> None:
    from src.search_engine.generator import answer_question, format_answer_with_links

    metadata_df, engine = _load_query_dependencies(args)

    print("video-visual-rag interactive query shell. Type 'exit' or Ctrl+D to quit.\n")
    while True:
        try:
            question = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"exit", "quit"}:
            if question.lower() in {"exit", "quit"}: break
            continue

        answer = answer_question(
            question=question,
            faiss_index=None,
            bm25_corpus=None,
            metadata_df=metadata_df,
            engine=engine,
        )
        print("\n" + format_answer_with_links(answer) + "\n")


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_extract(args)
    cmd_caption(args)
    cmd_index(args)
    logger.info("Full pipeline complete.")


# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    cfg = load_config()
    paths = cfg["paths"]
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--wandb", type=str, default=None, metavar="API_KEY")
    base_parser.add_argument("--wandb-project", type=str, default="video-visual-rag")
    base_parser.add_argument("--wandb-run-id", type=str, default=None, metavar="RUN_ID")
    
    parser = argparse.ArgumentParser(prog="main.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = subparsers.add_parser("extract", parents=[base_parser])
    p_extract.add_argument("--videos-dir", default=paths["raw_videos_dir"])
    p_extract.add_argument("--keyframes-dir", default=paths["keyframes_dir"])
    p_extract.add_argument("--output-csv", default=paths["keyframe_map_csv"])
    p_extract.add_argument("--no-recursive", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    # caption
    p_caption = subparsers.add_parser("caption", parents=[base_parser])
    p_caption.add_argument("--keyframe-map", default=paths["keyframe_map_csv"])
    p_caption.add_argument("--video-info-dir", default=paths["video_info_dir"])
    p_caption.add_argument("--keyframes-dir", default=paths["keyframes_dir"])
    p_caption.add_argument("--output-parquet", default=paths["metadata_parquet"])
    p_caption.set_defaults(func=cmd_caption)

    # index
    p_index = subparsers.add_parser("index", parents=[base_parser])
    p_index.add_argument("--metadata", default=paths["metadata_parquet"])
    p_index.set_defaults(func=cmd_index)

    # query
    p_query = subparsers.add_parser("query", parents=[base_parser])
    p_query.add_argument("question", type=str, nargs="?", default=None)
    _query_mode_group = p_query.add_mutually_exclusive_group()
    _query_mode_group.add_argument("--s", dest="s", type=str, default=None)
    _query_mode_group.add_argument("--q", dest="q", type=str, default=None)
    _query_mode_group.add_argument("--qa", dest="qa", type=str, default=None)
    p_query.add_argument("--out-csv", dest="out_csv", type=str, default=None)
    p_query.add_argument("--top-n", dest="top_n", type=int, default=int(cfg["phase4"].get("cli_top_n_default", 20)))
    p_query.add_argument("--rerank", action="store_true")
    p_query.add_argument("--video-id", dest="video_id", type=str, default=None)
    p_query.add_argument("--select-frame", dest="select_frame", action="store_true")
    p_query.add_argument("--metadata", default=paths["metadata_parquet"])
    p_query.set_defaults(func=cmd_query)

    # trake
    p_trake = subparsers.add_parser("trake", parents=[base_parser])
    p_trake.add_argument("--stages", nargs="+", required=True)
    p_trake.add_argument("--query", type=str, default=None)
    p_trake.add_argument("--video-id", dest="video_id", type=str, default=None)
    p_trake.add_argument("--out-csv", dest="out_csv", type=str, default=None)
    p_trake.add_argument("--search-top-n", dest="search_top_n", type=int, default=int(cfg["phase4"].get("trake_search_top_n", 50)))
    p_trake.add_argument("--metadata", default=paths["metadata_parquet"])
    p_trake.set_defaults(func=cmd_trake)

    # shell
    p_shell = subparsers.add_parser("shell", parents=[base_parser])
    p_shell.add_argument("--metadata", default=paths["metadata_parquet"])
    p_shell.set_defaults(func=cmd_shell)

    # pipeline
    p_pipeline = subparsers.add_parser("pipeline", parents=[base_parser])
    p_pipeline.add_argument("--videos-dir", default=paths["raw_videos_dir"])
    p_pipeline.add_argument("--video-info-dir", default=paths["video_info_dir"])
    p_pipeline.add_argument("--keyframes-dir", default=paths["keyframes_dir"])
    p_pipeline.add_argument("--output-csv", default=paths["keyframe_map_csv"])
    p_pipeline.add_argument("--no-recursive", action="store_true")
    p_pipeline.add_argument("--keyframe-map", default=paths["keyframe_map_csv"])
    p_pipeline.add_argument("--output-parquet", default=paths["metadata_parquet"])
    p_pipeline.add_argument("--metadata", default=paths["metadata_parquet"])
    p_pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "wandb") and args.wandb:
        import wandb
        print("\n ---Đang thiết lập Weights & Biases...---")
        
        os.environ["WANDB_API_KEY"] = args.wandb.strip()
        wandb.login()
        
        init_kwargs = {
            "project": args.wandb_project,
            "name": f"run_{args.command}"
        }
        
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