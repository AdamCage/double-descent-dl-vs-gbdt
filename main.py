"""Historical text pipeline. New research uses scripts/research.py."""

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Double Descent Research Pipeline")
    parser.add_argument("--legacy", action="store_true", help="Explicitly replay the historical text pipeline")
    parser.add_argument("--result-dir", help="Isolated destination under artifacts/legacy-replays/<name>")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default=None,
        help="Override MODE from .env (fast = first 100 rows)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Run a single stage (e.g. 's01' or '1'). Omit to run all.",
    )
    args = parser.parse_args()

    if not args.legacy:
        parser.error("Historical pipeline is archived. Use python scripts/research.py status or next. "
                     "A deliberate replay requires --legacy --stage and --result-dir.")
    if not args.stage or not args.result_dir:
        parser.error("A legacy replay requires an explicit --stage and --result-dir; running all stages is retired.")
    root = Path(__file__).resolve().parent
    destination = Path(args.result_dir).resolve()
    replay_root = (root / "artifacts" / "legacy-replays").resolve()
    if not destination.is_relative_to(replay_root) or destination == replay_root:
        parser.error("--result-dir must be a named directory inside artifacts/legacy-replays/; result/ is historical evidence.")

    # Delay scientific imports until the archival guard has accepted the request.
    from src.config import get_settings
    from src.logging_setup import setup_logging

    cfg = get_settings()
    cfg.RESULT_DIR = str(destination)
    if args.mode:
        cfg.MODE = args.mode

    logger = setup_logging()
    cfg.result_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(cfg.result_path / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                                                "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)
    logger.info("Pipeline started  mode=%s  seed=%d", cfg.MODE, cfg.SEED)

    # Import a stage only when it is selected. s03 must not load torch or CatBoost.
    stages = {
        "s01": ("01 Data loading", "src.stages.s01_data"),
        "s02": ("02 EDA", "src.stages.s02_eda"),
        "s03": ("03 Embeddings", "src.stages.s03_embeddings"),
        "s04": ("04 Embedding analysis", "src.stages.s04_embedding_analysis"),
        "s05": ("05 DNN training", "src.stages.s05_dnn_train"),
        "s06": ("06 DNN metrics", "src.stages.s06_dnn_metrics"),
        "s07": ("07 CatBoost training", "src.stages.s07_catboost_train"),
        "s08": ("08 CatBoost metrics", "src.stages.s08_catboost_metrics"),
        "s09": ("09 Comparison", "src.stages.s09_comparison"),
        "s10": ("10 Metadata", "src.stages.s10_meta"),
    }

    timings: dict[str, float] = {}
    ctx: dict = {}

    try:
        if args.stage:
            key = args.stage if args.stage.startswith("s") else f"s{int(args.stage):02d}"
            if key not in stages:
                logger.error("Unknown stage: %s", args.stage)
                sys.exit(1)
            selected = {key: stages[key]}
        else:
            selected = stages
        for key, (label, module_path) in selected.items():
            logger.info("=== %s ===", label)
            t0 = time.perf_counter()
            fn = importlib.import_module(module_path).run
            fn(cfg, logger, ctx=ctx)
            timings[key] = time.perf_counter() - t0
            logger.info("Stage %s finished in %.1fs", key, timings[key])
    except Exception:
        logger.exception("Pipeline failed")
        raise

    ctx["timings"] = timings
    logger.info("Pipeline completed. Total %.1fs", sum(timings.values()))


if __name__ == "__main__":
    main()
