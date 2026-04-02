"""Double Descent Research Pipeline — entry point."""

import argparse
import sys
import time

from src.config import get_settings
from src.logging_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Double Descent Research Pipeline")
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

    cfg = get_settings()
    if args.mode:
        cfg.MODE = args.mode

    logger = setup_logging()
    logger.info("Pipeline started  mode=%s  seed=%d", cfg.MODE, cfg.SEED)

    from src.stages.s01_data import run as run_s01
    from src.stages.s02_eda import run as run_s02
    from src.stages.s03_embeddings import run as run_s03
    from src.stages.s04_embedding_analysis import run as run_s04
    from src.stages.s05_dnn_train import run as run_s05
    from src.stages.s06_dnn_metrics import run as run_s06
    from src.stages.s07_catboost_train import run as run_s07
    from src.stages.s08_catboost_metrics import run as run_s08
    from src.stages.s09_comparison import run as run_s09
    from src.stages.s10_meta import run as run_s10

    stages = {
        "s01": ("01 Data loading", run_s01),
        "s02": ("02 EDA", run_s02),
        "s03": ("03 Embeddings", run_s03),
        "s04": ("04 Embedding analysis", run_s04),
        "s05": ("05 DNN training", run_s05),
        "s06": ("06 DNN metrics", run_s06),
        "s07": ("07 CatBoost training", run_s07),
        "s08": ("08 CatBoost metrics", run_s08),
        "s09": ("09 Comparison", run_s09),
        "s10": ("10 Metadata", run_s10),
    }

    timings: dict[str, float] = {}
    ctx: dict = {}

    if args.stage:
        key = args.stage if args.stage.startswith("s") else f"s{int(args.stage):02d}"
        if key not in stages:
            logger.error("Unknown stage: %s", args.stage)
            sys.exit(1)
        label, fn = stages[key]
        logger.info("=== %s ===", label)
        t0 = time.perf_counter()
        fn(cfg, logger, ctx=ctx)
        timings[key] = time.perf_counter() - t0
        logger.info("Stage %s finished in %.1fs", key, timings[key])
    else:
        for key, (label, fn) in stages.items():
            logger.info("=== %s ===", label)
            t0 = time.perf_counter()
            fn(cfg, logger, ctx=ctx)
            timings[key] = time.perf_counter() - t0
            logger.info("Stage %s finished in %.1fs", key, timings[key])

    ctx["timings"] = timings
    logger.info("Pipeline completed. Total %.1fs", sum(timings.values()))


if __name__ == "__main__":
    main()
