"""Render the complete fixed gallery from separately completed audit summaries."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--confirmation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--intake-profile", type=Path)
    args = parser.parse_args()
    from src.cot_figures import DESIGN_NOTE_PATH, FigureError, Gallery, load_sources, render
    output = args.output.resolve()
    for source in (args.discovery.resolve(), args.confirmation.resolve()):
        if output == source or source in output.parents or output in source.parents:
            raise FigureError("Gallery must be a new directory outside input summary directories")
    gallery = Gallery(output)
    gallery.source(Path(__file__))
    gallery.source(Path(__file__).resolve().parents[1] / "src" / "cot_figures.py")
    gallery.source(Path(__file__).resolve().parents[1] / "uv.lock")
    gallery.source(Path(__file__).resolve().parents[1] / DESIGN_NOTE_PATH)
    study = load_sources(gallery, args.discovery, args.confirmation, args.intake_profile)
    if len(study.models) != 8 or study.datasets != ["bbh", "gpqa", "mmlu"]:
        raise FigureError("R18 requires the frozen eight-model, three-dataset complete matrix")
    render(gallery, study)
    print(json.dumps({"output": str(output), "figure_png_svg_pairs": len(gallery.figures),
                      "csv_tables": len(gallery.tables), "offline_index": str(output / "index.html")}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
