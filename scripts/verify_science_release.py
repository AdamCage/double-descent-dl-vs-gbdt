"""Verify delivered figure bytes and replay frozen R10 without model fitting."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_TABLE_ROWS = {
    "r10_rates": 72, "r10_scan_scores": 7188,
    "r10_localization_flagged": 6020, "r10_example_trajectories": 30720,
    "r11_primary_contrasts": 4, "r11_clock_diagnostic_contrasts": 4,
    "r11_all_trajectory_metrics": 55512,
    "r12_scan_score_contrasts": 4, "r12_fixed_boundary_gain_contrasts": 4,
    "r12_fixed_late_acceleration_gates": 6, "r12_all_trajectory_metrics": 221400,
    "r12_first_zero_errors": 72, "r12_descriptive_diagnostics": 144,
    "r2_existing_confirmation_cells": 1440,
    "r3_existing_confirmation_cells": 145, "r4_existing_confirmation_cells": 155,
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command(argv: list[str], cwd: Path) -> dict:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True,
                            encoding="utf-8", errors="replace",
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES=""))
    record = {"argv": argv, "exit_code": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise RuntimeError(json.dumps(record))
    return record


def confined_file(root: Path, relative: str, directory: str) -> Path:
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative
            or not relative.startswith(directory + "/")
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Unsafe artifact path")
    path = root / relative
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("Artifact paths cannot contain symlinks/junctions")
        cursor = cursor.parent
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact is not a confined regular file")
    return path


def verify_gallery(gallery: Path, *, expected_static: int = 187, expected_interactive: int = 2,
                   expected_tables: dict[str, int] | None = None) -> dict:
    expected_tables = EXPECTED_TABLE_ROWS if expected_tables is None else expected_tables
    manifest = json.loads((gallery / "figure_manifest.json").read_text(encoding="utf-8"))
    rows = manifest["figures"]
    count = expected_static + expected_interactive
    if len(rows) != count or manifest.get("figure_count") != count or len({r["id"] for r in rows}) != count:
        raise ValueError("Incomplete or duplicate figure IDs")
    static = interactive = 0
    expected = set()
    for row in rows:
        if set(row["paths"]) not in ({"png", "svg"}, {"html"}):
            raise ValueError("Each entry must be one PNG/SVG pair or one HTML")
        if set(row["sha256"]) != set(row["paths"]):
            raise ValueError("Figure hash kinds differ from path kinds")
        for kind, relative in row["paths"].items():
            if relative in expected:
                raise ValueError("Reused figure output path")
            path = confined_file(gallery, relative, "figures")
            if path.suffix != "." + kind:
                raise ValueError("Figure extension differs from declared kind")
            if digest(path) != row["sha256"][kind]:
                raise ValueError(f"Figure hash mismatch: {relative}")
            expected.add(relative)
            if kind == "png":
                with Image.open(path) as bitmap:
                    if bitmap.width < 600 or bitmap.height < 300:
                        raise ValueError(f"Undersized figure: {relative}")
                    bitmap.verify()
            elif kind == "svg":
                if not ET.parse(path).getroot().tag.endswith("svg"):
                    raise ValueError(f"Invalid SVG: {relative}")
            elif kind == "html":
                if "<html" not in path.read_text(encoding="utf-8").lower():
                    raise ValueError(f"Invalid HTML: {relative}")
        static += "png" in row["paths"] and "svg" in row["paths"]
        interactive += "html" in row["paths"]
    actual = {p.relative_to(gallery).as_posix() for p in (gallery / "figures").rglob("*") if p.is_file() or p.is_symlink()}
    if actual != expected or static != expected_static or interactive != expected_interactive:
        raise ValueError(f"Incomplete figure inventory: {static} static, {interactive} interactive")
    tables = manifest["tables"]
    if not expected_tables or len(tables) != len(expected_tables):
        raise ValueError("Incomplete table inventory")
    expected_paths = {f"tables/{name}.csv" for name in expected_tables}
    table_paths = [table["path"] for table in tables]
    if len(set(table_paths)) != len(tables) or set(table_paths) != expected_paths:
        raise ValueError("Missing, duplicate or unexpected source table")
    for table in tables:
        path = confined_file(gallery, table["path"], "tables")
        if digest(path) != table["sha256"]:
            raise ValueError("Table hash mismatch")
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("Source table has no columns")
            table_count = sum(1 for _ in reader)
        if table_count != expected_tables[path.stem] or table.get("rows") != table_count:
            raise ValueError("Source table has an incomplete row matrix")
    actual_tables = {p.relative_to(gallery).as_posix() for p in (gallery / "tables").rglob("*") if p.is_file() or p.is_symlink()}
    if actual_tables != expected_paths:
        raise ValueError("Unregistered source table files")
    if not (gallery / "index.html").is_file() or not (gallery / "index.html").stat().st_size:
        raise ValueError("Missing gallery index")
    return {"static_png_svg_pairs": static, "interactive_html": interactive, "source_tables": len(tables)}


def check_replayed_values(original: np.ndarray, replay: np.ndarray) -> list[int]:
    if original.shape != (7188, 256) or replay.shape != (7188, 256):
        raise ValueError("R10 replay requires the complete 7188x256 matrix")
    if not np.isfinite(original).all() or not np.isfinite(replay).all():
        raise ValueError("R10 replay contains nonfinite values")
    np.testing.assert_array_equal(original, replay)
    return list(original.shape)


def verify(gallery: Path, package: Path, original: Path, output: Path) -> dict:
    figures = verify_gallery(gallery)
    commands = [command([sys.executable, str(package / "verify.py")], output.parent)]
    replay = output.parent / "r10-portable-replay"
    commands.append(command([sys.executable, str(package / "replay.py"), "R10", "--output", str(replay)], output.parent))
    with np.load(original / "curves.npz", allow_pickle=False) as a, np.load(replay / "curves.npz", allow_pickle=False) as b:
        shape = check_replayed_values(a["loss"], b["loss"])
    for name in ("analysis.json", "calibration.json"):
        if json.loads((original / name).read_text()) != json.loads((replay / name).read_text()):
            raise ValueError(f"Portable R10 numerical mismatch: {name}")
    if digest(original / "curves.jsonl") != digest(replay / "curves.jsonl"):
        raise ValueError("Portable R10 curve metadata mismatch")
    commands.append(command([sys.executable, str(package / "verify.py")], output.parent))
    return {"verified": True, **figures, "replayed_r10_array_shape": shape,
            "raw_values_exactly_equal": True, "package_unchanged_after_replay": True,
            "commands": commands, "model_fits": 0, "download_bytes": 0,
            "independent_scientific_replication": False, "scientific_validity": "not_assessed"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gallery", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--original-r10", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    try:
        result = verify(args.gallery.resolve(), args.package.resolve(), args.original_r10.resolve(), args.output)
    except Exception as exc:
        result = {"verified": False, "error": f"{type(exc).__name__}: {exc}", "scientific_validity": "not_assessed"}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "commands"}))
    raise SystemExit(0 if result["verified"] else 1)
