"""Build a standalone R10/R11/R12 scientific replay package without model fits.

Attempts remain complete evidence copies. Runnable scientific snapshots are
kept separately per stage, never combined into a supposedly shared code version.
Dependency metadata is inspected locally; this builder makes no network calls.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "package-manifest.json"
MAX_FILES = 50000
MAX_BYTES = 8 * 1024 ** 3
STAGES = {
    "R10": {
        "runner": "scripts/r10_calibrate.py",
        "modules": ["src/acceleration_audit.py", "src/detector_catalog.py", "src/dynamics.py"],
        "tests": ["tests/test_acceleration_audit.py"],
        "roots": ["numpy", "scipy"],
    },
    "R11": {
        "runner": "scripts/r11_run.py",
        "modules": ["src/acceleration_boosting.py"],
        "tests": ["tests/test_acceleration_boosting.py"],
        "roots": ["numpy", "scipy", "scikit-learn", "catboost"],
    },
    "R12": {
        "runner": "scripts/r12_run.py",
        "modules": ["src/acceleration_randomness.py", "src/acceleration_boosting.py",
                    "src/acceleration_audit.py", "src/detector_catalog.py", "src/dynamics.py"],
        "tests": ["tests/test_acceleration_randomness.py"],
        "roots": ["numpy", "scipy", "scikit-learn", "catboost"],
    },
}
SUPPLEMENTS = (
    "docs/methods/acceleration-identifiability.md",
    "docs/methods/acceleration-identifiability-review.json",
    "docs/literature/2026-11-07-v2-audit.md",
    "docs/decisions/0004-v2-research-refactor.md",
    "docs/reproducibility-v2.md",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"Unsafe relative package path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or PureWindowsPath(value).drive or any(
            part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Unsafe relative package path: {value!r}")
    for part in pure.parts:
        lowered = part.casefold()
        if lowered in (".env", ".git", ".venv") or lowered.startswith(".env."):
            raise ValueError(f"Credential/environment/VCS paths cannot be packaged: {value!r}")
    return pure.as_posix()


def checked_absolute(path: Path) -> Path:
    result = Path(os.path.abspath(path))
    for part in (result, *result.parents):
        if part.is_symlink() or (hasattr(part, "is_junction") and part.is_junction()):
            raise ValueError(f"Symlink or junction cannot be packaged: {part}")
    return result


def file_tree(root: Path) -> list[tuple[str, Path]]:
    root = checked_absolute(root)
    if not root.is_dir():
        raise ValueError(f"Expected a directory: {root}")
    files: list[tuple[str, Path]] = []
    pending, seen = [root], 0
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                seen += 1
                if seen > MAX_FILES:
                    raise ValueError("Package input exceeds the entry-count cap")
                path = Path(entry.path)
                relative = safe_relative(path.relative_to(root).as_posix())
                checked_absolute(path)
                if entry.is_file(follow_symlinks=False):
                    files.append((relative, path))
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                else:
                    raise ValueError(f"Nonregular evidence entry: {path}")
    return sorted(files)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def bound_file(root: Path, relative: str, expected: str) -> Path:
    path = checked_absolute(root / safe_relative(relative))
    if not path.is_file() or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Missing or invalid evidence binding: {relative}")
    if digest(path) != expected:
        raise ValueError(f"Evidence binding changed: {relative}")
    return path


def frozen_argument(value: str, inputs: dict[str, str]) -> str:
    normalized = value.replace("\\", "/")
    matches = [name for name in inputs if normalized == name or normalized.endswith("/" + name)]
    if len(matches) != 1:
        raise ValueError(f"Runner argument is not one uniquely frozen input: {value!r}")
    return matches[0]


def inspect_attempt(stage: str, attempt: Path) -> dict[str, Any]:
    attempt = checked_absolute(attempt)
    tree = file_tree(attempt)
    manifest = load_object(attempt / "manifest.json")
    plan = load_object(attempt / "plan.json")
    if manifest.get("stage_id") != stage or plan.get("stage_id") != stage:
        raise ValueError(f"Stage/attempt identity mismatch for {stage}")
    if manifest.get("status") != "complete":
        raise ValueError(f"Select a complete attempt for runnable stage {stage}; preserve failed attempts separately")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", attempt.name):
        raise ValueError("Attempt directory name is not a safe identifier")
    if manifest.get("attempt_id", attempt.name) != attempt.name:
        raise ValueError("Manifest attempt_id disagrees with its directory")
    if manifest.get("plan_sha256") != digest(attempt / "plan.json"):
        raise ValueError("Frozen machine plan hash changed")
    inputs = manifest.get("input_hashes")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Attempt needs a nonempty source-input inventory")
    for name, expected in inputs.items():
        bound_file(attempt / "inputs", name, expected)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Attempt needs recorded task evidence")
    runner = STAGES[stage]["runner"]
    selected = []
    for task in tasks:
        if task.get("status") != "complete" or task.get("exit_code") != 0 or task.get("timed_out"):
            raise ValueError("Selected attempt contains an incomplete/failed task")
        cwd = safe_relative(task.get("cwd", ""))
        hashes = task.get("file_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("Recorded task is missing its complete file inventory")
        actual = {name for name, _ in file_tree(attempt / cwd)}
        if actual != set(hashes):
            raise ValueError("Recorded task inventory does not cover exactly the evidence files")
        for name, expected in hashes.items():
            bound_file(attempt / cwd, name, expected)
        outputs = task.get("output_hashes")
        if not isinstance(outputs, dict) or not outputs or any(hashes.get(k) != v for k, v in outputs.items()):
            raise ValueError("Declared output hashes disagree with task files")
        argv = task.get("argv", [])
        if any(isinstance(arg, str) and arg.replace("\\", "/").endswith("/" + runner) for arg in argv):
            selected.append(task)
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one recorded {runner} task")
    task = selected[0]
    argv = task["argv"]
    if argv.count("--protocol") != 1:
        raise ValueError("Scientific replay needs one frozen protocol argument")
    protocol = frozen_argument(argv[argv.index("--protocol") + 1], inputs)
    expected_protocol = f"research/stages/{stage}/inputs/protocol.json"
    if protocol != expected_protocol:
        raise ValueError("Scientific protocol path differs from the declared stage allowlist")
    calibration = None
    if "--calibration" in argv:
        if stage != "R12" or argv.count("--calibration") != 1:
            raise ValueError("Unexpected calibration argument")
        calibration = frozen_argument(argv[argv.index("--calibration") + 1], inputs)
    required = ["src/__init__.py", runner, *STAGES[stage]["modules"],
                *STAGES[stage]["tests"], f"research/stages/{stage}/PLAN.md", protocol, "uv.lock"]
    optional = ["scripts/run_unit_checks.py", "tests/__init__.py",
                f"research/stages/{stage}/inputs/design-review.json"]
    if any(name not in inputs for name in required):
        raise ValueError(f"Incomplete runnable source closure for {stage}: {sorted(set(required) - set(inputs))}")
    code_files = sorted(required + [name for name in optional if name in inputs])
    for name in code_files:
        if not name.endswith(".py"):
            continue
        parsed = ast.parse((attempt / "inputs" / name).read_text(encoding="utf-8-sig"), filename=name)
        for node in ast.walk(parsed):
            imports = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                       [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
            for module in imports:
                if module.startswith("src."):
                    source = module.replace(".", "/") + ".py"
                    if source not in code_files:
                        raise ValueError(f"Scientific import {module} is outside the frozen allowlist for {stage}")
    return {"stage": stage, "attempt": attempt, "tree": tree, "manifest": manifest,
            "code_files": code_files, "protocol": protocol, "calibration": calibration,
            "source_task": task["id"], "original_argv": argv}


def dependency_closure(roots: list[str]) -> dict[str, Any]:
    environment = default_environment()
    environment["extra"] = ""
    records: dict[str, Any] = {}
    pending = list(roots)
    while pending:
        name = canonicalize_name(pending.pop())
        if name in records:
            continue
        installed = metadata.distribution(name)
        active = []
        for raw in installed.requires or []:
            requirement = Requirement(raw)
            if requirement.marker is not None and not requirement.marker.evaluate(environment):
                continue
            dependency = metadata.distribution(requirement.name)
            if requirement.specifier and not requirement.specifier.contains(dependency.version, prereleases=True):
                raise ValueError(f"Installed dependency violates metadata: {installed.metadata['Name']}: {raw}")
            active.append(raw)
            pending.append(requirement.name)
        records[name] = {"name": installed.metadata["Name"], "version": installed.version,
                         "active_base_requirements": sorted(active)}
    if any(name in records for name in ("epistemeos", "torch")):
        raise ValueError("Unexpected research-framework or torch dependency in minimal replay closure")
    return {"schema_version": 1, "source": "Installed importlib.metadata; no network resolution",
            "roots": sorted(set(map(canonicalize_name, roots))), "extras": [],
            "marker_environment": environment, "python": platform.python_version(),
            "packages": [records[name] for name in sorted(records)],
            "scope": "Exact versions for this Python/platform's active base dependency closure; not wheel hashes or a cross-platform binary-reproducibility promise."}


VERIFY_SCRIPT = r'''"""Check package inventory and all bytes without scientific dependencies."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys

def safe(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe package path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or PureWindowsPath(value).drive or any(p in ("", ".", "..") for p in value.split("/")):
        raise ValueError("Unsafe package path")
    if any(p.casefold() in (".env", ".git", ".venv") or p.casefold().startswith(".env.") for p in pure.parts):
        raise ValueError("Credential/environment/VCS entry in package")
    return value

def verify(root):
    root = Path(root).absolute()
    for path in (root, *root.parents):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise ValueError("Package path traverses a symlink or junction")
    manifest_path = root / "package-manifest.json"
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = document.get("files", [])
    if document.get("schema_version") != 1 or not rows:
        raise ValueError("Invalid package manifest")
    expected = {}
    for row in rows:
        name = safe(row["path"])
        if name == "package-manifest.json" or name in expected:
            raise ValueError("Duplicate or reserved package inventory entry")
        expected[name] = row
    actual = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in dirs + files:
            path = base / name
            relative = safe(path.relative_to(root).as_posix())
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("Symlink/junction in package")
            if path.is_file():
                actual.add(relative)
            elif not path.is_dir():
                raise ValueError("Nonregular package entry")
    if actual != set(expected) | {"package-manifest.json"}:
        raise ValueError("Package file inventory changed")
    total = 0
    for name, row in expected.items():
        path = root / name
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        if path.stat().st_size != row["bytes"] or h.hexdigest() != row["sha256"]:
            raise ValueError("Package bytes changed: " + name)
        total += row["bytes"]
    return {"verified": True, "files": len(expected), "bytes": total, "scientific_validity": "not_assessed"}

if __name__ == "__main__":
    try:
        print(json.dumps(verify(Path(__file__).absolute().parent)))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
'''


REPLAY_SCRIPT = r'''"""Replay one frozen scientific stage into a new directory outside the package."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("R10", "R11", "R12"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 and the packaged requirements.txt")
    root = Path(__file__).absolute().parent
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, str(root / "verify.py")], env=env, check=True)
    document = json.loads((root / "package-manifest.json").read_text(encoding="utf-8"))
    entry = document["stages"].get(args.stage)
    if entry is None:
        parser.error("That stage was not included in this package")
    output = Path(os.path.abspath(args.output))
    for path in (output, *output.parents):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            parser.error("Output paths cannot traverse symlinks or junctions")
    if output == root or root in output.parents:
        parser.error("Replay output must be outside the verified package")
    if output.exists():
        parser.error("Replay output must be a new directory; previous outputs are retained")
    code = root / entry["code_root"]
    argv = [sys.executable, str(code / entry["runner"]), "--protocol", str(code / entry["protocol"])]
    if entry["calibration"] is not None:
        argv += ["--calibration", str(root / entry["calibration"])]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    # Import caches would alter the verified code copy; outputs and logs stay here.
    with (output / "replay-stdout.log").open("xb") as stdout, (output / "replay-stderr.log").open("xb") as stderr:
        result = subprocess.run(argv, cwd=output, env=env, stdout=stdout, stderr=stderr)
    record = {"stage": args.stage, "source_attempt": entry["attempt_id"], "argv": argv,
              "python": sys.version, "exit_code": result.returncode,
              "scientific_validity": "not_assessed",
              "note": "A replay is a new execution, not automatically an independent scientific replication or proof of the original claims."}
    with (output / "replay-record.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"stage": args.stage, "output": str(output), "exit_code": result.returncode}))
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
'''


def build_package(runs: dict[str, Path], manuscript: Path, output: Path,
                  figures: Path | None = None, *, review_packets: Path | None = None,
                  source_root: Path = ROOT) -> dict[str, Any]:
    if not runs or set(runs) - set(STAGES):
        raise ValueError("Choose at least one of R10, R11, R12")
    output, manuscript = checked_absolute(output), checked_absolute(manuscript)
    if output.exists():
        raise ValueError("Package output must be a new path; previous packages are never overwritten")
    if not manuscript.is_file():
        raise ValueError("A current manuscript file is required")
    specs = [inspect_attempt(stage, path) for stage, path in sorted(runs.items())]
    dependencies = dependency_closure([root for stage in runs for root in STAGES[stage]["roots"]]
                                      + (["matplotlib", "plotly"] if figures is not None else []))
    for spec in specs:
        recorded = {canonicalize_name(k): v for k, v in
                    spec["manifest"].get("environment", {}).get("packages", {}).items()}
        for distribution in dependencies["packages"]:
            name = canonicalize_name(distribution["name"])
            if name in recorded and recorded[name] != distribution["version"]:
                raise ValueError(f"Installed dependency version differs from {spec['stage']} evidence: {name}")
    copies: list[tuple[Path, str]] = [(manuscript, "paper/" + safe_relative(manuscript.name))]
    stages = {}
    input_roots = [spec["attempt"] for spec in specs]
    for spec in specs:
        stage, attempt = spec["stage"], spec["attempt"]
        evidence_root = f"evidence/{stage}/{attempt.name}"
        copies += [(path, f"{evidence_root}/{name}") for name, path in spec["tree"]]
        copies += [(attempt / "inputs" / name, f"code/{stage}/{name}") for name in spec["code_files"]]
        stages[stage] = {"attempt_id": attempt.name, "evidence_root": evidence_root,
                         "code_root": f"code/{stage}", "runner": STAGES[stage]["runner"],
                         "protocol": spec["protocol"], "source_task": spec["source_task"],
                         "calibration": None if spec["calibration"] is None else f"{evidence_root}/inputs/{spec['calibration']}",
                         "original_argv": spec["original_argv"],
                         "source_files": [{"path": name, "sha256": spec["manifest"]["input_hashes"][name]}
                                          for name in spec["code_files"]],
                         "review_present": (attempt / "review.json").is_file()}
    if figures is not None:
        figures = checked_absolute(figures)
        input_roots.append(figures)
        copies += [(path, "figures/" + name) for name, path in file_tree(figures)]
        plotter = checked_absolute(source_root / "scripts/plot_v2_research.py")
        if plotter.is_file():
            copies.append((plotter, "supplement/scripts/plot_v2_research.py"))
    if review_packets is not None:
        review_packets = checked_absolute(review_packets)
        input_roots.append(review_packets)
        copies += [(path, "reviews/" + name) for name, path in file_tree(review_packets)]
    for name in SUPPLEMENTS:
        source = checked_absolute(source_root / name)
        if source.is_file():
            copies.append((source, "supplement/" + name))
    if any(output == source or source in output.parents for source in input_roots):
        raise ValueError("Package output must be outside input/evidence directories")
    destinations = [safe_relative(relative) for _, relative in copies]
    if len(destinations) != len(set(destinations)):
        raise ValueError("Package destinations collide")
    if len(copies) + 6 > MAX_FILES or sum(path.stat().st_size for path, _ in copies) > MAX_BYTES:
        raise ValueError("Package exceeds its file-count or byte cap")
    # All preflight checks precede creation. Failed copies stay visible for audit.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    for source, relative in copies:
        expected = digest(source)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as destination:
            for block in iter(lambda: incoming.read(1024 * 1024), b""):
                destination.write(block)
        if digest(target) != expected or digest(source) != expected:
            raise ValueError(f"Source changed while copying: {source}")
    requirements = ["# Python 3.12; exact active base dependency closure from installed metadata.",
                    "# Original per-stage uv.lock targets the main repository and its EpistemeOS dependency."]
    requirements += [f"{entry['name']}=={entry['version']}" for entry in dependencies["packages"]]
    commands = "\n".join(f"python replay.py {stage} --output ../replay-{stage.lower()}" for stage in stages)
    readme = f"""# Scientific reproduction package\n\nPython 3.12 is required. Create the environment outside this package, install\n`requirements.txt`, then run `python verify.py`. Example:\n\n```text\npython3.12 -m venv ../science-env\n../science-env/bin/python -m pip install -r requirements.txt\n../science-env/bin/python verify.py\n{commands}\n```\n\nUse the environment's Python for every replay command. On Windows use\n`py -3.12 -m venv ..\\science-env` and `..\\science-env\\Scripts\\python.exe`.\n\n`evidence/` contains complete selected attempts: frozen inputs, machine plans,\nlogs, raw arrays, metrics, saved models and any bound report/review. `code/Rxx/`\nis a separate scientific allowlist copied from that attempt's frozen inputs;\nsource versions from different stages are never merged. Archived workflow\ncontrols remain provenance in evidence; the minimal replay never imports them.\n\nEach original `uv.lock` is retained verbatim for provenance. It describes the\nfull main repository, including EpistemeOS, and is **not** the installation\nroute for this standalone package. `requirements.txt` instead pins the locally\nobserved active base dependency closure, with no requested extras and no network\nresolution by the builder. It contains neither EpistemeOS nor torch. Platform\nmarkers were evaluated in the environment recorded in `dependency-lock.json`;\nwheels, OS and hardware can differ elsewhere. Bitwise agreement is not promised.\n\nReplay verifies every packaged file first and requires a new output path outside\nthe package. It executes the frozen scientific runner with absolute packaged\nprotocol/calibration paths; imports cannot write caches into the package. A\nfailed replay keeps its output, stdout/stderr and exit record. Replays are new\nexecutions, not automatically independent scientific replications.\n\nOptional figures, manuscript and supplements are delivery artifacts; they are\nnot silently promoted to frozen experimental inputs. The plotting source, when\npresent, is under `supplement/scripts/`; any secondary historical controls need\ntheir separately archived evidence to regenerate. Local reviews and hashes\nprovide integrity bindings, not external novelty certification or peer review.\nThe manifest covers every file except itself; use the separately recorded\narchive/manifest digest as the external trust anchor.\n"""
    generated = {"requirements.txt": "\n".join(requirements) + "\n", "verify.py": VERIFY_SCRIPT,
                 "replay.py": REPLAY_SCRIPT, "README.md": readme,
                 "dependency-lock.json": json.dumps(dependencies, indent=2) + "\n"}
    for name, content in generated.items():
        with (output / name).open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    files = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
             for name, path in file_tree(output)]
    document = {"schema_version": 1, "python_requirement": "3.12", "stages": stages, "files": files,
                "scientific_validity": "not_assessed", "external_novelty_certification": False,
                "manifest_scope": "Every packaged file except this manifest; no identity authentication.",
                "source_policy": "Complete attempts plus separate per-stage frozen scientific code; no whole EpistemeOS checkout.",
                "dependency_policy": "Installed exact base closure; original full-project uv.lock retained only as provenance.",
                "builder": {"path": "scripts/build_science_package.py", "sha256": digest(Path(__file__))}}
    with (output / MANIFEST).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    result = subprocess.run([sys.executable, str(output / "verify.py")], text=True, capture_output=True,
                            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    if result.returncode:
        raise ValueError(f"Package verification failed: {result.stderr}")
    return {"package": str(output), "stages": sorted(stages), "files": len(files),
            "bytes": sum(row["bytes"] for row in files), "verified": True,
            "manifest_sha256": digest(output / MANIFEST), "scientific_validity": "not_assessed"}


def self_test() -> dict[str, Any]:
    """Exercise packaging and replay using tiny fake runners; never fit a model."""
    with tempfile.TemporaryDirectory(prefix="science-package-fixture-") as directory:
        root = Path(directory)
        runs = {}
        for stage in ("R10", "R11", "R12"):
            attempt = root / "attempts" / stage / f"fixture-{stage.lower()}"
            inputs = attempt / "inputs"
            protocol = f"research/stages/{stage}/inputs/protocol.json"
            names = ["src/__init__.py", STAGES[stage]["runner"], *STAGES[stage]["modules"],
                     *STAGES[stage]["tests"], f"research/stages/{stage}/PLAN.md", protocol, "uv.lock"]
            calibration = f"research/stages/{stage}/inputs/calibration.json"
            if stage == "R12":
                names.append(calibration)
            for name in names:
                target = inputs / name
                target.parent.mkdir(parents=True, exist_ok=True)
                value = "# Fake evidence fixture; not a scientific result.\n"
                if name == "src/__init__.py":
                    value += f"SNAPSHOT = '{stage}'\n"
                if name == protocol:
                    value = json.dumps({"fixture": stage})
                if stage == "R12" and name == calibration:
                    value = json.dumps({"fixture_calibration": True})
                if name == STAGES[stage]["runner"]:
                    value = ("import argparse,json,sys\nfrom pathlib import Path\n"
                             "sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import src\n"
                             "p=argparse.ArgumentParser();p.add_argument('--protocol',required=True)\n"
                             "p.add_argument('--calibration')\n"
                             "a=p.parse_args(); d=json.loads(Path(a.protocol).read_text())\n"
                             "if a.calibration: d['calibration']=json.loads(Path(a.calibration).read_text())\n"
                             "Path('fixture-output.json').write_text(json.dumps(d))\n")
                target.write_text(value, encoding="utf-8")
            task = attempt / "tasks/science"
            task.mkdir(parents=True)
            for name in ("metrics.json", "stdout.log", "stderr.log"):
                (task / name).write_text("{}\n" if name.endswith(".json") else "", encoding="utf-8")
            plan = {"stage_id": stage, "schema_version": 1}
            (attempt / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            hashes = {name: digest(inputs / name) for name in names}
            manifest = {"stage_id": stage, "attempt_id": attempt.name, "status": "complete",
                        "input_hashes": hashes, "plan_sha256": digest(attempt / "plan.json"),
                        "tasks": [{"id": "science", "status": "complete", "exit_code": 0, "timed_out": False,
                                   "cwd": "tasks/science", "output_hashes": {"metrics.json": digest(task / "metrics.json")},
                                   "file_hashes": {name: digest(path) for name, path in file_tree(task)},
                                   "argv": [sys.executable, str(inputs / STAGES[stage]["runner"]),
                                            "--protocol", str(inputs / protocol)]}]}
            if stage == "R12":
                manifest["tasks"][0]["argv"] += ["--calibration", str(inputs / calibration)]
            (attempt / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            runs[stage] = attempt
        manuscript = root / "fixture-paper.md"
        manuscript.write_text("Fake packaging test only; not scientific evidence.\n", encoding="utf-8")
        packets = root / "review-packets"
        packets.mkdir()
        (packets / "fixture-claim.json").write_text('{"fixture":true}\n', encoding="utf-8")
        package = root / "package"
        result = build_package(runs, manuscript, package, review_packets=packets, source_root=root)
        assert load_object(package / "reviews/fixture-claim.json") == {"fixture": True}
        assert (package / "code/R10/src/__init__.py").read_bytes() != (package / "code/R11/src/__init__.py").read_bytes()
        assert not (package / "code/R10/src/research_workflow.py").exists()
        assert "torch==" not in (package / "requirements.txt").read_text(encoding="utf-8")
        replay = subprocess.run([sys.executable, str(package / "replay.py"), "R10", "--output", str(root / "replayed")],
                                text=True, capture_output=True)
        if replay.returncode:
            raise AssertionError(replay.stderr)
        assert load_object(root / "replayed/fixture-output.json") == {"fixture": "R10"}
        replay12 = subprocess.run([sys.executable, str(package / "replay.py"), "R12", "--output", str(root / "replayed12")],
                                  text=True, capture_output=True)
        assert replay12.returncode == 0, replay12.stderr
        assert load_object(root / "replayed12/fixture-output.json") == {
            "fixture": "R12", "calibration": {"fixture_calibration": True}}
        duplicate = subprocess.run([sys.executable, str(package / "replay.py"), "R10", "--output", str(root / "replayed")],
                                   text=True, capture_output=True)
        assert duplicate.returncode != 0
        verify = subprocess.run([sys.executable, str(package / "verify.py")], text=True, capture_output=True)
        assert verify.returncode == 0, verify.stderr
        try:
            build_package(runs, manuscript, package, source_root=root)
        except ValueError:
            pass
        else:
            raise AssertionError("Existing package was overwritten")
        tampered = package / "evidence/R10/fixture-r10/tasks/science/metrics.json"
        tampered.write_text("tampered fixture\n", encoding="utf-8")
        verify = subprocess.run([sys.executable, str(package / "verify.py")], text=True, capture_output=True)
        assert verify.returncode != 0
        secret = runs["R10"] / "inputs/.env"
        secret.write_text("fixture_only=true\n", encoding="utf-8")
        try:
            build_package(runs, manuscript, root / "forbidden-package", source_root=root)
        except ValueError:
            pass
        else:
            raise AssertionError("Credential-named fixture was copied")
        assert not (root / "forbidden-package").exists()
        return {"passed": True, "fixture_stages": result["stages"], "scientific_fits": 0,
                "checks": ["complete evidence copy", "separate stage versions", "minimal dependency closure",
                           "standalone fake replay", "frozen calibration replay", "explicit review packets",
                           "package integrity after replay", "exclusive outputs",
                           "tamper detection", "credential-name rejection"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="STAGE:ATTEMPT")
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--review-packets", type=Path, help="Explicit evidence/claim/review intake copied under reviews/")
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true", help="Tiny fake evidence/replay only; no scientific fits")
    args = parser.parse_args()
    try:
        if args.self_test:
            if args.run or args.figures or args.review_packets or args.manuscript or args.output:
                parser.error("--self-test cannot be combined with package inputs")
            result = self_test()
        else:
            if not args.run or args.manuscript is None or args.output is None:
                parser.error("--run, --manuscript and --output are required")
            runs = {}
            for value in args.run:
                stage, separator, attempt = value.partition(":")
                if not separator or stage not in STAGES or not attempt or stage in runs:
                    parser.error("Each --run must name a unique R10/R11/R12:attempt-path")
                runs[stage] = Path(attempt)
            result = build_package(runs, args.manuscript, args.output, args.figures,
                                   review_packets=args.review_packets)
    except (ValueError, OSError, KeyError, TypeError, metadata.PackageNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
