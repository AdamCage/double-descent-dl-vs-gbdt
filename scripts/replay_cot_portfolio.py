"""Replay the frozen CoT portfolio numerics from a portable evidence directory.

No network, API, model, training, or repository imports are required. The exact
producer is loaded from evidence/src after its reference SHA256 is checked.
Passing establishes computational reproduction, not independent replication.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import sys
import time
import traceback
from unittest.mock import patch

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

ATOL = 1e-12
PHASES = ("discovery", "confirmation")
FILES = ("analysis.json", "dataquality.json", "per_question.parquet", "bootstrap.npz", "pairs.csv", "protocol.json")
SHARED_PROTOCOL_KEYS = ("models", "datasets", "context_definition", "metadata_conflict_policy", "input_sha256", "spec_sha256",
                        "metadata_audit_sha256", "split_seed", "split_modulus", "discovery_buckets", "confirmation_buckets",
                        "reserve_buckets", "bootstrap_replicates", "bootstrap_batch", "bootstrap_seed", "min_clusters",
                        "min_overlap_coverage", "practical_effect", "tie_band", "min_joint_finite_bootstrap_fraction")
_MISSING = object()


class ReplayError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    def reject_constant(value):
        raise ReplayError(f"Nonfinite JSON literal in {path.name}: {value}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def safe_value(value):
    if value is _MISSING:
        return "<missing>"
    if isinstance(value, bytes):
        return {"hex": value.hex()[:512], "bytes": len(value)}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_value(v) for v in value]
    if hasattr(value, "item"):
        return safe_value(value.item())
    if isinstance(value, str) and len(value) > 512:
        return {"prefix": value[:512], "characters": len(value)}
    return value


class Comparison:
    def __init__(self, name: str, maximum_examples: int = 100):
        self.name, self.checked, self.mismatches = name, 0, 0
        self.examples, self.maximum_examples = [], maximum_examples
        self.max_abs_float_difference = 0.

    def fail(self, path: str, reason: str, expected=_MISSING, actual=_MISSING, *, count: int = 1):
        self.mismatches += count
        if len(self.examples) < self.maximum_examples:
            self.examples.append({"path": path, "reason": reason, "expected": safe_value(expected), "actual": safe_value(actual)})

    def compare(self, expected, actual, path: str = "$"):
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in sorted(set(expected) | set(actual), key=str):
                here = f"{path}/{key}"
                if key not in expected or key not in actual:
                    self.fail(here, "key_set", expected.get(key, _MISSING), actual.get(key, _MISSING))
                else:
                    self.compare(expected[key], actual[key], here)
            return
        if isinstance(expected, (list, tuple)) and isinstance(actual, type(expected)):
            self.checked += 1
            if len(expected) != len(actual):
                self.fail(path, "length", len(expected), len(actual))
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.compare(left, right, f"{path}/{index}")
            return
        self.checked += 1
        if type(expected) is not type(actual):
            self.fail(path, "type", type(expected).__name__, type(actual).__name__)
        elif isinstance(expected, float):
            if math.isnan(expected) or math.isnan(actual):
                if not (math.isnan(expected) and math.isnan(actual)):
                    self.fail(path, "NaN_mask", expected, actual)
            elif math.isfinite(expected) and math.isfinite(actual):
                difference = abs(expected - actual)
                self.max_abs_float_difference = max(self.max_abs_float_difference, difference)
                if difference > ATOL:
                    self.fail(path, "float_absolute_tolerance", expected, actual)
            elif expected != actual:
                self.fail(path, "nonfinite_value", expected, actual)
        elif expected != actual:
            self.fail(path, "exact_value", expected, actual)

    def summary(self):
        return {"name": self.name, "status": "passed" if self.mismatches == 0 else "failed",
                "checked_values": self.checked, "mismatches": self.mismatches,
                "max_abs_float_difference": self.max_abs_float_difference, "examples": self.examples,
                "additional_mismatches_not_listed": max(0, self.mismatches - len(self.examples))}


def compare_json(reference: Path, replayed: Path, *, analysis: bool = False) -> dict:
    result = Comparison(reference.name)
    expected, actual = load_json(reference), load_json(replayed)
    if analysis:
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            raise ReplayError("analysis.json must be an object")
        # Intentionally root-only: nested scientific settings with these names are compared.
        expected = {k: v for k, v in expected.items() if k not in ("timing", "environment")}
        actual = {k: v for k, v in actual.items() if k not in ("timing", "environment")}
    result.compare(expected, actual)
    summary = result.summary()
    summary["ignored_root_keys"] = ["timing", "environment"] if analysis else []
    return summary


def compare_npz(reference: Path, replayed: Path) -> dict:
    import numpy as np
    result = Comparison(reference.name)
    with np.load(reference, allow_pickle=False) as left, np.load(replayed, allow_pickle=False) as right:
        required = {"samples", "metric_names", "cell_ids"}
        if set(left.files) != required or set(right.files) != required:
            raise ReplayError("Bootstrap archive requires exactly samples, metric_names and cell_ids")
        result.compare(sorted(left.files), sorted(right.files), "$/members")
        for name in sorted(set(left.files) & set(right.files)):
            expected, actual = left[name], right[name]
            result.compare(str(expected.dtype), str(actual.dtype), f"$/{name}/dtype")
            result.compare(list(expected.shape), list(actual.shape), f"$/{name}/shape")
            if expected.shape != actual.shape:
                continue
            result.checked += expected.size
            if expected.dtype.kind in "fc" and actual.dtype.kind in "fc":
                equal = np.isclose(expected, actual, atol=ATOL, rtol=0, equal_nan=True)
                finite = np.isfinite(expected) & np.isfinite(actual)
                if np.any(finite):
                    result.max_abs_float_difference = max(result.max_abs_float_difference,
                                                          float(np.max(np.abs(expected[finite] - actual[finite]))))
            elif expected.dtype.kind == actual.dtype.kind:
                equal = expected == actual
            else:
                result.fail(f"$/{name}", "incompatible_array_kinds", str(expected.dtype), str(actual.dtype))
                continue
            failures = np.flatnonzero(~equal)
            if failures.size:
                result.mismatches += int(failures.size)
                for flat in failures[:max(0, result.maximum_examples - len(result.examples))]:
                    index = np.unravel_index(int(flat), expected.shape)
                    result.examples.append({"path": f"$/{name}/{','.join(map(str, index))}", "reason": "array_value_or_NaN_mask",
                                            "expected": safe_value(expected[index]), "actual": safe_value(actual[index])})
    return result.summary()


def compare_parquet(reference: Path, replayed: Path) -> dict:
    import pyarrow.parquet as pq
    result = Comparison(reference.name)
    left, right = pq.ParquetFile(reference), pq.ParquetFile(replayed)
    result.checked += 1
    if not left.schema_arrow.equals(right.schema_arrow, check_metadata=True):
        result.fail("$/schema", "logical_schema_or_metadata", str(left.schema_arrow), str(right.schema_arrow))
    result.compare(left.metadata.metadata or {}, right.metadata.metadata or {}, "$/key_value_metadata")
    result.compare(left.metadata.num_rows, right.metadata.num_rows, "$/num_rows")
    def rows(parquet):
        for batch in parquet.iter_batches(batch_size=2048, use_threads=False):
            yield from batch.to_pylist()
    compared_rows = 0
    for index, (expected, actual) in enumerate(zip_longest(rows(left), rows(right), fillvalue=_MISSING)):
        if expected is _MISSING or actual is _MISSING:
            result.fail(f"$/rows/{index}", "missing_row", expected, actual)
        else:
            result.compare(expected, actual, f"$/rows/{index}")
            compared_rows += 1
    summary = result.summary()
    summary.update(compared_rows=compared_rows, physical_writer_metadata={"reference_created_by": left.metadata.created_by,
                                                                         "replay_created_by": right.metadata.created_by},
                   row_groups_are_physical_layout=True)
    return summary


def csv_schema(reference_analysis: dict) -> dict[str, str]:
    schema = dict.fromkeys(("cell_id", "dataset", "model_a", "model_b", "ineligible_reasons"), "str")
    schema.update(dict.fromkeys(("eligible", "material_effect"), "bool"))
    schema.update(dict.fromkeys(("delta_total_low", "delta_total_high"), "float"))
    for cell in reference_analysis["cells"]:
        for key in cell["estimates"]:
            schema[key] = "float"
        for key, value in cell["counts"].items():
            kind = "int" if type(value) is int else "float" if type(value) is float else None
            if kind is None or key in schema and schema[key] != kind:
                raise ReplayError(f"Ambiguous count type in reference CSV schema: {key}")
            schema[key] = kind
    return schema


def parse_csv_value(text: str, kind: str):
    if kind == "str":
        return text
    if text == "":
        if kind == "float":
            return None
        raise ReplayError(f"Missing nonnullable CSV {kind}")
    if kind == "bool":
        if text not in ("True", "False"):
            raise ReplayError("CSV booleans must be explicit True/False")
        return text == "True"
    if kind == "int":
        if not re.fullmatch(r"-?(0|[1-9][0-9]*)", text):
            raise ReplayError("CSV count must be an exact decimal integer")
        return int(text)
    if kind == "float":
        return float(text)
    raise ReplayError(f"Unknown CSV type: {kind}")


def compare_csv(reference: Path, replayed: Path, schema: dict[str, str]) -> dict:
    result = Comparison(reference.name)
    with reference.open(encoding="utf-8", newline="") as a, replayed.open(encoding="utf-8", newline="") as b:
        left, right = csv.DictReader(a), csv.DictReader(b)
        result.compare(left.fieldnames, right.fieldnames, "$/headers")
        for headers in (left.fieldnames, right.fieldnames):
            if not headers or len(set(headers)) != len(headers) or set(headers) != set(schema):
                raise ReplayError("CSV headers are missing, duplicated, or outside the frozen scientific schema")
        row_count = 0
        for index, (expected, actual) in enumerate(zip_longest(left, right, fillvalue=_MISSING)):
            row_count += 1
            if expected is _MISSING or actual is _MISSING:
                result.fail(f"$/rows/{index}", "missing_row", expected, actual)
                continue
            if None in expected or None in actual:
                raise ReplayError("CSV row has extra columns")
            for key in schema:
                try:
                    lvalue, rvalue = parse_csv_value(expected[key], schema[key]), parse_csv_value(actual[key], schema[key])
                except (ReplayError, ValueError, TypeError) as exc:
                    result.fail(f"$/rows/{index}/{key}", "typed_parse: " + str(exc), expected.get(key), actual.get(key))
                else:
                    result.compare(lvalue, rvalue, f"$/rows/{index}/{key}")
    return {**result.summary(), "rows_checked": row_count, "column_types": schema}


@contextmanager
def offline_guard():
    def denied(*_args, **_kwargs):
        raise ReplayError("Network access is forbidden during portable numerical replay")
    with patch.object(socket, "create_connection", denied), patch.object(socket, "getaddrinfo", denied), \
            patch.object(socket.socket, "connect", denied), patch.object(socket.socket, "connect_ex", denied):
        yield


def validate_bindings(evidence: Path) -> tuple[dict, dict]:
    inputs, references, protocols, hashes = evidence / "inputs", {}, {}, {}
    required = [evidence / "src" / name for name in ("__init__.py", "cot_audit.py", "cot_figures.py")]
    required += [inputs / name for name in ("slim.parquet", "download-spec.json", "metadata_audit.json", "intake_profile.json")]
    for phase in PHASES:
        required += [evidence / "protocols" / f"{phase}.json"]
        required += [evidence / "reference" / phase / name for name in FILES]
    for path in required:
        if not path.is_file():
            raise ReplayError(f"Required evidence file is absent: {path.relative_to(evidence)}")
        hashes[path.relative_to(evidence).as_posix()] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    source_hash = hashes["src/cot_audit.py"]["sha256"]
    metadata = load_json(inputs / "metadata_audit.json")
    if metadata.get("metadata_only") is not True or metadata.get("outcome_columns_read") is not False or metadata.get("status") != "complete":
        raise ReplayError("Complete outcome-free metadata audit required")
    for key, relative in (("slim_sha256", "inputs/slim.parquet"), ("spec_sha256", "inputs/download-spec.json")):
        if metadata.get("input_binding", {}).get(key) != hashes[relative]["sha256"]:
            raise ReplayError(f"Metadata preflight input differs: {key}")
    profile = load_json(inputs / "intake_profile.json")
    if isinstance(profile.get("slim"), dict) and profile["slim"].get("sha256") != hashes["inputs/slim.parquet"]["sha256"]:
        raise ReplayError("Intake profile binds a different slim payload")
    for phase in PHASES:
        path = evidence / "protocols" / f"{phase}.json"
        protocol = load_json(path)
        reference = load_json(evidence / "reference" / phase / "analysis.json")
        references[phase], protocols[phase] = reference, protocol
        if protocol.get("execution_authorized") is not True or protocol.get("context_definition") != "scope_only":
            raise ReplayError("Only the activated frozen H2 scope_only phases may be replayed")
        expected = {"input_sha256": hashes["inputs/slim.parquet"]["sha256"], "spec_sha256": hashes["inputs/download-spec.json"]["sha256"],
                    "metadata_audit_sha256": hashes["inputs/metadata_audit.json"]["sha256"]}
        for key, value in expected.items():
            if protocol.get(key) != value:
                raise ReplayError(f"{phase} frozen input binding differs: {key}")
        binding = reference["input_binding"]
        for key, value in (("slim_sha256", expected["input_sha256"]), ("spec_sha256", expected["spec_sha256"]),
                           ("protocol_sha256", hashes[f"protocols/{phase}.json"]["sha256"]), ("source_sha256", source_hash)):
            if binding.get(key) != value:
                raise ReplayError(f"{phase} reference binding differs: {key}; copy original protocol bytes, not the reserialized output")
        comparison = Comparison(f"{phase} frozen protocol")
        comparison.compare(protocol, load_json(evidence / "reference" / phase / "protocol.json"))
        comparison.compare(protocol, reference["settings"])
        comparison.compare(metadata["metadata_integrity"]["conflict_ledger"],
                           load_json(evidence / "reference" / phase / "dataquality.json")["metadata_integrity"]["conflict_ledger"])
        if comparison.mismatches:
            raise ReplayError(f"{phase} reference protocol or metadata-population ledger differs")
        if reference.get("split") != phase or reference.get("context_definition") != "scope_only":
            raise ReplayError(f"{phase} reference has a different partition/estimand")
    for key in SHARED_PROTOCOL_KEYS:
        if key not in protocols["discovery"] or key not in protocols["confirmation"]:
            raise ReplayError(f"Missing shared frozen protocol setting: {key}")
        if protocols["discovery"][key] != protocols["confirmation"][key]:
            raise ReplayError(f"Frozen phases disagree on scientific setting: {key}")
    binding_report = {"files": hashes, "producer_source_sha256": source_hash, "critical_protocols_agree": True,
                      "original_frozen_protocol_bytes_verified": True}
    for name in ("uv.lock", "requirements.txt", "reference_origins.json"):
        path = evidence / name
        if path.is_file():
            binding_report.setdefault("additional_package_records", {})[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            if name == "reference_origins.json":
                binding_report["reference_origins"] = load_json(path)
    return references, binding_report


def write_report(path: Path, report: dict):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def replay(evidence: Path, output: Path) -> dict:
    evidence, output = evidence.resolve(), output.resolve()
    if not evidence.is_dir() or output.exists() or output == evidence or evidence in output.parents or output in evidence.parents:
        raise ReplayError("Evidence must exist; output must be a new separate directory outside its ancestry")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"schema_version": 1, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "evidence": str(evidence), "output": str(output), "scope": "computational reproduction of both frozen H2 phases",
              "tolerance": {"atol": ATOL, "rtol": 0, "equal_nan": True, "counts_booleans_metadata_exact": True},
              "json_exclusions": {"analysis.json": ["root timing", "root environment"]},
              "network_policy": "socket DNS/connect disabled during producer import and execution",
              "reference_output_locations": {p: str(evidence / "reference" / p) for p in PHASES}, "phases": {}}
    report_path = output / "replay_report.json"
    write_report(report_path, report)
    try:
        references, report["bindings"] = validate_bindings(evidence)
        with offline_guard():
            source = evidence / "src" / "cot_audit.py"
            name = "portable_cot_audit_" + report["bindings"]["producer_source_sha256"][:16]
            spec = importlib.util.spec_from_file_location(name, source)
            if spec is None or spec.loader is None:
                raise ReplayError("Cannot load the pinned producer module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            import numpy as np
            import pyarrow as pa
            pa.set_cpu_count(4)
            pa.set_io_thread_count(4)
            report["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__,
                                     "arrow_cpu_count": pa.cpu_count(), "arrow_io_thread_count": pa.io_thread_count(), "cpu_env_threads": 4}
            for phase in PHASES:
                phase_report = {"execution": "running", "comparisons": [], "output": str(output / phase)}
                report["phases"][phase] = phase_report
                write_report(report_path, report)
                print(json.dumps({"phase": phase, "status": "running"}), flush=True)
                logs = output / "logs"
                logs.mkdir(exist_ok=True)
                phase_report["logs"] = {name: f"logs/{phase}.{name}.log" for name in ("stdout", "stderr")}
                with (logs / f"{phase}.stdout.log").open("x", encoding="utf-8") as out, (logs / f"{phase}.stderr.log").open("x", encoding="utf-8") as err:
                    try:
                        with redirect_stdout(out), redirect_stderr(err):
                            module.run_analysis(evidence / "inputs" / "slim.parquet", evidence / "inputs" / "download-spec.json",
                                                evidence / "protocols" / f"{phase}.json", output / phase, phase)
                        phase_report["execution"] = "complete"
                    except Exception as exc:
                        traceback.print_exc(file=err)
                        phase_report.update(execution="failed", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                for filename in FILES:
                    reference, actual = evidence / "reference" / phase / filename, output / phase / filename
                    if not actual.is_file():
                        phase_report["comparisons"].append({"name": filename, "status": "failed", "mismatches": 1, "reason": "missing_replayed_output"})
                        continue
                    try:
                        if filename.endswith(".json"):
                            comparison = compare_json(reference, actual, analysis=filename == "analysis.json")
                        elif filename.endswith(".npz"):
                            comparison = compare_npz(reference, actual)
                        elif filename.endswith(".parquet"):
                            comparison = compare_parquet(reference, actual)
                        else:
                            comparison = compare_csv(reference, actual, csv_schema(references[phase]))
                    except Exception as exc:
                        comparison = {"name": filename, "status": "failed", "mismatches": 1,
                                      "reason": "comparison_error", "error": f"{type(exc).__name__}: {exc}"}
                    comparison.update(reference_sha256=sha256_file(reference), replay_sha256=sha256_file(actual))
                    phase_report["comparisons"].append(comparison)
                phase_report["scientific_match"] = phase_report["execution"] == "complete" and all(
                    check["status"] == "passed" for check in phase_report["comparisons"])
                write_report(report_path, report)
                print(json.dumps({"phase": phase, "scientific_match": phase_report["scientific_match"]}), flush=True)
        report["status"] = "passed" if all(report["phases"].get(p, {}).get("scientific_match") is True for p in PHASES) else "failed"
    except BaseException as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        if not isinstance(exc, Exception):
            raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        write_report(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = replay(args.evidence, args.output)
    print(json.dumps({"status": report["status"], "report": str(args.output / "replay_report.json")}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
