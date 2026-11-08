"""Pinned public-data intake, with no model calls or statistical endpoints.

The only network operation is a bounded GET for an explicitly listed payload.
The Parquet reducer preserves every row and all declared analysis columns.
Omitted transcript strings retain exact UTF-8 hashes, lengths and provenance.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


class IntakeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise IntakeError("Unsafe relative payload path")
    pure = PurePosixPath(value)
    if not pure.parts or pure.is_absolute() or pure.as_posix() != value or any(p in (".", "..") for p in pure.parts):
        raise IntakeError("Unsafe relative payload path")
    reserved = {"con", "prn", "aux", "nul"} | {f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)}
    for part in pure.parts:
        if part.endswith((".", " ")) or part.split(".", 1)[0].casefold() in reserved or any(c in '<>"|?*' or ord(c) < 32 for c in part):
            raise IntakeError("Nonportable payload path")
        if part.casefold() in {".env", ".git", ".venv"} or part.casefold().startswith(".env."):
            raise IntakeError("Environment/VCS paths are not dataset payloads")
    return Path(*pure.parts)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def validate_spec(spec: dict, *, require_authorized: bool = False) -> None:
    if spec.get("schema_version") != 1:
        raise IntakeError("Unsupported intake specification")
    if require_authorized and spec.get("download_authorized") is not True:
        raise IntakeError("Payload execution is not activated in the frozen specification")
    dataset = spec.get("dataset", {})
    repo = dataset.get("repo_id", "")
    revision = dataset.get("revision", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise IntakeError("Pinned public dataset/revision required")
    if dataset.get("private") is not False or dataset.get("gated") is not False:
        raise IntakeError("Only the explicitly public, ungated source is supported")
    if not dataset.get("license") or not dataset.get("license_url"):
        raise IntakeError("The source license must be recorded before acquisition")
    files = spec.get("files")
    if not isinstance(files, list) or not files or len(files) != spec.get("file_count"):
        raise IntakeError("Explicit complete file matrix required")
    seen = set()
    for item in files:
        relative = safe_relative(item.get("path"))
        key = relative.as_posix().casefold()
        if key in seen or relative.suffix != ".parquet":
            raise IntakeError("Duplicate or non-Parquet payload")
        seen.add(key)
        expected_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/" + urllib.parse.quote(relative.as_posix(), safe="/._-")
        if item.get("url") != expected_url:
            raise IntakeError("Unpinned or unexpected payload URL")
        if type(item.get("bytes")) is not int or item["bytes"] <= 0 or not re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")):
            raise IntakeError("Every payload needs positive exact bytes and SHA256")
    total = sum(item["bytes"] for item in files)
    if spec.get("total_payload_bytes") != total:
        raise IntakeError("File sizes do not match declared total")
    limits = spec.get("limits", {})
    for key in ("max_download_bytes", "max_http_requests", "max_attempts_per_file", "max_workers", "timeout_seconds", "max_slim_bytes", "batch_rows"):
        if type(limits.get(key)) is not int or limits[key] <= 0:
            raise IntakeError(f"Positive integral limit required: {key}")
    if limits["max_download_bytes"] > 1_000_000_000 or total > limits["max_download_bytes"]:
        raise IntakeError("Payload/retry transfer cap must be <=1GB and cover the matrix")
    if limits["max_workers"] > 4 or limits["max_attempts_per_file"] > 2 or limits["max_http_requests"] > 144:
        raise IntakeError("HTTP concurrency/attempt/request limit exceeds mandate")
    analysis, omitted = spec.get("analysis_columns"), spec.get("hashed_text_columns")
    if not isinstance(analysis, list) or not analysis or not isinstance(omitted, list) or not omitted:
        raise IntakeError("Explicit retained and hashed text columns required")
    if len(set(analysis + omitted)) != len(analysis + omitted):
        raise IntakeError("Column policies overlap or contain duplicates")
    features = spec.get("observed_hf_features", {})
    if set(analysis + omitted) != set(features):
        raise IntakeError("Every source column must have an explicit retention policy")
    if not {"core_id", "model", "id", "scope", "model_answer", "baseline_model_answer", "cue_target_answer", "correct_answer", "cue_acknowledgment"}.issubset(analysis):
        raise IntakeError("Required join/analysis fields cannot be omitted")
    for name in analysis + omitted:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name.startswith("source_") or name in {"row_content_sha256", "choices_ordered_sha256"}:
            raise IntakeError("Invalid or reserved source column")
    generated = {name + suffix for name in omitted for suffix in ("_sha256", "_chars", "_utf8_bytes")}
    if generated.intersection(analysis + omitted):
        raise IntakeError("Source field collides with an omitted-text descriptor")
    if any(features[name] != {"dtype": "string", "_type": "Value"} for name in omitted):
        raise IntakeError("Only declared string transcripts may be replaced with UTF8 hashes")


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        with self.path.open("x", encoding="utf-8"):
            pass

    def append(self, value: dict) -> None:
        with self.lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": utc_now(), **value}, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


@dataclass
class TransferBudget:
    limits: dict
    log: AuditLog
    started: float = field(default_factory=time.monotonic)
    received: int = 0
    reserved: int = 0
    requests: int = 0
    stopped: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def check(self) -> None:
        if self.stopped.is_set():
            raise IntakeError("Acquisition stopped after another payload failed")
        if time.monotonic() - self.started > self.limits.get("max_wall_seconds", 900):
            raise IntakeError("Intake wall-clock limit exhausted")

    def request(self, source: dict, attempt: int, *, redirect_host: str | None = None) -> int:
        self.check()
        with self.lock:
            if self.requests >= self.limits["max_http_requests"]:
                raise IntakeError("Physical HTTP request cap exhausted")
            self.requests += 1
            index = self.requests
        # Exactly one countable record is written before each physical request.
        # Redirect query strings/signed URLs and response headers are not logged.
        self.log.append({"kind": "http_request", "url": source["url"], "attempt": attempt,
                         "request_index": index, "source_path": source["path"], "redirect_host": redirect_host})
        return index

    def reserve(self, requested: int) -> int:
        self.check()
        with self.lock:
            left = self.limits["max_download_bytes"] - self.received - self.reserved
            if left <= 0:
                raise IntakeError("Download byte cap exhausted")
            amount = min(requested, left)
            self.reserved += amount
            return amount

    def commit(self, reserved: int, actual: int, source: dict, attempt: int) -> None:
        if not 0 <= actual <= reserved:
            raise IntakeError("HTTP reader exceeded reserved transfer allowance")
        with self.lock:
            self.reserved -= reserved
            self.received += actual
        if actual:
            # Dataset bytes include failed/partial attempts, without being
            # incorrectly classified as document bytes by resource_ledger.
            self.log.append({"kind": "dataset_download", "status": "received_chunk", "bytes": actual,
                             "source_path": source["path"], "file_attempt": attempt})

    def snapshot(self) -> dict:
        with self.lock:
            return {"http_requests": self.requests, "transferred_bytes": self.received,
                    "inflight_reserved_bytes": self.reserved, "max_download_bytes": self.limits["max_download_bytes"],
                    "max_http_requests": self.limits["max_http_requests"], "elapsed_seconds": time.monotonic() - self.started}


class _BoundedRedirect(urllib.request.HTTPRedirectHandler):
    def __init__(self, budget: TransferBudget, source: dict, attempt: int):
        self.budget, self.source, self.attempt, self.count = budget, source, attempt, 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme != "https" or not (host == "huggingface.co" or host.endswith((".huggingface.co", ".hf.co"))):
            raise IntakeError("Unexpected/non-HTTPS dataset redirect")
        self.count += 1
        if self.count > 5:
            raise IntakeError("Redirect chain exceeds five hops")
        self.budget.request(self.source, self.attempt, redirect_host=host)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_opener(budget: TransferBudget, source: dict, attempt: int):
    return urllib.request.build_opener(_BoundedRedirect(budget, source, attempt))


def download_one(index: int, source: dict, root: Path, budget: TransferBudget,
                 opener_factory: Callable = _default_opener) -> dict:
    destination = root / "data" / safe_relative(source["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt_dir = root / "download_attempts" / f"{index:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    attempts = []
    for attempt in range(1, budget.limits["max_attempts_per_file"] + 1):
        partial = attempt_dir / f"attempt-{attempt}.partial"
        started = time.monotonic()
        received, http_status, failure = 0, None, None
        try:
            budget.request(source, attempt)
            opener = opener_factory(budget, source, attempt)
            request = urllib.request.Request(source["url"], headers={"User-Agent": "cot-audit-pinned-intake/1", "Accept-Encoding": "identity"})
            with partial.open("xb") as stream, opener.open(request, timeout=budget.limits["timeout_seconds"]) as response:
                http_status = response.getcode()
                if http_status != 200:
                    raise IntakeError("Payload response is not HTTP200")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != source["bytes"]:
                    raise IntakeError("HTTP Content-Length differs from pinned payload size")
                while True:
                    reserved = budget.reserve(min(1024**2, source["bytes"] + 1 - received))
                    try:
                        block = response.read(reserved)
                    except BaseException:
                        budget.commit(reserved, 0, source, attempt)
                        raise
                    budget.commit(reserved, len(block), source, attempt)
                    if not block:
                        break
                    received += len(block)
                    stream.write(block)
                    if received > source["bytes"]:
                        raise IntakeError("Payload exceeds pinned byte count")
                stream.flush()
                os.fsync(stream.fileno())
            if received != source["bytes"] or file_sha256(partial) != source["sha256"]:
                raise IntakeError("Payload size/SHA256 mismatch")
            if destination.exists():
                raise IntakeError("Refusing to overwrite a payload destination")
            partial.rename(destination)
        except Exception as exc:
            # Do not persist network exception text: redirect URLs may carry
            # ephemeral signed query strings. Error type/status is sufficient.
            failure = str(exc) if isinstance(exc, IntakeError) else type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                http_status = exc.code
        row = {"source_path": source["path"], "file_attempt": attempt, "status": "failed" if failure else "verified",
               "http_status": http_status, "transferred_bytes": received, "elapsed_seconds": time.monotonic() - started,
               "expected_bytes": source["bytes"], "expected_sha256": source["sha256"], "error": failure,
               "path": (partial if failure else destination).relative_to(root).as_posix()}
        if partial.exists():
            row.update(partial_bytes=partial.stat().st_size, partial_sha256=file_sha256(partial))
        attempts.append(row)
        budget.log.append({"kind": "file_download_result", **row})
        if not failure:
            return {"path": source["path"], "status": "verified", "bytes": received, "sha256": source["sha256"], "attempts": attempts}
        if budget.stopped.is_set():
            break
    budget.stopped.set()
    return {"path": source["path"], "status": "failed", "attempts": attempts}


def _json_hash_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": "nan" if math.isnan(value) else "inf" if value > 0 else "-inf"}
    if isinstance(value, list):
        return [_json_hash_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_hash_value(item) for key, item in value.items()}
    return value


def _matches_feature(arrow_type, feature: Any, pa) -> bool:
    if isinstance(feature, list):
        return len(feature) == 1 and (pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type)) and _matches_feature(arrow_type.value_type, feature[0], pa)
    dtype = feature.get("dtype")
    return {"string": pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type),
            "bool": pa.types.is_boolean(arrow_type), "float64": pa.types.is_float64(arrow_type),
            "int64": pa.types.is_int64(arrow_type)}.get(dtype, False)


def slim_parquets(spec: dict, root: Path, *, check: Callable[[], None] = lambda: None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    kept, omitted = spec["analysis_columns"], spec["hashed_text_columns"]
    all_columns = kept + omitted
    reserved = ["source_file", "source_row_index", "source_payload_sha256", "row_content_sha256", "choices_ordered_sha256"]
    target = root / "slim.parquet"
    partial = root / "slim.parquet.partial"
    if target.exists() or partial.exists():
        raise IntakeError("Slim output already exists; use a new attempt")
    writer = None
    profiles, total_rows = [], 0
    total_nulls = {name: 0 for name in all_columns}
    scalar_sets = {"model": set(), "scope": set()}
    try:
        for source in spec["files"]:
            check()
            path = root / "data" / safe_relative(source["path"])
            if path.stat().st_size != source["bytes"] or file_sha256(path) != source["sha256"]:
                raise IntakeError("Unverified/changed source cannot be transformed")
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            if set(schema.names) != set(all_columns):
                raise IntakeError(f"Unexpected/missing source columns: {source['path']}")
            for name in all_columns:
                if not _matches_feature(schema.field(name).type, spec["observed_hf_features"][name], pa):
                    raise IntakeError(f"Unexpected source field type: {name}")
            if writer is None:
                fields = [pa.field(name, schema.field(name).type, nullable=True) for name in kept]
                fields.extend(pa.field(name, pa.string() if name != "source_row_index" else pa.int64()) for name in reserved)
                for name in omitted:
                    fields.extend([pa.field(name + "_sha256", pa.string()), pa.field(name + "_chars", pa.int64()), pa.field(name + "_utf8_bytes", pa.int64())])
                output_schema = pa.schema(fields, metadata={b"dataset_revision": spec["dataset"]["revision"].encode(),
                    b"row_content_hash_recipe": b"sha256(canonical JSON kept values plus omitted UTF8 sha256/chars/bytes descriptors; explicit null)"})
                writer = pq.ParquetWriter(partial, output_schema, compression="zstd", compression_level=6, use_dictionary=True)
            source_rows, nulls = 0, {name: 0 for name in all_columns}
            for batch in parquet.iter_batches(batch_size=spec["limits"]["batch_rows"], columns=all_columns, use_threads=False):
                check()
                raw = batch.to_pydict()
                out = {name: raw[name] for name in kept}
                size = batch.num_rows
                out.update(source_file=[source["path"]] * size, source_row_index=list(range(source_rows, source_rows + size)),
                           source_payload_sha256=[source["sha256"]] * size, row_content_sha256=[], choices_ordered_sha256=[])
                descriptors = {name: [] for name in omitted}
                for name in all_columns:
                    count = sum(value is None for value in raw[name])
                    nulls[name] += count
                    total_nulls[name] += count
                for name in omitted:
                    hashes, lengths, byte_lengths = [], [], []
                    for text in raw[name]:
                        if text is None:
                            sha, length, byte_length = None, None, None
                        else:
                            encoded = text.encode("utf-8")
                            sha, length, byte_length = hashlib.sha256(encoded).hexdigest(), len(text), len(encoded)
                        hashes.append(sha); lengths.append(length); byte_lengths.append(byte_length)
                        descriptors[name].append({"sha256": sha, "chars": length, "utf8_bytes": byte_length})
                    out[name + "_sha256"], out[name + "_chars"], out[name + "_utf8_bytes"] = hashes, lengths, byte_lengths
                for i in range(size):
                    identity = {name: _json_hash_value(raw[name][i]) for name in kept}
                    identity.update({name: {"__omitted_utf8__": descriptors[name][i]} for name in omitted})
                    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
                    out["row_content_sha256"].append(hashlib.sha256(payload).hexdigest())
                    choices = raw["choices"][i]
                    out["choices_ordered_sha256"].append(None if choices is None else hashlib.sha256(
                        json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest())
                for name in scalar_sets:
                    scalar_sets[name].update(value for value in raw[name] if value is not None)
                writer.write_table(pa.Table.from_pydict(out, schema=output_schema), row_group_size=size)
                source_rows += size
                total_rows += size
                if partial.stat().st_size > spec["limits"]["max_slim_bytes"]:
                    raise IntakeError("Slim Parquet byte limit exceeded; partial output preserved")
            if source_rows != parquet.metadata.num_rows:
                raise IntakeError("Streamed source row count mismatch")
            profiles.append({"source_file": source["path"], "source_sha256": source["sha256"], "rows": source_rows,
                             "arrow_schema": str(schema), "nulls": nulls})
    finally:
        if writer is not None:
            writer.close()
    if writer is None or partial.stat().st_size > spec["limits"]["max_slim_bytes"]:
        raise IntakeError("Empty or oversized slim result")
    partial.rename(target)
    result = {"schema_version": 1, "rows": total_rows, "source_files": len(profiles), "source_profiles": profiles,
              "retained_columns": kept, "hashed_text_columns": omitted, "nulls": total_nulls,
              "models": sorted(scalar_sets["model"]), "scopes": sorted(scalar_sets["scope"]),
              "slim": {"path": "slim.parquet", "bytes": target.stat().st_size, "sha256": file_sha256(target)},
              "row_filtering": "none; all source rows retained", "value_normalization": "none; nullable values retained",
              "hash_recipe": "Exact UTF8 per omitted string; row_content_sha256 hashes kept values plus typed omitted-text hash/length descriptors, not normalized text.",
              "ordered_choices_hash_recipe": "SHA256(JSON UTF8 ordered choices list with ensure_ascii=False and compact separators); null is null, empty list is hashed.",
              "statistical_endpoints_computed": False, "scientific_validity": "not_assessed"}
    atomic_json(root / "schema_profile.json", result)
    return result


def run_intake(spec: dict, output: Path, *, spec_path: Path | None = None,
               opener_factory: Callable = _default_opener) -> dict:
    validate_spec(spec, require_authorized=True)
    if output.is_symlink() or (hasattr(output, "is_junction") and output.is_junction()):
        raise IntakeError("Intake output cannot be a symlink/junction")
    output = output.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise IntakeError("Intake output must be new or empty")
    else:
        output.mkdir(parents=True, exist_ok=False)
    if output.is_symlink() or (hasattr(output, "is_junction") and output.is_junction()):
        raise IntakeError("Intake output cannot be a symlink/junction")
    log = AuditLog(output / "acquisition.jsonl")
    budget = TransferBudget(spec["limits"], log)
    manifest = {"schema_version": 1, "dataset": spec["dataset"], "status": "running", "files": [],
                "spec_sha256": file_sha256(spec_path) if spec_path else hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                "started_at_utc": utc_now(), "pid": os.getpid(), "model_calls": 0, "gpu_jobs": 0,
                "scientific_validity": "not_assessed"}
    atomic_json(output / "download_manifest.json", manifest)
    try:
        with ThreadPoolExecutor(max_workers=spec["limits"]["max_workers"]) as executor:
            futures = {executor.submit(download_one, i, source, output, budget, opener_factory): source for i, source in enumerate(spec["files"])}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as exc:
                    budget.stopped.set()
                    result = {"path": futures[future]["path"], "status": "failed", "error": type(exc).__name__}
                manifest["files"].append(result)
                manifest["network"] = budget.snapshot()
                atomic_json(output / "download_manifest.json", manifest)
        manifest["files"].sort(key=lambda item: item["path"])
        if len(manifest["files"]) != len(spec["files"]) or any(item["status"] != "verified" for item in manifest["files"]):
            raise IntakeError("Complete pinned download matrix did not verify")
        manifest["status"] = "download_verified"
        atomic_json(output / "download_manifest.json", manifest)
        profile = slim_parquets(spec, output, check=budget.check)
        result = {"schema_version": 1, "status": "complete", "dataset": spec["dataset"], "spec_sha256": manifest["spec_sha256"],
                  "network": budget.snapshot(), "rows": profile["rows"], "slim": profile["slim"],
                  "raw_files": len(spec["files"]), "all_source_rows_retained": True, "statistical_endpoints_computed": False,
                  "model_calls": 0, "gpu_jobs": 0, "scientific_validity": "not_assessed"}
        import pyarrow
        result["environment"] = {"python": platform.python_version(), "pyarrow": pyarrow.__version__,
                                 "arrow_cpu_threads": pyarrow.cpu_count(), "arrow_io_threads": pyarrow.io_thread_count(),
                                 "http_workers": spec["limits"]["max_workers"]}
        manifest.update(status="complete", finished_at_utc=utc_now(), network=budget.snapshot())
        atomic_json(output / "download_manifest.json", manifest)
        atomic_json(output / "intake_report.json", result)
        return result
    except BaseException as exc:
        manifest.update(status="failed", finished_at_utc=utc_now(), network=budget.snapshot(), error=type(exc).__name__)
        atomic_json(output / "download_manifest.json", manifest)
        atomic_json(output / "intake_report.json", {"status": "failed", "error_type": type(exc).__name__,
                    "network": budget.snapshot(), "partial_artifacts_preserved": True, "scientific_validity": "not_assessed"})
        raise
