"""Small, offline research execution ledger; scientific judgement stays explicit.

The ledger detects accidental changes and binds reviews to evidence. It is not a
security sandbox or an independent reviewer, and never asserts scientific validity.
Only the standard library is needed, including for an engineering smoke run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Iterator


class WorkflowError(ValueError):
    """A plan, evidence bundle or transition fails the operating contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def identifier(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise WorkflowError(f"Unsafe identifier: {value!r}")
    return value


def confined(base: Path, relative: Any, *, must_exist: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise WorkflowError(f"Use a nonempty forward-slash relative path: {relative!r}")
    part = Path(relative)
    if part.is_absolute() or ":" in relative or any(p in ("..", ".") for p in relative.split("/")):
        raise WorkflowError(f"Path escapes its evidence root: {relative}")
    base = base.resolve()
    candidate = base / part
    if not candidate.resolve().is_relative_to(base):
        raise WorkflowError(f"Path escapes through a symlink: {relative}")
    cursor = candidate
    while cursor != base:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise WorkflowError(f"Symlinks/junctions are not evidence files: {relative}")
        cursor = cursor.parent
    if must_exist and not candidate.is_file():
        raise WorkflowError(f"Required file is missing: {relative}")
    return candidate


class Workflow:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.registry_path = confined(self.root, "research/stages.json", must_exist=True)
        registry = load_json(self.registry_path)
        if registry.get("schema_version") != 1 or not isinstance(registry.get("stages"), list):
            raise WorkflowError("Registry must have schema_version=1 and stages list")
        self.stages: dict[str, dict[str, Any]] = {}
        for entry in registry["stages"]:
            if not isinstance(entry, dict):
                raise WorkflowError("Each registry stage must be an object")
            stage = identifier(entry.get("id"))
            if stage in self.stages or entry.get("kind") not in ("engineering", "science"):
                raise WorkflowError(f"Duplicate stage or invalid kind: {stage}")
            if not isinstance(entry.get("depends_on"), list):
                raise WorkflowError(f"depends_on must be a list: {stage}")
            for flag in ("preparation_authorized", "execution_authorized"):
                if flag in entry and not isinstance(entry[flag], bool):
                    raise WorkflowError(f"{flag} must be boolean: {stage}")
            confined(self.root, entry.get("plan"))
            self._authorization_binding(entry)
            self.stages[stage] = entry
        visiting: set[str] = set()
        seen: set[str] = set()

        def visit(stage: str) -> None:
            if stage not in self.stages:
                raise WorkflowError(f"Unknown dependency: {stage}")
            if stage in visiting:
                raise WorkflowError(f"Dependency cycle at {stage}")
            if stage in seen:
                return
            visiting.add(stage)
            for dep in self.stages[stage]["depends_on"]:
                visit(identifier(dep))
            visiting.remove(stage)
            seen.add(stage)

        for stage in self.stages:
            visit(stage)
        self.state = confined(self.root, ".research/workflow")
        self.artifacts = confined(self.root, "artifacts/research")

    def stage(self, stage: str) -> dict[str, Any]:
        if stage not in self.stages:
            raise WorkflowError(f"Unknown stage: {stage}")
        return self.stages[stage]

    def _authorization_binding(self, entry: dict[str, Any], *, base: Path | None = None) -> dict[str, str] | None:
        """A declaration binds bytes; it does not authenticate its author or science."""
        binding = entry.get("authorization_decision")
        if binding is None:
            return None  # Older engineering records need no invented authority.
        if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                or not isinstance(binding.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", binding["sha256"])):
            raise WorkflowError("authorization_decision needs exactly a relative path and lowercase SHA-256")
        confined(self.root if base is None else base, binding.get("path"))
        self._safe_input(binding["path"])
        return {"path": binding["path"], "sha256": binding["sha256"]}

    def validate_authorization(self, stage: str) -> dict[str, str] | None:
        """Check an optional live charter without requiring an executable plan."""
        binding = self._authorization_binding(self.stage(stage))
        if binding is not None:
            path = confined(self.root, binding["path"], must_exist=True)
            if sha256(path) != binding["sha256"]:
                raise WorkflowError(f"Authorization decision changed: {stage}")
        return binding

    @contextmanager
    def lock(self, stage: str) -> Iterator[None]:
        """One writer for the project; a dead process leaves an explicit recovery record."""
        path = confined(self.root, ".research/workflow/active.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as stream:
                json.dump({"stage": stage, "pid": os.getpid(), "started_at": utc_now()}, stream)
        except FileExistsError as exc:
            raise WorkflowError(f"Workflow lock exists: {path}; inspect owner before recovery") from exc
        try:
            yield
        finally:
            path.unlink()

    def closure_path(self, stage: str) -> Path:
        self.stage(stage)
        return confined(self.root, f".research/workflow/closures/{stage}.json")

    def status(self) -> dict[str, Any]:
        closures: dict[str, str | None] = {}
        for stage in self.stages:
            if self.closure_path(stage).is_file():
                try:
                    self.verify_closure(stage)
                    closures[stage] = None
                except (WorkflowError, OSError, KeyError, TypeError) as exc:
                    closures[stage] = str(exc)
        rows = []
        for stage, entry in self.stages.items():
            closed = stage in closures and closures[stage] is None
            missing = [dep for dep in entry["depends_on"] if dep not in closures or closures[dep] is not None]
            authorization_error = None
            if not closed:
                try:
                    self.validate_authorization(stage)
                except (WorkflowError, OSError) as exc:
                    authorization_error = str(exc)
            state = ("closed" if closed else "invalid" if stage in closures or authorization_error
                     else "waiting" if missing else "ready")
            execution = entry.get("execution_authorized", entry["kind"] == "engineering")
            preparation = entry.get("preparation_authorized", False)
            action = ("complete" if state == "closed" else "escalate" if state == "invalid"
                      else "wait_for_dependencies" if state == "waiting"
                      else "execute" if execution else "prepare" if preparation else "await_authorization")
            rows.append({"id": stage, "kind": entry["kind"],
                         "state": state,
                         "waiting_for": missing,
                         "closure_error": closures.get(stage),
                         "preparation_authorized": preparation,
                         "execution_authorized": execution,
                         "authorization_decision": entry.get("authorization_decision"),
                         "authorization_error": authorization_error,
                         "recommended_action": action,
                         "scientific_validity": "not_assessed"})
        return {"stages": rows, "scientific_validity": "not_assessed"}

    def next_stage(self) -> dict[str, Any]:
        rows = self.status()["stages"]
        invalid = [row["id"] for row in rows if row["state"] == "invalid"]
        if invalid:
            return {"next": None, "recommended_action": "escalate", "blocking_stages": invalid}
        ready = next((row for row in rows if row["state"] == "ready"), None)
        action = (ready["recommended_action"] if ready else
                  "wait_for_dependencies" if any(row["state"] == "waiting" for row in rows) else "complete")
        return {"next": ready, "recommended_action": action, "blocking_stages": []}

    def validate_plan(self, stage: str) -> dict[str, Any]:
        entry = self.stage(stage)
        authorization = self.validate_authorization(stage)
        path = confined(self.root, entry["plan"], must_exist=True)
        plan = load_json(path)
        if plan.get("schema_version") != 1 or plan.get("stage_id") != stage:
            raise WorkflowError("Plan schema/stage_id does not match registry")
        if entry["kind"] == "science" and plan.get("status") != "frozen":
            raise WorkflowError("Science execution requires a frozen plan, not a draft")
        for key in ("question", "scope"):
            if not isinstance(plan.get(key), str) or not plan[key].strip():
                raise WorkflowError(f"Plan needs {key}")
        budget = plan.get("budget", {}).get("max_wall_seconds")
        if not isinstance(budget, (int, float)) or isinstance(budget, bool) or not math.isfinite(budget) or not 0 < budget <= 604800:
            raise WorkflowError("Budget max_wall_seconds must be positive and <= 7 days")
        max_attempts = plan["budget"].get("max_attempts", 3)
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or not 0 < max_attempts <= 100:
            raise WorkflowError("Budget max_attempts must be an integer from 1 to 100")
        total = plan["budget"].get("total_wall_seconds")
        if total is not None and (not isinstance(total, (int, float)) or isinstance(total, bool) or not math.isfinite(total) or not 0 < total <= 60480000):
            raise WorkflowError("Optional total_wall_seconds must be finite, positive and <= 700 days")
        tasks = plan.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise WorkflowError("Executable plan needs a nonempty registered task matrix")
        ids: set[str] = set()
        for task in tasks:
            task_id = identifier(task.get("id"))
            if task_id in ids:
                raise WorkflowError(f"Duplicate task ID: {task_id}")
            ids.add(task_id)
            argv = task.get("argv")
            if not isinstance(argv, list) or not argv or any(not isinstance(a, str) or not a for a in argv):
                raise WorkflowError(f"Task {task_id} needs a nonempty argv list")
            timeout = task.get("timeout_seconds")
            if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= budget:
                raise WorkflowError(f"Task {task_id} timeout exceeds stage budget")
            for key in ("inputs", "outputs"):
                values = task.get(key)
                if not isinstance(values, list) or not values or len(set(values)) != len(values):
                    raise WorkflowError(f"Task {task_id} needs distinct nonempty {key}")
                for value in values:
                    if key == "inputs":
                        self._safe_input(value)
                        confined(self.root, value, must_exist=True)
                    else:
                        confined(self.artifacts / stage / "validation", value)
                        if value in ("stdout.log", "stderr.log"):
                            raise WorkflowError("Outputs cannot shadow execution logs")
        for key in ("exit_criteria", "predictions"):
            entries = plan.get(key)
            if not isinstance(entries, list) or not entries:
                raise WorkflowError(f"Plan needs nonempty {key}")
            seen: set[str] = set()
            for item in entries:
                item_id = identifier(item.get("id"))
                if item_id in seen:
                    raise WorkflowError(f"Duplicate {key} ID: {item_id}")
                seen.add(item_id)
                text_key = "description" if key == "exit_criteria" else "statement"
                if not isinstance(item.get(text_key), str) or not item[text_key].strip():
                    raise WorkflowError(f"Missing {key} description: {item_id}")
                if key == "exit_criteria":
                    refs = item.get("task_ids")
                    if not isinstance(refs, list) or not refs or not set(refs).issubset(ids):
                        raise WorkflowError(f"Criterion {item_id} must reference registered tasks")
        return {"stage_id": stage, "plan_sha256": sha256(path), "tasks": sorted(ids),
                "authorization_decision": authorization,
                "valid": True, "scientific_validity": "not_assessed"}

    def attempt(self, stage: str, attempt_id: str | None = None) -> Path:
        self.stage(stage)
        base = confined(self.root, f"artifacts/research/{stage}")
        if attempt_id:
            path = confined(base, identifier(attempt_id))
        else:
            choices = sorted(p for p in base.glob("*") if p.is_dir() and (p / "manifest.json").is_file())
            if not choices:
                raise WorkflowError(f"No recorded attempts for {stage}")
            path = choices[-1]
        if not path.is_dir():
            raise WorkflowError(f"Missing attempt: {path}")
        return path

    def _source_paths(self, plan: dict[str, Any]) -> list[str]:
        paths = {value for task in plan["tasks"] for value in task["inputs"]}
        paths.add("research/stages.json")
        paths.add(self.stage(plan["stage_id"])["plan"])
        authorization = self.validate_authorization(plan["stage_id"])
        if authorization is not None:
            paths.add(authorization["path"])
        for value in ("src/research_workflow.py", "src/resource_ledger.py", "scripts/research.py", "pyproject.toml", "uv.lock"):
            if (self.root / value).is_file():
                paths.add(value)
        for value in paths:
            self._safe_input(value)
            if value.startswith(("artifacts/research/", ".research/workflow/")):
                raise WorkflowError("Mutable workflow evidence cannot be a source input")
        return sorted(paths)

    @staticmethod
    def _safe_input(value: str) -> None:
        """Do not duplicate credentials, VCS internals or installed environments."""
        parts = value.lower().replace("\\", "/").split("/")
        if any(part in (".git", ".venv") or part == ".env" or part.startswith(".env.") for part in parts):
            raise WorkflowError(f"Credential/environment/VCS paths cannot be snapshotted: {value}")

    def _attempt_budget(self, stage: str, plan: dict[str, Any]) -> float:
        """Count interrupted attempts too; unknown elapsed time consumes a full attempt."""
        base = confined(self.root, f"artifacts/research/{stage}")
        previous = [p for p in base.glob("*") if p.is_dir()]
        budget = plan["budget"]
        if len(previous) >= budget.get("max_attempts", 3):
            raise WorkflowError(f"Stage max_attempts exhausted: {stage}")
        remaining = float(budget["max_wall_seconds"])
        if "total_wall_seconds" in budget:
            spent = 0.0
            for attempt in previous:
                try:
                    elapsed = load_json(attempt / "manifest.json").get("elapsed_seconds", budget["max_wall_seconds"])
                    if not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or not math.isfinite(elapsed) or elapsed < 0:
                        raise WorkflowError("Invalid prior elapsed time")
                    spent += elapsed
                except (WorkflowError, OSError):
                    spent += budget["max_wall_seconds"]
            remaining = min(remaining, budget["total_wall_seconds"] - spent)
            if remaining <= 0:
                raise WorkflowError(f"Stage total_wall_seconds exhausted: {stage}")
        return remaining

    def _git(self) -> dict[str, Any]:
        def call(*args: str) -> str | None:
            try:
                result = subprocess.run(["git", *args], cwd=self.root, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=15)
                return result.stdout.strip() if result.returncode == 0 else None
            except (OSError, subprocess.SubprocessError):
                return None
        return {"sha": call("rev-parse", "HEAD"), "status": call("status", "--porcelain")}

    @staticmethod
    def _execute(argv: list[str], cwd: Path, timeout: float) -> tuple[int, bool]:
        options: dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        with (cwd / "stdout.log").open("wb") as stdout, (cwd / "stderr.log").open("wb") as stderr:
            child = subprocess.Popen(argv, cwd=cwd, stdout=stdout, stderr=stderr, shell=False, **options)
            def terminate() -> None:
                if child.poll() is not None:
                    return
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
            try:
                return child.wait(timeout=timeout), False
            except subprocess.TimeoutExpired:
                terminate()
                return child.returncode, True
            except BaseException:
                terminate()
                raise

    def run(self, stage: str) -> dict[str, Any]:
        # Import at call time: resource_ledger uses this module's hashing helper.
        from src.resource_ledger import append_entry, report as resource_report

        entry = self.stage(stage)
        if not entry.get("execution_authorized", entry["kind"] == "engineering"):
            raise WorkflowError(f"Execution is not authorized for {stage}; register the bounded plan first")
        if self.closure_path(stage).exists():
            raise WorkflowError(f"Stage is already operationally closed: {stage}")
        for dep in entry["depends_on"]:
            if not self.closure_path(dep).is_file():
                raise WorkflowError(f"Dependency is not closed: {dep}")
            self.verify_closure(dep)
        self.validate_plan(stage)
        with self.lock(stage):
            authorization = self.validate_authorization(stage)
            plan_path = confined(self.root, entry["plan"], must_exist=True)
            plan = load_json(plan_path)
            # Older declarations (Markdown or JSON without resource budgets)
            # retain their original semantics; a declared budget must validate.
            if authorization is not None and Path(authorization["path"]).suffix.lower() == ".json":
                charter = load_json(confined(self.root, authorization["path"], must_exist=True))
                if "budgets" in charter:
                    try:
                        resources = resource_report(self.root, charter_path=authorization["path"],
                                                    expected_sha256=authorization["sha256"])
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        raise WorkflowError(f"Cannot validate cumulative resource budget: {exc}") from exc
                    if not resources["computational_activation_ready"]:
                        raise WorkflowError("Cumulative resource preflight is not ready: "
                                            f"unmeasured={resources['unmeasured_attempts']}, "
                                            f"reconciliation={resources['reconciliation_problems']}; "
                                            "inspect the resource report before recovery")
                    if resources["remaining_wall_seconds"] < plan["budget"]["max_wall_seconds"]:
                        raise WorkflowError("Cumulative remaining wall budget cannot cover declared max_wall_seconds")
            attempt_wall_limit = self._attempt_budget(stage, plan)
            # Bind the precise accepted ancestry, not merely dependency names.
            for dep in entry["depends_on"]:
                self.verify_closure(dep)
            dependency_hashes = {dep: sha256(self.closure_path(dep)) for dep in entry["depends_on"]}
            attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
            attempt = confined(self.root, f"artifacts/research/{stage}/{attempt_id}")
            attempt.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(plan_path, attempt / "plan.json")
            sources = {value: sha256(confined(self.root, value, must_exist=True)) for value in self._source_paths(plan)}
            manifest: dict[str, Any] = {
                "schema_version": 1, "stage_id": stage, "attempt_id": attempt_id,
                "registered_plan_path": entry["plan"],
                "dependency_closure_hashes": dependency_hashes,
                "authorization_decision": authorization,
                "started_at": utc_now(), "status": "running", "plan_sha256": sha256(attempt / "plan.json"),
                "input_hashes": sources, "code_hashes": {p: digest for p, digest in sources.items() if p.endswith(".py")},
                "git": self._git(),
                "environment": {"python": sys.version, "executable": sys.executable,
                                "platform": platform.platform(), "machine": platform.machine(),
                                "processor": platform.processor(), "cpu_count": os.cpu_count(),
                                "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
                                "thread_controls": {key: os.environ[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES") if key in os.environ}},
                "tasks": [], "scientific_validity": "not_assessed",
            }
            write_json(attempt / "manifest.json", manifest, exclusive=True)
            started = time.monotonic()
            failure: str | None = None
            try:
                for value in sources:
                    target = confined(attempt / "inputs", value)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(confined(self.root, value, must_exist=True), target)
                    if sha256(target) != sources[value]:
                        raise WorkflowError(f"Source changed while snapshotting: {value}")
                for task in plan["tasks"]:
                    taskdir = confined(attempt, f"tasks/{task['id']}")
                    taskdir.mkdir(parents=True, exist_ok=False)
                    argv = [arg.replace("{python}", sys.executable).replace("{repo}", str(self.root)).replace("{run}", str(attempt)) for arg in task["argv"]]
                    remaining = attempt_wall_limit - (time.monotonic() - started)
                    if remaining <= 0:
                        raise WorkflowError("Stage wall-clock budget exhausted")
                    record: dict[str, Any] = {"id": task["id"], "argv": argv,
                                             "cwd": taskdir.relative_to(attempt).as_posix(),
                                             "started_at": utc_now(), "status": "running"}
                    manifest["tasks"].append(record)
                    write_json(attempt / "manifest.json", manifest)
                    code, timed_out = self._execute(argv, taskdir, min(task["timeout_seconds"], remaining))
                    record.update(exit_code=code, timed_out=timed_out, finished_at=utc_now())
                    outputs = {name: sha256(confined(taskdir, name, must_exist=True)) for name in task["outputs"]}
                    record["output_hashes"] = outputs
                    for child in taskdir.rglob("*"):
                        confined(taskdir, child.relative_to(taskdir).as_posix())
                    record["file_hashes"] = {path.relative_to(taskdir).as_posix(): sha256(confined(taskdir, path.relative_to(taskdir).as_posix(), must_exist=True))
                                              for path in sorted(taskdir.rglob("*")) if path.is_file() or path.is_symlink()}
                    record["status"] = "complete" if code == 0 and not timed_out else "failed"
                    if code != 0 or timed_out:
                        raise WorkflowError(f"Task {task['id']} failed (exit={code}, timeout={timed_out})")
                for value, digest in sources.items():
                    if sha256(confined(self.root, value, must_exist=True)) != digest:
                        raise WorkflowError(f"Source input changed during execution: {value}")
            except BaseException as exc:
                failure = f"{type(exc).__name__}: {exc}"
                manifest["failure"] = failure
            finally:
                manifest.update(status="failed" if failure else "complete", finished_at=utc_now(),
                                elapsed_seconds=time.monotonic() - started)
                write_json(attempt / "manifest.json", manifest)
                try:
                    append_entry(self.root, {
                        "kind": "attempt", "stage_id": stage, "attempt_id": attempt_id,
                        "elapsed_seconds": manifest["elapsed_seconds"], "status": manifest["status"],
                        "note": "Automatically recorded by the workflow runner; scientific_validity not_assessed.",
                    })
                except (OSError, ValueError) as exc:
                    raise WorkflowError(f"Resource ledger append failed; evidence retained at {attempt}: {exc}") from exc
            if failure:
                raise WorkflowError(f"Attempt {attempt_id} failed; evidence retained at {attempt}: {failure}")
            return {"stage_id": stage, "attempt_id": attempt_id, "status": "complete",
                    "path": str(attempt), "scientific_validity": "not_assessed"}

    def verify(self, stage: str, attempt_id: str | None = None, *, live_sources: bool = True) -> dict[str, Any]:
        attempt = self.attempt(stage, attempt_id)
        manifest = load_json(attempt / "manifest.json")
        plan = load_json(attempt / "plan.json")
        if manifest.get("stage_id") != stage or manifest.get("attempt_id") != attempt.name or manifest.get("status") != "complete":
            raise WorkflowError("Attempt identity/status is not complete")
        dependency_hashes = manifest.get("dependency_closure_hashes")
        if not isinstance(dependency_hashes, dict) or set(dependency_hashes) != set(self.stage(stage)["depends_on"]):
            raise WorkflowError(f"Attempt dependency identities changed: {stage}")
        for dep, digest in dependency_hashes.items():
            if sha256(self.closure_path(dep)) != digest:
                raise WorkflowError(f"Dependency closure changed: {stage} -> {dep}")
        if sha256(attempt / "plan.json") != manifest.get("plan_sha256"):
            raise WorkflowError("Frozen plan changed")
        if plan.get("schema_version") != 1 or plan.get("stage_id") != stage:
            raise WorkflowError("Frozen plan has wrong schema/stage identity")
        if live_sources and sha256(confined(self.root, self.stage(stage)["plan"], must_exist=True)) != manifest["plan_sha256"]:
            raise WorkflowError("Live plan changed; create a new registered attempt")
        sources = manifest.get("input_hashes")
        if not isinstance(sources, dict) or not sources:
            raise WorkflowError("Empty source hash inventory")
        frozen_registry = load_json(confined(attempt / "inputs", "research/stages.json", must_exist=True))
        entries = frozen_registry.get("stages")
        if not isinstance(entries, list):
            raise WorkflowError("Frozen registry has no stage declarations")
        frozen_entries = [entry for entry in entries if isinstance(entry, dict) and entry.get("id") == stage]
        if len(frozen_entries) != 1:
            raise WorkflowError("Frozen registry must declare this stage exactly once")
        authorization = self._authorization_binding(frozen_entries[0], base=attempt / "inputs")
        if manifest.get("authorization_decision") != authorization:
            raise WorkflowError("Attempt authorization differs from frozen registry")
        if authorization is not None and sources.get(authorization["path"]) != authorization["sha256"]:
            raise WorkflowError("Source inventory does not bind the authorization decision")
        if live_sources and self.validate_authorization(stage) != authorization:
            raise WorkflowError("Live authorization differs from this attempt")
        required_inputs = {value for task in plan["tasks"] for value in task["inputs"]}
        required_inputs.update(("research/stages.json", manifest.get("registered_plan_path")))
        if authorization is not None:
            required_inputs.add(authorization["path"])
        if not required_inputs.issubset(sources):
            raise WorkflowError("Source inventory does not cover registered inputs and controls")
        for value, digest in sources.items():
            if sha256(confined(attempt / "inputs", value, must_exist=True)) != digest:
                raise WorkflowError(f"Input snapshot changed: {value}")
            if live_sources and sha256(confined(self.root, value, must_exist=True)) != digest:
                raise WorkflowError(f"Live input changed: {value}; rerun before closing")
        tasks = manifest.get("tasks", [])
        if [task["id"] for task in tasks] != [task["id"] for task in plan["tasks"]]:
            raise WorkflowError("Recorded tasks do not match the registered matrix")
        checked = 0
        for task, spec in zip(tasks, plan["tasks"]):
            if task.get("status") != "complete" or task.get("exit_code") != 0 or task.get("timed_out"):
                raise WorkflowError(f"Task is not complete: {task['id']}")
            if set(task.get("output_hashes", {})) != set(spec["outputs"]) or not task.get("file_hashes"):
                raise WorkflowError(f"Missing/non-matching output inventory: {task['id']}")
            taskdir = confined(attempt, f"tasks/{task['id']}")
            if not {"stdout.log", "stderr.log"}.issubset(task["file_hashes"]):
                raise WorkflowError(f"Task logs are absent from inventory: {task['id']}")
            actual_files: set[str] = set()
            for child in taskdir.rglob("*"):
                confined(taskdir, child.relative_to(taskdir).as_posix())
                if child.is_file():
                    actual_files.add(child.relative_to(taskdir).as_posix())
            if actual_files != set(task["file_hashes"]):
                raise WorkflowError(f"Task file inventory changed: {task['id']}")
            for name, digest in task["file_hashes"].items():
                if sha256(confined(taskdir, name, must_exist=True)) != digest:
                    raise WorkflowError(f"Output changed: {task['id']}/{name}")
                checked += 1
            for name, digest in task["output_hashes"].items():
                if task["file_hashes"].get(name) != digest:
                    raise WorkflowError(f"Output hash records disagree: {task['id']}/{name}")
        return {"stage_id": stage, "attempt_id": attempt.name, "verified": True,
                "checked_files": checked, "manifest_sha256": sha256(attempt / "manifest.json"),
                "plan_sha256": manifest["plan_sha256"], "scientific_validity": "not_assessed"}

    def _requires_subagent_review(self, attempt: Path) -> bool:
        """Use the run's frozen charter, never a later live policy revision."""
        binding = load_json(attempt / "manifest.json").get("authorization_decision")
        if binding is None:
            return False
        path = confined(attempt / "inputs", binding["path"], must_exist=True)
        if path.suffix.lower() != ".json":
            return False  # Legacy declarations may be Markdown, without this policy.
        required = load_json(path).get("require_distinct_stage_review_subagent", False)
        if not isinstance(required, bool):
            raise WorkflowError("require_distinct_stage_review_subagent must be boolean")
        return required

    def review_template(self, stage: str, report: str, attempt_id: str | None = None) -> dict[str, Any]:
        result = self.verify(stage, attempt_id)
        attempt = self.attempt(stage, result["attempt_id"])
        plan = load_json(attempt / "plan.json")
        report_path = confined(self.root, report, must_exist=True)
        review = {"schema_version": 1, "stage_id": stage, "attempt_id": attempt.name,
                "reviewer": {"name": "", "harness": "", "role": "stage_reviewer",
                             "independence": "not_established"},
                "plan_sha256": result["plan_sha256"], "manifest_sha256": result["manifest_sha256"],
                "report": {"path": report, "sha256": sha256(report_path)},
                "criteria": [{"id": item["id"], "verdict": "pending", "evidence": []} for item in plan["exit_criteria"]],
                "predictions": [{"id": item["id"], "outcome": "", "evidence": []} for item in plan["predictions"]],
                "limitations": [], "verdict": "pending", "scientific_validity": "not_assessed"}
        if self._requires_subagent_review(attempt):
            review.update(author_agent="", reviewer_agent="", review_mode="subagent_review")
        return review

    def _check_review(self, stage: str, attempt: Path, review: dict[str, Any], *, live_sources: bool = True,
                      verify_live_sources: bool | None = None) -> None:
        result = self.verify(stage, attempt.name, live_sources=live_sources if verify_live_sources is None else verify_live_sources)
        plan = load_json(attempt / "plan.json")
        manifest = load_json(attempt / "manifest.json")
        if any(review.get(key) != value for key, value in {
            "schema_version": 1, "stage_id": stage, "attempt_id": attempt.name,
            "plan_sha256": result["plan_sha256"], "manifest_sha256": result["manifest_sha256"],
            "scientific_validity": "not_assessed"}.items()):
            raise WorkflowError("Review is not bound to this exact plan/attempt/manifest")
        reviewer = review.get("reviewer", {})
        if any(not isinstance(reviewer.get(key), str) or not reviewer[key].strip() for key in ("name", "harness", "role", "independence")):
            raise WorkflowError("Review must label reviewer identity, harness, role and independence")
        if self._requires_subagent_review(attempt):
            harness = reviewer["harness"].strip().casefold()
            if harness not in ("cursor", "human", "codex"):
                raise WorkflowError("Chartered review requires Cursor subagent or an actual human/Codex reviewer")
            if harness in ("cursor", "codex"):
                agents = [review.get(key) for key in ("author_agent", "reviewer_agent")]
                if (any(not isinstance(agent, str) or not agent.strip() for agent in agents)
                        or agents[0].strip().casefold() == agents[1].strip().casefold()):
                    raise WorkflowError("Agent review requires nonempty distinct author_agent and reviewer_agent")
                if review.get("review_mode") != "subagent_review":
                    raise WorkflowError("Agent review must record review_mode=subagent_review")
                # These are accountable caller-supplied labels, not proof of
                # independent identities, contexts, or scientific correctness.
        report = review.get("report", {})
        report_path = confined(self.root, report.get("path"), must_exist=True) if live_sources else confined(attempt, "report.md", must_exist=True)
        if sha256(report_path) != report.get("sha256"):
            raise WorkflowError("Reviewed report changed")
        evidence = {f"{task['id']}/{name}" for task in manifest["tasks"] for name in task["output_hashes"]}
        for key, plan_key in (("criteria", "exit_criteria"), ("predictions", "predictions")):
            rows = review.get(key)
            if not isinstance(rows, list) or [r.get("id") for r in rows] != [r["id"] for r in plan[plan_key]]:
                raise WorkflowError(f"Review must score every registered {key} exactly once")
            for row in rows:
                refs = row.get("evidence")
                if not isinstance(refs, list) or not refs or not set(refs).issubset(evidence):
                    raise WorkflowError(f"Review {row['id']} needs registered output evidence")
                if key == "criteria" and row.get("verdict") not in ("pass", "partial", "fail"):
                    raise WorkflowError(f"Review criterion is unscored: {row['id']}")
                if key == "criteria":
                    criterion = next(item for item in plan["exit_criteria"] if item["id"] == row["id"])
                    if not {ref.split("/", 1)[0] for ref in refs}.issubset(criterion["task_ids"]):
                        raise WorkflowError(f"Criterion {row['id']} cites a task outside its registered evidence")
                if key == "predictions" and (not isinstance(row.get("outcome"), str) or not row["outcome"].strip()):
                    raise WorkflowError(f"Prediction is unscored: {row['id']}")
        if not isinstance(review.get("limitations"), list) or not review["limitations"] or any(not isinstance(v, str) or not v.strip() for v in review["limitations"]):
            raise WorkflowError("Review needs explicit limitations")
        if review.get("verdict") not in ("accept", "revise", "reject"):
            raise WorkflowError("Review needs a verdict")

    def review(self, stage: str, review_path: Path, attempt_id: str | None = None, *, historical: bool = False) -> dict[str, Any]:
        attempt = self.attempt(stage, attempt_id)
        review = load_json(review_path)
        if historical and review.get("verdict") not in ("revise", "reject"):
            raise WorkflowError("Historical review import only permits revise or reject; accepting review requires live sources")
        self._check_review(stage, attempt, review, verify_live_sources=not historical)
        with self.lock(stage):
            target = attempt / "review.json"
            if target.exists():
                raise WorkflowError("Review already frozen; new evidence/review requires a new attempt")
            shutil.copyfile(confined(self.root, review["report"]["path"], must_exist=True), attempt / "report.md")
            if sha256(attempt / "report.md") != review["report"]["sha256"]:
                raise WorkflowError("Report changed while freezing review")
            write_json(target, review, exclusive=True)
        return {"stage_id": stage, "attempt_id": attempt.name, "review_recorded": True,
                "verdict": review["verdict"], "scientific_validity": "not_assessed"}

    def close(self, stage: str, attempt_id: str | None = None) -> dict[str, Any]:
        attempt = self.attempt(stage, attempt_id)
        review = load_json(attempt / "review.json")
        self._check_review(stage, attempt, review)
        if review["verdict"] != "accept" or any(row["verdict"] != "pass" for row in review["criteria"]):
            raise WorkflowError("Operational closure needs accepted review and all exit criteria passing")
        for dep in self.stage(stage)["depends_on"]:
            self.verify_closure(dep)
        result = {"schema_version": 1, "stage_id": stage, "attempt_id": attempt.name,
                  "closed_at": utc_now(), "status": "operationally_closed",
                  "manifest_sha256": sha256(attempt / "manifest.json"),
                  "review_sha256": sha256(attempt / "review.json"),
                  "dependency_closure_hashes": load_json(attempt / "manifest.json")["dependency_closure_hashes"],
                  "scientific_validity": "not_assessed"}
        with self.lock(stage):
            if self.closure_path(stage).exists():
                raise WorkflowError("Closure already exists; append a new stage rather than overwrite history")
            write_json(self.closure_path(stage), result, exclusive=True)
        return result

    def verify_closure(self, stage: str, _ancestors: set[str] | None = None) -> dict[str, Any]:
        ancestors = set(_ancestors or ())
        if stage in ancestors:
            raise WorkflowError(f"Closure dependency cycle at {stage}")
        ancestors.add(stage)
        closure = load_json(self.closure_path(stage))
        expected = {"schema_version": 1, "stage_id": stage,
                    "status": "operationally_closed", "scientific_validity": "not_assessed"}
        if any(closure.get(key) != value for key, value in expected.items()):
            raise WorkflowError(f"Invalid operational closure identity/authority: {stage}")
        attempt = self.attempt(stage, closure.get("attempt_id"))
        manifest = load_json(attempt / "manifest.json")
        dependency_hashes = closure.get("dependency_closure_hashes")
        if not isinstance(dependency_hashes, dict) or dependency_hashes != manifest.get("dependency_closure_hashes") or set(dependency_hashes) != set(self.stage(stage)["depends_on"]):
            raise WorkflowError(f"Closed dependency bindings changed: {stage}")
        for dep, digest in dependency_hashes.items():
            if sha256(self.closure_path(dep)) != digest:
                raise WorkflowError(f"Dependency closure changed: {stage} -> {dep}")
            self.verify_closure(dep, ancestors)
        # A closed stage records historical inputs; later source changes do not erase it.
        self.verify(stage, attempt.name, live_sources=False)
        if sha256(attempt / "manifest.json") != closure.get("manifest_sha256") or sha256(attempt / "review.json") != closure.get("review_sha256"):
            raise WorkflowError(f"Closed evidence changed: {stage}")
        review = load_json(attempt / "review.json")
        self._check_review(stage, attempt, review, live_sources=False)
        if review.get("verdict") != "accept" or review.get("scientific_validity") != "not_assessed" or not review.get("criteria") or any(item.get("verdict") != "pass" for item in review["criteria"]):
            raise WorkflowError(f"Closed review lacks operational acceptance: {stage}")
        if sha256(confined(attempt, "report.md", must_exist=True)) != review.get("report", {}).get("sha256"):
            raise WorkflowError(f"Closed report snapshot changed: {stage}")
        return closure
