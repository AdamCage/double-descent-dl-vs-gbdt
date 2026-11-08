"""Portable content-addressed evidence ZIP; repeated paths restore as full files.

Used when frozen attempts contain repeated source/evidence snapshots. The index
retains every path, size and SHA-256; each distinct byte stream is stored once.
No links, environment files, databases, overwrite, network or model fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

INDEX = "EVIDENCE-INDEX.json"
TOOL = "RESTORE.py"
MAX_FILES = 50000
MAX_BYTES = 20 * 1024**3


def digest(stream) -> str:
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024**2), b""):
        value.update(block)
    return value.hexdigest()


def safe_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Unsafe evidence path")
    pure = PurePosixPath(value)
    if not pure.parts or pure.is_absolute() or value != pure.as_posix() or any(p in (".", "..", "") for p in pure.parts):
        raise ValueError("Unsafe evidence path")
    reserved = {"con", "prn", "aux", "nul"} | {f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1, 10)}
    if any(p.endswith((".", " ")) or p.split(".", 1)[0].casefold() in reserved or any(ord(c) < 32 for c in p) for p in pure.parts):
        raise ValueError("Nonportable Windows evidence path")
    if any(p.casefold() in {".env", ".git", ".venv"} or p.casefold().startswith(".env.") for p in pure.parts):
        raise ValueError("Environment/VCS files are not evidence")
    if any(".sqlite3" in p.casefold() for p in pure.parts):
        raise ValueError("Database requires coherent backup")
    return value


def validate_index(index: dict) -> list[dict]:
    if index.get("schema_version") != 1 or index.get("format") != "sha256-blobs-v1":
        raise ValueError("Unsupported evidence index")
    rows = index.get("files")
    if not isinstance(rows, list) or not 0 < len(rows) <= MAX_FILES:
        raise ValueError("Invalid file count")
    names = set()
    total = 0
    for row in rows:
        name = safe_path(row["path"])
        # Windows portability: prevent file/directory aliases and case collisions.
        folded = name.casefold()
        if folded in names:
            raise ValueError("Duplicate evidence path")
        names.add(folded)
        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Invalid content digest")
        size = row.get("bytes")
        if type(size) is not int or size < 0:
            raise ValueError("Invalid file size")
        total += size
    if total > MAX_BYTES or any(parent.as_posix().casefold() in names for row in rows for parent in PurePosixPath(row["path"]).parents if str(parent) != "."):
        raise ValueError("Evidence size or file/directory conflict")
    sizes = {}
    for row in rows:
        previous = sizes.setdefault(row["sha256"], row["bytes"])
        if previous != row["bytes"]:
            raise ValueError("One digest has inconsistent sizes")
    return rows


def verify(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or INDEX not in names or TOOL not in names:
            raise ValueError("Missing index or duplicate ZIP entry")
        if archive.getinfo(INDEX).file_size > 32 * 1024**2:
            raise ValueError("Oversized index")
        index = json.loads(archive.read(INDEX))
        rows = validate_index(index)
        if archive.getinfo(TOOL).file_size > 1024**2 or hashlib.sha256(archive.read(TOOL)).hexdigest() != index.get("restore_tool_sha256"):
            raise ValueError("Restore tool is unbound or oversized")
        blobs = {row["sha256"]: row["bytes"] for row in rows}
        if set(names) != {INDEX, TOOL} | {f"blobs/{sha}" for sha in blobs}:
            raise ValueError("ZIP does not exactly cover declared content")
        for entry in entries:
            mode = entry.external_attr >> 16
            if entry.is_dir() or stat.S_IFMT(mode) not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                raise ValueError("Links/directories/encrypted entries not allowed")
        for sha, size in blobs.items():
            entry = archive.getinfo(f"blobs/{sha}")
            if entry.file_size != size:
                raise ValueError("Content size mismatch")
            with archive.open(entry) as stream:
                if digest(stream) != sha:
                    raise ValueError("Content digest mismatch")
    return {"verified": True, "files": len(rows), "unique_blobs": len(blobs),
            "restored_bytes": sum(row["bytes"] for row in rows),
            "unique_bytes": sum(blobs.values()), "scientific_validity": "not_assessed"}


def create(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir() or output.is_relative_to(source):
        raise ValueError("Require source directory and external exclusive output")
    rows, representatives = [], {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or not path.resolve().is_relative_to(source):
            raise ValueError("Source link/junction is not allowed")
        relative = safe_path(path.relative_to(source).as_posix())
        if path.is_file():
            with path.open("rb") as stream:
                sha = digest(stream)
            rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha})
            representatives.setdefault(sha, path)
    tool = Path(__file__).read_bytes()
    index = {"schema_version": 1, "format": "sha256-blobs-v1", "files": rows,
             "restore_tool_sha256": hashlib.sha256(tool).hexdigest(),
             "scientific_validity": "not_assessed"}
    validate_index(index)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.writestr(INDEX, json.dumps(index, sort_keys=True, indent=2) + "\n")
        archive.writestr(TOOL, tool)
        for sha, path in sorted(representatives.items()):
            archive.write(path, f"blobs/{sha}")
    return verify(output)


def restore(archive_path: Path, output: Path) -> dict:
    result = verify(archive_path)
    if output.exists():
        raise ValueError("Restore requires a new output directory")
    output.mkdir(parents=True, exist_ok=False)
    root = output.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for row in json.loads(archive.read(INDEX))["files"]:
            path = root / safe_path(row["path"])
            if not path.resolve().is_relative_to(root):
                raise ValueError("Restore escaped target")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(f"blobs/{row['sha256']}") as src, path.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024**2)
            with path.open("rb") as stream:
                if digest(stream) != row["sha256"]:
                    raise ValueError("Restored file differs")
    return {**result, "restored": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("create")
    pack.add_argument("--source", required=True, type=Path)
    pack.add_argument("--output", required=True, type=Path)
    check = commands.add_parser("verify")
    check.add_argument("archive", type=Path)
    unpack = commands.add_parser("restore")
    unpack.add_argument("archive", type=Path)
    unpack.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (create(args.source, args.output) if args.command == "create" else
              verify(args.archive) if args.command == "verify" else restore(args.archive, args.output))
    print(json.dumps(result, indent=2))
