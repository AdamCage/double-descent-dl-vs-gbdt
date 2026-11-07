"""Create and verify portable archives of an explicit artifact directory.

No secret environment, database live-copy, external symlink or implicit result
directory inclusion. Archives are exclusive-create; verification works detached
from the checkout and compares bytes against its own manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


MANIFEST = "BUNDLE-MANIFEST.json"


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def create_bundle(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir() or output.is_relative_to(source):
        raise ValueError("Use an existing source directory and an output outside it")
    files = []
    for path in sorted(source.rglob("*")):
        if (path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())
                or not path.resolve().is_relative_to(source)):
            raise ValueError(f"Symlink not allowed: {path}")
        parts = [part.casefold() for part in path.relative_to(source).parts]
        if any(part in {".env", ".git", ".venv"} or part.startswith(".env.") for part in parts):
            raise ValueError("Secret/source environment directories are not evidence bundles")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if ".sqlite3" in path.name or path.name == MANIFEST:
            raise ValueError("Database files need a coherent backup; reserved manifest name is disallowed")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": hash_file(path)})
    if not files:
        raise ValueError("Refusing an empty evidence bundle")
    manifest = {"schema_version": 1, "scientific_validity": "not_assessed", "files": files}
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    import io
    output.parent.mkdir(parents=True, exist_ok=True)
    # x mode is essential: a prior portable evidence package is never overwritten.
    with output.open("xb") as destination:
        with tarfile.open(fileobj=destination, mode="w:gz") as archive:
            info = tarfile.TarInfo(MANIFEST)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
            for row in files:
                archive.add(source / row["path"], arcname=row["path"], recursive=False)
    verify_bundle(output)
    return manifest


def verify_bundle(path: Path) -> dict:
    with tarfile.open(path, mode="r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or MANIFEST not in names:
            raise ValueError("Missing manifest or duplicate archive members")
        for member in members:
            pure = PurePosixPath(member.name)
            if (not member.isfile() or pure.is_absolute() or ".." in pure.parts
                    or "\\" in member.name or ":" in member.name):
                raise ValueError("Unsafe archive member")
        manifest = json.load(archive.extractfile(MANIFEST))
        rows = manifest.get("files", [])
        if manifest.get("schema_version") != 1 or not rows:
            raise ValueError("Invalid bundle manifest")
        expected = [row["path"] for row in rows]
        if len(expected) != len(set(expected)) or set(names) != set(expected) | {MANIFEST}:
            raise ValueError("Manifest coverage mismatch")
        for row in rows:
            member = archive.getmember(row["path"])
            handle = archive.extractfile(member)
            h = hashlib.sha256()
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(block)
            if member.size != row["bytes"] or h.hexdigest() != row["sha256"]:
                raise ValueError(f"Evidence mismatch: {row['path']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    pack = actions.add_parser("create")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    verify = actions.add_parser("verify")
    verify.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        manifest = (create_bundle(args.source, args.output) if args.action == "create"
                    else verify_bundle(args.archive))
    except (ValueError, OSError, tarfile.TarError, KeyError, TypeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({"verified": True, "files": len(manifest["files"]),
                      "scientific_validity": "not_assessed"}))


if __name__ == "__main__":
    main()
