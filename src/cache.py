"""Hash-based caching: skip expensive computation when inputs haven't changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _stable_hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
    return h.hexdigest()[:16]


def compute_hash(**kw: Any) -> str:
    # Формула ниже заморожена: s03 пишет её в partial_meta.json живого прогона.
    return _stable_hash(*sorted(kw.items()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_digest(path: Path) -> str:
    """Короткий хеш исходника этапа. В ключ s03 не входит."""
    return sha256_file(path)[:16]


def is_cached(stage_dir: Path, expected_hash: str) -> bool:
    hf = stage_dir / "cache_hash.json"
    if not hf.exists():
        return False
    try:
        stored = json.loads(hf.read_text(encoding="utf-8"))
        return stored.get("hash") == expected_hash
    except Exception:
        return False


def write_hash(stage_dir: Path, hash_value: str) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "cache_hash.json").write_text(
        json.dumps({"hash": hash_value}), encoding="utf-8"
    )
