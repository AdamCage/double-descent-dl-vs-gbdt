"""Stage 03 — corpus embeddings via RouterAI (qwen/qwen3-embedding-8b)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings, get_settings

ROUTERAI_BACKEND = "routerai"
ROUTERAI_MODEL = "qwen/qwen3-embedding-8b"
_PARTIAL_FILES = ("partial.dat", "partial_flags.npy", "partial_meta.json")
_RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


class RouterAIHTTPError(RuntimeError):
    def __init__(self, status: int, detail: str = "") -> None:
        super().__init__(f"routerai HTTP {status}")
        self.status = status
        self.detail = detail


def _api_key(cfg: Settings) -> str:
    value = cfg.ROUTERAI_API_KEY.get_secret_value().strip()
    if not value:
        raise RuntimeError("ROUTERAI_API_KEY is empty")
    return value


def _require_routerai(cfg: Settings) -> str:
    if cfg.EMBEDDING_BACKEND != ROUTERAI_BACKEND:
        raise RuntimeError(
            f"refusing embedding backend {cfg.EMBEDDING_BACKEND}; routerai is required"
        )
    if cfg.EMBEDDING_MODEL != ROUTERAI_MODEL:
        raise RuntimeError(
            f"refusing embedding model {cfg.EMBEDDING_MODEL}; {ROUTERAI_MODEL} is required"
        )
    return _api_key(cfg)


def _redact(text: str, secret: str) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    text = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"(?i)(api[_-]?key[\"']?\s*[:=]\s*[\"']?)[^\s\"']+",
        r"\1[REDACTED]",
        text,
    )
    return text


def _partial_dir(out_dir: Path, backend: str, model: str) -> Path:
    """Model-specific partials. The 768-d embeddinggemma files stay in out_dir."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{backend}__{model}").strip("._")
    path = (out_dir / "partials" / slug).resolve()
    root = out_dir.resolve()
    if path == root or root not in path.parents:
        raise RuntimeError("refusing partial cache path outside the embeddings directory")
    return path


async def _embed_routerai_batch(
    session: aiohttp.ClientSession,
    texts: list[str],
    model: str,
    base_url: str,
    api_key: str,
) -> list[list[float]]:
    endpoint = base_url.rstrip("/") + "/embeddings"
    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.post(endpoint, json=payload, headers=headers) as resp:
            if resp.status >= 400:
                body = await resp.text()
                detail = re.sub(r"\s+", " ", _redact(body, api_key)).strip()[:180]
                if api_key and api_key in detail:
                    detail = ""
                raise RouterAIHTTPError(resp.status, detail)
            try:
                data = await resp.json()
            except (aiohttp.ContentTypeError, json.JSONDecodeError) as exc:
                raise RuntimeError("routerai response was not JSON") from exc
    except RouterAIHTTPError:
        raise
    except TimeoutError as exc:
        raise RuntimeError("routerai timeout") from exc
    except aiohttp.ClientError as exc:
        raise RuntimeError(f"routerai connection error: {type(exc).__name__}") from exc

    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise RuntimeError("routerai response missing data")
    if rows and isinstance(rows[0], dict) and "index" in rows[0]:
        rows = sorted(rows, key=lambda row: row["index"])
    vectors: list[list[float]] = []
    try:
        for row in rows:
            embedding = row["embedding"]
            if not isinstance(embedding, list):
                raise TypeError("embedding is not a list")
            vectors.append(embedding)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("routerai response missing embeddings") from exc
    return vectors


async def _embed_all(
    texts: list[str],
    cfg: Settings,
    logger: logging.Logger,
    out_dir: Path,
    cache_hash: str,
) -> np.ndarray:
    api_key = _require_routerai(cfg)
    batch_size = cfg.EMBEDDING_BATCH_SIZE
    semaphore = asyncio.Semaphore(cfg.EMBEDDING_MAX_CONCURRENT)

    batches: list[list[str]] = [
        texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
    ]
    partial_dir = _partial_dir(out_dir, cfg.EMBEDDING_BACKEND, cfg.EMBEDDING_MODEL)
    partial_dir.mkdir(parents=True, exist_ok=True)
    data_path = partial_dir / "partial.dat"
    flags_path = partial_dir / "partial_flags.npy"
    meta_path = partial_dir / "partial_meta.json"
    probe_path = partial_dir / "probe.json"
    expected_dim: int | None = None
    if probe_path.exists():
        expected_dim = int(json.loads(probe_path.read_text(encoding="utf-8"))["dim"])

    matrix: np.memmap | None = None
    flags: np.ndarray | None = None
    if meta_path.exists() and data_path.exists() and flags_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        flags_loaded = np.load(flags_path)
        same_cache = (
            meta.get("hash") == cache_hash
            and meta.get("n_texts") == len(texts)
            and meta.get("backend") == cfg.EMBEDDING_BACKEND
            and meta.get("model") == cfg.EMBEDDING_MODEL
            and meta.get("batch_size") == batch_size
            and int(flags_loaded.shape[0]) == len(batches)
        )
        if same_cache and expected_dim is not None and int(meta.get("dim", -1)) != expected_dim:
            raise RuntimeError("partial cache dimension does not match the probe")
        if same_cache:
            matrix = np.memmap(
                data_path,
                dtype=np.float32,
                mode="r+",
                shape=(int(meta["n_texts"]), int(meta["dim"])),
            )
            flags = flags_loaded
            logger.info(
                "Resuming embeddings: %d/%d batches already stored",
                int(flags.sum()),
                len(batches),
            )

    alloc_lock = asyncio.Lock()
    done = 0 if flags is None else int(flags.sum())
    progress = tqdm(total=len(batches), desc="Embedding batches", unit="batch")

    async def _process(batch_idx: int, batch: list[str]) -> None:
        nonlocal matrix, flags, done
        if flags is not None and flags[batch_idx]:
            progress.update(1)
            return
        async with semaphore:
            last_error: Exception | None = None
            embs: list[list[float]] | None = None
            for attempt in range(8):
                try:
                    embs = await _embed_routerai_batch(
                        session, batch, cfg.EMBEDDING_MODEL, cfg.ROUTERAI_BASE_URL, api_key,
                    )
                    break
                except RouterAIHTTPError as exc:
                    last_error = exc
                    if exc.status not in _RETRY_STATUSES:
                        raise
                    logger.warning(
                        "embedding batch %d HTTP %s, retry %d",
                        batch_idx, exc.status, attempt + 1,
                    )
                    await asyncio.sleep(min(60, 2 ** attempt))
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "embedding batch %d error %s, retry %d",
                        batch_idx, type(exc).__name__, attempt + 1,
                    )
                    await asyncio.sleep(min(60, 2 ** attempt))
            if embs is None:
                raise RuntimeError(f"embedding batch {batch_idx} failed") from last_error
            if len(embs) != len(batch):
                raise RuntimeError(
                    f"embedding batch {batch_idx} returned {len(embs)} vectors for {len(batch)} texts"
                )
        async with alloc_lock:
            width = len(embs[0])
            if expected_dim is not None and width != expected_dim:
                raise RuntimeError(
                    f"embedding width {width} does not match probe dimension {expected_dim}"
                )
            if matrix is None:
                matrix = np.memmap(
                    data_path, dtype=np.float32, mode="w+", shape=(len(texts), width),
                )
                flags = np.zeros(len(batches), dtype=np.uint8)
                meta_path.write_text(json.dumps({
                    "hash": cache_hash,
                    "n_texts": len(texts),
                    "dim": width,
                    "batch_size": batch_size,
                    "backend": cfg.EMBEDDING_BACKEND,
                    "model": cfg.EMBEDDING_MODEL,
                }), encoding="utf-8")
            elif matrix.shape[1] != width:
                raise RuntimeError("embedding width changed between batches")
            start = batch_idx * batch_size
            matrix[start:start + len(embs)] = np.asarray(embs, dtype=np.float32)
            matrix.flush()
            assert flags is not None
            flags[batch_idx] = 1
            np.save(flags_path, flags)
            done = int(flags.sum())
        progress.update(1)
        if done == 1 or done % 50 == 0 or done == len(batches):
            logger.info("Embedding progress: %d/%d batches", done, len(batches))

    timeout = aiohttp.ClientTimeout(total=300, sock_connect=30, sock_read=300)
    connector = aiohttp.TCPConnector(limit=cfg.EMBEDDING_MAX_CONCURRENT)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await asyncio.gather(*(_process(i, batch) for i, batch in enumerate(batches)))

    progress.close()
    if matrix is None:
        raise RuntimeError("no embeddings were returned")
    return matrix


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
    _require_routerai(cfg)
    out_dir = cfg.result_path / "03_embeddings"

    splits: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        if ctx and f"{name}_df" in ctx:
            splits[name] = ctx[f"{name}_df"]
        else:
            splits[name] = pd.read_parquet(cfg.result_path / "01_data" / f"{name}.parquet")

    all_texts = pd.concat([splits["train"], splits["val"], splits["test"]])["text"].tolist()
    h = compute_hash(
        backend=cfg.EMBEDDING_BACKEND,
        model=cfg.EMBEDDING_MODEL,
        batch_size=cfg.EMBEDDING_BATCH_SIZE,
        n_texts=len(all_texts),
        first_text_hash=hashlib.sha256(all_texts[0].encode()).hexdigest()[:8],
        last_text_hash=hashlib.sha256(all_texts[-1].encode()).hexdigest()[:8],
    )

    if is_cached(out_dir, h):
        logger.info("Stage 03 cached — loading embeddings from .npy")
        emb_train = np.load(out_dir / "train.npy")
        emb_val = np.load(out_dir / "val.npy")
        emb_test = np.load(out_dir / "test.npy")
    else:
        partial_dir = _partial_dir(out_dir, cfg.EMBEDDING_BACKEND, cfg.EMBEDDING_MODEL)
        n_batches = (len(all_texts) + cfg.EMBEDDING_BATCH_SIZE - 1) // cfg.EMBEDDING_BATCH_SIZE
        logger.info(
            "Extracting embeddings: backend=%s  model=%s  texts=%d  batches=%d  concurrency=%d  partial=%s",
            cfg.EMBEDDING_BACKEND,
            cfg.EMBEDDING_MODEL,
            len(all_texts),
            n_batches,
            cfg.EMBEDDING_MAX_CONCURRENT,
            partial_dir,
        )
        all_emb = asyncio.run(_embed_all(all_texts, cfg, logger, out_dir, h))
        logger.info("Embedding shape: %s", all_emb.shape)

        n_train = len(splits["train"])
        n_val = len(splits["val"])
        emb_train = all_emb[:n_train]
        emb_val = all_emb[n_train : n_train + n_val]
        emb_test = all_emb[n_train + n_val :]

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "train.npy", emb_train)
        np.save(out_dir / "val.npy", emb_val)
        np.save(out_dir / "test.npy", emb_test)
        for leftover in _PARTIAL_FILES:
            (partial_dir / leftover).unlink(missing_ok=True)
        write_hash(out_dir, h)
        logger.info("Stage 03 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["emb_train"] = emb_train
        ctx["emb_val"] = emb_val
        ctx["emb_test"] = emb_test


def _write_probe(cfg: Settings, dim: int, text: str) -> Path:
    partial_dir = _partial_dir(cfg.result_path / "03_embeddings", cfg.EMBEDDING_BACKEND, cfg.EMBEDDING_MODEL)
    partial_dir.mkdir(parents=True, exist_ok=True)
    path = partial_dir / "probe.json"
    path.write_text(json.dumps({
        "backend": cfg.EMBEDDING_BACKEND,
        "model": cfg.EMBEDDING_MODEL,
        "base_url": cfg.ROUTERAI_BASE_URL,
        "dim": dim,
        "text_chars": len(text),
    }), encoding="utf-8")
    return path


def probe(cfg: Settings | None = None, text: str = "ping") -> int:
    cfg = cfg or get_settings()
    api_key = _require_routerai(cfg)

    async def _once() -> int:
        timeout = aiohttp.ClientTimeout(total=60, sock_connect=30, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            vectors = await _embed_routerai_batch(
                session, [text], cfg.EMBEDDING_MODEL, cfg.ROUTERAI_BASE_URL, api_key,
            )
        if len(vectors) != 1 or not vectors[0]:
            raise RuntimeError("routerai probe returned no vector")
        return len(vectors[0])

    return asyncio.run(_once())


def _main() -> None:
    parser = argparse.ArgumentParser(description="Probe the RouterAI embedding model")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if not args.probe:
        parser.error("pass --probe")
    cfg = get_settings()
    text = "ping"
    try:
        dim = probe(cfg, text)
    except RouterAIHTTPError as exc:
        print(f"probe_failed http_status={exc.status}")
        raise SystemExit(1)
    path = _write_probe(cfg, dim, text)
    print(
        f"probe_ok dim={dim} model={cfg.EMBEDDING_MODEL} "
        f"backend={cfg.EMBEDDING_BACKEND} probe_file={path}"
    )


if __name__ == "__main__":
    try:
        _main()
    except RouterAIHTTPError as exc:
        print(f"probe_failed http_status={exc.status}", file=sys.stderr)
        raise SystemExit(1)
