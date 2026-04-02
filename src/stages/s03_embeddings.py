"""Stage 03 — Embedding extraction via Ollama or vLLM (async, parallel)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.cache import compute_hash, is_cached, write_hash
from src.config import Settings


# ── Ollama backend ────────────────────────────────────────────────────────────

async def _embed_ollama_batch(
    session: aiohttp.ClientSession,
    texts: list[str],
    model: str,
    url: str,
) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    async with session.post(f"{url}/api/embed", json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["embeddings"]


# ── vLLM backend (OpenAI-compatible) ──────────────────────────────────────────

async def _embed_vllm_batch(
    session: aiohttp.ClientSession,
    texts: list[str],
    model: str,
    url: str,
) -> list[list[float]]:
    payload = {"model": model, "input": texts}
    async with session.post(f"{url}/v1/embeddings", json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [d["embedding"] for d in sorted_data]


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def _embed_all(
    texts: list[str],
    cfg: Settings,
    logger: logging.Logger,
) -> np.ndarray:
    batch_size = cfg.EMBEDDING_BATCH_SIZE
    semaphore = asyncio.Semaphore(cfg.EMBEDDING_MAX_CONCURRENT)

    batches: list[list[str]] = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i : i + batch_size])

    embed_fn = _embed_ollama_batch if cfg.EMBEDDING_BACKEND == "ollama" else _embed_vllm_batch
    base_url = cfg.OLLAMA_URL if cfg.EMBEDDING_BACKEND == "ollama" else cfg.VLLM_URL
    model = cfg.EMBEDDING_MODEL

    all_embeddings: list[list[float]] = [[] for _ in range(len(texts))]
    progress = tqdm(total=len(batches), desc="Embedding batches", unit="batch")

    async def _process(batch_idx: int, batch: list[str]) -> None:
        async with semaphore:
            embs = await embed_fn(session, batch, model, base_url)
            start = batch_idx * batch_size
            for j, emb in enumerate(embs):
                all_embeddings[start + j] = emb
            progress.update(1)

    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_process(i, b) for i, b in enumerate(batches)]
        await asyncio.gather(*tasks)

    progress.close()
    return np.array(all_embeddings, dtype=np.float32)


def _texts_hash(texts: list[str], model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    for t in texts:
        h.update(t.encode())
    return h.hexdigest()[:16]


def run(cfg: Settings, logger: logging.Logger, *, ctx: dict[str, Any] | None = None) -> None:
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
        logger.info(
            "Extracting embeddings: backend=%s  model=%s  texts=%d",
            cfg.EMBEDDING_BACKEND, cfg.EMBEDDING_MODEL, len(all_texts),
        )
        all_emb = asyncio.run(_embed_all(all_texts, cfg, logger))
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
        write_hash(out_dir, h)
        logger.info("Stage 03 artifacts saved to %s", out_dir)

    if ctx is not None:
        ctx["emb_train"] = emb_train
        ctx["emb_val"] = emb_val
        ctx["emb_test"] = emb_test
