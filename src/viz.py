"""Plot subsampling. Histograms still use every row; scatters and n-grams do not.

UMAP and exhaustive n-gram counters on the full training split are a memory
hazard on this machine. The cap is fixed before any model curve is read.
It does not enter a preregistered estimand.
"""

from __future__ import annotations

import numpy as np


PLOT_ROWS = 20000
PLOT_SEED = 42


def stratified_indices(y: np.ndarray, limit: int = PLOT_ROWS, seed: int = PLOT_SEED) -> np.ndarray:
    labels = np.asarray(y)
    count = len(labels)
    if count <= limit:
        return np.arange(count)
    classes, sizes = np.unique(labels, return_counts=True)
    raw = sizes / sizes.sum() * limit
    take = np.floor(raw).astype(int)
    remaining = int(limit - take.sum())
    order = np.argsort(-(raw - take))
    for cursor in range(remaining):
        take[order[cursor % len(take)]] += 1
    rng = np.random.RandomState(seed)
    chosen = []
    for class_id, size in zip(classes, take):
        pool = np.flatnonzero(labels == class_id)
        size = min(int(size), len(pool))
        if size:
            chosen.append(rng.choice(pool, size=size, replace=False))
    return np.sort(np.concatenate(chosen))
