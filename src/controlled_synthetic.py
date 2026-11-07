"""Synthetic designs for the controlled double-descent stages.

The generator is deterministic from the protocol salt, phase and seed channels.
Discovery and confirmation channels differ. No row is downloaded.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PHASE_CODE = {"discovery": 1, "confirmation": 2}
KIND_CODE = {
    "data": 1,
    "noise": 2,
    "feature": 3,
    "model": 4,
    "eval": 5,
    "eval_noise": 6,
}


def channels(protocol: dict[str, Any], phase: str, kind: str, seed: int) -> list[int]:
    if phase not in PHASE_CODE:
        raise ValueError(f"Unknown phase: {phase}")
    if kind not in KIND_CODE:
        raise ValueError(f"Unknown seed kind: {kind}")
    salt = protocol["protocol_salt"]
    if isinstance(salt, bool) or not isinstance(salt, int) or salt < 0:
        raise ValueError("protocol_salt must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return [salt, PHASE_CODE[phase], KIND_CODE[kind], seed]


def generator(protocol: dict[str, Any], phase: str, kind: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(channels(protocol, phase, kind, seed)))


def column_std(protocol: dict[str, Any], spectrum: str, p_max: int) -> np.ndarray:
    index = np.arange(p_max, dtype=np.float64)
    if spectrum == "isotropic":
        return np.ones(p_max, dtype=np.float64)
    if spectrum == "geometric":
        return np.sqrt(np.power(np.float64(protocol["geometric_ratio"]), index))
    raise ValueError(f"Unknown spectrum: {spectrum}")


def teacher_beta(protocol: dict[str, Any], data_rng: np.random.Generator, p_max: int) -> np.ndarray:
    weights = np.power(np.float64(protocol["beta_geometric_ratio"]), np.arange(p_max, dtype=np.float64))
    signs = data_rng.integers(0, 2, size=p_max).astype(np.float64) * 2.0 - 1.0
    beta = signs * weights
    norm = float(np.linalg.norm(beta))
    if not math.isfinite(norm) or norm <= 0:
        raise RuntimeError("Teacher norm is not positive")
    beta /= norm
    return beta


def sigma_from_snr(snr: float) -> float:
    if not math.isfinite(snr) or snr <= 0:
        raise ValueError("snr must be positive")
    return 1.0 / math.sqrt(float(snr))


def prepare_draw(protocol: dict[str, Any], phase: str, data_seed: int) -> dict[str, Any]:
    """Draw beta, covariates, label noise and one frozen feature matrix.

    The model channel is addressed so it stays independent, and it is not consumed.
    """
    p_max = int(protocol["p_max"])
    n_train = int(protocol["n_train"])
    data_rng = generator(protocol, phase, "data", data_seed)
    noise_rng = generator(protocol, phase, "noise", data_seed)
    feature_rng = generator(protocol, phase, "feature", data_seed)
    _model_rng = generator(protocol, phase, "model", data_seed)
    del _model_rng
    beta = teacher_beta(protocol, data_rng, p_max)
    covariates = data_rng.standard_normal((n_train, p_max))
    label_noise = noise_rng.standard_normal(n_train)
    weight_scale = math.sqrt(2.0 / p_max)
    feature_weights = feature_rng.standard_normal((p_max, p_max)) * weight_scale
    return {
        "data_seed": int(data_seed),
        "beta": beta,
        "covariates": covariates,
        "label_noise": label_noise,
        "feature_weights": feature_weights,
        "channels": {
            "data": channels(protocol, phase, "data", data_seed),
            "noise": channels(protocol, phase, "noise", data_seed),
            "feature": channels(protocol, phase, "feature", data_seed),
            "model": channels(protocol, phase, "model", data_seed),
        },
    }


def prepare_evaluation(protocol: dict[str, Any], phase: str) -> dict[str, Any]:
    n_eval = int(protocol["n_eval"])
    p_max = int(protocol["p_max"])
    design_seed = int(protocol[f"{phase}_eval_seed"])
    noise_seed = int(protocol[f"{phase}_eval_noise_seed"])
    covariates = generator(protocol, phase, "eval", design_seed).standard_normal((n_eval, p_max))
    label_noise = generator(protocol, phase, "eval_noise", noise_seed).standard_normal(n_eval)
    return {
        "eval_seed": design_seed,
        "eval_noise_seed": noise_seed,
        "covariates": covariates,
        "label_noise": label_noise,
        "channels": {
            "eval": channels(protocol, phase, "eval", design_seed),
            "eval_noise": channels(protocol, phase, "eval_noise", noise_seed),
        },
    }


def design_matrix(covariates: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return np.asarray(covariates, dtype=np.float64) * np.asarray(scales, dtype=np.float64)


def latent_standardizer(train_latent: np.ndarray) -> dict[str, np.ndarray]:
    """Train-only location and scale of the latent Gaussian columns.

    Spectrum scales are applied after this step and are not divided out again.
    """
    matrix = np.asarray(train_latent, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("latent design must have at least two rows")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0, ddof=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise RuntimeError("A latent training column has nonpositive scale")
    return {"mean": mean, "scale": scale}


def apply_latent_spectrum(latent: np.ndarray, fitted: dict[str, np.ndarray],
                          spectrum_scale: np.ndarray) -> np.ndarray:
    matrix = np.asarray(latent, dtype=np.float64)
    scales = np.asarray(spectrum_scale, dtype=np.float64)
    if matrix.shape[1] != fitted["mean"].shape[0] or scales.shape != fitted["mean"].shape:
        raise ValueError("spectrum scales do not match the latent width")
    if np.any(scales <= 0) or not np.isfinite(scales).all():
        raise ValueError("spectrum scales must be positive and finite")
    standardized = (matrix - fitted["mean"]) / fitted["scale"]
    return standardized * scales


def teacher_signal(features: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(features, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)
