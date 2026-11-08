"""Technical controls for finite-label-response studies of squared-loss boosting.

Fixed-partition leaf refitting and representation collision floors are known
controls. None of the functions in this module declares a scientific effect.
The operator acts on the actual native (float32-representable) label vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class Threshold:
    raw_feature: int
    probability: float
    value: float


@dataclass(frozen=True)
class NestedBinaryFeatures:
    train: np.ndarray
    evaluation: np.ndarray
    thresholds: tuple[Threshold, ...]
    prefix_budgets: tuple[int, ...]
    raw_feature_count: int
    quantile_method: str = "linear"

    def prefix(self, budget: int) -> tuple[np.ndarray, np.ndarray]:
        if budget not in self.prefix_budgets:
            raise ValueError("Budget is not a declared nested prefix")
        return self.train[:, :budget], self.evaluation[:, :budget]

    def manifest(self) -> dict[str, Any]:
        return {
            "fit_on": "training_covariates_only", "comparison": "raw_feature > threshold",
            "quantile_method": self.quantile_method,
            "raw_feature_count": self.raw_feature_count,
            "prefix_budgets": list(self.prefix_budgets),
            "thresholds": [{"column": i, "raw_feature": t.raw_feature,
                            "probability": t.probability, "value": t.value}
                           for i, t in enumerate(self.thresholds)],
            "duplicate_threshold_policy": "retain; nominal candidate budgets do not imply unique thresholds or signatures",
        }


def _matrix(value: Any, label: str, *, allow_empty: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 1 or (not allow_empty and array.shape[0] < 1) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite 2D matrix")
    return array


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def nested_binary_features(x_train: Any, x_evaluation: Any, raw_feature_count: int = 4,
                           threshold_levels: Iterable[int] = (1, 3, 7, 15)) -> NestedBinaryFeatures:
    """Freeze train-only quantiles in dyadic order, round-robin over raw features.

    For four raw features the declared levels produce prefixes 4, 12, 28, 60.
    Every raw feature is already represented in the first prefix. Tied quantiles
    remain tied columns; they must not be silently replaced after seeing data.
    """
    train = _matrix(x_train, "x_train")
    evaluation = _matrix(x_evaluation, "x_evaluation", allow_empty=True)
    if train.shape[1] != evaluation.shape[1]:
        raise ValueError("Train/evaluation raw dimensions differ")
    if type(raw_feature_count) is not int or not 1 <= raw_feature_count <= train.shape[1]:
        raise ValueError("raw_feature_count exceeds supplied raw features")
    levels = tuple(threshold_levels)
    if not levels or levels[0] != 1 or any(type(v) is not int or v < 1 or (v + 1) & v for v in levels) or tuple(sorted(set(levels))) != levels:
        raise ValueError("Threshold levels must increase through dyadic counts 2**k-1 and start at 1")
    probabilities = []
    depth = 1
    while len(probabilities) < levels[-1]:
        probabilities.extend(numerator / (2 ** depth) for numerator in range(1, 2 ** depth, 2))
        depth += 1
    thresholds = tuple(Threshold(feature, probability,
                                 float(np.quantile(train[:, feature], probability, method="linear")))
                       for probability in probabilities for feature in range(raw_feature_count))
    binary_train = np.column_stack([train[:, t.raw_feature] > t.value for t in thresholds]).astype(np.uint8)
    binary_evaluation = np.column_stack([evaluation[:, t.raw_feature] > t.value for t in thresholds]).astype(np.uint8)
    return NestedBinaryFeatures(_read_only(binary_train), _read_only(binary_evaluation), thresholds,
                                tuple(raw_feature_count * level for level in levels), raw_feature_count)


def canonicalize_targets(targets: Any) -> dict[str, np.ndarray]:
    """Expose native-label rounding; callers must save both nominal and actual y.

    In a finite perturbation study use delta = perturbed_native - base_native
    (promoted to float64). epsilon*z is the nominal direction and generally has
    a different rounding residual. A Gaussian Frobenius expectation for the
    nominal direction is not automatically exact for the rounded delta.
    """
    nominal = np.asarray(targets, dtype=np.float64)
    if nominal.ndim != 1 or not len(nominal) or not np.isfinite(nominal).all():
        raise ValueError("Targets must be a finite nonempty vector")
    with np.errstate(over="ignore", invalid="ignore"):
        native = nominal.astype(np.float32)
    if not np.isfinite(native).all():
        raise ValueError("Targets overflow native float32 representation")
    actual = native.astype(np.float64)
    return {"nominal": nominal.copy(), "native": native, "actual": actual,
            "rounding_delta": actual - nominal}


def _actual_targets(targets: Any, n_train: int) -> np.ndarray:
    canonical = canonicalize_targets(targets)
    if len(canonical["actual"]) != n_train:
        raise ValueError("Target count differs from training rows")
    if not np.array_equal(canonical["actual"], canonical["nominal"]):
        raise ValueError("Pass the actual float32-representable labels explicitly; canonicalize_targets exposes rounding")
    return canonical["actual"]


def _binary_pair(x_train: Any, x_evaluation: Any) -> tuple[np.ndarray, np.ndarray]:
    train = _matrix(x_train, "x_train")
    evaluation = _matrix(x_evaluation, "x_evaluation", allow_empty=True)
    if train.shape[1] != evaluation.shape[1]:
        raise ValueError("Binary feature dimensions differ")
    if not np.isin(train, (0, 1)).all() or not np.isin(evaluation, (0, 1)).all():
        raise ValueError("Control supports only explicit binary numeric features")
    return train, evaluation


def _threads(threads: int) -> None:
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError("threads must be an integer in 1..4")


def _data_hash(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    descriptor = json.dumps({"shape": list(value.shape), "dtype": value.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(descriptor + b"\0" + contiguous.tobytes()).hexdigest()


def validate_control_model(model: Any) -> dict[str, Any]:
    """Refuse settings that change the specified linear leaf-refitting rule."""
    params = model.get_all_params()
    required = {
        "loss_function": "RMSE", "depth": 2, "boosting_type": "Plain",
        "bootstrap_type": "No", "leaf_estimation_method": "Gradient",
        "leaf_estimation_iterations": 1, "leaf_estimation_backtracking": "No",
        "score_function": "L2", "boost_from_average": False,
        "border_count": 1, "grow_policy": "SymmetricTree", "task_type": "CPU",
    }
    for name, value in required.items():
        if params.get(name) != value:
            raise ValueError(f"Unsupported resolved control parameter {name}: {params.get(name)!r}")
    for name, expected in (("l2_leaf_reg", 1.0), ("random_strength", 0.0), ("rsm", 1.0),
                           ("model_shrink_rate", 0.0)):
        if name not in params or not math.isclose(float(params[name]), expected, rel_tol=0, abs_tol=1e-15):
            raise ValueError(f"Unsupported resolved control parameter {name}")
    actual_eta = float(params.get("learning_rate", float("nan")))
    if not math.isclose(actual_eta, float(np.float32(0.1)), rel_tol=0, abs_tol=1e-12):
        raise ValueError("Resolved learning rate differs from the specified float32(.1)")
    if params.get("langevin", False) or params.get("posterior_sampling", False):
        raise ValueError("Posterior/Langevin sampling is unsupported")
    if model.get_cat_feature_indices() or model.get_text_feature_indices() or model.get_embedding_feature_indices():
        raise ValueError("Only numeric binary features are supported")
    scale, bias = model.get_scale_and_bias()
    if not math.isclose(float(scale), 1, rel_tol=0, abs_tol=1e-15) or not math.isclose(float(bias), 0, rel_tol=0, abs_tol=1e-15):
        raise ValueError("Control requires unit scale and exactly zero initial bias")
    return {"resolved_parameters": params, "actual_learning_rate": actual_eta,
            "actual_l2_leaf_reg": float(params["l2_leaf_reg"]), "scale": float(scale), "bias": float(bias)}


def fit_control(x_train: Any, targets: Any, *, iterations: int = 64, seed: int = 0,
                threads: int = 4):
    """Fit only the tightly specified unweighted RMSE technical control.

    No sample/object/group weights or initial model are accepted. This is not a
    scientific runner. Fitting tests are opt-in, and never use study seeds.
    """
    _threads(threads)
    supplied_train = _matrix(x_train, "x_train")
    train, _ = _binary_pair(supplied_train, np.empty((0, supplied_train.shape[1])))
    actual_y = _actual_targets(targets, len(train))
    if type(iterations) is not int or iterations < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive iterations and nonnegative integer seed required")
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=iterations, depth=2, learning_rate=0.1, l2_leaf_reg=1.0,
        loss_function="RMSE", boosting_type="Plain", bootstrap_type="No",
        random_strength=0.0, leaf_estimation_method="Gradient",
        leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
        score_function="L2", boost_from_average=False, use_best_model=False,
        allow_writing_files=False, border_count=1, thread_count=threads,
        task_type="CPU", random_seed=seed, verbose=False, rsm=1.0,
        model_shrink_rate=0.0, langevin=False,
        metadata={"boosting_response_control": "unweighted-gradient-rmse-v1",
                  "control_training_x_sha256": _data_hash(train),
                  "control_native_targets_sha256": _data_hash(actual_y)},
    )
    model.fit(train, actual_y)
    if model.tree_count_ != iterations:
        raise ValueError("Incomplete control tree budget")
    validate_control_model(model)
    return model


def _leaf_matrix(value: Any, label: str, *, allow_empty: bool = False) -> np.ndarray:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or matrix.shape[1] < 1 or (not allow_empty and matrix.shape[0] < 1) or not np.issubdtype(matrix.dtype, np.integer) or (matrix < 0).any():
        raise ValueError(f"{label} must contain nonnegative integer leaf indices")
    return matrix.astype(np.int64, copy=False)


def operators_from_partitions(train_leaves: Any, evaluation_leaves: Any, *, eta: float,
                              l2: float, checkpoints: Iterable[int] | None = None,
                              leaf_counts: Iterable[int] | None = None) -> dict[int, dict[str, np.ndarray]]:
    """Pure exact fixed-partition recurrence; empty leaves predict zero updates."""
    train = _leaf_matrix(train_leaves, "train_leaves")
    evaluation = _leaf_matrix(evaluation_leaves, "evaluation_leaves", allow_empty=True)
    if train.shape[1] != evaluation.shape[1]:
        raise ValueError("Train/evaluation tree counts differ")
    if not math.isfinite(eta) or eta <= 0 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("Finite positive eta and nonnegative l2 required")
    n_train, n_trees = train.shape
    chosen = (n_trees,) if checkpoints is None else tuple(checkpoints)
    if not chosen or any(type(t) is not int or not 0 <= t <= n_trees for t in chosen) or len(chosen) != len(set(chosen)):
        raise ValueError("Checkpoints must be unique integers in 0..n_trees")
    chosen_set = set(chosen)
    if leaf_counts is None:
        counts = [max(int(train[:, tree].max()), int(evaluation[:, tree].max()) if len(evaluation) else 0) + 1 for tree in range(n_trees)]
    else:
        counts = list(leaf_counts)
        if len(counts) != n_trees or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("One positive structural leaf count per tree required")
    a_train = np.zeros((n_train, n_train), dtype=np.float64)
    a_evaluation = np.zeros((len(evaluation), n_train), dtype=np.float64)
    identity = np.eye(n_train)
    snapshots = {}
    if 0 in chosen_set:
        snapshots[0] = {"train": _read_only(a_train.copy()), "evaluation": _read_only(a_evaluation.copy())}
    for tree in range(n_trees):
        train_ids, eval_ids, count = train[:, tree], evaluation[:, tree], counts[tree]
        if int(train_ids.max()) >= count or (len(eval_ids) and int(eval_ids.max()) >= count):
            raise ValueError("Leaf index exceeds structural leaf count")
        # R is fixed before either training or evaluation update for this tree.
        residual_operator = identity - a_train
        summed = np.zeros((count, n_train), dtype=np.float64)
        np.add.at(summed, train_ids, residual_operator)
        occupancy = np.bincount(train_ids, minlength=count)
        denominator = occupancy + l2
        leaf_operator = np.divide(summed, denominator[:, None], out=np.zeros_like(summed), where=denominator[:, None] > 0)
        a_train += eta * leaf_operator[train_ids]
        a_evaluation += eta * leaf_operator[eval_ids]
        if not np.isfinite(a_train).all() or not np.isfinite(a_evaluation).all():
            raise ValueError("Nonfinite replay operator")
        at = tree + 1
        if at in chosen_set:
            snapshots[at] = {"train": _read_only(a_train.copy()), "evaluation": _read_only(a_evaluation.copy())}
    return {at: snapshots[at] for at in sorted(chosen)}


def frozen_operators(model: Any, x_train: Any, x_evaluation: Any,
                     checkpoints: Iterable[int] | None = None, *, targets: Any,
                     threads: int = 4, atol: float = 1e-10,
                     rtol: float = 1e-10) -> dict[str, Any]:
    """Build actual-partition operators and require baseline replay agreement.

    This gate compares A@y_actual with model.predict at every requested prefix.
    A mismatch is an implementation/solver-semantics failure, not a new effect.
    """
    _threads(threads)
    if not math.isfinite(atol) or atol < 0 or not math.isfinite(rtol) or rtol < 0:
        raise ValueError("Finite nonnegative replay tolerances required")
    train, evaluation = _binary_pair(x_train, x_evaluation)
    actual_y = _actual_targets(targets, len(train))
    control = validate_control_model(model)
    metadata = model.get_metadata()
    if metadata.get("boosting_response_control") != "unweighted-gradient-rmse-v1" or metadata.get("control_training_x_sha256") != _data_hash(train) or metadata.get("control_native_targets_sha256") != _data_hash(actual_y):
        raise ValueError("Model does not bind this unweighted control training matrix and native label vector")
    train_leaves = np.asarray(model.calc_leaf_indexes(train, thread_count=threads), dtype=np.int64)
    evaluation_leaves = np.asarray(model.calc_leaf_indexes(evaluation, thread_count=threads), dtype=np.int64) if len(evaluation) else np.empty((0, model.tree_count_), dtype=np.int64)
    leaf_counts = np.asarray(model.get_tree_leaf_counts(), dtype=np.int64)
    actual_weights = np.asarray(model.get_leaf_weights(), dtype=np.float64)
    expected_weights = np.concatenate([np.bincount(train_leaves[:, tree], minlength=int(count)) for tree, count in enumerate(leaf_counts)])
    if actual_weights.shape != expected_weights.shape or not np.array_equal(actual_weights, expected_weights):
        raise ValueError("Actual leaf weights differ from unweighted training occupancy")
    operators = operators_from_partitions(train_leaves, evaluation_leaves,
                                         eta=control["actual_learning_rate"], l2=control["actual_l2_leaf_reg"],
                                         checkpoints=checkpoints, leaf_counts=leaf_counts.astype(int).tolist())
    checks = []
    for at, operator in operators.items():
        for name, features in (("train", train), ("evaluation", evaluation)):
            expected = np.asarray(model.predict(features, prediction_type="RawFormulaVal", ntree_end=at, thread_count=threads), dtype=np.float64) if at and len(features) else np.zeros(len(features))
            replay = operator[name] @ actual_y
            error = float(np.max(np.abs(replay - expected))) if len(features) else 0.0
            if not np.allclose(replay, expected, atol=atol, rtol=rtol):
                raise ValueError(f"Known-control replay mismatch at tree {at}, split {name}, max error {error}; diagnose solver semantics before science")
            checks.append({"trees": at, "split": name, "n": len(features), "max_absolute_error": error,
                           "atol": atol, "rtol": rtol, "passed": True})
    return {**control, "operators": operators, "baseline_checks": checks,
            "train_leaf_indices": train_leaves, "evaluation_leaf_indices": evaluation_leaves,
            "leaf_counts": leaf_counts, "unit_weights_verified": True,
            "target_space": "actual native float32-representable labels promoted to float64"}


def decomposition(full_delta: Any, frozen_delta: Any, epsilon: float) -> dict[str, Any]:
    """Norm identity with a signed cross term; not an extra-variance estimate."""
    full = np.asarray(full_delta, dtype=np.float64)
    frozen = np.asarray(frozen_delta, dtype=np.float64)
    if full.ndim != 1 or full.shape != frozen.shape or not len(full) or not np.isfinite(full).all() or not np.isfinite(frozen).all() or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Two equal finite nonempty vectors and positive epsilon required")
    structural = full - frozen
    denominator = epsilon * epsilon * len(full)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Nonfinite or underflowed normalization")
    full_norm = float(np.dot(full, full) / denominator)
    frozen_norm = float(np.dot(frozen, frozen) / denominator)
    structural_norm = float(np.dot(structural, structural) / denominator)
    twice_cross = float(2 * np.dot(frozen, structural) / denominator)
    if not np.isfinite([full_norm, frozen_norm, structural_norm, twice_cross]).all():
        raise ValueError("Overflowed response norm decomposition")
    closure = full_norm - (frozen_norm + structural_norm + twice_cross)
    return {"normalized_full": full_norm, "normalized_frozen": frozen_norm,
            "normalized_structural_discrepancy": structural_norm,
            "normalized_twice_cross": twice_cross, "closure_error": closure,
            "relative_closure_error": closure / max(1.0, abs(full_norm)),
            "structural_delta": structural, "n_evaluation": len(full), "epsilon": epsilon,
            "interpretation": "Squared approximation discrepancy and signed cross term; cross may be negative, so this is not an additive nonnegative extra-variance decomposition."}


def representation_mse_floor(signatures: Any, targets: Any) -> dict[str, Any]:
    """Known empirical MSE floor for predictors constant on identical signatures."""
    signatures = _matrix(signatures, "signatures")
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1 or len(targets) != len(signatures) or not np.isfinite(targets).all():
        raise ValueError("One finite target per signature required")
    unique, inverse, counts = np.unique(signatures, axis=0, return_inverse=True, return_counts=True)
    means = np.bincount(inverse, weights=targets, minlength=len(unique)) / counts
    fitted = means[inverse]
    residual = targets - fitted
    sse = float(np.dot(residual, residual))
    return {"mse_floor": sse / len(targets), "sse_floor": sse,
            "n": len(targets), "n_groups": len(unique), "group_ids": inverse,
            "group_counts": counts, "group_means": means, "fitted": fitted,
            "residual": residual, "scope": "Empirical train representation floor only; zero is necessary but not sufficient for depth-limited finite-boosting interpolation.",
            "known_control": True}
