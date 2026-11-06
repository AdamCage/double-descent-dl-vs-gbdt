"""Preregistered curve rules for one frozen double-descent run.

The thresholds below were fixed before the full-data curves existed.
They are properties of this pack version, not quantities fit to a plot.
A synthetic fixture and the study bundle share the arithmetic. The fixture
is not evidence about the corpus.
"""

import math


PACK_ID = "double_descent_text_v1"
METRIC = "modelwise_peak_excess"
PEAK_EXCESS_MIN = 0.01
MODELWISE_TRAIN_ACC = 0.99
MODELWISE_TRAIN_LOSS = 0.05
EPOCHWISE_TRAIN_ACC = 0.99
EPOCHWISE_TRAIN_LOSS = 0.05
PLATEAU_MIN_ITERATION = 100
PLATEAU_WIDTH = 500
PLATEAU_RANGE = 0.005
DROP_MIN = 0.05
VAL_RISE_MAX = 0.01
ECE_BINS = 10
LOGLOSS_CLIP = 1e-15
LABEL_NOISE = 0.1
STUDY_SEED = 42
EXPECTED_TASK = "GPU"
EXPECTED_EMBEDDING = "embeddinggemma:latest"


def _num(value, name):
    if type(value) is bool or type(value) not in (int, float):
        raise ValueError(f"{name} is not a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} is not finite")
    return value


def _series(points, key):
    return [_num(row[key], key) for row in points]


def _argmin_threshold(points, acc_key, loss_key, acc_min, loss_max):
    for index, row in enumerate(points):
        if _num(row[acc_key], acc_key) >= acc_min or _num(row[loss_key], loss_key) <= loss_max:
            return index
    return None


def _peak(values, anchor):
    """Peak of a validation curve on the logged grid.

    The grid point nearest the interpolation index, among points that have
    a neighbour on both sides, is the candidate. The excess is that point
    minus the higher of the best point strictly before it and the best point
    strictly after it. Fewer than three points, or an anchor on the edge
    with no interior neighbour, cannot show a second descent.
    """
    count = len(values)
    if anchor is None or count < 3:
        return {"verdict": "inconclusive", "interpolation_index": anchor,
                "peak_index": None, "peak_excess": 0.0, "n_points": count}
    eligible = [index for index in range(1, count - 1) if abs(index - anchor) <= 1]
    if not eligible:
        return {"verdict": "inconclusive", "interpolation_index": anchor,
                "peak_index": None, "peak_excess": 0.0, "n_points": count}
    peak = max(eligible, key=lambda index: (values[index], -index))
    left = min(values[:peak])
    right = min(values[peak + 1:])
    excess = values[peak] - max(left, right)
    verdict = "supports" if excess >= PEAK_EXCESS_MIN else "refutes"
    return {"verdict": verdict, "interpolation_index": anchor, "peak_index": peak,
            "peak_excess": excess, "n_points": count}


def modelwise_pattern(points):
    ordered = sorted(points, key=lambda row: (_num(row["num_params"], "num_params"),
                                               _num(row["width"], "width")))
    anchor = _argmin_threshold(ordered, "train_acc", "train_loss",
                               MODELWISE_TRAIN_ACC, MODELWISE_TRAIN_LOSS)
    found = _peak(_series(ordered, "val_loss"), anchor)
    found["train_acc_threshold"] = MODELWISE_TRAIN_ACC
    found["train_loss_threshold"] = MODELWISE_TRAIN_LOSS
    found["points"] = [
        {"width": int(row["width"]), "num_params": int(row["num_params"]),
         "train_loss": _num(row["train_loss"], "train_loss"),
         "val_loss": _num(row["val_loss"], "val_loss"),
         "train_acc": _num(row["train_acc"], "train_acc"),
         "val_acc": _num(row["val_acc"], "val_acc")}
        for row in ordered]
    return found


def epochwise_pattern(points):
    ordered = sorted(points, key=lambda row: _num(row["epoch"], "epoch"))
    anchor = _argmin_threshold(ordered, "train_acc", "train_loss",
                               EPOCHWISE_TRAIN_ACC, EPOCHWISE_TRAIN_LOSS)
    found = _peak(_series(ordered, "val_loss"), anchor)
    found["train_acc_threshold"] = EPOCHWISE_TRAIN_ACC
    found["train_loss_threshold"] = EPOCHWISE_TRAIN_LOSS
    found["points"] = [
        {"epoch": int(row["epoch"]),
         "train_loss": _num(row["train_loss"], "train_loss"),
         "val_loss": _num(row["val_loss"], "val_loss"),
         "train_acc": _num(row["train_acc"], "train_acc"),
         "val_acc": _num(row["val_acc"], "val_acc")}
        for row in ordered]
    return found


def _range(values):
    return max(values) - min(values)


def boosting_pattern(train, val):
    if len(train) != len(val) or len(train) < 2:
        raise ValueError("boosting curves must be paired and nonempty")
    train = [_num(value, "train_logloss") for value in train]
    val = [_num(value, "val_logloss") for value in val]
    start = None
    for index in range(0, len(train) - PLATEAU_WIDTH + 1):
        if index + 1 < PLATEAU_MIN_ITERATION:
            continue
        train_window = train[index:index + PLATEAU_WIDTH]
        val_window = val[index:index + PLATEAU_WIDTH]
        if _range(train_window) <= PLATEAU_RANGE and _range(val_window) <= PLATEAU_RANGE:
            start = index
            break
    empty = {"plateau_start_iteration": None, "plateau_end_iteration": None,
             "drop_iteration": None, "train_at_plateau_end": None,
             "val_at_plateau_end": None, "train_at_drop": None, "val_at_drop": None}
    if start is None:
        return {"verdict": "refutes", **empty}
    end = start + PLATEAU_WIDTH - 1
    for index in range(end + 1, len(train)):
        if train[index] > train[end] - DROP_MIN or val[index] > val[end] - DROP_MIN:
            continue
        if any(val[cursor] > val[end] + VAL_RISE_MAX for cursor in range(end + 1, index + 1)):
            continue
        return {"verdict": "supports",
                "plateau_start_iteration": start + 1,
                "plateau_end_iteration": end + 1,
                "drop_iteration": index + 1,
                "train_at_plateau_end": train[end],
                "val_at_plateau_end": val[end],
                "train_at_drop": train[index],
                "val_at_drop": val[index]}
    return {"verdict": "refutes", "plateau_start_iteration": start + 1,
            "plateau_end_iteration": end + 1, "drop_iteration": None,
            "train_at_plateau_end": train[end], "val_at_plateau_end": val[end],
            "train_at_drop": None, "val_at_drop": None}


def _clip(probability):
    if probability <= 0.0:
        return LOGLOSS_CLIP
    if probability >= 1.0:
        return 1.0 - LOGLOSS_CLIP
    return probability


def _logloss(labels, probabilities):
    total = 0.0
    for label, probability in zip(labels, probabilities):
        probability = _clip(probability)
        total += -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
    return total / len(labels)


def _brier(labels, probabilities):
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def _accuracy(labels, probabilities):
    correct = sum((probability >= 0.5) == (label == 1.0) for label, probability in zip(labels, probabilities))
    return correct / len(labels)


def _ece(labels, probabilities):
    count = len(labels)
    total = 0.0
    for bin_index in range(ECE_BINS):
        lower = bin_index / ECE_BINS
        upper = (bin_index + 1) / ECE_BINS
        chosen = []
        for label, probability in zip(labels, probabilities):
            in_bin = probability >= lower and (probability < upper or (bin_index == ECE_BINS - 1 and probability <= upper))
            if in_bin:
                chosen.append((label, probability))
        if not chosen:
            continue
        confidence = sum(item[1] for item in chosen) / len(chosen)
        accuracy = sum(item[0] for item in chosen) / len(chosen)
        total += (len(chosen) / count) * abs(accuracy - confidence)
    return total


def _pearson(left, right):
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 0.0 or var_right <= 0.0:
        return None
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    return covariance / math.sqrt(var_left * var_right)


def _auc(labels, probabilities):
    order = sorted(range(len(labels)), key=lambda index: probabilities[index])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0.0 or negatives == 0.0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and probabilities[order[end + 1]] == probabilities[order[index]]:
            end += 1
        average_rank = (index + 1 + end + 1) / 2.0
        for cursor in range(index, end + 1):
            if labels[order[cursor]] == 1.0:
                rank_sum += average_rank
        index = end + 1
    return (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def _score(labels, probabilities, prefix):
    area = _auc(labels, probabilities)
    return {
        f"{prefix}_logloss": _logloss(labels, probabilities),
        f"{prefix}_brier": _brier(labels, probabilities),
        f"{prefix}_accuracy_0.5": _accuracy(labels, probabilities),
        f"{prefix}_ece_10": _ece(labels, probabilities),
        f"{prefix}_auc": 0.0 if area is None else area,
    }


def comparison(labels, dnn, catboost):
    if not labels or len(labels) != len(dnn) or len(labels) != len(catboost):
        raise ValueError("test labels and both probability vectors must share one length")
    labels = [_num(value, "label") for value in labels]
    dnn = [_num(value, "dnn_proba") for value in dnn]
    catboost = [_num(value, "cb_proba") for value in catboost]
    if any(label not in (0.0, 1.0) for label in labels):
        raise ValueError("test labels must be 0 or 1")
    if any(probability < 0.0 or probability > 1.0 for probability in dnn + catboost):
        raise ValueError("probabilities must lie in [0, 1]")
    scored = {"n_test": len(labels)}
    scored.update(_score(labels, dnn, "dnn"))
    scored.update(_score(labels, catboost, "cb"))
    pearson = _pearson(dnn, catboost)
    disagree = sum((left >= 0.5) != (right >= 0.5) for left, right in zip(dnn, catboost)) / len(labels)
    scored["pearson_dnn_cb"] = 0.0 if pearson is None else pearson
    scored["pearson_defined"] = pearson is not None
    scored["disagree_rate_0.5"] = disagree
    dnn_loss = scored["dnn_logloss"]
    cb_loss = scored["cb_logloss"]
    scored["logloss_difference_cb_minus_dnn"] = cb_loss - dnn_loss
    return scored


def _curve(block, name):
    train = block.get("train_logloss")
    val = block.get("val_logloss")
    if type(train) is not list or type(val) is not list:
        raise ValueError(f"{name} logloss curves are missing")
    return boosting_pattern(train, val)


def evaluate_bundle(bundle):
    """Return the raw record fields that do not depend on a roster unit."""
    if bundle.get("schema_version") != 1:
        raise ValueError("bundle schema_version must be 1")
    origin = bundle.get("origin")
    if origin not in ("synthetic_fixture", "full_pipeline"):
        raise ValueError("bundle origin is not recognised")
    if _num(bundle.get("label_noise"), "label_noise") != LABEL_NOISE:
        raise ValueError("bundle label noise differs from the preregistered 0.1")
    if int(bundle.get("seed")) != STUDY_SEED:
        raise ValueError("bundle seed differs from the preregistered seed 42")
    meta = bundle.get("meta")
    if type(meta) is not dict:
        raise ValueError("bundle meta is missing")
    if origin == "full_pipeline":
        if bundle.get("mode") != "full":
            raise ValueError("full pipeline bundle must declare mode full")
        if meta.get("catboost_task_type") != EXPECTED_TASK:
            raise ValueError("full pipeline bundle must declare CatBoost task_type GPU")
        if not str(meta.get("embedding_model", "")).startswith("embeddinggemma"):
            raise ValueError("full pipeline bundle must declare embeddinggemma")
    modelwise = modelwise_pattern(bundle["modelwise"])
    epochwise = epochwise_pattern(bundle["epochwise"])
    catboost = _curve(bundle["catboost_main"], "catboost_main")
    sweeps = bundle.get("sweeps")
    if set(sweeps) != {"random_strength", "l2_leaf_reg", "learning_rate"}:
        raise ValueError("sweeps must be random_strength, l2_leaf_reg and learning_rate")
    sweep_verdicts = {}
    for name, runs in sweeps.items():
        if type(runs) is not list or not runs:
            raise ValueError(f"sweep {name} is empty")
        sweep_verdicts[name] = []
        for run in runs:
            found = _curve(run, name)
            sweep_verdicts[name].append({"label": str(run.get("label", "")), "verdict": found["verdict"]})
    compared = comparison(bundle["y_test"], bundle["dnn_proba"], bundle["cb_proba"])
    return {"origin": origin, "meta": meta, "modelwise": modelwise, "epochwise": epochwise,
            "catboost": catboost, "sweep_verdicts": sweep_verdicts, "comparison": compared}
