"""Stdlib runner sources. Profile v1 cannot import this package.

The arithmetic is duplicated from ``model.py``. The pack test runs both and
rejects a divergence. Reanalysis reads the primary raw record and writes
those same bytes back; it is a second implementation of the metric, not a
new experiment and not an independent laboratory.
"""

PRIMARY_SOURCE = r'''"""Score one captured curve bundle. The roster seed is not the study seed."""
import json
import math
import pathlib
import sys


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


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def num(value, name):
    if type(value) is bool or type(value) not in (int, float):
        raise ValueError(name + " is not a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name + " is not finite")
    return value


def series(points, key):
    return [num(row[key], key) for row in points]


def argmin_threshold(points, acc_key, loss_key, acc_min, loss_max):
    for index, row in enumerate(points):
        if num(row[acc_key], acc_key) >= acc_min or num(row[loss_key], loss_key) <= loss_max:
            return index
    return None


def peak(values, anchor):
    count = len(values)
    if anchor is None or count < 3:
        return {"verdict": "inconclusive", "interpolation_index": anchor,
                "peak_index": None, "peak_excess": 0.0, "n_points": count}
    eligible = [index for index in range(1, count - 1) if abs(index - anchor) <= 1]
    if not eligible:
        return {"verdict": "inconclusive", "interpolation_index": anchor,
                "peak_index": None, "peak_excess": 0.0, "n_points": count}
    chosen = max(eligible, key=lambda index: (values[index], -index))
    left = min(values[:chosen])
    right = min(values[chosen + 1:])
    excess = values[chosen] - max(left, right)
    verdict = "supports" if excess >= PEAK_EXCESS_MIN else "refutes"
    return {"verdict": verdict, "interpolation_index": anchor, "peak_index": chosen,
            "peak_excess": excess, "n_points": count}


def modelwise_pattern(points):
    ordered = sorted(points, key=lambda row: (num(row["num_params"], "num_params"), num(row["width"], "width")))
    anchor = argmin_threshold(ordered, "train_acc", "train_loss", MODELWISE_TRAIN_ACC, MODELWISE_TRAIN_LOSS)
    found = peak(series(ordered, "val_loss"), anchor)
    found["points"] = [{
        "width": int(row["width"]), "num_params": int(row["num_params"]),
        "train_loss": num(row["train_loss"], "train_loss"), "val_loss": num(row["val_loss"], "val_loss"),
        "train_acc": num(row["train_acc"], "train_acc"), "val_acc": num(row["val_acc"], "val_acc"),
    } for row in ordered]
    return found


def epochwise_pattern(points):
    ordered = sorted(points, key=lambda row: num(row["epoch"], "epoch"))
    anchor = argmin_threshold(ordered, "train_acc", "train_loss", EPOCHWISE_TRAIN_ACC, EPOCHWISE_TRAIN_LOSS)
    found = peak(series(ordered, "val_loss"), anchor)
    found["points"] = [{
        "epoch": int(row["epoch"]),
        "train_loss": num(row["train_loss"], "train_loss"), "val_loss": num(row["val_loss"], "val_loss"),
        "train_acc": num(row["train_acc"], "train_acc"), "val_acc": num(row["val_acc"], "val_acc"),
    } for row in ordered]
    return found


def curve_range(values):
    return max(values) - min(values)


def boosting_pattern(train, val):
    if len(train) != len(val) or len(train) < 2:
        raise ValueError("boosting curves must be paired and nonempty")
    train = [num(value, "train_logloss") for value in train]
    val = [num(value, "val_logloss") for value in val]
    start = None
    for index in range(0, len(train) - PLATEAU_WIDTH + 1):
        if index + 1 < PLATEAU_MIN_ITERATION:
            continue
        if (curve_range(train[index:index + PLATEAU_WIDTH]) <= PLATEAU_RANGE
                and curve_range(val[index:index + PLATEAU_WIDTH]) <= PLATEAU_RANGE):
            start = index
            break
    if start is None:
        return {"verdict": "refutes", "plateau_start_iteration": None, "plateau_end_iteration": None,
                "drop_iteration": None}
    end = start + PLATEAU_WIDTH - 1
    for index in range(end + 1, len(train)):
        if train[index] > train[end] - DROP_MIN or val[index] > val[end] - DROP_MIN:
            continue
        if any(val[cursor] > val[end] + VAL_RISE_MAX for cursor in range(end + 1, index + 1)):
            continue
        return {"verdict": "supports", "plateau_start_iteration": start + 1,
                "plateau_end_iteration": end + 1, "drop_iteration": index + 1}
    return {"verdict": "refutes", "plateau_start_iteration": start + 1,
            "plateau_end_iteration": end + 1, "drop_iteration": None}


def clip(probability):
    if probability <= 0.0:
        return LOGLOSS_CLIP
    if probability >= 1.0:
        return 1.0 - LOGLOSS_CLIP
    return probability


def logloss(labels, probabilities):
    total = 0.0
    for label, probability in zip(labels, probabilities):
        probability = clip(probability)
        total += -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
    return total / len(labels)


def brier(labels, probabilities):
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def accuracy(labels, probabilities):
    correct = sum((probability >= 0.5) == (label == 1.0) for label, probability in zip(labels, probabilities))
    return correct / len(labels)


def ece(labels, probabilities):
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
        hit = sum(item[0] for item in chosen) / len(chosen)
        total += (len(chosen) / count) * abs(hit - confidence)
    return total


def pearson(left, right):
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 0.0 or var_right <= 0.0:
        return None
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    return covariance / math.sqrt(var_left * var_right)


def auc(labels, probabilities):
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


def score(labels, probabilities, prefix):
    area = auc(labels, probabilities)
    return {
        prefix + "_logloss": logloss(labels, probabilities),
        prefix + "_brier": brier(labels, probabilities),
        prefix + "_accuracy_0.5": accuracy(labels, probabilities),
        prefix + "_ece_10": ece(labels, probabilities),
        prefix + "_auc": 0.0 if area is None else area,
    }


def comparison(labels, dnn, catboost):
    if not labels or len(labels) != len(dnn) or len(labels) != len(catboost):
        raise ValueError("test vectors differ in length")
    labels = [num(value, "label") for value in labels]
    dnn = [num(value, "dnn_proba") for value in dnn]
    catboost = [num(value, "cb_proba") for value in catboost]
    if any(label not in (0.0, 1.0) for label in labels):
        raise ValueError("labels must be binary")
    if any(probability < 0.0 or probability > 1.0 for probability in dnn + catboost):
        raise ValueError("probabilities must lie in [0, 1]")
    scored = {"n_test": len(labels)}
    scored.update(score(labels, dnn, "dnn"))
    scored.update(score(labels, catboost, "cb"))
    correlation = pearson(dnn, catboost)
    scored["pearson_dnn_cb"] = 0.0 if correlation is None else correlation
    scored["pearson_defined"] = correlation is not None
    scored["disagree_rate_0.5"] = sum((left >= 0.5) != (right >= 0.5) for left, right in zip(dnn, catboost)) / len(labels)
    scored["logloss_difference_cb_minus_dnn"] = scored["cb_logloss"] - scored["dnn_logloss"]
    return scored


def one_curve(block):
    return boosting_pattern(block["train_logloss"], block["val_logloss"])


def evaluate(bundle):
    if bundle.get("schema_version") != 1:
        raise ValueError("schema")
    origin = bundle.get("origin")
    if origin not in ("synthetic_fixture", "full_pipeline"):
        raise ValueError("origin")
    if num(bundle.get("label_noise"), "label_noise") != LABEL_NOISE or int(bundle.get("seed")) != STUDY_SEED:
        raise ValueError("preregistered noise or seed differs")
    meta = bundle["meta"]
    if origin == "full_pipeline":
        if bundle.get("mode") != "full" or meta.get("catboost_task_type") != EXPECTED_TASK:
            raise ValueError("full bundle config differs")
        if not str(meta.get("embedding_model", "")).startswith("embeddinggemma"):
            raise ValueError("embedding model differs")
    modelwise = modelwise_pattern(bundle["modelwise"])
    epochwise = epochwise_pattern(bundle["epochwise"])
    catboost = one_curve(bundle["catboost_main"])
    sweeps = bundle["sweeps"]
    if set(sweeps) != {"random_strength", "l2_leaf_reg", "learning_rate"}:
        raise ValueError("sweeps")
    sweep_verdicts = {}
    for name, runs in sweeps.items():
        sweep_verdicts[name] = [{"label": str(run.get("label", "")), "verdict": one_curve(run)["verdict"]} for run in runs]
    compared = comparison(bundle["y_test"], bundle["dnn_proba"], bundle["cb_proba"])
    return modelwise, epochwise, catboost, sweep_verdicts, compared


def main():
    if len(sys.argv) != 4 or sys.argv[1:3] != ["input.dat", "--seed"]:
        raise ValueError("expected input.dat --seed N")
    seed = int(sys.argv[3])
    bundle = json.loads(pathlib.Path("input.dat").read_text(encoding="utf-8"))
    modelwise, epochwise, catboost, sweep_verdicts, compared = evaluate(bundle)
    raw = {
        "schema_version": 1, "domain": "double_descent_text_v1", "roster_unit": seed,
        "origin": bundle["origin"], "meta": bundle["meta"],
        "modelwise_points": modelwise["points"], "epochwise_points": epochwise["points"],
        "catboost_train": bundle["catboost_main"]["train_logloss"],
        "catboost_val": bundle["catboost_main"]["val_logloss"],
        "sweeps": bundle["sweeps"], "y_test": bundle["y_test"],
        "dnn_proba": bundle["dnn_proba"], "cb_proba": bundle["cb_proba"],
        "modelwise": {key: modelwise[key] for key in ("verdict", "interpolation_index", "peak_index", "peak_excess", "n_points")},
        "epochwise": {key: epochwise[key] for key in ("verdict", "interpolation_index", "peak_index", "peak_excess", "n_points")},
        "catboost": catboost, "sweep_verdicts": sweep_verdicts, "comparison": compared,
    }
    pathlib.Path("raw.json").write_bytes(encoded(raw))
    pathlib.Path("metrics.json").write_bytes(encoded({"modelwise_peak_excess": modelwise["peak_excess"]}))


if __name__ == "__main__":
    main()
'''.encode("utf-8")

REANALYSIS_SOURCE = r'''"""Recompute the primary metric from the primary raw record and keep those bytes."""
import json
import pathlib
import sys


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def peak_excess(points):
    if len(points) < 3:
        return 0.0
    anchor = None
    for index, row in enumerate(points):
        if float(row["train_acc"]) >= 0.99 or float(row["train_loss"]) <= 0.05:
            anchor = index
            break
    if anchor is None:
        return 0.0
    eligible = [index for index in range(1, len(points) - 1) if abs(index - anchor) <= 1]
    if not eligible:
        return 0.0
    values = [float(row["val_loss"]) for row in points]
    chosen = max(eligible, key=lambda index: (values[index], -index))
    left = min(values[:chosen])
    right = min(values[chosen + 1:])
    return values[chosen] - max(left, right)


def main():
    if len(sys.argv) != 4 or sys.argv[1:3] != ["input.dat", "--seed"]:
        raise ValueError("expected input.dat --seed N")
    seed = int(sys.argv[3])
    source = pathlib.Path("input.dat").read_bytes()
    document = json.loads(source)
    if document.get("domain") != "double_descent_text_v1" or document.get("roster_unit") != seed:
        raise ValueError("reanalysis input is not the primary raw record")
    if type(document.get("modelwise_points")) is not list:
        raise ValueError("reanalysis points are missing")
    pathlib.Path("raw.json").write_bytes(source)
    pathlib.Path("metrics.json").write_bytes(encoded({"modelwise_peak_excess": peak_excess(document["modelwise_points"])}))


if __name__ == "__main__":
    main()
'''.encode("utf-8")
