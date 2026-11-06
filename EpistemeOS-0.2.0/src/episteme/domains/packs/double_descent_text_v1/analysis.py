"""Recompute the preregistered curve rules from raw records."""

import hashlib
import math

from episteme.domains import api

from .model import METRIC, PACK_ID, evaluate_bundle, modelwise_pattern


PACK_VERSION = "1"
TOLERANCE = 1e-8

LIMITATIONS = [
    "The claim describes one frozen run. It is not a population estimate and not a reviewer approval.",
    "Model-wise and epoch-wise rules use the validation curve on the logged grid. The test split is used only for the preregistered DNN versus CatBoost comparison.",
    "Model-wise training accuracy is computed on the labels the pipeline trained on, including the preregistered 10 percent flips. Epoch-wise training uses clean labels.",
    "The interpolation threshold is the first logged point with train accuracy at least 0.99 or train loss at most 0.05. If that point is missing or has no interior neighbour, the model-wise outcome is inconclusive.",
    "A peak excess of at least 0.01 at that neighbourhood is the preregistered support rule. A smaller excess, once the threshold is identified with both sides present, refutes the pattern on this run.",
    "The CatBoost rule looks for a 500-iteration plateau, range at most 0.005 on train and validation, starting at iteration 100 or later, then a joint drop of at least 0.05 without a validation rise above 0.01. Ablation sweeps use the same rule and do not multiply the primary claim.",
    "The kernel calls the second program independent_reanalysis because its bytes differ. The same pack, OS user and curves produced it. That is not an independent reanalysis and not a replication.",
    "scientific_validity stays not_assessed. A descriptive label is not human scientific review.",
]


def _raw(cas, key, seed):
    document = cas.json(key, "raw data")
    if (type(document) is not dict or document.get("schema_version") != 1
            or document.get("domain") != PACK_ID or document.get("roster_unit") != seed):
        raise ValueError("raw data does not match the double-descent schema")
    return document


def _metric(document):
    value = document.get(METRIC)
    if type(value) is bool or type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError("metrics file lacks a finite modelwise peak excess")
    return float(value)


def _close(left, right):
    if left is None or right is None:
        return left is right
    if type(left) is str or type(right) is str:
        return left == right
    return abs(float(left) - float(right)) <= TOLERANCE


def validate_outputs(context, cas):
    checks = []
    primary_raw = {}
    for row in context.slots:
        findings = []
        try:
            document = _raw(cas, row["outputs"]["raw_data"], row["roster_unit"])
            _metric(cas.json(row["outputs"]["metrics"], "metrics"))
            pattern = modelwise_pattern(document["modelwise_points"])
            if not _close(pattern["peak_excess"], document["modelwise"]["peak_excess"]):
                raise ValueError("stored model-wise excess differs from the recomputed excess")
            if row["mode"] == "primary":
                primary_raw[row["roster_unit"]] = row["outputs"]["raw_data"]
            elif primary_raw.get(row["roster_unit"]) != row["outputs"]["raw_data"]:
                raise ValueError("reanalysis did not keep the primary raw data")
        except (api.EnvelopeError, ValueError, TypeError, KeyError) as exc:
            findings.append(str(exc))
        checks.append(api.OutputCheck(
            schema_version=1, slot=row["slot"], roster_unit=row["roster_unit"], mode=row["mode"],
            result=row["result"], raw_data=row["outputs"]["raw_data"],
            metrics=row["outputs"]["metrics"], status="failed" if findings else "passed",
            findings=findings))
    return checks


def recompute_metrics(context, cas):
    result = []
    for unit in context.draft.roster:
        primary = context.slot(f"primary:{unit}")
        reanalysis = context.slot(f"reanalysis:{unit}")
        document = _raw(cas, primary["outputs"]["raw_data"], unit)
        if reanalysis["outputs"]["raw_data"] != primary["outputs"]["raw_data"]:
            raise ValueError("reanalysis did not keep the primary raw data")
        value = modelwise_pattern(document["modelwise_points"])["peak_excess"]
        recorded = _metric(cas.json(primary["outputs"]["metrics"], "metrics"))
        repeated = _metric(cas.json(reanalysis["outputs"]["metrics"], "metrics"))
        result.append(api.Recomputation(
            schema_version=1, roster_unit=unit, metric=METRIC, recomputed=value,
            primary_recorded=recorded, reanalysis_recorded=repeated,
            primary_absolute_difference=abs(value - recorded),
            reanalysis_absolute_difference=abs(value - repeated),
            tolerance=context.numeric_tolerance))
    return result


def _summary(pattern):
    return {key: pattern[key] for key in ("verdict", "interpolation_index", "peak_index", "peak_excess", "n_points")}


def analyse(context, cas, checks, recomputations):
    failed = [check for check in checks if check.status != "passed"]
    if failed:
        raise ValueError("double-descent outputs failed pack checks")
    design = context.draft.typed_design()
    unit = context.draft.roster[0]
    document = _raw(cas, context.slot(f"primary:{unit}")["outputs"]["raw_data"], unit)
    modelwise = modelwise_pattern(document["modelwise_points"])
    bundle = {
        "schema_version": 1, "origin": document["origin"], "label_noise": 0.1, "seed": 42,
        "mode": "full" if document["origin"] == "full_pipeline" else "fixture",
        "meta": document["meta"], "modelwise": document["modelwise_points"],
        "epochwise": document["epochwise_points"],
        "catboost_main": {"train_logloss": document["catboost_train"], "val_logloss": document["catboost_val"]},
        "sweeps": document["sweeps"], "y_test": document["y_test"],
        "dnn_proba": document["dnn_proba"], "cb_proba": document["cb_proba"],
    }
    evaluated = evaluate_bundle(bundle)
    if evaluated["modelwise"]["verdict"] != modelwise["verdict"]:
        raise ValueError("model-wise verdicts diverged")
    outcome = modelwise["verdict"]
    origin = document["origin"]
    excess = recomputations[0].recomputed
    if abs(excess - modelwise["peak_excess"]) > 1e-9:
        raise ValueError("point estimate differs from the model-wise rule")
    epoch = _summary(evaluated["epochwise"])
    boost = evaluated["catboost"]["verdict"]
    compared = evaluated["comparison"]
    fixture = origin != "full_pipeline"
    statement = (
        f"На этом замороженном прогоне model-wise peak excess равен {excess:.6g}; "
        f"вердикт правила — {outcome}. Epoch-wise вердикт — {epoch['verdict']} "
        f"(excess {epoch['peak_excess']:.6g}). CatBoost plateau-then-drop — {boost}. "
        f"Разница test logloss CatBoost минус DNN равна {compared['logloss_difference_cb_minus_dnn']:.6g} "
        f"на {compared['n_test']} строках. Это описательный итог одного прогона, не оценка популяции"
        + (" и не результат по корпусу: источник — синтетическая фикстура." if fixture
           else ". Научный review человеком не вынесен.")
    )
    if len(statement) > 4000:
        raise ValueError("statement exceeds the report limit")
    meta = document["meta"]
    sensitivities = [
        {"name": "epochwise_pattern", "timing": "preregistered", "result": epoch["verdict"]},
        {"name": "catboost_plateau_then_drop", "timing": "preregistered", "result": boost},
        {"name": "test_logloss_cb_minus_dnn", "timing": "preregistered",
         "result": f"{compared['logloss_difference_cb_minus_dnn']:.6g}"},
    ]
    for name, runs in evaluated["sweep_verdicts"].items():
        counts = {}
        for run in runs:
            counts[run["verdict"]] = counts.get(run["verdict"], 0) + 1
        sensitivities.append({"name": f"sweep_{name}", "timing": "preregistered",
                              "result": ", ".join(f"{key}={counts[key]}" for key in sorted(counts))})
    report = api.StatisticalReport.from_dict({
        "schema_version": 1, "protocol_hash": context.protocol_hash,
        "estimand": api.supplied({"protocol_hash": context.protocol_hash, "text": design.estimand}),
        "estimator": api.supplied({
            "name": "logged_grid_peak_excess", "hook": "recompute_metrics",
            "description": "Validation-loss peak excess at the preregistered interpolation neighbourhood."}),
        "point_estimate": api.supplied({
            "metric": METRIC, "unit": "validation logloss", "value": excess,
            "by_roster_unit": [{"roster_unit": unit, "value": recomputations[0].recomputed}]}),
        "uncertainty": api.not_applicable("Preregistered uncertainty.method is not_applicable."),
        "confidence_interval": api.not_applicable(
            "One frozen run is not a sample, so no interval is declared."),
        "effect_size": api.not_applicable(
            "Descriptive design: peak excess is a property of this run, not a population effect size."),
        "assumptions": api.supplied([
            {"assumption": "The pack scores captured curves and does not refit the networks or the trees.",
             "status": "unchecked", "reference": None},
            {"assumption": "Model-wise train metrics include the preregistered 10 percent label flips; validation labels are clean.",
             "status": "unchecked", "reference": None},
            {"assumption": "Embeddinggemma context is 2048 tokens; longer essays are truncated by the embedding server.",
             "status": "unchecked", "reference": None},
        ]),
        "sample_size": api.supplied({
            "experimental_unit": design.experimental_unit, "unit_scope": "total",
            "planned": design.sample_size, "planned_total": design.sample_size,
            "analysed": 1, "exclusions": [], "missing_slots": []}),
        "multiple_testing": api.supplied({
            "family": [METRIC], "claim_position": "1", "correction": "none", "adjusted_result": None}),
        "stopping_rule": api.supplied({
            "rule_sha256": hashlib.sha256(design.stopping_rule.rule.encode("utf-8")).hexdigest(),
            "roster_complete": True, "interim_looks": 0}),
        "sensitivity_analysis": api.supplied(sensitivities),
        "deviations": api.supplied([]),
    })
    details = {
        "origin": origin,
        "modelwise": _summary(modelwise),
        "epochwise": epoch,
        "catboost_verdict": boost,
        "catboost_plateau_start_iteration": evaluated["catboost"]["plateau_start_iteration"],
        "catboost_drop_iteration": evaluated["catboost"]["drop_iteration"],
        "comparison": {key: compared[key] for key in (
            "n_test", "dnn_logloss", "cb_logloss", "logloss_difference_cb_minus_dnn",
            "dnn_brier", "cb_brier", "dnn_ece_10", "cb_ece_10", "dnn_accuracy_0.5",
            "cb_accuracy_0.5", "dnn_auc", "cb_auc", "pearson_dnn_cb", "disagree_rate_0.5")},
        "n_train": meta.get("n_train"), "n_val": meta.get("n_val"), "feature_dim": meta.get("feature_dim"),
        "pca_explained_variance_at_256": meta.get("pca_explained_variance_at_256"),
        "umap_plot_rows": meta.get("umap_plot_rows"),
        "catboost_task_type": meta.get("catboost_task_type"),
        "embedding_model": meta.get("embedding_model"),
    }
    return api.AnalysisReport(
        pack_id=PACK_ID, pack_version=PACK_VERSION, protocol_hash=context.protocol_hash,
        statement=statement, limitations=list(LIMITATIONS), outcome=outcome,
        inference_mode="descriptive", details=details, statistical_report=report)
