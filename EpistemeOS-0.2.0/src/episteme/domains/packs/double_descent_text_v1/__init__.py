"""Descriptive scoring of one frozen double-descent run.

The heavy training stays outside this runner. The runner only applies the
rules in ``model.py`` to a captured ``bundle.json``. A confirmatory population
claim is not available: the design mode is descriptive and uncertainty is
not applicable. ``scientific_validity`` stays not assessed.
"""

import hashlib
import json

from episteme.domains import api

from . import analysis, programs
from . import capture as reader
from .model import LABEL_NOISE, METRIC, PACK_ID, STUDY_SEED


PACK_VERSION = reader.PACK_VERSION
NUMERIC_TOLERANCE = 1e-6
WALL_SECONDS = 300
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
OUTPUTS = {"raw_data": "raw.json", "metrics": "metrics.json"}
ESTIMAND = (
    "On one frozen full-data run, the validation-logloss peak excess of the "
    "DNN model-wise curve at the preregistered interpolation neighbourhood."
)
STOPPING = (
    "Score the single captured bundle once, in one primary program and one "
    "same-data reanalysis. Do not add widths, epochs, trees, or rows."
)

CAUTIONS = [
    "Training of the networks and trees is outside the runner. This pack scores captured curves.",
    "Descriptive support or refutation applies only to the captured run.",
    "The second program is same-pack reanalysis, not an independent reimplementation and not replication.",
    "A synthetic fixture bundle is a fixture. It is not the corpus result.",
    "No human scientific review is created by this pack.",
]

MANIFEST = {
    "contract_version": 1, "pack_id": PACK_ID, "pack_version": PACK_VERSION,
    "description": "Descriptive double-descent and boosting-curve rules on one captured run.",
    "roster_semantics": ["deterministic_single"],
    "metrics": [{"name": METRIC, "unit": "validation logloss",
                 "description": "Validation logloss at the interpolation neighbourhood minus the higher shoulder."}],
    "outputs": {"raw_data": {"path": "raw.json", "schema_id": "double_descent_text_v1/raw-v1"},
                "metrics": {"path": "metrics.json", "schema_id": "double_descent_text_v1/metrics-v1"}},
    "numeric_tolerance": NUMERIC_TOLERANCE, "hidden_inputs": [],
    "execution_profiles": ["trusted_local_python_v1"],
    "statistical_capabilities": ["estimand", "estimator", "point_estimate", "assumptions",
                                 "sample_size", "multiple_testing", "stopping_rule",
                                 "sensitivity_analysis", "deviations"],
    "interpretation_cautions": CAUTIONS,
    "capture": True,
}

PARAMETERS_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {}}


def describe():
    return api.ParameterCatalog(
        schema_version=1, pack_id=PACK_ID, pack_version=PACK_VERSION,
        description=MANIFEST["description"], parameters_schema=PARAMETERS_SCHEMA, metric=METRIC,
        outputs=dict(OUTPUTS), limitations=list(analysis.LIMITATIONS))


def validate_parameters(parameters):
    if api.thaw(parameters) != {}:
        raise ValueError("double-descent pack takes no parameters")


def _bundle(request):
    if api.thaw(request.host_inputs) != {}:
        raise ValueError("double-descent pack takes no host inputs")
    if request.capture is None or set(request.capture.files) != {"bundle.json"}:
        raise ValueError("double-descent pack requires a captured bundle.json")
    return request.capture.files["bundle.json"]


def compile_protocol(request):
    payload = _bundle(request)
    document = json.loads(payload.decode("utf-8"))
    if document.get("origin") not in ("synthetic_fixture", "full_pipeline"):
        raise ValueError("bundle origin is not recognised")
    digest = hashlib.sha256(payload).hexdigest()
    design = {
        "schema_version": 1, "mode": "descriptive", "experimental_unit": "frozen full-data run",
        "estimand": ESTIMAND,
        "primary_metric": {"name": METRIC, "unit": "validation logloss"},
        "secondary_metrics": [],
        "sample_size": 1,
        "sample_size_rationale": (
            "The experimental unit is the single captured run, seed "
            f"{STUDY_SEED}, label noise {LABEL_NOISE}. It is not a sample of runs."
        ),
        "uncertainty": {
            "method": "not_applicable", "resampling_unit": None,
            "rationale": "One frozen run is the whole analysis. No resampling interval is declared.",
        },
        "exclusions": [],
        "stopping_rule": {"kind": "fixed_sample", "rule": STOPPING},
        "multiple_testing": {
            "family": [METRIC], "correction": "none",
            "rationale": "Epoch-wise, CatBoost and the comparison are preregistered sensitivities, not extra primary tests.",
        },
        "data_splits": [
            {"id": "observed_run", "digest": digest, "role": "discovery", "exposure_policy": "open"},
        ],
    }
    plan = (
        "Apply the frozen interpolation rule: the first logged point with train accuracy "
        "at least 0.99 or train loss at most 0.05. The peak is the maximum validation loss "
        "among interior logged points within one step of that index. The excess is that "
        "value minus the higher of the best point before it and the best point after it. "
        "Support requires excess at least 0.01. The CatBoost rule and the test-set comparison "
        "are reported beside that primary number and do not change it."
    )
    return api.ProtocolDraft(
        schema_version=1,
        design=("Descriptive scoring of one captured curve bundle. The bundle is declared seen. " + plan),
        analysis_plan=plan, metric=METRIC, stopping_rule=STOPPING, statistical_design=design,
        roster=[0], roster_semantics="deterministic_single", sample_size_scope="total",
        run_limit=2, replication_tolerance=NUMERIC_TOLERANCE, seen_data=[digest])


def compile_execution(request):
    payload = _bundle(request)
    return api.ExecutionPlan(
        primary_program=programs.PRIMARY_SOURCE, reanalysis_program=programs.REANALYSIS_SOURCE,
        input=payload, outputs=dict(OUTPUTS), wall_seconds=WALL_SECONDS,
        max_output_bytes=MAX_OUTPUT_BYTES, required_capabilities=[],
        execution_profile="trusted_local_python_v1",
        environment_requirements=api.environment_requirements())


def capture(source):
    return reader.read_directory(source)


def _fail(condition, message):
    if condition:
        raise ValueError(message)


def validate_protocol(context):
    protocol = context.protocol
    design = protocol["statistical_design"]
    inventory = {row["path"]: row["sha256"] for row in context.capture["inventory"]}
    bundle = inventory["bundle.json"]
    _fail(api.thaw(context.parameters) != {}, "double-descent protocol has parameters")
    _fail(protocol.get("protocol_mode") != "descriptive", "double-descent protocol is not descriptive")
    _fail(protocol["metric"] != METRIC or list(protocol["seeds"]) != [0],
          "double-descent protocol metric or roster differs")
    _fail(protocol["data"] != bundle, "program input is not the captured bundle")
    _fail(bundle not in protocol["seen_data"], "captured bundle is not declared seen")
    _fail(design["uncertainty"]["method"] != "not_applicable", "uncertainty method differs")
    _fail(design["sample_size"] != 1, "sample size is not the single run")
    _fail(design["mode"] != "descriptive", "design mode differs")
    _fail(list(design["multiple_testing"]["family"]) != [METRIC], "multiple-testing family differs")


def validate_outputs(context, cas):
    return analysis.validate_outputs(context, cas)


def recompute_metrics(context, cas):
    return analysis.recompute_metrics(context, cas)


def analyse(context, cas, checks, recomputations):
    return analysis.analyse(context, cas, checks, recomputations)
