"""Bind the double-descent study inside EpistemeOS.

Запись в `.research/double-descent` — исходная пререгистрация до ревью
2026-11-04. Её нельзя переписывать. Новый вопрос и протокол лежат в
`docs/protocol_revision_2026-11-04.md`. Этот скрипт не выносит scientific review
и не превращает отсутствие измерений в успех.

`question` freezes the rules before any full-data curve is treated as confirmatory.
`evaluate` captures bundle.json and runs one descriptive stage. Actor IDs are
labels of one OS user; they are not independent reviewers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from episteme.commands import CommandService
from episteme.domains import registry
from episteme.execution import freeze_environment
from episteme.stages import stage_open, stage_run
from episteme.store import Store
from episteme import domain_packs


ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = ROOT / ".research" / "double-descent"
CAPTURE = ROOT / "episteme_capture"
STUDY = "double-descent-ai-human"
PLANNER = "dd-planner"
ANALYST = "dd-analyst"
EXECUTOR = "dd-executor"
REPLICATOR = "dd-reanalyst"


def _execute(store: Store, action: str, payload: dict, *, actor: str, role: str) -> dict:
    receipt = CommandService(store).execute({
        "context": {
            "command_id": f"{action}-{len(store.events())}-{actor}",
            "expected_revision": len(store.events()),
            "actor": actor,
            "role": role,
            "study_id": STUDY,
            "correlation_id": "double-descent-study",
            "causation_id": None,
        },
        "request": {"version": 1, "action": action, "payload": payload},
    })
    return receipt if isinstance(receipt, dict) else {"id": receipt}


def question() -> None:
    raise SystemExit(
        "Historical registration is frozen. Its embeddinggemma/GPU/10%-noise "
        "scope does not describe the clean Qwen/CPU run. Use "
        "scripts/episteme_bridge.py status|bootstrap for the new question and "
        "docs/automation.md for stage execution. No historical store was opened."
    )
    # Historical implementation retained below for provenance; never execute.
    loaded = registry.load_pack("double_descent_text_v1")
    scope = {
        "corpus": "andythetechnerd03/AI-human-text mirror of shanegerami/ai-vs-human-text",
        "split": "stratified 70/15/15 seed 42 after exact-text dedup",
        "primary_metric": "modelwise_peak_excess",
        "pack_id": loaded.pack_id,
        "pack_version": loaded.pack_version,
        "pack_code_digest": loaded.pack_code_digest,
    }
    constraints = [
        "Rules below are fixed before the full-data curves are read as evidence.",
        "Model-wise train metrics use 10 percent label flips; validation labels stay clean. Epoch-wise training uses clean labels.",
        "Interpolation is the first logged point with train accuracy >= 0.99 or train loss <= 0.05.",
        "Peak excess is validation loss at an interior point within one logged step of interpolation, minus the higher shoulder. Support needs excess >= 0.01.",
        "If interpolation is absent or has no interior neighbour, the model-wise outcome is inconclusive.",
        "CatBoost rule: earliest 500-iteration window at or after iteration 100 with train and validation range <= 0.005, then a joint drop >= 0.05 without a validation rise above 0.01.",
        "Ablations of random_strength, l2_leaf_reg and learning_rate are sensitivities, not extra primary tests.",
        "Test probabilities compare the final epoch-wise DNN and the final CatBoost iteration. They do not choose the peak.",
        "UMAP and n-gram plots use a stratified subsample of at most 20000 rows. PCA and both models use every row.",
        "embeddinggemma:latest context is 2048 tokens. A probe of a long string returned prompt_eval_count 2048.",
        "CatBoost task_type is GPU. DNN uses CUDA torch. One OS user runs every actor label. Labels are not independence.",
        "Competing explanations stay open: label noise, epoch budget, embedding truncation, length features, learning-rate schedule, Bayesian subsample, logloss versus accuracy, and near-duplicate leakage across the random split.",
    ]
    stopping = [
        "One full-mode pipeline run, seed 42.",
        "One primary runner program and one same-data reanalysis of the captured bundle.",
        "No second seed and no change of the peak rule after the curves exist.",
    ]
    hypotheses = [
        ("На валидационной logloss model-wise кривой DNN есть пик у порога интерполяции и второй спуск.",
         "modelwise_peak_excess >= 0.01 и вердикт supports.",
         "Порог интерполяции найден, обе стороны есть, а excess < 0.01 (refutes), либо порог не найден (inconclusive)."),
        ("На epoch-wise кривой самой широкой сети с чистыми метками есть тот же пик валидационной logloss.",
         "Правило пика на логах эпох даёт supports.",
         "Правило даёт refutes или inconclusive."),
        ("У CatBoost есть плато logloss, затем совместный спуск train и validation без роста validation.",
         "Правило плато-затем-спуск даёт supports на основном прогоне.",
         "Плато нет, либо после плато нет совместного спуска без роста validation."),
        ("Наблюдаемый рисунок объясняется шумом меток, бюджетом эпох, обрезкой эмбеддинга, текстовыми длинами, расписанием learning rate, байесовским subsample, выбором метрики или утечкой близких текстов.",
         "Хотя бы одно из этих условий остаётся достаточным объяснением кривой.",
         "Кривая сохраняется в заранее объявленной абляции, которая убирает это условие. Такой абляции в этом прогоне нет для всех пунктов, поэтому гипотеза не закрывается одним графиком."),
        ("Итоговые test logloss DNN и CatBoost различаются на одном и том же split.",
         "Разница CatBoost минус DNN не равна нулю на записанных вероятностях.",
         "Разница равна нулю. Интервал и превосходство на популяции не заявляются."),
    ]
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    with Store(STUDY_ROOT) as store:
        if store.events():
            raise SystemExit("study store already has events; refusing to rewrite the preregistration")
        question_id = _execute(store, "planning.question", {
            "study_id": STUDY,
            "statement": "Появляется ли model-wise double descent валидационной logloss DNN на этом корпусе, и есть ли у CatBoost плато с последующим совместным спуском train и validation?",
            "objective": "Заморозить правила до полного прогона и затем описать один прогон, не выдавая его за популяционную оценку.",
            "scope": scope,
            "constraints": constraints,
            "stopping_criteria": stopping,
        }, actor=PLANNER, role="planner")
        hypothesis_ids = []
        for statement, prediction, falsifier in hypotheses:
            hypothesis_ids.append(_execute(store, "kernel.hypothesis", {
                "statement": statement,
                "prediction": prediction,
                "falsifier": falsifier,
                "scope": scope,
            }, actor=PLANNER, role="planner")["id"])
        explanation_id = _execute(store, "planning.explanation_set", {
            "question": question_id["id"],
            "hypotheses": hypothesis_ids,
            "comparison_plan": "Первичное правило — model-wise peak excess. Epoch-wise, CatBoost и сравнение на test — чувствительности. Конкурирующие объяснения не снимаются этим одним прогоном.",
        }, actor=PLANNER, role="planner")
        pointer = {
            "study_root": str(STUDY_ROOT),
            "question": question_id["id"],
            "hypotheses": hypothesis_ids,
            "explanation_set": explanation_id["id"],
            "pack_code_digest": loaded.pack_code_digest,
            "isolation": "one Windows user, os_isolation=none, sandbox=false; role labels are not independence",
        }
    (STUDY_ROOT / "question_pointer.json").write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(pointer, ensure_ascii=False, indent=2))


def evaluate() -> None:
    raise SystemExit(
        "Historical evaluation is disabled: the text pack's fixed "
        "embeddinggemma/GPU/10%-noise assumptions mismatch current provenance. "
        "Do not relabel artifacts to satisfy the old pack. Use the new workflow "
        "in docs/automation.md; historical stores remain unchanged."
    )
    # Historical implementation retained below for provenance; never execute.
    if not (CAPTURE / "bundle.json").is_file():
        raise SystemExit("missing episteme_capture/bundle.json")
    pointer = json.loads((STUDY_ROOT / "question_pointer.json").read_text(encoding="utf-8"))
    loaded = registry.load_pack("double_descent_text_v1")
    if loaded.pack_code_digest != pointer["pack_code_digest"]:
        raise SystemExit("pack code digest changed after the question; refusing a silent rebind")
    with Store(STUDY_ROOT) as store:
        captured = domain_packs.store_capture(store, "double_descent_text_v1", CAPTURE)
        environment = freeze_environment(store)
        registered = _execute(store, "pack.preregister", {
            "explanation_set": pointer["explanation_set"],
            "pack_id": loaded.pack_id,
            "pack_version": loaded.pack_version,
            "pack_code_digest": loaded.pack_code_digest,
            "parameters": {},
            "host_inputs": {},
            "capture": captured["capture"],
            "environment": environment,
        }, actor=PLANNER, role="planner")
        opened = stage_open(
            store, actor=_actor(PLANNER), study=STUDY, question=pointer["question"],
            explanation_set=pointer["explanation_set"], pack_id=loaded.pack_id,
            preregistration=registered["binding"], attempt_ceiling=2,
            criteria=[{"kind": "analysis_recorded"}])
        closed = stage_run(store, opened["stage"], analyst=_actor(ANALYST),
                           executor=EXECUTOR, replicator=REPLICATOR)
    summary = {
        "capture": captured["capture"],
        "protocol": registered.get("protocol"),
        "binding": registered.get("binding"),
        "stage": opened["stage"],
        "close": closed,
    }
    (STUDY_ROOT / "evaluation_pointer.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("protocol", "binding", "stage")}, ensure_ascii=False, indent=2))
    print(json.dumps({key: closed.get(key) for key in (
        "status", "disposition", "outcome", "inference_mode", "claim", "analysis",
        "scientific_validity", "review")}, ensure_ascii=False, indent=2))


def _actor(actor_id: str):
    from episteme.kernel import Actor
    return Actor(actor_id, "planner" if actor_id == PLANNER else "analyst")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"question", "evaluate"}:
        raise SystemExit("usage: episteme_study.py question|evaluate")
    question() if sys.argv[1] == "question" else evaluate()


if __name__ == "__main__":
    main()
