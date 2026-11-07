# Изменение автономии — инженерное review

Дата: 2026-11-07. Основание: прямое указание пользователя о почти полностью
автономном исследовании Cursor с субагентами. Это governance/code change,
не scientific R1 attempt и не новый эксперимент.

Выполнено: единая авторизация R1–R7, preparation/execute preflight, explicit
next action, SHA-256 bound authorization, distinct-subagent stage/claim review,
конкретные R1 contracts и обновлённые live правила без per-stage Codex sign-off.
Ordinary negative/inconclusive outcomes не требуют эскалации; критические
неожиданности/развилки/central integrity/мандат имеют evidence escalation path.

Validation: `checks.json`, structural_problems=[], 81 tests (80 pass, один
Windows symlink skip). R0 historical verification и closure прошли; frozen
plan/report/review/archive/closure R0 не изменялись. `next`: R1 ready,
preparation_authorized=true, execution_authorized=false,
recommended_action=prepare, blockers=[]. Никакого реального R1 attempt нет.

Ревью реализации выполнялось в разных агентских контекстах той же команды:
`r1_operational_review` — workflow gates/tests;
`autonomous_claim_gate` — claim admission/tests;
`r1_scientific_design` — read-only итоговая проверка научной/операционной
согласованности и live decision digests. Внешняя независимость не заявляется.

Ограничение: общий charter resource budget пока требует coordinator/task-level
accounting сверх generic runner caps. В R1 PLAN явно поручено Cursor реализовать
и проверить общий ledger до вычислений R2–R7. Это разрешённая реализация,
не причина ждать ещё одного решения Codex.

Current execution=false не считать scientific authorization blocker: Cursor
сам реализует/freeze/validate/activate stage по уже принятому мандату.
Все source snapshots, журналы, claims и manuscript остаются evidence-bound;
team review не означает human approval или доказательство научной истины.
