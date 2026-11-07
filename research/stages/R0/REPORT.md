# R0 — исследовательский workflow готов к передаче

Дата: 7 ноября 2026. Scope: инженерный reset по запросу пользователя.
Новых научных экспериментов, обучения моделей или платных API-вызовов нет.
`scientific_validity=not_assessed`; научного одобрения результатов нет.

## План и evidence

- Attempt: `20261107T070335-5521b09a17`.
- Plan SHA-256: `ab204a193f8c3fe385fe921b0be9c1cb23d3af25b0e7ea1899832da966f5d233`.
- Manifest SHA-256: `a321178e4d9c7ffad7bf4444d5b2dd353b33eb94c69bb930e6b2523d3479f206`.
- Evidence: `artifacts/research/R0/20261107T070335-5521b09a17/tasks/repository-check/checks.json`.
- Frozen source/input files: 193. Environment and package versions are in manifest;
  original Git HEAD: `115d501`, branch `codex/research-harness-reset`, dirty/untracked
  state explicitly recorded and declared source bytes copied into `inputs/`.
- Commands: `validate-plan R0`, `run R0`, `verify R0 --attempt 20261107T070335-5521b09a17`.
  Task ran `scripts/check_repository.py`; runner elapsed approximately 13.1 seconds.

Это единственный зарегистрированный R0 attempt. До freeze выполнялись локальные
unit/structural проверки при разработке, без научных вычислений; они не включены
как подтверждающие scientific replicates и не заменяют final attempt.

## Критерии

| ID | Исход | Evidence и смысл |
| --- | --- | --- |
| C1 | PASS | `checks.json`: structural_problems=[], locked environment установлен `uv sync --frozen`; R0 plan валиден, R1–R7 draft/unauthorized |
| C2 | PASS | 64 tests, 63 pass и 1 skip; corruption, transitive closure, exact matrix, missing outputs, retries, timeout, stale review и scientific-claim authority rejection |
| C3 | PASS | Structural check сверяет переносимые исторические snapshots с inventory SHA-256; большие матрицы не аттестованы |
| C4 | PASS | Round-trip bundle test проходит после удаления исходного каталога; изменённые bytes и `.env`/`.env.*` отвергаются |

P1 наблюдено: offline checks завершены, structural failures нет, research training
и API не запускались. P2 наблюдено: negative fixtures отвергаются gates.
Артефакт для обоих — `repository-check/checks.json`; исходные test names и
unittest output включены в JSON и stderr log. Эти тесты проверяют инженерные
контракты, а не эффект double descent.

Skip: тест создания symlink вне repo не выполнен из-за Windows WinError 1314.
Остальные path confinement tests прошли. Это явный предел host verification,
а не обещание sandbox или недоступности секретов произвольному task.

## Что изменено

Введены автономный Cursor stage loop, DAG, frozen plans, task-level logs,
source snapshots, time/retry caps, integrity verification, structured operational
review и immutable closures с рекурсивной проверкой dependency evidence.
Новый EpistemeOS bridge фиксирует только вопрос в отдельном store; старые
неверные EmbeddingGemma/GPU/noise exporter/evaluate не исполняются.
Новая программа охватывает linear/ridge, random features, MLP и trees/GBDT.
Старый основной вопрос о позднем ускорении CatBoost стал отдельной линией.

История сохранена: исходные result/DB не менялись; 48 файлов перечислены в
legacy inventory, 17 небольших logs/JSON/notes скопированы с SHA-256. Частичная
MLP сетка, рост loss на width 512, отсутствие правого плеча, округлённая accuracy
и metadata mismatch 768/4096 явно ограничивают интерпретацию. Рукопись теперь
помечена исторической pilot note; фактические pilot числа сохранены.

## Ревью и ограничения

Отдельный агентский контекст выполнил read-only engineering review. Найдены и
исправлены: (1) обход integrity gate через повреждённого предка R0→R1→R2;
(2) включение `.env.local` в bundle. Оба случая имеют regression coverage в
финальном suite. Это проверка кода в рамках одной команды; независимость
научного рецензента не установлена. Итоговый bound operational review фиксирует
автор refactor, честно как self-review инженерной миграции.

Ограничения: Cursor runtime не является внешним постоянно работающим сервисом;
возобновление идёт из packet. Общий runner не ограничивает денежные расходы или
размер task outputs. Список transitive inputs проверяет reviewer. Нового
cross-model EpistemeOS DomainPack, научных адаптеров, scientific protocols и
подтверждённых claims ещё нет; их реализация относится к разрешённым будущим
этапам исследования. Начальный реестр 8 первичных источников проверен на уровне
metadata/abstract, полнотекстовые snapshots и анализ запланированы в R1.

## Передача

После freeze review и closure полный R0 evidence экспортируется в
`research/evidence/R0-runs-2026-11-07.tar.gz`, closure — в `research/evidence/R0-closure.json`.
Digest и restore/check команды записываются в HANDOFF, отдельно от этого
hash-bound отчёта. Финальный архив не является циклическим доказательством
собственной готовности. Далее — обсуждение научной программы; R1 пока не запущен.
