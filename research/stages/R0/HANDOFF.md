# R0 — готово к передаче в Cursor

> R0 закрыт. Текущий вход после уточнения пользователя —
> [R1/HANDOFF.md](../R1/HANDOFF.md) и решение 0003 об автономном Cursor R1–R7.
> Указанное ниже прежнее ожидание разрешения R1 историческое и заменено общей
> авторизацией; R0 evidence и closure не меняются.

Дата: 7 ноября 2026. Ветка: `codex/research-harness-reset`.
R0 закрыт операционно, `scientific_validity=not_assessed`.
Attempt `20261107T070335-5521b09a17`, один зарегистрированный запуск,
13.1 секунды runner wall time, 64 tests: 63 pass + 1 skip (Windows symlink).
Все обязательные структурные проверки прошли. Активного dispatcher lock нет.

## Evidence

- PLAN: `research/plans/R0.json`.
- REPORT: `research/stages/R0/REPORT.md`.
- Bound review: `research/reviews/R0.json` (инженерная self-review Codex;
  scientific independence не устанавливается).
- Archive: `research/evidence/R0-runs-2026-11-07.tar.gz`, 200 файлов.
  SHA-256: `79fa3aca01148d7dadcbc91ab26dc6c3bc513b0ae30c7875dc6e5baa36c62669`.
- Closure: `research/evidence/R0-closure.json`.
  SHA-256: `d6320367dc01ac4d75f62a48cfca391df5ccebb143dcef8ec96c2b7dac33e2f5`.
- Restore check: `research/evidence/R0-recovery.json`. В новом temporary root
  распакованы файлы, восстановлены source snapshots/closure, загружен именно
  snapshotted verifier и выполнены closure verification и next. Получено R0
  closed, R1 ready, execution_authorized=false. Это проверка восстановления
  evidence, не повторное научное вычисление.

## Точная следующая команда

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

Ожидается R1 ready/unauthorized. Пользователь намерен продолжить исследование
после refactor; до этого новые эксперименты не стартуют. Для обсуждения читать
`docs/research-program.md`, `docs/legacy-assessment.md`, `docs/literature/README.md`.
Далее зафиксировать научный дизайн/ресурсы и scope авторизации R1. Cursor
заполняет machine plan и PLAN.md до запуска и самостоятельно идёт по
операционному loop в пределах принятого дизайна. Skeleton budgets не расходуются.

## Восстановление в чистом checkout

Сначала проверить архив:

```powershell
uv run python scripts/evidence_bundle.py verify research/evidence/R0-runs-2026-11-07.tar.gz
```

Убедиться, что `artifacts/research/R0` и `.research/workflow/closures/R0.json`
отсутствуют: существующие данные не заменять. Затем создать пустой каталог,
распаковать архив в него и скопировать closure в отсутствующий целевой файл.
Команды для нового checkout:

```powershell
if (Test-Path -LiteralPath artifacts/research/R0) { throw 'R0 evidence already exists' }
if (Test-Path -LiteralPath .research/workflow/closures/R0.json) { throw 'R0 closure already exists' }
New-Item -ItemType Directory -Path artifacts/research/R0 -Force
tar -xzf research/evidence/R0-runs-2026-11-07.tar.gz -C artifacts/research/R0
New-Item -ItemType Directory -Path .research/workflow/closures -Force
Copy-Item -LiteralPath research/evidence/R0-closure.json -Destination .research/workflow/closures/R0.json
uv run python scripts/research.py verify R0 --attempt 20261107T070335-5521b09a17 --historical
uv run python scripts/research.py status
```

`status` проверяет closure и всю ancestry. Любой invalid запрещает продолжение.
Обычный live verify после новых правок кода может отказать правильно: для
истории используются frozen snapshots. Старые result/DB не восстанавливаются
из R0 bundle как новые подтверждённые эксперименты.

## Открытые вопросы

Старые большие матрицы не хешированы/не архивированы этим reset; corpus validity
не аттестована. Новые scientific adapters, cross-model EpistemeOS DomainPack,
полнотекстовый обзор, точные сетки/статистики и механизмные вмешательства
требуют отдельных frozen plans. EpistemeOS bridge ещё не инициализирован в
реальном новом store и фиксирует только question при явном bootstrap.
Новых accepted empirical claims нет. `llm-semantic-afterlife` не изменялся.
