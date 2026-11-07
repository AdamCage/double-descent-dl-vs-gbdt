# R1 — следующий разрешённый unit of work для Cursor

Текущая задача: подготовить и выполнить R1, затем самостоятельно продолжить
R2–R7 по [общему мандату](../../../docs/decisions/0003-cursor-autonomous-research.md).
R1 конкретизирован решением 0002. R0 закрыт и не исполняется повторно.
Научного утверждения о результатах этим handoff не делается.

## Начать здесь

Прочитать AGENTS.md, research/STATE.md,
`docs/decisions/0002-r1-audit-authorization.md`,
`research/decisions/R1-authorization.json`, R1 PLAN.md/TASK-CONTRACT.json и
`docs/automation.md`. Проверить decision digest в R1 registry и plan.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

CLI отображает `preparation_authorized=true`, `recommended_action=prepare`
и decision binding. Это разрешает подготовку при execution=false.
False означает необходимость завершить технический preflight, **а не ждать
повторного научного разрешения**. Scope и бюджет уже приняты Codex.

## Подготовка — выполняется сейчас

1. Убедиться, что R0 closure валиден, lock отсутствует, R1 attempts ещё нет.
   Если появился active процесс — проверить владельца, не запускать второй.
2. Начать append-only preparation log с фактическими timestamps, exposure,
   retrieval attempts/bytes, расходом времени/диска и implementation actions.
3. В read-only режиме выполнить ограниченное чтение исторической evidence,
   собрать разрешённые версии L01–L08, реальные assumption cards и proposal R2.
   Недоступность честно зафиксировать; не запускать модели и не качать datasets.
4. Реализовать scripts/tests задачи capture/validation. Machine JSON уже
   содержит непустой task/output contract, но future scripts/notes ещё отсутствуют.
   Дополнить inputs всеми фактическими source/notes/intake файлами. Сохранять
   source registry baseline; результаты R1 писать в новый attempt. Полнотекстовое
   чтение выполняет Cursor, scripts лишь фиксируют и проверяют evidence.
5. Заморозить `research/plans/R1.json` (`status=frozen`) и actual exposure/input
   closure; проверить decision/contract digest и `validate-plan R1`.
   До создания inputs ожидаемый отказ validate-plan для draft не является blocker.
6. После PASS самостоятельно поменять **только R1** execution flag на true.
   Это делегированная активация уже принятого решения, не новое научное решение.
   Не ослаблять runner/check_repository и не создавать dummy outputs ради PASS.

## Исполнение после preflight

```powershell
uv run python scripts/research.py validate-plan R1
uv run python scripts/check_repository.py --output checks.json
uv run python scripts/research.py run R1
```

Записать полученный attempt ID. Затем verify; REPORT со всеми dispositions,
P1–P3, C1–C6, бюджетами и missing evidence; bound operational review; close;
evidence bundle с SHA-256 и проверкой восстановления; обновить STATE/HANDOFF.
Для review/close использовать команды docs/automation.md с R1 и точным attempt.
Итоговая scientific_validity остаётся not_assessed. Нужны реальные отдельные
author/reviewer subagent IDs, review_mode=subagent_review и содержательное
review, а не подпись автора. После R1 продолжить prepare/freeze/activate R2
самостоятельно; human/Codex нужен только по критериям критической эскалации 0003.

## Не выходить за границы

120 минут Cursor; 900s/attempt, 1800s runner total, максимум 2 attempts;
200 MiB network, 25 MiB/doc, 2 retrieval attempts/source, 48 HTTP requests;
8 обязательных source dispositions и до 4 конкретно мотивированных дополнений;
2 GiB новых файлов. GPU/training/embeddings/datasets/paid API: 0.
Не менять исторические result/DB, frozen R0 evidence и llm-semantic-afterlife.
Если scope/бюджет недостаточен или integrity не восстанавливается — сохранить
промежуточный packet и вернуть точное решение Codex; не обнулять лимиты.
