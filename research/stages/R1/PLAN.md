# R1 — аудит истории, первичных работ и проект R2

Статус: **frozen после design review**.
Preflight на 2026-11-07: L02–L08 сохранены как PDF и проверены по текстовому слою;
L01 получил HTTP 403 на обеих попытках и не заменён препринтом. Сравнение осей
L05–L07 оставлено неразрешённым. Предложение R2 не запускает модели.
Разрешены подготовка Cursor и условная активация исполнения после preflight.
Решение: `docs/decisions/0002-r1-audit-authorization.md` и
`research/decisions/R1-authorization.json`. Точные outputs/criteria:
`research/stages/R1/TASK-CONTRACT.json`. Это описательный аудит, не confirmation.

## Вопрос и уже доступные результаты

Какую evidence можно использовать для дизайна исследования механизмов, а какие
свойства остаются неизвестными? Уже прочитаны ревью 4/6 ноября, программа reset,
историческая оценка, округлённые строки MLP до width 512 и метаданные пилота.
Нельзя объявлять эти известные наблюдения независимой проверкой новой гипотезы.
Cursor добавляет в `preparation-log.jsonl` actual source exposure, retrieval,
implementation/checks и расход общего бюджета до freeze.

## Предсказания и условия опровержения

- **P1:** существующих небольших артефактов недостаточно для утверждения о
  double descent на старом корпусе. Если найден согласованный финальный
  full-precision результат, противоречащий этому ожиданию, записать его bytes,
  hash и locator и пересмотреть disposition; не подгонять аудит под P1.
- **P2:** сравнение определения осей, loss и допущений существенно для переноса
  выводов tree/GBDT работ. Если сопоставимость нельзя установить по доступному
  full text, исход unresolved, а не выдуманное согласование работ.
- **P3:** R2 можно предложить как различающие контрасты noise/spectrum/ridge/
  random features с измеряемой интерполяцией. Если допущения/численная процедура
  не обоснованы, proposal получает blocked/limited readiness, без запуска.

Это ожидания аудита, не набор statistical tests; p-values или confidence
intervals для наличия источника не строятся. Единица учёта — artifact/source/
proposed contrast. Список missing/unavailable items обязателен.

## Подготовка и точный исполнимый этап

Код реализации и source closure создаёт Cursor. До freeze он читает разрешённые
тексты, составляет agent-authored notes, сохраняет acquisition metadata и
разрешённые snapshots в новом intake `artifacts/intake/R1/<intake-id>/`.
Snapshots неизменяемы, все они становятся declared inputs. Не смешивать intake
с `result/`, старыми DB или outputs зарегистрированного attempt.

Planned entry point: `scripts/r1_capture.py` с task subcommands. Допустима иная
внутренняя модульная организация при сохранении научного task/output contract.
Никакой script не вызывает внешнюю LLM и не притворяется автором scientific
review. Frozen notes содержат реальные выводы Cursor и locators; deterministic
scripts проверяют их структуру, hashes, coverage и экспортируют результаты.
Все raw notes, acquisition index и подготовленные proposals включаются во входы;
их provenance и pre-freeze exposure отражаются честно.

Порядок machine tasks: legacy-evidence → literature → r2-design → audit-validate
→ repository-check. Последний проверяет инфраструктуру, а не научную истину.
Validator сопоставляет все item IDs и evidence paths с frozen input manifests,
отклоняет пропуски, структурно неверные locators и повышение abstract-only статуса без
соответствующего full text. Семантику locators агент проверяет при ревью.

## Выходы и критерии

До активации вычислительных R2–R7 Cursor реализует и тестирует общий resource
ledger/accounting: cumulative wall time всех attempts, включая failed/interrupted,
storage/network и остаток charter. Общий runner сейчас проверяет finite
plan/task/attempt caps, но сам не доказывает соблюдение общих 24 часов/20 GiB.
`audit-validate` сообщает, что учёт реализован и проверен, либо явно оставляет
вычислительную активацию зависимых stages неготовой. Обычный validate-plan PASS
не заменяет эту проверку; новое согласование Codex для реализации учёта не нужно.

История: `legacy-audit.json/md`; литература: `literature-audit.json`,
`source-index.json`, карточки L01–L08; предложение: `R2-design-proposal.json/md`;
валидация: `validation.json`; инфраструктура: `checks.json`.
Точные поля и критерии C1–C6 заданы TASK-CONTRACT и machine draft.
Полнота означает обоснованный disposition каждого item, а не обязательное
подтверждение старых результатов или доступность каждой статьи.

## Ресурсы и завершение

120 минут Cursor суммарно; runner 900s/attempt, 1800s всего, 2 attempts;
до 200 MiB загрузок, 25 MiB/document, 2 retrieval attempts/source,
48 requests, максимум 4 дополнительных первичных источника, 2 GiB новых файлов.
GPU/training/datasets/embeddings/paid APIs = 0. Допуски бюджетов не расширять
самостоятельно. Не тратить ресурсы на новые model runs или полный hash матриц.

После operational review/close сохранить evidence и точные limitations,
получить содержательное review R2 proposal отдельным Cursor субагентом и
продолжить подготовку/активацию R2 по решению 0003. Codex/человеку передавать
критические вопросы, а не routine переход. `scientific_validity=not_assessed`;
локальные claims требуют отдельного bound subagent review. R1 loop уже
разрешён; повторного разрешения на код, freeze и activation не нужно.
