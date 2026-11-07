# R6 — операционно закрыт как not_run

Attempt `20261107T111120-40cf0c74f2` закрыт 2026-11-07T11:14:07Z.
Review SHA-256 `859cf65cfae08a8e1049b20728665dc2b006c539adce1be65a4fe1bca7f212c8`.
Операционный reviewer `b603ed46-4e6d-425e-b193-aa9eec4ce674`, verdict accept.
Design reviewer `5f1d2237-db12-4c0f-97ca-78f274f68af7`.
`scientific_validity=not_assessed`.

Реальная таблица не скачивалась и не обучалась. Архив
`research/evidence/R6-runs-2026-11-07-closed.tar.gz`, SHA-256
`cb059f512515c1b0699db277434c367019198ca7aecba73136b527a418748e74`.
Closure copy: `research/evidence/R6-closure.json`.

Причина: R2–R5 не дали одного механизма для переноса. Это не измерение
отсутствия эффекта на реальных строках.

Следующая команда — подготовить R7:

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```
