# R5 — операционно закрыт

Confirmation attempt `20261107T105955-6e7f800ff1` закрыт 2026-11-07T11:07:01Z.
Review SHA-256 `0735cf70372781ed77e5648e3280a3ba5d15d019516c86fd373fd8b8f0e0b338`.
Операционный reviewer `8da4cdef-70ec-45ba-9a94-2d10eeb452f3`, verdict accept.
Design reviewer `d07fa088-13ac-4dfa-bad6-eb4fad7ed629`.
`scientific_validity=not_assessed`.

Discovery attempt `20261107T105818-f129d71a55` сохранён. Архив
`research/evidence/R5-runs-2026-11-07-closed.tar.gz`, SHA-256
`aaed282379380969de2e572303ea9048d5c2002f5f2c5a73fa4d422991c02ea9`.
Closure copy: `research/evidence/R5-closure.json`.

Четыре заранее заданных контраста на seeds 8101–8105 получили `supported`.
Это не один механизм. Результат MLP про 40 эпох и кривая глубины бустинга
не переоценивались. Claim `R5-replication-not-one-mechanism` локальный.
Эскалации 0003 нет.

Следующая команда — подготовить R6:

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```
