# R4 — операционно закрыт

Confirmation attempt `20261107T104140-5bc2019cef` закрыт 2026-11-07T10:48:32Z.
Review SHA-256 `16ac881579ff2de3f6d821ceaf6b99e502793e34e3646dbfe9b7b83d274e32fe`.
Операционный reviewer `9506bcbc-93e4-4b02-8fba-30b05c17fbab`, verdict accept.
Design reviewer `c5b0bc65-238c-4285-9953-71a88c525f67`.
`scientific_validity=not_assessed`.

Discovery attempt `20261107T103949-29045fc56f` сохранён. Архив
`research/evidence/R4-runs-2026-11-07-closed.tar.gz`, SHA-256
`eb360f4c0aa7be3a296dc09ab7dba42f90bcb1db78866dcab4348eef63b51e4b`.
Closure copy: `research/evidence/R4-closure.json`.

Шесть сеток не объединены. Рост листьев поднял чистый риск при падении
train-остатка; рост леса при фиксированном потолке листьев чистый риск снизил.
CatBoost не обучался. Claim `R4-separate-axes` локальный. Эскалации 0003 нет.

Следующая команда — подготовить R5:

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```
