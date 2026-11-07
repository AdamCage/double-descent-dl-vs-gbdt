# Состояние на 7 ноября 2026 — R6 закрыт как not_run, дальше R7

R0–R6 операционно закрыты. `scientific_validity=not_assessed`.
Критической эскалации по решению 0003 нет.

**Следующая работа — подготовить R7.** `next` рекомендует `prepare`.
`preparation_authorized=true` и `execution_authorized=false` — незавершённый
preflight, не новый запрос Codex.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

R6 attempt `20261107T111120-40cf0c74f2`, disposition `not_run`, review SHA-256
`859cf65cfae08a8e1049b20728665dc2b006c539adce1be65a4fe1bca7f212c8`.
Архив `research/evidence/R6-runs-2026-11-07-closed.tar.gz`, SHA-256
`cb059f512515c1b0699db277434c367019198ca7aecba73136b527a418748e74`.
Скачиваний и обучений: 0.

R5 архив SHA-256 `aaed282379380969de2e572303ea9048d5c2002f5f2c5a73fa4d422991c02ea9`.
R4 архив SHA-256 `eb360f4c0aa7be3a296dc09ab7dba42f90bcb1db78866dcab4348eef63b51e4b`.
R3 архив SHA-256 `49700f5b51e730343e57b041c6844903c6ae9f073cb03dd4895d159cf956ff2b`.
R2 архив SHA-256 `2131e6e42c29a7be202a9a67ee0d15f4668ebe51c7a8771a416f69c670cbf055`.

Локальные claims не являются внешним подтверждением. Один dispatcher,
CPU до 4 потоков, GPU и платные API равны 0.
