# Состояние на 7 ноября 2026 — R5 закрыт, дальше R6

R0–R5 операционно закрыты. `scientific_validity=not_assessed`.
Критической эскалации по решению 0003 нет.

**Следующая работа — подготовить R6.** `next` рекомендует `prepare`.
`preparation_authorized=true` и `execution_authorized=false` — незавершённый
preflight, не новый запрос Codex.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

R5 confirmation `20261107T105955-6e7f800ff1`, review SHA-256
`0735cf70372781ed77e5648e3280a3ba5d15d019516c86fd373fd8b8f0e0b338`.
Архив `research/evidence/R5-runs-2026-11-07-closed.tar.gz`, SHA-256
`aaed282379380969de2e572303ea9048d5c2002f5f2c5a73fa4d422991c02ea9`.
Репликация не собрала линейный шум, листья и лес в один механизм.

R4 архив SHA-256 `eb360f4c0aa7be3a296dc09ab7dba42f90bcb1db78866dcab4348eef63b51e4b`.
R3 архив SHA-256 `49700f5b51e730343e57b041c6844903c6ae9f073cb03dd4895d159cf956ff2b`.
R2 архив SHA-256 `2131e6e42c29a7be202a9a67ee0d15f4668ebe51c7a8771a416f69c670cbf055`.

Один dispatcher, CPU до 4 потоков, GPU и платные API равны 0.
Скачивание датасета для R6 не начато.
