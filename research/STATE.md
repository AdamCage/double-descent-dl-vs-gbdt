# Состояние на 7 ноября 2026 — R4 закрыт, дальше R5

R0–R4 операционно закрыты. `scientific_validity=not_assessed`.
Критической эскалации по решению 0003 нет.

**Следующая работа — подготовить R5.** `next` рекомендует `prepare`.
`preparation_authorized=true` и `execution_authorized=false` означают
незавершённый preflight, не новый запрос Codex.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

R4 confirmation `20261107T104140-5bc2019cef`, review SHA-256
`16ac881579ff2de3f6d821ceaf6b99e502793e34e3646dbfe9b7b83d274e32fe`.
Архив `research/evidence/R4-runs-2026-11-07-closed.tar.gz`, SHA-256
`eb360f4c0aa7be3a296dc09ab7dba42f90bcb1db78866dcab4348eef63b51e4b`.
Оси листьев, размера леса, шагов, глубины и доли признаков не склеены.

R3 confirmation `20261107T102511-b1cf6c7be9`. Короткий бюджет MLP имел меньший
чистый риск, чем обучение до допуска 0.001. Архив SHA-256
`49700f5b51e730343e57b041c6844903c6ae9f073cb03dd4895d159cf956ff2b`.

R2 confirmation `20261107T095343-6bc0c457c7`. Архив SHA-256
`2131e6e42c29a7be202a9a67ee0d15f4668ebe51c7a8771a416f69c670cbf055`.

Один dispatcher, CPU до 4 потоков, GPU и платные API равны 0. L01 и L05–L07
не продвинуты.
