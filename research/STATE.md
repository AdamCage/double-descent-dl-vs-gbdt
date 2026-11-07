# Состояние на 7 ноября 2026 — R3 закрыт, дальше R4

R0, R1, R2 и R3 операционно закрыты. `scientific_validity=not_assessed`.
Критической эскалации по решению 0003 нет.

**Следующая работа — подготовить R4.** `next` рекомендует `prepare`.
`preparation_authorized=true` и `execution_authorized=false` — незавершённый
preflight, не запрос нового разрешения Codex.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

R3 confirmation `20261107T102511-b1cf6c7be9`, review SHA-256
`6c17b3010fdb68ae525962cfec673fac90579a878ca23d99eaaa1edd22756743`.
Операционный reviewer `6d8384f1-1695-4596-8b33-91a0b28af508`.
Архив `research/evidence/R3-runs-2026-11-07-closed.tar.gz`, SHA-256
`49700f5b51e730343e57b041c6844903c6ae9f073cb03dd4895d159cf956ff2b`.
Короткий бюджет из 40 эпох имел меньший чистый риск, чем обучение до допуска
0.001. Claim `R3-confirmation-budget` локальный.

R2 confirmation `20261107T095343-6bc0c457c7`, архив
`research/evidence/R2-runs-2026-11-07-closed.tar.gz`, SHA-256
`2131e6e42c29a7be202a9a67ee0d15f4668ebe51c7a8771a416f69c670cbf055`.
Commit закрытия R2: `06a3afe32b329100f2119b665443281c5991383f`.

Общий runner бюджет — 24 часа, CPU до 4 потоков, один dispatcher, GPU и
платные API равны 0. L01 и оси L05–L07 по-прежнему не склеены.
