# Состояние на 7 ноября 2026 — R2 закрыт, дальше R3

По прямому уточнению пользователя основное исследование выполняет Cursor с
субагентами. Человек/Codex — верхнеуровневая критика и критические моменты.
Действуют решение `docs/decisions/0003-cursor-autonomous-research.md` и bound
`research/decisions/autonomous-research-authorization.json`.

**Следующая работа — подготовить R3.** `next` рекомендует `prepare`.

R2 операционно закрыт: confirmation attempt `20261107T095343-6bc0c457c7`,
review SHA-256 `a4e389ae2bf47a1920145439eae84fd9621ff5d6fe407dfbb423762728d79b7b`,
`scientific_validity=not_assessed`. Design review
`a80b8102-ffa6-4830-a59c-a18a59ee8989`. Операционный review
`50e3cc99-ebd6-4007-bdaa-cee30bac73cf`, verdict accept. Discovery attempt
`20261107T095017-e76868f52e` сохранён. Архив
`research/evidence/R2-runs-2026-11-07-closed.tar.gz`, SHA-256
`2131e6e42c29a7be202a9a67ee0d15f4668ebe51c7a8771a416f69c670cbf055`.
Claim `R2-confirmation-contrasts` — локальный `evidence_supported` /
`cursor_team_reviewed`. Критической эскалации 0003 нет.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

В пределах программы Cursor сам выбирает подробные протоколы, discovery и
confirmation, делает анализ/репликации и пишет статью. Обязательны evidence,
общие ресурсы и review отдельным субагентом. Human approval не выдумывать.
Обычный отрицательный/inconclusive исход не блокирует автономную работу.
Общий runner бюджет — 24 часа, CPU до 4 threads/job, один compute dispatcher,
GPU/paid APIs = 0.

R0 и R1 операционно закрыты и не переписываются. R1: attempt
`20261107T085630-b4345b8c44`. L01 PDF недоступен после двух HTTP 403.
Сравнение осей L05–L07 неразрешено и не переносилось в R2.

Источники состояния: `research/stages.json`, live attempts/closures и CLI;
решений — `docs/decisions/` и `research/decisions/`; выводов —
`research/claims.json`.
