# Состояние на 7 ноября 2026 — автономный Cursor R1–R7

По прямому уточнению пользователя основное исследование выполняет Cursor с
субагентами. Человек/Codex — верхнеуровневая критика и критические моменты,
не обязательный sign-off каждого stage, дизайна или claim.
Действуют решение `docs/decisions/0003-cursor-autonomous-research.md` и bound
`research/decisions/autonomous-research-authorization.json`.

**Следующая работа — подготовить R2.** `next` рекомендует `prepare`.
R1 операционно закрыт: attempt `20261107T085630-b4345b8c44`, review SHA-256
`c9f57b1459fe0fe5f1ac9139c2fa5db2b2e18a6c3816cd0bdd24ecee60391444`,
`scientific_validity=not_assessed`. Design review
`03f147be-80d7-42b8-9499-5473f6a5ffc9`. Операционный review
`88fc6e9f-cf60-40ad-a8c9-95625863d952`, verdict accept. L01 PDF недоступен
после двух HTTP 403. Сравнение осей L05–L07 неразрешено. Модели не обучались.
Архив с review: `research/evidence/R1-runs-2026-11-07-closed.tar.gz`, SHA-256
`9f0ebf4f76c0152e24c7b558025ed16e9a5872c215c037f732898b33cf030007`.
Вход R2: `research/stages/R2/HANDOFF.md`.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

В пределах программы Cursor сам выбирает подробные протоколы, discovery и
confirmation, делает анализ/репликации и пишет статью. Обязательны evidence,
общие ресурсы и review отдельным субагентом; поддержанные выводы допускаются
как `evidence_supported/cursor_team_reviewed`, без выдуманного human approval.
Критические неожиданности после диагностики, развилки общей цели, центральная
неустранимая integrity проблема или выход за мандат передаются человеку/Codex
с exact evidence packet. Обычный отрицательный/inconclusive исход не блокирует
автономную работу. Общий runner бюджет — 24 часа, CPU до 4 threads/job, один
compute dispatcher, GPU/paid APIs = 0; подробности и более строгий R1 — в charter.

R0 операционно закрыт: attempt `20261107T070335-5521b09a17`, его frozen evidence
и closure не менялись. Исходный R0 suite: 64 tests (63 pass, 1 Windows symlink
skip). Архив `research/evidence/R0-runs-2026-11-07.tar.gz` и recovery receipt
сохранены. Текущие tests отдельно проверяют новую автономную authorization lane.

Новых R1 model runs, новых scientific claims и реального нового EpistemeOS
bootstrap пока нет. Исторические result/DB, dependency snapshot и
`llm-semantic-afterlife` не изменять. Частичная старая MLP сетка и metadata
конфликты остаются not_assessed за пределами сохранённых малых наблюдений.

Источники состояния: `research/stages.json`, live attempts/closures и CLI;
решений — `docs/decisions/` и `research/decisions/`; выводов —
`research/claims.json`. Не использовать старое R0-only правило как текущую границу.
