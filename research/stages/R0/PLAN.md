# R0 — рефакторинг исследовательского workflow

Статус: engineering; запрос пользователя 7 ноября 2026 разрешает рефакторинг.
Новые научные эксперименты не входят в scope.

## Вопрос и предсказания

Можно ли исполнять stages в Cursor с воспроизводимыми inputs/source snapshots,
артефактами и точным операционным review, сохранив независимый научный маршрут?
P1: все offline checks пройдут без обучения и платных API.
P2: negative gate tests отклоняют corrupt evidence, неподтверждённые claims,
неавторизованные stages, пропуски, таймауты и stale reviews.
Фальсификатор инженерной готовности — любой падающий обязательный gate.

## Дизайн

Один repository-check task запускает unittest suite и structural checks.
Данные корпуса не читаются; маленькие процессы и temp fixtures проверяют runner.
Полный список source/input файлов фиксируется в research/plans/R0.json.
Per-attempt 120s, cumulative 240s, максимум 2 attempts после freeze.
Если понадобится менять проверяемый код, новый attempt сохраняет предыдущий исход.

## Критерии выхода

C1: executable plans, authority split и dependency lock согласованы.
C2: corruption/missing-output/timeout/review/claim negative tests проходят.
C3: исторические snapshots и их SHA-256 проходят structural verification.
C4: переносимость архива проверена offline round-trip тестами, включая
проверку без исходного каталога и отказ на изменённых bytes/секретах.
Stage report, exact bound review, closure и handoff оформляются после task;
финальный evidence archive создаётся после closure и не используется как
циклическая проверка собственной готовности.

Научные stages R1–R7 остаются выключенными. Closure R0 операционный и не
удостоверяет никакой эффект двойного спуска.
