# 0005 — ограниченный переход к проверке применимости frozen response

8 ноября 2026. Codex coordinator `/root`, на основании прямой авторизации
пользователя менять stages/начать заново и последнего указания пересматривать
подходы, если ценных результатов недостаточно. Родительская авторизация0004
и все общие budgets остаются прежними. Это не одобрение научного результата.

R10–R12 дают полезный audit loss-based inference, но не устанавливают новый
механизм double descent или полноценное сравнение MLP–GBDT. Эти результаты
сохраняются, R13 завершает их производную поставку. Количество рисунков не
превращает их в новый механизм. Исходный текстовый domain с нерешённым
provenance не используется для новых claims.

Вместо дальнейшего перебора формы loss-кривой R14 проверяет новый candidate:
можно ли до полного adaptive refit предсказать finite-label-response ошибку
точного условного frozen-tree replay. Это вопрос применимости объяснения
адаптивной модели. Known LeafRefit, EFC, operator noise gain и representation
floor являются baseline/controls, не нашей новизной. Отдельный literature audit
и confirmation нужны даже при положительном discovery.

Научный дизайн, estimands, все seeds/grids и stop/go criteria фиксируются в
research/stages/R14/PLAN.md и JSON до генерации study data. Рецензент отдельный
от автора реализации/вывода. R14 execution только после закрытого R13, полного
preflight и cumulative resource check. Не переносить synthetic calibration
на новые curves. Не адаптировать grid по endpoint и не считать correlated
contexts независимыми draws.

Если discovery prediction/cost gates провалены, этот candidate останавливается
с честным negative/inconclusive report. Ни удачная known-control curve, ни
переименование результата не означают достижения полной первоначальной цели.
Любой следующий содержательный pivot требует отдельного записанного решения.
