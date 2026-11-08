# Draft: alignment portfolio и отдельная журнальная траектория

8 ноября 2026, Europe/Moscow. Действует решение0006. Это новый дизайн под
уточнённую цель; он не утверждает, что старое исследование стало успешным,
и пока не запускает numerical stages. Никакой новый manuscript не считается
готовым к журналу или гарантией отбора в Residency.

Официальная [Residency](https://alignment.anthropic.com/2026/anthropic-residency/)
этого набора принимает заявки до22ноября2026; она ориентирована на established
technical leaders, опыт ML/PhD/публикации не обязательны. Профиль пользователя
и его фактическая дата подачи пока неизвестны. Поэтому двухнедельный roadmap
условен на этот набор. Он должен показывать его настоящий опыт и суждение.

## Предварительный основной вопрос

Сохраняет ли AI-monitor заявленную способность находить нарушения, когда
prompt/monitor выбран на development set, а затем меняется семейство задач?
Что происходит при фиксированном бюджете false alarms/человеческих проверок?
Может ли family-separated selection/calibration давать более надёжные выводы
о deployment performance, чем обычная процедура выбора лучшей конфигурации?

Это кандидат, не приоритет и не заранее положительный результат. Known nested
validation/conformal calibration/base-rate effects — controls, не новая теория.
Новый вклад возможен как проверенный вывод о конкретном alignment evaluation
protocol и actionable method; если ближайшая работа уже полностью отвечает
на вопрос или effects practically negligible, candidate останавливается.

Он связан с официальными [behavioral monitoring/scalable oversight directions](https://alignment.anthropic.com/2025/recommended-directions/).
Ближайшие priors/конкуренты и альтернативы должны быть проверены в
alignment-options-2026-11-08.md и independent-strategy-review-2026-11-08.md.
Доступ к готовым raw outputs/scores не предполагать: code/example release не
равен полноценному открытому dataset. Никакие private/gated данные не получены.

## Предлагаемые stages

| Stage | Работа и проверяемый выход | Условие остановки/продолжения |
| --- | --- | --- |
| R15: topic/prior/data gate | Сравнить3узких вопроса; точные ближайшие priors, доступные raw examples/labels/scores, license/access, compute estimate, одна выбранная гипотеза и её reviewer critique | В течение48h подготовки должен появиться внятный пробел и реально исполнимый pilot. Иначе остановить candidate, не строить большой framework |
| R16: frozen pilot | Один primary endpoint, независимая task-family единица, paired controls, простые сильные baselines, отдельно выбор конфигурации/калибровка/evaluation, caps/meaningful effect заранее | Negative/inconclusive сохраняется. Не расширять prompts/models/tasks ради успеха; confirmation только по заранее заданному go rule |
| R17: independent confirmation | Новые held-out task families/данные и заранее заданный model-transfer check, locked pipeline, cluster-wise uncertainty и latency/token/API cost | Непереносимость эффекта ограничивает claim; не скрывать её средним по всем задачам |
| R18: evidence/report/portfolio | Привязанные claims, raw runs/models/prompts/version metadata, reproducible script, графики с controls, краткий English report и user explanation of decisions/limitations | Публичный report/заявка только после проверки; journal expansion отдельно по достаточности evidence |

R15 не означает, что уже разрешено запуститьR16. IDs пока design-only и не
добавлены в machine registry. Для первого numerical stage авторизация/JSON
будут зарегистрированы только с complete input closure и отдельным preflight.
Текущие CPU/API/GPU/storage caps не меняются; повторный общий resource check
обязателен. Бюджет model API или GPU нельзя предположить из цели portfolio.

## Что требуется от результата

Для ближайшей заявки полезен ограниченный честный artefact: ясно проверяемый
вопрос, самостоятельное решение пользователя о controls/interpretation,
working implementation и1–2убедительных figures с полными supplementary data.
Полная галерея сохраняет все объявленные conditions; число figures само по
себе не критерий успеха. Нужны также воспроизводимый запуск и понятные границы.

Пользователь должен реально участвовать: сформулировать мотивацию, выбрать
между competing explanations, посмотреть случаи ошибок и защитить вывод.
Исходный вклад и AI assistance документируются, не выдумываются. Официальная
[AI guidance](https://www.anthropic.com/candidate-ai-guidance) подчёркивает
реальный опыт и собственное мышление кандидата. Заявку/личный first draft
этот исследовательский процесс за пользователя не создаёт.

Для журнальной версии нужны generalizable/actionable findings и достаточно
независимого evidence; одна демонстрация/повторение не считается достаточным.
[Критерии TMLR](https://jmlr.org/tmlr/acceptance-criteria.html) требуют убедительной
поддержки claims и интереса/ясности для аудитории; новая методика не обязательна.
Venue и выполнение требований конкретной аспирантуры ещё не выбраны.
За две недели не обещается accepted journal paper; portfolio MVP и journal
study имеют разные критерии и сроки.

Нужные уточнения: фактический technical/research background пользователя,
срок целевой подачи и доступный CPU/API/GPU бюджет. Read-only подготовка
идёт без них; зависимые ресурсоёмкие runs до их разрешения не начинаются.
