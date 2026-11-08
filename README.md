# Аудит сопоставимости CoT-оценок

8 ноября 2026 завершён воспроизводимый анализ публичных cached evaluations:
восемь исторических моделей, BBH/GPQA/MMLU, все 28 пар. На отложенных вопросах
**16 из 56 поддержанных сравнений** прошли заранее заданный критерий сдвига
(9 BBH, 7 MMLU); максимальный сдвиг составил **15,52 процентного пункта**.
Вывод касается опубликованной judge-ACK аннотации и популяции сравнения.

Итоговый [GitHub Release](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/tag/research-cot-audit-2026-11-08):

- [Девятистраничный PDF](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/download/research-cot-audit-2026-11-08/manuscript.pdf).
- [Переносимый ZIP: 111 рисунков PNG/SVG, восемь CSV, код, данные, replay и reviews](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/download/research-cot-audit-2026-11-08/cot-audit-portfolio-2026-11-08.zip).
- После распаковки ZIP открыть `cot-audit-portfolio/portfolio/gallery/index.html`.
- [Английская рукопись](manuscript/cot_input_comparability.md) и [краткое описание для портфолио](docs/portfolio/cot-audit-overview.md).
- [Текущий статус и восстановление](research/STATE.md), [ограниченный научный claim](research/decisions/H2-sensitivity-claim.json).
- [Состав релиза, SHA256 и команды восстановления](docs/releases/research-cot-audit-2026-11-08.md).

Обе фазы полностью воспроизведены из portable evidence: все 12 сравнений файлов
без численных расхождений. Отдельные агенты независимо проверили числа,
111 PNG/SVG и все девять PDF-страниц. 239 unit tests прошли, девять opt-in
fitting tests пропущены. API/GPU/обучение новых моделей не использовались.
Резервные 35% outcome-данных сохранены неиспользованными.

Метод сопоставления и общая проблема selection уже известны. Вклад — конкретный
эмпирический аудит этого pinned release и воспроизводимый инструмент проверки;
внешний приоритет и готовность к сильному журналу не установлены. H1 с излишним
условием по literal cue, его неудачная integrity-попытка и визуальная ревизия
первой сборки сохранены. Подробности: [научный отчёт R17](research/stages/R17/REPORT.md),
[итог поставки R18](research/stages/R18/REPORT.md),
[проверка приоритета](docs/literature/cot-audit-priority-2026-11-08.md).

## Архив исследования двойного спуска

Прежняя программа R0-R14 закрыта и сохранена отдельно. Новый механизм double
descent у MLP/CatBoost не установлен; известный линейный контроль не является
такой новизной. Материалы ниже остаются историей, а не очередью новых запусков.

Цель — исследовать, при каких условиях двойной спуск возникает, исчезает и
смещается у линейных моделей, случайных признаков, нейросетей и деревьев/GBDT.
Отсутствие эффекта — допустимый результат. Позднее ускорение LogLoss в CatBoost
остаётся отдельным вопросом, а старый текстовый эксперимент — историческим кейсом.

Проект перестроен 7 ноября 2026 и дополнен по [рецензии v2](reviews/07-11-2026-v2.md).
Научная программа: [docs/research-program.md](docs/research-program.md).
Текущий статус: [research/STATE.md](research/STATE.md).

Результаты новой ревизии изложены в [рукописи](manuscript/acceleration_audit_v2.md).
R10–R12 выполнены и получили отдельные evidence-bound reviews:

- R10:7 188 траекторий с отдельной калибровкой и оценкой. Некалиброванный
  локальный detector отметил все2 400 null eval curves; калибровка снижает
  ложные сигналы, но не гарантирует высокую чувствительность при сильном шуме.
- R11:72 парных CatBoost fit. Знак step-up contrast зависит от заранее
  объявленной шкалы времени; один step-down contrast остаётся inconclusive.
- R12:72 continuous fit при постоянном шаге. Увеличение random_strength
  повышает максимальный scan score, однако все6 fixed late-event gates дают
  not_established. Это не доказательство отсутствия событий в других режимах.

Общий механизм double descent или естественного позднего ускорения этим не
установлен. Известные scan/rank методы не выдаются за новый общий алгоритм.
R13 закрыт: [галерея187 PNG/SVG pairs и2 HTML](artifacts/research/R13/20261107T214015-f184806e46/tasks/figures/gallery/index.html),
16 исходных CSV и [самостоятельный пакет](artifacts/research/R13/20261107T214015-f184806e46/tasks/package/science-package/).
Полный portable R10 replay совпал точно; все187 PNG просмотрены в обзорных
листах и9 ключевых — отдельно. [Review и ограничения](research/reviews/R13-attempt3-v2.json).
[Discovery R14: прогноз ошибки frozen-tree response](research/stages/R14/PLAN.md)
также завершён; его результаты и ограничения сохранены в
[отчёте](research/stages/R14/REPORT.md). Новые запуски не поставлены в очередь.

## Работа в Cursor и Codex

Cursor с выбранным пользователем Grok 4.7 Extra High и субагентами автономно
планирует, реализует, исполняет, анализирует и рецензирует исследование, ведёт
доказательства и пишет статью. Человек и Codex дают верхнеуровневую критику и
принимают критические решения. Подпись Codex на каждом stage/claim не нужна;
нужен отдельный reviewer субагент и проверяемая evidence. Прямой запрос
пользователя разрешил также Codex и новые stages по
[решению0004](docs/decisions/0004-v2-research-refactor.md); harness и авторство
каждого review записываются честно. Общий resource budget сохраняется.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

Подробные команды, восстановление и архивирование:
[docs/automation.md](docs/automation.md). В Cursor доступны команды
`/research-next`, `/research-review` и локальные skills стадий.
Правила: [AGENTS.md](AGENTS.md), [единая авторизация R1–R7](docs/decisions/0003-cursor-autonomous-research.md).
Все 20 этапов R0–R18, включая R17D, операционно закрыты. `next` возвращает
`complete`; не запускать заново закрытые сетки по устаревшему handoff.

## Доказательства и воспроизведение

Каждый attempt хранит frozen plan, source/input snapshots, resolved argv,
метаданные среды, stdout/stderr, выходные хеши и привязанное к ним ревью.
Неудачные попытки сохраняются. Переносимые архивы проверяются без исходного
каталога. `uv.lock` фиксирует зависимости; полного битового равенства расчётов
на другом железе это не гарантирует.

Source checkout содержит исходники и control/review records. Большие массивы,
живые attempts и intake/QA-копии распространяются в проверенных Release assets.
CI выполняет отдельную проверку source checkout и unit tests, явно отмечая
недоступные evidence bindings как непроверенные. Полная команда ниже требует
восстановления evidence и не ослаблена для публикации. Git сохраняет точные
байты файлов, включая исторические CRLF, поскольку SHA256 привязан к байтам.

```powershell
uv run python scripts/check_repository.py --output checks.json
uv run python scripts/episteme_bridge.py status
```

EpistemeOS подключён из неизменённого `EpistemeOS-0.2.0/` (внутренняя версия
пакета — `0.1.0`). Новый bridge фиксирует вопрос, а не пререгистрацию
несуществующего эксперимента. Старый текстовый DomainPack не используется
для новой межмодельной программы.

Источники и границы их проверки: [docs/literature/](docs/literature/README.md).
Численные утверждения статьи требуют review отдельного Cursor субагента либо
реального human/Codex review и точных bindings в [research/claims.json](research/claims.json).

## История

[Оценка старых результатов](docs/legacy-assessment.md) объясняет, что можно
сохранить и с какими ограничениями. `result/`, исходные `.research/double-descent*`,
статья и старые протоколы сохранены. Частичный переносимый снимок исторических
логов/JSON: [research/evidence/legacy-2026-11-07/](research/evidence/legacy-2026-11-07/inventory.json).
Большие локальные матрицы в этом снимке не объявляются воспроизводимыми.

`objective.md`, `docs/plan/`, `.cursor/plans/` и протоколы 4/6 ноября являются
историей. `python main.py` по умолчанию больше не запускает старую сетку.
Намеренный replay требует `--legacy --stage ... --result-dir
artifacts/legacy-replays/<name>` и подготовленных входов в отдельном каталоге.
Оригинальный `result/` защищён входной командой от перезаписи.

Образцы: [EpistemeOS](https://github.com/AdamCage/EpistemeOS) и
[llm-semantic-afterlife](https://github.com/AdamCage/llm-semantic-afterlife).
Локальный `llm-semantic-afterlife` использован только для чтения.
