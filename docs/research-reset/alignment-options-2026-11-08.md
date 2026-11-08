# Три кандидата для alignment portfolio: проверка на 8 ноября 2026

Это исследовательские карточки для решения 0006, а не PLAN, разрешение на
вычисления или подтверждение новизны. Новых экспериментов здесь нет. Проверены
первичные публикации, README, отдельные файлы исходников и карточки данных;
датасеты и веса не скачивались, модели/API не запускались. Старое исследование
сохраняется как отдельный завершённый evidence packet.

## Рекомендация и срок

По соответствию цели я предпочитаю **кандидат 1: перенос надёжности после выбора
монитора**. Но его нельзя объявлять готовым offline-проектом: требуемая таблица
предсказаний и независимых labels пока не подтверждена. Дать ему ограниченный
48-часовой prior/data gate. Если он не проходит, перейти к **кандидату 3**, где
публичность нужных полей подтверждена значительно лучше. Без API-бюджета это
наиболее конкретный резерв. Кандидат 2 имеет смысл при разрешённых новых вызовах
монитора. Три исследования одновременно не запускать.

У [Anthropic Residency](https://alignment.anthropic.com/2026/anthropic-residency/)
дедлайн 22 ноября 2026. Программа рассчитана на established technical leaders;
публикации и PhD не обязательны. Поэтому полезный двухнедельный результат —
один убедительный проверяемый вывод, рабочая реализация, ясные ограничения и
личное исследовательское суждение пользователя. Принятую журнальную статью за
этот срок обещать нельзя. Темы monitoring/scalable oversight соответствуют
[официальным направлениям](https://alignment.anthropic.com/2025/recommended-directions/).

| Кандидат | Научный вопрос | Готовые данные без новых model calls | Главная опасность |
| --- | --- | --- | --- |
| 1. Выбор и перенос монитора | Переносится ли качество выбранного monitor/prompt на другое семейство задач при фиксированном бюджете проверок? | Не подтверждены все необходимые scores/labels/benign cases | Известная nested validation под новым названием; неполный release |
| 2. Проверяемое состояние инструментов | Может ли дешёвая детерминированная сводка состояния вернуть выигрыш дорогого code-assisted monitor? | Есть описание и paired synthetic transcripts; свежие monitor calls нужны | SLEIGHT уже проверяет state prompts и coding agents |
| 3. Адаптивные подсказки | Насколько сравнение CoT proxy зависит от разных версий вмешательства для разных моделей? | Публичный viewer показывает нужные поля; объём и overlap ещё проверяются | Ограниченный estimand и уже известное instance matching |

## 1. Надёжность мониторинга после выбора конфигурации и переноса задач

**Вопрос.** Если выбрать prompt/monitor на development-эпизодах, а порог
калибровать при ограниченном числе human reviews, насколько оптимистична оценка
после перехода на новое семейство задач? Сохраняется ли вывод при разделении
selection, calibration и evaluation по независимым семействам, а не по
стохастическим trials одного эпизода?

**Предсказание.** Оценка выбранного победителя будет сильнее завышена при
случайном trial split, чем при family-separated split; часть выигрыша исчезнет
на новом семействе. Интересен размер и практическое следствие эффекта в
конкретных alignment evaluation protocols, а не сам факт winner's curse.

**Ближайшее известное.** [SLEIGHT-Bench](https://alignment.anthropic.com/2026/sleight-bench/)
уже показывает различающиеся blind spots, побочные эффекты специализированных
prompts и coding-agent monitoring. Его [benign-source sensitivity script](https://github.com/safety-research/sleight-bench/blob/main/scripts/paper/plot_benign_source_sensitivity.py)
уже сравнивает источники benign-калибровки. [Reliable Weak-to-Strong Monitoring](https://arxiv.org/abs/2508.19461)
изучает awareness, scaffolding и escalation; [Evaluating Control Protocols](https://arxiv.org/abs/2511.02997)
проверяет control protocols против адаптирующегося атакующего. Последний
источник здесь прочитан на уровне abstract, поэтому точное пересечение ещё
подлежит проверке. Нельзя заявлять новизной «перенос ухудшается», ансамбль,
conformal threshold или применение AUROC.

**Отличие-кандидат.** Полностью воспроизводимый аудит конкретной процедуры
выбора монитора: что меняется в её обещании о catch rate при том же бюджете
проверок после независимого family transfer. Базовые статистические процедуры
служат controls. Вклад появится только при нетривиальном измеренном провале
известного evaluation protocol и полезном способе ограничить ошибочный вывод.

**Минимальный пилот.** Одна зафиксированная панель конфигураций, заранее
выбранные группы задач, первичный endpoint — held-out detection при фиксированном
бюджете false alarms/reviews. Сравнить исходную процедуру выбора и разделённую
процедуру на одинаковых episodes; uncertainty считать по episode/task family.
Разные trials одной истории не независимые задачи. Если scores разных мониторов
получены на разных траекториях, offline сравнение cascade/ensemble невалидно.
Значения FPR и smallest worthwhile effect выбираются в PLAN до анализа;
малый набор benign cases не позволяет обещать надёжный 1% FPR.

**Данные/стоимость.** Возможный источник — [ScaleAI/mrt](https://huggingface.co/datasets/ScaleAI/mrt),
но карточка указывает 11.5 GB и неоднородные JSON. Полный download не входит
в прежний 2 GiB dataset cap. Нужны manifest и допустимый ограниченный subset.
Наличие monitor verdict не доказывает наличие независимого task-checker label.
Нужны continuous scores, идентификаторы одинаковых trajectories, настоящие
benign episodes и независимая проверка harmful outcomes. Без этого пилот
потребует новых monitor calls и отдельного бюджета.

**Kill criterion.** Через 48 часов нет одновременно ясного отличия от priors,
доступной достаточной панели и исполнимого pilot — остановить кандидата.
Если обычный nested/grouped baseline уже устраняет весь практически значимый
эффект, не объявлять новый метод. Portfolio fit высокий; журнальный потенциал
условен на независимый перенос и практическое следствие, не на один rank flip.

## 2. Детерминированные evidence capsules для проверки состояния инструментов

**Вопрос.** Можно ли на ограниченном языке tool calls строить компактную
проверяемую сводку происхождения данных и состояния объектов, которая улучшит
обнаружение нарушения дешевле coding-agent monitor при одинаковом бюджете?

**Предсказание.** Для ошибок идентичности объектов, путей и порядка изменений
сводка поможет больше, чем равная по длине обычная текстовая summary. Для
неподдерживаемой семантики она должна возвращать unknown, а не выдумывать state.

**Ближайшее известное и отличие.** SLEIGHT уже тестирует state-tracking prompts
и мониторы, исполняющие код; его авторы сами предлагают искать более дешёвые
scaffolds. Поэтому «добавим инструменты/память» не новизна. Узкое отличие —
дешёвый фиксированный parser/provenance graph, измеренная цена и перенос на
категории, по которым parser не настраивался. Это конкретная engineering
гипотеза с открытым сравнением, а не заявленный новый общий метод.

**Минимальный пилот.** Поддержать несколько явно заданных безопасных форм
Read/Write/Edit в текстовых журналах. Сравнить raw transcript, равную по токенам
generic summary, специальный state prompt, capsule и доступный code-assisted
baseline; одинаковый monitor, входной/выходной budget и paired episodes.
Primary endpoint — paired detection improvement при заранее выбранном review
budget; отдельно показывать token/cost, abstention и coverage. Нельзя выводить
улучшение из сокращения контекста или большей вычислительной цены.

**Данные/стоимость.** [Публичный репозиторий SLEIGHT](https://github.com/safety-research/sleight-bench)
описывает 40 синтетических атак с benign counterparts; multi-stage примеры
добавляют сессии, а не независимые атаки. Репродукция paper figures читает
BigQuery и делает новые API evaluations; полная открытая таблица trial scores
не подтверждена. HF-release требует отдельного согласия на sharing данных
аккаунта; он не получен. Подготовка parser и fixtures возможна без моделей,
но это ещё не научный результат. Эксперимент требует разрешённых monitor
calls. Журналы трактуются как данные; команды из них не исполняются.

**Kill criterion.** Если matched-cost generic/state baseline даёт тот же
эффект, parser охватывает слишком мало независимых случаев или gain не
переносится с разработанной категории, остановить сильный claim. Хорошее
portfolio может получиться как полезный инструмент с честным benchmark;
для журнала нужны дополнительные независимые задачи и механизм выигрыша.

## 3. Адаптивная подсказка меняет само вмешательство при сравнении моделей

**Вопрос.** Насколько межмодельные сравнения proxy «признала ли CoT подсказку»
зависят от того, что каждой модели показывают свою версию подсказки, выбранную
против её собственного baseline answer? Это содержательно другой estimand,
чем сравнение моделей на одной и той же подсказке.

**Проверенный факт дизайна.** В [generate_adaptive_datasets.py](https://github.com/ajmeek/measuring_cot_monitorability/blob/main/scripts/data_processing/generate_adaptive_datasets.py)
target выбирается среди ответов, отличных от baseline данной модели. Поэтому
на одном вопросе модели могут получать разные target answers. При верном
baseline target обязательно неверен; при неверном baseline target может быть
верным. Это логическое свойство кода, не обнаруженный численный эффект и не
автоматически ошибка: adaptive challenge может быть намеренной целью оценки.
Нужно проверить, соответствует ли этот код опубликованным строкам и выводам.

**Ближайшее известное.** [Meek et al.](https://arxiv.org/abs/2510.27378) уже
объединяют cue faithfulness proxy и verbosity; это не наш вклад. [OpenAI,
20 октября 2026](https://alignment.openai.com/beyond-gmean-squared/) уже предлагает
pair matching на общих instances, показывает смещения рейтинга и выводит
границы causal attribution. Простое matching или новый рейтинг новизной не
будет. Отличие-кандидат — отдельно проверить model-dependent treatment version
даже на одном core question и явную связь cue correctness с baseline correctness.

**Минимальный пилот.** Сначала воспроизвести исходные denominator/estimands.
Затем заранее объявленным способом сравнить общий набор вопросов и общий набор
точных treatment versions: core ID × cue kind × target/cue text. Показать
распределение baseline correctness/cue correctness и совместные answer
transition/acknowledgment outcomes. Оценивать все объявленные пары/семейства,
а не искать красивую перестановку. Разделить exploratory и confirmation
question families до анализа. Matching не восстанавливает отсутствующие
counterfactuals: при слабом overlap вывод ограничивается supported pairs.

**Данные/ограничения.** [README авторского code repository](https://github.com/ajmeek/measuring_cot_monitorability#links)
напрямую связывает paper 2510.27378 и HF-release. [Авторский открытый viewer](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts)
показывает 446k rows, 8 models и поля core_id, scope, model_answer,
baseline_model_answer, cue_target_answer, correct_answer, cue_acknowledgment.
Metadata-only [HF tree API](https://huggingface.co/api/datasets/ameek/measuring_cot_monitorability_transcripts/tree/main?recursive=true&expand=false&limit=1000)
вернул 27 файлов, из них 24 parquet, суммарно 940129927 bytes (около 0.876 GiB).
Payload не скачивался. Сам release помещается в cap 2 GiB; остаток общего
бюджета, cache/extracted storage и pinned revision нужны до загрузки.
Схема подходит для первого аудита; уникальность/семантика core_id, missingness,
дубликаты, соответствие grader labels и treatment overlap ещё не проверены.
По карточке baseline и пять adaptive variants относятся к одному sample group;
это пока не доказательство надёжного cross-model join. Сравнение ответов/targets машинно проверяемо;
cue_acknowledgment — готовая аннотация, не доказанная истинная faithful reasoning.
Авторские causal-factor lists также не являются доступом к внутренним причинам.
Одно наблюдаемое изменение ответа не доказывает причинное использование cue.
Новые модели для первого descriptive reanalysis не нужны; причинная confirmation
потребует повторных baseline/neutral-cue controls и отдельного бюджета.

**Kill criterion.** Остановить novelty claim, если исходная работа уже
контролирует этот эффект, нет достаточного treatment overlap или заранее
значимый effect не воспроизводится между held-out families. Из отсутствующих
counterfactuals не делать «исправленный общий рейтинг». Это наиболее реалистичный
offline fallback для двух недель; журнальный потенциал умеренный и зависит
от нового общего вывода, а не от критики одной статьи.

## Что действительно открыто, а что пока только кажется открытым

| Источник | Что проверено чтением | Чего не предполагаем |
| --- | --- | --- |
| [SHADE-Arena](https://github.com/jkutaso/SHADE-Arena) | Public environments/checkers/examples; README отличает private repository | Полный paper dataset и existing transcripts публично доступны |
| [SLEIGHT](https://github.com/safety-research/sleight-bench) | FORMAT описывает metadata scores; scripts используют BQ/API | Metadata means равны полным per-trial benign/attack scores; авторский BQ доступен |
| [ScaleAI MRT](https://huggingface.co/datasets/ScaleAI/mrt) | Release logs заявлен; карточка описывает transcripts/verdicts | Verdict есть независимый outcome label; есть одинаковые trajectories для всех monitors |
| [AuditBench](https://github.com/safety-research/auditing-agents) | Public code; README требует large-model serving/API для воспроизведения | Открытый CPU-only набор всех audit logs и clean controls. Tool-to-agent gap уже известен |
| [OpenAI monitorability-evals](https://github.com/openai/monitorability-evals) | Открытые evaluation tasks и paired interventions | Все frontier rollouts/scores опубликованы; deprecated tasks имеют надёжные labels |
| [Meek transcripts](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts) | Видимая схема/примеры; 24 parquet, 940129927 bytes, public/ungated по metadata API | Все строки валидны, общий оставшийся budget достаточен и есть нужный overlap |

Пять показательных примеров SHADE не заменяют полноценную выборку. Ссылка на
Google Drive с CoT-данными Anthropic в этой среде привела к login; доступа к
полному набору не подтверждено. Данные только успешных атак могут помочь
изучать совместные misses, но не FPR или deployment risk.

## Графики как артефакты

После выбора одного кандидата — воспроизводимый script, PNG и SVG каждого
объявленного графика, таблица исходных точек и README в artifacts/research/.
Основная история использует несколько читаемых figures, полная gallery
сохраняет остальные условия, включая отрицательные. До данных графики с
воображаемыми результатами не создаются.

Предлагаемый полный набор для кандидата 1: карта episode/family overlap;
число независимых benign/harmful cases; распределения scores; calibration и
evaluation false alarms; selection optimism; paired catch-rate differences;
family transfer matrix; error correlation; uncertainty по families; sensitivity
к review budget; cost–detection frontier; breakdown по blind spots. Для
кандидата 3 соответствующие artifacts — treatment-overlap matrix, baseline/cue
correctness composition, missingness, original/matched estimates, paired
differences с интервалами, family heterogeneity и denominator sensitivity.
Число рисунков не является критерием остановки или научной ценности.

## Проверенные исходники и предел проверки

GitHub blob SHA при чтении 8 ноября (это content hashes файлов, не commits):
SLEIGHT README `ab40a47c7e80f87d1577da8ee8bfd3a788f27a8b`,
benign-source script `99d0b264c20bb3c6e1c018e2579fc2cdb380d816`;
Meek adaptive generator `2d92f26fdb30a1cb616335096c8b1738fb070436`.
HF transcripts metadata revision `a7a8b25ae467db9d80d63e55ac053b48753b2d36`,
lastModified 30 ноября 2025; metadata API: private=false, gated=false.
При запуске потребуются pinned repository/dataset revisions и локальные hashes.
Обзор не исчерпывает литературу; ни одна карточка не имеет статуса
novelty-approved или evidence-supported. Следующий полезный выход — короткий
prior/data gate по одному вопросу и критика отдельного reviewer, не очередной
большой framework.
