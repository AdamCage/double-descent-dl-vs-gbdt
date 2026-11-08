# Сегодня: воспроизводимый аудит сравнимости CoT acknowledgment

Пользователь уточнил срок: **сегодня завершить проект**. Двухнедельный roadmap
в PORTFOLIO-PROGRAM.md — ранняя стратегия; эта постановка имеет приоритет.
Выбран CPU-only кандидат3 из alignment-options, поскольку авторский public
release существует и помещается в имеющийся dataset/storage cap. PaidAPI/GPU0.

Выход дня: проверенные данные+код+все исходы, отдельные discovery/confirmation
по непересекающимся question clusters, статические scientific figures и gallery,
краткий английский report/portfolio artefact с реальными ограничениями.
Accepted journal paper, подтверждённая мировая новизна и admission в Residency
не являются обещанными результатами одного дня. Вклад человека/агентов
фиксируется честно; научный report ещё потребуется лично разобрать пользователю.

Вопрос: **насколько межмодельная разность частоты released judge acknowledgment
меняется при разных явно названных comparison populations и какая часть
публичного release поддерживает exact-input comparison?**
Исходная adaptive policy Meek намеренная, не ошибка. Matching/stratification
и maxT bootstrap известны, не заявляются новой метрикой/теорией. Работа не
оценивает latent causal faithfulness, hidden thoughts или современные модели2026.
Она аудирует исторические cached outputs2025 и интерпретацию их сравнения.

В каждой model pair/dataset сравниваются три популяции:

1. Общие question×cue-contexts, каждое со всеми наблюдаемыми версиями для модели.
2. Только contexts с exact version overlap, но всё ещё все native versions.
3. Те же overlap contexts, только одинаковые released input versions.

Разность3−1 раскладывается на composition2−1 и version restriction3−2.
При одной версии наcontext вторая часть может быть тождественно0: это будет
показано, а не переименовано в эффект. Ни одна часть не causal correction
несуществующих counterfactuals. Неизвестные/неподдержанные cases не imputed0.

R15 — pinned24-file intake и полный slim/provenance без outcome statistics.
R16 — frozen discovery (30% clusters), go только по integrity/analyzability,
без отбора положительного эффекта. R17 — locked confirmation(35%) с теми же
estimands, всеми28pairs×3datasets и отдельным bootstrapseed. Это новые вопросы
в том же benchmark, не независимые новые model runs/OODfamily transfer.
R18 — полный numerical/visual review, English report и воспроизводимая поставка.

После source audit и до outcomes фиксируются5pp practical shift,2pp tie band,
минимум independent support clusters и25%overlap, simultaneous cluster maxT
интервалы. Нулевой/неопределённый исход допустим. Нет изменения thresholds,
выбора красивых model pairs или придуманной journal readiness ради дедлайна.

Уточнение до outcomes: пользователь дал четыре часа, до13:12UTC8ноября.
35%question clusters оставлены reserve для одной отдельно preregistered новой
гипотезы, если потребуется; все попытки раскрываются. Текущий тест reserve
ACK outcomes не читает. Split изменён до первичного анализа с отдельным review.
