# R8 — отчёт

`scientific_validity=not_assessed`. Нулевой код возврата не является научной валидностью.

## План и attempt

Единственная попытка: `20261107T140907-bd7e4a5a5e`, статус `complete`.
Неудачных попыток нет. Задача `score-catalog`, exit code 0, timeout false.
Wall: `0.3899999996647239` с.

Plan SHA-256 `7304d89c4b57cd9d1ac8e4a8facd482ed54e32e694886d0ad16e62560a1f3780`.
Manifest SHA-256 `25291153cd7352fbe10653ce50f997af008cb38a61f1152ea44d97047a14aa8e`.
Каталог SHA-256 `1680b0d4eded65c3ac695388ba4988f0d3824cafd582828b25f92fb0c4d060ee`.
`scores.json` SHA-256 `36882c2612f0c2b58285b099d7b817e29d3a6013dd21610f3d986bafbf1331fc`.
`uv.lock` SHA-256 `ac89eef670e3774086fff7a06df62f1ba760f85d6c76378ac915ca32b76523b4`.
Авторизация SHA-256 `4f2eca57802835816b688668838545df60fd568c99069c291d2e42cb4f5371c9`.

Команда задачи: `scripts/r8_score.py --catalog research/stages/R8/inputs/catalog.json`.
Сырой файл: `artifacts/research/R8/20261107T140907-bd7e4a5a5e/tasks/score-catalog/scores.json`.
Stdout и stderr пустые. Снимок входов лежит в `artifacts/research/R8/20261107T140907-bd7e4a5a5e/inputs/`.

Git HEAD на старте: `a6f135c921a005dce1140f84506badea7e2f2d4e`. Рабочее дерево было
грязным: новые файлы R8/R9 и правки программы не входили в этот commit.
Связка доказательства — хеши входов попытки, не commit.

Среда: Windows 11, Python 3.12.11, AMD64. Модель не обучалась. GPU, платные
API и скачивания: 0. Seed нет: пять формул детерминированы. Окно локального
детектора 5, глобальная доля 0.2, порог RSS 0.05.

Design review до запуска: reviewer `7078a728-72d6-430c-b416-5aacf7cc7b3c`,
verdict accept. Автор плана и кода: `research-owner-r8r9`.

## Критерии и предсказания

Числа ниже сняты из `scores.json` этой попытки.

- C1. Пять рядов, у каждого есть `global` и `local` с `candidate`, `t_star`,
  `velocity_gain`, `rss_improvement`. `models_fit=0`, `download_bytes=0`,
  `http_requests=0`, `mechanism_claim=false`, `scientific_validity=not_assessed`.
  Операционно: pass.
- C2. `catalog_sha256` совпадает с байтами каталога. `two_drops` имеет
  `claim_role=excluded_multiple`. Исключения каталога скопированы. Лишних
  рядов нет. Операционно: pass.

- P1. `fast_plateau_moderate`: `global.candidate=false`. Исход: supported.
- P2. Тот же ряд: `local.candidate=true`, `local.t_star=70`,
  `velocity_gain=0.004999999999999993`. Исход: supported.
- P3. `smooth_exponential`: оба `candidate=false`. Глобальный `t_star=79`,
  локальный `t_star=94`, локальный gain отрицательный. Исход: supported.
- P4. `late_tail`: `global.t_star=79`, `local.candidate=true`,
  `local.t_star=90`. Глобальный `candidate` при этом true: правило сработало
  на последней допустимой границе, а не на индексе 90. Исход предсказания
  о позиции: supported.
- P5. `single_late_drop`: `global.t_star=50`. Исход: supported.
- P6. Тот же ряд: `global.candidate=true`. Исход: supported.
- P7. `two_drops` остаётся `excluded_multiple`, корневой `mechanism_claim`
  false. Исход: supported. Локальное правило всё же вернуло `t_star=45` и
  `candidate=true`. Это не интерпретируется.

## Результат и конкурирующие объяснения

На этом каталоге глобальное среднее скорости не помечает ряд «быстрый спад,
плато, умеренный спад»: лучшая допустимая граница 79 имеет
`reason=velocity_did_not_increase`. Локальное окно 5 ставит границу на
заранее известный индекс 70.

На ряде со спадом только после индекса 90 глобальное правило ставит 79.
Это граница допуска `min_fraction=0.2`, а не индекс начала спада. Локальное
окно ставит 90.

Гладкая экспонента не получает кандидата ни у одного правила. Контрольный
один спад оба правила ставят на индекс 50.

Ряд с двумя спадами исключён протоколом. Локальное правило всё равно
выбирает одну границу. Значит, положительный `candidate` не отделяет
однократный переход от нескольких. Для такого ряда протокол запрещает
механический вывод, и файл это сохраняет.

Это discovery на пяти формулах. Это не confirmation, не оценка неопределённости
по независимым запускам и не наблюдение CatBoost. Позднее ускорение на
данных не установлено и не опровергнуто: модель не обучалась.

Конкурирующее объяснение совпадения P2 и P4: окно 5 и узлы формул были
выбраны так, чтобы излом попадал на границу окна. Совпадение проверяет это
правило на этих формулах. Оно не переносится на другие окна и другие кривые.

## Отклонения и следующий шаг

Отклонений протокола нет. Окно после просмотра оценок не менялось.
Предложенный узкий claim о пяти рядах требует отдельного review и до него
не записан в `research/claims.json`.

Следующий этап очереди — R9, discovery CatBoost, после закрытия R8.
Сетка R9 не использует эти оценки.
