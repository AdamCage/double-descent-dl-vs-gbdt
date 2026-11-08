# R5 — отчёт confirmation

Confirmation attempt: `20261107T105955-6e7f800ff1`. Plan SHA-256 `2be0f54a502a68a50491728c45a2d4714ea89c7036490d289fcefc2cb3e29f8e`. Manifest SHA-256 `324bb798223c02dbef8e5b171d05ca0aa56d4ef7fc6aca0480e1d90fcde2ed8b`. `verify` прошёл, `checked_files=12`. Elapsed 4.828 с. `scientific_validity=not_assessed`.

Discovery attempt `20261107T105818-f129d71a55` сохранён, elapsed 4.687 с, manifest SHA-256 `9174946128d6bcae51ecd82cd8d6d0dd054800c0290f59ba9cbc90556349bd20`. Два discovery draw дали `inconclusive_few_brackets` и меню не меняли. Design review `d07fa088-13ac-4dfa-bad6-eb4fad7ed629`. Автор: `ddb85535-1ef1-4069-95c4-46578067887a`.

Копии: `analysis.json` SHA-256 `56e48cf85af3f1f0aaeccf729e506c80b9bae7b0909ae3048c506ffa3d6d03cc`, `seed-audit.json` SHA-256 `ccff4a6517cb0d981c9690d65efc1c64c68ff294f477f9ce313fc12b6105ceb6`, `bound-discovery.json` SHA-256 `5b8113d801f3d018ce1c74b9bee19bee14d0e0bb7809bec04ac67a2b588313f1`.

## Критерии

C1. `tests_returncode` 0.

C2. 50 ячеек, seeds ровно 8101–8105, `discovery_seeds_absent` true. Есть линейное окно, листья 2 и 256, лес 1 и 50.

C3. Шум, остаток листьев, чистый риск листьев и чистый риск леса посчитаны. Все четыре исхода `supported` при n = 5. Отрицательный исход был допустим.

C4. `kept_full_menu` true, `used_clean_risk` false, export SHA-256 `41d101f54ed74d61ec71293addc4b26558e583cac6e651457b96ca52ef3ebf01`. В `analysis.json` записаны non-identifiability и три непереоценённых альтернативы, включая 40 эпох MLP и кривую глубины бустинга.

## Предсказания

P1. Поддержано. Оконный чистый риск SNR 1 минус SNR 5: mean 25.924, SE 2.545, min 17.081, max 32.916, n = 5. Все приращения положительны. Это не доверительный интервал и не снимает тяжёлый хвост, виденный в R2 на других seeds.

P2. Поддержано. Остаток 2 листьев минус 256: mean 0.926, SE 0.004. Чистый риск 256 минус 2: mean 0.988, SE 0.118, min 0.659.

P3. Поддержано. Чистый риск одного ограниченного дерева минус 50 деревьев: mean 0.797, SE 0.028, min 0.707.

P4. Записано, не снято репликацией. Совпадение направлений не идентифицирует общий механизм. Результат R3 про 40 эпох и кривая глубины бустинга не переоценивались. L05–L07 и CatBoost остаются неразрешёнными.

## Пределы

Пять новых draw, фиксированные концы сеток, без CatBoost и без MLP. Репликация не выбирает единое объяснение для линейного шума, листьев и леса. Эскалации 0003 нет: исходы совпали с заранее записанными направлениями, а неидентифицируемость сохранена.
