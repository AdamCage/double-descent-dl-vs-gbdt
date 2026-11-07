# R1 — отчёт операционного аудита

Attempt: `20261107T085630-b4345b8c44`. Plan SHA-256: `88d7d77d88902668c22ea656c71affca3f7169f309bd09d24851f6ec38af47a5`. Manifest SHA-256: `6a04f9da18d475212ed6360cb7a14ad738c55adb1e5d2d4ab3049ea2db49fdff`. `verify` прошёл, `checked_files=26`. `scientific_validity=not_assessed`. Модели не обучались.

Автор заметок: `ddb85535-1ef1-4069-95c4-46578067887a`. Чтение L02–L04: `144f9191-3044-4fe3-87d1-0f162ee1fb60`. Чтение L05–L08: `001810aa-bf22-41e9-92fd-4cdc05feb1c1`. Design review: `03f147be-80d7-42b8-9499-5473f6a5ffc9`, verdict accept, blocking issues пусты. Это локальное team review, не внешняя экспертиза.

## Критерии

C1. В `legacy-evidence/legacy-audit.json` 28 наблюдений. Поддерживаются округлённые строки s05, рост validation loss на width 512 без завершённого правого плеча, пустой `result/05_dnn`, конфликт dim 768 и 4096, слабый cache key и расхождение hash partial/cache. Десять крупных матриц и parquet имеют disposition `not_assessed`, sha256 null и статус inventory `local_only_not_hashed`. Противоречащего full-precision результата в этих файлах нет.

C2. L02–L08 — full text: PDF, SHA-256, текстовый слой PyMuPDF 1.28.0 и цитаты, которые входят в слой. L01 недоступен: две попытки PNAS PDF вернули HTTP 403, препринт не подставлен. Это записано в `literature/cards/L01.md` и в `unavailable_sources` файла `audit-validate/validation.json`.

C3. `tree_axis_comparison.resolved` равен false. L05 и L06 меняют leaf/boosting capacity и ensemble size при squared loss. L07 меняет число split candidates и в заключении держит ensemble size фиксированным, метрика RMSE. Согласование не выдумано.

C4. `r2-design/R2-design-proposal.json` задаёт teacher, вложенные признаки, нормировку, раздельные seeds, меню p/n, SNR, SVD/rcond/rank/residual, estimand, общие evaluation points, четыре вмешательства с фальсификаторами и readiness `ready_for_autonomous_planning`. `models_executed` false. Числовой residual tolerance и размер confirmation grid остаются открытыми и не блокируют этот описательный этап.

C5. Подготовка запечатана. Загрузки: 16 420 945 байт, 9 HTTP-запросов, datasets 0. `repository-check/checks.json` имеет `passed` true (91 tests, 1 skip). В момент `audit-validate` текущий attempt ещё не имел `elapsed_seconds`, поэтому учёт поставил консервативные 28 800 с и `computational_activation_ready=false`, не подставив ноль. После завершения manifest даёт 11,828 с; вместе с R0 13,094 с пост-проверочный пересчёт даёт 24,922 с, пустой список unmeasured и ready true. Этот пересчёт не переписывает frozen `validation.json`.

C6. Автор и design reviewer различны. Отрицательные и not_assessed пункты сохранены. Операционное review этого отчёта выполняет другой субагент.

## Предсказания

P1. Согласуется с ожиданием: малые артефакты не устанавливают double descent на старом корпусе. Противоречащий полный результат не найден и не скрыт.

P2. Согласуется с ожиданием: сравнение осей нужно, и сопоставимость L05–L07 остаётся неразрешённой.

P3. Согласуется с ожиданием: ограниченный R2 различает шум, спектр и ridge у линейной модели и отдельно описывает frozen random features. Запуск не делался. Неидентифицируемость при отсутствии скобок интерполяции записана.

## Бюджет и пределы

Runner attempt 11,828 с из 900 с. Это первый R1 attempt. Новые файлы аудита около 41 МиБ при потолке R1 2 ГиБ. GPU, paid API и датасеты: 0.

Пределы: L01 не прочитан; формулы PDF не восстанавливались; крупные матрицы не хешировались; design review и этот отчёт не оценивают научную истинность.
