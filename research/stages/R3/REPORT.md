# R3 — отчёт confirmation

Confirmation attempt: `20261107T102511-b1cf6c7be9`. Plan SHA-256 `2ca61ba925d577c2c822b9c31b1a15cca4b3585f09786dce3a9062f6f80e13bb`. Manifest SHA-256 `0e944a762326befb10cfd4e62ce49f2f35de119bed89716524761e7883888e02`. `verify` прошёл, `checked_files=12`. Elapsed 20.297 с. `scientific_validity=not_assessed`.

Discovery attempt `20261107T102058-3d5e221c96` сохранён, elapsed 16.485 с, live verify до смены плана: plan SHA-256 `8d5ca37d71f9d370b627f6c420caa19870179a237472f059e55241e8603fed2a`, manifest SHA-256 `4f532a4f27ec2f3a3d97164753412fba5241e2ada8a369036e718ed75fa4a0e7`. Он не является confirmation.

Design review `65e42b62-6bef-4fbf-976d-5fc030638b5c`: accept дизайна, revise пока PLAN описывал discovery, затем accept confirmation-контракта. Автор кода и отчёта: `ddb85535-1ef1-4069-95c4-46578067887a`.

Копии: `research/stages/R3/exports/analysis.json` SHA-256 `9b6d5e3064407b819089fc9b497818dcd7e78024ed1e9b626fb0b61cfadc785d`. `seed-audit.json` SHA-256 `2ec9544709efc1395a12444a7d94c4919a2c798722f068e961136265497f8ed3`. `bound-discovery.json` SHA-256 `c4cab46f81303f002bf33c439d1328e70b755c3d215d7e8d996c32c586369c9d`.

## Критерии

C1. `solver-checks/checks.json` имеет `tests_returncode` 0.

C2. `confirmation/cells.jsonl` содержит 145 ячеек. Seeds ровно 4101–4105. `discovery_seeds_absent` true. У `until_residual`, SNR 1, decay 0 ширина 4 не интерполирует (остаток около 0.46–0.53), ширина 256 интерполирует (остаток ≤ 0.001).

C3. `analysis.json` имеет `label=confirmation`. Шум: `supported`. Бюджет 40 эпох: `refuted`. Weight decay: `supported`. Отрицательный исход бюджета допустим и не мешает критерию.

C4. `bound-discovery.json`: `used_clean_risk` false, ширины 4, 8, 16, 32, 64, 128, 256. Export SHA-256 `5eb5ef3c71267a06d6822cff9eb75aacac459d4febf0d7307bac8539e30dd2f9`. Проекция 37.23 с была ниже 1800 с.

C5. `runtime.json`: device cpu, `cuda_initialized` false, `torch_threads` 4, переменные потоков равны 4.

## Предсказания

P1. Поддержано. Среднее по ширинам чистого риска SNR 1 минус SNR 5 на `until_residual`: mean 1.157, SE 0.081, min 0.917, max 1.389, n = 5.

P2. Опровергнуто. Разность чистого риска «40 эпох минус until_residual» на SNR 1: mean −0.927, SE 0.083, min −1.190, max −0.729. Короткий бюджет имеет меньший чистый риск, чем доведение до допуска 0.001. Предсказание, что короткий бюджет хуже, не выполнено. Это отрицательный результат, не повод менять допуск или сетку.

P3. Поддержано на срезе ширина 64, SNR 1, `until_residual`. Чистый риск decay 0 минус decay 0.001: mean 0.225, SE 0.026, min 0.152, max 0.316.

P4. Скобка остатка есть: ширина 4 выше допуска, ширина 256 не выше. Пик чистого риска не требовался.

## Пределы

Один скрытый слой, вход из 20 столбцов, n = 200, пять confirmation draw, CPU. Стандартные ошибки не доверительные интервалы. Допуск 0.001 — порог оптимизации, не линейный 1e-8. Опровержение P2 говорит, что на этой сетке доведение train-остатка до допуска ухудшает чистый риск относительно 40 эпох. Это не переносится на деревья и реальные таблицы и не требует эскалации 0003: предсказание было фальсифицируемым и фальсифицировано в заранее записанном правиле.
