# R12 — постоянный шаг и вмешательство random_strength

Attempt `20261107T194939-7c18953a98`; plan SHA256
`8ae9d070bba865d347db6d56f767285054d1b7b170f2c534a5d234c82639b32b`, manifest
`fc4d0c3da64ec28f5c6ad73c5d5ab869f92c4daef0b23dbb6dbf0ed21e4b225a`.
Verify прошёл,83 files. Реализация и исходный scientific design:
`/root/catboost_study`; отчёт составляет `/root` по frozen outputs. Требуется
отдельное result review до admission нового claim и закрытия. Ниже авторский
анализ, а не внешнее научное одобрение. scientific_validity not_assessed.

## Полнота критериев

C1:72 клетки,2 datasets×12 новых seeds×3 random_strength; один continuous fit,
1024 дерева,1025 raw timepoints на каждом split. Сохранились все72 CBM,
predictions/data NPZ, cells.jsonl, analysis.json, models.json, environment.json.
Unit task явно включал opt-in tiny fitting smoke; никакой default CI fit.

C2: в task проверено совпадение final staged_predict с model.predict, fitted
bias с train prevalence; data/split/prediction hashes, leaf indices, фактические
и occupied leaf counts записаны. Независимое повторное сравнение ещё требуется.

C3: четыре paired scan-score contrasts, четыре fixed-boundary contrasts и
шесть conjoined gates сохранены. У каждого12 исходных значений, SE, обычный
Student-t95% и отдельные family4/family12 Bonferroni intervals. Ни один
неудобный seed не исключён. Все исходы ниже — правила frozen протокола.

C4: LR=.05 неизменен; нет init_model, test-driven выбора, GPU/API/download.
Inference и обучение явно ограничены4 потоками. calibration.json R10 не
передавался: descriptive_candidate и calibration_binding остаются null.
double_descent_declared=false, noise_annealing_mechanism_identified=false.

## Предсказания и контрасты

P1: rs10−rs0 повышает максимальный train slope-scan score в обоих окружениях
по отдельной family4. rs1−rs0 не имеет предсказанного знака и остаётся
exploratory. Нельзя переобозначать exploratory как confirmatory после открытия.

| Dataset | Контраст | Mean | Family4 adjusted CI | Outcome |
| --- | --- | --- | --- | --- |
| synthetic | rs1−rs0 | .00707807602 | [−.000508598680,.0146647507] | exploratory |
| synthetic | rs10−rs0 | .0171360169 | [.00759461815,.0266774156] | supported |
| breast_cancer | rs1−rs0 | .000138131971 | [−.00225905280,.00253531674] | exploratory |
| breast_cancer | rs10−rs0 | .00323859775 | [.00141283205,.00506436344] | supported |

Это изменение whole-trajectory diagnostic scalar, а не изолированного закона
annealing: параметр меняет весь путь деревьев, градиентов и общий range loss.
Синтетическая значимость R10 на эти кривые не переносится.

P2: все шесть fixed late-event gates дали `not_established`. Boundary512,
64 точки до/после, family12 для gain/drop, practical after-drop>.01 и
не менее10/12 positive gains были заданы до вычисления и не менялись.

| Dataset | rs | Mean fixed-window gain | Positive seed gains | Mean after-drop | Gate |
| --- | --- | --- | --- | --- | --- |
| synthetic | 0 | −3.61343711e-5 | 4/12 | .0112841015 | not_established |
| synthetic | 1 | −1.80624049e-5 | 4/12 | .0140426108 | not_established |
| synthetic | 10 | −4.04536068e-5 | 5/12 | .0150359500 | not_established |
| breast_cancer | 0 | −1.29133046e-6 | 4/12 | .00164390538 | not_established |
| breast_cancer | 1 | −1.72321946e-7 | 6/12 | .00274187355 | not_established |
| breast_cancer | 10 | −1.38782655e-5 | 2/12 | .00318317063 | not_established |

Нижняя family12 граница среднего gain не положительна ни в одной группе.
Даже когда объём уменьшения loss практически заметен, это не означает
увеличения СКОРОСТИ уменьшения. Непрохождение gate не исключает иной момент,
окно, конфигурацию или датасет. Искать другой момент в confirmation запрещено.
Четыре дополнительные fixed-boundary paired differences имеют adjusted
intervals, содержащие0; это отдельная диагностическая семья4.

P3: вся matrix сохранена; scientific claims double descent/annealing не
объявлены. Неудачных attempts R12 нет. Время runner205.10900000017136s;
сырые границы, числа ошибок и все отрицательные исходы остаются в evidence.

## Интерполяция и ограничения

Описательно,63/72 моделей достигли точного zero train classification errors
в пределах бюджета; first-zero находился между123 и988 деревьями. В этот
момент LogLoss мог быть .14022221881182134. Это измерения разных величин:
нулевая0–1 ошибка не означает нулевой surrogate loss. Эти дополнительные
числа требуют result review и не заменяют заранее заданные P1–P3.

Два малых окружения, одна библиотека и depth4. Breast Cancer — повторы
разбиений/моделей одного корпуса, а не независимые корпуса. t-интервалы
приближённые. Первичный максимум сканирует всю кривую и может относиться к
ранней динамике. Сильный вывод об естественном позднем событии не установлен;
механизм split-noise annealing не идентифицирован. Чистое совпадение signs,
закрытие operational stage или множество графиков этого не заменяют.
