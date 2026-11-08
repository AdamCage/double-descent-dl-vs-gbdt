# R14 — early-probe forecast не прошёл заранее заданный порог полезности

Автор отчёта `/root`. Реализация: `/root/catboost_study` и
`/root/r14_implementation`; renderer: `/root/literature` и `/root/r14_figures`.
Независимый reviewer `/root/r14_reviewer` не автор этих файлов или вывода.
Статус numerical attempt complete; научная интерпретация/operational verdict
подлежат отдельному bound review. Исходная полная научная цель остаётся открытой.

Attempt `20261108T073937-26f816e66d`, elapsed521.766s, все3tasks exit0.
Manifest SHA256 `29d2cf114c9886b61f10ad5b7f5955ac4b63f157edc6fdc46e2e012919347f05`;
plan `2cdb3c1411ea7155a3a89c0abc1bfe8b4f58e85b9c8ae3a552347cdc336f41df`.
Live verify прошёл1119files. Предварительно отдельно проверены native
predictions/leaves всех416savedCBM и все raw/OOF equations; окончательные
assessment и scope фиксирует reviewer в REVIEW, не этот авторский отчёт.

## Что именно проверено

32contexts =2fixed synthetic teachers ×4nested candidate prefixes ×4data
draws14001–14004.128train/256evaluation; nominalε=.01/.1.128cheap32-tree
perturbed fits и256full512-tree perturbed fits плюс32basefits =416fit,
384direction records и64context/amplitude endpoints. Независимая единица —
data draw, **не64rows или416моделей**. Направления парные и повторяются между
teachers/prefixes/amplitudes; pilot/response streams раздельны.

D — средний по4новым направлениям ||Δfull−A512δactual||²/(256ε²).
C — такой же32-tree diagnostic по2другим направлениям. Target forecast
u=log10(1+D); ridgealpha1, стандартизация только training folds, все observations
одного draw исключаются вместе. Ни модели прогнозов, ни endpoints/grids/seeds,
ни gain/cost thresholds после данных не менялись. Это discovery LODO, не
independent confirmation или результат об истинном teacher-risk double descent.

## Предсказания и исходы

**P1 known-control supported.** Все128baseline split/checkpoint checks прошли
при неизменномatol1e-10/rtol0; max absolute error7.993605777301127e-15.
Все32operators/actual configurations/weights сохранены. Invalid zeroδ=0.
Максимальная absolute squared-norm closure error5.684341886080802e-14;
signed cross сохранён, не объявлен nonnegative extra variance.

**P2 prespecified≥20% forecast advantage refuted.** Aggregate MAE(u), с равным
весом contexts внутри draw и затем draws:

| Model | Содержание | MAE(u) | Уменьшение MAE у A относительно baseline |
| --- | --- | ---: | ---: |
| B0 | training-fold mean |0.517545614568|53.1329%|
| B1 | metadata ε/prefix/teacher |0.256022876744|5.2590%|
| B2 | B1+frozen geometry/EFC/quality/floor |0.254704551297|4.7686%|
| B3 | B1+cheap diagnostic |0.250140933179|3.0312%|
| B4 | cheap diagnostic+slope1 calibrated offset |0.366651554364|33.8449%|
| A | B2+cheap diagnostic |0.242558701584|—|

A достигает требуемого выигрыша противB0/B4, но неB1/B2/B3. Этот факт не
переименовывается в успех на основании выигрыша над слабым constant baseline.
Остальные conjunctive gates проходят: A не хужеB2/B3/B4 соответственно
в3/3/4из4draw folds и отдельно в обоих teachers; q90−q10(u)=1.5720428594518647
превышает.1. Прохождение этих частей не отменяет failure основного20%gate.
Нет p-value/CI из64correlated rows и нет заявления об отсутствии любого
возможного полезного forecast. Source `decision.go=false`,
`outcome=stop_this_forecast_candidate` соблюдаются.

**P3: положителен только узкий computed probe gate; стоимость всего forecast
при исходном условии «есть basefit/operator» не установлена.** Сохранённый
analysis.json показываетP3_cost=true для3.654164098203182s cheap
fit+nativepredict+decomposition против24.260860392358154s full response,
ratio0.15061972407846652<.5. Это верно для дополнительной пробы поверх уже
вычисленныхB2diagnostics, а не для построения всего forecaster с нуля.

Независимый cost audit выявил материальную границу измерения: общая подготовка
22.53748700255528s включает frozen operators, EFC/geometry features и все512
base train prediction stages. В base_prediction_seconds=14.976943s обычное
prediction и требуемая дляEFC trajectory смешаны; точную дополнительную
стоимостьEFC при наличии толькоbasefit/operator из этих timers не выделить.
Source feature timer=.234944s; accounting bounds с features/forecast дают
примерно.1605–.7778 отfull response cost, пересекают.5. Поэтому исходный более
широкий P3 не объявляется supported. Cold forecast/full-refits source ratio
1.079786074243844; secondary ratio при общей подготовке в обоих denominators
0.5597748779413033. Это разные scopes, не альтернативные пороги успеха.
IO276.8415187s и leaf-evidence extraction записаны отдельно. Все timings
относятся к этой среде; ни общий speedup, ни перенос на deployment не доказаны.
Immutable source analysis с computedP3=true не переписан; assessment уточнён
в этом отчёте и отдельном review. P2false уже даётgo=false при любом из этих
cost scopes. Повторных fits ради спасения candidate не запланировано.

## Полнота evidence и графиков

Все inputs/nominal/native labels/deltas/directions/thresholds,32context NPZ,
416fit NPZ/CBM, leaf indices, raw predictions, operators, feature snapshots,
OOF coefficients/scalers/predictions, timings/budget/progress/environment
сохранены. Gallery113PNG/SVG pairs и7CSV:64forecast rows,32contexts,
384directions,147456raw response rows,24fold fits,2816timing components и
11cost scopes. Все seeds и направления показаны, включая отрицательный gate.
Root просмотрел4primary panels; окончательный широкий visualQA — в отдельном
review. Графики отображают сохранённый source decision, его cost interpretation
нужно читать с оговоркой P3 выше; renderer не проводит нового statistical test.

Author secondary analysis (AUTHOR-ANALYSIS.md) сохранил все16groups и signed
cross: среднийD растёт поprefix4→60 в каждом teacher/ε при улучшении baseline
clean-risk;31rows имеютC>D,33 —C<D. Эти описательные совпадения не означают
причинного механизма, double descent, нового scaling law или hypothesis
confirmation.73из256full evaluation cross terms отрицательны; squared
discrepancy нельзя складывать с frozen response как неотрицательную variance.
Отдельная known norm identity не является новой теоремой.

Reload metadata caveat: CatBoost1.2.10 округляет learning_rate и
bayesian_matrix_reg в get_all_params загруженногоCBM (.10000000149011612→
.1000000015). Reviewer использует отдельное metadata-display tolerance,
не ослабляя study/native prediction gates. Predictions всех416savedmodels
совпали точно; для operator reconstruction берётся сохранённая actual eta и
partitions, а не заново округлённое display значение. Helper fresh-model
guard не заявляется универсальным roundtripAPI для произвольныхCBM.

## Критерии выхода и продолжение

C1/C2 имеют полную evidence фиксированного matrix/replay; C3 содержит полный
negative accuracy result и явно неустановленную broaderP3 cost hypothesis;
C4 имеет полную заявленную delivery matrix. Оценку pass/partial назначает
отдельный reviewer по фактическим данным/визуальной проверке. Complete exit0
сам по себе не является scientific acceptance.

Candidate A остановлен. R15confirmation этого forecast по прежнему плану
**не разрешена**, поскольку conjunctive go gate провален. Любая новая
содержательная гипотеза — отдельное записанное решение, frozen design и новые
discovery/confirmation данные; никаких изменений ridge/features/амплитуд ради
успеха на этих64rows. Существующие результаты сохраняются как bounded negative
benchmark. Priority, новое объяснение double descent и full cross-model study
не установлены. Послеreview:close, exclusive archive+detached verify, STATE.
