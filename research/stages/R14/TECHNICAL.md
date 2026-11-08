# R14 — технический fixed-partition контроль, не научный протокол

Автор: `/root/catboost_study`. Дата: 8 ноября 2026, Europe/Moscow.
Этот файл не PLAN, не авторизация вычислений и не claim о новизне. Реализация
подготавливает известные алгебраические контроли для отдельного будущего
falsification pilot. Научные данные и study seeds здесь не генерируются.

Прочитан [scout](../../../docs/literature/next-question-scout-2026-11-07.md).
Ближайшая литература уже содержит fixed-structure refitting/influence,
data-dependent smoother/telescoping constructions, split-candidate оси и
search degrees of freedom. Наблюдение «деревья изменились» и сам linear replay
новыми результатами не являются. Потенциальный будущий вопрос — прогноз ошибки
такого surrogate на новых конечных perturbations и новых draws.

## API и границы реализации

`nested_binary_features(X_train, X_evaluation, raw_feature_count=4,
threshold_levels=(1,3,7,15))` возвращает readonly максимальные binary matrices,
frozen threshold tuple и prefixes `(4,12,28,60)`. Quantiles вычисляются только
по train с `numpy.quantile(method="linear")`. Порядок вероятностей:
1/2, затем1/4,3/4, затем1/8,3/8,5/8,7/8 и далее; внутри каждой вероятности
признаки идут round-robin0…3. Сравнение — `x_j > threshold`. Все четыре raw
features входят в первый prefix; teacher будущего протокола должен оставаться
одним и тем же и использовать те же raw features, а не активироваться вместе
с ростом candidate budget. Равные quantiles сохраняются, не заменяются.
Номинальные candidate counts не гарантируют уникальность signatures.

`canonicalize_targets(y)` явно возвращает nominal float64, native float32,
actual native labels promoted to float64 и rounding delta. `fit_control` и
`frozen_operators` требуют **фактические native-representable метки**, а не
молча ослабляют tolerance. Для finite perturbation следует сохранить nominal
`epsilon*z`, base/perturbed native y и actual delta = y_pert_native−y_base_native.
Именно actual delta подаётся в A@delta. Rounding residual не является
структурным эффектом. Формула Gaussian Frobenius expectation для nominal z
не переносится автоматически как точное равенство на rounded actual delta.

`fit_control(X_train_binary, actual_y, iterations=64, seed=0, threads=4)` —
узко ограниченный CatBoostRegressor: RMSE, depth2, Plain, bootstrapNo,
random_strength0, L2=1, learning_rate=.1, Gradient leaf estimation, один шаг,
backtrackingNo, scoreL2, zero initial bias/boost_from_averageFalse,
border_count1, rsm1, model_shrink_rate0, LangevinFalse. Нет sample weights,
init_model, categorical/text/embedding features или eval-based выбора.
`get_all_params` проверяет реально разрешённые настройки. Metadata связывает
модель с unweighted train matrix и actual label vector. Все fit/predict/leaf
API получают явный thread count1…4. Метод не запускается из default CI.

`operators_from_partitions` — чистая алгебра на заданных leaf indices.
`frozen_operators(model,X_train,X_evaluation,checkpoints,targets=actual_y)`
использует actual leaf indices и actual resolved learning_rate и L2.
CatBoost сохраняет eta какfloat32: используется значение из get_all_params,
а не идеализированное десятичное .1. Scale должен быть1, bias0; leaf weights
должны совпасть с unit training occupancy. До интерпретации чего-либо
каждый requested prefix проверяется по A@y_actual против `model.predict` с
atol/rtol1e-10 (smoke задаёт atol1e-10, rtol0). Несовпадение — ошибка
реализации или известных solver semantics; оно запрещает научную интерпретацию.

Рекурсия при нулевой инициализации: A_train=A_eval=0. Перед каждым деревом
R=I−A_train. Для train leaf g вычисляется B_g=sum_{i in g}R_i/(n_g+L2).
Оба обновления используют **этот же R до изменения A_train**:
A_train+=eta B[train_leaf], A_eval+=eta B[eval_leaf]. Пустой train leaf
имеет нулевое обновление. Снимки prefix operators readonly. Это точный
условный fixed-partition контроль в ограниченных настройках, не объяснение
полного адаптивного refit и не безусловная причинная модель.

`decomposition(full_delta,frozen_delta,epsilon)` сохраняет structural delta,
нормированные squared norms и **signed** 2cross с denominator epsilon²*n_eval,
а также численную closure error. Cross может быть отрицателен; discrepancy
не называется неотрицательной добавочной variance. Это norm identity,
а не новый theorem.

`representation_mse_floor(signatures,y)` возвращает групповые средние,
residual и эмпирический MSE floor для точных одинаковых signatures. Это
известный train-only контроль. Нулевой floor — необходимое, но не достаточное
условие интерполяции depth-limited regularized finite ensemble.

## Проверки и что ещё требуется

Default unit suite проверяет ручные partitions, пустые leaves, использование
pre-update residual operator, signed negative cross, train-only nested
quantiles, tied quantiles, shapes, явное native rounding и collision floor.
Никаких fit по умолчанию.

Tiny CatBoost smoke включается только через `RESEARCH_MODEL_SMOKE=1`; seed71,
12 детерминированных toy train rows, 3 evaluation rows, 5 деревьев. Он проверяет
baseline prefix replay, actual eta и изменение leaf values на новых dyadic
labels при тех же partitions. Это техническая проверка, не scientific pilot.

После подтверждения root, что dispatcher R13 завершён, tiny smoke выполнен
на CatBoost1.2.10, threads1, seed71. Все9 проверок прошли (8 pure algebra и
один fitting smoke). Actual eta=0.10000000149011612, actual L2=1.
Для prefixes0/1 max absolute baseline error равен0 на train/evaluation;
prefix3: train1.734723475976807e-18, evaluation1.3877787807814457e-17;
prefix5: train/evaluation2.7755575615628914e-17. Изменённые dyadic labels
при тех же partitions дают max absolute evaluation error2.7755575615628914e-17.
Tolerance остался atol1e-10, rtol0. Для nominal .1 явный float32 rounding delta
равен1.4901161138336505e-09; эти разные масштабы не смешиваются.

Во вспомогательном тесте обнаружены процедурные особенности CatBoost:
`model.copy().set_leaf_values` отвергается как non-solid representation;
`set_leaf_values` на исходной disposable toy model меняет сохранённые values,
но ранее созданный prediction calcer сохраняет старые values. Public
save/load создаёт свежий calcer и восстанавливает проверенное совпадение.
Поэтому тест проверяет новые leaf values и читает их через свежий save/load;
научные модели не редактируются. Эти API/storage особенности не являются
новым научным эффектом и не потребовали ослаблять численный tolerance.

До научных вычислений нужны отдельные PLAN.md/machine plan, источники с
проверенными assumptions и snapshots, budgets/stops, раздельные discovery и
confirmation inputs, отдельный reviewer и проверка строгого baseline replay.
Наличие этого модуля не означает, что новый вопрос уже решён или новизна доказана.

## Возобновление: short-prefix контроль и отдельная стоимость leaf audit

8 ноября 2026. Продолжение реализации: `/root/r14_implementation`; исходное
авторство `/root/catboost_study` сохранено. Ни study seed, ни scientific fit
не запускались. Следующий nonstudy precision smoke разрешён coordinator:
12 фиксированных toy train rows,3 evaluation rows,seed71,threads1.

Точный вызов:
```powershell
$env:RESEARCH_MODEL_SMOKE='1'; uv run python -m unittest discover -s tests -p test_boosting_response.py -k fresh_short_fit -v
```

Фактический объединённый stdout/stderr этого вызова:
```text
test_fresh_short_fit_matches_full_prefix_and_long_replay (test_boosting_response.BoostingResponseModelSmoke.test_fresh_short_fit_matches_full_prefix_and_long_replay) ... ok
{"technical_smoke": "fresh-short32-vs-full512-prefix32", "catboost": "1.2.10", "seed": 71, "threads": 1, "n_train": 12, "n_evaluation": 3, "resolved_full_control": {"iterations": 512, "depth": 2, "learning_rate": 0.10000000149011612, "l2_leaf_reg": 1, "loss_function": "RMSE", "boosting_type": "Plain", "random_strength": 0, "bootstrap_type": "No", "leaf_estimation_method": "Gradient", "leaf_estimation_iterations": 1, "leaf_estimation_backtracking": "No", "score_function": "L2", "boost_from_average": false, "border_count": 1, "grow_policy": "SymmetricTree", "task_type": "CPU", "rsm": 1, "model_shrink_rate": 0}, "resolved_short_control": {"iterations": 32, "depth": 2, "learning_rate": 0.10000000149011612, "l2_leaf_reg": 1, "loss_function": "RMSE", "boosting_type": "Plain", "random_strength": 0, "bootstrap_type": "No", "leaf_estimation_method": "Gradient", "leaf_estimation_iterations": 1, "leaf_estimation_backtracking": "No", "score_function": "L2", "boost_from_average": false, "border_count": 1, "grow_policy": "SymmetricTree", "task_type": "CPU", "rsm": 1, "model_shrink_rate": 0}, "actual_scale_and_bias": [1.0, 0.0], "short_full_prefix_max_absolute_error": {"train": 0.0, "evaluation": 0.0}, "leaf_indices_and_values_exactly_equal": true, "baseline_checks": [{"trees": 32, "split": "train", "n": 12, "max_absolute_error": 2.220446049250313e-16, "atol": 1e-10, "rtol": 0, "passed": true}, {"trees": 32, "split": "evaluation", "n": 3, "max_absolute_error": 1.1102230246251565e-16, "atol": 1e-10, "rtol": 0, "passed": true}, {"trees": 512, "split": "train", "n": 12, "max_absolute_error": 2.220446049250313e-16, "atol": 1e-10, "rtol": 0, "passed": true}, {"trees": 512, "split": "evaluation", "n": 3, "max_absolute_error": 7.771561172376096e-16, "atol": 1e-10, "rtol": 0, "passed": true}]}

----------------------------------------------------------------------
Ran 1 test in 0.609s

OK
```

Это техническая проверка budget semantics на toy input, не независимая
репликация и не доказательство prefix invariance для всех данных. Gate
atol1e-10/rtol0 остаётся обязательным для каждого научного context.

После независимого замечания reviewer P3 timing исправлен до scientific run:
`prediction_seconds` включает только native prediction и его проверки;
`calc_leaf_indexes`, leaf counts/weights вынесены в отдельный
`leaf_evidence_seconds` для base/cheap/full. Raw leaf evidence сохраняется
полностью; её стоимость не входит в primary fit+predict+decomposition P3.
Default mocked clock regression проверяет это разделение.

Default unit suite дополнен полной mocked pipeline-проверкой: два не-study
seed label5/6,16 toy contexts,48 pure mock objects вместо CatBoost fits,
буквальные фиксированные arrays вместо RNG. Проверены все13 зарегистрированных
итоговых файлов,16 context NPZ,48 input/prediction NPZ и48 mock model files,
complete progress, OOF rows, отдельные leaf timings и честный false range gate.
Эти временные mock files не являются научными моделями или evidence пилота.
`uv run python -m unittest discover -s tests -p test_response_forecast.py -v`:
14 tests passed; никакой генерации study data и CatBoost fitting нет.
