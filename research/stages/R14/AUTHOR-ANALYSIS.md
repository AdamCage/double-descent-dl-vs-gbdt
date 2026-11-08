# R14 — авторский описательный анализ полного response-study

Автор: `/root/r14_implementation`, продолжение реализации
`/root/catboost_study`. Это **author analysis, не независимый review**, не
новая пререгистрация и не разрешение нового эксперимента. Источник — completed
task `response-study`, attempt `20261108T073937-26f816e66d`.
Ни fit, ни inference сохранённых моделей при подготовке этой записки не
выполнялись. Исходники, PLAN, protocol и tests не менялись.

Все нижеуказанные файлы находятся в
`artifacts/research/R14/20261108T073937-26f816e66d/tasks/response-study/`.
Точные input/source/lock bindings сохранены в attempt; отдельный reviewer
оценивает их независимо. Анализ этого task не удостоверяет завершение всего
stage с производными фигурами или исходной научной цели.

## Заранее заданный исход: forecast candidate останавливается

Матрица завершена:32 contexts,64 context/amplitude rows,128 pilot directions,
256 full-response directions,416 fits. Независимая единица —4 data draws,
seeds14001–14004. Teachers, nested prefixes, две amplitudes и probe directions
внутри draw связаны. Это не64 или384 независимые репликации.

По `analysis.json`, поля `decision`, `forecast.summaries` и `costs`:

- P1 passed: все128 baseline replay checks при32/512 trees и на train/evaluation
  прошли atol1e-10/rtol0. Max absolute error `7.993605777301127e-15`.
- P2 failed: A не достиг заранее требуемого20% MAE выигрыша над каждым B0–B4.
  Прибавка против B1 —5.25897%, B2 —4.76860%, B3 —3.03118%. Это меньше20%.
  Против B0 —53.13288%, B4 —33.84490%, но победы над этими более слабыми
  comparators недостаточно для conjunctive gate.
- Per-baseline draw gate прошёл: A не хуже B2 в3/4 draws, B3 в3/4, B4 в4/4.
  В seed14003 A хуже B2; в seed14004 хуже B3. Сравнения с каждым из B2/B3/B4
  в обоих teachers прошли. Эти выполненные части не заменяют failed20% gate.
- Range gate прошёл: q10(u)=0.0000617054710813,
  q90(u)=1.572104564922946, размах1.572042859451865 при требовании≥.1.
  Здесь отрицательный outcome не объясняется отсутствием variation endpoint.
- P3 passed в объявленном conditional-audit scope: дополнительный cheap
  fit+prediction+decomposition wall3.654164098s против24.260860392s full response,
  ratio0.1506197241 при требовании≤.5. Это измерение стоимости, не accuracy claim.

`decision.go=false`, `outcome=stop_this_forecast_candidate`. Новые alpha,
features, seed, amplitudes или более удобный comparator по этому endpoint
не подбирались. Confirmation этого candidate по данному PLAN не запускается.
Небольшой наблюдаемый MAE выигрыш не доказывает полного отсутствия predictive
signal; он не удовлетворяет заранее выбранной минимальной практической прибавке.

### Все заранее заданные forecasters

Target u=log10(1+D); MAE усреднён равновесно внутри draw, затем по4 draws.
Все значения — discovery cross-validation, без p-values или iid64 intervals.

| Model | Overall MAE(u) | Draw14001 | Draw14002 | Draw14003 | Draw14004 | Linear teacher | Interaction teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 training mean | .517546 | .454327 | .550308 | .493167 | .572380 | .422252 | .612839 |
| B1 metadata | .256023 | .237824 | .281705 | .239684 | .264878 | .196604 | .315442 |
| B2 metadata+geometry/EFC | .254705 | .227314 | .270868 | .229945 | .290692 | .184304 | .325105 |
| B3 metadata+cheap probe | .250141 | .237583 | .275365 | .235518 | .252098 | .187871 | .312411 |
| B4 unit-slope probe+training offset | .366652 | .338189 | .384984 | .336980 | .406453 | .246985 | .486318 |
| A B2+cheap probe | .242559 | .226732 | .236375 | .232798 | .274329 | .179259 | .305858 |

Источник: `analysis.json:forecast.summaries`; все64 held-out predictions,
features и residuals для всех6 models — `forecast_rows.jsonl`, keyed by `id`.
Единственный используемый ridge alpha1 и train-fold-only scaling не менялись.

## Что показывает полная карта response error

D — среднее по4 новым full-response directions
`||full_delta−frozen_delta||²/(256 ε²)` на evaluation при512 trees.
C — аналогичная величина по2 другим pilot directions при32 trees.
Это ошибка приближения реакции модели, **не риск относительно teacher**.
Evaluation covariates доступны probe; clean teacher labels не входят в forecast.

Таблица включает **все16 teacher×prefix×amplitude groups**, в каждой ровно4
draws. Mean/min/max — описательные значения этих4 draws, не confidence interval.
Mean raw discrepancy MSE=ε² mean(D), чтобы нормировку наε² не спутать с
ненормированной величиной изменения предсказания.

| Teacher | Prefix | ε | Mean D | Observed D min–max | Mean C | Mean ε²D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Linear | 4 | .01 | .000158378 | .00000153288–.000331045 | .443622 | 1.58378e-8 |
| Linear | 4 | .1 | .0000323695 | .000000239006–.0000611264 | .0318325 | 3.23695e-7 |
| Linear | 12 | .01 | 2.02917 | .705135–3.49811 | 6.57313 | .000202917 |
| Linear | 12 | .1 | .142096 | .0807975–.185021 | .0919636 | .00142096 |
| Linear | 28 | .01 | 5.18222 | 3.82120–6.18667 | 10.4875 | .000518222 |
| Linear | 28 | .1 | .376825 | .367717–.398861 | .291857 | .00376825 |
| Linear | 60 | .01 | 13.5343 | 10.3797–18.4246 | 16.3755 | .00135343 |
| Linear | 60 | .1 | .582495 | .451167–.706499 | .378313 | .00582495 |
| Interaction | 4 | .01 | .00454798 | .000181464–.0113397 | .676234 | 4.54798e-7 |
| Interaction | 4 | .1 | .00569007 | .000127899–.0151571 | .120010 | .0000569007 |
| Interaction | 12 | .01 | 6.49175 | 3.34848–10.8212 | 9.80381 | .000649175 |
| Interaction | 12 | .1 | .627158 | .307629–1.27735 | .785903 | .00627158 |
| Interaction | 28 | .01 | 90.9742 | 14.1198–121.299 | 5.83216 | .00909742 |
| Interaction | 28 | .1 | 1.88081 | 1.13141–2.48812 | .438041 | .0188081 |
| Interaction | 60 | .01 | 176.918 | 48.3984–401.103 | 6.20509 | .0176918 |
| Interaction | 60 | .1 | 4.56870 | 1.99290–6.20258 | .566322 | .0456870 |

Источник: все64 `forecast_rows.jsonl:{teacher,prefix,epsilon,full_discrepancy,
cheap_discrepancy}`; row IDs перечислены в OOF folds. Ни seed, ни directions
по величине эффекта не отбирались. Means совпадают с direct averages из
`directions.jsonl` и соответствующих `predictions.npz` vectors.

В этих group means D растёт от prefix4 к60 при обоих teachers и amplitudes.
Ранний C не является величиной с постоянным единичным переносом на D:
в31 из64 rows C>D, в33 C<D. Например, **полная группа** Interaction/ε.01
имеет mean C по prefixes4/12/28/60 = .676234/9.80381/5.83216/6.20509,
а mean D = .00454798/6.49175/90.9742/176.918. Это описание всех четырёх
prefix groups, не выбранной удачной кривой. B4 уже проверял простейший
unit-slope перенос с training-fold offset; его MAE хуже A, B1/B2/B3.

Диапазон всех64 D: `2.3900632409388945e-7`–`401.1032041911199`;
C: `.0000733598121264`–`24.146957556683354`. Малое normalized D при большойε
не означает меньшей абсолютной ошибки: mean ε²D при переходе .01→.1 растёт
во всех8 teacher×prefix groups таблицы. Из двух amplitudes не оценивается
закон масштабирования или пределε→0.

## Достаточность baseline replay и качество модели — разные вопросы

Точный baseline A*y_actual восстановлен во всех128 checks, но D существенно
меняется между contexts. Поэтому проверка baseline replay удостоверяет
условную leaf-refitting алгебру и solver conventions; она сама по себе не
удостоверяет точность реакции **полного адаптивного** refit.

Следующая таблица также включает все8 teacher×prefix groups. Train MSE —
относительно actual noisy labels, clean evaluation MSE — относительно известного
teacher. Collision floor — известный empirical representation control;
нулевой floor не доказывает интерполяции depth-limited512-tree ensemble.

| Teacher | Prefix | Mean train MSE | Mean clean evaluation MSE | Mean collision floor | Observed unique train signatures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear | 4 | .363845 | .419476 | .348880 | 16–16 |
| Linear | 12 | .0974330 | .198256 | .0313704 | 101–107 |
| Linear | 28 | .0211966 | .112270 | .000246250 | 127–128 |
| Linear | 60 | .00673900 | .0706675 | .0000897223 | 127–128 |
| Interaction | 4 | .632835 | .665235 | .608470 | 16–16 |
| Interaction | 12 | .179786 | .393337 | .0470622 | 101–107 |
| Interaction | 28 | .0346982 | .322388 | .0000292739 | 127–128 |
| Interaction | 60 | .0178227 | .248458 | .0000105448 | 127–128 |

Источник: все32 `contexts.jsonl`, `base_features`,
`base_clean_evaluation_mse`, `n_train_signatures`. В этих group means clean
prediction quality улучшается с prefix, одновременно с ростом mean D.
Это демонстрирует различие измеряемых величин в данном пилоте; не устанавливает
общий trade-off, причину расхождения, механизм double descent или его отсутствие.

Conditional frozen evaluation noise gain `||A_eval||F²/256` имеет общий
range .0850581902–.2445849893, значительно уже normalized full-response gain
на отдельных directions. Сопоставление не опровергает Gaussian conditional
formula: full adaptive response имеет другой estimand, а actual roundedδ
не объявлялся идеальнымεz. Baseline B2 уже включал этот conditional feature.

## Signed cross-term нельзя переименовать в добавочную variance

Для256 full-response directions evaluation `2<frozen,structural>/(256 ε²)`
отрицателен в73 случаях, observed range −2.68929732…2.37654637.
На train отрицательных58/256, range −.500440487….928344300.
Для128 pilot directions отрицательных evaluation34/128, train31/128.
Единицы направления не являются независимыми data draws.

На full-response evaluation normalized full squared norm находится в
.0275473854…550.168190, frozen norm в.0287040728….410066427,
structural discrepancy в2.91456394e-9…550.437980. Structural norm сам по
себе не равен разнице full и frozen gains из-за signed cross-term.
Max absolute norm closure error по всем384 directions и обоим splits:
`5.684341886080802e-14`.

Источники: `directions.jsonl:metrics.{train,evaluation}` и raw vectors
по `array_keys` в `predictions.npz`. Автор перечитал все768 split/direction
combinations: `structural=full−frozen` совпало точно; все четыре raw-derived
normalized norms/cross совпали с JSON точно; mean C/D по всем64 rows совпали
точно. Это author consistency check, не независимая репликация/review.

## Практическая стоимость и пределы вывода

В scope уже имеющегося baseline/operator короткие probes дешевле полного
reference response:3.654s против24.261s. Но common basefit+предсказания+
operators+features занимают22.537s. Secondary end-to-end ratio с common cost
в обеих сторонах=.559774878; если acquisition common cost включить только
в forecast и сравнить с full refits без этой подготовки, ratio=1.079786074.
Второй вариант не даёт speedup. Forecast CV computation=.004888100s.

I/O сохраняемой evidence отдельно276.841519s при task elapsed330.808025s;
leaf extraction отдельно base .074039s, cheap .171706s, full .589015s.
Эти величины не скрыты в prediction timer и не входят в primary P3.
Итоговая стоимость полного воспроизводимого исследования существенно больше
одних model-fit seconds. Источники: `timings.json`, `analysis.json:costs`.

Получена воспроизводимая **ограниченная карта ошибок приближения и failed
forecast candidate**. Она показывает, почему стоимость probe, точность
conditional replay и полезность прогнозирования нужно проверять отдельно.
Это не новый общий алгоритм, не установленный механизм и не подтверждение
приоритета. Known LeafRefit/EFC/norm controls остаются known controls.

Ограничения:4 draws; два фиксированных synthetic teachers присутствуют в
каждом training fold; одна CatBoost configuration;2 amplitudes; finite512-tree
budget; всего2/4 pilot/response directions на context/amplitude. Random
directions не охватывают adversarial или label-edit directions. Нет new-teacher,
real-data, neural-network или independent-confirmation результата. Исходная
полноценная межмодельная научная цель остаётся открытой. Любой следующий
содержательный вопрос требует нового решения и frozen protocol, без изменения
этого отрицательного исхода или расширения сетки ради положительного эффекта.

## Точные evidence hashes использованного task

| Run-relative file | SHA-256 |
| --- | --- |
| analysis.json | 31279f4e2da0411f00806a27ff1e42c5cd5e038bde4398c0124b0b2587badbfa |
| contexts.jsonl | e3b6de78e8d741522b4fa9dee14c4673b56879bee2357923159fcb6185fd9f7f |
| directions.jsonl | 2b571db5ae250cd7eaa7e54bc26e026e6d2d68fd0e24d3d5866e232ebfc745a3 |
| forecast_rows.jsonl | 6af3d32f3e609bffd4b4c16fb647e0bc03fc21e69d2108ca2834a0e7b1c7ae93 |
| predictions.npz | 8e6d5feefdd224b2187e9d5e412e38735360b155da6cf1beaf8422acc7a6537b |
| data.npz | 54b5a5f0aaaec505a27ed2cc77372bf4829e4ab92ed60fb671edf15f62142620 |
| operators.npz | 4b284e58d113e42a5107b2dd2043a8f9e6e6febb318cbbbb41b6a2254e9a53fc |
| timings.json | d5ef12e0a607f96f80732f9dead2f4390fca978bae2087f96e9923ea3bfe296c |
