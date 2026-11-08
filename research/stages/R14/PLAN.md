# R14 — discovery: прогноз ошибки frozen-tree response

Статус: frozen для independent implementation preflight; execution запрещён
до его acceptance, validate-plan и активации. R13 операционно закрыт.
Автор дизайна `/root`; первоначальный technical helper/runner author
`/root/catboost_study`, продолжение `/root/r14_implementation`; первоначальный
renderer author `/root/literature`, продолжение `/root/r14_figures`.
Отдельный reviewer `/root/r14_reviewer` не является одним из авторов.
Scientific design review сохранён в draft-design-review.md; окончательный
source-bound preflight привязывается в inputs/design-review.json. Этот этап
не обещает новизну и не ищет нужную кривую.

## Вопрос и возможная ценность

Может ли короткая проба label perturbations предсказать out-of-sample squared
ошибку точного conditional frozen-partition replay при последующем полном
адаптивном refit лучше, чем доступные до refit complexity/geometry baselines?
Это условие применимости одного объяснения бустинга, связанное с исходной
программой механизмов double descent. Сам double descent здесь не endpoint.
Новый вклад возможен только как переносимый и экономящий вычисления forecast;
известные LeafRefit-type replay, изменение структуры, EFC, collision floors
и norm identities не являются заявками на новизну.

Первичные antecedents и доступ: docs/literature/C2-source-cards-2026-11-08.md
и next-question-scout-2026-11-07.md; disambiguation weak-tree representation:
docs/literature/C2-author-code-note-2026-11-08.md. Priority остаётся открытым. Вычисления
не подменяют проверку литературы. EFC — явная адаптация Eq2/3 к новым weak
trees CatBoost с actual zero initialization и фактическим SSE, а не побитовая
репликация эксперимента Allerbo/Schön. Не переносим influence-LOO estimand.

## До вычисления: фиксированная матрица

32 contexts = 2 teachers × 4 nested candidate prefixes × 4 independent data
draws. Seeds14001..14004; draw является cluster/unit повторения. У каждого
draw независимые train128 и unlabeled evaluation256, 4 raw coordinates iid
N(0,1). Teachers: linear=sum(x)/2; interaction=(x0*x1+x2*x3)/sqrt(2).
Обе имеют population variance1. Train labels teacher+N(0,.1²), без empirical
rescaling. Train/eval covariates seed channels +100000/+200000, label noise
+300000. Оба teacher используют одни covariates/noise для парного дизайна.
Никаких внешних данных, загружаемых корпусов, embeddings или paid API.

Train-only nested dyadic quantiles из helper: levels1/3/7/15, prefixes4/12/28/60.
Все4 raw features представлены с первого prefix. Ties сохраняются, actual
unique thresholds/signatures и collision floors сохраняются; prefix не
называется эффективным числом параметров или порогом интерполяции.

Модель строго helper control: RMSE, symmetric depth2,512trees, nominal eta.1,
L2=1, random_strength0, No bootstrap, Plain, Gradient one leaf step,
backtrackingNo, border_count1 над binary columns, zero bias, one fit,
use_best_modelFalse, threads≤4. Seeds seed+400000; одинаковый seed при refits.
Никакого подбора eta/depth/числа деревьев по outcome.

Для каждого context: base512; eps .01 и .1. Для каждой eps две cheap pilot
directions и четыре новые full-response directions. Directions iid N(0,I),
общие между prefixes/teachers для парности, отдельные streams seed+500000
(pilot) и seed+600000 (response). Один direction повторяется между eps,
но pilot и response streams не пересекаются. Cheap fits32trees; response
fits512. 13fits/context,416fits всего. Сохраняются все attempts, модели,
номинальные/actual labels, z, thresholds, leaf indices, operators, predictions,
raw per-direction metrics и fit/replay timings. Никакого optional stopping
по промежуточному эффекту: весь fixed matrix либо техническая остановка.

## Проверяемый контроль и estimands

Perturbation nominal target равен base_actual+eps*z; baseline nominal target
и его первоначальное округление сохраняются отдельно. Pair rounding delta
равна actual_delta−eps*z. Не добавлять eps*z к первоначальному nominal base.

До интерпретации в каждом context baseline A_t*y_actual должен воспроизвести
native predictions на train/evaluation при t32/512, atol1e-10, rtol0.
Метки до fit явно cast float32; δ=actual perturbed−actual base, а не εz.
Zero δ сохраняется как invalid direction, не заменяется новой пробой. Весь
fixed matrix сохраняется, но при таком случае OOF/decision не вычисляется:
scientific interpretation=inconclusive, go=false, строка не удаляется.
Mismatch/неполная модель/NaN/таймаут — integrity/technical failure, не эффект.
Нельзя увеличить tolerance, удалить неудобный context или добрать seed.

Base train predictions сохраняются на всех512 итерациях для EFC/SSE weights;
для response fits нужны final train/evaluation predictions и полные train/
evaluation leaf indices, CBM и targets, не все prediction trajectories.

Для каждой новой response direction full=prediction_refit−prediction_base,
frozen=A_eval,512*δ, structural=full−frozen. Primary context/eps endpoint
D=mean_d ||structural||²/(256 ε²). Signed2cross и полная norm closure
сохраняются; structural discrepancy не называется extra variance. Идеальная
Gaussian expectation Frobenius не объявляется точной для rounded labels.
Evaluation teacher labels не нужны для endpoint и не подбирают forecast.

Cheap diagnostic C — такое же D на evaluation при t32 по двум другим
directions; train-only counterpart также сохраняется как secondary.
Использование unlabeled evaluation covariates явно входит в доступ forecast.
Endpoint не является ошибкой истинного teacher и не доказывает ухудшения
prediction quality от perturbation. Secondary clean teacher MSE и train
residuals имеют только описательный статус.

## Forecast, baselines и анализ discovery

Target u=log10(1+D). Заранее фиксированы ridge alpha1, fitted intercept,
train-fold standardization (нулевая variance→scale1). Никакого tuning alpha,
выбора feature subset или обработки outliers по endpoint. Четыре leave-one-
draw-out folds; все teachers/prefixes/eps одного draw исключаются вместе.
Это discovery cross-validation, НЕ независимая confirmation и НЕ64 iid units.

Baselines: B0 mean training-fold u; B1 ridge по log10(eps), log10(prefix),
teacher indicator; B2 ridge B1 плюс log10(1+trainMSE512), collision MSE floor,
trajectory-weighted EFC, log10(1+||A_eval512||F²/256),
log10(1+||A_train512||F²/128), log10(1+largest singular value(A_train512)).
EFC_t=1−sum_l n_l(n_l−1)/(128*127) на каждом новом weak tree;
weights=max(0,SSE(t−1)−SSE(t)), initial SSE=||actual_y||². Если weights sum0,
это undefined, integrity для B2; не imputing0. All features фиксируются до
full response endpoint. A-model = B2 плюс v=log10(1+C).
Дополнительные сильные simple-probe baselines: B3 ridge по B1+v без512
geometry/EFC; B4=v+mean_train(u−v), фиксированный slope1 и только training-fold
offset. Никакого выбора slope/calibration по held-out rows. B3/B4 нужны,
чтобы не выдать пользу очевидного partial-response measurement за новый
geometry-aware forecast.

Practical scope — conditional audit: пользователь уже имеет full basefit и
frozen operator, применимость которого проверяет. Их стоимость показывается
отдельно. Secondary end-to-end ratio дополнительно включает basefit/operator
и показывает предел практического переноса; общего cheap standalone service
из conditional gate не следует.

Primary metric MAE(u) на64 held-out context/eps predictions. Error averaging
сначала равновесно внутри draw; сравниваются aggregate и4 draw-specific MAE.
Никаких p-values/узких confidence intervals из64 correlated rows. Также
сохранить каждую prediction, residual, full feature table и ошибки по teacher,
prefix, amplitude, direction. Никакой test-adapted grid.

## Предсказания, фальсификаторы, stop/go

P1 known control: все32 baseline replays при32/512 проходят fixed tolerance;
все actual configurations/hash/leaf-weight gates проходят. Иначе нет scientific
interpretation, только technical diagnosis с новым attempt, максимум3.
P2 forecast: A-model MAE минимум на20% ниже КАЖДОГО B0/B1/B2/B3/B4;
A не хуже КАЖДОГО из B2/B3/B4 минимум в3из4 draw folds и отдельно в каждом
из2 teachers. Сравнения и fold gates сохраняются для всех baselines, без
выбора победителя по endpoint. Фальсификатор —
невыполнение любой части. Результат не переименовывается в успех при другом
ridge/feature/eps или графике correlation. Target q90−q10≥.1 log units нужен
для meaningful forecast range; иначе inconclusive lack of challenge.
P3 practical: total extra cheap-fit+predict+decomposition wall≤50% такого же
fit+predict+decomposition wall четырёх full-response refits, aggregated over
amplitudes/contexts. Все компоненты записываются отдельно; чтение/запись
архивов не входит в этот comparative wall и показывается отдельно. Общий
basefit/512-operator учитывается отдельно и показывается. Это измерение в
данной среде, не универсальный speedup; считать не только число деревьев.

Proceed к отдельному R15 confirmation на новых draw seeds/data только если
P1/P2/P3 и range gate выполнены. Замораживается весь A/B pipeline без нового
selection; confirmation sample size/power и новая область переноса фиксируются
ДО генерации данных. Если P2/P3/range gate fails — остановить этот forecast
candidate и честно оформить negative/inconclusive, не расширять сетку ради
успеха. Новая содержательная гипотеза требует отдельного решения.

## Бюджеты и выход

Draft cap:1800s/attempt,3600s cumulative,3attempts; CPU4,one model/dispatcher,
GPU/API/download0; new storage≤1GiB. Parent24h/20GiB ledger учитывает прежние
stages, intake и retries. Не запускать до закрытия R13 и independent preflight.
Сначала pure tests и отдельный tiny implementation smoke; study seeds в тестах
запрещены. План/протокол/source snapshot/uv.lock/data hashes обязательны.
Reviewer не автор runner или оцениваемого вывода. Verify→REPORT со всеми
исходами→bound review→close→archive/restore verify→STATE. Новые claims только
с точными evidence scope и реальным отдельным review.

Графические артефакты должны показывать все contexts: D/C по каждой amplitude,
actual vs frozen responses, signed cross terms, clean quality, collision floors,
EFC/noise gain, out-of-fold predictions/residuals для всех моделей, fold/teacher
ошибки, runtime costs. PNG+SVG, machine tables и gallery создаются отдельным
производным figures task после response-study в этом stage, без выборочного
показа удачных seeds и без увеличения n. До scientific execution зафиксирован
inputs/figure-spec.json: complete OOF —113PNG/SVG pairs и7CSV; invalid actualδ
и unavailable OOF —89pairs и6CSV, с явной unavailable annotation.
Обе ветки сохраняют все32contexts/384directions,147456response rows,
2816component timing rows и11cost-scope rows. Figure branching определяется
заранее заданной integrity policy, а не выбором более удачной картинки.

Nonstudy implementation controls до freeze: seed71,12train/3eval,threads1,
fresh32fit vs full512prefix32 точно совпали по native predictions, leaves и
values; baseline operator replay max error7.77e-16 при прежнемatol1e-10/rtol0.
Команда/actual raw log/version сохранены в TECHNICAL.md. Это technical control,
не independent science replication и не n пилота. Стоимость извлечения leaf
evidence сохраняется отдельно; primary P3 считает fit+nativepredict+decomp,
как определено выше. Full mock pipeline/pure tests не генерировали study data.
