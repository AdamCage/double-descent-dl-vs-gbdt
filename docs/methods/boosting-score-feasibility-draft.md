# CatBoost split-score: source-only feasibility note

8 ноября 2026. Автор `/root/score_feasibility`. Только чтение официальных
исходников и алгебра, **не научный эксперимент, PLAN, сертификат или claim**.
Модели не обучались, scientific data R14 не читались, новые inputs/сетки не
создавались. Ветка сертификатов не активирована; дальнейший поиск остановлен
по новому направлению основной программы.

## Версия и граница проверки

Через read-only GitHub connector проверен официальный annotated tag
[v1.2.10](https://api.github.com/repos/catboost/catboost/git/ref/tags/v1.2.10):
tag object `89061789840cf67d911dae098487da78ba909ade` указывает на commit
`b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`, tag date17 февраля2026.
[Tag object](https://api.github.com/repos/catboost/catboost/git/tags/89061789840cf67d911dae098487da78ba909ade).
Поиск использовал default branch только для обнаружения путей; все изложенные
ниже implementation facts перепроверены на этом pinned commit. Это не
аттестация происхождения установленного binary wheel. Локальные snapshots
исходников не скачивались; указаны immutable primary links и прочитанные lines.

Область: single-host CPU, Plain, scalar RMSE, L2 split score, No bootstrap,
random_strength0, rsm1, symmetric tree max_depth2, unweighted numeric binary
features, no estimated/categorical features, no feature penalties/constraints,
zero initialization, один Gradient leaf-estimation step. Не переносить на
default CatBoost, GPU, ordered boosting, Newton, classification или CTR.

## Подтверждённые implementation facts

| Деталь | Проверенный исходник pinned commit |
| --- | --- |
| RMSE first derivative в используемой convention — `target−approx`; target передаётся float, second derivative−1 | [error_functions.h:379–402](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo_helpers/error_functions.h#L379-L402) |
| Leaf average — сумма derivatives/(count+scaledL2), а для пустого leaf0 | [online_predictor.h:112–126](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo_helpers/online_predictor.h#L112-L126) |
| Plain L2 scorer суммирует leaf average × leaf derivative sum; следовательно leaf contribution — s²/(n+scaledL2) | [score_calcers.cpp:20–33](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/score_calcers.cpp#L20-L33), [score_calcers.h:76–93](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/score_calcers.h#L76-L93) |
| Regularizer для scoring масштабируется как float(L2Reg)×mean training weight | [scoring.cpp:726–769](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/scoring.cpp#L726-L769) |
| DevLeafwiseScoring по умолчанию false. Classic и alternative leafwise ветки нельзя молча смешивать, хотя формула scaledL2 совпадает | [data_processing_options.cpp:9–33](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/options/data_processing_options.cpp#L9-L33), [leafwise_scoring.cpp:443–483](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/leafwise_scoring.cpp#L443-L483) |
| No bootstrap задаёт sample weights1. Bernoulli sample rate для не-Bernoulli bootstrap1, и control сохраняет все rows | [tensor_search_helpers.cpp:543–565](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/tensor_search_helpers.cpp#L543-L565), [calc_score_cache.h:55–61](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/calc_score_cache.h#L55-L61), [calc_score_cache.cpp:1172–1180](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/calc_score_cache.cpp#L1172-L1180) |
| random_strength0 обнуляет score standard deviation при конечных derivatives; RNG всё ещё присутствует в control flow | [greedy_tensor_search.cpp:851–866](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/greedy_tensor_search.cpp#L851-L866), [rand_score.h:42–47](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/rand_score.h#L42-L47) |
| Symmetric grow выбирает один общий split на depth. При равных scores/gains остаётся первый в фактическом traversal: обновление использует strict `>` | [tensor_search_helpers.cpp:716–756](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/tensor_search_helpers.cpp#L716-L756), [greedy_tensor_search.cpp:934–969](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/greedy_tensor_search.cpp#L934-L969) |
| Кандидаты проходят available-feature layout, compression в bundles/binary packs/groups, затем rsm filter. rsm1 не означает traversal по исходным column indices | [greedy_tensor_search.cpp:149–308](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/greedy_tensor_search.cpp#L149-L308), [311–389](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/greedy_tensor_search.cpp#L311-L389) |
| Feature weight default1, first-use/per-object penalties default0; иные values изменяют сравниваемые gains | [feature_penalties_options.h:13–25](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/options/feature_penalties_options.h#L13-L25), [feature_penalties_calcer.cpp:159–205](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/feature_penalties_calcer.cpp#L159-L205) |
| max_depth2 не гарантирует4leaves: после выбранного split проверяется redundancy, избыточный уровень удаляется и построение останавливается | [greedy_tensor_search.cpp:1171–1273](https://github.com/catboost/catboost/blob/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084/catboost/private/libs/algo/greedy_tensor_search.cpp#L1171-L1273) |

Следствие: CPU L2 score с regularization нельзя заменить обычным уменьшением
нерегуляризованной SSE. При leaf value a=s/(n+λ) фактическое SSE improvement
равно2as−na², тогда как scorer используетas. Приλ>0 это разные величины.
Эта алгебра следует из приведённого кода, не является новым результатом.

## Условная квадратичная форма: только real-arithmetic derivation

Если предыдущие деревья/leaf updates фиксированы, начальное prediction0 и
остаток перед деревом t равен r=R_{t−1}y, то для candidate b на текущем depth
его refinement cells имеют индикаторы u_l и occupancy n_l. В указанном
unweighted controlλ=1. Формула scorer имеет вид

\[
 S_{t,d,b}(y)=\sum_{l:n_l>0}\frac{(u_l^TR_{t-1}y)^2}{n_l+\lambda}
 =y^TQ_{t,d,b}y,
\quad
 Q_{t,d,b}=R_{t-1}^T\Big[\sum_l\frac{u_lu_l^T}{n_l+\lambda}\Big]R_{t-1}.
\]

R_{t−1} остаётся одним и тем же для обеих depth decisions внутри дерева;
его нельзя обновлять после первого split так, словно уже добавлено дерево.
После всего фактически сохранённого дерева с leaf operatorP_t real-arithmetic
recurrence — R_t=(I−ηP_t)R_{t−1}. Если redundant level удалён, P_t относится
к оставшемуся дереву. Эти условные expressions не воспроизводят сами по себе
выбор ветвей C++ и не являются новым whole-path theorem.

При unit feature weights scoreBeforeSplit одинаков для всех challengers на
этом depth; сравнение gain эквивалентно сравнению scores. Для сохранения
winner w в идеализированном порядке нужен strict gap к каждому более раннему
challenger; к более позднему достаточно nonnegative gap. Консервативное
требование строгих gaps ко всем — sufficient, но может оказаться vacuous.
Нулевой observed gap не доказывает нестабильность: identical quadratic forms
с фиксированным tie order требуют отдельного разбора.

## Что не установлено и почему нативного сертификата пока нет

- Одного списка выбранных splits недостаточно для доказанного восстановления
  полного candidate traversal и всех проигравших scores. Доступность таких
  diagnostics через public API здесь не проверена. Packing/bundling,
  ignored/constant features и tie order нужно явно связать с quantized data;
  порядок raw feature indices не является доказанной заменой.
- Strict real-arithmetic score gap — не доказательство результата floating
  C++ comparisons. Summation, permutation, caching/subtraction и compiler/SIMD
  rounding требуют отдельного error bound или outward interval treatment.
  В этой задаче соответствие нативных scores численно не проверялось.
- Native targets float32: обсуждаемое изменение должно быть actual representable
  δ, а не произвольное nominal εz. Continuum ε→0 и машинная grid различны.
- Сохранение целого path включает winner, tie, redundant-split removal и stop
  decisions каждого уровня/дерева, не только final leaves или отдельные margins.
- Здесь не проверены полезность радиусов, tightness, стоимость, neural baseline
  или приоритет относительно Timber/AGT/search degrees of freedom. Обычная
  quadratic-gap inequality уже отмечена как элементарный baseline в scout.

Если ветка когда-либо будет возобновлена, до scientific execution понадобятся
отдельно разрешённый nonstudy score/tie/arity smoke, fixed source/binary scope,
soundness review независимого автора и заранее заданные критерии полезного
radius/cost. Это перечень отсутствующей верификации, **не новый план запуска**.
Ни сертификат CatBoost, ни научная новизна, ни practical feasibility сейчас не
установлены. Никаких новых вычислений или продолжения этой ветки эта записка
не авторизует.
