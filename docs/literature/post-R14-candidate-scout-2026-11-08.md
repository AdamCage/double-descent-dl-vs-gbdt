# После R14: регулярность отклика и область точности frozen replay

8 ноября 2026. Автор scout: `/root/r14_figures`. Это отдельная проверка
первичной литературы и критика кандидатов, **не PLAN, не результат нового
эксперимента и не независимое review R14**: этот агент был соавтором renderer.
Ни study data, ни модели, ни вычислительные планы здесь не создавались.
Проверены пять работ; поиск и чтение — через web, без локальных download
snapshots, исполнения чужого кода или обхода закрытого доступа. Приоритет не
установлен; отсутствие найденной статьи не доказывает новизну.

**Рекомендация:** общую историю «hard splits дают скачки, smooth NN — гладкий
отклик» не делать главным новым результатом. Более предметный кандидат —
проверяемая, вычислительно полезная область, в которой conditional replay
адаптивного бустинга гарантированно точен. Но базовые split-margin bounds
элементарны, а обучение с сертификатами label-poisoning уже исследуется.
Нужны существенно более сильные tightness/coverage/cost результаты и связь с
исходным сравнением NN–GBDT; одной положительной границы недостаточно.

## Пять первичных источников и реальная граница чтения

### 1. Wang–Zhao–Fan: адаптивный поиск уже входит в complexity

[Авторский preprint v2, 19 июля 2024](https://arxiv.org/html/2308.13630v2),
[журнальный DOI, JCGS2025](https://doi.org/10.1080/10618600.2024.2388545).
Прочитаны abstract, §1.1–1.2, Definition1, §4.1 и relevant discussion.
Журнальный DOI снова вернул403; авторский arXiv HTML доступен. Это не новая
копия журнального полного текста и не замена прежнего acquisition record.

Для представления fitted response как `S(y)y` определена разность между
covariance-based degrees of freedom и `E_y tr S(y)` при фиксированном X.
Для regression tree conditional trace равен числу leaves, но полная
адаптивная процедура несёт стоимость поиска partitions. Это уже запрещает
называть новую разность conditional/final sensitivity первым обнаружением
«стоимости изменения структуры». Их estimand усредняет по y; наш
finite-perturbation squared response с делением на epsilon² ему не равен.
Точный закон `1/epsilon` и сертификат всего boosting path из прочитанных
разделов не следуют.

### 2. Brophy–Hammoudeh–Lowd: fixed structure и label edits уже сравнивались

[JMLR2023, полный авторизованный PDF](https://jmlr.org/papers/volume24/22-0449/22-0449.pdf).
Прочитаны Simplifying Assumption1, §3.1, §4.3/4.5 и соответствующая discussion
о структурных изменениях; это не повторение всех48 страниц экспериментов.

LeafRefit оставляет splits фиксированными, меняя leaf values; targeted label
edits уже адаптированы и сопоставлены с полным retraining. Обсуждается, почему
игнорирование структурных изменений иногда помогает group-influence ranking.
Следовательно, «frozen ошибается» и «полный refit меняет структуру» — известные
наблюдения; большее расхождение не обязано означать худшую downstream utility.
Здесь не проверялся сертификат continuous-label ball, сохраняющий всю
последовательность boosting splits. Influence ranking, точность individual
counterfactual response и robustness certificate — разные endpoints.

### 3. Kanoh: ось candidates и шумочувствительная геометрия уже заняты

[arXiv:2608.03111v1](https://arxiv.org/html/2608.03111v1), 4 августа2026.
Повторно прочитаны §3, kernel-view limitation, результаты noise/candidate
interventions и описание label swaps. Источник уже входит в прежний intake;
нового snapshot здесь нет.

Split-candidate scaling, reference tree kernel и связь peak с геометрией,
depth/n/noise уже исследуются. Reference kernel прямо не объявлен точной
моделью adaptive training. Label swaps изменяют training targets при
сохранённой candidate configuration. Поэтому новый вклад должен различать
conditional geometry и adaptive selection на заранее фиксированной оси,
а не повторять DD-кривую с новым названием сложности. В прочитанных местах
нет готового whole-path сертификата exact frozen replay; это ограничение
scout, не доказательство отсутствия такого результата во всей литературе.

### 4. Timber: split stability при label flips уже используется алгоритмически

[Calzavara–Cazzaro–Vettori, авторский PDF](https://www.dais.unive.it/~calzavara/papers/satml25.pdf),
2025. Прочитаны §II, §III-C/D/E, Algorithms1–3 и declared ensemble scope.

Для binary classification узлы аннотируются instances, single-label flip
которых сохраняет выбранный split. Сравниваются изменённые information gains
всех challengers, учитывается tie order. Это позволяет найти первый
затронутый subtree и переобучать его. Рассматривается расширение на
traditional random forests. Таким образом, поиск устойчивой структуры по
split-score comparisons — уже содержательный prior. Но это attack/efficient
retraining method для discrete flips; нельзя приписать ему общий continuous
L2-radius theorem для sequential residual-dependent RMSE boosting.
Background упоминает GBDT, что само по себе не расширяет заявленный метод
на такой boosting path.

### 5. Abstract Gradient Training: сертификаты обучения NN уже существуют

[Sosnin–Wicker–Collyer–Tsay, JMLR27(128),2026](https://jmlr.org/papers/volume27/25-2206/25-2206.pdf).
Прочитаны abstract, §5 assumptions/abstract dynamics, §6.4.3 bounded
perturbations/Algorithm1 и relevant label-poisoning examples §8.2/9.3.
Все58 страниц proofs и supplementary details не проверялись.

Работа ограничивает reachable parameter sets при first-order training,
включая bounded feature/label perturbations. Fixed initialization, ordering
и hyperparameters входят в assumptions. Интервальные over-approximations
могут терять tightness; bounds на prediction через parameter enclosure не
тождественны exact-structure radius. Это близкий prior для cross-model
сертификатов: нельзя объявлять первой идею сертифицировать NN training против
label poisoning. Сравнение с новым GBDT сертификатом потребует одинаковой
угрозы/нормы и честного учёта качества bounds и стоимости, а не переноса
NN-гарантии на произвольный optimizer.

## Кандидат A: какие именно пределы нельзя смешивать

Далее — собственная аналитическая оценка дизайна, не theorem из пяти работ.
Пусть `F_T(y)` — вектор предсказаний после фиксированного T при фиксированных
X, initialization, ordering и hyperparameters. Различны:

\[
R_y(\epsilon)=\mathbb E_Z\frac{\|F_T(y+\epsilon Z)-F_T(y)\|^2}{\epsilon^2},
\qquad R_{\rm pop}(\epsilon)=\mathbb E_Y R_Y(\epsilon).
\]

У finite hard-split procedure с конечным candidate set и строгими score gaps
обычно есть область вокруг **данного** y, где branch sequence постоянна.
При squared-loss linear leaf updates conditional map в ней affine. Для
bounded perturbation внутри доказанной области frozen replay точен; для
Gaussian Z никакой конечный radius не покрывает абсолютно все directions,
но вероятность выхода может убывать при уменьшении epsilon. Это не
популяционное усреднение по всем Y около selection boundaries.

Если Y имеет плотность около регулярной поверхности переключения, вероятность
её пересечения может иметь порядок epsilon. Если предсказание при crossing
испытывает ненулевой скачок, его вклад в normalized squared response может
иметь порядок `1/epsilon`. Это общий геометрический механизм, который здесь
не выдаётся за новое открытие или установленный закон CatBoost. Нужны
условия на boundary, jump energy, density и tails; непрерывная смена branch
или нулевая плотность меняют вывод. Из covariance-based search DF нельзя
без доказательства получить этот second-moment estimand.

Для smooth activation/loss и фиксированного конечного числа gradient steps
локальная Taylor expansion по labels ожидается из обычной дифференцируемости
композиции updates. Она не превращается в утверждение обо всех NN: ReLU,
discrete selection/early stopping и изменение optimizer path требуют отдельной
проверки. Кроме того, clamped hidden features и полный Jacobian обучения
`dF_T/dy` — разные контроли. Сравнить последний с conditional GBDT operator
честнее, чем назвать произвольный frozen-feature NN аналогом exact replay.
Внутри strict GBDT region его linearization вообще может быть точнее
нелинейной NN Taylor approximation. Общее «деревья локально хуже NN» неверно.

Наконец, epsilon→0 при fixed y, усреднение по Y, T→infinity и native precision
— разные пределы. Float32 rounding может полностью обнулить малые actual
label changes. Подгонять slope по nominal epsilon через такой plateau нельзя.
R14 с двумя amplitudes не устанавливает scaling exponent. Candidate A имеет
смысл только как заранее ограниченное сравнение **практически значимых**
noise scales/complexity interventions, с отдельным измерением crossing
frequency, jump size и within-region response. Сам generic exponent слаб
как главный novelty claim.

## Кандидат B: где возможен более предметный пробел

Кандидат для проверки: sound, достаточно tight и дешёвый **whole boosting
path** certificate, который гарантирует exact conditional response при
ограниченном изменении labels. Содержательный вопрос — не существование
очевидного score-gap bound, а остаётся ли область полезной после многих
residual-dependent steps и можно ли её практически вычислить.

Как известный элементарный baseline: если фиксированные winner/challenger
scores записаны как `y^T Q y`, разность matrices обозначена H, а observed
gap `g=y^T H y>0`, то при `||delta||2<=r`

\[
g(y+\delta)\ge g-2r\|Hy\|_2-r^2\|H\|_{\rm op}.
\]

Это обычная достаточная quadratic inequality, **не новая теорема**. Для
sequential boosting нужны корректные conditional residual operators и
одновременное сохранение всех предыдущих decisions. Нельзя заменить их
независимыми margins каждого дерева. Zero observed margin тоже не доказывает
нестабильность: identical score matrices и неизменный tie order могут
сохранять решение. Сертификат структуры и сертификат одинаковых predictions
следует различать, особенно при эквивалентных partitions.

Главные falsifiers/stop conditions для возможного будущего дизайна:

- Любой refit, нарушающий заявленную точность **внутри** certified ball,
  опровергает certificate implementation/assumptions. Это не новая физика
  hard splits; все научные claims о soundness останавливаются до diagnosis.
- Если фактический vendor split score/candidate screening/tie semantics
  не воспроизводятся, сертификат относится только к отдельно названному
  controlled learner. Нельзя незаметно назвать его сертификатом CatBoost.
- Если остаются лишь trivial/zero radii, значения ниже practically relevant
  native-label scale либо cost выше полного проверяемого refit, ограничить
  вывод negative feasibility. Meaningful radius, coverage и cost thresholds
  нужно объявить до новых данных; random probes дают upper checks, а не proof
  отсутствия нарушения во всём ball.
- Для исходной NN–GBDT цели нужны согласованные norm/threat model, fixed
  training controls, независимые data draws и сравнение с simple score-gap
  baseline, local Jacobian/curvature и ближайшими certification methods.
  Положительный certificate сам по себе не объясняет double descent.

По пяти проверенным работам whole residual-dependent boosting-path radius
не оказался уже реализованным готовым решением. Это **предварительный пробел
по ограниченному scout**, не приоритет. До большого нового stage сначала
нужны proof-level feasibility и более узкая проверка ближайших algorithmic
certification priors; иначе разумнее остановить candidate, чем превращать
элементарный margin bound в заявленную научную новизну.
