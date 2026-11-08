# C2: первичные источники и точность baseline

Дата: 8 ноября 2026, Europe/Moscow. Автор: `/root/literature`.
Это проверка источников и требований к будущему контролю, не результат
обучений, не независимое review реализации C2 и не установление новизны.

## Acquisition и ограничения

План записан до retrieval:
`docs/literature/C2-acquisition-plan-2026-11-08.json`.
Log: `artifacts/intake/C2-literature/acquisition.jsonl`.
Новый intake: **5 HTTP requests, 2 792 827 payload bytes**, три полных текста.
У Wang et al. оба официальных URL вернули 403; тела отказов не загружались,
попытки сохранены, обхода доступа не было. Последующие attempts не разрешены
этим планом. Cumulative ledger после intake: 25 371 622 document bytes,
22 HTTP requests, dataset bytes 0, при caps 200 MiB и 48 requests.

| Работа, фиксированная версия | Локальный snapshot относительно repo root | Bytes; SHA-256 |
| --- | --- | --- |
| Allerbo–Schön, [arXiv:2605.21167v2](https://arxiv.org/html/2605.21167v2), 2026-11-04 | `artifacts/intake/C2-literature/allerbo-schon-2605.21167v2-attempt1.html` | 807385; `edb8c057af99bc20f86e8aec505895b1bc051596f378899f6d2b2afa13b8b4f4` |
| Brophy–Hammoudeh–Lowd, [JMLR 24(154):1–48](https://jmlr.org/papers/v24/22-0449.html), 2023 | `artifacts/intake/C2-literature/brophy-hammoudeh-lowd-jmlr23-attempt1.pdf` | 1331976; `5e6b3cee1cd656d95cd9fe0c2aa3f237677bda96733af2da5c2c55415a6de96a` |
| Jeffares–Curth–van der Schaar, [arXiv:2411.00247v1](https://arxiv.org/html/2411.00247v1), 2024-10-31 | `artifacts/intake/C2-literature/jeffares-curth-2411.00247v1-attempt1.html` | 653466; `0513fe3831ef41525543adcb94177d15ca75dbdfbe08bfd29da8777f879ec28b` |
| Wang–Zhao–Fan, [JCGS DOI](https://doi.org/10.1080/10618600.2024.2388545), 2025 | Не получен: official `/doi/full/` и `/doi/abs/` вернули 403 | 0 bytes; hash отсутствует |
| Kanoh, [arXiv:2608.03111v1](https://arxiv.org/html/2608.03111v1), 2026-08-04 | Повторно использован `artifacts/intake/v2-literature/kanoh-2608.03111v1-attempt1.html`; повторного download нет | SHA-256 `29cc4b9f3f1ba56b4d0357f421d5c052299bef4ff25b2f964574ee35bbdb010f` |

Все новые успешные snapshots получены 2026-11-08; точные UTC timestamps,
content type, attempt, locator, bytes и plan hash есть в log. Хеши проверены
по сохранённым байтам. PDF JMLR прочитан с text extraction; страницы 7–9
дополнительно rendered и визуально проверены по PDF skill. Это не
воспроизведение численных результатов статей.

## Allerbo–Schön v2: что именно определено

Проверены §3, Definitions 1–2, Eq. (1)–(3), §4.2, §5, §6, Appendix B.2,
D.2 и E. Полные доказательства не проверялись. Используется **v2 FC/EFC**,
а не майская v1 с обозначением GAC.

Для features φ(x), kernel k(x,x′)=φ(x)ᵀφ(x′) и
\(\bar k(x,x')=k(x,x')/\sqrt{k(x,x)k(x',x')}\):

\[
\mathrm{FC}=1-\mathbb E_{x,x'}[\bar k(x,x')^2].
\]

Для **m входов, на которых оценивается metric**, Eq. (2):

\[
\mathrm{EFC}=1-\frac{1}{m(m-1)}\sum_{i\ne j}
\frac{k(x_i,x_j)^2}{k(x_i,x_i)k(x_j,x_j)}.
\]

Диагональ исключена. Деление на m² или включение diagonal — другая оценка.
Требуются m≥2 и ненулевые norms/diagonals.

Eq. (3) агрегирует changing features по trajectory:

\[
\mathrm{EFC}_{\rm trajectory}=
\frac{\sum_{t=1}^{T}\mathrm{EFC}(\phi_t;X_0)\,\Delta L_t}
{\sum_{t=1}^{T}\Delta L_t},\qquad
\Delta L_t=\max(L_{t-1}-L_t,0).
\]

Это loss-weighted average, не final/max/unweighted EFC. Если denominator
нулевой, формула не определена: нельзя молча выдавать 0.

§3 для меняющихся parametric features использует
\(\phi(x,\hat\theta)=\nabla_{\hat\theta}\hat f(x,\hat\theta)\);
§4.2/D.2 дают decision-tree kernel как same-leaf indicator. Appendix B.2
указывает per-iteration computation для boosting. Однако **прочитанный текст
не определяет однозначно**, означает ли φ_t именно новый weak tree,
concatenated features текущего ensemble или иные update features в их коде.
Ссылка авторов: [feature_complexity](https://github.com/allerbo/feature_complexity).
Этот linked code в текущий ограниченный intake не загружался и не был
проверен. Не выдавать неразрешённый выбор за точную replication реализации.

### Собственная алгебра для ясно обозначенного C2 proxy

Для одного фиксированного дерева с leaf indicators и m оцениваемыми входами,
из которых m_l попали в leaf l, применение опубликованной Eq. (2) даёт

\[
\mathrm{EFC}_{\rm one\ tree}
=1-\frac{\sum_l m_l(m_l-1)}{m(m-1)}.
\]

Это finite-sample версия pairwise leaf diversity. Замена на
\(1-\sum_l(m_l/m)^2\) без поправки — не точная Eq. (2). При постоянно четырёх
сбалансированных leaves proxy закономерно имеет ограниченную вариативность;
это не свидетельство ошибки метода авторов и не повод требовать от него
измерять finite-label-refit error, другой estimand.

Допустимое название нашего вычисления — **loss-weighted per-tree EFC proxy,
using Allerbo–Schön Eq. (2)–(3)** с явно указанными X_0, loss и φ_t. Точная
same-leaf часть соответствует их DT kernel. Пока linked implementation не
проверен, название «точно воспроизведённый опубликованный GBDT EFC baseline»
не обосновано. EFC всего ensemble, EFC усреднённого kernel и среднее per-tree
EFC вообще не обязаны совпадать.

## Brophy et al.: fixed structure, LeafRefit и estimand

Проверены §2.1, Eq. (3)–(7), Simplifying Assumption 1 и §3.1–3.3, summary
на странице 13, §5.4 и discussion на страницах 22–24. Assumption 1 фиксирует
tree topology и feature splits при исключении training examples. LeafRefit
пересчитывает **все leaf values последовательно**, сохраняя структуру;
LeafInfluence — производная по весу наблюдения с cascade промежуточных
предсказаний. Ни один не равен независимой замене каждого leaf mean при
неизменных старых residuals. Published target — изменение target loss после
удаления/upweighting, а не MSE ошибки отклика на одновременный конечный шум
всех labels. Уже исследуются label edits и structural changes; это не новый
факт C2. Авторы подчёркивают ценность дальнейшего исследования структуры.

В их Eq. (6) shrinkage уже входит в leaf update:

\[
\theta_{t,l}=-\eta\frac{\sum_{i\in I_{t,l}}g_{t,i}}
{\sum_{i\in I_{t,l}}h_{t,i}+\lambda}.
\]

Нельзя повторно умножать сохранённые shrunk leaf values на η. Gradient/Hessian
и λ должны отвечать выбранной loss convention. Формула абстрактного
Newton-step не доказывает совпадение со всеми CatBoost leaf-estimation modes.

### Условия точного C2 контроля — наши требования к реализации

Для наиболее прозрачного squared-loss, без leaf L2, bootstrap и иных
path-dependent weights, leaf averaging задаёт B_t на train и C_t из train
в evaluation rows. По сравнению с unnormalized same-leaf kernel, элемент
B_t содержит **1/n_l**, где n_l — число train-строк в leaf. В reused Kanoh
normalization это K_t/n, а K_t имеет leaf weight n/n_l.

При learning rate η и условно фиксированных partitions линейный replay:

\[
A^{(t)}=A^{(t-1)}+\eta B_t(I-A^{(t-1)}),\qquad
A_*^{(t)}=A_*^{(t-1)}+\eta C_t(I-A^{(t-1)}).
\]

Инициализация — часть оператора: при fit-mean bias
\(A^{(0)}=\mathbf1\mathbf1^T/n\),
\(A_*^{(0)}=\mathbf1_*\mathbf1^T/n\); при строго фиксированном bias его
производная нулевая. Это не новые influence theorems, а линейный контроль
заданной процедуры. Порядок деревьев и новый residual каждого шага сохраняются.

Faithful **conditional leaf-refit control** должен сохранить фактические borders,
leaf assignments, learning-rate convention, bias, float precision, loss,
weights, регуляризацию, число leaf-estimation steps и empty-leaf behavior.
Глобальный least-squares refit на всех terminal indicators — другой алгоритм.
Переобучение новой CatBoost-модели на новых residuals может выбрать новые splits
и не является fixed-structure replay.

CatBoost может преобразовывать targets во float32. Для проверки реакции нужно
использовать реально представимое изменение
\(\delta y=\operatorname{float32}(y+\epsilon z)-\operatorname{float32}(y)\),
а не предполагать точное εz. Это требование нашего численного контроля, не
утверждение о методе статьи. Случаи нулевого δy и numerical mismatch сохраняются;
оператор, не воспроизводящий baseline и frozen refit до объявленного допуска,
не разрешает научно интерпретировать adaptive discrepancy.

Название «LeafRefit-type frozen replay for finite label perturbations» честно
описывает адаптацию. Точная replication опубликованного LeafRefit influence
требует его LOO estimand и проверенной процедуры. C2 не должен сравнивать их
как будто prediction-response error и target-loss rankings — одна метрика.

## Jeffares et al.: сравнение и ограничения

Проверены §3, Eq. (4)–(7), §4.1–4.2 и Appendix B.1. Telescoping approximation
использует последовательность локальных first-order updates. Effective
parameters для evaluation set X_0 размера m имеют вид

\[
p^0=\frac{n}{m}\sum_{j=1}^{m}\|\hat s(x_j^0)\|_2^2,
\qquad \tilde f(x)=\hat s(x)y+c^0(x)
\]

при соответствующем telescoping представлении/условиях; n — число training
labels, а c^0 сохраняет вклад initialization и не должен молча отбрасываться.
Сохранённые trajectory weights не автоматически равны полной производной
по labels при переобучении. Это различие estimands, а не опровержение статьи.
NN telescoping approximation, bias/init и optimizer details нельзя заменить
одним final NTK и назвать faithful reproduction.

В C2 exact frozen operator A_* позволяет вычислить
\(\|A_*\|_F^2/m\) как normalized conditional noise gain. При указанных
условиях p^0/n имеет ту же нормировку. Такую известную связь нельзя продавать
как новый complexity metric или новый закон для адаптивного full refit.

## Wang et al.: честный предел доступа

В intake полного текста или abstract snapshot нет. Официальные full/abstract
URL вернули 403, оба record сохранены. Предыдущий browser scout видел
официальные abstract и passages о modified search degrees of freedom.
Это позволяет указать близкий prior, но недостаточно для protocol baseline,
который претендует на точную реализацию их estimator. Не выводить новый
search-cost theorem и не объявлять full-text approval. Внешняя ссылка —
официальный DOI; обход paywall или использование нелегальной копии не нужны.

## Ограниченный дополнительный scout: насколько нов сам forecast-вопрос

Целевые primary-only запросы об influence trustworthiness, approximation error,
early refit diagnostics и structural sensitivity не дали основания заявить
приоритет нового forecast-метода. Самый близкий подтверждённый antecedent —
[Sharchilev et al., ICML 2018](https://proceedings.mlr.press/v80/sharchilev18a.html):
там уже предложены fixed-structure LeafRefit/LeafInfluence и контролируемые
приближения с trade-off accuracy/computation. JMLR 2023 их расширяет и
показывает несоответствие changing-structure LOO и fixed-structure оценок.
Эти страницы дополнительно прочитаны через web, без новых download snapshots;
они не подменяют четыре запланированных intake sources.

Отличающийся возможный результат C2 должен заключаться в **заранее проверяемом
прогнозе**, когда и насколько finite-label perturbation full refit расходится
с точным условным replay, на независимых contexts/amplitudes и против
ближайших baselines. «Структура меняется», «frozen approximation ошибается» и
«быстрый partial refit иногда дешевле» уже не новый вклад. Совпадение такого
прогноза с опубликованными экспериментами или отсутствие нового predictive
выигрыша означает replication/negative benchmark, а не повод переименовать
известный результат. Приоритет пока открыт.
