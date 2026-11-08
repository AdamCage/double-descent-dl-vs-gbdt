# R14: отдельное design review до исполнения

Дата: 8 ноября 2026. Reviewer: Codex-субагент `/root/literature`.
Автор scientific plan: `/root`; technical helper: `/root/catboost_study`.
Проверенная версия `PLAN.md`:
SHA-256 `1b0ae5047fa82c135054beb5dcded16156de3d1646e6bfeaa57482791457c7f3`.

**Вердикт: revise до научного исполнения.** Два содержательных пункта ниже
требуют решения в тексте до freeze. Это не запрет на техническую подготовку
или на корректный отрицательный результат. Execution сейчас не авторизуется.

Ограничение независимости: reviewer написал литературный scout, source cards
и code note; он не автор конкретного R14 forecast, runner или implementation.
Это отдельная локальная design-рецензия, не внешнее peer review. Рецензия
научного вывода после результатов должна выполняться другим назначенным
агентом; собственные source-derived утверждения здесь не получают независимой
подписи этого же автора.

## Что в дизайне научно корректно

Estimand чёткий: out-of-sample discrepancy полного label-refit и точного
условного fixed-partition replay, а не teacher prediction error, LOO ranking,
общая extra variance или double descent. Сохраняется signed cross term.
Actual float32 label changes отделены от nominal εz; известная линейная
Frobenius формула не объявляется точной для округлённого perturbation law.

Парность организована внутри data draw. Holdout исключает все teachers,
prefixes и amplitudes draw вместе. Pilot directions и response directions
разделены; использование unlabeled evaluation covariates честно указано.
Фиксированные ridge/standardization и P2 не выбирают красивый subset после
endpoint. 64 context/eps rows не названы 64 независимыми повторами; нет
псевдоточных p-values. Четыре draw дают discovery, не confirmation.

Go зависит от replay correctness, predictive gain, challenge range и реального
wall-cost; failure останавливает этот кандидат. Предусмотрено сохранение
технических сбоев, ограниченные retries и независимый R15 при положительном
discovery. Ни наличие risk peak, ни поддержка предсказания не нужны для
честного завершения R14. Связь с исходной программой содержательная, хотя
это уже узкий вопрос о надёжности объяснения бустинга, а не завершённый
межмодельный механизм double descent.

## Blocking B1: comparator с тем же ранним измерением

Все B0/B1/B2 не используют C, тогда как A-model добавляет раннее измерение
того же типа discrepancy на 32 шагах. Положительный P2 в текущем виде покажет,
что такое раннее измерение полезно против baselines без него. Он не покажет
ценность полного geometry/EFC pipeline по сравнению с простым cheap-only /
partial-target forecast и не установит, нужны ли дорогие A512/operator features.

Для заявленной научной/практической ценности следует **до вычисления** либо
зафиксировать соответствующий простой early-probe comparator и его роль
в stop/go, либо сузить claim до feasibility/ценности самого probe и честно
отказаться от более сильной заявки на новый forecasting pipeline. Выбор
конкретной модели, feature set и правила остаётся за автором; reviewer не
подменяет его новым forecast в плане. Параметры нельзя выбирать по R14 endpoint.

Это существенный контроль, а не требование большей инфраструктуры: основная
опасность C2 — переименовать известную идею partial refit / approximation
trade-off в новый научный вклад. Primary literature уже включает LeafRefit,
FastLeafRefit и empirical evaluation их приближений. Последующая независимая
confirmation важнее добавления ещё одной post-hoc корреляции.

## Blocking B2: исход для invalid directions

Текст сохраняет нулевой actual δ как invalid direction и запрещает replacement,
но не задаёт, как после этого вычисляется context D/C, MAE и итоговый go/no-go.
Нельзя по умолчанию усреднить оставшиеся направления, удалить context или
подставить 0: изменятся endpoint и повторная единица. Нужен заранее заданный
terminal/invalid outcome и его влияние на scientific interpretability и
stop/go. Это касается также любых иных сохранённых invalid endpoint rows.

При заявленных amplitudes такой случай может быть редким; именно поэтому
его обработку легко определить до открытия evidence.

## Обязательные ограничения будущего вывода, не новые blockers

- Все LODO folds содержат оба teacher в train. Это перенос на новые draws
  тех же двух generators, а не доказательство transfer на новый teacher,
  реальные таблицы или иную конфигурацию CatBoost.
- B2 содержит explicit conditional-operator geometry, но не полную replication
  всех kernel baselines Kanoh. Победа над B2 не опровергает reference-kernel
  theory и не доказывает превосходство над всеми published complexity metrics.
- P3 должен показывать измеренную стоимость common basefit/operator отдельно,
  весь дополнительный probe/response analysis и стоимость обучения/применения
  forecast. Нельзя переводить число деревьев в универсальный speedup или
  прятать стоимость получения discovery targets.
- Native replay smoke должен проверять применимые solver semantics на нужной
  длительности и сохранять точные versions/settings; mismatch — technical
  failure. Положительный toy test на пяти деревьях не заменяет gates на 32/512.
- Before freeze добавить точные machine protocol/matrix, source closure,
  `uv.lock`, helper, runner, meaningful tests и документированные источники.
  В частности, включить `C2-author-code-note-2026-11-08.md`, если используется
  подтверждённое там per-new-tree interpretation EFC. Wang full text недоступен:
  его нельзя объявить реализованным faithful baseline.

Scientific draft может быть принят после явного решения B1/B2 и повторной
короткой проверки изменённой версии. Machine freeze и закрытие R13 остаются
отдельными необходимыми preflight gates. Отрицательный исход или отсутствие
новизны после корректной проверки не являются основанием менять сетку.

## Повторное review стабильной версии — актуальное решение

Дата: 8 ноября 2026. Текст и digest получены из **одного чтения bytes**.
Проверенный `PLAN.md` SHA-256:
`f1f75de19972dfce2e59f5813e5ef869948b389306c8d9e1d77cfc7c7a09f4d3`.

**Актуальный вердикт: accept scientific discovery design. Blocking замечаний
к этой версии плана нет.** Это принятие дизайна, не scientific validation,
не independent confirmation, не утверждение приоритета и не самостоятельная
авторизация запуска. Freeze/source integrity, закрытие R13, отдельная проверка
runner и execution controls остаются необходимыми.

Историческая оговорка: digest в начале первого review был снят отдельным
чтением после первоначального чтения текста во время concurrent draft edits.
К моменту снятия того digest zero-δ policy уже обновлялась. Поэтому первое
описание B2 не следует трактовать как точно bound blocker версии с тем
digest. Начальная дискуссия сохранена; актуальная оценка связана с указанным
здесь атомарно прочитанным snapshot.

B1 разрешён: зафиксированы B3 с ранним C без expensive512 geometry/EFC и B4
с заранее заданным unit slope и training-fold-only offset. P2 требует выигрыша
над каждым B0–B4 и сохраняет fold/teacher сравнения против каждого B2/B3/B4,
без выбора удобного comparator по endpoint. Конкретные baselines определил
автор после критики reviewer; reviewer не писал forecast или runner.

B2 разрешён: нулевой actual δ остаётся в полной matrix без replacement;
OOF/decision не вычисляются, scientific outcome inconclusive, go=false.
Perturbation строится от actual base labels, pair rounding residual объявлен.

Practical scope сужен до conditional audit уже полученной модели/operator;
common costs показаны отдельно, end-to-end ratio secondary. P3 остаётся
измерением стоимости, не доказательством точности cheap-only forecast.
Непрохождение A-model gate не должно превращаться в новый положительный
scientific claim о B3/B4 без отдельного заранее заданного критерия и review.

Сохраняются существенные пределы: всего четыре независимых draw, оба teacher
присутствуют в обучающих folds, target — finite-response approximation error,
а не teacher risk. Нет подтверждения переноса на новые generators, реальные
данные или arbitrary CatBoost. Даже поддержка P2/P3 здесь разрешает только
проверку нового замороженного confirmation-протокола; заявка на научную новизну
остаётся открытой и должна пройти отдельное содержательное review.
