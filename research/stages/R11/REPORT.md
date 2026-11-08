# R11 — парные расписания и часы оптимизации

Основной attempt `20261107T194028-229b223740`, plan SHA256
`33328abb7ecf995c370af4cc3ded98ba79597f20e280a78b317ba20caec4afb4`, manifest
`e1ff4eb1c8ad4f6cae4026906e59f47a7f48d4765d70a6f091e6d7a11525e66b`.
Verify: true,83 files. Авторы реализации `/root/catboost_study` и `/root`;
автор отчёта `/root`, отдельный reviewer `/root/science_critique`, harness codex.
Scientific validity not_assessed. Evidence: task `schedule-study/` этого attempt.

## Первая попытка и исправление

Attempt `20261107T192904-5420604589` завершён и сохранён, время196.67200000025332s.
Отдельный review=revise: fit ограничен4 потоками, inference оставлен с default-1.
Фактический перерасход не измерялся; гарантия не была обеспечена исходником.
REPORT-attempt1.md и scored review сохранены. После явных ограничений всех
staged_predict/predict и повторного preflight получен новый attempt. Сетка,
данные и seeds не менялись, n осталось12; это технический повтор, не независимая
репликация. Числа первой попытки не выбирали ни исправление, ни основной исход.

## Критерии и полнота

C1: ровно72 cells,2 окружения×12seed×3расписания, каждая модель256 деревьев,
каждый split257 raw timepoints; все72 CBM и data/predictions NPZ сохранены.
unit-checks/checks.json:9 tests, failures/errors0, включая opt-in tiny smoke.

C2: data/split hashes совпадают внутри каждой пары; для каждого fit проверено
final staged_predict=model.predict. t0 — fitted bias, проверенный по train
prevalence logit; nonzero bias и prefix continuation проверены technical smoke.

C3: analysis.json сохраняет4 primary и4 LR-clock diagnostic contrasts,
каждый со всеми12 paired values, SE, Student-t95% и Bonferroni family4 CI.
Две семьи4 не составляют совместную семью8 с95% гарантией. Сохраняется один
inconclusive primary outcome. Ни окно, ни число моделей после результатов не менялись.

C4: CPU4 явно в fit/inference, GPU/API/download0. Test не выбирал настройки.
double_descent_declared=false, natural_acceleration_mechanism_identified=false,
synthetic_null_pvalues_transportable=false. Никакого human approval.

## Предсказания

Среднее paired gain(schedule)−gain(constant), единицы LogLoss/дерево.
Интервалы — family4 Bonferroni при t-приближении, не произвольная гарантия.

| Окружение | Контраст | Mean | Adjusted CI | Исход |
| --- | --- | --- | --- | --- |
| synthetic | stepup−constant | .00425687913 | [.00319455579,.00531920246] | supported |
| synthetic | stepdown−constant | −.000841410484 | [−.00120546082,−.000477360148] | supported |
| breast_cancer | stepup−constant | .00166257643 | [.00137099688,.00195415598] | supported |
| breast_cancer | stepdown−constant | −.000135930481 | [−.000283184142,.0000113231788] | inconclusive |

P1 поддержано для обоих окружений. P2 поддержано только synthetic;
breast_cancer inconclusive, хотя обычный нескорректированный95% интервал
полностью отрицателен. Выбор более удобного интервала после результата запрещён.
P3 — полная matrix, парность и отсутствие test-driven выбора — выполнено.

## Заранее заданная диагностика часов

На тех же32 точках OLS по sum learning rate, в единицах LogLoss/единицу LR:

| Окружение | Контраст | Mean | Adjusted CI family4 |
| --- | --- | --- | --- |
| synthetic | stepup−constant | −.0241218336 | [−.0337149664,−.0145287007] |
| synthetic | stepdown−constant | .00154323688 | [−.000629270933,.00371574470] |
| breast_cancer | stepup−constant | −.0513655804 | [−.0535267323,−.0492044285] |
| breast_cancer | stepdown−constant | .000932005468 | [−.000419365813,.00228337675] |

У stepup положительный контраст на tree-clock и отрицательный на LR-clock
в обоих окружениях. Численные величины разных clocks имеют разные единицы;
их нельзя вычитать или трактовать как одинаковые effect sizes. Знак сравнивает
одноимённые schedule-minus-constant contrasts внутри соответствующих часов.
Часы sum LR не являются доказанными точными часами квадратичной модели.

Причинное вмешательство — полное расписание, включая разную предысторию до128.
Это не изолированный скачок при одинаковых первых деревьях. Равенство256
деревьев и sum LR28.16 не делает деревья, градиенты или margin path одинаковыми.
Результат подтверждает необходимость указывать часы, но не механизм естественного
позднего ускорения и не двойной спуск.

## Пределы и продолжение

Два малых окружения, одна библиотека и глубина;12 новых синтетических выборок,
а у cancer12split/model повторов на одном корпусе. t-интервалы приближённые.
Margin/gradient/validation/test trajectories описательные, не mediation analysis.
Сырой test доступен в frozen outputs; он не менял дизайн. Контроль расписания
не заменяет исследование естественной динамики. R12 имеет новый frozen протокол
с постоянным LR, одним непрерывным fit и вмешательством random_strength.
