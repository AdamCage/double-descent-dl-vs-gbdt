# R13 — производный научный пакет, фигуры и переносимый replay

Статус: **frozen scientific assortment; technical attempt3 preflight pending**.
Автор `/root/workflow_audit` по контракту coordinator `/root`. Attempt1 failed,
attempt2 complete; visual QA нашёл перекрытие числовых tick labels. Явная
поправка AMENDMENT-attempt3.md расширяет только technical retry cap до3 в
пределах родительского cap3. Исправление formatter авторства
`/root/catboost_study`; остальные sources и exact evidence не меняются.
Execution повторно активируется только после отдельного preflight review.

## Вопрос, область и единица проверки

Можно ли выпустить полную воспроизводимую производную поставку уже закрытых
R10–R12: все заранее перечисленные фигуры, самостоятельный научный runtime,
точную evidence/claim/review цепочку и фактический replay R10 после переноса
путей? Это проверка полноты и integrity, **не новый primary scientific test**,
не поиск эффекта и не подтверждение новизны. Операционное закрытие не повышает
`scientific_validity=not_assessed`.

Исходные предсказания R10–R12, положительные, отрицательные и inconclusive
исходы остаются прежними. Нет нового выбора seed, data, окна, LR, random_strength
или test-driven изменения сетки. R10 replay использует прежние seed/протокол и
не является независимой научной репликацией. Единицы delivery-проверки —
файл/полная фигура/полная таблица/связь hash; они не увеличивают исходное n.
R13 завершает производную поставку методологической линии B. Исходная
межмодельная программа механизмов double descent остаётся незавершённой;
продолжение A требует отдельных frozen scientific stages после этого milestone.

## Фиксированные источники и перенос без ослабления gates

Основные attempts:

| Stage | Выбранный attempt | Научный task |
| --- | --- | --- |
| R10 | `20261107T192414-36aee5d99b` | `calibrate-evaluate` |
| R11 | `20261107T194028-229b223740` | `schedule-study` |
| R12 | `20261107T194939-7c18953a98` | `randomness-study` |

Оригинальные attempts read-only. Workflow запрещает mutable
`artifacts/research/**` как source inputs. Поэтому проверенные byte copies
помещены в **учитываемый общим storage ledger** `artifacts/intake/R13/`:
`frozen-runs/<stage>/<attempt>` и `review-packets/`. Figure/package/validation
argv читают эти копии. `inputs/intake-manifest.json` перечисляет original path,
intake path, SHA-256, размер и назначение каждого файла. Эта операция меняет
пути, не числа и не прежние claims/reviews. Inputs перечислены файлами в JSON.

Review packets содержат текущий claim registry и exact decisions/evidence для
всех его accepted claims; методы и отдельное математическое review; авторизации;
операционные records/closures; литературный audit, acquisition plan/log и четыре
снимка первичных web-источников. Первоначальные evidence paths сохранены в
зеркале `review-packets/`, чтобы связи можно было читать относительно него.
Оригинальный первый R11 attempt `20261107T192904-5420604589` сохранён полностью
в `prior-attempts/`: техническое revise, не дополнительное n. Ресурсный ledger
и report скопированы как **снимки через R12**, не как live inputs, которые
изменятся во время R13. Source packet manifest сам не включает свой digest.

R2/R3/R4 — только прежние подтверждающие secondary controls. Их точные
`cells.jsonl` из attempts `20261107T095343-6bc0c457c7`,
`20261107T102511-b1cf6c7be9`, `20261107T104140-5bc2019cef` сохранены в
`review-packets/secondary-controls/<stage>/`. Это позволяет пересоздать все
данные фигуры offline, не обучая старые модели и не объединяя разные оси.

Coordinator объявил рукопись `manuscript/acceleration_audit_v2.md` стабильной
после трёх исправлений по отдельному review. Её SHA-256
`90a02f57759bd976e123732125405f091212f0adc6596ce55d2c6d3e7dda4684`,
review `inputs/manuscript-review.json` от `/root/science_critique` имеет digest
`2360bc0016886189cd6d6167a7876c24328765bd93ffe5f12a7b27765acf9e8d`.
Оба файла скопированы в review packets и привязаны в intake manifest.
Review принимает научную верность ограниченных claims условно на реальную
проверку ещё не созданных R13 assets; полного достижения исходной цели,
приоритета и журнальной готовности не утверждает. Новые изменения текста,
claim registry или источников требуют явно обновлённого intake/review до
запуска, не изменения закрытых attempts.

## Фиксированная последовательность задач

1. **figures**: `scripts/plot_v2_research.py` читает все R10 rates/calibration/
   metadata/raw curves, все R11/R12 analysis/cells и три исторические control
   таблицы. Строит полную галерею: **187 static PNG/SVG pairs и 2 interactive
   HTML**, всего189 figure entries, source CSV tables, index и hash manifest.
   Это полный объявленный ассортимент, не selection показательных seed.
   Кривые train/validation/test и оси сложности раздельны. Min/max обозначены
   наблюдаемыми диапазонами, условные и скорректированные интервалы подписаны.
2. **package**: `scripts/build_science_package.py` копирует целые выбранные
   attempts, весь явный review packet, gallery и итоговую рукопись. В каждом
   `code/Rxx` собственные frozen научные исходники/PLAN/protocol/tests/uv.lock.
   Версии shared modules не склеиваются. Исторические workflow controls остаются
   evidence, но standalone runners их не импортируют. Main-repo `uv.lock`
   честно помечен как provenance с EpistemeOS; устанавливается минимальный
   exact `requirements.txt` из локальной active-base metadata closure при
   `extra=''`, без torch/EpistemeOS и без resolver/network. `verify.py`
   проверяет все bytes, `replay.py` запускает выбранный frozen scientific runner
   с абсолютными packaged protocol/calibration путями в новом внешнем каталоге.
3. **validation**: `scripts/verify_science_release.py` проверяет manifest,
   все187 PNG/SVG pairs и2 interactive HTML, hashes таблиц, PNG читаемость и
   размеры, валидность SVG. Затем проверяет standalone inventory, реально
   запускает frozen R10 в новом внешнем task subdirectory и сравнивает
   **все7188×256 raw loss values точно**, JSON analysis/calibration и curve
   metadata с intake оригиналом. После replay повторно проверяет пакет.
   Model fits=0; это повторный расчёт синтетических рядов, не model training.

Все три child argv выполняются через frozen
`inputs/run_derivative_task.py`. Он устанавливает OMP/MKL/OPENBLAS/NUMEXPR=4,
CUDA empty, отключает pyc и использует локальный matplotlib cache. Сохраняет
exit/elapsed/thread controls и storage check в `task-runtime.json`. Таймаут,
process-tree termination и project lock принадлежат основному workflow.

## Предсказания, фальсификаторы и критерии выхода

Это delivery predictions, сформулированные до производного запуска, а не
новые пререгистрированные научные гипотезы.

- P1: полный manifest содержит189 entries, именно187 static pairs и2 HTML;
  все объявленные raw sources и source tables связаны hashes. Фальсификатор:
  пропуск/дубликат, несовпадение hashes, нечитабельный PNG/SVG или другая count.
- P2: package inventory проверяется, frozen stage source versions раздельны,
  минимальный runtime не требует EpistemeOS/torch, claim/review provenance и
  первоначальный технический R11 attempt сохранены. Фальсификатор: любой
  пропуск/перезапись, изменённый source hash, shared-code merge или dependency leak.
- P3: фактический standalone R10 воспроизводит всю raw матрицу и численный
  analysis точно и оставляет package неизменным. Фальсификатор: любой отличающийся
  элемент/JSON/hash, failed child process или последующий inventory mismatch.

C1: все источники/фигуры/таблицы полны, runtime controls записаны.
C2: standalone runtime минимален и сохранил полную выбранную evidence,
claim/review/литературную цепочку и technical retry без повышения n.
C3: фактический portable R10 smoke прошёл на всех исходных значениях, пакет
не изменился. Каждый критерий оценивается отдельным operational review;
успешный renderer exit сам по себе недостаточен.

## Бюджеты, остановка, ошибки и handoff

≤1800s attempt, ≤3600s суммарно, максимум3 attempts по явной технической
поправке AMENDMENT-attempt3.md (первоначально2). Task timeouts:
figures900s, package240s, validation480s. Один dispatcher/task за раз, CPU≤4,
GPU/API/network/download=0. Новые outputs текущего attempt ≤8GiB; wrapper
проверяет cap на границах задач, это не live OS quota. Пакет дополнительно
ограничен8GiB/50000 entries своим builder. Global24h/20GiB не обнуляются;
R13 intake копии учитываются `resource_ledger.NEW_ARTIFACT_ROOTS`.

Остановка: один проход fixed delivery matrix либо сохранённый technical
failure/таймаут. Никаких новых обучений ради фигуры; никакой адаптации научных
данных или контрастов. При техническом исправлении сохраняются все успешные
и неудачные tasks, новый attempt получает новый ID и проходит preflight.
При воспроизводимом numerical mismatch — сначала diagnosis frozen paths,
dependencies и кода, затем evidence escalation при unresolved integrity.

После run: verify → REPORT со всеми predictions/criteria → отдельный bound
review → close → exclusive evidence archive и detached archive verify → STATE
с точными путями галереи/рукописи/пакета и следующей командой. Внешней
публикации, рассылки и утверждения журнальной готовности здесь нет.
