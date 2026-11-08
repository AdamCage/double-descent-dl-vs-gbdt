# Завершено 2026-11-08: подтверждённый H2-аудит и готовое портфолио

Это действующее состояние. Все записи ниже разделителя «История» сохранены
как журнал и не задают очередь запуска. Все 20 зарегистрированных этапов
R0–R18, включая R17D, операционно закрыты без closure errors; `next` возвращает
`recommended_action=complete`, `next=null`, `blocking_stages=[]`.
Активного dispatcher нет. Последний принятый запуск R18:
`20261108T111319-cf1d82f2a8`. Новые эксперименты не запланированы.
Автоматическая цель ранее снята пользователем и не создавалась заново.

После закрытия научных stages пользователь отдельно авторизовал commit, push
и GitHub Release. Материалы доставки и точные SHA256:
[release notes](../docs/releases/research-cot-audit-2026-11-08.md),
[asset manifest](../docs/releases/research-cot-audit-2026-11-08-assets.json),
[GitHub Release](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/tag/research-cot-audit-2026-11-08).
Git сохраняет точные source/control bytes; большие новые архивы и live/intake
выходы распространяются в Release assets. CI для source checkout явно
отмечает невосстановленные evidence как unverified; полный локальный gate
`scripts/check_repository.py` не ослаблен.

На ранее не использованных для выбора дизайна confirmation outcomes H2
подтверждён: 16 из 56 поддержанных сравнений (9 BBH, 7 MMLU) прошли заранее
заданный критерий. Максимальный сдвиг межмодельного разрыва — 15,52 п.п.
Все 28 GPQA-сравнений не имеют достаточной поддержки. Вывод относится к
публичной исторической judge-ACK аннотации и изменению популяции сравнения;
это не доказательство причинной faithfulness или истинного рейтинга моделей.
Общая проблема selection/matching известна. Внешний приоритет конкретного
аудита и готовность к сильному журналу не установлены. Новый механизм
двойного спуска в прежнем MLP/CatBoost-исследовании не установлен.

Готовые материалы:

- [Английская статья, 9 страниц PDF](../artifacts/research/R18/20261108T111319-cf1d82f2a8/tasks/package/portfolio/manuscript.pdf).
- [111 рисунков, каждый PNG/SVG, и 8 CSV](../artifacts/research/R18/20261108T111319-cf1d82f2a8/tasks/package/portfolio/gallery/index.html).
- [Переносимый ZIP](../artifacts/research/delivery/cot-audit-portfolio-2026-11-08.zip)
  с исходниками, данными, обоими replay, протоколами и review receipts.
- [Рукопись](../manuscript/cot_input_comparability.md),
  [описание для портфолио](../docs/portfolio/cot-audit-overview.md),
  [итоговая передача и восстановление](stages/R18/HANDOFF-2026-11-08.md).

Проверки: 239 unit tests PASS, 9 opt-in fitting tests SKIP. Все 12 полных
сравнений файлов двухфазного portable replay совпали без численных расхождений.
Отдельные reviewer-агенты проверили научные выводы, все 111 PNG/SVG и все
9 PDF-страниц. Первая визуально отклонённая сборка и все научные неудачные
попытки сохранены. Три H2 claims допущены отдельно; всего в реестре 13 claims.
Локальное team review не выдаётся за внешнее рецензирование. Машинное поле
`scientific_validity=not_assessed` не переименовано в человеческое одобрение.
35% reserve outcomes остаются неиспользованными; общая metadata проверялась.
Платные API, GPU, обучение и новые model calls не использовались.

Архивы проверены отдельными процессами без чтения исходных package-файлов:

- Portable ZIP: 125071397 bytes, SHA256
  `3de3f516eaa2edffdae4d76fdc85cbbd5079bef5d1eefb081acd9dd372639491`;
  все 324 файла полностью восстановлены и проверены.
- Оба R18 attempts: `evidence/R18-runs-2026-11-08-v1.zip`, SHA256
  `e6b792879d2d4100d7daf35ca618f5b49e58b8e8de56d3e0f3ce84790479181e`;
  все 473 blobs / 1347 paths проверены, пять файлов восстановлены отдельно.
  Повторное полное размещение 603 MB workflow-дерева не выполнялось.
- Независимые review/QA proofs: `evidence/cot-review-proof-2026-11-08-v1.zip`,
  SHA256 `ac51c26f76f141e7295b1051cbdfdb541cff29d784240728695f93ffb5fb8105`;
  все 812 members проверены, семь proof samples восстановлены.

Точные receipts: [release](evidence/R18-release-2026-11-08.json),
[workflow archive](evidence/R18-archive-2026-11-08.json),
[proof archive](evidence/cot-review-proof-2026-11-08.json),
[closure](evidence/R18-closure.json),
[финальные status/next](evidence/R18-final-workflow-status.json).
Четыре ранних неисполненных R17 planning variants не сохранились как полные
байты; это явно отражено в proof receipt. Все реально исполненные inputs,
source snapshots и независимо привязанные численные/QA proofs сохранены.

Следующая команда — только проверка завершения:

```powershell
uv run python scripts/research.py next
```

При необходимости read-only проверки итогового запуска:

```powershell
uv run python scripts/research.py status
uv run python scripts/research.py verify R18 --attempt 20261108T111319-cf1d82f2a8
```

PDF operation marker уже выполнен ровно один раз; при чтении/восстановлении
его не повторять. К моменту закрытия R18 внешней публикации и отправки заявки
не было; последующая доставка на GitHub разрешена отдельно и описана выше.
Подачи статьи в журнал или заявки на residency не выполнялось. Дальнейшее
независимое исследование на другом benchmark/crossed targets с проверенными
judge labels потребует нового замороженного плана; это открытая перспектива
для журнальной работы, а не незавершённый этап текущей поставки.

---

# История — завершённые состояния, не инструкции к запуску

## R18 attempt1 REVISE frozen; technical attempt2 in preparation

Attempt1 `20261108T104100-7836d70ed3` complete278.735s; all111PNG/SVG/8CSV data
and12portable replay comparisonsPASS, all9PDFpages visuallyPASS. IndependentQA
found bottom N/A annotation overlaps ticks in shared56pair panels. Operational
REVISE imported before source change. Preserve all successful old task data.
Renderer author now fixing ONLYcategoricalaxis padding plusbbox regression.
R18execution=false. Wait finalsource -> updateportableplot/test copy/origins/
evidence-spec-attempt2 +CI -> distinctdesign-review-attempt2 ->runONEremaining
attempt2. No new hypothesis/data/metric/seed/threshold, reserveunused.
PDFmarkerALREADYRAN exactlyonce: NEVERrepeat onretry. Same13acceptedclaims,
mainmanuscript230a andfourpaperplots/captions unchanged. Deadline13:12UTC.
Noactive dispatcher. Alloldstatusesbelowarehistory, notqueue.

# LIVE R18 delivery - DO NOT RESTART OR REPEAT PDF MARKER

Attempt `20261108T104100-7836d70ed3`, unified session1776, running. Production
figures COMPLETE:111PNG/SVGpairs+eightCSV. Single detached replay is running;
discovery sixfilecomparisonsPASS, confirmation currently executing. Package/PDF
will run serially afterreplay. Boundpreflight83af79a5ac7142451bc520f30f13b7118caf0ea8c73db1594c8577477f2642fa accepted;
PDFskillmarker alreadyexecutedEXACTLYONCE, log artifacts/intake/R18-pdf-operation-started.json.
DO NOT repeatmarker orstartanotherdispatcher. Observeexistingmanifest/session.
Allsource/evidence/protocol inputs frozen; do notedit untilresultreview/close.
Independent /root/cot_visual_reviewer handles everyPDFpage/all111PNG+SVGvisualQA;
/root/reset_critic handles completeCSV/source/replay/package numericQA.
Parent has visually inspected allfouractualpaper_*PNGs, no layoutissuefound.
Next: observe completion -> verifyR18 -> REPORT -> actual independentREVIEW ->close ->
archive+detachedverify+normalportableZIP -> finalSTATE/README anddeliverylinks.
SupportedH2/unusedreserve andscientificlimits belowremain unchanged.

# Current delivery status - confirmed H2, R18 preflight

2026-11-08: R15/R16/R17D/R17 operationally CLOSED. Final H2 confirmation
`20261108T100025-c526dea890` supported the frozen criterion:16/56eligiblecells,
9BBH+7MMLU, largest15.5207pp; GPQA28unsupported. Distinct reviewer reproduced
all88772perquestionrows/all5000draws/allintervalfamilies. Three bounded H2
claims admitted; registry13includes10olderclaims. All84versioncomponents0
underobservedstructure, NOTcausalversioneffect/truefaithfulness/rankingproof.
H1anditsfailedattemptandunexecutedconfirmationplan fullypreserved. R17D uses
exploratoryH1questionreuse; freshR17confirmation was unusedbeforeH2. Reserve35%
ACK/factor outcomes remain unused. PaidAPI/GPU/training0; deadline13:12UTC.

Scientific manuscript finalcandidateSHA230a21f145ade23000f43ac82aa913d2459d81dca014ce55395fc44ac70ebf2e,
acceptedscientificreviewv2. Literaturemechanismalreadyknown; specificempirical
case-studyvalue supported, externalpriority/journalreadiness NOTestablished.
R18 registeredexecution=false, machineSHA2634aabfaa4b7e6615fa92d15acb411e774ebc9bfbc91ee4094d6f6bf348c026.
Preflightpacket:111PNG/SVGpairs/eightCSV, fourlarge-fontpaperfigures, single
portable two-phase replay, EnglishHTML/PDF bundle. Allsourcesfinal;237repo tests
passed(skip9). Reviewer binding finalR18design-review.json next, thenactivate.
NO productiongallery/detachedreplay/PDF hasrun yet; NOartifactmarker hasrun.
BeforefirstR18authoring dispatch, run PDFskillmarker exactlyonce andrecordit.
Do not duplicate marker afterresume; inspect artifacts/intake/R18-pdf-operation-started.json.

Commands after acceptedpreflight andactivation:
```powershell
uv run python scripts/research.py validate-plan R18
uv run python scripts/research.py run R18
```
Historical evidencecheck (currentregistryhasaddedR18):
```powershell
uv run python scripts/research.py verify R17 --attempt 20261108T100025-c526dea890 --historical
```
No active dispatcher. CompleteR17D/R17archives separatelydetachedverified.
Older status entries below are historical and do not replace this header.

Обновление: R16attempt2 `20261108T093900-48ff41116f` COMPLETE/verify11files.
H1narrowdiscovery:57supportcells/25validCI/material0; BBH/MMLU maxshift<.0022.
НЕgeneralrobustness: cueTEXTcontainsanswer, U0alreadyconditionsontarget; H1
неисходныйnativeadaptivecomparison. ОригинальныйR17H1НЕзапущен иbytepreserved
вresearch/history/R17-H1-unexecuted-planning-2026-11-08/PRESERVATION.json.
Текущийшаг: independentR16resultreview -> close, ДОизмененияanalysiscode/registry.
НоваяH2(R17D exploratorydiscovery +R17freshconfirmation) context=scopeONLY.
Decision0007/childcharter76eb82a9b82c8ad0c7832d0c55b7d3e445e9f412579e57390aca56f7c1bc66cd.
Не запускатьстарыйR17. ОбеH2постановки/sourceпредварительноfreeze/reviewдоH2.
Reserve35% ACK/factorgradesне использованы. Deadline13:12UTC,API/GPU0.

Обновление09:23UTC: R15 operationally CLOSED и archive detachedverified. R16
attempt `20261108T091936-7fa306e46e` FAILED на metadata consistency ДО ACK
aggregation (0.094s). Никаких научных эффектов пока нет. Failed evidence сохранено.
Сейчас source diagnosis + полный metadata-only audit перед reviewed amendment
о quarantine целых конфликтных baseline-union clusters. Это technical integrity
repair, не отрицательный H1 и не разрешение обхода gates. R17 не запускался.
Не повторять run R16 до нового bound preflight. Deadline13:12UTC сохраняется.

# Текущий приоритет — закончить alignment portfolio сегодня

Обновление 2026-11-08 09:12 UTC. Пользователь дал ещё четыре часа: доставка
до **13:12 UTC / 16:12 Москвы**. Новые гипотезы разрешены при неудаче прежней;
все попытки раскрываются, критерии не подгоняются. Paid API/GPU/training=0.
Автоматическая goal-сессия очищена пользователем, новая не создаётся.

R15 engineering intake COMPLETE: `20261108T090628-8efd502cb8`, 79.703s,
24/24 payload hashes, 940116134 network bytes, 446208 rows, slim88613869bytes.
Никакие scientific ACK rates не считались. Accounting log зеркалирован.
Текущий шаг: verify/review/close R15, freeze R16/R17 before outcomes.
Новый заранее выбранный split: discovery30%, confirmation35%, reserve35%;
reserve outcomes не читаются без отдельного reviewed плана новой гипотезы.
Reviewer `/root/reset_critic` независимо принял этот design tradeoff до outcomes.
GPQA при minimum100 confirmation может быть только descriptive; порог не снижается.
Старая DD/CatBoost ветка R0–R14 закрыта; положительный новый механизм не установлен.

Точные текущие команды:
```powershell
uv run python scripts/research.py verify R15 --attempt 20261108T090628-8efd502cb8
uv run python scripts/research.py review R15 --prepare research/reviews/R15-v1-draft.json --report research/stages/R15/REPORT.md --attempt 20261108T090628-8efd502cb8
```

Исторические записи ниже сохраняются; их статус/очередь не заменяют это обновление.

# Состояние на 8 ноября 2026 — основной научный результат ещё открыт

**НОВЫЙ ПРИОРИТЕТ по прямому уточнению пользователя:** alignment portfolio
для Anthropic Residency; отдельно — статья для аспирантуры в хорошем журнале.
Решение0006 и draft R15–R18: docs/research-reset/PORTFOLIO-PROGRAM.md.
Старая DD/CatBoost программа R0–R14 закрыта и сохраняется как bounded audit;
новый механизм/полное первоначальное исследование не объявлены достигнутыми.
GBDT-certificate continuation отложена. Новые numerical runs не начинались.
Автоматическая goal-сессия очищена пользователем, новая не создавалась.
Профиль пользователя, срок его подачи и бюджет запрошены; сейчас разрешена
read-only подготовка. Paid API/GPU остаются0. R15–R18 пока design-only,
не machine queue: сначала prior/data/resource gate и один frozen pilot.

**РАБОТА ВОЗОБНОВЛЕНА пользователем.** Предыдущая пауза снята. R13 закрыт,
галерея/standalone package сохранены. Создание общего архива session7602
прервано на проверенном worker PID25068; частичный
research/evidence/R13-runs-2026-11-08-v2.tar.gz сохранён и НЕ verified.
Не перезаписывать/не считать этот архив готовым: detached verify установил
EOFError, запись действительно неполна. После
resume status/verify подтверждают closed R0–R13, R131315files intact,
старого archive worker/active dispatcher нет. Создан и отдельно проверен
полный content-addressed ZIP всех3attempts:4449paths/1358blobs,
791114717bytes, SHA256dcf9c43d990677e64b2408f76d97557267459bc2e3a3f5a2fc62dbce3bc38324.
Handoff:research/stages/R13/HANDOFF-2026-11-08.md. Старый failed tar сохранён.
R14 после независимого preflight активирован; текущий attempt указан ниже.
После resume завершены незаконченные sources/fixture tests и требовавшийся
fresh32-fit vs512-prefix32 control на nonstudy toy inputs. Final preflight
проверил все24bindings и принят до генерации scientific data.

Прямой запрос пользователя и расширение цели разрешили новые stages и работу
Codex с отдельными субагентами. Решение: docs/decisions/0004-v2-research-refactor.md.
Общие бюджеты не обнулены. Человек не одобрял результаты задним числом.

R10 закрыт: attempt `20261107T192414-36aee5d99b`, review
`8240fd32643039dec1c0f606d3b89610f256f6de0f91d9c163211091522eeb24`.
7188 траекторий, отдельные calibration/evaluation. P3 опровергнуто;
local baseline отметил все2400 null eval curves. Слабая power scan при большом
шуме и условный FPR10% в одной ячейке сохранены. Claim и review привязаны.
Архив research/evidence/R10-runs-2026-11-07-v2.tar.gz проверен отдельно,
SHA256 `61e5fd8a15ba77a20527b21470b27d7fb925f693db3b6555e4a237ccb04a1cca`.

R11 закрыт на принятом attempt `20261107T194028-229b223740`; первая техническая
попытка сохранена с revise из-за неявного inference thread_count. Парное
изменение learning-rate schedule даёт ускорение на оси числа деревьев, но
обратный знак контраста на оси накопленного learning rate. Это ограничение
интерпретации кривой, не новый механизм double descent.

R12 закрыт: `20261107T194939-7c18953a98`. Random strength меняет scan-score,
но ни одна из шести заранее заданных late-event gates не установлена.
Это не доказательство отсутствия всех поздних ускорений.

R13 — производный выпуск, НЕ завершение исходной цели. Attempt1
`20261107T210201-651abc88d9` failed и сохранён. Attempt2
`20261107T210937-844511dd10` complete и verify passed: 187 PNG/SVG pairs,
2 offline HTML, 16 исходных CSV, standalone package и точный R10 replay.
Визуальная проверка нашла перекрытие малых десятичных ticks в R12 gate plot.
Явный AMENDMENT-attempt3.md разрешил
cap2→3 в пределах родительского cap3; общий wall/storage cap не изменён.
Отдельный preflight принял formatter correction и 5 regression tests.
Третий attempt `20261107T214015-f184806e46` complete147.156s и ЗАКРЫТ:
live verify1315files, отдельный review C1–C3 pass. Все187 PNG просмотрены в
7 обзорных листах и9 ключевых отдельно; ограничения QA сохранены. Frozen
review SHA256 `0f296dc1498f817649a4cb383a4f3d3144239d5841d8e7a7bc84c125b88840f8`.
Активного dispatcher нет. Архив всех3attempts проверен отдельно:
research/evidence/R13-runs-2026-11-08-deduplicated-v3.zip;
closure отдельно research/evidence/R13-closure.json.

Следующие команды только читают состояние:

```powershell
uv run python scripts/research.py status
uv run python scripts/research.py verify R14 --attempt 20261108T073937-26f816e66d
```

R14 COMPUTATION COMPLETE: `20261108T073937-26f816e66d`,521.766s, все3tasks
exit0, live verify1119files passed. Dispatcher terminal, active.lock отсутствует.
Нового run/attempt не требуется: machine ready/execute пока означает только
готовые зависимости; сейчас требуется result review существующего attempt.
193 repository checks passed, final independent preflight accept
SHA4fef9061ce3530f0d2be3e60a119904fc8e36d6d9e479fbc60ab3ec20b9c319c,
machine plan2cdb3c1411ea7155a3a89c0abc1bfe8b4f58e85b9c8ae3a552347cdc336f41df.
416fits/32contexts/4independent draws, all-baseline20% gain иcost stop gate;
113PNG/SVG pairs/7CSV при complete OOF,89/6 при invalidδ.
Technical controls: fresh32fit иprefix32 полного512 точно совпали;512-replay
max error7.77e-16 приatol1e-10/rtol0. Не интерпретировать незавершённый attempt.
REPORT.md готов. P2>=20%gain refuted: A MAE.242558701584 противB3.250140933179,
gain3.031%; go=false,candidateSTOP. P1exactcontrols pass. P3positivecomputed
probe ratio.15062 не доказывает дешевизну всегоforecast: EFCtrajectory timer
смешан сbaseprediction, broadercostNOT_ESTABLISHED; coldcostratio1.0798.
113PNG/SVGpairs и7CSV созданы. Итоговый review отдельного
`/root/r14_final_review` принят после повторного all-raw/all416-native-model
audit и visualQA113overview/9fullres; C1–C4pass для negative/inconclusive
assessment, не positiveP3method claim. R14операционно ЗАКРЫТ;
frozen reviewSHAce07b5ff5c936f729113ec5945420b4c4e3545b8899c28aa4ab176680449a5b1.
Архив research/evidence/R14-runs-2026-11-08-v1.zip отдельно verified,
1152files/160306762bytes, SHA256
fc150cb12bb8c26878c6eca6f8949f1bc6d769c6a96da45477cbd8dbbb676af0.
Closure отдельно research/evidence/R14-closure.json.
Confirmation старого forecastA запрещена go=false; новые R15–R18 обозначают
другую design-программу по0006, а не разрешение повторить A.
Проверяется прогноз ошибки frozen approximation при конечных label perturbations.
Приоритет и полезность ещё не установлены;
«структура деревьев меняется» и conditional noise gain уже известны.
Нужны независимый дизайн-review и отдельная confirmation на новых данных.

Новый механизм, сравнимый межмодельный результат и публикационная готовность
не установлены. Описательный double descent есть в известном min-norm linear
контроле R2; для MLP/CatBoost убедительного подтверждения нет. Ниже — история,
а не действующая очередь. Старые нерешённые authorization-вопросы заменены
прямой авторизацией пользователя и решением 0004; результаты не переодобрены.

## Предыдущий handoff R0–R9

`scientific_validity=not_assessed`. Ревизия по `reviews/07-11-2026.md`
записана в `docs/research-program.md` и `research/escalations/2026-11-07-review.md`.
Рекомендация сделать позднее ускорение CatBoost главным вопросом не принята:
это развилка решения 0003. Ответа человека или Codex на этот пакет нет.
Локальные claims не являются внешним подтверждением.

**Следующая команда только читает состояние. Нового stage нет.**

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

Charged runner wall после R8 и R9: 121.13999999966472 с из 86400 с.
`reconciliation_problems` пуст. GPU, платные API и скачивания в R8–R9: 0.

| Stage | Attempt | Review SHA-256 | Archive SHA-256 |
| --- | --- | --- | --- |
| R8 | `20261107T140907-bd7e4a5a5e` | `92a010aba675c41b72cecba3285180f993b646d8210560f801c3f50af7530e03` | `7816eff3283d690a4e7c8be3341a89c8daeed8618b89ed68cc802c0e44728fad` |
| R9 | `20261107T141434-ba64acb5eb` | `355781c82ce0b05c35b06f1dd5e446bff2933b3390c0bf22205a2c94282709c9` | `4fcfb1b0ba74d1f3f35bf53c54ff4ff244802796060a46ecb88f5be3ee948bfc` |

R0–R7 остаются закрытыми, как в предыдущей записи состояния. Их архивы не
переписывались.

## Что сделано в ревизии

R8, discovery. Пять детерминированных рядов. Глобальный детектор не помечает
`fast_plateau_moderate` (`candidate=false`, `t_star=79`). Окно 5 помечает
`t_star=70`. На `late_tail` глобальный `t_star=79`, локальный `t_star=90`.
`two_drops` исключён, хотя локальное правило вернуло границу 45. CatBoost
не обучался. Claim `R8-detector-catalog`, reviewer
`e078bfb0-20d5-4dcf-8549-220576f37ac2`.

R9, discovery. Salt `20261111`, seeds 7101 и 7102, итерации 1…64, допуск
`1e-8`. Все 14 ячеек имеют `interpolating=false`. Оба seed:
`not_identifiable`. Наименьшие остатки: `0.005397518627629901` и
`0.0047802282799166524`, оба при 64 деревьях. Двойной спуск не объявлен.
Claim `R9-iteration-threshold-not-identified`, reviewer
`b424b442-3a85-4957-b13b-8122a3040724`.

Design review обоих планов: `7078a728-72d6-430c-b416-5aacf7cc7b3c`.
Автор реализации: `research-owner-r8r9`. После закрытия попыток импорт CatBoost
перенесён внутрь функции обучения, а импорт torch в `src/r3_mlp.py` — внутрь
функций, которые его вызывают. Иначе сбор тестов видел torch до проверки
runtime R2. Снимки попыток хранят код, который считал ячейки. Числа не
пересчитывались. `checks.json`: `passed=true`, тесты 0, структурных проблем нет.

## Нерешённые вопросы

- Нужен ли ответ человека на пакет `research/escalations/2026-11-07-review.md`,
  если позднее ускорение должно стать главным вопросом. Пока ветка поиска
  конфигурации с нужной кривой остановлена.
- Порог интерполяции R9 внутри сетки не найден. Новый допуск или новая ось
  требуют нового протокола до обучения. Продолжение R9 добавлением итераций
  не поставлено.
- Логистический baseline и аудит близости текстов Qwen не запускались.
  Метаданные эмбеддингов по-прежнему не согласованы (`dim=768` и документы
  о 4096). `result/` не изменялся.
- Madelon, Bioresponse и HIGGS не скачивались.
- L05–L07 из R4 остаются неразрешёнными. Общего механизма у семейств нет.
- Окно 5 не проверялось на других формулах. Положительный локальный кандидат
  на `two_drops` не является идентификацией нескольких переходов.
