# R10 — независимая оценка аудита ускорения

Attempt `20261107T192414-36aee5d99b`, plan SHA-256
`669285e14b959243a077631ab85691ad0d8df77f683a2a98db80caf2fe4ab256`, manifest
`42dcdc5ddc70026e364190d572bc2a9a2acdb3dd76090526aa8cba70df4d3e1c`.
`verify` прошёл, checked_files=10. Автор `/root`, design reviewer
`/root/science_critique`, harness codex. Scientific validity not_assessed.
Все пути ниже относительно task `calibrate-evaluate/` этого attempt.

## Полнота и критерии

C1 выполнен: curves.npz и curves.jsonl содержат 7188 траекторий — 2388
calibration, 4800 evaluation. Полная замороженная matrix; фазы seed1/2
разделены. Все данные сохраняются, без отбора по детекции.

C2 выполнен: calibration.json фиксирует thresholds и order190/199; analysis.json
содержит72 rates с pointwise exact binomial CI и локализацией. Environment записан.
`mechanism_claim=false`, `double_descent_declared=false`, GPU/API/download0.

C3 выполнен: unit-checks/checks.json —6 meaningful tests, failures/errors0.
Frozen declared sources, workflow/resource controls, protocol и uv.lock сохранены
в snapshots attempt. Проверка источников до/после прошла. Failed attempts R10 нет.

## Все предсказания, включая опровергнутые

P1 supported: локальное правило R8 отмечает 200/200 noisy linear кривых в каждом
из четырёх noise conditions (CP95% [0.981725,1]). Более того, оно отмечает
все2400 null evaluation кривых, включая exponential и early_plateau.
Это измерение на заданном каталоге, не доказательство FPR1 для любого шума.

P2 supported по заявленному слабому порогу observed FPR<=.10. У calibrated scan
noisy linear rates: small_iid12/200=.06, small_ar8/200=.04,
large_iid8/200=.04, large_ar20/200=.10; на остальных восьми null conditions0/200.
CP95% для large_ar linear [0.0621594,0.150213], поэтому нельзя утверждать
условный FPR<=.05 после конкретной калибровки. Теоретический rank bound
маргинален по calibration, а оценка eval условна на один общий threshold.
Различие явно сохраняется. CI pointwise, не simultaneous.

P3 REFUTED: small_iid/last_tenth оба global и scan отмечают200/200. Scan не
имеет предсказанного преимущества по detection rate. Median ошибки локализации
среди flagged —global25 точек, scan6; это отдельное описательное измерение,
не переопределение P3 после получения результата.

P4 supported: максимальная ошибка положительной affine-инвариантности
8.359979375427429e-14 <1e-10. Масштабированные копии не увеличивают n.

## Область применимости

Global на fast_plateau_tail отмечает0/200 при обоих small noise,
2/200 large_iid и1/200 large_ar. Scan даёт200/200 при small noise, но
только14/200=.07 и3/200=.015 при large_iid/large_ar. Улучшение детекции
не универсально, и высокую power при сильном шуме заявлять нельзя.

На linear с AR(1) global ложные сигналы97/200=.485 и103/200=.515,
а при iid0/200. Значит, RSS floor старого global правила не даёт
автоматического контроля ошибок при зависимости внутри кривой.

На late_hinge scan detection200/200 во всех noise, но локализация в пределах
32 точек падает с1 до.63/.56 при large noise; наличие сигнала не равно точной
локализации. Полные72 строки, а не только выбранные сравнения, находятся в
analysis.json и exports/analysis.json. Их рисунки — производные, не новые повторы.

## Ограничения и следующая работа

Известные формы и Gaussian AR(1), одна длина/сетка окон, одна calibration.
Не оценены другие null laws, calibration uncertainty, различные горизонты,
временные деформации или частные сочетания jumps/нескольких переходов.
Scan и rank calibration — известные методы; вклад здесь эмпирическая карта
ошибок конкретных правил. На CatBoost эти thresholds только описательны.
R11 использует отдельное frozen расписание и новые data/model seeds.

Операционный review/closure и точный архивный digest фиксируются отдельно,
после независимой проверки отчёта. Никакого human approval не заявляется.
