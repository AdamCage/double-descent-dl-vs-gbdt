# Stage ID — вопрос и границы

Статус: draft / frozen; science decision reference; allowed execution scope.
Документ дополняет machine PLAN.json, не подменяет его.

## Вопрос и связь с литературой

Какой механизм различаем? Какие источники прочитаны и какие только найдены?
Что уже было доступно планировщику (discovery/test exposure)?

## Гипотезы, предсказания и фальсификаторы

Для каждого объяснения: наблюдаемое предсказание, вмешательство, альтернативы,
условия refutes/inconclusive и критерий остановки. Не требовать наличия пика.

## Дизайн и ресурсы

Семейство и одна ось сложности; остальные настройки фиксированы. Генератор /
data revision, лицензия, split/group hashes, seeds разделены на draw/split/noise/
model, шум только train, преобразования fit только train. Grid и seed matrix.
Единица повторения и estimand; convergence/interpolation definitions; метрики
в полной точности; uncertainty и multiplicity. Discovery и confirmation отдельно.

Объявить timeout/task, wall-clock/attempt/storage/API budget, recovery,
исключения и лимит refinement до старта. Нет бесконечного autonomous retry.

## Артефакты и критерии выхода

Точные task IDs и пути сырых данных/метрик, model diagnostics, provenance,
фигуры, audit и report. Критерии сопоставлены с task IDs в PLAN.json.
Отрицательный результат и недостаточная идентифицируемость имеют собственные
выходы. Любое отклонение — ADR до нового attempt.
