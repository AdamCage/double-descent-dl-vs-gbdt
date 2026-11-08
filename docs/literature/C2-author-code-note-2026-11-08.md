# C2: уточнение EFC по официальной реализации

Прочитано 8 ноября 2026, `/root/literature`, через web:
[Allerbo `feature_complexity/cifar_boost.py` на main](https://github.com/allerbo/feature_complexity/blob/main/cifar_boost.py).
Code не скачивался и не исполнялся. Это изменяемая ветка; commit permalink и
локальный code hash здесь не установлены. Замечание дополняет, не переписывает
`C2-source-cards-2026-11-08.md` и разрешает указанную там неопределённость φ_t.

Проверены функции `get_efc` и `run_for_depth`. На каждой итерации авторы
обучают **новое слабое дерево** на текущих residuals и вычисляют EFC именно
этого дерева, а не concatenated features накопленного ensemble. По decision
paths без корня строится pairwise overlap; сравнение с максимумом каждой
колонки даёт same-terminal-leaf indicator для обычного sklearn tree.
Off-diagonal normalization равна m(m−1). Total EFC агрегирует per-tree EFC
по неотрицательным уменьшениям training loss. Это подтверждает выбор
per-weak-tree occupancy metric в C2; адаптация Eq. (2)–(3) к иной процедуре
обучения всё равно не является побитовой replication авторского эксперимента.

В прочитанном коде начальное значение `loss_old` задаётся средним train target,
а следующие значения loss — суммой squared residuals. Это выглядит как
несогласованность инициализации весов первой итерации с SSE convention.
Исходный эксперимент не исполнялся, влияние на результаты статьи не измерено;
ошибка опубликованного научного вывода отсюда не следует. C2 должен брать
фактический L₀ собственного baseline и объявленную loss convention.

Авторский CIFAR experiment также использует mean initialization и momentum,
тогда как контролируемый CatBoost C2 может иметь zero initialization и иной
update rule. Эти различия следует перечислить; нельзя требовать от EFC как
complexity measure непосредственно оценивать ошибку finite-label influence.
