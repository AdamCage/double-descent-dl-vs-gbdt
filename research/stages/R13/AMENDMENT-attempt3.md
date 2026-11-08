# R13 — явная техническая поправка перед третьим attempt

8 ноября 2026. Решение coordinator `/root` в пределах прямой авторизации
пользователя и 0004; это не заднее одобрение результатов.

Attempt1 `20261107T210201-651abc88d9` failed при намеренно отсутствующем
историческом значении. Attempt2 `20261107T210937-844511dd10` complete:
validation и точный portable replay прошли. Фактический visual QA root и
отдельного `/root/science_critique` нашёл перекрытие десятичных tick labels
в левом panel `r12_fixed_late_acceleration_gates`. Поэтому C1 ещё не pass.
Оба attempts и все частичные/полные outputs сохраняются неизменными.

Поправка: max_attempts **2 → 3**, родительский max_attempts_per_stage=3.
max_wall_seconds=1800, total_wall_seconds=3600, global24h/20GiB,
CPU≤4, one dispatcher, GPU/API/network=0 и stage storage≤8GiB не меняются.
Повторная resource reconciliation обязательна до запуска. Третий attempt
допустим только после отдельного preflight review новых source hashes.

Изменение исходника только техническое: общий ScalarFormatter переключается
на scientific notation при малых/больших порядках и общий MathText multiplier.
Categorical/log formatters сохранены. Новая регрессия сравнивает реальные
rendered bounding boxes labels до/после. Данные, модели, grids/seeds, контрасты,
CSV counts и189 figure entries не меняются; новое n не возникает.

Никакого четвёртого attempt эта поправка не разрешает. При остающейся
нечитабельности — сохранённый revise и отдельное решение о способе поставки.
Ни выпуск графиков, ни его закрытие не завершают исходную научную цель.
