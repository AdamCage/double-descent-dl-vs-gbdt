# Double descent: механизмы в разных ML-моделях

Цель — исследовать, при каких условиях двойной спуск возникает, исчезает и
смещается у линейных моделей, случайных признаков, нейросетей и деревьев/GBDT.
Отсутствие эффекта — допустимый результат. Позднее ускорение LogLoss в CatBoost
остаётся отдельным вопросом, а старый текстовый эксперимент — историческим кейсом.

Проект перестроен 7 ноября 2026 по [ревью](reviews/06-11-2026.md).
Научная программа: [docs/research-program.md](docs/research-program.md).
Текущий статус: [research/STATE.md](research/STATE.md).

## Работа в Cursor

Cursor с выбранным пользователем Grok 4.7 Extra High и субагентами автономно
планирует, реализует, исполняет, анализирует и рецензирует исследование, ведёт
доказательства и пишет статью. Человек и Codex дают верхнеуровневую критику и
принимают критические решения. Подпись Codex на каждом stage/claim не нужна;
нужен отдельный reviewer субагент и проверяемая evidence.

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

Подробные команды, восстановление и архивирование:
[docs/automation.md](docs/automation.md). В Cursor доступны команды
`/research-next`, `/research-review` и локальные skills стадий.
Правила: [AGENTS.md](AGENTS.md), [единая авторизация R1–R7](docs/decisions/0003-cursor-autonomous-research.md).
R0 и R1 операционно закрыты; следующий шаг — [подготовка R2](research/stages/R2/HANDOFF.md).
`next` рекомендует `prepare`: Cursor сам завершает preflight и активирует stage.

## Доказательства и воспроизведение

Каждый attempt хранит frozen plan, source/input snapshots, resolved argv,
метаданные среды, stdout/stderr, выходные хеши и привязанное к ним ревью.
Неудачные попытки сохраняются. Переносимые архивы проверяются без исходного
каталога. `uv.lock` фиксирует зависимости; полного битового равенства расчётов
на другом железе это не гарантирует.

```powershell
uv run python scripts/check_repository.py --output checks.json
uv run python scripts/episteme_bridge.py status
```

EpistemeOS подключён из неизменённого `EpistemeOS-0.2.0/` (внутренняя версия
пакета — `0.1.0`). Новый bridge фиксирует вопрос, а не пререгистрацию
несуществующего эксперимента. Старый текстовый DomainPack не используется
для новой межмодельной программы.

Источники и границы их проверки: [docs/literature/](docs/literature/README.md).
Численные утверждения статьи требуют review отдельного Cursor субагента либо
реального human/Codex review и точных bindings в [research/claims.json](research/claims.json).

## История

[Оценка старых результатов](docs/legacy-assessment.md) объясняет, что можно
сохранить и с какими ограничениями. `result/`, исходные `.research/double-descent*`,
статья и старые протоколы сохранены. Частичный переносимый снимок исторических
логов/JSON: [research/evidence/legacy-2026-11-07/](research/evidence/legacy-2026-11-07/inventory.json).
Большие локальные матрицы в этом снимке не объявляются воспроизводимыми.

`objective.md`, `docs/plan/`, `.cursor/plans/` и протоколы 4/6 ноября являются
историей. `python main.py` по умолчанию больше не запускает старую сетку.
Намеренный replay требует `--legacy --stage ... --result-dir
artifacts/legacy-replays/<name>` и подготовленных входов в отдельном каталоге.
Оригинальный `result/` защищён входной командой от перезаписи.

Образцы: [EpistemeOS](https://github.com/AdamCage/EpistemeOS) и
[llm-semantic-afterlife](https://github.com/AdamCage/llm-semantic-afterlife).
Локальный `llm-semantic-afterlife` использован только для чтения.
