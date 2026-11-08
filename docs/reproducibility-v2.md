# Отдельный пакет воспроизведения v2

`scripts/build_science_package.py` собирает выбранные завершённые attempts
R10, R11 и R12 в самостоятельный каталог. Сборка читает готовую evidence и
метаданные установленных Python-пакетов; не обучает модели и не обращается к
сети. Полный dependency snapshot EpistemeOS и старые большие данные в пакет
не переносятся. Приоритет и научная новизна этим инструментом не удостоверяются.

## Сборка из проверенных attempts

Сначала выполнить обычные `research.py verify`, REPORT, отдельное review и
`close` соответствующих stages. Сборщик дополнительно проверяет входные хеши,
замороженный machine plan, полные task inventories и выходные хеши.

PowerShell, значения идентификаторов заменить на фактически выбранные attempts:

```powershell
uv run python scripts/build_science_package.py --self-test

uv run python scripts/build_science_package.py `
  --run "R10:artifacts/research/R10/<attempt-R10>" `
  --run "R11:artifacts/research/R11/<attempt-R11>" `
  --run "R12:artifacts/research/R12/<attempt-R12>" `
  --figures "artifacts/research/<derived-gallery-directory>" `
  --review-packets "research/stages/<packaging-stage>/inputs/review-packets" `
  --manuscript "manuscript/acceleration_audit_v2.md" `
  --output "artifacts/research/<packaging-stage>/<attempt>/tasks/<package-task>/science-package"
```

Разрешены абсолютные и относительные пути attempts; `R11:C:\...` разбирается
по первому двоеточию. Для каждого stage выбирается один attempt. `--figures`
необязателен. Выходной каталог должен быть новым и находиться вне выбранных
evidence/figures. Существующий пакет не перезаписывается; неудачная копия
сохраняется для диагностики. Пределы сборки — 50 тысяч файлов и 8 GiB.

Переносимый вызов Python, без PowerShell continuation:

```text
python scripts/build_science_package.py --run R10:/path/to/R10/attempt --run R11:/path/to/R11/attempt --run R12:/path/to/R12/attempt --manuscript manuscript/acceleration_audit_v2.md --output /path/to/new-science-package
```

## Что содержит пакет

- `evidence/Rxx/<attempt>/`: полный выбранный attempt, включая исходные
  `inputs`, immutable plan, сырые массивы/метрики, модели, stdout/stderr,
  manifest, REPORT и review, если они уже сохранены в attempt.
- `code/Rxx/`: отдельная минимальная научная версия из **снимка этого attempt**,
  с runner, нужными `src` модулями, научными тестами, PLAN.md, protocol и
  оригинальным `uv.lock`. Разные версии общих модулей не объединяются.
  Registry и workflow controls исполняемому научному коду не нужны; их
  исторические копии остаются только внутри полной evidence.
- `reviews/`: явный intake из необязательного `--review-packets`, например
  актуальный claim registry, соответствующие scientific decisions, closures,
  operational reviews и метаданные литературных snapshots. Сборщик сохраняет
  структуру и bytes выбранного каталога; root coordinator готовит его до
  замораживания packaging stage. На него действуют те же запреты секретов и
  symlinks. Выходной каталог пакета должен находиться вне этого intake.
- `requirements.txt` и `dependency-lock.json`: точные установленные версии
  базовых зависимостей NumPy/SciPy и, для модельных stages, scikit-learn/CatBoost.
  Для графиков добавляются Matplotlib/Plotly. Сборщик рекурсивно читает
  `importlib.metadata`, проверяет active requirement specifiers и вычисляет
  markers для текущей среды при `extra=''`. Сеть и новый resolver не вызываются.
- `paper/`, необязательные `figures/`, `supplement/`: текущая рукопись,
  готовая галерея, методологическая записка и её отдельное bound review,
  литературная записка и исходник построения фигур, если он доступен.
  Это delivery artifacts; их наличие не превращает их в прежние frozen inputs.
- `verify.py`, `replay.py`, `package-manifest.json`: проверка каждого файла и
  запуск выбранного научного stage без обращения к исходному checkout.

Оригинальный `uv.lock` описывает **главный репозиторий**, включая локальную
зависимость EpistemeOS. Он сохраняется для provenance, но не служит способом
установки standalone-пакета. Здесь используется `requirements.txt`; в нём
нет EpistemeOS и torch. Его exact pins относятся к активной dependency closure
среды сборки. Для другого OS/железа могут потребоваться другие wheels; точное
битовое равенство результатов не обещается. Нужен Python 3.12.

## Проверка и запуск после переноса

Проверка hashes не требует научных зависимостей:

```powershell
py -3.12 "C:\path\science-package\verify.py"
py -3.12 -m venv "C:\path\science-env"
& "C:\path\science-env\Scripts\python.exe" -m pip install -r "C:\path\science-package\requirements.txt"
& "C:\path\science-env\Scripts\python.exe" "C:\path\science-package\replay.py" R10 --output "C:\path\new-replay-r10"
& "C:\path\science-env\Scripts\python.exe" "C:\path\science-package\replay.py" R11 --output "C:\path\new-replay-r11"
& "C:\path\science-env\Scripts\python.exe" "C:\path\science-package\replay.py" R12 --output "C:\path\new-replay-r12"
```

Portable Python:

```text
python3.12 /path/to/science-package/verify.py
python3.12 -m venv /path/to/science-env
/path/to/science-env/bin/python -m pip install -r /path/to/science-package/requirements.txt
/path/to/science-env/bin/python /path/to/science-package/replay.py R10 --output /path/to/new-replay-r10
```

Установка зависимостей пользователем может обращаться к публичному индексу;
сам сборщик и научные runners этого не делают. В окружении с заранее
подготовленными wheels использовать обычные pip `--no-index --find-links`.

Среду и replay outputs создавать **вне** проверяемого пакета. `replay.py`
сначала проверяет весь inventory, затем выполняет точный runner из отдельного
`code/Rxx`, передавая абсолютные пути к исходному protocol и, если он был
зафиксирован в argv R12, calibration. Выходной каталог должен быть новым;
stdout/stderr и `replay-record.json` сохраняются и при ошибке. Import caches
отключены, чтобы replay не менял проверяемый пакет.

Научные тесты выбранного stage можно запустить его отдельной средой и каталогом
`code/Rxx`; тесты с реальным tiny fit требуют явного `--model-smoke` в
`scripts/run_unit_checks.py` и являются вычислительной задачей вне обычного CI.
`--self-test` сборщика использует только крошечные фиктивные данные и runners,
проверяет раздельность версий, replay, запрет перезаписи, tamper detection и
отказ от `.env` без научного обучения.

## Архив и границы гарантий

Manifest содержит hashes всех файлов, кроме себя. Его SHA-256 печатается при
сборке; внешний digest нужно сохранить в packaging REPORT/closure или в
отдельном handoff. После закрытия packaging stage допустим отдельный
эксклюзивно создаваемый архив через `scripts/evidence_bundle.py` и его проверка
без исходного каталога. Архив нельзя создавать внутри самого пакета.

Связанные hashes и локальные reviews не удостоверяют личность автора, внешний
исследовательский приоритет, научную истинность или одобрение журнала.
`scientific_validity=not_assessed` сохраняется. Replay на исходных seed — новое
исполнение прежнего протокола, а не независимое подтверждение на новых данных.
Графики старых R2–R4, если они присутствуют в экспортированной галерее, требуют
отдельной исторической evidence для полного пересчёта; данный runner ограничен
явно выбранными R10–R12.
