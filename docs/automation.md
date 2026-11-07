# Cursor: автономное исполнение и проверяемое исследование

Действует вместе с `AGENTS.md` и `docs/research-program.md`. Это локальный
операционный workflow, не sandbox, не механизм доказательства научной истины
и не сервис, который сам запускает Cursor после перезагрузки компьютера.
Cursor продолжает работу по сохранённому stage packet в активной сессии и
возобновляет её из handoff в следующей. Выбор Grok 4.7 Extra High выполняется
в настройках Cursor пользователем; репозиторий не подменяет модель.

## Единственный маршрут

```powershell
uv sync --frozen
uv run python scripts/research.py status
uv run python scripts/research.py next
```

`next` показывает ближайший готовый stage и флаг авторизации. `ready` означает
готовые зависимости, а не право запускать неавторизованный научный план.
Registry — `research/stages.json`; machine plans — `research/plans/`;
сопроводительные пакеты — `research/stages/<ID>/`.

R0 закрыт. По [решению 0003](decisions/0003-cursor-autonomous-research.md) Cursor
с субагентами автономно выполняет R1–R7, включая детальный scientific design,
анализ, критику и статью. Codex/человек нужны для верхнеуровневой критики и
критических событий, а не для подписи каждого stage/claim.

`next.recommended_action=prepare` и `preparation_authorized=true` означают:
самостоятельно подготовить план/код, получить review другого субагента,
заморозить, выполнить validate-plan, затем самому активировать execution этого
stage. Пока plan draft, execution=false — правильный preflight, не новый
authorization blocker. Подготовку выполнить по R1 HANDOFF и общей авторизации;
следующие stages готовить после закрытия зависимостей. Исполнение draft-плана
и ослабление gates запрещены. Budget и scope берутся из bound charter, а не
из устаревшего R0 handoff.

Coordinator делегирует code/analysis/literature и отдельное review субагентам,
сохраняет задания, реальные agent IDs и результаты. Reviewer конкретного
артефакта отличается от автора. Рутинные отрицательные/inconclusive исходы,
исправления и refinement внутри discovery не требуют Codex. Существенная
неожиданность после диагностики, развилка основной цели, центральная нерешённая
integrity проблема и выход за общий мандат требуют evidence escalation packet.

## План → run → review → close

1. Cursor читает вопрос, источники, ранее доступные результаты и HANDOFF.
   Заполняет PLAN.md из `research/templates/PLAN.md`, machine JSON и ADR
   отклонений. Внутренний дизайн и план проверяет отдельный субагент, без нового
   sign-off Codex. Научный JSON должен иметь `status="frozen"` до исполнения;
   наличие skeleton не есть preregistration. Новые источники фиксирует с
   permalink, версией, retrieval date, locator и SHA-256 разрешённого snapshot.
2. Выполняет `validate-plan <ID>`. Входы перечислены **файлами**, не каталогами;
   включить все импортируемые исходники, используемые данные, конфиги, split и
   seed manifest, `uv.lock`, релевантные dependency sources и протокол.
   Workflow автоматически сохраняет свои controls, но не обнаруживает все
   косвенные imports и зависимости: полноту source closure проверяет reviewer.
   `.env`, `.git`, `.venv` и секреты входами не являются.
3. Выполняет `run <ID>`. Создаётся новый immutable attempt с frozen plan и
   копиями declared inputs. Задачи запускаются argv без shell, рабочий каталог
   каждой — `artifacts/research/<ID>/<attempt>/tasks/<task>/`. Значения
   `{python}`, `{repo}`, `{run}` разворачиваются в пути. Код читает исходники и
   входы из repo; их хеши проверяются до/после, а копии сохраняются для replay.
   Выходы — относительные файлы внутри task dir. Таймаут завершает дерево
   процесса; budget считается на attempt, `total_wall_seconds` — суммарно по
   stage, `max_attempts` ограничивает повторы. Storage/API бюджеты дополнительно
   контролирует конкретный task; общий runner не является финансовым лимитером.
4. Выполняет `verify <ID> --attempt <attempt>`. Нужны точная зарегистрированная
   matrix задач, успешный exit, непустые выходные hashes, полные stdout/stderr,
   неизменные snapshots/inputs. Пустой каталог, пропущенный seed и нулевой exit
   без ожидаемого файла не проходят. Content-addressed reuse автоматически не
   реализован: run всегда новый. Кэш будущего task проверяет полный fingerprint
   данных/порядка/кода/конфига и всё равно записывает происхождение результата.
5. Пишет REPORT.md с каждым критерием и предсказанием, всеми failed attempts,
   отклонениями и пределами вывода. Готовит bound review:

   ```powershell
   uv run python scripts/research.py review R0 --attempt <attempt> --report research/stages/R0/REPORT.md --prepare research/reviews/R0-draft.json
   ```

   Заполняет reviewer name/harness/independence, для каждого критерия
   `verdict=pass|partial|fail`, для каждого prediction outcome, evidence вида
   `repository-check/checks.json`, limitations и `verdict=accept|revise|reject`.
   Не принимать критерию PASS только по наличию ключевого слова. Читать
   исходные результаты и смысл проверки. Для Cursor lane заполнить
   `review_mode=subagent_review`, различные `author_agent` и `reviewer_agent`.
   Charter-bound Cursor stage требует review другого субагента; имена ролей
   одного автора не заменяют его. Это локальное team review, не внешняя экспертиза.
6. Фиксирует scored review и closure:

   ```powershell
   uv run python scripts/research.py review R0 --attempt <attempt> --review research/reviews/R0-draft.json
   uv run python scripts/research.py close R0 --attempt <attempt>
   ```

   `close` требует всех PASS и accepted операционного review; review/report/run
   связаны хешами. Изменения после freeze требуют нового attempt/stage, а не
   перезаписи истории. Rejected review сохраняется; исправление и новое ревью
   идут в новой попытке. Итог всегда `scientific_validity=not_assessed`.
7. Обновляет HANDOFF и `research/STATE.md`; если следующий stage разрешён и
   укладывается в общий charter, продолжает сам через prepare/activate loop.
   Обращение к Codex требуется только по критериям критической эскалации 0003.

Минимальный machine plan (пример формы, не эксперимент):

```json
{
  "schema_version": 1,
  "stage_id": "EXAMPLE",
  "status": "frozen",
  "question": "Does the registered offline task produce its evidence?",
  "scope": "engineering smoke only",
  "budget": {"max_wall_seconds": 120, "total_wall_seconds": 240, "max_attempts": 2},
  "tasks": [{
    "id": "repository-check",
    "argv": ["{python}", "{repo}/scripts/check_repository.py", "--root", "{repo}", "--output", "checks.json"],
    "inputs": ["scripts/check_repository.py"],
    "outputs": ["checks.json"],
    "timeout_seconds": 100
  }],
  "exit_criteria": [{"id": "C1", "description": "Offline checks pass", "task_ids": ["repository-check"]}],
  "predictions": [{"id": "P1", "statement": "Checks pass without scientific training"}]
}
```

Этот короткий пример не является достаточным source closure: реальный R0
перечисляет также тесты, импортируемые модули, документы и dependency snapshot.
Для научных задач argv указывает на версионированную конфигурацию с полной
сеткой, seed, stopping rules и schema сырых метрик. Один task может вычислять
matrix, но тогда он обязан записать и проверить каждую ячейку, не скрывая сбоев.

## Возобновление и перенос

Failed/interrupted attempts не удаляются. Runner намеренно не продолжает
неизвестный частично записанный task автоматически. Checkpoint каждого
длительного научного task реализуется вместе с этим task: atomic запись,
full fingerprint и собственная проверка resume. Без неё новый attempt начинает
task заново; скрытого обещания exactly-once выполнения нет.

После обрыва проверить `.research/workflow/active.lock` и PID через
`Get-Process -Id <PID> -ErrorAction SilentlyContinue`, сопоставить команду и
дочерние процессы. Только при доказанном отсутствии владельца сохранить lock
как recovery evidence и убрать именно этот файл. Не удалять locks по таймеру.
Если старый процесс ещё работает — наблюдать его, не запускать второй.

После изменения кода старые закрытые stages проверяются по снимкам:

```powershell
uv run python scripts/research.py verify R0 --attempt <attempt> --historical
```

Live runs и closures игнорируются Git. До передачи/публикации сохранять
непустой переносимый архив и его SHA-256 в отчёте:

```powershell
uv run python scripts/evidence_bundle.py create --source artifacts/research/R0 --output research/evidence/R0-runs.tar.gz
uv run python scripts/evidence_bundle.py verify research/evidence/R0-runs.tar.gz
```

Проверка сверяет каждый файл, inventory и hashes непосредственно в архиве,
без обращения к исходному каталогу. Для восстановления сначала verify, затем
распаковать в **новый пустой** `artifacts/research/R0` (`tar -xzf ... -C ...`).
Closure из `.research/workflow/closures/` архивируется отдельно или переносится
в пакет handoff с digest; после восстановления `status`/исторический `verify`
и проверка closure обязательны. Выходной архив всегда создаётся эксклюзивно,
поэтому версии именуются явно. Это локальные integrity bindings, не цифровая
подпись; автор, способ хранения и внешний digest остаются частью provenance.

`research/evidence/legacy-2026-11-07/inventory.json` честно помечает большие
старые матрицы как local-only без вычисленного hash; не считать их включёнными
в переносимый снимок. Полное архивирование таких файлов — отдельная измеряемая
задача. SQLite live DB и WAL нельзя переносить случайным file-copy: использовать
read-only EpistemeOS export либо согласованный SQLite backup. DB файловый архив
через `evidence_bundle.py` запрещён.

## EpistemeOS и научные claims

```powershell
uv run python scripts/episteme_bridge.py status
# Только после принятия вопроса; записывает НОВЫЙ store, не запускает модель:
uv run python scripts/episteme_bridge.py bootstrap
uv run python scripts/episteme_bridge.py verify
```

Bridge фиксирует только question и digest программы. Это **не** новый DomainPack,
не experiment preregistration, не batch analysis и не scientific approval.
Подключение cross-model pack — будущая задача отдельного дизайна, если нужны
именно kernel claim admission guarantees. Пока evidence ledger локален,
проверяем и явно ограничен; старые question/evaluate/export закрыты из-за
несовпадения EmbeddingGemma/GPU/noise с Qwen/CPU/clean фактом.

Accepted local empirical claims заносятся в `research/claims.json` после
bound review отдельным Cursor субагентом в пределах общей авторизации либо
реально выполненного human/Codex review. `src/research_claims.py` проверяет exact UTF-8
statement hash, claim ID, непустые evidence `{path, sha256}`, scope, limitations
и хеш внешнего decision JSON из `scientific_review={path,sha256}`. Decision
содержит `claim_id`, `statement_sha256`, тот же exact evidence set, `reviewer`,
`harness=human|codex|cursor`, `verdict=accepted`. Для Cursor claim требуются
`assessment=evidence_supported`, `review_status=cursor_team_reviewed`, а в
decision — `review_mode=subagent_review`, разные непустые `author_agent` и
`reviewer_agent`, exact `authorization_decision` binding из
`claims_doc.autonomy_authorization`. Сама авторизация должна разрешать этот
lane и быть неизменной по SHA-256. Один лишь operational stage PASS не
создаёт claim: нужны содержательный review и exact evidence/statement bindings.
Метки личности логические: код не удостоверяет, кто написал файл, внешнюю
независимость или истинность утверждения. `scientific_validity=not_assessed`
сохраняется; human approval не придумывается. Сейчас accepted claims нет.

## Проверки инфраструктуры

`uv run python scripts/check_repository.py --output checks.json` проверяет
структуру, authority/evidence bindings и unit tests. CI повторяет проверки на
Windows, не обучая модели и не вызывая платные API. Тесты гоняют маленькие
временные процессы для проверки corruption/timeout/review gates. Они не являются
экспериментальными результатами о двойном спуске. Новый экспериментальный
адаптер, статистика и фигуры получают отдельные соответствующие проверки.
