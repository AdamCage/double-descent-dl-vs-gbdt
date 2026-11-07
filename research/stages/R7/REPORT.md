# R7 — отчёт локальной записки

Attempt: `20261107T112024-93842d83aa`. Plan SHA-256 `bb098c64ad5f779380a0083bd6c8b6f632577219ac99746286fa66c802e7c737`. Manifest SHA-256 `6b02217e3981263049a809ef6b702da615948bb8143e786e677b2723a495999b`. `verify` прошёл, `checked_files=3`. Elapsed 0.172 с. `scientific_validity=not_assessed`.

Design review `6d4303d0-c56b-4402-bf06-845ac8bc0ee6`, accept. Автор записки: `ddb85535-1ef1-4069-95c4-46578067887a`. Внешняя подача не выполнялась.

## Критерии

C1. `synthesis-check/synthesis-check.json`: `claims_valid` true, `claim_problems` пуст, `passed` true.

C2. `missing_phrases` пуст, `external_submission` false. Текст включает опровергнутый бюджет MLP, раздельные оси деревьев, фразу о том, что репликации не идентифицируют один общий механизм, и disposition `not_run`.

## Предсказание

P1. Поддержано проверкой: отрицательный бюджет, раздельные оси, неидентифицированный механизм и непроведённый перенос остались в записке.

## Пределы

Проверка сверяет маркеры и ledger, а не доказывает механизм. `scientific_validity` не меняется. Эскалации 0003 нет.
