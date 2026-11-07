# R6 — отчёт: перенос не запускался

Attempt: `20261107T111120-40cf0c74f2`. Plan SHA-256 `d3a680956570bcef400c22055ad41f21c7afd413e3d8bd97e91b499958ae58e2`. Manifest SHA-256 `174a1097e4c1eba051c2596e9317e0c32c31daec6b1ae9d9f8aab3e158e014d9`. `verify` прошёл, `checked_files=3`. Elapsed 0.156 с. `scientific_validity=not_assessed`.

Design review `5f1d2237-db12-4c0f-97ca-78f274f68af7`, accept. Автор аудита: `ddb85535-1ef1-4069-95c4-46578067887a`.

## Критерии

C1. `transfer-audit/disposition.json`: `disposition=not_run`, `download_bytes=0`, `models_fit=0`, `datasets_opened=[]`, `http_requests=0`. Кандидат sklearn diabetes записан как не открытый.

C2. Тот же файл перечисляет claim ids `R2-confirmation-contrasts`, `R3-confirmation-budget`, `R4-separate-axes`, `R5-replication-not-one-mechanism`. Перенесённый эффект не заявлен.

## Предсказание

P1. Поддержано операционно: реальная таблица не оправдана, меню датасетов не скачивалось. Причины — разные направления чистого риска у листьев и леса, опровергнутый бюджет MLP и отсутствие одного механизма в R5.

## Пределы

Это решение не обучать, а не измерение на реальных строках. Оно не говорит, что на другой таблице эффекта нет. Эскалации 0003 нет: блокировка предусмотрена заданием этапа.
