# R15 — pinned public CoT data intake, без анализа outcomes

Статус: frozen для независимого preflight; execution пока false. Пользователь
уточнил цель: законченный portfolio project сегодня,8ноября2026. Это
ограниченный offline statistical audit, не обещание accepted journal paper.
Автор плана `/root`, intake implementation `/root/cot_intake`; независимый
reviewer `/root/reset_critic` не автор этих файлов. R14 закрыт и не повторяется.

Вопрос engineering stage: можно ли получить ровно24объявленных публичных
parquet files по immutable revision, проверить каждый byte/hash и построить
полную компактную таблицу для отдельного frozen scientific analysis?
Никакие acknowledgment rates, rankings, hypothesis tests или выбор моделей
по outcomes здесь не вычисляются. Dataset membership задан публичным release,
не выбором удобных результатов. План R16/R17 и split замораживаются отдельно.

Источник: авторский HF release ameek/measuring_cot_monitorability_transcripts,
revision a7a8b25ae467db9d80d63e55ac053b48753b2d36.24parquets суммарно
940116134bytes; repository metadata с3дополнительнымифайлами940129927bytes.
URLs/LFS SHA256/размер каждого payload перечислены в inputs/download-spec.json.
README и LICENSE отдельно получены до payload, их hashes проверены; source
audit связывает release с paper/code. Использование — statistical reanalysis,
без обучения моделей/дистилляции и без новыхmodelAPI. License/attribution
передаются с evidence, third-party model training restrictions не обходятся.

P1: все24payload byte counts/SHA256 совпадают. Фальсификатор: любой wrong/missing
file/revision, HTTP exhaustion или content mismatch. Сбой сохраняется.
P2: все строки и21analysis columns сохраняются; input/output/reasoning заменены
точными UTF8SHA256/char+byte lengths, null отличается отempty. Сохранены
sourcepath,rowindex,payloadhash,row-content hash и ordered choices hash.
Фальсификатор: неверная схема/потерястрок/неполныйprovenance/слишкомбольшойslim.
P3: intake остаётся в заранее заданных network/storage/CPU рамках и не делает
численных выводов об alignment или latent faithfulness. Фальсификатор:
превышение caps, незарегистрированный источник, model call/fit/исполнениеtranscript.

Одна unit-check задача, затем serial intake task. Файлы downloads допускают
до4parallelHTTP streams; это не model jobs. CPU/env≤4, APIspend/GPU/modeltraining0.
≤1800s/attempt,≤3600s total,≤2attempts (parentmax3).Tasktimeouts unit90s,
intake1500s. Payload cap1e9actualtransferredbytes/attempt включает retries,
physicalHTTPcap144 сredirects,≤2tries/file,90s/socket operation. Все physical
requests/chunks учитываются; signed redirect query/headers не сохраняются.
Failedpartialbytes/logs не удаляются и не переписываются. Slim cap256MiB,
batch2048, raw+slim≤1.3GB плюс небольшой metadata/source snapshot; общий
20GiB и2GiBdataset cap не обнуляются. До retry учитываются и failed bytes.

Global guard использует НОВЫЙ alignment charter, явно переданный stage:
research/decisions/alignment-portfolio-authorization-2026-11-08.json.
Legacy r1_budgets schema сохраняется, но новое dataset allowance2GiB применяется
к этому scope; старые003/004не модифицируются. Для нового отчёта бюджета всегда
явно указывается этот charter; defaultстаройR1проверки не переопределяется.
После terminal независимо отsuccess acquisition.jsonl зеркалируется byte-exact
в новый artifacts/intake/R15-accounting/<attempt>/acquisition.jsonl ДО новогоrun;
resource_report сканирует этот корень. Никаких скрытых успешных downloads.

Критерии выхода: C1 все24hash/size иcaptured network log; C2 полный проверяемый
slim/схема/provenance/длины; C3 caps/отсутствиеoutcomeanalysis и executabledata.
Число строк узнаётся из input, не увеличивает число independent questions;
row count не научный результат. Worker ID/attempt фиксируются workflow lock.

Остановка после одной полнойматрицы либо техническогоfailure. Не дополнять
release другимrevision/другимифайлами молча. Verify→REPORT→отдельный REVIEW→close.
Сырыеbytes сохраняются один раз в этомattempt; для R16/R17 послеclose копируются
только slim+verified manifests в учитываемыйintake, с явным hash mapping.
Это ограничивает storage и не меняет исходные данные. Экспорт/архив stage —
exclusive content-addressed bundle+detachedverify. Результаты исходного научного
вопроса не объявляются до отдельных R16/R17 планов и их независимого review.
