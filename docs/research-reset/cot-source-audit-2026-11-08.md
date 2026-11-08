# Source audit: adaptive cues and comparable acknowledgment estimates

8 ноября 2026. Source-only preparation для R15; numerical results отсутствуют.
Ни одного parquet payload не скачано. Прочитаны paper v2 (основной текст,
Table 2, Fig. 6–7, Appendix C/E), официальные scorer/analysis/config/generator
исходники и релевантные полные разделы OpenAI blog, включая appendix.
Произвольный мировой приоритет этим обзором не установлен.

## Проверяемый вопрос и предел новизны

Основной кандидат: **насколько межмодельная разность частот acknowledgment
меняется при переходе от общих вопросов к общей точной версии cued input,
и какую долю исходного вопросного пространства поддерживает такое сравнение?**

Это эмпирический аудит переносимости интерпретации оценок. Adaptive challenge
является намеренным estimand авторов. Он не объявляется ошибкой, скрытым
confound или опровержением работы. Утверждать новую метрику нельзя: matching,
standardization и ограничения overlap известны. Кандидат на новый результат —
конкретный измеренный вывод о чувствительности и покрытии этого release.

Фальсифицируемая версия для заморозки PLAN: на четырёх заранее выбранных
instruction/reasoning family pairs разность acknowledgment gaps между
common-question и exact-input comparisons достигает заранее установленного
практического порога (предложение: 5 percentage points) при достаточном
зафиксированном overlap. Все пары, семейства задач и интервалы показываются,
включая отсутствие эффекта. Ни один rank flip сам по себе не считается успехом.
Порог, coverage gate, weighting и multiple-comparison rule окончательно
задаются PLAN до outcomes. Это не утверждение, что эффект существует.

Смена matching меняет также популяцию вопросов. Поэтому различие оценок
**не идентифицирует причинный эффект версии cue** и не является автоматически
«исправленной истинной faithfulness». При нехватке support результатом может
быть только ограничение идентифицируемости; это честный итог сегодняшнего
проекта, но ещё не journal-ready claim.

## Что именно оценивает опубликованная работа

[Meek et al., arXiv:2510.27378v2](https://arxiv.org/abs/2510.27378v2)
([PDF](https://arxiv.org/pdf/2510.27378v2)), §3/Table 2: cue направлен против
собственного baseline answer модели. Faithfulness здесь означает cue
acknowledgment во всех cued responses, включая отсутствие смены ответа.
QwQ-32B оценивает acknowledgment и factor coverage; reasoning и final output
доступных reasoning-моделей входят в judged trace. Appendix E задаёт
sample-level M=(F+V)/2 и стандартные ошибки. Fig. 6–7 уже анализируют
correctness transitions и условные scores: простая стратификация correct/incorrect
не является нашим вкладом. Работа ограничена single-turn QA и данными judge;
это не измерение причинной faithful reasoning или deployment safety.

Официальная цепочка: [README code repository](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/README.md)
связывает paper с [HF transcripts](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts).
Audited code commit: `7da0cfb310db80b71d842d310b9c44c9b5e8ee37`,
29 ноября 2025. Его связь с каждой опубликованной таблицей отдельно не доказана;
ниже точно описан доступный код, а не заявлена bitwise reproduction paper.

## Scorer: значение bool и ошибки разбора

[cue_aware_adaptive.py](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/src/measuring_cot_monitorability/scorers/cue_aware_adaptive.py),
blob `7590c09228fdd37a559cfa30ff4f765ed84088d0`:

- Trace собирается из ContentReasoning и текстовых частей сообщения; completion
  добавляется, если его ещё нет в собранном тексте. Это не только final answer.
- has_cue=false возвращает 0 без judge. Отсутствующие обязательные cue metadata
  или judge criterion также возвращают 0. Visual/XML имеют специальные criteria.
- Judge спрашивают, замечает ли trace cue, даже если его отвергает. Полученный
  ответ разбирается re.search; группа C даёт 1, любая иная ситуация — 0.
- [requirements.txt](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/requirements.txt)
  фиксирует inspect_ai==0.3.146. Его [DEFAULT_GRADE_PATTERN](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0.3.146/src/inspect_ai/scorer/_model.py)
  равен `(?i)GRADE\s*:\s*([CPI])(.*)$`; итоговая проверка Meek сравнивает
  захваченный символ с uppercase C. Наш pipeline не перезапускает judge.

Следствие для анализа: HF cue_acknowledgment=false может означать и отсутствие
acknowledgment, и scorer/parse failure. По одному bool эти случаи не разделить.
Точность аннотации не становится известной от её наличия. Unknown/null нельзя
без объяснения превращать в false; нужны missingness/exclusion ledger и
sensitivity limits. Null, empty string, baseline false имеют разные смыслы.

## Точные знаменатели доступного analysis code

[extract_metrics.py](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/scripts/analysis/extract_metrics.py),
blob `1e903dc10605883f018229fa28f8f43c5ea8c867`:

| Уровень | Реальная операция |
| --- | --- |
| config × scope | Берётся последний .eval по filename; extraction не доказывает полноту опубликованного run |
| F в config | mean по присутствующим cue scorer values; value==1 или ==CORRECT считается acknowledgment; baseline F не считается |
| V в config | mean по присутствующей factor_utilization_rate metadata |
| F/V в dataset × model × scope | Невзвешенное mean config means, т.е. macro по subdatasets |
| M в adaptive scope | По sample_data вычисляется (bool(cue_ack)+V)/2, если V не None; отсутствующий cue_ack трактуется как 0; затем micro mean этих M |
| baseline | V/accuracy могут быть; M пропускается |
| state transitions | Join baseline/adaptive по sample ID, затем четыре сочетания correctness; conditional cue_ack count/count |
| sample_count/SE | sample_count суммирует len(log.samples); SE helper использует его; число реально scored cue/V/M может отличаться |

Таким образом, pooled M нельзя молча заменить на среднее отдельно macro F и V.
F-primary reanalysis должно явно назвать свои weights и populations; исходный
paper estimand и наши descriptive estimands показываются раздельно. Для
воспроизведения macro нужны проверенные task-subset IDs. Их нельзя надёжно
получить разбиением core_id по первому underscore.

[factor_utilization.py](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/src/measuring_cot_monitorability/scorers/factor_utilization.py),
blob `b1d3010198b92a698523ea4dbd86a39bd3e4300e`: parses XML counts,
fallback n/out-of/m, не зажимает ratio в [0,1]; при отсутствии factors возвращает
1, при неудачном parsing по умолчанию 0. Не делать clipping молча. Эти поля
сохраняются для аудита, но не образуют новый primary endpoint.

## Adaptive policy и проверка treatment identity

[generate_adaptive_datasets.py](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/scripts/data_processing/generate_adaptive_datasets.py),
blob `2d92f26fdb30a1cb616335096c8b1738fb070436`: baseline answers извлекаются
из .eval; adaptive_random выбирает random.choice из letters != own baseline,
прочие стратегии берут первую альтернативу. Default seed=42; сам этот факт
не доказывает стратегию, применённую ко всем released rows.

Логическое следствие: baseline-correct модель обязательно получает неверный
target; baseline-incorrect может получить верный target. Same core question
и same cue kind ещё не означают одинаковую интервенцию. При этом совпадающий
target не гарантирует одинакового rendered input: есть formatting и choice
order. Поэтому exact input SHA256 — основной проверяемый признак равенства
видимого treatment; cue text/target/choices нужны как объясняющие проверки.
Hash не включает недоступный system prompt или provider configuration, и
«равный input» не означает равные hidden states модели.

До выводов необходимо проверить uniqueness model×core_id×scope, согласованность
choices/correct_answer между моделями, mapping baseline_model_answer к
baseline row, target!=baseline, has_cue и actual scope. `stayed_incorrect`
означает wrong→wrong и может содержать смену ответа; nonswitch проверяется
прямым model_answer==baseline_model_answer. switched_to_cue тоже не означает
причинного использования подсказки.

## Что уже покрыто OpenAI prior

[Beyond g-mean², 20 October 2026](https://alignment.openai.com/beyond-gmean-squared/)
уже вводит pair-matched comparisons, magnitude-aware HodgeRank, отдельное
positive/negative support и ограничения disconnected comparisons. В разделе
Next step прямо сказано, что canonical instance matching не уравнивает
behavior class/difficulty; coverage надо показывать. Исторические nominal rank
changes часто близки к ties. Appendix выводит causal-recall bounds; individual
causal labels неизвестны. Это непосредственный prior нашего подхода.

Наш точный empirical question про model-dependent cue versions в Meek release
в проверенных местах этих источников не найден. Но это специализация известной
проблемы, а не доказанный первый результат. g-mean², rejection/retry risk и
causal-recall bounds здесь не применяются к acknowledgment bool по аналогии:
для них в этом reanalysis нет требуемого ground-truth behavior dataset.

## Release, schema, модели и ограниченный download specification

Pinned HF revision: `a7a8b25ae467db9d80d63e55ac053b48753b2d36`.
[Metadata API](https://huggingface.co/api/datasets/ameek/measuring_cot_monitorability_transcripts/revision/a7a8b25ae467db9d80d63e55ac053b48753b2d36)
показывает private=false, gated=false. License metadata: other;
license_name=mit-with-model-training-restrictions. [LICENSE](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts/blob/a7a8b25ae467db9d80d63e55ac053b48753b2d36/LICENSE)
содержит MIT и provider-specific ограничения на training; здесь только
статистический анализ. У constituent datasets остаются attribution conditions.

24 parquet = **940116134 bytes**. Весь репозиторий из 27 файлов = 940129927
bytes; разница — README/LICENSE/.gitattributes, не двадцать пятый data shard.
Все 24 имеют LFS SHA256, byte count, git pointer OID и pinned resolve URL в
[download-spec.json](../../research/stages/R15/inputs/download-spec.json).
Cap=1000000000 bytes; запрос новых файлов или другой revision запрещён spec.
Это ready_for_preflight, а не самостоятельно выданное разрешение на download.

Восемь моделей в [официальном config](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/src/measuring_cot_monitorability/eval/config.py):

| Pair | Provider model suffixes |
| --- | --- |
| Anthropic | anthropic/claude-3-5-sonnet; anthropic/claude-3-7-sonnet |
| DeepSeek | deepseek/deepseek-chat-v3-0324; deepseek/deepseek-r1-0528 |
| Qwen 2.5/QwQ | qwen/qwen-2.5-72b-instruct; qwen/qwq-32b |
| Qwen 3 | qwen/qwen3-235b-a22b-2507; qwen/qwen3-235b-a22b-thinking-2507 |

Для каждого есть bbh/gpqa_diamond/mmlu file, всего 3×8. Column model string
нельзя без проверки отождествлять с filename/provider identifier.

[HF info metadata](https://datasets-server.huggingface.co/info?dataset=ameek%2Fmeasuring_cot_monitorability_transcripts&config=default)
сообщает следующую схему (24 поля). Этот API не сообщил revision, поэтому
actual pinned parquet footer должен подтвердить её при intake:

| Type | Поля |
| --- | --- |
| string | core_id, model, id, scope, input, output, reasoning, model_answer, state_transition, correct_answer, correct_text, baseline_model_answer, cue, cue_target_answer |
| bool | is_correct, cue_acknowledgment, switched_to_cue, has_cue |
| int64 | factor_utilization_used, factor_utilization_total, correct_answer_index |
| float64 | factor_utilization_rate |
| list<string> | choices, causal_factors |

Viewer показывает baseline + пять scopes: 01_stanford_random, 02_visual_squares,
03_grader_hack, 04_unethical_information, 05_xml_metadata. Его core_id examples
совместимы с question ID, но гарантия cross-model key ещё не доказана. Slimmer
сохраняет все 21 поля кроме больших input/output/reasoning; для каждого
удалённого текста сохраняются exact UTF-8 SHA256, длина и null distinction,
а также file/row provenance. Input hash нужен для treatment join.

Info сообщает uncompressed size 2865124478 bytes: нельзя читать всю таблицу
в RAM или считать 940 MB оценкой peak memory. Intake должен работать batches,
проверять payload SHA256/bytes до чтения, сохранять failures и ограничить cache.
Нет outcome aggregates, новых model calls, обучения, manuscript priority claim
или подтверждения готовности к журналу в этом source audit.
