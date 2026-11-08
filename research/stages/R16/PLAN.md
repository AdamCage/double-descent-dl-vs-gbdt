# R16 — discovery: сравнимость released acknowledgment proxy

Статус draft до input binding/source preflight; outcomes не читались.
Автор scientific design `/root`; implementation `/root/alignment_brainstorm`;
independent reviewer `/root/reset_critic`. Задача — законченный offline project
сегодня; это не гарантия публикации или подтверждения мировой новизны.
Первичные источники/точный legitimate native estimand — cot-source-audit.md.

## Данные, единица и раздельные split

Полный24parquet public release pinned byR15:3file-parent datasets bbh/gpqa/mmlu,
все8released historicalmodels ивсе28unorderedpairs. Имена фиксируются изR15
schema metadata до статистики, не выбираются по scores. Используется verified
slim со всеми исходнымистроками иexactinput/orderedchoices hashes/provenance.
F здесь released judge cue_acknowledgment во всех has_cue=true случаях с
наблюдаемымbool; non-switch не исключаются. Null не превращается вfalse;
falseможетвключатьстарыйscorerparsefailure. M/latentfaithfulness не оцениваются.
Weights отличаются отpaper macro-by-subdataset, поэтому exact reproduction
опубликованной таблицы не заявляется. Native question-weighted released proxy
и наши populations названы явно.

Question q=(fileparentdataset,core_id); контекст c=(scope,cue); version v=
(exactreleasedinputSHA,orderedchoicesSHA,exactcue_target_answer). Target letter
сам по себе не означает одинаковый treatment. Missing ACK labels исключаются
с явным ledger, не imputedfalse. Для validcuedACK строк missing/invalidversion/
input/choices/target keys — hard integrity failure, не unknown-version bucket.
Всеcoverage denominators сохраняются.
Дубликаты generations внутри model/q/c/v усредняются, затем равный вес version
вcontext, равный вес общих contexts вq и равный вес releasedcoreq. Choices /
correctness integrity проверяется, contradictory same-input metadata не сливается.
До outcomes проверяются source_file∈24manifest, payloadSHA, uniqueness(file,row),
одинmodel/file иdataset mapping; orderedchoicesSHA пересчитывается изretainedlist.
Дляодногоq проверяются unorderedchoice multiset иcorrect_text гдеониизвестны;
буква/индексправильногоответа сравнивается толькоприодинаковомorderedchoices.
Каждая используемаяcuedq/model должнаиметьvalidnonnullnonemptyBASELINEinputSHA
anchor; иначеfailure сcoverageledger. BaselineinputSHA вариации междуmodels
отдельноreport, анеавтоматическоедоказательствоэквивалентностиpromptformat.

Dependency clusters — union q и exactnonnullBASELINEinputSHA; aliases не дают
нового independent n. Point population остаётся equal releasedcoreq; количество
core IDs и независимых clusters показываются отдельно. Cluster key=minlex
canonical 'dataset|core_id'; separator вID запрещён. SHA256('20261108|'+key),
int(fullhex,16)%20<6 — discovery30%, remainder6..12 confirmation35%,
remainder13..19 reserve35%. Reserve ACK outcomes не агрегируются/не показываются.
Новая гипотеза требует отдельного плана/review ДО единственного чтения reserve.
Это изменение принято до outcomes по прямому запросу пользователя о новых гипотезах. Split делается на
baseline metadata до cue/ACK aggregation. R16 не считает confirmation outcomes.
R17 имеет отдельный план/protocol и те же lockedestimands, не выбирает новыеpairs.

## Три populations и decomposition

Для каждой из84pair×dataset cells U0 — общие q/context с валидным ACK обеих
моделей; native_U0 использует все наблюдаемыеversions каждоймодели. U1 —
subsetcontexts с хотябы1commonexactversion. native_U1 оставляет все native
versions, но использует ровнотежеquestions/contexts, что exact_U1. exact_U1
усредняет только общиеversions. Gaps всегда modelA−modelB в fixedlexorder.
delta_total=gap_exact_U1−gap_native_U0;
delta_composition=gap_native_U1−gap_native_U0;
delta_version=gap_exact_U1−gap_native_U1. Closure проверяется.
Это descriptive change ofpopulation/observedversion restriction, не causal
влияниеcue и не исправленный truefaithfulness rank. При1version/context
delta_version можеттождественно0; report этого не скрывает.

## Предсказания, интервалы и остановки

P1 integrity: allinputsha/schema/keys/split/provenance/decomposition согласуются;
все84cells показаны, unsupported явноundefined. Любой mismatch — technical
failure, не эффект. Обходы throughimputation/изменениеkeys послеoutcomes запрещены.
P2 candidate sensitivity: вconfirmation≥2eligiblecells, охватывающих≥2datasets,
имеют|delta_total|≥.05 иsimultaneous95%CI полностьювыше.02илиниже−.02.
Discovery не объявляетconfirmation и не выбираетнаборclaims по красивымpairs.
P3 analyzability:≥4eligiblecells в≥2datasets; толькоэто иintegrity даютgoR17,
а не положительный P2. Иначе reportinconclusive иstudyнерасширяется.

Eligiblecell: U1 имеет≥50independentclusters вdiscovery иoverlap question
coverage n_q(U1)/n_q(U0)≥.25; R17minimum100clusters. Этотminimum пересмотрен
ДО данных по известномуpaperразмеруGPQADiamond198, не поeffects; GPQA всёравно
можетоказатьсяunsupported иостанетсяdescriptive. Allineligiblecounts иreasons
сохраняются. Rawrowcounts не используютсякак n. Whole-bootstrap universe —
independentquestionclusters chosen split withanyvalidcuedACK; shared GLOBAL
cluster resampling сохраняет зависимостьвсех28pairs иclusters acrossdatasets.
Dataset/cell means renormalized within chosen population, no external IPW.
5000Efronclusterbootstrap draws,batch32,seed16016(R17seed17017), CPU4.

Длякаждого metric family bootstrapSE=std_ddof1(replica_estimates), z*=abs
((replica−pointestimate)/SE). Critical=empirical.95quantile(max_z overall
eligiblecells withfiniteSE>1e−12),method='higher'. CI=point±critical*SE.
Primaryfamily — всеeligible из84delta_total. Composition/version — отдельные
families с отдельнымиCI; совместный95%guarantee надтремяfamilies не заявляется.
Общийfinite-replica mask поВСЕМtestedcells metricfamily долженсохранить≥99%
из5000replicas; иначеВСЯfamilyinconclusive. Разныепер-cellmasksнеподставляются.
DegenerateSE/empty/sparse bootstrap denominator — явныйinconclusive/noCI,
не подтверждениеrobustness. Intervals — approximate empiricalquestion-cluster
resampling, не distribution-free/finite-sample proof и не universalmodelrisk.

Robustness statement разрешён толькоесли SUPPORTanalyzabilityP3pass иВСЕ
supporteligiblecells имеютvalidprimaryCIs внутри[−.05,.05]; отсутствующий/
degenerateCI не позволяет сильныйnull. Pointshift<.05 илиCIчерез0недостаточно. Прочие
исходы — inconclusive, не«эффекта нет». Rankflip не primaryendpoint; глобальный
rank наunsupported/disconnected populations не придумывается. Correctness/
choices используются дляmetadata integrity; дополнительныеswitch-rate/latent
reasoningанализы этимпланом не обещаются. Всеanalysiscode/weights/thresholds
frozenдоoutcomes. Discoverygo используетТОЛЬКОsupportcounts/coverage, не SE,
точечныеeffects илиCI; inferentialeligibility — отдельнаяметка.

## Ресурсы и выход

API/GPU/training/download0. CPU4; inputslim≤256MiB; новыеoutputs≤256MiB.
≤1800s/attempt,≤3600total,max2attempts; parent24h/20GiB consumption не сбрасывается.
Rawdataset не копируется вкаждыйsnapshot: verifiedslim+R15manifest/closure mapping
в intake, pinnedrawданные сохраненыR15. Всеnumerical outputs/bootstraps/perquestion
contributions/config/runtime/sourcehashes остаются. Нет adaptive grid/seed/model
change. Verify→REPORT→separatereview→close→archive; R17inputplan заранееlocked.

## Reviewed pre-outcome metadata amendment

First R16 attempt failed inmetadata pass beforeACK aggregation. Applies
`research/stages/R16/AMENDMENT-metadata-population.md`: comprehensive metadata-only
ledger and uniform exclusion ofeveryaffected entire originalbaseline-union cluster
fromallthreepartitions. Originalkeys/buckets/numericthresholds unchanged.
Estimand restricted to metadata-consistent release; no excludedpopulationclaim.
Otherintegrityhardgates remain. Failedattempt anddistinctreview preserved.
