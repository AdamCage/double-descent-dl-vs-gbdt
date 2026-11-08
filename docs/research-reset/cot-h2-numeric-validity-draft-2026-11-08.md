# H2a draft: material impact of numerically inadmissible factor scores

Source-only proposal, 8 November 2026. This is not a frozen PLAN and does not
authorize reserve access. No reserve scores, ACK, generated answers or traces
were inspected to write it. H1 remains the current hypothesis.

## Question and prior knowledge

Can definitely inadmissible numeric factor grades materially change a
model-comparison conclusion on the cached release? The target is the stability
of a reported score, not latent reasoning faithfulness, safety or a newly
discovered general problem with LLM judges.

[Meek et al., Appendix C/E](https://arxiv.org/pdf/2510.27378v2) already discuss
judge overcounting and define verbosity as a fraction. Their
[factor scorer](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/src/measuring_cot_monitorability/scorers/factor_utilization.py)
allows an unbounded ratio; XML and fallback parsing use potentially different
denominators. Zero factors have an explicit special-case score. The prior HF
schema/viewer inspection also exposed aggregate numeric ranges. Therefore
**the mere existence of invalid values is not a blind new finding**.

The potentially useful claim concerns material comparison impact under a
fully declared, conservative correction sensitivity analysis. This is a
concrete audit of this release, not a new bounding method. Journal novelty is
uncertain; an isolated parser defect would primarily be an engineering result.

## Population and independence

Use only baseline rows from the untouched reserve partition, with original
baseline-union cluster assignments and the uniform metadata quarantine fixed
before H1 outcomes. ACK and cued responses are unnecessary. Keep all eight
models and all 28 pairs in all three datasets; preserve unsupported cases.

Before V access, verify factor-list identities/field semantics using code and
design metadata. Differences in factor lists across models require an explicit
restricted comparison population, rather than silently comparing different
grading rubrics. Missing or inconsistent source-field mapping is a stop gate.
Repeated generations are averaged within the declared baseline version; then
versions and released questions receive fixed equal weights. Clusters describe
dependence; aliases must not be counted as independent observations.

## Validity rules to freeze

1. A reported fraction must be finite and lie in [0,1]. Values outside this
   domain are definitely inadmissible for the declared fraction metric.
2. After confirming the export mapping, used/total must be integral,
   nonnegative and satisfy used<=total for positive total. The denominator
   should agree with the supplied factor-list length.
3. Compare rate with used/total under a fixed numeric tolerance. A mismatch is
   a definition/serialization inconsistency, not proof of a false semantic
   judgment. Use it as an invalid-score rule only if source mapping is verified.
4. No-factors, missing-basis and missing-rate cases are named separately.
   The scorer's intentional no-factors convention must not be relabeled a
   discovered bug. Their inclusion in the unknown-score sensitivity set must
   be declared before reserve inspection.

A numerically valid grade can still be semantically wrong. This study cannot
measure that error without independent annotation. Do not claim otherwise.
Keep the raw grade and every reason flag; never silently clip or substitute a
preferred corrected value.

## Estimand and falsifier

Let valid recorded grades remain fixed. Replace every definitely inadmissible
or explicitly unevaluable grade by an unknown number in [0,1]. Propagate those
intervals through the fixed averaging weights. For model m this gives [L_m,U_m].
For pair A−B, the sensitivity envelope is [L_A−U_B,U_A−L_B]. It is conditional
on the recorded admissible grades; it is **not a confidence interval for true
factor coverage**.

Compare the native finite gap g with its envelope. The compulsory correction
distance is dist(g,envelope), zero when g lies inside it. A proposed materiality
threshold is .05. A stronger finite-release result would require compulsory
distance>=.05 in at least two predeclared cells covering two datasets. The
threshold and global rule need reviewer approval before use. Native nonfinite
means are undefined, with an explicit failure/coverage report, not imputed zero.

An envelope that merely allows a rank reversal supports only possible
sensitivity; it does not show that the ranking is wrong. Small gaps within the
declared tie band are not meaningful rank reversals. Bootstrap sampling
uncertainty, if requested, is an additional layer and needs its own frozen
reviewed protocol; a finite cached-corpus sensitivity statement can be reported
without claiming generalization to new questions or contemporary models.

**Falsifier/stop:** no qualifying numeric violations, no compulsory material
change, or insufficient support means H2a is unsupported. Preserve the result
and stop this branch. Do not broaden the invalidity definition, change the
threshold, add models or spend the reserve again to obtain a positive result.

## Practical output and cost

An invariant validator, complete violation ledger, model/pair sensitivity
envelopes, retained/excluded denominators and reproducible numeric tables.
Useful figures include validity rates by reason/model, raw versus bounded
score envelopes, all-pair gap envelopes, compulsory correction distances,
coverage and clustering summaries. Caption every interval as measurement
sensitivity rather than causal/sampling certainty.

All required scalar/list fields and text-length descriptors are in the pinned
24-file release. CPU only, no API, new data download, training or manual
relabeling. The reserve is consumed once under a new distinct PLAN/review.
This can produce a useful validation artifact today; it does not guarantee a
substantial positive finding or a journal-ready paper.

## Other alternatives considered without outcome access

H2b: reconstruct whether metadata baseline_model_answer and cue_target_answer
agree with the released baseline response reference. The generator searches
the first answer pattern in a string representation; a mismatch could also
reflect attempt drift or a reference parser issue. It would be a coherence
audit, not proof of the cause. Do not execute alongside H2a on the same reserve.

H2c: adaptive target support itself depends on baseline agreement. If each
model excludes its own baseline letter, two-choice questions with different
baselines have no shared admissible target. Under *independent uniform*
selection among K−1 alternatives, common-target probabilities are 1/(K−1)
for equal baselines and (K−2)/(K−1)^2 otherwise. The source code's seeded RNG
does not establish actual independence or the policy of every released row.
The support obstruction is analytic; the probability formulas require their
assumption. This specializes a known positivity problem and needs a clear
practical design recommendation to become a useful empirical contribution.
