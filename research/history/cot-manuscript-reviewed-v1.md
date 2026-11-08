# Exact-Input Support Changes Cross-Model Comparisons of a CoT Acknowledgment Proxy

Research report - 8 November 2026

AI-assisted research audit. Evidence, code, internal reviews and a numerical replay accompany this report. This work has not received external peer review.

## Abstract

Cross-model evaluations can share benchmark questions while presenting different interventions to each model. We audit this distinction in a public release of chain-of-thought evaluations, where cues are adapted to models' baseline answers. Before a held-out analysis, we fixed all eight historical models, 28 pairs, three datasets, estimands, support requirements and a simultaneous bootstrap procedure. On previously unused confirmation questions, 16 of 56 supported comparisons met the prespecified materiality criterion: a gap change of at least five percentage points with its simultaneous interval outside a two-point band. The largest change was 15.52 points, with an approximate simultaneous 95% interval of 11.84 to 19.20 points in magnitude. All 28 GPQA comparisons lacked the required confirmation support. The observed changes enter the population-composition component; the within-support version component is algebraically zero in this release. This is evidence that the comparison population matters for this historical judge annotation, not a causal estimate or a corrected ranking of faithful reasoning.

## 1. Question and contribution

Meek et al. evaluate cue acknowledgment and factor coverage as proxies relevant to CoT monitorability. Their adaptive cues deliberately challenge each model relative to its own baseline answer. This is a legitimate evaluation policy, but a common question and cue kind do not ensure an identical input across models. Our outcome is the released acknowledgment boolean, including cases where the answer does not switch. We do not analyse the combined monitorability score or judge internal reasoning. [Meek et al., 2025](https://arxiv.org/abs/2510.27378v2).

The contribution is a specific empirical audit: how much do model-pair acknowledgment gaps change when native observed versions are restricted to common exact inputs, and how much support remains? Pair matching and selection effects are established ideas. OpenAI's complementary monitorability metrics already compare shared instances and discuss model-dependent instance selection. We claim neither a new matching method nor the first recognition of this problem. [Guan, Xie and Baker, 2026](https://alignment.openai.com/beyond-gmean-squared/).

## 2. Data, provenance and protocol evolution

We pinned the author release at revision `a7a8b25ae467db9d80d63e55ac053b48753b2d36`. Its 24 Parquet files contain 446,208 rows from eight historical 2025 models on BBH, GPQA and MMLU. All 940,116,134 payload bytes matched the declared hashes. A compact table retained every row, 21 analysis fields, exact UTF8 hashes and lengths for three large text fields, ordered-choice hashes, source payload hashes and row identities. The compact input is 88,613,869 bytes. No new model calls, training, GPU use or paid APIs were required. The original release and its licence remain the data authority. [Pinned author dataset](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts/tree/a7a8b25ae467db9d80d63e55ac053b48753b2d36).

Before acknowledgment analysis, a complete metadata audit found four BBH IDs in two baseline-input clusters with conflicting reference answers for identical visible problems. A uniform rule excluded both entire clusters, comprising 192 rows, from every analysis partition. The conflicts' upstream cause is unknown; they do not prove that acknowledgment labels are wrong. Every excluded ID and reason is retained.

An initial discovery question, H1, conditioned on both cue kind and literal cue text. Because the text contains the target answer, that question already conditioned on common targets. It did not answer the intended native-policy question, and its confirmation plan was not executed. Its failed integrity attempt, successful narrow analysis and unexecuted confirmation plan are preserved. H2 instead conditions on cue kind only. Both H2 plans, code and thresholds were frozen before H2 discovery. H2 discovery reused inspected questions and is explicitly exploratory; confirmation acknowledgment values had not been used for design. This overconditioning could have been recognised from source before the first run. The revision is disclosed rather than presented as a retroactive repair of H1; no H1 confirmation or general robustness is claimed.

## 3. Three comparison populations

A question is the file-parent dataset plus released core ID. A context is the cue-kind `scope`. An exact version consists of the released input hash, ordered-choice hash and target answer. Literal cue text is not a native-population conditioning key. Equality concerns released text, not inaccessible provider or system context.

For each model pair and dataset, U0 contains common questions and contexts with valid acknowledgment labels for both models. The native U0 gap uses all observed versions for each model. U1 retains contexts with at least one exact version shared by the models. The native U1 gap still uses all native versions, now on exactly those questions and contexts. The exact U1 gap uses only shared versions.

Let their signed model-A-minus-model-B gaps be g0, g1 and gE:

```text
Delta_total       = gE - g0
Delta_composition = g1 - g0
Delta_version     = gE - g1
Delta_total       = Delta_composition + Delta_version
```

Generations average within model/question/context/version, then versions within context, common contexts within question, and released question IDs. These equal-question weights differ from the source paper's subdataset-macro estimator. Aliased IDs retain their declared point weights but share uncertainty clusters. Missing labels are counted and excluded, never treated as false.

This decomposition describes changes in observed support. Composition can change both questions and cue-kind mixtures. It does not estimate the causal effect of changing a cue on an otherwise fixed population, impute missing counterfactuals or identify a true model ranking.

## 4. Held-out test and uncertainty

Questions sharing any nonempty exact baseline-input hash form global dependency clusters. A fixed hash assigns 30% to discovery, 35% to confirmation and 35% to reserve. Whole-cluster exclusions preserve the original assignments. Confirmation contains 3,268 retained IDs in 3,215 clusters. Reserve acknowledgment and factor-grade outcomes remain unused.

The complete grid is 28 model pairs by three datasets. Confirmation eligibility requires at least 100 independent exact-overlap clusters and 25% question overlap. The primary criterion requires at least two eligible cells spanning two datasets with absolute total shift at least 0.05 and a simultaneous interval wholly outside [-0.02, 0.02]. Discovery cannot select successful pairs or change this rule.

We use 5,000 global cluster-bootstrap resamples, sharing cluster multiplicities across all comparisons. Centered deviations are scaled by marginal bootstrap standard errors; the 95th percentile of the maximum absolute standardized deviation calibrates the total-shift family. All 5,000 joint-finite confirmation draws were retained; its family contains 56 comparisons and its critical value is 3.32979. Other decomposition endpoints have separate families. Zero standard error gives an unavailable interval, not proof of robustness. These are approximate empirical resampling intervals, not finite-sample guarantees, independent model reruns or coverage over the whole research programme.

## 5. Confirmation results

Exploratory H2 discovery had 17 material cells (10 BBH, 7 MMLU) among 57 supported comparisons. This is not an independent replication of H1 or confirmation of selected successful pairs. H2 passed the frozen primary criterion on confirmation. Sixteen comparisons qualified: nine on BBH and seven on MMLU. All 56 supported comparisons had usable primary intervals. GPQA had only 13 to 68 exact-overlap clusters per pair and all 28 cells remained unsupported under the fixed minimum. No cutoff was lowered.

| Dataset | Eligible comparisons | Material comparisons | Exact-overlap question coverage | Total-shift range, percentage points |
| --- | ---: | ---: | --- | --- |
| BBH | 28 of 28 | 9 | 56.49% to 87.98% | -15.52 to +13.58 |
| GPQA | 0 of 28 | 0 | 19.70% to 87.18% | -8.38 to +11.71, descriptive only |
| MMLU | 28 of 28 | 7 | 31.25% to 35.76% | -8.37 to +11.63 |

The largest absolute eligible shift is BBH, Qwen3-235B-A22B-2507 minus QwQ-32B. Its native gap is +4.20 points; on exact-overlap support it is -11.32 points. The change is -15.52 points, with simultaneous interval [-19.20, -11.84]. The native population contains 1,022 questions; exact support contains 616 questions in 615 clusters.

On MMLU, DeepSeek-chat-v3-0324 minus Qwen3-235B-A22B-2507 changes from -2.87 to +8.76 points. The change is +11.63 points, interval [8.87, 14.39], with 2,156 native questions and 728 exact-overlap questions in 722 clusters. These examples are the largest absolute eligible shifts in their datasets, chosen descriptively. The intervals already cover the complete primary family. We did not register intervals for the separate raw gaps, so their observed sign changes are not significance tests of each raw gap.

All 84 version components are exactly zero. Each model/question/cue-kind has a single observed version, so native U1 and exact U1 coincide on matching contexts. Consequently the total changes equal the composition changes. This algebraic fact does not establish that versions have no causal effect. The retained comparison population, rather than an identified within-context intervention effect, changes the observed gap.

## 6. A simple design constraint

There is also a useful elementary support fact. Let A be the common set of K semantic answers and let B contain the models' valid baseline answers. If each model permits only targets different from its own baseline, common admissible targets are exactly A minus B. Their number is K minus the number of distinct baseline answers. With two answers and two disagreeing baselines, no common contrastive target exists. No independence or random-target assumption is needed.

This is a conditional set identity, not a new theorem or a claim that every released row followed a verified randomization strategy. A target contrasting with a model's baseline is also not necessarily objectively incorrect. The constraint explains why native adaptive and shared-input questions can require different populations without either evaluation being intrinsically wrong.

A coordinated shared target keeps one evaluation per model and cue kind but can coincide with a model's baseline. A fully crossed design evaluates all K targets for every model, including the own-baseline condition. Its nominal call count rises from M times C to K times M times C per question. It supports several explicitly different estimands; it does not create a common contrastive target where none exists. These are design counts, not measured runtime savings.

## 7. Practical use and limitations

Before quoting an acknowledgment gap, report its native policy, weighting, shared-input support and restricted-population gap together. Preserve unavailable comparisons and cue-kind breakdowns. Decide whether the scientific question concerns a model-specific challenge or identical interventions; the same benchmark name does not settle that choice. The artifact supplies this complete audit for the pinned release, not a universal ranking service.

The main limitation is measurement: a released judge boolean is not ground truth about internal reasoning, and false values can include source parsing failures. We did not regrade traces or calibrate the judge. Curated benchmark questions, historical models and conditional resampling limit generalization. Exact input hashes do not verify hidden prompts. Conservative metadata exclusions narrow the population. No claim refutes the source's intentional adaptive estimand, reproduces its published aggregate table exactly or certifies external priority. A specific held-out empirical finding and a reproducible diagnostic are the contribution; journal acceptance remains a separate assessment.

## 8. Reproducibility and contributions

The delivery includes all 84 cells in both H2 phases, per-question contributions, all bootstrap arrays, source tables, input hashes, frozen protocols, exclusions, source snapshots, `uv.lock`, failed attempts and distinct internal reviews. The portable bundle contains the verified compact input and a detached numerical replay. Raw public payloads remain in the separately verified intake archive and can be recovered from their pinned manifest. Packages must be installed before offline numerical replay.

The human set the research motivation, portfolio objective, deadline and resource constraints. Codex agents implemented and analysed the audit; a distinct agent independently reconstructed every confirmation point, all 88,772 per-question rows, all 5,000 bootstrap draws and all three interval families. Internal review is not external peer review or proof of independent human authorship. The prospective portfolio owner must understand, critically review and be able to defend these choices and limitations.

## References and artifact entry points

- [Meek et al. Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity. arXiv:2510.27378v2, 2025.](https://arxiv.org/abs/2510.27378v2)
- [Pinned public evaluation transcripts, revision a7a8b25.](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts/tree/a7a8b25ae467db9d80d63e55ac053b48753b2d36)
- [Pinned official evaluation implementation, commit 7da0cfb.](https://github.com/ajmeek/measuring_cot_monitorability/tree/7da0cfb310db80b71d842d310b9c44c9b5e8ee37)
- [Guan, Xie and Baker. Beyond g-mean squared: complementary metrics for monitorability. OpenAI, 2026.](https://alignment.openai.com/beyond-gmean-squared/)
- [Complete figure and table gallery](gallery/index.html)
- [Portable numerical replay instructions](evidence/README.md)
