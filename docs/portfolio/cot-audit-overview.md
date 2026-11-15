# CoT Audit: are we comparing models on the same inputs?

An AI-assisted research case study in evaluation validity and the measurement
foundations of scalable oversight. Based on the completed report
[Exact-Input Support Changes Cross-Model Comparisons](../../manuscript/cot_input_comparability.md).

## In thirty seconds

Two models can answer the same benchmark question yet receive different hints.
If each hint depends on the model's original answer, a difference in how often
models acknowledge that hint can also reflect which inputs are being compared.
This audit asks how the reported gap changes when inputs must match exactly.
On held-out questions, 16 of 56 supported comparisons met the fixed change
criterion; the largest shift was 15.52 percentage points. The practical output
is a reproducible check of comparison support, not a ranking of reasoning quality.

## Why this is relevant to Anthropic and scalable oversight

Anthropic studies whether a model's stated chain of thought faithfully reflects
the factors influencing its answer, including whether it acknowledges hints.
See its primary research on
[CoT faithfulness](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)
and [unacknowledged hints in reasoning models](https://www.anthropic.com/research/reasoning-models-dont-say-think).
Its [technical safety research directions](https://red.anthropic.com/2025/recommended-directions/)
also identify detecting or ensuring CoT faithfulness as a research interest.
These sources were consulted for this portfolio framing on 15 November 2026,
not used to redesign the completed experiment.

The connection is methodological: before using a CoT acknowledgment metric to
compare models or choose an oversight strategy, establish what population and
intervention each score measures. This project checks that prerequisite. It
does not evaluate an operational oversight system, establish monitor detection
performance, or demonstrate faithful internal reasoning. The fit to Anthropic's
research is an interpretation of these public sources, not an affiliation,
endorsement or claim about a current hiring requirement. The audited dataset
is the public Meek et al. release, not an Anthropic evaluation commissioned for
this project.

## Read or reproduce

- [Research report](../../manuscript/cot_input_comparability.md): definitions, inference and limitations.
- [Evidence release](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/tag/research-cot-audit-2026-11-08): PDF, portable data/code, gallery and replay.
- [Recovery and asset hashes](../releases/research-cot-audit-2026-11-08.md): choose the portable ZIP and follow its replay instructions.
- [Claim ledger](../../research/claims.json): exact evidence and review bindings.

## The question

When evaluations adapt a cue to each model's baseline answer, how much can the
observed difference between models change if the comparison requires identical
released inputs? This project audits that question on public cached evaluations,
with no new model calls, paid APIs, GPU jobs or training.

## The result

The preregistered test used previously unused question clusters. Sixteen of 56
supported comparisons qualified under the fixed rule, across BBH and MMLU.
The largest change was 15.52 percentage points. In that BBH example, Qwen3 minus
QwQ changed from +4.20 points on native support to -11.32 on exact common support;
the approximate bootstrap interval for the change was [-19.20, -11.84] points, simultaneous across the
56 primary comparisons. All 28 GPQA comparisons lacked the frozen support minimum.

This concerns a historical released judge annotation. It does not identify
which model has more faithful internal reasoning or a corrected overall ranking.
All observed version components were algebraically zero; population composition
changes explain the reported decomposition, without causal attribution.

## What the artifact demonstrates

- Translating a measurement concern into explicit populations and estimands.
- Preserving a failed integrity attempt and an initially overconditioned question.
- Changing the question before using held-out confirmation, with fixed numeric rules.
- Handling aliases, dependent model pairs, missing support and simultaneous intervals.
- Verifying every numerical result independently and providing a portable replay.
- Presenting all comparisons, including unsupported and inconclusive outcomes.

The general matching/selection issue was already described in prior work.
The contribution is a specific, quantified audit of this pinned release and
usable evidence artifacts. External priority and readiness for a strong journal
are not established. A crossed-target or independent-benchmark extension with
validated labels would strengthen a future journal submission.

## Ownership and preparation

The human supplied the motivation, research direction, deadline and resource
constraints. Codex agents implemented, analysed and separately reviewed the work.
This is an AI-assisted project, not evidence of independently written human code.
The record does not establish that the human personally selected every estimator,
designed the bootstrap, authored the code or reproduced the results. Separate
reviewer agents checked agent-produced work; this is internal team review, not
external peer review or an independent experimental replication.

### Portfolio wording supported by the current record

> I directed an AI-assisted audit of cross-model CoT evaluation comparability,
> setting the research direction and resource constraints. Codex agents carried
> out implementation, statistical analysis and separate internal review. The
> held-out audit found material gap changes in 16 of 56 supported comparisons
> when requiring common exact inputs, with a largest shift of 15.52 percentage
> points. The result concerns a released acknowledgment proxy and comparison
> populations; it does not establish causal faithfulness or a true model ranking.

For a CV:

> Directed an AI-assisted CoT evaluation audit: 16/56 supported held-out
> comparisons met a fixed gap-change criterion (maximum 15.52 pp); released
> reproducible evidence, code and distinct internal agent reviews.

Personal claims about scientific problem formulation, methodology selection,
critical decisions or interpretation should be added only for decisions the
owner actually made and can explain with concrete examples. Understanding a
method after delivery does not establish authorship of its original design.

Before presenting it, the portfolio owner should run the replay, inspect the
source tables and be able to explain:

1. Why a common benchmark question does not imply a common cue or population.
2. Why the five-point estimate threshold and the two-point interval band differ.
3. Why the zero version component is an algebraic identity rather than a causal finding.
4. Why GPQA remains unsupported and why model-pair rows are dependent.
5. Why H1 changed, and how the new confirmation data remained unused for design.
6. What computational reproduction establishes beyond, and separately from, scientific validity.

The English research report, 111 figure PNG/SVG pairs, eight complete CSV tables,
input/protocol hashes, all bootstrap arrays, reviews and replay script are the
deliverable. The 35% reserve outcomes remain unused for future independent work.
