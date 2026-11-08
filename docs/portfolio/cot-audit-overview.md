# Portfolio case study: comparison support in CoT evaluations

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
