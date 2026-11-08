# Contrastive cues, shared interventions, and question retention

Source-only design note, 8 November 2026. No model-answer, ACK or factor-score
values were inspected for this note. The symbols below are hypothetical;
they do not describe observed models. This is an elementary common-support
argument, not a claim of a new theorem or a first discovery of selection bias.

## Assumptions and source scope

Fix one question, one realized baseline episode per model, and a common answer
space A with K>=2 options. Answer identities must be aligned across models:
the same letter cannot denote different options. Let b_m in A be the valid
recorded baseline answer used to assign model m's cue. Freeze that reference;
stochastic reruns or a different parser can produce a different design.

The [official adaptive generator](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/scripts/data_processing/generate_adaptive_datasets.py)
forms the available letters and removes its supplied baseline answer before
selecting a target. This verifies the code's intended constraint t_m!=b_m
when b_m is valid. It does not prove that every released row used this exact
code/configuration or that its parsed reference matches an independently
adjudicated final answer.

[Meek et al., Table 2 and Section 3](https://arxiv.org/pdf/2510.27378v2)
explicitly describe contrast with the model's own baseline. Their acknowledgment
score includes cued responses without answer changes. The adaptive policy is
intentional, and its policy-level estimand is legitimate.

## A support obstruction

For model m, the permitted contrastive targets are

\[
T_m=A\setminus\{b_m\}.
\]

Let B={b_m:m in M} be the set of distinct baseline answers across the compared
models. A target shared by every model, while remaining contrastive for every
model, must lie in

\[
\bigcap_{m\in M}T_m
=\bigcap_{m\in M}\left(A\setminus\{b_m\}\right)
=A\setminus B.
\]

The second equality follows directly: an option belongs to every T_m exactly
when it differs from every b_m. The shared contrastive target set therefore
has size K−|B|. It is empty if and only if the baselines cover all K options.
This needs no randomization, uniformity or independence assumption.

For a two-option question and two disagreeing baselines, B=A, so no shared
contrastive target exists. Adding more compared models can only shrink the
intersection. A method cannot simultaneously retain every such question,
show every model the same target, and require that target to differ from every
model's baseline. Weighting cannot create a target in an empty set.

| Hypothetical two-option baselines | Allowed shared contrastive targets |
| --- | --- |
| A, A | {B} |
| A, B | Empty |
| B, B | {A} |

Equal target identities alone do not establish equal rendered interventions:
choice order, cue wording and formatting also matter. Exact visible-input
identity is a further requirement; it does not equate hidden model states.

## Contrastive does not mean objectively wrong

If there is a well-defined reference answer y in A, a contrastive cue is
necessarily wrong only when b_m=y. If b_m!=y, the correct answer y remains an
allowed contrastive target. Thus opposition to a recorded baseline and
objective incorrectness are separate design properties.

Demanding a shared target that is also objectively wrong gives the stronger
condition

\[
t\in A\setminus(B\cup\{y\}).
\]

This set can be empty even when A\B is nonempty. Conversely, a shared wrong
target chosen from A\{y} may equal an incorrect model baseline. These are
different evaluation questions; neither should silently replace the other.
Where reference labels are ambiguous, the adjective "objectively" is not
justified by a stored answer key alone.

## Counterfactual design choices and cost

Let M denote the number of models and C the number of cue types. Counts below
are cued evaluations per question, excluding the M baseline evaluations and
any repeated sampling.

| Design | What it preserves | What it gives up | Cued cost |
| --- | --- | --- | --- |
| Native per-model adaptive target | Every valid cue opposes that model's baseline; all questions can be retained | Same-target intervention across models | M C |
| Coordinated shared target independent of individual baselines | One common intervention and the full question population | Some models receive a target equal to their baseline | M C |
| Shared target selected from A\B | Common target and opposition to every baseline | Questions with empty intersection; resulting scope is conditional | M C on retained questions |
| Fully crossed K targets for every model | Every model is observed under each common target; distinct target policies can be reported | Not every evaluation is contrastive; greater cost | K M C |

A coordinated target could be chosen with a frozen question seed. If a reliable
y is available, selecting a shared target from A\{y} ensures a wrong cue, but
still does not ensure opposition to every baseline. Because acknowledgment
can be measured even without switching, a same-baseline target need not be
discarded when the intended estimand is response to a common cue.

A fully crossed experiment must include the **cued** t=b_m condition; an
uncued baseline is not its substitute. Crossing all K targets allows a common
fixed-target policy to be evaluated without adaptive target-availability
selection. It also allows each model's own contrastive policy to be evaluated
separately. It does not make a logically impossible all-model-contrastive
shared target exist when B=A.

## Interpretation and practical use

Native adaptive-policy comparisons and common-intervention comparisons answer
different questions. Conditioning on a shared contrastive target can change
the mix of questions and baseline agreements. Report both the estimand and
coverage rather than naming one universal model ranking. This is consistent
with [OpenAI's discussion of common-instance versus behavior-class support](https://alignment.openai.com/beyond-gmean-squared/).

The proof establishes a design constraint, not a large empirical effect.
Small or inconclusive observed shifts do not refute it, and it cannot be used
to manufacture evidence that shifts matter in practice. A useful evaluation
specification should choose its priority explicitly: native contrastive
challenge, common intervention, or a crossed design supporting both distinct
questions. The appropriate choice depends on the monitoring use case and
budget. No new model calls or experiments are authorized by this note.
