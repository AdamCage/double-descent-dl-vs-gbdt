# Priority check: exact-input support audit of the Meek release

8 November 2026. Targeted source-only review of
[the current manuscript](../../manuscript/cot_input_comparability.md).
No reserve data, local model responses, ACK/factor values or experiments were
accessed. The numerical findings below are the manuscript's reported results,
not a fresh calculation. Analysis/source files remain unchanged.

**Assessment:** a concrete empirical case study and reproducible audit appear
defensible. The general selection/comparability mechanism is strongly
anticipated. An exact duplicate of this particular release-level analysis was
not identified in the inspected material; this does not establish priority.
The right novelty claim is narrow and provisional.

## Closest primary prior

| Primary source and reading scope | Already established | Difference still plausibly specific to this report |
| --- | --- | --- |
| [Meek et al., v2, 30 November 2025](https://arxiv.org/pdf/2510.27378v2): main text, Table 2, Figures 6–7, Appendix C calibration, D figure descriptions and E definitions/table structure; [author project page](https://ajmeek.github.io/cot_monitorability_website/) and [pinned generator](https://github.com/ajmeek/measuring_cot_monitorability/blob/7da0cfb310db80b71d842d310b9c44c9b5e8ee37/scripts/data_processing/generate_adaptive_datasets.py) | Baseline-contrastive cues are intentional. Acknowledgment includes non-switch responses. Cue/dataset differences and correctness-transition strata are already analysed. | The inspected sections do not present our native versus pair-specific exact-input-support gap audit. This is a reading-scope statement, not proof of absence in all versions or related work. |
| [Dilgren and Wiegreffe, 17 August 2026](https://www.lesswrong.com/posts/z9fPtghFxELBXebY3/evaluating-chain-of-thought-monitorability-is-still-an-open): original research commentary, intervention-evaluation section and comparison tables | They explicitly warn that model-specific retained datasets impede cross-model comparisons and empirically demonstrate severe filtering in OpenAI's intervention evaluations. They also distinguish an intervention proxy from detecting harmful behavior. | Their reported experiment concerns OpenAI's filtering/evaluation suite, not the specific Meek-release acknowledgment comparison reported here. This prior must be cited; the general warning is not ours. |
| [Guan, Xie and Baker, 20 October 2026](https://alignment.openai.com/beyond-gmean-squared/): pair matching, synthetic experiments, historical results, behavior-class limitation and appendix | Shared-instance comparisons, composition-driven apparent ranking changes, relative rankings, support limits and uncertainty about causal attribution are explicit. They already caution that equal instances need not imply equally difficult behavior. | Our additional empirical object is a baseline-adaptive cue release and its observable exact-input support. The quoted effect sizes and supported-cell results are specific measurements, not a new matching principle. |
| [Chen et al., 8 May 2025](https://arxiv.org/pdf/2505.05410), Sections 2–3 | Hint-conditioned disclosure tests and filtering on answer changes predate both our report and Meek's all-cued acknowledgment proxy. | Our outcome is the released annotation on all eligible cued responses; it is not Chen's hint-use-conditioned causal quantity or a new faithfulness definition. |

The August commentary cites the Meek causal-factor work in its taxonomy and
acknowledges feedback from a Meek coauthor. It should not be characterized as
unaware of that research. Its explicit model-specific-subset criticism is a
particularly relevant omission from the present reference list.

Additional 2026 screening included [MonitorBench](https://arxiv.org/abs/2603.28590v4)
(abstract/taxonomy) and [Efficient Reasoning Training Does Not Always Harm CoT
Faithfulness and Monitorability](https://arxiv.org/html/2610.03509v1)
(abstract and Sections 3–4). These address conditional monitorability and
different training/evaluation questions. This limited screening did not reveal
the same Meek-release audit; it was not a complete audit of their appendices.

## What the reported finding adds—and does not add

The manuscript reports 16 of 56 supported confirmation comparisons passing its
materiality rule, shifts up to 15.52 percentage points, and zero within-support
version components. Taken with the frozen full grid and reproducibility
artifacts, these are useful release-specific evidence that restricting support
can materially change this historical acknowledgment comparison.

They do not show a new selection phenomenon, sixteen independent discoveries,
an error in Meek's legitimate adaptive-policy estimand, a corrected ranking of
internal faithfulness, or better deployment monitoring. The pair comparisons
share models and questions. Their count describes a correlated comparison
grid. Statistical materiality is not a novelty certificate.

In particular, the zero version component follows from one observed version
per model/question/cue kind. The result is about the composition of the retained
population, not an identified effect of changing a cue on a fixed question.
The elementary common-target support identity supplies a design explanation,
but is not a new theorem. More plots or resamples cannot recover the absent
counterfactual versions.

## Positioning and manuscript action

Keep the current narrow empirical framing. Add the August commentary beside
the October OpenAI prior, and explicitly describe the report as an empirical
case-study extension of established comparability concerns. The current
disclaimers about source weights, judge annotations, causal attribution and
unexecuted H1 confirmation are appropriate. Do not claim to overturn the
source paper's combined monitorability table: that estimator is not reproduced.

For a portfolio, the strongest evidence is research judgment plus a complete
working artifact: a precise question, an acknowledged failed formulation,
held-out analysis, source-integrity diagnosis, independent reconstruction and
portable replay. This is materially better evidence than a large figure count
alone, provided the portfolio owner can explain and defend the work.

For a journal, my judgment is more cautious: current priority is unverified and
the conceptual increment is modest. A research note is a credible positioning;
readiness for a strong journal is not established by these results. A valuable
next extension would test a coordinated/shared or crossed-target design on
fresh questions, or reproduce the audit on an independent evaluation with
validated annotations. That would test a practical remedy or generalization,
rather than only illustrate the known support problem. None of those
experiments has been performed or authorized by this note.

## Search scope and uncertainty

Searches combined the exact paper title/arXiv identifier, Meek, adaptive cues,
baseline answers, common/shared inputs, matching, selection and reanalysis.
Only primary papers, original author/research posts and official repository
material support the assessment above. The official repository's
[all-issues API](https://api.github.com/repos/ajmeek/measuring_cot_monitorability/issues?state=all&per_page=100)
returned no entries; its [commit history](https://github.com/ajmeek/measuring_cot_monitorability/commits/main/)
and author project page did not expose a matching reanalysis. The HF
[discussion index](https://huggingface.co/datasets/ameek/measuring_cot_monitorability_transcripts/discussions)
showed a conversion-bot item. These are weak checks, not evidence of world-first
priority. Search snippets that surfaced dataset rows were not used as evidence
or analysed; no local reserve outcomes were opened.

Unindexed work, unpublished drafts, other terminology and subsequent versions
remain possible. No author contact or external peer review was performed.
**Status: specific empirical contribution plausible; general mechanism known;
priority and journal acceptance not certified.**
