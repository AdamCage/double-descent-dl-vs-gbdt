# R17D — exploratory H2 sensitivity, awaiting untouched confirmation

Author `/root`; implementation `/root/alignment_brainstorm`; distinct reviewer
`/root/reset_critic`. Attempt `20261108T095410-f87b41e0ef` completed both tasks in
28.735s, analysis27.031s. Both H2 plans/source were independently frozen before
this run. Context is explicitly `scope_only`, the cue kind. R16's narrower H1
and the unexecuted superseded H1 confirmation plan remain preserved.

These are the original30%discovery clusters already inspected for H1, so this
is exploratory reuse, not independent confirmation/replication. Neither the
original35%confirmation nor35%reserve ACK/factor outcomes were used. Numeric
thresholds, all84cells, weights,5000bootstrap draws and seeds were unchanged.
No API, GPU, new data acquisition, model calls or training.

## Complete discovery results

| Dataset | Support-eligible | Primary-CI-eligible | Material cells | Delta_total range | U1/U0 question coverage |
| --- | ---: | ---: | ---: | --- | --- |
| BBH |28|28|10|[-.142996,.124487]|[.586458,.888770]|
| GPQA |1|1|0|[-.102222,.119365]|[.178571,.809524]|
| MMLU |28|28|7|[-.095553,.121365]|[.308208,.352683]|

All84cells are retained. Overall57support-eligible,57valid primary intervals,
17material cells across BBH/MMLU. A material cell has abs(Delta_total)>=.05
and its endpoint-specific simultaneous95%bootstrap interval wholly outside
[-.02,.02]. This is the fixed descriptive discovery rule, not a confirmed H2.
The structural discovery gate is true:>=4support cells across>=2datasets.
Go was based on integrity/support only, not the positive observed effects.

P1 integrity and decomposition supported subject to separate postrun audit.
P2 H2 remains `not_evaluated_on_discovery`; sensitivity_rule_pass=true is an
exploratory observation only. P3 support gate supported. P4 blanket robustness
is inconclusive and is not claimed. GPQA's unsupported cells remain descriptive;
the confirmation minimum100clusters is not relaxed. Separate endpoint/split
families do not supply joint programme-wide95%coverage.

The complete pre-outcome metadata conflict ledger and exclusions are unchanged.
The comparison uses equal released core-question weights, not the source paper's
subdataset-macro table estimand. Exact matching changes observed question and
cue-kind composition; it does not identify a causal cue effect, corrected true
ranking, latent faithfulness or modern-model behavior. Native challenge policy
is intentional, not a demonstrated flaw in the source study.

Evidence: `tasks/ack-audit/audit/{analysis.json,dataquality.json,pairs.csv,
per_question.parquet,bootstrap.npz,protocol.json}`, unit checks and frozen
source/input/lock inventories. All predictions, unsupported comparisons and
raw contributions are preserved. Independent numerical audit and operational
review are required before closure.

Next command after accepted review/closure: run the ALREADYFROZEN R17H2
confirmation on its statistically untouched35%. No changes to the model grid,
context definition, thresholds, seeds, support or reporting from these results.
If confirmation fails or is inconclusive, disclose it and use a separately
reviewed reserve question if appropriate; never relabel discovery as confirmation.
