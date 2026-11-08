# H2 protocol: native adaptive and exact-input comparisons

Frozen before any H2 analysis after the H1 same-literal-cue discovery. H1 is
preserved, not retroactively repaired. Decision0007 and its child authorization
apply. Research question: how sensitive are cross-model released judge ACK gaps
to restricting a common cue-kind comparison to exactly shared released inputs?

## Units, estimands and populations

All8historical models and28unordered pairs from R15; file-parent datasets
BBH/GPQA/MMLU. Model order lexicographic, signs A minus B. ACK is the released
judge boolean on all has_cue=true cued responses, including non-switches.
Missing/nonboolean ACK is excluded and counted, never substituted false. A false
annotation can also reflect source judge parsing failure. No latent faithfulness,
causal cue effect, contemporary-model evaluation or exact paper-table reproduction.
The source publication macro-weights subdatasets; this study equal-weights
released core question IDs, a separately named population.

Question q=(file-parent dataset,core_id). Context c is **scope only**, the cue
kind. The literal cue text and target are NOT conditioning keys for native U0.
Exact version v=(SHA256 of exact released input, ordered choices SHA256, exact
cue-target answer). Equality excludes unavailable provider/system configurations.
Target letter alone is insufficient. Duplicate generations average within
model/q/c/v; then equal observed versions within c, equal common contexts within
q, equal released q. Aliased IDs retain that point weight but not independent n.

For each of84pair/dataset cells, U0 comprises q/contexts with valid observed ACK
for both models. native_U0 uses all each-model observed versions. U1 keeps only
contexts with at least one exact version shared by both. native_U1 uses ALL
native versions on exactly the same q/contexts as exact_U1; exact_U1 uses only
common versions. The latter need not represent the full native adaptive policy.
Gaps g0,g1,gE are averages under those definitions.

Delta_total=gE-g0; Delta_composition=g1-g0; Delta_version=gE-g1. Exact algebraic
closure is checked. Composition can change both questions and cue-kind mixtures.
With one observed version per model/q/context, Delta_version may be identically
zero; do not turn this identity into causal evidence. No corrected true ranking,
missing-counterfactual imputation, IPW, new matching theory or global rank across
unsupported populations. Separate cue kinds are not independent replications.

## Integrity, population amendment and data separation

Every source file, raw hash, row identity and one-model/file mapping must match
the pinned24-file release. Recompute ordered choices hashes; check choice
multisets/correct-text and order-appropriate reference labels. Every retained
cued q/model must have a nonempty valid baseline-input anchor; missing exact
version/design keys remain hard failures. Full metadata-only audit precedes H1
outcomes; its uniform conservative exclusion removed4BBHIDs/2wholebaseline-union
clusters (192source rows) with conflicting same-input reference metadata.
Do not repair labels or infer the source cause/ACK invalidity from this conflict.
Both H2 protocols bind the same exact audit SHA and complete exclusion ledger.

Dependency clusters union all q sharing any nonempty exact baseline-input hash,
globally across datasets, before exclusions. Preserve original canonical min-q
keys. For key k, h=int(SHA256(UTF8('20261108|'+k)),16)%20. h0..5 is discovery,
h6..12 confirmation,h13..19 reserve. Exclude whole flagged original clusters
uniformly; no rebucketing. R17D reuses30%discovery already inspected for H1,
explicitly exploratory. R17 uses35%confirmation ACK not previously aggregated,
displayed or used for design. Arrow can decode unselected values in batches;
statistical non-use is promised, not byte-unread files. Reserve35% remains unused
for ACK/factor grades. Metadata was inspected across all partitions and is not
an untouched outcome for a future metadata-prevalence hypothesis.

Both H2 plans, all numeric rules, source, model grid and seeds freeze and receive
distinct preflight BEFORE R17D. No confirmation-informed adaptation. R17D cannot
select positive pairs, shrink the grid, change seeds or tune practical thresholds.

## Predictions, uncertainty and stopping

P1 all integrity/decomposition/disjointness controls pass; mismatch is technical
failure. P2 primary confirmation: at least2eligible cells spanning at least2
datasets have abs(Delta_total)>=.05 AND a simultaneous95%bootstrap interval
wholly above.02 or below-.02. H2 discovery cannot confirm/refute this endpoint.
P3 analyzability: at least4SUPPORT-eligible cells across at least2datasets.
Discovery go depends ONLY on P1/P3, not point shifts, positive findings or SE.
If it fails, stop confirmation and report unsupported; do not lower requirements.
P4 robustness only if P3holds and ALLsupport-eligible primary intervals exist
inside[-.05,.05]. Missing/zero-SE intervals block a blanket robustness claim.
Otherwise outcome is inconclusive unless the primary sensitivity rule passes.

Support-eligible: at least50independent U1clusters in discovery or100confirmation,
question coverage n_q(U1)/n_q(U0)>=.25, finite estimands. GPQA may remain only
descriptive at this frozen support minimum. No raw-row-as-n or relaxed cutoff.

Global Efron question-cluster bootstrap5000draws,batch32. Shared global cluster
multiplicities preserve dependence across all84cells and cross-dataset aliases;
each cell/dataset mean renormalizes within its observed population. Bootstrap
universe contains selected clusters with any valid cued ACK. Seeds16016discovery,
17017confirmation. For each separate Delta endpoint family, require a JOINT
finite mask over all support-eligible cells retaining>=99%replicas. Otherwise
that family is inconclusive. For finite SE>1e-12, standardize centered replicas
by marginal bootstrap SE(ddof1); critical is the empirical.95quantile(method
higher) of the maximum absolute standardized deviation across all tested cells.
Intervals are point +/- critical*SE. SE<=1e-12 gives N/A, not proof of no effect.

Primary family is all eligible Delta_total cells from84. Composition/version
families are separately simultaneous; no joint95%coverage over endpoints,
phases, programme hypotheses or population safety. This is approximate empirical
cluster-resampling uncertainty, not a distribution-free finite-sample guarantee
or independent replication/new model run. All cells, counts, exclusions,
bootstrap arrays and per-question contributions remain in artifacts.

CPU4, API/GPU/training/download0, output<=256MiB per stage,1800s/attempt,
3600total,max2attempts. Cumulative parent24h/20GiB/2GiBdownload not reset.
One dispatcher; preserve all failures. Verify -> REPORT -> distinct review ->
close -> archive/detached verify. A null/unsupported result is valid reporting,
not permission for an unrecorded parameter search. Journal novelty is assessed.
