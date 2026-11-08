# Auditing Late Loss Acceleration in Gradient Boosting: Detection Limits and Learning-Rate Clock Controls

**Research manuscript, 7 November 2026.** R10–R12 numerical claims have
separate, evidence-bound local review.
The versioned protocols and exact claim registry govern admission; this
document does not establish priority for a general statistical method.

## Abstract

A second period of rapid training-loss reduction is sometimes interpreted as
evidence of a capacity transition or a new learning mechanism. Such an
interpretation requires distinguishing optimization speed, held-out risk, and
interpolation. We develop an audit for that distinction. A constructive argument
shows that any finite positive nonincreasing loss sequence can be generated on
one fixed quadratic by varying its step size. A complementary spectral argument
identifies the narrower stationary positive-semidefinite squared-loss model
under which absolute loss decrements cannot increase. These elementary controls
motivate three predeclared studies: whole-trajectory calibration of an
acceleration scan against specified dependent-noise laws, and paired CatBoost
learning-rate interventions with equal cumulative prescribed step size, followed
by continuous-fit split-score-randomness interventions at constant step size. The
empirical contribution is the measured failure envelope of competing rules and
the effect of a controlled change of training clock. Across 2,400 synthetic null
evaluation trajectories, the ungated local rule flagged every curve. A
whole-curve calibrated scan had observed null rates of 0–10%, while its power
for one plateau-tail alternative fell to 1.5% under the larger correlated noise.
In 72 CatBoost schedule fits, step-up-minus-constant descent-speed contrasts
were positive in tree-index time but negative in the prespecified cumulative
learning-rate clock in both studied environments. One step-down contrast
remained inconclusive after multiplicity correction. In 72 continuous-fit
randomness-intervention cells, with 12 paired seed units per environment,
stronger split-score randomness increased the
maximum scan score, but none of six prespecified groups established the fixed
sustained late-window event. A score increase and an identified late event are
therefore distinct empirical outcomes under this design.
We do not treat a scheduled intervention as reproduction of a naturally occurring
CatBoost phenomenon, or an acceleration flag as evidence of double descent.

## 1. Introduction

The shape of a learning curve is an observation, not a unique explanation.
Training loss can fall rapidly, plateau, and fall more rapidly again. The same
plot may reflect changing optimization steps, adaptive update directions,
stochasticity, measurement effects, or several of these together. It also need
not predict the behavior of unseen examples.

This study addresses a specific practical problem: how can a researcher assess
an apparent late acceleration without first searching for a visually attractive
curve and then treating its shape as a mechanism? We distinguish three questions:
(i) whether the speed of training-loss reduction increases on a declared clock;
(ii) whether held-out risk increases and subsequently decreases along one fixed
axis; and (iii) whether measurements and interventions support an interpolation
interpretation or a causal explanation. A training-loss acceleration alone
answers neither of the latter questions.

Our contribution is deliberately bounded. We provide an auditable diagnostic
study with declared null laws, whole-curve calibration, localization assessment,
and a causal learning-rate control. The algebra below supplies counterexamples
and restricted falsifiers rather than a new theory of gradient boosting. A useful
outcome can be limited sensitivity, excessive false flags, or inconclusive
intervention evidence. None requires changing the protocol to obtain a positive
curve.

## 2. Related work and scope of novelty

Kanoh studies double descent in CatBoost, XGBoost, and LightGBM using
split-candidate scaling, with kernel diagnostics and controlled interventions.
Our study does not claim that observing double descent in these libraries is
new. It instead audits what a change in optimization speed can identify.
[Kanoh, 2026](https://arxiv.org/abs/2608.03111v1).

Curth, Jeffares, and van der Schaar examine how distinct complexity axes and
smoother-based effective parameters alter interpretations of non-neural
double-descent curves. This motivates preserving the identity of each axis
rather than combining different ways of growing a model in one line.
[Curth et al., 2023](https://papers.nips.cc/paper_files/paper/2023/hash/aec5e2847c5ae90f939ab786774856cc-Abstract-Conference.html).

Kernel connections for symmetric-tree boosting also predate this work.
Those results do not make every adaptive CatBoost Logloss trajectory an exact
instance of stationary squared-loss kernel descent.
[Ustimenko et al., 2023](https://arxiv.org/abs/2206.05608v3).

Slope-change detection and error control are established statistical problems.
MOSUM methods study piecewise-linear changes, and testing local extrema after
smoothing and differentiation provides another approach to structural breaks.
Our scan is an application and validation of known ideas, not the first local
change-point detector.
[Kim et al., 2023](https://arxiv.org/abs/2208.04900v3),
[He et al., 2026](https://arxiv.org/abs/2308.04368v5).

The official CatBoost documentation describes learning rate as controlling the
update step and required training duration. It also describes time-dependent
randomness in split scoring. These are motivations for controls, not evidence
that either explains an observed transition.
[CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning).

The targeted literature audit is recorded in
`docs/literature/2026-11-07-v2-audit.md`. It is not a systematic search and does
not certify research priority. Metadata or abstract verification is not silently
promoted to validation of detailed theorem assumptions.
Four primary HTML snapshots, versions, locators, retrieval records and SHA-256
hashes are retained in `artifacts/intake/v2-literature/`; the acquisition used
four requests and no dataset downloads.

The scoped empirical contribution can be separated from its known ingredients:

| Prior ingredient | Contribution measured here | Boundaries |
| --- | --- | --- |
| General change-point scans and calibration | Complete error, power and localization map of three declared acceleration rules under four noise conditions | Finite synthetic catalog; no new general test |
| Learning-rate effects on optimization | Paired, equal-total CatBoost schedules with a reversal of step-up contrast sign across prespecified clocks | Whole schedules and two small environments |
| Randomized split scoring | Whole-run rs10-versus-rs0 score increases together with six failed fixed late-event gates | Not an intervention on the annealing law itself |
| Double-descent mechanisms and effective complexity | Separate optimization, risk and interpolation audit, with earlier family controls retained | No new universal double-descent mechanism |

These are original measurements of this declared study, not an assertion that
the statistical ingredients or optimization facts were previously unknown.
The literature comparison bounds the contribution; it does not certify an
exhaustive absence of similar empirical findings.

## 3. What can one loss curve identify?

### 3.1 A fixed-objective construction

Consider any finite sequence
\(L_0\geq L_1\geq\cdots\geq L_T>0\). Set
\(f(x)=x^2/2\), \(x_0=\sqrt{2L_0}\), and

\[
\eta_t=1-\sqrt{L_{t+1}/L_t},\qquad 0\leq\eta_t<1.
\]

Gradient descent gives
\(x_{t+1}=(1-\eta_t)x_t=\sqrt{2L_{t+1}}\), and hence
\(f(x_t)=L_t\) at every observed point. A plateau corresponds to a zero step.
Thus the entire monotone curve, including a late acceleration, can occur without
a change of objective, representation, or dimension. Knowing that the step size
is constant rules out this particular construction, but does not identify the
remaining mechanism.

For this construction the exact clock

\[
\tau_t=-2\sum_{s<t}\log(1-\eta_s)
\]

gives \(L_t=L_0e^{-\tau_t}\). This identity depends on the quadratic model.
For CatBoost, \(\sum_s\eta_s\) is only a declared diagnostic clock; curvature
and adaptive updates prevent interpreting it as an exact equivalent.

### 3.2 A restricted stationary spectral control

Suppose training residuals evolve as
\(r_{t+1}=(I-\eta K)r_t\), with constant \(\eta\), symmetric PSD \(K\),
and \(0\leq\eta\lambda_i\leq2\). For
\(L_t=\lVert r_t\rVert^2/(2n)\), spectral decomposition yields

\[
L_t=\frac{1}{2n}\sum_i a_i^2q_i^t,
\qquad q_i=(1-\eta\lambda_i)^2\in[0,1].
\]

Therefore

\[
D_t=L_t-L_{t+1}
=\frac{1}{2n}\sum_i a_i^2(1-q_i)q_i^t,
\qquad 0\leq D_{t+1}\leq D_t.
\]

Increasing absolute loss decrements contradict this restricted model. A
contradiction does not distinguish a changing operator from changing steps,
regularization, stochasticity, a different loss, or measurement error. It is not
a universal falsifier for kernel explanations, and does not apply automatically
to CatBoost Logloss. Both arguments are elementary controls, not claims of a new
general theorem. Their authored derivation is preserved in
`docs/methods/acceleration-identifiability.md`.

## 4. Methods

### 4.1 R10: synthetic calibration and detection limits

R10 compares the existing global mean-velocity rule, its uncalibrated local
window-5 counterpart, and a multiscale OLS slope scan. For observations
\(y_0,\ldots,y_{255}\), the scan uses windows \(w\in\{8,16,32\}\).
At an eligible boundary \(b\), OLS segments cover \([b-w,b]\) and
\([b,b+w]\), including the boundary in each. With slopes
\(\hat\beta^-_{b,w}\) and \(\hat\beta^+_{b,w}\), the score is

\[
S(y)=\max\!\left(0,
\max_{b,w:\hat\beta^+_{b,w}<0}
\frac{w(\hat\beta^-_{b,w}-\hat\beta^+_{b,w})}
{\max_t y_t-\min_t y_t}\right).
\]

The constant-curve score is zero. The score is invariant to positive affine
changes of loss units. The global baseline retains its 20% minimum-segment rule
and 5% piecewise-linear RSS-improvement requirement. The local baseline flags
a positive maximum difference in mean velocities; its RSS diagnostic does not
gate the primary local flag. Neither baseline is a calibrated significance test.

Let \(u=t/255\) and \((x)_+=\max(x,0)\). The complete deterministic catalog is:

| Role | Family | Mean trajectory |
| --- | --- | --- |
| Null | Linear | \(1-0.5u\) |
| Null | Exponential | \(0.2+0.8e^{-5u}\) |
| Null | Early plateau | \(1-0.6\min(u/0.2,1)-0.02u\) |
| Alternative | Late hinge | \(1-0.05u-0.3(u-u_{b_{.7}})_+\) |
| Alternative | Fast decline, plateau, tail | \(1-0.6\min(u/0.2,1)-0.02u-0.6(u-u_{b_{.7}})_+\) |
| Alternative | Last tenth | \(1-0.05u-1.2(u-u_{b_{.9}})_+\) |

Here \(b_c=\lfloor255c\rfloor\). Stationary Gaussian AR(1) level noise has
marginal standard deviation \(\sigma\in\{0.002,0.01\}\) and correlation
\(\rho\in\{0,0.8\}\), giving four noise conditions. A complete independently
generated trajectory is one repetition; adjacent points are not repetitions.

Each null/noise cell supplies 199 calibration trajectories. Its threshold is the
190th ordered score; the threshold for a noise condition is the maximum of its
three null thresholds. A flag requires strict exceedance. Independent
evaluation uses 200 trajectories in each of six family/four noise cells. The
declared total is 2,388 calibration plus 4,800 evaluation trajectories, with
separate SeedSequence phase channels and no evaluation-based tuning.

When all 199 calibration scores and a future score are jointly exchangeable
under one included null law, the order-statistic construction bounds marginal exceedance by 0.05;
taking the maximum threshold preserves this bound for each included law. This
is marginal over calibration, not a conditional guarantee for the realized
threshold. It does not cover unknown noise laws or CatBoost curves.

False-flag rates, power, and exact pointwise 95% Clopper–Pearson intervals are
reported separately by family/noise/detector. These intervals condition on the
shared calibration and do not include calibration uncertainty or simultaneous
coverage across the table. Localization summaries include median absolute
boundary error among flagged alternatives and the proportion of all alternatives
detected within 32 points. The latter avoids hiding missed detections.

### 4.2 R11: paired learning-rate interventions

R11 asks whether a predeclared learning-rate schedule changes descent speed
around a known continuation boundary. Its two environments are a fixed
`make_classification` generator and the bundled Wisconsin Breast Cancer dataset.
The synthetic generator has 768 rows, 10 features, five informative features,
zero redundant features, `flip_y=0.08`, and `class_sep=0.8`. Stratified splits
contain 256 training, 256 validation, and 256 test rows. Breast Cancer uses
stratified 60/20/20 splits with sklearn integer rounding. No external dataset is
downloaded. Independent synthetic seeds regenerate data; Breast Cancer repeats
are conditional on one fixed corpus.

The 12 seeds are 12001–12012. Data, split, and model channels are respectively
seed, seed+100000, and seed+200000. The three schedules are constant
\(0.11\to0.11\), step-up \(0.02\to0.20\), and step-down
\(0.20\to0.02\). Each segment has 128 trees, so every schedule has total
prescribed learning rate 28.16. All arms use the same two-fit continuation
procedure, including the constant arm, to control for the restart boundary.

CatBoost settings are CPU, four threads, depth 4, leaf L2 3, no bootstrap,
random strength 1, Plain boosting, Logloss, 254 borders, and one leaf-estimation
iteration. Best-model selection is disabled. `boost_from_average` is true for
the first segment and false for continuation, as required with `init_model`.
All declared settings and library versions are saved with the attempt.

The primary statistic uses a fixed boundary at tree 128. OLS is fit to
`loss[97:129]` and `loss[128:160]`, each containing 32 points and sharing the
boundary. Gain equals after-descent-speed minus before-descent-speed. Each
schedule gain is differenced against the constant-arm gain within its seed.
Step-up predicts a positive paired difference; step-down predicts a negative
difference. There are four primary contrasts: two schedules by two environments.

We report all 12 paired values, their mean and SE, a two-sided 95% Student-t
interval, and a Bonferroni interval for a familywise 95% level across the four
contrasts. Support requires the adjusted interval to lie strictly on the
predicted side of zero; the opposite side refutes the prediction, and overlap
is inconclusive. These intervals rely on independent paired repeats and an
adequate t approximation; 12 repeats do not establish that approximation.

A separate diagnostic family repeats the same fixed windows using cumulative
prescribed learning rate as the x coordinate. It has no predicted sign and no
mechanistic support/refutation label. Equal cumulative prescribed steps do not
make trained models equivalent: schedules can change the update path and
selected splits. Validation/test losses, exact classification errors, margins,
gradient RMS, and concentration of absolute gradients are descriptive. None
selects the grid, boundary, or primary claim. Synthetic R10 thresholds do not
become valid CatBoost p-values.

### 4.3 R12: constant-step split-score-randomness intervention

The next study uses the same two declared environments with new seeds
13001–13012. Each of 72 models is a single uninterrupted 1,024-tree fit at
learning rate 0.05; random strength is 0, 1, or 10. Depth, L2, bootstrap,
quantization and other settings are fixed as in the scientific protocol.
No schedule switch, best-model truncation, or restart is used. Raw margins are
retained for every object at every iteration; exact error counts and gradient
summaries are split-level aggregates. Tree-leaf assignments are retained for
training objects. All fitting and inference calls explicitly use at most four threads.

The primary scalar is the maximum train-loss OLS scan with windows 8, 16 and
32, as defined in R10. Paired rs1-minus-rs0 and rs10-minus-rs0 differences in
each environment form a four-contrast interval family. Only rs10 has a
prespecified positive sign prediction. This statistic scans the whole path;
an increase does not by itself establish a late event or identify noise
annealing. No synthetic R10 threshold is applied to these longer curves.

A stronger late-event criterion is fixed at tree 512, using 64 loss points on
each side: indices 449:513 and 512:576. A group passes only if its adjusted
mean-gain interval is positive, its adjusted mean after-window loss-drop
interval lies above 0.01 absolute LogLoss, and at least 10 of its 12 gains are
positive. Six group-gain and six group-drop intervals use a Bonferroni family
of 12. Four paired fixed-window gain differences are a separate diagnostic
family. Failure means this event definition was not established; it excludes
no other location or configuration. Whole-run random strength changes the
training path as well as split randomness, so it does not isolate an annealing
law. These choices were frozen before the new trajectories were opened.

## 5. Results

### 5.1 Analytic identifiability

The construction in §3 proves observational ambiguity for any positive finite
nonincreasing sequence when the step schedule is unconstrained. The stationary
PSD result supplies a restricted falsifier. These statements hold by algebra;
their scope does not depend on whether an empirical prediction is supported.
Separate reviewer `/root/workflow_audit` accepted the exact proof document in
`docs/methods/acceleration-identifiability-review.json`. The reviewer did not
contribute the construction; the scientific reviewer who did contribute it was
excluded from this independent sign-off. These are known elementary controls,
not a claim of theorem priority.

### 5.2 Detection failure envelope

Attempt `20261107T192414-36aee5d99b` contains all 7,188 finite trajectories,
including 2,388 calibration and 4,800 independent evaluation curves. Claim
`R10-acceleration-failure-envelope` binds all raw curves, score rows, thresholds,
and 72 summary rows to separate Codex review. The local rule flagged every one
of the 2,400 null evaluation curves. Its improved sensitivity on deterministic
R8 examples therefore did not validate it as an inferential procedure.

The calibrated scan's observed null rates were 0–10%. On the linear null they
were 12/200, 8/200, 8/200, and 20/200 for small iid, small AR, large iid, and
large AR noise. The last cell's pointwise exact interval was [0.0622, 0.1502].
The marginal rank bound must not be read as conditional 5% control after one
calibration draw. This selected cell is descriptive, not a simultaneous
table-wide rejection. Exponential and early-plateau nulls had zero scan flags
in all eight corresponding cells.

| Noise condition | Linear-null global/local/scan flags | Fast-plateau-tail global/local/scan flags |
| --- | --- | --- |
| Small iid | 0 / 200 / 12 | 0 / 200 / 200 |
| Small AR(1) | 97 / 200 / 8 | 0 / 200 / 200 |
| Large iid | 0 / 200 / 8 | 2 / 200 / 14 |
| Large AR(1) | 103 / 200 / 20 | 1 / 200 / 3 |

Each count has denominator 200. All 72 rates and exact intervals, including
the other shapes, are in `research/stages/R10/exports/analysis.json` and the
complete figure tables. The scan's fast-plateau-tail power fell from 100% under
small noise to 7% and 1.5% under large iid and AR noise. On the late-hinge
alternative it flagged all curves, but localization within 32 points fell to
63% and 56% at large noise; the global rule localized 99.5% and 98.5% there.
Thus neither detection nor localization uniformly dominates the baseline.

| Prespecified prediction | Required evidence | Outcome |
| --- | --- | --- |
| Local baseline flags more than half of noisy linear trajectories in every noise cell | 200/200 in each cell | Supported |
| Calibrated scan observed null false-flag rate is at most 0.10 in every null/noise cell | Maximum 20/200 | Supported at the declared empirical threshold |
| Scan detects more last-tenth alternatives than the global rule under small iid noise | Both 200/200 | Refuted |
| Positive affine rescaling changes scan score by less than \(10^{-10}\) | Maximum error \(8.36\times10^{-14}\) | Supported |

On the prespecified last-tenth contrast, median boundary error was 25 points
for the global rule and six for the scan. This descriptive localization result
does not replace the refuted detection-rate prediction. The full benchmark was
retained; no window, null law, or evaluation seed was changed after opening it.

No table-wide significance claim should be inferred by selecting individual
confidence intervals. Any low-power regime must remain in the failure envelope.

### 5.3 CatBoost intervention and clock comparison

Accepted retry `20261107T194028-229b223740` contains all 72 models and 216 raw
split-wise prediction arrays. Independent review recomputed all eight paired
contrast vectors and intervals and checked saved-model predictions. Claim
`R11-schedule-clock-dependence` binds these data. A preceding completed attempt
was retained with revise because inference thread limits were implicit; the
technical retry fixed those limits without changing data, seeds or scientific
choices. Predictions matched exactly, and the retry did not increase n.

| Environment | Contrast | Tree-clock mean and adjusted interval | Outcome | LR-clock diagnostic |
| --- | --- | --- | --- | --- |
| Synthetic | Step-up minus constant | .00425688 [.00319456, .00531920] | Supported | −.0241218 [−.0337150, −.0145287] |
| Synthetic | Step-down minus constant | −.000841410 [−.00120546, −.000477360] | Supported | .00154324 [−.000629271, .00371574] |
| Breast Cancer | Step-up minus constant | .00166258 [.00137100, .00195416] | Supported | −.0513656 [−.0535267, −.0492044] |
| Breast Cancer | Step-down minus constant | −.000135930 [−.000283184, .0000113232] | Inconclusive | .000932005 [−.000419366, .00228338] |

Tree-clock units are LogLoss per tree; LR-clock units are LogLoss per unit of
prescribed learning rate. Magnitudes across clocks are not directly comparable.
The positive-to-negative step-up contrast occurs within each clock’s matching
schedule comparison. The two separately adjusted four-contrast families do not
give a joint eight-contrast 95% guarantee. In particular, the unadjusted
Breast Cancer step-down interval is negative, but the declared adjusted interval
crosses zero and determines the inconclusive outcome.

These are effects of complete schedules, including differing prehistories
before tree 128. They do not isolate an instantaneous step change applied to
identical first-stage models. Equal tree count and cumulative prescribed steps
do not ensure identical update paths. The result supports recording training
clock and schedule as part of the phenomenon’s definition.

Report held-out behavior independently of train acceleration. No double descent
or natural late-acceleration mechanism is declared by this intervention protocol.

### 5.4 Continuous-fit randomness intervention

Verified attempt `20261107T194939-7c18953a98` contains the complete fixed
72-model matrix. The automatic protocol summaries report rs10-minus-rs0 scan
differences of .0171360 [.00759462, .0266774] on synthetic data and .00323860
[.00141283, .00506436] on Breast Cancer, with their declared family-four
intervals. The rs1 contrasts remain exploratory; neither adjusted interval
excludes zero. Separate review recomputed all raw metrics, scan positions,
paired contrasts and six gates; claim `R12-randomness-without-fixed-late-event`
binds the complete evidence.

All six fixed late-event groups are `not_established`: none has a positive
adjusted mean-gain lower bound, and positive gain counts are 4/12, 4/12 and
5/12 for synthetic rs0/1/10 and 4/12, 6/12 and 2/12 for Breast Cancer. A
larger maximum acceleration score therefore did not establish the independently
specified sustained late-window event. This distinction is limited to the
declared window and criterion and does not show universal absence of natural
acceleration. Neither a noise-annealing mechanism nor double descent is claimed.

Descriptively, 63 of 72 models reached exact zero training classification
errors within the budget, first doing so between trees 123 and 988. Training
LogLoss at first zero was as high as 0.140222. Nine first-zero times are
right-censored by the budget. The released plots distinguish those censored
cases and retain loss at first zero: surrogate loss and exact interpolation by
classification error cannot be substituted for one another.

### 5.5 Earlier model-family controls are secondary evidence

Earlier R2–R5 studies supply context, not new results of this audit. Their
reports describe a linear noise/ridge control, an MLP stopping-budget contrast,
separate tree/forest/boosting axes, and new-seed replication of selected linear
and tree contrasts. R3's short-budget prediction was refuted; this is retained.
They do not establish a common mechanism across model families and were not
rerun or reinterpreted to support the new acceleration claims.

| Existing study | Confirmation attempt | Source |
| --- | --- | --- |
| R2: linear/random-feature controls | `20261107T095343-6bc0c457c7` | `research/stages/R2/REPORT.md` |
| R3: MLP optimization controls | `20261107T102511-b1cf6c7be9` | `research/stages/R3/REPORT.md` |
| R4: separate tree/ensemble axes | `20261107T104140-5bc2019cef` | `research/stages/R4/REPORT.md` |
| R5: selected new-seed replication | `20261107T105955-6e7f800ff1` | `research/stages/R5/REPORT.md` |

## 6. Practical interpretation protocol

For a new apparent acceleration, first record the loss, data split, observation
frequency, clock, training schedule, and stopping rule. Preserve the full series
and record whether the event lies near a censored boundary. State a null model
before applying a scan; a positive score has no generic false-alarm meaning.

Validate the complete selection procedure on independent null and alternative
trajectories matching the intended measurement conditions. Include both missed
events and false flags. If calibration is synthetic, report the mismatch when
the rule is applied to model training; do not transfer the nominal error rate.

Then measure unseen-example risk separately and measure interpolation directly
where that interpretation is proposed. For a mechanism, declare a controlled
intervention and its predicted contrast before observing outcomes. A scheduled
positive control establishes diagnostic response to that intervention; it does
not establish a naturally occurring mechanism. Report dependence on clock and
retain uncertainty when competing explanations remain.

## 7. Limitations

The null catalog is finite and uses Gaussian AR(1) level noise. It omits jumps,
multiple transitions, other dependence laws, heteroskedasticity, unknown
sampling mechanisms, and long censored tails. It therefore cannot validate a
universal detector. The observed calibration thresholds are random and shared;
pointwise evaluation intervals do not remove that uncertainty.

The CatBoost studies have two small environments, one library, one fixed depth,
a two-segment continuation design and a separate continuous-fit design.
Breast Cancer repetitions do not provide
independent corpora. The intervention can change more than the speed of an
otherwise identical path. Internal margin/gradient measurements are descriptive
and cannot independently establish mediation or a universal causal mechanism.

The historical Qwen embedding case remains outside the new evidence: corpus
proximity and metadata concerns are unresolved. This manuscript does not claim
cross-corpus AI-text detection, a comprehensive MLP–GBDT comparison, a novel
general change-point test, a natural CatBoost phase transition, or journal
readiness. Research priority requires a broader literature assessment.

## 8. Reproducibility and review provenance

New numerical evidence belongs under `artifacts/research/`. Each attempt must
bind its frozen scientific and machine plans, all declared inputs, dirty and
untracked source snapshot, `uv.lock`, seeds, resolved commands, raw outputs,
environment, logs, and output hashes. Failed attempts are retained. A zero exit
code or operational closure is not a scientific validity judgment.

R10 protocol: `research/stages/R10/inputs/protocol.json`.
R11 protocol: `research/stages/R11/inputs/protocol.json`.
R12 protocol: `research/stages/R12/inputs/protocol.json`.

| Study | Accepted attempt | Claim ID |
| --- | --- | --- |
| Synthetic diagnostic audit | `20261107T192414-36aee5d99b` | `R10-acceleration-failure-envelope` |
| Schedule/clock control | `20261107T194028-229b223740` | `R11-schedule-clock-dependence` |
| Continuous randomness intervention | `20261107T194939-7c18953a98` | `R12-randomness-without-fixed-late-event` |
| Analytic controls | Proof document, separately reviewed | `A1-loss-clock-nonidentifiability` |

Exact plan, manifest, raw-output and decision hashes appear in each REPORT,
the accepted decisions and `research/claims.json`. Portable archive digests and
final release commands are recorded in `research/STATE.md` and the R13 handoff.
The graphical appendix provides every declared condition, all per-seed panels,
source CSVs, static PNG/SVG pairs, and offline interactive companions. Observed
seed envelopes are labeled as ranges, not confidence intervals.

The standalone release separates per-stage frozen code versions and retains
all selected attempts, including the earlier R11 technical history. It pins a
minimal scientific dependency closure without requiring the whole EpistemeOS
checkout or torch for these CatBoost studies. A manifest verifier and a replay
entry point use the packaged protocols. Regeneration on the same seeds checks
portability; it is not an independent scientific replication.

The earlier R2–R5 reports identify their own Cursor authors and local team
reviews. This revision is implemented in Codex; it must not acquire a fictitious
Cursor signature. The analytic document and R10 implementation are authored by
`/root` (the constructive argument also had input from `/root/science_critique`);
R11/R12 implementations are authored by `/root/catboost_study`, with R11
preflight fixes by `/root`. The literature
audit and this manuscript scaffold are authored by `/root/literature`. A
different agent must review each evaluated result or conclusion, with actual
identities and hashes recorded. Local team review is not an external peer
review or an independently collected replication.

The article promotes only claims registered in
`research/claims.json` with their exact evidence scope and separate bound review.
Numerical claims were reviewed by `/root/science_critique`; the analytic proof
was independently reviewed by noncontributor `/root/workflow_audit`.
Publication-level significance and the breadth of future mechanistic claims
remain questions for scientific assessment, not outcomes of the software gates.
