# When Is a Second Loss Drop Not Double Descent? A Controlled Study of CatBoost Training Dynamics

**Historical pilot note.** Original working manuscript: 4 November 2026; status annotation: 7 November 2026. This note is retained for its limited pilot results and the distinction between late acceleration and double descent. It is not the current primary research plan, a corpus-results preprint, or evidence of scientific approval. `scientific_validity` is `not_assessed`. See the [current research program](../docs/research-program.md), [legacy assessment](../docs/legacy-assessment.md), and [verified literature registry](../docs/literature/README.md).

No new experiment or reanalysis was run for this annotation. The pilot's recorded numbers and historical rules below are preserved; they are not automatically the protocol for a new study. No human reviewer approval is recorded here.

## Abstract

The repository began as a search for double descent in a multilayer perceptron and in CatBoost trained on frozen text embeddings. A second drop in log loss can be ordinary optimization, and Kanoh (2026) studies a separate split-candidate capacity intervention in gradient boosting. This historical note asks which mechanisms could produce a late acceleration of CatBoost training loss, and when that acceleration would constitute double descent. It records the earlier evaluation protocol and a small CPU pilot. The project's primary program now studies double-descent mechanisms across linear, random-feature, neural, and tree-based models; late acceleration is an optional separate line. Essay embeddings and embedding analysis have local artifacts, but the partial MLP log does not establish double descent or a completed test evaluation.

## Question

Double descent, as used here, is a rise and later fall in test error as a complexity axis grows, near an interpolation regime. A two-phase log-loss curve is a stall followed by faster decrease of both training and validation log loss. The second pattern need not contain any worsening of generalization. A faster decline of log loss is not, by itself, a phase transition. The statistical unit is an independent run, not a neighboring boosting iteration.

The logistic loss \(\ell(m)=\log(1+e^{-m})\) has derivative \(d\ell/dm=-1/(1+e^{m})\). Once the margin \(m=yf(x)\) is positive and growing at a constant rate, the loss contributed by that point decelerates. Crossing zero margin therefore does not explain a sharp late acceleration. An explanation needs a change in the speed of the margin, in the distribution of hard points, or in the trees being selected.

## Historical pilot protocol

Thresholds for F1 and Youden’s J, and the prevalence threshold, are frozen on the validation split. Accuracy, F1, log loss, Brier score, and zero-one error are then computed on the test split. The same array cannot be passed as both splits.

The comparison between the network and CatBoost uses clean labels. A 10% label-noise arm exists only as a separate DNN regime and is off unless `DNN_RUN_NOISE_ARM` is set. A comparison whose noise rates differ is marked `blocked_noise_regime_mismatch` and is not a controlled contrast.

The DNN used for that comparison is the model-wise width with the best validation log loss among widths whose logged training loss has stopped falling by more than 5% over the last fifth of the logged grid. If no width has settled, the selection rule says so. The widest network is retained only as the epoch-wise curve.

PCA keeps the smallest number of components that meets the variance target, default 0.95. It no longer replaces a 256-component fit by 128 components when 128 components explain less than the target. If the target is not met inside the fitted cap, the original embedding dimension is kept.

CatBoost complexity is not one axis. Depth, `border_count`, and the number of trees are varied one at a time. On the essay corpus that grid is specified and switched off (`CB_RUN_COMPLEXITY_AXES=false`) until a small-data check exists. The corpus run still uses one seed, 42. Three seeds, 42, 43, and 44, are the replicate list for the pilot. The interval across those runs is the observed minimum and maximum, not a population confidence interval.

A change point in validation log loss \(L(t)\) is a time \(t^\star\) after which the mean velocity \(v(t)=-dL/dt\) is higher, and a two-segment line reduces residual sum of squares by at least 5%. Replication requires at least three runs whose fractions \(t^\star/T\) lie within 0.15 of their median. One curve returns `not_available`.

Hypothesis labels are closed before looking at the pilot:

- H1 (`random_strength=0` removes a late acceleration) is `supports` only if the reference acceleration replicates and the zero-strength runs do not. If the reference acceleration is absent, H1 is `inconclusive`.
- H2 (learning rate and L2, and alignment on `iteration × learning_rate`) is descriptive.
- H3 (hard-example margins versus AUC) is descriptive.
- H4 (a test-error peak on a separate complexity axis) is never `supports` on a three-point grid. A peak on all seeds is `coarse_peak_not_double_descent`.

## What this study is not claiming over Kanoh (2026)

Kanoh, “Double Descent in Gradient Boosting Decision Trees via Split-Candidate Scaling,” arXiv:2608.03111, 4 August 2026, treats the number of split candidates as a capacity axis for GBDT, links it to the spectrum of an empirical tree kernel, and reports test-error peaks for CatBoost, XGBoost, and LightGBM. That preprint is not a consensus; a peak on one more dataset would be a replication rather than an automatic novelty claim. The distinct target of this historical pilot is the long plateau and later acceleration of CatBoost log loss, and the conditions under which that shape is or is not double descent. This does not replace the broader primary question of the current research program.

The same caution comes from Curth, Jeffares, and van der Schaar (NeurIPS 2023), who show that a second descent in trees can be an artifact of gluing different complexity axes, and from Comesaña Cimadevila (arXiv:2509.25216), who finds the peak when learner capacity and ensemble size are scaled together and not when one axis is held fixed. Belkin, Hsu, Ma, and Mandal (PNAS 2019, doi:10.1073/pnas.1903070116) and Nakkiran et al. (ICLR 2020, arXiv:1912.02292) already established double descent outside and inside deep networks. Ustimenko, Beliakov, and Prokhorenkova (ICLR 2023, arXiv:2206.05608) connect certain symmetric, randomized boosting schemes to kernel ridge regression; that equivalence is not a license to treat an arbitrary CatBoost Logloss configuration as a Gaussian process. Tabular comparisons of networks and trees remain those of Gorishniy et al. (NeurIPS 2021) and Grinsztajn, Oyallon, and Varoquaux (NeurIPS 2022). CatBoost’s documented decay of split-score noise motivates H1. It does not identify the mechanism.

The historical essay experiment was designed as a comparison of an MLP head and a CatBoost head on one frozen embedding, not a comparison of end-to-end deep learning with gradient boosting. A completed matched comparison has not been established. Source, length, and generator artifacts in the AI-versus-human essays can be predictors. Near-duplicate leakage across the random split is unchecked.

## Pilot results

`scripts/pilot_late_acceleration.py` fit CatBoost on CPU with two threads. It did not read the essay embeddings. Synthetic draws used `make_classification` (600 rows, 10 features, a new draw per seed). The second set was scikit-learn’s breast-cancer data, with the split and the model seed changing together. Reference settings were 200 iterations, learning rate 0.1, depth 6, `l2_leaf_reg` 3, `random_strength` 20, `border_count` 32, Bayesian bootstrap, and `use_best_model=false`. The validation set was the only evaluation set during fitting.

On both datasets and all three seeds the change-point rule found no candidate: velocity did not increase. Replication is `not_reproduced`. H1 is `inconclusive`, because there was no reference acceleration for `random_strength=0` to remove. H4 on depth, `border_count`, and number of boosting trees (historically labeled `ensemble_size`) is `not_observed`. H3 records hard-example margins and validation AUC and draws no mechanism conclusion. For H2, all six learning-rate pairs had a smaller spread on `iteration × learning_rate` than on raw iterations, but 8 of 12 synthetic curves and 6 of 12 breast-cancer curves placed the log-loss minimum on the last iteration. That alignment is the shared iteration budget, not evidence for H2.

Held-out F1 thresholds differed from thresholds that would have been tuned on the test split (synthetic 0.747 versus 0.410; breast cancer 0.468 versus 0.137). Those pairs only show that the old test-tuned rule can move. They are not corpus metrics.

The essay-corpus status of every hypothesis remains `not_assessed`.

## Historical essay log and conflicts left explicit

The embedding artifacts use `qwen/qwen3-embedding-8b` at 4096 dimensions through RouterAI. The earlier plan named EmbeddingGemma. Local s03/s04 artifacts exist; the previous statement that embeddings were still being computed is obsolete. The historical training code holds torch on CPU. The stage-01 split is exact-duplicate removal plus a stratified random split. Preserving that historical artifact is not a validation of the split. The original EpistemeOS preregistration is unchanged and describes the older question, including a GPU note that differs from the recorded CPU/clean-label run.

The partial local `result/s05_dnn.log`, read on 7 November, records validation losses 0.0063, 0.0048, 0.0037, 0.0025, 0.0023, 0.0020, and 0.0038 for widths 8, 16, 32, 64, 128, 256, and 512 respectively. All printed training accuracies are 1.000; these rounded values cannot certify zero errors. Several widths, including 256 and 512, have `settled=False`. The next recorded event starts width 1024, without a completed row at the time of inspection. A start event does not establish that an OS process is still running.

Thus monotonic decline describes only the logged widths 8–256; width 512 rises, with no observed subsequent right shoulder. There is no measured bracket around the interpolation transition, no completed test evaluation, and no mechanism identification. These observations are historical local log evidence, not a confirmed double-descent result. The final output directory was empty at inspection; the original log is mutable and is ignored by Git. A small-file snapshot and inventory were preserved during the reset in `research/evidence/legacy-2026-11-07/`, as described in the legacy assessment; preserving the log does not validate the underlying arrays or resolve metadata mismatches.

## References

Belkin, M., Hsu, D., Ma, S., and Mandal, S. Reconciling modern machine-learning practice and the classical bias–variance trade-off. *PNAS*, 2019. doi:10.1073/pnas.1903070116.

Comesaña Cimadevila, G. Evaluating Double Descent in Machine Learning: Insights from Tree-Based Models Applied to a Genomic Prediction Task. arXiv:2509.25216, 2025.

Curth, A., Jeffares, A., and van der Schaar, M. A U-turn on Double Descent: Rethinking Parameter Counting in Statistical Learning. NeurIPS, 2023.

Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting Deep Learning Models for Tabular Data. NeurIPS, 2021.

Grinsztajn, L., Oyallon, E., and Varoquaux, G. Why do tree-based models still outperform deep learning on tabular data? NeurIPS, 2022.

Kanoh, R. Double Descent in Gradient Boosting Decision Trees via Split-Candidate Scaling. arXiv:2608.03111, 2026.

Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., and Sutskever, I. Deep Double Descent: Where Bigger Models and More Data Hurt. ICLR, 2020. arXiv:1912.02292.

Ustimenko, A., Beliakov, A., and Prokhorenkova, L. Gradient Boosting Performs Gaussian Process Inference. ICLR, 2023. arXiv:2206.05608.
