# CoT comparison audit and complete research evidence

Release tag: `research-cot-audit-2026-11-08`.

## Main result

On previously unused confirmation question clusters, 16 of 56 support- and
inference-eligible model-pair/dataset comparisons met the frozen materiality
criterion: nine BBH and seven MMLU comparisons. The largest native-to-common-
input gap shift was 15.52 percentage points. All 28 GPQA comparisons lacked
the frozen minimum support. This audits historical released judge-ACK
annotations and comparison populations; it does not identify latent reasoning
faithfulness, causal cue effects or corrected true model rankings.

The general matching/selection concern has prior work. The contribution is a
quantified audit of this pinned public release and a reproducible diagnostic
tool. External priority and readiness for a strong journal are not established.
The earlier double-descent/CatBoost work and negative/inconclusive attempts are
preserved; no new general double-descent mechanism is claimed.

The project was implemented, analysed and separately reviewed by Codex agents.
Internal independent agent review is not external peer review or independent
new-benchmark replication. Human direction, resource constraints and AI roles
are documented in `docs/portfolio/cot-audit-overview.md`.

## Start here

- `manuscript.pdf`: the nine-page English research report.
- `cot-audit-portfolio-2026-11-08.zip`: self-contained portable package with
  111 figures in both PNG and SVG, eight CSV tables, compact public data,
  source, frozen protocols, bootstrap arrays, replay and review receipts.
- Open `cot-audit-portfolio/portfolio/gallery/index.html` after extraction.
- `research-cot-audit-2026-11-08-assets.json` and `SHA256SUMS.txt`: exact byte
  sizes, SHA256 hashes, repository-relative source locations and asset URLs.

Download from the [release page](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/tag/research-cot-audit-2026-11-08).

Both audit phases were reproduced with all 12 full-file comparisons equal and
no float differences. Before publication, 239 unit tests passed and nine
opt-in fitting tests were skipped; publication adds separately recorded source
checkout checks. All 111 PNG/SVG pairs and nine PDF pages received distinct
visual review. API, GPU, training and new model calls were not used. Reserve
outcomes remain unused. Numerical reproduction is not scientific certification.

## Portable verification

The portable archive was fully restored: all 324 files were byte/hash checked
by a verifier extracted from the ZIP, run in a separate process without original
package reads. Its SHA256 is
`3de3f516eaa2edffdae4d76fdc85cbbd5079bef5d1eefb081acd9dd372639491`.

Verify the downloaded assets against `SHA256SUMS.txt`. After extracting the
portable archive, use its top-level `VERIFY.py` to verify or restore into a
new directory:

```text
python VERIFY.py cot-audit-portfolio-2026-11-08.zip --restore NEW_DIRECTORY
```

The extracted `portfolio/evidence/README.md` gives the pinned package setup and
two-phase replay commands. This replay needs no model calls or data download.
It reproduces the numerical analysis of cached public data.

## Full historical evidence

Complete R0–R18 archives, including R17D and the historical pre-closure R1
archive, are attached separately. Every complete archive received fresh outer
SHA256 and internal indexed-file/blob verification before publication.
The deduplicated ZIPs contain `RESTORE.py` and `EVIDENCE-INDEX.json`; use the
standalone tool to verify or restore into a new empty directory. Earlier
TAR.GZ bundles have their own inventory/restore route documented in
`docs/automation.md`. `cot-review-proof-2026-11-08-v1.zip` retains the distinct
reviewers' checkers, QA and associated proofs.

The interrupted `R13-runs-2026-11-08-v2.tar.gz` is retained locally and omitted:
its gzip stream ends prematurely and it is not a usable release archive.
All three original R13 attempts are fully included in the verified
`R13-runs-2026-11-08-deduplicated-v3.zip`. Omitting the damaged transfer copy
does not omit a research attempt. Four earlier unexecuted R17 planning drafts
were not retained as complete bytes; that historical limitation is disclosed
in `research/evidence/cot-review-proof-2026-11-08.json`.

Each full historical workflow tree was not materialized again: every indexed
byte was streamed and checked, with representative files restored. The normal
portable CoT ZIP separately received a complete restore test. Raw public
source payloads are preserved in the R15 archive. Large legacy matrices
explicitly marked local-only in the original inventory remain outside this
release; no stronger historical completeness is claimed.

## Source checkout and full evidence validation

Git contains source, protocols, controls, scientific decisions, reports and
small historical exports. Generated live runs, intake/QA copies and new large
archives ship as Release assets. Source-only CI records unavailable evidence
bindings as unverified; a passing source check does not mean full evidence
validation or a new scientific result.

After restoring the required evidence, the unchanged full local check is:

```powershell
uv sync --frozen
uv run python scripts/check_repository.py --output checks.json
```

All 20 stages are operationally closed. No new scientific run is queued.
The source/review records retain their exact original bytes, including mixed
historical LF/CRLF line endings, because evidence SHA256 bindings are byte exact.
Release publication was explicitly authorized by the user after the scientific
stages closed; their original frozen plans are not retroactively rewritten.
