# R15 — complete pinned public-data intake

Author `/root`; intake implementation `/root/cot_intake`; independent reviewer
`/root/reset_critic`. This is an engineering report, not a scientific finding.

Attempt `20261108T090628-8efd502cb8` completed both registered tasks in
79.703 seconds. All 24 pinned Parquet payloads matched their declared size and
SHA256 at revision `a7a8b25ae467db9d80d63e55ac053b48753b2d36` of
`ameek/measuring_cot_monitorability_transcripts`.

The intake retained all 446,208 source rows and all 21 declared analysis
columns. Three omitted long-text columns have exact UTF8 hashes and lengths,
with source file, source row index, payload hash, row-content hash and ordered
choices hash. This does not establish the validity of the released judge labels.
No acknowledgment rates, model rankings or scientific endpoints were calculated.

The slim file is 88,613,869 bytes, SHA256
`cf41ae989be031362fed849a89387be87ca5d8231f89faa94ff8c1a96763f6ec`.
There are eight historical model identifiers and three file-parent datasets;
these metadata fix subsequent analysis membership before outcomes.

The transfer used 48 physical HTTP requests and 940,116,134 returned payload
bytes, within the frozen caps of 144 requests and 1,000,000,000 bytes. No retries
or model calls, training, GPU jobs or paid APIs were required. Arrow and HTTP
workers were capped at four. The exact acquisition log has been mirrored into
`artifacts/intake/R15-accounting/20261108T090628-8efd502cb8/acquisition.jsonl`
before the next run so the cumulative resource ledger counts the acquisition.

Evidence is under
`artifacts/research/R15/20261108T090628-8efd502cb8/tasks/intake/intake/`:
`download_manifest.json`, `schema_profile.json`, `intake_report.json`,
`acquisition.jsonl`, `slim.parquet` and all 24 raw payloads. The unit task passed
all seven pure/mock tests. Live workflow verification and separate operational
review determine closure; a zero exit code alone is not scientific validation.

P1 payload integrity: supported. P2 all-row lossless metadata transformation
within the slim cap: supported by the intake checks, subject to independent
review. P3 engineering budget and no outcome computation: supported. These are
engineering predictions only. API and GPU budgets remain zero.

Next: freeze separately reviewed R16/R17 scientific protocols. The user granted
four further hours at 09:12 UTC, with a delivery deadline of 13:12 UTC on
8 November 2026. Before scientific outcomes, a 30% discovery / 35% confirmation /
35% untouched reserve split replaces the earlier draft 30/70 split. Any new
hypothesis requires a new preregistration and independent review before reserve
outcomes. All attempted hypotheses will be disclosed.
