# R16 attempt 1 — preserved metadata integrity failure

Attempt `20261108T091936-7fa306e46e` failed before any acknowledgment aggregation.
The 16 pure fixture tests passed. The analysis stopped in the first metadata
pass after 0.094 seconds: identical released baseline input SHA256
`ba08a665eb39a1524b7a74cf47429e3fe90aa92e4c35fa1e389d93f79cbbbcaa`
has inconsistent correct-answer metadata across BBH core IDs
`bbh_causal_judgement_0036` (No/B) and `bbh_causal_judgement_0104` (Yes/A).
Both list ordered choices Yes, No. Root inspected only provenance, input
descriptors, choice and correct-answer metadata. No judge ACK rates, output,
reasoning or selected hypothesis endpoints have been inspected.

This is an engineering integrity failure, not a refutation of H1. The raw
source, slim, frozen plan/source snapshot and failed logs remain unchanged.
No numerical results, graphs or successful observations exist for this attempt.
Source-level diagnosis and an independent review are required before any retry.
Any revision must explicitly change the eligible population before outcomes,
retain an exclusion ledger and preserve this failed attempt.

Evidence: attempt manifest, frozen input snapshot, task stderr,
`tasks/ack-audit/audit/failure.json` and `dataquality.json`. The workflow
currently only verifies/reviews attempts with complete status; therefore an
external bound failure review is recorded separately. No operational close or
successful scientific validity is fabricated for this failed attempt.

Author `/root`; implementation `/root/alignment_brainstorm`; separate reviewer
`/root/reset_critic`. No API/GPU/model training or new downloads.
