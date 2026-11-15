# CoT Audit: portfolio framing and transparent contributions

Release tag: `cot-audit-portfolio-2026-11-15`.

This documentation release makes the completed CoT audit easier to assess as
a research portfolio case study. Start with
[the portfolio overview](../portfolio/cot-audit-overview.md).

- Added a plain-language summary and links to the report, evidence and claims.
- Explained relevance to Anthropic's public CoT faithfulness research and the
  measurement foundations of scalable oversight, with primary-source links.
- Added usable portfolio and CV wording with explicit human/agent attribution.
- Distinguished documented human direction from unverified personal authorship
  of methodology, implementation or interpretation.

The scientific result remains the completed historical judge-ACK audit: 16/56
supported held-out comparisons met the frozen criterion, maximum shift 15.52 pp.
No new experiment, claim admission, external review or human approval is implied.

The [8 November evidence release](https://github.com/AdamCage/double-descent-dl-vs-gbdt/releases/tag/research-cot-audit-2026-11-08)
remains the authority for the PDF and portable numerical artifacts. Those frozen
assets predate this portfolio wording; this release does not rebuild them.
GitHub's source archives contain the updated documentation.

Validation: `uv sync --frozen`; full
`uv run python scripts/check_repository.py --output checks.json` passed
(248 tests, 9 skipped; no structural problems). Historical R18 verification
passed for attempt `20261108T111319-cf1d82f2a8`, checking 569 files.
