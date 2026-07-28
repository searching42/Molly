# Generic RunPlan Provenance Audit Acceptance

This acceptance check proves that the existing generic `run_plan_execute` worker queue path can execute citation and license provenance auditing:

```text
run_plan_execute
-> track_citation_provenance_adapter
-> citation_provenance_report + audit_summary
```

The test uses synthetic local parsed document, evidence hit, and extracted-record fixtures. No PDF fixture is created or read.

## Safety Boundary

This path writes a citation provenance report and audit summary only. It tracks citation and license status for review, but it does not grant reuse permission. Candidate records remain unconfirmed. The test does not confirm, promote, train, predict, parse PDFs, call MinerU, call GROBID, fetch URLs, resolve DOIs, scan corpus directories, call LLMs, use sentence-transformers, download embedding models, spawn subprocesses, or mutate registry/promotion/publication/release/global append artifacts.

## Scope

This acceptance test does not add a CLI and does not modify the OLED local demo allowlist. It uses direct synthetic fixtures so the focused test executes only `track_citation_provenance_adapter`.
