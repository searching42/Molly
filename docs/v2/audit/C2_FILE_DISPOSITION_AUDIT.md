# C2 exact-HEAD file disposition audit

Status: `PASS`

Audit commit: `bc379401fd4d88efc714fb878628cb29e49765d2`

The exact file inventory is
`docs/v2/audit/V1_FILE_DISPOSITION_INVENTORY.csv`.

## Scope and method

The inventory covers every tracked Python implementation file matched by
`src/ai4s_agent/*.py` and `src/ai4s_agent/**/*.py` at the audit commit. It has
453 unique rows, with one explicit disposition per file. The closed vocabulary
is `MIGRATE`, `SIMPLIFY`, `PLUGIN`, `REFERENCE_ONLY`, `ARCHIVE`,
`DELETE_AFTER_CUTOVER`, and `DEFER`.

The audit used source inspection plus an AST import and inbound-use scan. All
453 files parsed successfully; the source contained 5,144 import nodes. The
representative stale paths were resolved to the actual implementations:

- `src/ai4s_agent/provenance/artifact_registry.py`
- `src/ai4s_agent/scientific_agent_harness_controller.py`
- `src/ai4s_agent/oled_bounded_discovery_controller.py`
- `src/ai4s_agent/scientific_agent_replanner.py`
- `src/ai4s_agent/run_plan_replan_application.py`
- `src/ai4s_agent/run_plan_replan_application_artifacts.py`
- `src/ai4s_agent/run_plan_replan_application_audit_memory.py`
- `src/ai4s_agent/run_plan_replan_proposal.py`

The audit records source-line counts and import-use counts so that later
milestones can distinguish a direct extraction candidate from a v1 authority
wrapper. It does not alter any v1 implementation and does not create a
production v2 package.

## Reconciliation

The high-level matrix had three factual representative-path corrections:

| Area | Reconciled path |
| --- | --- |
| Artifact registry | `src/ai4s_agent/provenance/artifact_registry.py` |
| Controller | `src/ai4s_agent/scientific_agent_harness_controller.py`; `src/ai4s_agent/oled_bounded_discovery_controller.py`; controller-related modules |
| Replanner | `src/ai4s_agent/scientific_agent_replanner.py`; `src/ai4s_agent/run_plan_replan_*.py` |

These are inventory/path reconciliations under the approved architectural
disposition. They are not a new scope decision.

```yaml
matrix_digest_at_owner_approval: 2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d
matrix_digest_after_C2_reconciliation: d26366996db3df2783b3c0fcc8b03981902c2400c1dd6128d436fcdfb2d4fca4
file_inventory_sha256: 6dc12b0a6d430e9fe6a31c38c3bfde2a12443dbe0dbdafb760859bd596ab83b3
tracked_python_file_count: 453
unresolved_dispositions: 0
```

## Decision

C2 is `PASS`. The exact file-level disposition is complete. Production
implementation remains separately gated by the complete C0-C7 readiness state;
this audit authorizes no CORE-01+ implementation by itself.
