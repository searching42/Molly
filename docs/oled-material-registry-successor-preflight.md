# OLED Material Registry successor snapshot write preflight (PR-X)

PR-X consumes one exact PR-W local Registry-entry adjudication and one newly
supplied current Material Registry snapshot. It rechecks every approved
candidate against that current state and, when the entire batch is clean,
produces a deterministic append-only successor snapshot plan. It does not
write a Registry snapshot or activate a Registry head.

## Exact inputs and time boundary

The file entry reads two distinct, non-symlinked JSON files:

1. one `oled_material_registry_entry_adjudication.v1` artifact;
2. one `oled_material_registry_snapshot.v1` artifact representing the current
   Registry state supplied by the operator.

The output records both construction-time file SHA-256 values and embeds both
validated models. Standalone validation can replay all semantic derivations,
but the original external JSON byte sequences are not embedded. Therefore
`standalone_input_bytes_revalidation_supported` remains `false`.

The current snapshot must have the same Registry ID as the snapshot bound by
PR-W and cannot predate it. Preflight generation cannot predate either PR-W or
the supplied current snapshot. The current snapshot SHA-256 and semantic
digest become the exact parent compare-and-swap boundary for PR-Y.

No separate Registry head or lineage-receipt format exists in this phase, so
`current_snapshot_lineage_receipt_bound=false` is explicit. If such a receipt
is introduced later, it must be supplied and bound rather than inferred from a
filename or mutable pointer.

## Full-batch conflict recheck

PR-X rebuilds the approved-candidate roster from the embedded PR-W artifact and
rechecks every candidate against both the current snapshot and the complete
candidate batch. It fails the whole preflight on any collision involving:

- material ID;
- preferred name or alias, using exact codepoint equality;
- canonical isomeric SMILES;
- standard InChI; or
- InChIKey.

PR-X never overwrites, merges, aliases, or silently drops a colliding
candidate. Candidate chemistry and entry digests are replayed through the
existing validated Registry-entry and PR-W models.

## Deterministic successor plan

For a clean non-empty roster, PR-X derives the successor Registry version from
the exact parent snapshot digest, PR-W adjudication digest, and ordered planned
addition digests. It constructs the complete expected successor snapshot with:

- every prior entry preserved exactly;
- only the PR-W-approved entries appended and material-ID sorted;
- the current Registry ID and chemistry runtime preserved;
- a bound expected successor snapshot digest; and
- exact source PR-W item, candidate, and dependent-cell counts for every
  planned addition.

An empty approved roster returns `no_registry_changes_required` and does not
invent a successor snapshot or version.

## Explicitly false after PR-X

- material ID reserved or assigned;
- authoritative Registry entry created;
- Registry snapshot written;
- Registry head activated;
- existing Registry mutated;
- observations or reviewed evidence materialized;
- Gold records, dataset views, or training eligibility created;
- device-only records admitted; and
- network, external service, LLM, or MinerU calls.

## File entry

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_successor_preflight \
  --entry-adjudication /absolute/path/to/entry_adjudication.json \
  --current-registry-snapshot /absolute/path/to/current_registry_snapshot.json \
  --output /absolute/fresh/path/to/registry_successor_preflight.json
```

The output must be fresh and cannot overlap either input. The runner pins the
output parent before input loading, rejects symbolic input/output path
components, and publishes with the repository's atomic no-replace helper.
CLI failures expose only a stable error code and exception type.

## Next boundary

PR-Y must consume this exact preflight plus the exact current snapshot bytes,
recheck the compare-and-swap boundary immediately before publication, and
publish the preflight's expected successor as an immutable, fresh, fsynced,
inode-bound unit. Head activation remains a separate explicit action.
