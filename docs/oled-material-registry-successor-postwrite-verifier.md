# OLED Material Registry successor post-write verifier (PR-Z)

PR-Z is a read-only verification boundary over one exact PR-Y publication
receipt and the separately published successor Registry snapshot. It
independently replays the transition before the snapshot may be supplied to a
new PR-N Registry resolution request.

PR-Z does not trust PR-Y's verification booleans as evidence. It does not
publish or activate a Registry head, mutate either Registry snapshot, or enter
the observation, reviewed-evidence, Gold, dataset, or training chains.

## Exact file inputs

The controlled file entry consumes two distinct, non-symlinked JSON files:

1. `oled_material_registry_successor_write.v1`;
2. the separately published `oled_material_registry_snapshot.v1`.

PR-Z records the exact SHA-256 of both files. PR-Y serializes both publication
files deterministically, so the verifier independently rebuilds their expected
publication bytes and requires the actual file hashes to match. A semantically
equivalent but reformatted receipt or snapshot fails closed.

The complete validated receipt and published snapshot are embedded in the
verification artifact. Standalone validation can replay their canonical
publication bytes and semantic transition, but cannot recover arbitrary
external input bytes; `standalone_input_bytes_revalidation_supported=false`
remains explicit.

## Independent transition replay

PR-Z derives the expected transition from PR-Y's embedded exact PR-X preflight,
prior snapshot, and separately supplied published snapshot. It checks:

- the published snapshot equals both PR-X's expected successor and PR-Y's
  published successor in full;
- every prior Registry entry remains byte-for-model identical;
- the published roster is exactly prior entries plus PR-X planned additions;
- every added entry reproduces the exact material ID, canonical name, aliases,
  canonical isomeric SMILES, standard InChI, InChIKey, and entry digest;
- material IDs remain sorted and unique;
- added material IDs and entry digests equal the exact PR-X/PR-Y roster;
- prior, added, dependent-cell, and final counts agree;
- Registry ID, prior/successor versions, snapshot digest, and generated times
  preserve the expected lineage; and
- receipt publication time does not predate the successor snapshot.

The Registry entry and snapshot models independently reparse all chemical
identifiers under the pinned chemistry runtime. PR-Z does not accept a writer
boolean such as `append_only_transition_verified=true` as a substitute for
these checks.

## Verification artifact boundary

On success, PR-Z records:

```text
status = registry_successor_publication_verified
published_registry_snapshot_verified = true
eligible_for_explicit_pr_n_input = true
registry_head_activated = false
observations_materialized = false
```

`eligible_for_explicit_pr_n_input=true` means only that an operator may supply
this exact verified snapshot file to a new PR-N request. It does not create a
mutable default head, automatically rerun PR-N, assign paper-local mappings, or
materialize observations.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_successor_postwrite_verifier \
  --write-artifact \
    /absolute/path/to/material_registry_successor_write.json \
  --published-registry-snapshot \
    /absolute/path/to/material_registry_snapshot.json \
  --output /absolute/fresh/path/to/registry_postwrite_verification.json
```

The output must be fresh and distinct from both inputs. Symbolic path
components, input overwrite, output-parent replacement, timestamp reversal,
non-canonical publication bytes, semantic tampering, and roster/count/lineage
changes fail closed. CLI errors expose only a stable error code and exception
type.

## Explicitly false after PR-Z

- Registry publication or mutation by the verifier;
- Registry head activation or activation receipt;
- automatic PR-N or PR-O execution;
- observation materialization or admission;
- reviewed-evidence staging;
- Gold records, dataset views, or training eligibility;
- device-only admission; and
- network, external service, LLM, or MinerU calls.

## Next boundary

After the real paper016 successor passes PR-Z, use its exact published snapshot
file to build a new PR-N resolution request. The seven paper-local structures
should surface the seven successor entries as exact structural candidates.
Human PR-O adjudication remains required before PR-P/PR-Q can attach those
material IDs to the 35 known property observations.
