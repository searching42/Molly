# OLED Final Registry Global-Append-Readiness Preflight

## Purpose

`oled_curated_final_registry_global_append_preflight.py` is a read-only preflight for local OLED final-registry candidate artifacts. It checks whether those artifacts are internally consistent and whether they collide with an optional existing final/global registry snapshot before a later writer gate appends anything.

Passing this preflight does not append, publish, validate, globally register, or create scientific conclusions.

## Inputs

- Final-registry candidate writer manifest JSON.
- Final-registry candidate entry JSON.
- Final-registry candidate index JSONL.
- Optional existing registry snapshot JSONL for duplicate and source-chain checks.

The snapshot is optional. Tests and pilot runs can use synthetic records; no real registry file is required.

## SHA256 Checks

The loader verifies SHA256 values recorded in the final-registry candidate writer manifest when they are present. Mismatches raise:

```text
final_registry_candidate_entry_sha256_mismatch:
final_registry_candidate_index_sha256_mismatch:
```

## Source Chain Checks

The preflight checks that the final-registry candidate entry preserves the source chain from the publication candidate stage, including:

- publication writer manifest id
- publication candidate entry id
- final-registry-readiness preflight status
- promoted registry entry id
- promotion writer manifest id
- candidate report id
- benchmark report manifest id

The final-registry preflight status must be `passed` or `passed_with_warnings`.

## Duplicate And Collision Checks

When an existing registry snapshot is supplied, the preflight checks for:

- duplicate final-registry entry ids
- duplicate source publication entry ids
- duplicate source candidate report plus benchmark report manifest pairs

Duplicate entry ids and duplicate source chains are errors by default. CLI flags can relax those checks for exploratory inspection.

## Final-Registry-Candidate Status

The entry and index must retain `final_registry_candidate` status. The report rejects metadata that claims benchmark validation, scientific-claim validation, publication, global registration, or global registry mutation.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_final_registry_global_append_preflight \
  --final-registry-writer-manifest /path/to/final_registry_candidate_manifest.json \
  --final-registry-candidate-base-dir /path/to/final_registry_candidate \
  --existing-registry-snapshot /path/to/existing_registry_snapshot.jsonl \
  --output-report /path/to/global_append_preflight_report.json
```

## Safety Boundary

This preflight is read-only. It does not write or mutate final/global registry files, publish benchmark entries, mark benchmark validation, validate scientific claims, rerun baselines, run models, train, predict, recompute metrics, call LLMs or MinerU, or read PDFs/images. The actual global registry writer remains a later explicit gate.
