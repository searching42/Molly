# OLED Publication Candidate Final Registry Writer Gate

## Purpose

`oled_curated_publication_candidate_final_registry_writer.py` is a controlled writer gate for the Phase 1 OLED real-literature grounding flow. It converts publication-candidate registry artifacts into local final-registry-candidate artifacts only after the final-registry-readiness preflight has passed and the caller gives explicit confirmation.

These outputs are local final-registry candidates. They are not external publications, benchmark-validated records, scientific performance claims, or mutations of any global registry.

## Inputs

- Publication writer manifest JSON from `oled_curated_promoted_registry_publication_writer.py`.
- Publication-candidate registry entry JSON.
- Publication-candidate registry index JSONL.
- Final-registry-readiness preflight report JSON from `oled_curated_publication_candidate_final_registry_preflight.py`.

The writer may verify SHA256 values recorded in the publication writer manifest while loading the source publication-candidate artifacts through the preflight loader APIs.

## Relationship To Final-Registry Preflight

The final-registry-readiness preflight remains the read-only readiness gate. This writer uses that report as an allowlist and status source:

- failed preflight blocks writing by default
- warnings are allowed by default but can be disallowed by policy
- validation, publication, or global-registry claims in source metadata are rejected

## Confirmation Requirement

Selection and writing require `confirm_publication_candidate_final_registry_write=True` unless dry-run mode is used. Calling the selection API without confirmation raises:

```text
confirmation_required:publication_candidate_final_registry_write
```

## Dry-Run Mode

Dry-run mode builds the final-registry candidate entry and index records in memory and may write only the audit manifest if requested. It does not write the final-registry candidate entry JSON or index JSONL files.

## Output Files

The default output filenames are deterministic:

```text
oled_final_registry_candidate_entry.json
oled_final_registry_candidate_index.jsonl
```

An optional writer manifest records file statuses, SHA256 values, policy, reason codes, and safety metadata.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_publication_candidate_final_registry_writer \
  --publication-writer-manifest /path/to/publication_candidate_registry_manifest.json \
  --final-registry-preflight-report /path/to/final_registry_preflight_report.json \
  --publication-candidate-base-dir /path/to/publication_candidate_registry \
  --output-dir /path/to/final_registry_candidate \
  --output-manifest /path/to/final_registry_candidate_manifest.json \
  --confirm-publication-candidate-final-registry-write
```

## Safety Boundary

This writer does not publish to any external service, append to or mutate a global registry, mark benchmark validation, mark scientific-claim validation, rerun baselines, run models, train, predict, recompute metrics, call LLMs or MinerU, or read PDFs/images. External publication and global registry mutation remain later explicit gates.
