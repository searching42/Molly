# OLED Release Candidate External Publication Preflight

## Purpose

`oled_curated_release_candidate_external_publication_preflight.py` is a read-only readiness gate for local OLED release-candidate registry artifacts.

It verifies that release-candidate artifacts are internally consistent and safe to consider for a later explicit external-publication gate. Passing this preflight does not publish, create GitHub releases, create git tags, upload artifacts, mutate a global registry, benchmark-validate results, or validate scientific claims.

## Inputs

- Release writer manifest JSON
- Release candidate entry JSON
- Release candidate delta JSONL
- Release candidate snapshot JSONL
- Optional prior registry snapshot JSONL

The preflight verifies SHA256 values recorded by the release writer manifest when available.

## Release Delta And Snapshot Checks

The release delta must contain release-candidate records that reference the selected release entry. By default the preflight expects a single release delta record.

The release snapshot must include the release delta record. When a prior registry snapshot is supplied, the prior snapshot must be preserved in the same order at the beginning of the release snapshot.

## Source Chain Checks

The release entry must retain the upstream source chain:

- global-append writer manifest and entry ids
- release preflight status
- final-registry, publication, promoted, registry, candidate-report, and benchmark-report source ids

The release preflight status must be `passed` or `passed_with_warnings`.

## Safety Boundary

The preflight rejects metadata claiming:

- benchmark validation
- scientific claim validation
- external publication
- GitHub release creation
- git tag creation
- artifact upload
- global registry mutation

It also rejects raw prediction payloads, feature dictionaries, raw paper text, image/PDF inputs, and obvious absolute local path leakage.

## CLI Example

```bash
python -m ai4s_agent.domains.oled_curated_release_candidate_external_publication_preflight \
  --release-writer-manifest /path/to/release_writer_manifest.json \
  --release-candidate-base-dir /path/to/release_candidate \
  --prior-registry-snapshot /path/to/prior_registry_snapshot.jsonl \
  --output-report /path/to/external_publication_preflight_report.json
```

Optional filters:

- `--baseline-kind`
- `--target-property-id`
- `--feature-view`
- `--allow-multiple-release-delta-records`
- `--allow-missing-prior-snapshot-preservation-check`

## Output

The optional report JSON is deterministic and redacted. It contains artifact summaries, entry summaries, finding code counts, status counts, and safety metadata. It does not include full release entries, raw predictions, feature dictionaries, raw paper text, or absolute local paths.

## Warning

This gate is read-only. It does not publish externally, create GitHub releases, create tags, upload artifacts, validate benchmark performance, make scientific conclusions, or mutate global registry files. External publication remains a later explicit gate.
