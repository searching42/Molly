# OLED exact-chain observation staging preflight (PR-P)

## Purpose

PR-P joins one exact PR-N material Registry resolution request with its exact
PR-O human Registry adjudication. It derives the subset of paper-local material
groups that have been mapped to an existing Registry entity and preserves the
complete roster of reviewed property-cell references that depend on that
identity.

PR-P is a preflight, not observation materialization. PR-N and PR-O contain the
reviewed cell coordinates and semantic/source digests, but they intentionally do
not contain the source cell's reported value, unit, decimal precision, or
measurement context. PR-P therefore cannot honestly construct an
`OledPropertyObservation`. It records that exact source-value replay is required
before any material ID may be attached to a materialized observation.

## Exact inputs

The file entry accepts:

1. one `oled_material_registry_resolution_request.v1` artifact from PR-N;
2. one `oled_material_registry_adjudication.v1` artifact from PR-O.

Both files are opened without following symbolic path components and are bound
by their exact SHA-256 values. The builder requires PR-O's stored PR-N file
SHA-256 to equal the SHA-256 of the supplied PR-N file, not merely its semantic
digest.

The output embeds both validated models and repeats:

- PR-N file SHA-256 and request artifact digest;
- PR-O file SHA-256 and adjudication artifact digest;
- carried PR-M adjudication file SHA-256 and semantic digest; and
- carried Registry snapshot file SHA-256 and semantic digest.

Standalone model validation replays the semantic digests and the complete
PR-N/PR-O item join. Exact input-file byte revalidation still requires those
external files, so
`standalone_input_bytes_revalidation_supported=false` remains explicit.

## Exact-chain join

PR-P requires:

- identical run and paper IDs;
- identical PR-M and Registry snapshot bindings;
- PR-O's request digest to equal the supplied PR-N artifact digest;
- the PR-O adjudicated-item roster to equal the complete PR-N resolution-item
  roster; and
- every PR-O embedded request item to be byte-for-byte equivalent at the
  validated JSON-model level to the corresponding PR-N item.

No PR-N item may be missing, added, substituted, or silently reordered.

## Staging-item eligibility

Only a PR-O item with all of the following derived truths is emitted:

```text
existing_registry_entity_mapped = true
material_identity_resolved = true
canonical_material_id_assigned = true
cross_paper_identity_mapping_human_confirmed = true
eligible_for_later_observation_staging = true
```

The selected Registry entry is replayed in full. Each staging item preserves:

- PR-N resolution item ID and digest;
- PR-O adjudicated item digest;
- PR-K/PR-M identity group ID and digest;
- selected existing material ID and complete Registry entry;
- reported paper-local subject literal;
- exact source, PDF, parsed-document, table, page, row, and subject-column
  bindings; and
- every identity-dependent cell's row, column, source-cell digest,
  PR-I disposition digest, and semantic-review item ID/digest.

The redundant literals are rederived during standalone validation. Rewriting a
reported subject, source coordinate, Registry entry, or cell reference and then
recomputing the item and outer digests still fails closed.

## Excluded outcomes

PR-O outcomes remain partitioned and counted separately:

- new-entity proposals;
- unresolved items; and
- conflict-deferred items.

None produces a staging item. In particular, a new-entity proposal has no
stable Registry ID yet and cannot be admitted merely because its graph was
accepted as paper-local evidence.

Ontology-review-pending cells and device-only records remain outside PR-P.
Device-only admission is fixed at zero.

## Controlled workflow

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_observation_staging_preflight \
  --request-artifact /operator/local/material_registry_resolution_request.json \
  --registry-adjudication /operator/local/material_registry_adjudication.json \
  --output /operator/local/observation_staging_preflight.json
```

The output must be fresh and distinct from both inputs. The output-parent
descriptor is pinned before either input is read and retained through
validation and publication. Symbolic input/output path components, input
overwrite, stale bindings, parent replacement, invalid models, and boundary
flag changes fail without publishing partial output. CLI failures emit only a
stable redacted error object.

## paper016-shaped automated boundary

The mapped test path preserves:

```text
7 upstream PR-M review items
1 PR-N resolution item
1 PR-O existing-entity mapping
1 PR-P staging item
5 exact dependent-cell references
14 ontology-pending cells excluded
0 device-only cells admitted
```

Separate tests prove that new-entity, unresolved, and conflict-deferred
outcomes produce no staging items. These fixtures do not establish the real
cross-paper identity of a paper016 material.

## Explicitly false after PR-P

- source property values present;
- material ID attached to a materialized observation;
- observation or schema-candidate materialization;
- reviewed-evidence staging or direct admission;
- Registry or alias mutation;
- Gold, dataset, feature, or training writes;
- source PDF or parsed-document reads; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

The next observation-materialization boundary must rejoin each PR-P cell with
the exact accepted PR-I semantic disposition and PR-J bounded source
transcription that contain the reported literal, unit, and precision. It must
also bind the applicable condition/context fields before constructing a layered
property observation. A digest-only cell reference or column header is not
sufficient evidence for a value.

PR-O new-entity proposals remain on a separate Registry-entry proposal and
human-validation branch. They must not enter the observation branch until a
stable material ID exists through that explicit gate.
