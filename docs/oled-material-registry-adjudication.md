# OLED material Registry human adjudication (PR-O)

## Purpose

PR-O records one exact-bound human Registry decision for every PR-N resolution
item. It distinguishes:

- mapping the paper-local material to one already surfaced Registry entity;
- proposing a new Registry entity without creating or naming it;
- keeping the paper-local material unresolved; and
- deferring a structural or semantic conflict.

PR-O is an adjudication artifact, not a Registry writer and not an observation
materializer. A confirmed mapping to an existing entity records a human
canonical material-ID association and may become eligible for a later staging
gate. It does not modify the Registry snapshot or write any property
observation.

## Exact inputs

The controlled adjudication entry accepts:

1. one `oled_material_registry_resolution_request.v1` artifact from PR-N;
2. one `oled_material_registry_decision_manifest.v1` supplied by the human
   reviewer.

The decision manifest repeats:

- run and paper IDs;
- exact PR-N file SHA-256 and semantic digest;
- exact PR-M adjudication file SHA-256 and semantic digest carried by PR-N;
- exact Registry snapshot file SHA-256 and semantic digest carried by PR-N;
- reviewer identity and timezone-aware review timestamp; and
- `adjudication_confirmed=true`.

PR-O opens both files without following symbolic path components, validates
the complete PR-N model (including its embedded PR-M and Registry snapshot),
requires exact decision coverage, and stores the exact decision-manifest file
SHA-256 plus deterministic semantic digest in the output.

## Mandatory acknowledgement coverage

Every decision entry must copy three complete PR-N rosters exactly:

```text
reviewed_structural_candidate_material_ids
reviewed_alias_hit_digests
reviewed_registry_conflict_digests
```

Missing, extra, reordered, duplicated, or stale values fail closed. Copying an
alias hit means only that the reviewer saw it; alias hits remain navigation
hints and never become identity evidence.

Every decision also binds the exact `resolution_item_id` and
`resolution_item_digest` and requires a non-empty safe review note.

## Allowed decision matrix

| PR-N lookup status | map existing | propose new | unresolved | defer conflict |
|---|---:|---:|---:|---:|
| no exact structural candidate | no | yes | yes | yes |
| partial structural-key match | yes | no | yes | yes |
| one consistent exact structural candidate | yes | no | yes | yes |
| ambiguous duplicate structural key | no | no | yes | yes |
| conflicting structural-key matches | no | no | yes | yes |

`map_to_existing_entity` requires exactly one `selected_existing_material_id`,
and the ID must be in the structural candidate union surfaced by PR-N. Alias-
only entries and arbitrary snapshot entries cannot be selected.

Duplicate-key and conflicting-key results cannot map directly to an existing
entity and cannot propose another potentially duplicate entity. They must
remain unresolved or be deferred for Registry cleanup.

`defer_conflict` requires exactly one reason:

- `duplicate_structural_key`;
- `structural_key_disagreement`;
- `reported_name_collision`; or
- `entity_scope_or_chemistry_conflict`.

All other decisions require `conflict_reason=none`.

The first three reasons must be supported by the corresponding bounded PR-N
match state/finding kind. `entity_scope_or_chemistry_conflict` is the only
reason that may record a new human semantic concern when no precomputed
Registry collision exists. The adjudication embeds the exact reviewed conflict
finding objects so this reason-to-evidence check remains auditable.

## Existing-entity mapping semantics

For a valid human mapping, PR-O copies the complete selected Registry entry
from the exact PR-N snapshot and replays the selected structural hit. The
adjudicated item records:

```text
existing_registry_entity_mapped = true
material_identity_resolved = true
canonical_material_id_assigned = true
cross_paper_identity_mapping_human_confirmed = true
eligible_for_later_observation_staging = true
registry_written = false
observations_materialized = false
```

Here, `canonical_material_id_assigned` means the adjudication records an
association to an ID that already existed in the bound snapshot. It does not
mean the Registry was mutated.

The output preserves the mapping group/cell counts separately. A later
observation-staging stage must jointly consume the exact PR-N request and PR-O
adjudication so it can replay the complete Registry snapshot binding before
using the selected ID.

## New-entity proposal semantics

`propose_new_entity` is allowed only when PR-N found no exact structural
candidate. PR-O derives a proposal from the PR-M-accepted graph and preserves:

- reported paper-local subject literal;
- candidate digest;
- canonical isomeric SMILES candidate;
- standard InChI candidate; and
- InChIKey candidate.

The proposal explicitly leaves all of the following false:

```text
material_id_assigned
canonical_name_assigned
aliases_assigned
registry_entry_created
registry_written
```

PR-V builds a separate, exact-bound local Registry-entry proposal review
request for those fields. It proposes only an unassigned opaque material-ID
handle and an unapproved paper-local preferred-name candidate; aliases remain
empty. A no-hit in the bound local snapshot is not a global novelty claim.
See `docs/oled-material-registry-entry-proposal-request.md`. New-entity
proposals remain ineligible for observation staging until later human
Registry-entry adjudication and a separately authorized Registry write.

## Unresolved and conflict outcomes

`keep_unresolved` does not exclude the paper-local row, delete its reviewed
property cells, or reinterpret a rejected response as a material exclusion.

`defer_conflict` records the human conflict category but performs no merge,
deduplication, alias mutation, or Registry repair. Both outcomes remain
ineligible for downstream observation staging.

## Controlled workflow

Render the PR-N evidence plus exact PR-O decision instructions:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_adjudication render \
  --request-artifact /operator/local/material_registry_resolution_request.json \
  --output-markdown /operator/local/material_registry_adjudication_review.md
```

After completing every decision, build the adjudication artifact:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_adjudication adjudicate \
  --request-artifact /operator/local/material_registry_resolution_request.json \
  --decision-manifest /operator/local/material_registry_decisions.json \
  --output /operator/local/material_registry_adjudication.json
```

Both commands pin the output-parent descriptor before input loading and retain
it through validation and publication. Inputs and outputs must be distinct;
inputs may not contain symbolic path components; outputs must be fresh. A
parent replacement during rendering or adjudication fails without publishing
into either the displaced or redirected directory. CLI failures emit a stable,
redacted error object.

## Generality and canary boundary

The production adjudicator is paper-agnostic and handles the complete dynamic
PR-N item roster. It contains no paper016 literal, table identifier, material
name, or fixed item/cell count. One invocation adjudicates one exact PR-N
request so every decision remains bound to a single paper-local evidence chain
while the supplied Registry snapshot provides the cross-paper identity state.

Automated tests preserve the existing synthetic paper016-shaped chain:

```text
7 PR-M review items
1 PR-N Registry-eligible paper-local graph candidate
5 identity-dependent property cells in that group
14 ontology-pending cells excluded
0 device-only cells admitted
```

The same candidate is exercised against an empty snapshot, one exact entry,
and two duplicate structural entries to test new-entity proposal, existing-
entity mapping, unresolved, and conflict-deferred outcomes. These fixtures do
not establish the real Registry identity of any paper016 material. A separate
seven-item regression deliberately uses a PR-N source order that differs from
the stable adjudication-ID order and verifies seven new-entity proposals and
35 dependent cells end to end.

## Explicitly false after PR-O

- automatic candidate merge;
- Registry entry creation, update, retirement, or mutation;
- canonical-name or alias assignment/mutation for new entities;
- observation or schema-candidate materialization;
- reviewed-evidence staging or direct/Gold admission;
- dataset, feature, or training write;
- source-PDF read or independent PR-M upstream replay; and
- network, external-service, LLM, or MinerU calls.

## Next boundaries

PR-O intentionally creates two separate downstream branches:

1. existing-entity mappings may enter a later exact-chain observation-staging
   preflight, which must replay PR-N and PR-O together before attaching the
   selected material ID to reviewed property cells;
2. PR-V converts new-entity proposals into a request-only local Registry-entry
   review packet. A later human adjudication must approve the single-entity
   scope, opaque material ID, source-supported preferred name, and exact alias
   list before a separately authorized writer may create Registry state. See
   `docs/oled-material-registry-entry-proposal-request.md`.

Unresolved and conflict-deferred items stop here until new evidence or Registry
repair changes the decision basis.
