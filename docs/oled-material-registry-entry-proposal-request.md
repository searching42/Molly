# OLED local Material Registry entry proposal review request (PR-V)

## Purpose

PR-V turns the `new_entity_proposal` branch of one exact PR-O human Registry
adjudication into a self-contained request for a later local Registry-entry
review. It carries forward the PR-M-accepted molecular graph and the exact
PR-N/PR-O evidence chain, but it does not approve or write a Registry entry.

The word `new` in the upstream PR-O disposition is strictly local to the
bound Molly Registry snapshot. PR-V means only that PR-N found no exact
canonical-SMILES or InChIKey candidate in that snapshot under its recorded
chemistry profile. It is not evidence that a molecule is globally novel,
previously unreported, absent from external databases, or outside patent or
literature prior art.

## Exact inputs and chain replay

The request consumes exactly:

1. one `oled_material_registry_resolution_request.v1` artifact from PR-N;
2. its exact `oled_material_registry_adjudication.v1` artifact from PR-O.

Both complete validated models are embedded in the output. The request binds
the exact input bytes and semantic digests for PR-N and PR-O and preserves the
Registry snapshot SHA-256/digest carried by PR-N. It jointly replays the
PR-N -> PR-O item coverage, selected-entry bindings, and causal timestamp
ordering before deriving any review item.

PR-V does not reopen the source PDF or independently rerun PR-G through PR-M.
Its source-evidence boundary is the exact graph, source anchors, chemistry
validation, and deterministic depiction already carried through the bound
PR-N/PR-O chain.

## Exact request roster

There is exactly one entry-review item for every PR-O adjudicated item whose
human decision is `propose_new_entity`. Each item embeds the complete source
adjudicated item and preserves:

- the paper-local reported subject literal;
- the PR-O new-entity proposal and proposal digest;
- the PR-M-accepted canonical isomeric SMILES, standard InChI, and InChIKey;
- automatic chemistry facts, including fragment, charge, and unassigned
  stereochemistry findings;
- the exact candidate depiction and source/table/row binding;
- the bound Registry ID, version, and snapshot digest; and
- the number of identity-dependent property cells held behind this review.

Existing-entity mappings, unresolved items, and conflict-deferred items are
excluded with separate counts. Ontology-review-pending cells remain upstream,
and device-only cells remain outside the request with a fixed count of zero.
Missing, added, duplicated, substituted, or reordered review items fail
closed.

## Local-snapshot-only meaning

Every artifact and reviewer-facing packet carries the fixed notice:

> No exact structural candidate was found only in the bound Molly Registry
> snapshot; global chemical novelty, literature prior art, patent novelty, and
> external database coverage were not assessed.

The following truths remain fixed:

```text
global_chemical_novelty_assessed = false
literature_prior_art_assessed = false
patent_novelty_assessed = false
external_database_search_performed = false
```

The request must not describe an item as a globally novel molecule, a
previously unknown material, or a first report. If a later or refreshed local
snapshot contains a matching entry, the item must be routed back to existing
Registry resolution rather than approved as another entry.

## Proposed entry fields are not approvals

PR-V deterministically proposes an opaque material ID from the Registry
namespace and the exact PR-O proposal digest. That ID is a review handle only:
it is not reserved, assigned, or written.

Before exposing that handle, PR-V requires it to be absent from every existing
`material_id` in the bound Registry snapshot. An occupied ID fails closed; the
request never presents an already-used identifier as an approvable proposal.

The paper-local reported subject literal is copied as an unapproved preferred
name proposal. It is not automatically accepted as the Registry preferred
name. The alias list starts empty; PR-V performs no case folding, punctuation
normalization, abbreviation expansion, fuzzy matching, external synonym
lookup, or automatic alias assignment.

Any later preferred name or alias must be explicitly human-approved and
supported by the bound source. Exact name/alias hits in the PR-N snapshot are
navigation and conflict-review hints, never standalone identity evidence.

## Fixed human-review contract

The request binds a complete versioned contract and exact contract digest. It
asks the later reviewer to decide:

1. whether the accepted graph represents one local Registry entity at the
   intended stereochemistry, charge/protonation, salt, mixture, complex, and
   source scope;
2. whether the reported paper-local literal is an acceptable preferred name,
   or should be replaced only by another source-supported literal; and
3. which exact source-supported literals, if any, should become aliases.

Version 1 is intentionally limited to one exact single-molecular-entity
graph. Unresolved salts, mixtures, formulations, coordination complexes,
polymers, protonation/charge conflicts, stereochemistry ambiguity, or unclear
source scope must be deferred rather than flattened into a Registry entry.

The allowed later human decisions are:

- `approve_local_registry_entry_candidate`;
- `keep_unresolved`;
- `defer_entity_policy`; and
- `route_to_existing_registry_resolution`.

PR-V records none of those decisions. It only creates the exact request on
which a later decision manifest and adjudication artifact can operate.

## Automatic facts versus human decisions

| Field or question | PR-V behavior | Later human responsibility |
|---|---|---|
| graph identifiers | exactly replay SMILES, InChI, and InChIKey | confirm they represent the intended source entity |
| charge and fragments | show deterministic chemistry facts | decide charge/protonation and salt/mixture semantics |
| stereochemistry | show assigned/unassigned findings | decide whether source scope is sufficiently specific |
| material ID | derive an opaque deterministic proposal | explicitly approve before assignment or write |
| preferred name | copy reported subject as an unapproved proposal | approve it or choose another exact source-supported literal |
| aliases | initialize to an empty list | explicitly approve each exact source-supported alias |
| local snapshot match | replay the exact PR-N no-hit | never reinterpret it as global novelty |
| source scope | preserve the exact upstream chain | reject/defer device, formulation, mixture, or unclear scope |

RDKit validity and deterministic identifiers are computational checks, not
scientific proof of entity scope. Likewise, a reported name is not identity
evidence by itself.

## Batch conflicts

The request detects duplicate proposed material IDs, canonical SMILES,
InChIKeys, and proposed preferred names within the exact PR-V batch. Findings
name every affected review-item ID, block automatic approval, and never merge
items. Batch conflicts change the status to
`batch_conflicts_require_human_review`; an empty new-proposal roster produces
`no_new_entity_proposals`; otherwise the status is
`ready_for_human_registry_entry_review`.

Review items retain the exact PR-N source ordering key
`(scope_id, table_id, row_index, identity_group_id)`, including the final
identity-group tie-breaker when multiple identities share one source row.

## Reviewer-facing Markdown

The deterministic renderer presents:

1. exact PR-N, PR-O, Registry snapshot, request, and contract bindings;
2. the local-snapshot-only and request-only boundaries;
3. derived item, cell, exclusion, and conflict counts;
4. any batch conflicts before individual items;
5. each accepted graph and automatic chemistry facts; and
6. each unapproved material ID, preferred-name proposal, empty alias list,
   and required human questions.

The renderer does not preselect a positive decision and does not present the
opaque ID or reported subject literal as approved Registry state.

## Controlled workflow

Build the exact-bound JSON request:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_entry_proposal_request build \
  --resolution-request /operator/local/material_registry_resolution_request.json \
  --registry-adjudication /operator/local/material_registry_adjudication.json \
  --output /operator/local/material_registry_entry_proposal_request.json
```

Render the validated request as reviewer-facing Markdown:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_entry_proposal_request render \
  --proposal-request /operator/local/material_registry_entry_proposal_request.json \
  --output-markdown /operator/local/material_registry_entry_proposal_review.md
```

Both commands require fresh outputs, reject symbolic input/output path
components and input overwrite, retain a pinned output-parent descriptor
through validation and publication, and emit a stable redacted error object
on failure.

## Generality and paper016 canary boundary

Production derivation is paper-agnostic. It derives the complete dynamic
review roster, counts, identifiers, names, chemistry facts, and conflicts from
the exact PR-N/PR-O inputs. It contains no paper016 material name, table ID,
row count, or fixed item/cell count.

A paper016-shaped fixture is only a canary for the real chain. Passing that
canary demonstrates deterministic contract behavior for its bounded inputs;
it does not establish the Registry identity of any real paper016 material,
scientific validity outside the reviewed evidence, or a production-scale
Registry corpus.

## Explicitly false after PR-V

- global chemical novelty, prior-art, patent, or external-database assessment;
- material-ID reservation or assignment;
- preferred-name or alias approval;
- Registry-entry creation or Registry mutation;
- automatic merge or deduplication;
- observation materialization or reviewed-evidence staging;
- Gold, dataset, feature, or training write;
- source-PDF read; and
- network, external-service, LLM, or MinerU calls.

## Next boundaries

The next stage must consume the exact PR-V request plus a complete human
decision manifest, bind one decision to every review item and batch finding,
and produce a Registry-entry adjudication artifact. Approval must explicitly
confirm the single-entity scope, material ID, source-supported preferred name,
and exact alias list.

Only a later, separately authorized writer may create a new Registry snapshot
or delta. That writer must recheck the target Registry state so a concurrently
added exact entity cannot be overwritten or duplicated. Existing Registry
files must not be mutated in place, and new entries remain ineligible for
observation staging until both human adjudication and the Registry write are
verified.
