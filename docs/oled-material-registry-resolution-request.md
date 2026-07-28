# OLED material Registry resolution request (PR-N)

## Purpose

PR-N is the first cross-paper material-identity boundary after the PR-M human
review. It compares only PR-M accepted paper-local structure candidates with
one explicitly supplied, immutable material Registry snapshot and produces a
self-contained human-review request.

It is a lookup/preflight stage, not an identity adjudicator. Even when both
canonical isomeric SMILES and InChIKey point to the same Registry entry, the
output calls that entry a **consistent exact structural candidate**. It does
not assign the entry's material ID to the paper-local row.

## Inputs and exact binding

The controlled file entry accepts exactly two JSON inputs:

1. one successful
   `oled_supplementary_material_identity_adjudication.v1` artifact from PR-M;
2. one `oled_material_registry_snapshot.v1` artifact.

The runner opens both inputs without following symbolic path components,
validates their complete Pydantic models, and records both the exact file
SHA-256 and semantic artifact digest. The complete validated PR-M artifact and
Registry snapshot are embedded in the output, so later validation can replay
coverage and lookup derivation without silently substituting either model.

PR-N does not reopen the source PDF and does not independently replay the full
PR-G through PR-M chain. Its trust boundary is the exact PR-M artifact bytes.
The output records this limitation explicitly.

## Registry snapshot contract

Each Registry entry contains:

- a stable `material_id`;
- one preferred `canonical_name` and a sorted, unique alias list;
- canonical isomeric SMILES;
- standard InChI;
- InChIKey; and
- an entry digest.

The snapshot binds a Registry ID/version, generation timestamp, chemistry
profile, RDKit version, InChI backend version, sorted unique entries, entry
count, and snapshot digest. Every entry is reparsed with the pinned runtime;
its canonical SMILES, standard InChI, and InChIKey must all agree. A snapshot
from another chemistry runtime fails closed instead of being compared under
an unrecorded normalization policy.

The snapshot is read-only. PR-N never adds, removes, updates, retires, or
merges an entry.

## Exact lookup and conflict taxonomy

For every PR-M group with all of the following truths:

```text
eligible_for_later_registry_review = true
paper_local_structure_candidate_accepted = true
source_anchors_human_validated = true
source_to_candidate_match_human_validated = true
```

PR-N performs two deterministic equality lookups against the snapshot:

- canonical isomeric SMILES candidate -> Registry material IDs;
- InChIKey candidate -> Registry material IDs.

The result is classified as:

| Status | Meaning | Allowed automatic action |
|---|---|---|
| `no_exact_structural_candidate` | neither key has an exact hit | none |
| `partial_structural_key_match` | only one key has a singleton hit | none |
| `one_consistent_exact_structural_candidate` | both keys have the same singleton hit | none |
| `ambiguous_duplicate_structural_key` | either key maps to multiple material IDs | none |
| `conflicting_structural_key_matches` | singleton key results disagree | none |

Duplicate canonical SMILES, duplicate InChIKeys, and duplicate reported-name
literals are explicit conflict findings. Only findings that touch one of the
bounded PR-M candidates are added to the review packet; unrelated Registry
maintenance problems do not inflate this paper's review workload. No finding
performs an automatic merge.

Standard InChI/InChIKey normalization can collapse distinctions that matter
for a scientific material entity. Therefore a consistent exact hit still
requires human review of stereochemistry, charge/protonation state, salts,
mixtures, complexes, source scope, and the intended Registry entity policy.

## Alias boundary

PR-N reports only codepoint-exact equality between the paper's reported
subject literal and an entry's preferred name or alias. It performs no case
folding, abbreviation expansion, fuzzy matching, synonym generation, or alias
normalization.

An alias hit is a navigation hint and is always stored with:

```text
exact_codepoint_match_only = true
identity_evidence = false
```

Names and aliases reject active markup, URLs, local paths, control text, and
credential-like content before they can enter JSON or Markdown artifacts.

## Generality and coverage boundary

The production contract is paper-agnostic. Each invocation consumes one exact
PR-M artifact and one explicitly supplied Registry snapshot; paper IDs, table
IDs, row counts, eligible-group counts, dependent-cell counts, and Registry
entry counts are all derived from those inputs. No paper016 literal or fixed
paper016 data shape exists in the production builder or runner.

The resolution item roster must exactly equal the PR-M
`eligible_for_later_registry_review` roster. Each item's complete adjudicated
group is preserved, including the exact paper-local candidate, human review
truths, source/table/row binding, and dependent property-cell digests.

The paper016-shaped automated canary therefore preserves:

```text
7 PR-M review items
1 Registry-eligible accepted graph candidate
5 Registry-eligible dependent property cells
14 ontology-pending cells outside PR-N
0 device-only cells admitted
```

This remains one synthetic paper016-shaped canary, not a production limit and
not a claim about a real paper016 Registry resolution. A separate multi-item
regression exercises seven Registry-eligible groups and 35 dependent cells so
single-item fixture ordering cannot mask production failures.

## Controlled workflow

Build the exact-bound request:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_resolution_request build \
  --source-adjudication /operator/local/material_identity_adjudication.json \
  --registry-snapshot /operator/local/material_registry_snapshot.json \
  --output /operator/local/material_registry_resolution_request.json
```

Render reviewer-facing Markdown:

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_material_registry_resolution_request render \
  --request-artifact /operator/local/material_registry_resolution_request.json \
  --output-markdown /operator/local/material_registry_resolution_request.md
```

Both outputs must be fresh. Before reading/validating the inputs, the runner
pins the output-parent directory descriptor and retains it through model
derivation and publication. It refuses to overwrite either input, rejects
symbolic input/output path components, rolls back an owned output if the named
parent is replaced during build or rendering, and emits a redacted stable
error record on CLI failure.

## Explicitly false after PR-N

- human Registry resolution completed;
- material identity resolved;
- canonical material ID assigned;
- alias normalized;
- cross-paper identity merge or automatic candidate merge;
- Registry mutation;
- observation or schema-candidate materialization;
- reviewed-evidence staging, Gold admission, dataset write, or training
  eligibility;
- source PDF read or full PR-M upstream replay; and
- network, external-service, LLM, or MinerU calls.

## Next boundary

PR-O now implements the separate exact-bound human Registry decision manifest
and adjudication artifact described above. It records existing-entity mapping,
new-entity proposal, unresolved, and conflict-deferred outcomes while keeping
Registry and observation writes disabled. See
`docs/oled-material-registry-adjudication.md`.

Existing-entity mappings and new-entity proposals then follow separate later
gates: exact-chain observation staging for the former, and Registry-entry
proposal/validation for the latter. Registry writes and alias lifecycle remain
explicitly later boundaries.
