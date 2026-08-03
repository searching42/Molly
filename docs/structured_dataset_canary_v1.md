# Structured Dataset Canary v1

Structured Dataset Canary v1 implements the BR1 engineering canary for broader organic emitter PLQY data. Public CI does not establish a TADF discovery claim. Its final artifact is named `Computational Top-N` and represents model-ranked computational candidates only.

## Authority chain

The canary is compiled into five planner-visible `ScientificToolSpec` / RunPlan tasks: prepare, confirm, train, generate and evaluate. The CI fixture first creates a proposal, receives Permission, calls approve-and-start and then lets the Harness Controller dispatch one task at a time through `RunPlanExecutor`. Task backends cannot register artifacts, write `StageState`, append Gate decisions, create Controller receipts or reconcile recovery. Those facts remain owned by Molly's existing authority chain.

Every backend input is the exact immutable path supplied by `RunPlanExecutor` from the current Artifact Registry/input manifest. Raw and Confirmed CSV content are separate registered artifacts (`raw_dataset_csv` and `confirmed_training_dataset_csv`), while their JSON publications bind the corresponding content digests. Evaluation also consumes the registered current checkpoint and generated candidate roster. No stage may rediscover an input through a fixed canary directory or an unregistered sibling file; path substitution, stale self-consistent siblings and publication/content mismatches fail closed.

`Raw Dataset is never training authority.` Raw CSV publication status is always `candidate_unconfirmed`. Review produces an immutable row roster and proposed actions. A trusted actor then approves the exact review snapshot through the shared training-config Gate; the confirmation receipt binds project, run, raw digest, review digest, included/excluded rows, target, role, condition policy, actor/source and decision digest.

`Training requires an exact-bound confirmation receipt and Confirmed Dataset publication.` Missing, stale, replaced or cross-scope authority fails closed. Client flags and LLM text cannot create a receipt.

## Condition-aware review snapshot v2

The frozen `prepare_structured_dataset_canary` task, default Tool Catalog,
`structured_dataset_review_snapshot.v1` reader and exact replay bytes remain
supported and unchanged. V1 is the CI/synthetic path and cannot infer or opt
into private mode from client-selected optional artifacts.

Private BR1 preparation uses a separately constructed server-owned v2 task
registry. That registry replaces the v1 prepare node with
`prepare_private_structured_dataset_canary_v2`, makes `uploaded_dataset`,
`source_dataset_manifest`, and `br1_mapping_policy` required inputs, and binds
the downstream confirmation dependency to that exact task. Trusted private
server bootstrap must inject the same registry into observation, planning,
proposal verification, Permission, authorization, Controller and executor.
The resulting versioned Tool Catalog and plan authorization bind the exact
manifest and mapping-policy bytes; no boolean inside client JSON creates owner
authority. Their content digests are also bound into the Raw Dataset,
condition-aware review and confirmation chain. The private files remain
private.

The checked-in schemas are validated at the adapter boundary. The source
manifest must bind dataset identity/version/DOI/license, download
date, original source digest and derived Raw CSV digest. The mapping policy
must freeze target/unit, scientific scope and downgrade,
single-solvent condition policy, molecular identity, duplicate tie-break and
`partially_comparable_single_solvent` semantics. Invalid or inconsistent scope
metadata fails before the review snapshot is created.

V2 separates three scientific identities:

- `scientific_observation_identity.v1` binds property, Standard InChIKey,
  normalized condition digest and source-context digest. Target value is an
  observed payload and is deliberately not identity material.
- `scientific_conflict_group.v1` binds property, Standard InChIKey and
  normalized condition only. It excludes paper/source identity so differing
  reports under the same condition remain visible as conflicts.
- train/test/external split grouping remains the InChIKey–paper bipartite
  connected component. It does not use observation identity.

`normalized_measurement_condition.v1` canonicalizes phase, solvent molecular
identity, host, doping fraction/basis, temperature in Kelvin, atmosphere,
concentration and method. Missing values remain explicit null/unknown values
and never silently collide with known values. Equivalent JSON key ordering and
Celsius/Kelvin representations produce the same condition digest.

V2 uses the review reasons `exact_duplicate_observation`,
`same_condition_conflicting_observation`, and
`condition_distinct_observation_retained`. Exact duplicate observations are
excluded deterministically; distinct-source conflicts and distinct-condition
observations are retained for human review. The current private mapping policy
uses `partially_comparable_single_solvent`; a stronger unqualified comparable
claim is rejected by v2. The human receipt is
`structured_dataset_confirmation_receipt.v2` and exact-binds the v2 snapshot
schema and digest. V1 publications are never rewritten.

The prepare-task Controller verifier runs the versioned review verifier for
both normal completion and committed-output crash reconstruction. V2
verification rebinds every review row to the exact Raw CSV, checks source and
mapping digests against the Raw publication, reconstructs source evidence,
checks action rosters and normalization policy, and cross-binds property,
molecule, condition, source and conflict identities. Malformed evidence,
missing source-row/paper identity, and DOI/paper disagreement deterministically
exclude the row before human confirmation.

## CI Reference Canary

The public path performs a real deterministic ridge fit on RDKit molecular descriptors from the current run's Confirmed Dataset. Split assignment operates on connected components of the InChIKey–paper bipartite graph; records sharing either a molecule or paper never cross train, test or external holdout, and insufficient independent components fail closed. Candidate generation executes a seed-bound deterministic molecule construction algorithm; it never reads a preset Top-N. Prediction consumes the current model checkpoint and current generated roster. Ranking is deterministic, displays OOD findings and excludes OOD candidates from Top-N according to an explicit rule.

Chemical validation records RDKit validity, canonical SMILES, Standard InChI/InChIKey, duplicates, training-set exact matches, nearest-neighbor similarity, scaffold novelty, AD/OOD status and a no-silent-loss summary.

The current run does not reuse an old model, prediction, generated candidate roster, or `existing_output`.

## Recovery and replay

Every request, checkpoint and publication has a semantic digest. Training request, current Controller dispatch receipt, checkpoint, Registry outputs, verified local publication and Controller action receipt form one exact chain. A pre-existing/copied checkpoint or request is rejected by the backend. If a dispatch occurred but the Controller publication/receipt is missing, only the existing Controller recovery verifier may reconcile it; the backend never adopts or redispatches it. Generation follows the same Controller-owned unknown-outcome rule and has no canary-specific dispatch ledger. Replaced source bytes, registry conflicts and cross-run packages fail closed.

Exact replay verifies the succeeded current Controller execution and its published artifacts without invoking training or generation again. Canonical JSON and semantic digests exclude telemetry. Telemetry is non-authoritative and fail-open.

## Scientific and claim boundary

The frozen engineering scope is organic small-molecule emitters and PLQY. Metal complexes, polymers, exciplex systems, host generation and transport-material generation are excluded. The raw contract retains emission mechanism, medium, host, doping ratio, temperature, measurement condition, paper evidence and comparability. Values from different conditions are never silently merged.

The final output is `Computational Top-N`, not experimental validation or material discovery. It does not assert synthesis feasibility, EQE, lifetime, TADF identity, DFT validation or a closed materials-discovery loop.

No M3H Gate V, M3.5 completion, or Molly v1 completion is claimed unless the corresponding owner-reviewed runtime evidence exists.

The 2026-08-03 private preflight did not create runtime evidence: it stopped
before formal execution because the server-owned remote resource authority
policy was absent and the supplied source CSV was not yet an authoritative BR1
Raw Dataset. `M3H-013` therefore remains without `V` or `DONE` evidence.
