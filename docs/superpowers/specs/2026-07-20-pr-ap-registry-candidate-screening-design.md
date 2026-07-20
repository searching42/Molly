# PR-AP Registry Candidate Screening Design

## Purpose

PR-AP turns one exact PR-AO model execution into an independent candidate-pool
screening run over one immutable OLED Material Registry snapshot. It closes the
gap between holdout evaluation and discovery-oriented prediction: PR-AO trains
and evaluates models on labeled dataset rows, while PR-AP applies those models
to Registry materials that were not used for training and produces a reviewable
shortlist.

PR-AP is an execution boundary. It does not retrain or register models, mutate
the Registry, generate molecules, assert experimental validity, or promote a
shortlist into Gold or a project asset.

## Chosen approach

PR-AP consumes the existing `OledMaterialRegistrySnapshot` directly rather than
introducing a candidate-pool manifest or accepting a legacy CSV. The Registry
already supplies stable material identity, canonical isomeric SMILES, entry
digests, aliases, and an immutable snapshot digest. Reusing it avoids a parallel
identity contract and makes train/candidate isolation independently verifiable.

The alternatives were rejected for this PR:

- A new candidate manifest would support external designed molecules but would
  add another provenance and validation lifecycle before screening works.
- A legacy candidate CSV would be quick to wire but would weaken identity and
  leakage checks and bypass the Registry lifecycle completed in PR-W through
  PR-Y.

External or generated candidate pools can be added later through an explicit
Registry admission path or a separately designed ephemeral-candidate boundary.

## Inputs and exact bindings

The command accepts:

1. a PR-AO execution directory containing `execution.json` and all referenced
   `model__<property>.json` files;
2. the exact PR-AI dataset `snapshot.json` bound by that execution; and
3. one immutable `OledMaterialRegistrySnapshot` JSON to use as the candidate
   pool.

The source dataset snapshot remains an explicit input even though model files
contain training material IDs. PR-AP independently re-derives the complete
training identity set from the source dataset rows and split assignments, then
cross-checks the derived train row/material roster against each model. This
prevents a forged or incomplete model declaration from weakening leakage
protection.

Every input is read from a pinned regular-file descriptor without following
symbolic path components. Duplicate JSON keys and non-finite values are
rejected. PR-AP records exact file SHA-256 values and validates:

- the PR-AO execution ID and execution version;
- execution receipt artifact hashes against exact model bytes;
- model execution ID, source dataset ID/digest/SHA, property ID, model kind,
  feature names, and training row/material IDs;
- the source dataset artifact, row digests, split assignments, snapshot digest,
  and row-material split binding;
- model training identities against the train split re-derived from the source
  dataset;
- Registry entry digests, sorted unique entries, identity uniqueness, and
  Registry snapshot digest.

Any missing binding or inconsistency fails before prediction or output
publication.

## Candidate independence

PR-AP derives three training identity sets from all train rows selected for the
screened properties:

- `selected_material_id`;
- `registry_entry_digest`; and
- canonical isomeric SMILES.

A Registry entry is excluded if it intersects any of these sets. Checking all
three dimensions protects against identifier drift, duplicate Registry entries,
or the same structure being represented under another material ID. Exclusion is
not silent: every rejected Registry entry is written with sorted reason codes
and its stable Registry identity.

Candidate entries must also have a nonempty canonical isomeric SMILES and yield
features compatible with every selected model. Feature failures and incomplete
property prediction are exclusions, not partial shortlist rows.

`eligible_candidates.csv` means candidates that pass independence, structure,
feature, and complete-prediction checks. Hard-constraint failures and Pareto
dominance do not remove a candidate from this file or `predictions.jsonl`; they
only prevent membership in `ranked_shortlist.csv`, with the decision flags and
reason codes retained on the prediction row. `excluded_candidates.jsonl` is
reserved for candidates that cannot legally or technically enter prediction.

## Feature generation and prediction

PR-AP reuses the PR-AO molecular feature implementation and prediction
semantics rather than maintaining a second mathematical path. Shared helpers
will be extracted only where required for exact equivalence:

- feature generation must produce the exact ordered feature vector declared by
  each model;
- only the supported `linear_kernel_ridge.v1` model kind is accepted;
- prediction uses the model's feature means, centered training features, target
  mean, and dual coefficients;
- every numeric output must be finite.

All selected models must agree on the source execution and feature contract.
PR-AP does not fall back to a different fingerprint backend or silently fill
missing feature columns.

## Constraints and scoring

The CLI accepts optional repeated predicted-value constraints:

```text
--max delta_e_st_ev=0.20
--min s1_ev=2.80
```

Constraints must reference a selected property, use finite numeric thresholds,
and contain no contradictory duplicate definitions. No domain threshold is
invented when the operator supplies none. Each candidate records the raw
predictions, each constraint result, and the overall hard-constraint status.

Objective directions come from the exact PR-AO execution configuration. PR-AP
does not use candidate-pool min/max normalization as an absolute utility score,
because an extreme candidate would change every other candidate's score.
Instead it computes:

1. Pareto dominance from raw predictions after applying objective directions;
2. a deterministic within-pool percentile rank per property; and
3. the unweighted mean percentile as a secondary ordering value.

The shortlist contains candidates that pass every hard constraint and are
Pareto non-dominated. It is ordered by descending mean percentile, then stable
material ID. The percentile aggregate is labeled as a ranking aid, not a
scientific utility or calibrated probability. Dominated or constraint-failing
candidates remain in the full predictions and exclusion/decision artifacts.

## Outputs and publication

The screening ID is deterministic over the exact PR-AO execution ID and SHA,
source dataset digest and SHA, Registry snapshot digest and SHA, selected
properties, constraints, objective directions, feature policy, and scoring
policy.

PR-AP publishes one versioned directory:

```text
<screening_id>/
  eligible_candidates.csv
  excluded_candidates.jsonl
  predictions.jsonl
  ranked_shortlist.csv
  screening.json
  report.md
```

`screening.json` binds every exact input and output SHA, counts candidates at
each decision stage, records configuration and reason-code counts, and carries
explicit boundary claims:

- `independent_registry_candidate_pool=true`;
- `training_identity_exclusion_applied=true`;
- `experimental_validation_claimed=false`;
- `benchmark_validated=false`;
- `production_ready=false`;
- `model_registered=false`;
- `registry_mutated=false`.

Publication reuses the repository's pinned-parent, invocation-owned temporary
directory, full-write, fsync, atomic no-replace rename, exact-byte post-write
validation, and owned-inode cleanup implementation. Existing targets are never
overwritten. Parent replacement, symbolic components, concurrent target
creation, short writes, or post-write changes fail closed.

## Failure semantics

Input and contract failures abort the complete invocation and publish nothing.
Examples include changed model bytes, a mismatched execution receipt, a missing
source dataset, forged training identities, Registry digest failure, unsupported
model kinds, and inconsistent feature contracts.

Candidate-local scientific input failures do not abort an otherwise valid run.
An invalid candidate SMILES, training overlap, incompatible feature vector, or
non-finite prediction is recorded in `excluded_candidates.jsonl`. The run fails
if no eligible candidate remains or if no candidate satisfies the complete
selected-property prediction contract.

CLI failures emit only a stable error code and exception type. Absolute local
paths, molecule structures, or private artifact content are not printed to
stdout/stderr.

## Testing and real-data validation

Focused tests cover:

- a valid independent Registry pool producing complete predictions and a
  deterministic shortlist;
- exclusion on training material ID, Registry digest, or canonical SMILES;
- re-signed source-dataset leakage and model-roster tampering;
- exact execution/model/source SHA and digest failures;
- duplicate Registry identities and malformed Registry snapshots;
- feature incompatibility and incomplete multi-property prediction;
- min/max constraints, Pareto dominance, percentile aggregation, and stable
  tie-breaking;
- duplicate JSON keys, symlink path components, input replacement, concurrent
  output targets, output-parent replacement, short writes, and post-write byte
  validation;
- zero-output behavior for invocation-level failures.

The PR also runs a real canary using the paper016 PR-AO three-model execution
and its seven-entry immutable Registry successor snapshot. Five Registry
materials used by PR-AO's train split should be excluded; the validation and
test materials should remain as two non-training candidates. PR-AP must not read
their target labels for prediction or scoring even though labels exist in the
source dataset used to verify training isolation. The result reports actual
total, training-overlap, feature-failure, eligible, Pareto, and shortlist
counts. This canary proves executable integration and train/candidate isolation
only. Reusing two labeled holdout materials does not establish external-corpus
model quality, experimental validity, or discovery success; a later run needs a
larger Registry containing genuinely unlabeled independent candidates.

## Acceptance criteria

PR-AP is complete when one command can consume exact PR-AO, PR-AI, and Registry
artifacts; independently exclude all training identities; predict every selected
property for eligible Registry entries; apply optional explicit constraints;
produce a deterministic Pareto-based shortlist; publish a no-replace versioned
artifact directory; pass adversarial input/publication tests; and complete the
paper016 real canary without making promotion or experimental claims.
