# OLED categorical Gold dataset admission (PR-AH)

## Purpose

PR-AH is the read-only admission boundary after PR-AG. It consumes the exact
PR-AG verification artifact and the separately published categorical Gold
snapshot, replays their binding, and publishes a deterministic decision for
every entry in the complete activated snapshot.

Admission answers only which current dataset-view policies may later consume
each Gold entry. It does not construct legacy `OledGoldDatasetRecord` objects,
build dataset-view rows, or make data training-eligible.

## Exact inputs

The controlled entry requires two distinct regular JSON files:

1. `oled_gold_successor_postwrite_verifier.v1`; and
2. its separately published `oled_categorical_gold_snapshot.v1`.

PR-AH records both exact SHA-256 values and reconstructs their deterministic
publication bytes. Whitespace or key reformatting, a different valid snapshot,
PR-AG digest tamper, symlinked inputs, and timestamp reversal fail closed.

The full PR-AG publication replay is rerun. The snapshot supplied to PR-AH must
equal the snapshot embedded in PR-AG and retain its deterministic entry IDs,
digests, internal uniqueness, ordering, lineage, and activation binding.

## Admission policy

Every Gold entry receives a complete, ordered evaluation of the existing
`OledDatasetViewKind` roster:

| Causal layer | Eligible current view |
| --- | --- |
| `molecule` | `curated_intrinsic` |
| `measurement` | `raw_all_measurements` |
| `interaction` | none |
| `device` | impossible under the upstream Gold contract |

`curated_device_baseline` is not admitted because categorical Gold does not
carry the device/confounder semantics required by that legacy view.
`best_reported` is not admitted because categorical Gold does not carry an
explicit best-reported flag.

Interaction-layer evidence is preserved in Gold but is not coerced into a
molecular, device, or measurement view. A later explicit interaction-view
contract may admit it.

Comparison-context status and hash are copied into each decision. No numeric
confidence is invented.

## Output

Success uses:

```text
status = categorical_gold_dataset_admission_complete
complete_snapshot_roster_replayed = true
causal_layer_view_policy_replayed = true
dataset_view_rows_written = false
dataset_materialized = false
training_eligible = false
```

The artifact contains one decision per snapshot entry, per-view reason codes,
view-level eligible counts, admitted/not-admitted counts, and the exact
upstream models and digests.

## CLI

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_categorical_gold_dataset_admission \
  --gold-successor-verification /operator/local/pr_ag_verification.json \
  --published-categorical-gold-snapshot /operator/local/gold_snapshot.json \
  --output /operator/local/dataset_admission.json
```

Inputs and output must be distinct. Output publication uses a pinned parent,
fresh-file requirement, filesystem sync, and atomic rename. CLI failures expose
only a stable error code and exception type.

## Boundary

PR-AH does not:

- materialize dataset-view rows or write datasets;
- create split assignments, features, training packages, or model inputs;
- construct legacy numeric-confidence Gold records or assign a score;
- write or activate Gold or mutate a head pointer;
- mutate reviewed evidence or the Material Registry; or
- read PDFs, use the network, call external services, LLMs, or MinerU.

The next boundary should materialize only explicitly admitted entries into
versioned dataset-view artifacts, with an exact PR-AH artifact binding. It must
remain separate from split, feature, and training-package generation.
