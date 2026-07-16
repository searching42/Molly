# OLED categorical dataset execution slice (PR-AI)

## Purpose

PR-AI turns the PR-AH admission result into a useful, reproducible dataset and
baseline smoke artifact. It deliberately combines the minimum materialization,
split, feature, and baseline steps in one vertical slice instead of creating
another sequence of governance-only PRs.

The source of truth remains one exact PR-AH artifact. Only decisions explicitly
admitted by PR-AH are materialized.

## Execution

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_categorical_dataset_execution \
  --dataset-admission /operator/local/pr_ah_admission.json \
  --output-root /operator/local/versioned_datasets
```

The command creates:

```text
versioned_datasets/
  <dataset_snapshot_id>/
    snapshot.json
    rows.jsonl
    split_assignments.jsonl
    baseline_predictions.jsonl
    baseline_metrics.json
    report.md
```

The snapshot ID binds the PR-AH digest, materialization policy version, and
ordered row digests. Existing snapshot directories are never overwritten.
Files are synced before the temporary directory is atomically renamed.

## Dataset rows

Each row preserves:

- PR-AH decision ID and digest;
- categorical Gold entry/candidate IDs and digests;
- Registry material ID, entry digest, and canonical isomeric SMILES;
- normalized target value/unit and reported precision;
- causal layer and comparison context;
- PDF/table/cell evidence references; and
- a deterministic 128-bit Morgan fingerprint, with the existing hashed
  fallback when RDKit is unavailable.

PR-AH routing remains authoritative:

- molecule → `curated_intrinsic`;
- measurement → `raw_all_measurements`;
- interaction and unadmitted decisions → zero rows.

## Split and baseline

Rows are grouped by Registry material ID before splitting, so one material
cannot cross train/validation/test. The same split applies across properties.

PR-AI calls the existing
`run_oled_mean_baseline_on_training_rows()` implementation for every numeric
property/view group:

- with holdout material groups, it reports train/validation/test metrics;
- with only one material group, it still writes a transparent train-only smoke
  result and records `insufficient_material_groups_for_holdout`.

Train-only metrics are not scientific evaluation. They prove that provenance,
features, splits, and the baseline execution path connect end to end.

## Boundary

PR-AI does not:

- invent numeric confidence or construct legacy Gold records;
- materialize entries rejected by PR-AH;
- claim benchmark validation, scientific model quality, or training readiness;
- register or promote a model;
- mutate Gold or Registry state; or
- call LLMs, MinerU, external services, or the network.

The next capability milestone should run this slice over genuinely adjudicated
real-paper Gold spanning enough distinct materials for holdout evaluation,
then report extraction yield, correction cost, and baseline error slices.
