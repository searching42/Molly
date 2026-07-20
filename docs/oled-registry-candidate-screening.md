# OLED Registry candidate screening

PR-AP applies one exact PR-AO model execution to an immutable Material Registry
snapshot and publishes a reviewable shortlist of materials that were not used
for model training.

## Command

```bash
PYTHONPATH=src .venv/bin/python \
  -m ai4s_agent.oled_registry_candidate_screening \
  --phase1-execution-dir /absolute/path/to/pr-ao-execution \
  --dataset-snapshot /absolute/path/to/pr-ai-snapshot.json \
  --registry-snapshot /absolute/path/to/material_registry_snapshot.json \
  --output-root /absolute/path/to/screenings \
  --max delta_e_st_ev=0.20 \
  --min s1_ev=2.80
```

`--min` and `--max` are optional and repeatable. A constraint must reference a
property selected by the exact PR-AO execution. PR-AP does not invent scientific
thresholds when none are supplied.

## Exact inputs and training isolation

The runner independently validates the bound PR-AI snapshot and immutable
Registry snapshot, then exactly replays the complete PR-AO publication from the
persisted dataset, configuration, and generation timestamp. The execution
directory basename, complete file roster, and every byte of every model,
prediction, metric, ranked CSV, receipt, and report must match the replay. It
re-derives train identities from the dataset rather than trusting model
metadata alone.

A Registry entry cannot enter prediction if its material ID, Registry entry
digest, canonical isomeric SMILES, regenerated standard InChI, or regenerated
InChIKey occurs in the train split. All matching reason codes are retained in
`excluded_candidates.jsonl`.

Before candidate inference, the runner regenerates every selected training
row's feature vector from its canonical SMILES using the exact 128-bit,
radius-2 feature profile. Feature version, backend/fallback state, column
roster, and every bit must equal the persisted PR-AI row. The verified
generator profile and RDKit runtime version are recorded in `screening.json`.

## Prediction and ranking

Eligible entries use the exact feature names and linear kernel-ridge prediction
parameters written by PR-AO. Every candidate must receive finite predictions
for every selected property.

The runner then records:

- explicit hard-constraint results;
- direction-aware Pareto dominance;
- a within-pool percentile for each property; and
- the mean percentile used only as a deterministic secondary rank.

The shortlist contains candidates that pass every supplied constraint and are
Pareto non-dominated. Percentiles are ranking aids, not calibrated scientific
utilities or probabilities.

## Published artifacts

The deterministic, atomic no-replace output directory contains:

- `eligible_candidates.csv`: candidates that passed identity, feature, and
  complete-prediction checks;
- `excluded_candidates.jsonl`: candidates blocked before prediction;
- `predictions.jsonl`: all complete predictions and screening decisions;
- `ranked_shortlist.csv`: hard-constraint-passing Pareto candidates;
- `screening.json`: exact source/output bindings, configuration, counts, reason
  codes, and bounded claims; and
- `report.md`: concise human-readable results.

PR-AP does not retrain or register models, mutate the Registry, validate a
benchmark, establish experimental performance, or promote shortlist entries.

## Agent-managed execution (PR-AQ)

`execute_oled_registry_candidate_screening` makes this runner available through
the controlled RunPlan agent path.  A Registry-screening goal in the planner or
chat preview selects this task instead of falling back to the generic report
workflow.

The caller must explicitly provide all three exact artifact paths under these
artifact IDs; the agent never searches for a latest version:

- `oled_phase1_execution_dir` — one complete PR-AO execution directory;
- `oled_dataset_snapshot` — the PR-AI categorical dataset snapshot used by
  that execution; and
- `oled_registry_snapshot` — one immutable Material Registry snapshot.

Optional task options are only repeated string lists named `minimums` and
`maximums`, using the same `property=value` syntax as the CLI.  The first
`/api/run-plan/execute` call records a content manifest for the exact inputs
and pauses at `gate_5_final_threshold`; it does not run screening.  Resume
through `/api/run-plan/resume` must provide the same input artifact paths and
unchanged constraints along with that gate approval.  Any changed path, input
bytes, execution-directory contents, or constraint fails closed before PR-AP
is invoked.

PR-AQ fixes publication below the project run directory at
`oled_registry_screening/<screening_id>/`.  It registers the PR-AP receipt,
shortlist, predictions, exclusions, eligible candidates, report, and the task
execution record as run artifacts.  There is no overwrite option: rerunning
the same deterministic screening inside the same run fails without replacing
the original publication.

## paper016 canary

The real paper016 canary binds the 21-row PR-AI snapshot, three PR-AO models,
and the verified seven-entry Registry successor. It excludes the five train
materials and predicts the validation/test materials as two non-training
candidates. Both reach the shortlist when no hard constraint is supplied.

Those two candidates have labels elsewhere in the source dataset, although
PR-AP does not read the labels for prediction or scoring. The canary therefore
proves exact integration and train/candidate isolation, not external-corpus
model quality or discovery success.
