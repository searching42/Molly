# OLED real Phase 1 execution

This command turns one published PR-AI dataset snapshot into an executable,
non-promotional modeling canary:

```bash
PYTHONPATH=src .venv/bin/python -m ai4s_agent.oled_real_phase1_execution \
  --dataset-snapshot /absolute/path/to/snapshot.json \
  --output-root /absolute/path/to/executions
```

The runner validates the complete snapshot contract, fits one deterministic
linear kernel-ridge model per numeric property using only `train` materials,
predicts the validation/test materials, computes split metrics, and produces a
multi-objective ranking. `delta_e_st_ev` is minimized by default; other
properties are maximized. Repeat `--minimize-property-id PROPERTY_ID` to change
an objective direction, and repeat `--property-id PROPERTY_ID` to select a
subset of targets.

The versioned output directory contains:

- `model__<property>.json`: self-contained, replayable fitted model parameters;
- `predictions.jsonl` and `metrics.json`: train/validation/test results;
- `ranked_candidates.csv`: validation/test materials only;
- `execution.json`: exact source snapshot binding, configuration, result
  summary, and artifact SHA-256 values;
- `report.md`: a short human-readable canary report.

The output is published with atomic no-replace directory semantics. It is a
real model execution, but remains explicitly `benchmark_validated=false`,
`production_ready=false`, and `model_registered=false`. Promotion requires a
larger, externally validated corpus and is outside this runner.
