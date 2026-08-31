# CORE-06A — BR1 scientific plugin contract

Status: PASS for the offline/local contract checkpoint. This checkpoint does
not establish B2 fresh-real BR1 parity, B3 remote restart evidence, or B4
cutover approval.

## Scope

CORE-06A adds the optional `molly.plugins.br1_inverse_design` package. It
keeps BR1 outside `molly.core` and provides the bounded production tool seam:

```text
reviewed dataset / migrated accepted-real dataset
→ applicability preflight
→ fresh training occurrence
→ generation occurrence
→ current-run model prediction
→ deterministic computational Top-N evaluation
```

The plugin binds each stage to exact input artifact IDs, the current run's
successful RunLedger occurrence, server-owned version/config/profile
references, and explicit seed/config digests. The output claim boundary is
`COMPUTATIONAL_ONLY`; no experimental claim is produced.

## Files

- `src/molly/plugins/br1_inverse_design/`: schemas, reviewed-dataset gate,
  deterministic contract runtime, stage services, current-run bindings, and
  AgentLoop ToolSpecs.
- `tests/molly/test_core06_br1_plugin.py`: offline end-to-end chain,
  restart inspection, foreign-model rejection, deterministic evaluation, and
  import-boundary checks.
- `tests/molly/test_core06_br1_acceptance_contract.py`: migrated dataset and
  server-owned configuration contract checks.

The implementation checkpoint is commit `c7e2b8b`; the macro branch base is
the merged Scientific Intake `main` commit
`d533d35751ebd6c88befaf63f540da9b10dbaedc`.

## Evidence

The focused CORE-06A command passed:

```text
PYTHONPATH=src PYTHONHASHSEED=0 python -m pytest -q \
  tests/molly/test_core06_br1_plugin.py \
  tests/molly/test_core06_br1_acceptance_contract.py
8 passed
```

`compileall` and `git diff --check` also passed for the checkpoint. Full CI is
intentionally deferred until the complete CORE-06 macro merge candidate.

## Limitations and next checkpoint

The runtime used by these tests is deterministic and contract-only; it does
not execute Uni-Mol, REINVENT4, GPU work, or remote work. The real reviewed
dataset import helper records exact source and transformation digests without
publishing source bytes or private paths. CORE-06B adds the durable local/
remote compute seam; CORE-06C must perform a fresh-real run before B2 can be
marked PASS. CORE-07 is not started.
