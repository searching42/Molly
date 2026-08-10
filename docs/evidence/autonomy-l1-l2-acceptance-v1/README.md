# Autonomy L1/L2 representative acceptance v1

This directory contains the reviewed, privacy-safe projection of the formal
Autonomy L1/L2 acceptance run. The `V` claim comes from the finite runtime
runner, not from ordinary pytest success:

```bash
python scripts/run_autonomy_l1_l2_acceptance.py \
  --expected-code-head <ACCEPTANCE_CODE_HEAD> \
  --output-dir <TEMP_OUTPUT>
```

The runner requires an exact clean code HEAD and exercises the existing Flask
control-plane/session, Permission, Authorization, StartIntent, Controller,
Execution Agent, Replanner, policy, budget, and filesystem-backed immutable
store paths. It uses only deterministic external-edge doubles (stub LLM,
synthetic remote lifecycle, injected clock, and fault injection). One L2
crash/replay scenario crosses separate Python processes; the L1 remote
adoption scenario is explicitly a durable receipt crash-window reconciliation,
not a process-boundary restart claim. PR #43 remains the complementary
evidence for the real BR1 remote restart substrate.
The L2 replay scenario also drives two near-concurrent `/replan` requests and
verifies one provider call and one successor publication.
The L1 budget scenarios enter the real conversation `tick()` path after
preloading exact durable evidence and verify zero Controller/provider effects
at the exhausted boundary. The L1→L2 handoff also enters a real B-side tick
and verifies that its budget snapshot is rebuilt from Controller B evidence.

The acceptance code was frozen at
`02deae19194a20c079253dabc917cfe7c9a05945`; after that freeze, only this
evidence projection, its contract checks, and the roadmap were finalized. The
final evidence commit is recorded in the pull request body.

Evidence files:

- [acceptance manifest](acceptance_manifest.json): exact code head, policy
  identities, scenario counts, and authority invariants.
- [scenario matrix](scenario_matrix.json): the stable `AUT-A01`–`AUT-A16`
  roster and bounded result projections.
- [restart/replay summary](restart_replay_summary.json): explicit
  exactly-once and process-boundary fields.
- [authority-boundary summary](authority_boundary_summary.json): human
  boundaries, stale-authority, retry, and read-only-effect assertions.

The scope is `control_plane_autonomy_only`. It does not rerun the real 1999-row
BR1 workflow, make scientific-performance claims, start BR2 implementation, or
start UI implementation. It proves only bounded, fail-closed, restart/replay,
exactly-once, and authority-preserving L1/L2 control-plane behavior.
