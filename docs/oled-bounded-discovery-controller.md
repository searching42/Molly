# PR-AU bounded closed-loop discovery controller

PR-AU adds a deterministic, executable control decision over the implemented
OLED candidate flow. It is an agentic workflow controller, not an autonomous
scientist and not a generator wrapper.

## Fixed action space

For an exact iteration history the controller may publish only:

- `stop`; or
- `request_generation_approval` for the existing
  `execute_oled_inverse_design` task and `gate_5_final_threshold`.

The controller never invokes REINVENT, approves a gate, predicts a structure,
or mutates Registry, Gold, dataset, or model state. A generated action remains
a separate gated execution, so one controller invocation cannot silently run
multiple generation rounds.

When, and only when, it requests generation, PR-AU emits a narrow
`generation_authorization.json`. It binds the existing
`execute_oled_inverse_design` task to the controller ID, the latest source-state
fingerprint, the exact requested candidate count, the fixed final-threshold
gate, and the exact PR-ARb/PR-AP/PR-AO/PR-AI/Registry/model source bindings.
The executor freezes this bundle before writing the PR-AS gate snapshot, then
replays it again before adapter dispatch. A direct PR-AS invocation remains
possible, but it cannot claim to be the controller-routed action unless it
consumes this authorization.

## Request and exact replay

The canonical request contains only:

- `request_version`;
- `limits`;
- one to three `iterations`.

Each iteration supplies the exact PR-ARb v2, PR-AT, PR-AS, original PR-ARb,
PR-AP, PR-AO, PR-AI, and Registry paths required by the existing independent
verifiers. The controller replays every iteration and records the resulting
decision, evaluation, and generation-publication identities and SHA-256 values.

Every iteration must have the same immutable loop fingerprint: Top-N target,
property constraints and directions, budget/currency, diversity threshold,
selection policy, PR-AP screening anchor, model binding, Phase-1 execution,
dataset snapshot, and Registry snapshot. Only the PR-AS publication, the
cumulative PR-AT evaluation, and the matching PR-ARb v2 decision may advance.

Candidate IDs must form a monotonic superset across iterations. The controller
also keeps a three-part chemical identity ledger over canonical isomeric
SMILES, standard InChI, and InChIKey. It rejects a reissued structure under a
different publication-scoped candidate ID, a candidate ID rebound to a new
structure, or a later cumulative pool that drops an earlier chemical identity.

## Hard ceilings

Callers may request smaller positive budgets, but never values above:

```yaml
max_iterations: 3
max_generation_rounds: 2
max_generated_candidates: 512
```

The generated-candidate limit is cumulative over unique PR-AS publications,
using each publication's accepted/source candidate count rather than only the
subset that later reaches PR-AT prediction. Thus identity overlaps and
feature/prediction failures still consume the generation budget. The controller
stops before requesting an action that would exceed a ceiling.

## Stop and continuation semantics

The controller stops when:

- PR-ARb v2 has a complete Top N;
- the complete PR-AT prediction pool has enough property-qualified supply and
  a non-supply policy prevents a complete Top N;
- any iteration, generation-round, or generated-candidate budget is exhausted.

Only a true property-qualified supply shortfall may produce
`request_generation_approval`. The receipt and authorization include the
shortfall-sized requested candidate count, the required gate, and the target
existing task. This is a bounded, executable routing grant—not evidence that
generation ran or was approved.

## Publication

The immutable publication contains exactly:

- `controller.json`;
- `controller_request.json`;
- `generation_authorization.json`;
- `report.md`.

The RunPlan executor independently replays the controller while keeping the
publication descriptor-pinned through atomic artifact registration. A completed
controller task is idempotent and does not dispatch again on retry.
