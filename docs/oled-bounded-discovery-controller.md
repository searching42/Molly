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

## Request and exact replay

The canonical request contains only:

- `request_version`;
- `limits`;
- one to three `iterations`.

Each iteration supplies the exact PR-ARb v2, PR-AT, PR-AS, original PR-ARb,
PR-AP, PR-AO, PR-AI, and Registry paths required by the existing independent
verifiers. The controller replays every iteration and records the resulting
decision, evaluation, and generation-publication identities and SHA-256 values.

Candidate IDs must form a monotonic superset across iterations. This prevents a
later generation round from silently discarding candidates found in an earlier
round. Until a cumulative PR-AT publication is available, a second iteration
that drops earlier generated candidates fails closed.

## Hard ceilings

Callers may request smaller positive budgets, but never values above:

```yaml
max_iterations: 3
max_generation_rounds: 2
max_generated_candidates: 512
```

The generated-candidate limit is cumulative over the latest verified monotonic
pool. The controller stops before requesting an action that would exceed a
ceiling.

## Stop and continuation semantics

The controller stops when:

- PR-ARb v2 has a complete Top N;
- the complete PR-AT prediction pool has enough property-qualified supply and
  a non-supply policy prevents a complete Top N;
- any iteration, generation-round, or generated-candidate budget is exhausted.

Only a true property-qualified supply shortfall may produce
`request_generation_approval`. The receipt includes the shortfall-sized
requested candidate count, the required gate, and the suggested existing task.
It is a routing request, not evidence that generation ran or was approved.

## Publication

The immutable publication contains exactly:

- `controller.json`;
- `report.md`.

The RunPlan executor independently replays the controller while keeping the
publication descriptor-pinned through atomic artifact registration. A completed
controller task is idempotent and does not dispatch again on retry.
