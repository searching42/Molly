# Evidence summary

Status: `BLOCKED` during preflight.

- PR #27 reviewed HEAD: `e7bb7fc45f0c514ccce000d7fff25ee66e967bd6`
- PR #27 squash merge: `e9316ad64fd219a26740b3424f541e00a9409d39`
- Resource profiles: current probes available for `unimol-train-v1` and `reinvent4-cpu-v1`
- Resource authority policy: not configured
- Raw source sufficiency: identity/target scale exists, but the BR1 Raw Dataset contract is incomplete
- Scientific scope: unassigned; neither TADF emitter PLQY nor broader organic emitter PLQY is claimed
- Formal acceptance ID/run: not created
- Human confirmation: not run
- Uni-Mol dispatch count: `0`
- REINVENT4 dispatch count: `0`
- Restart/exact replay: not run
- Inspection/Top-N: not created
- OTel: not applicable because execution did not start
- LangSmith: `not_applicable_no_llm_call`
- Focused tests: `45 passed`
- PR Fast: `1157 passed, 5472 deselected`
- Full CI / CodeQL: pending Draft PR

No old model, prediction, candidate roster, deterministic fake provider, or
`existing_output` was used. This evidence can only establish a fail-closed
preflight blocker.
