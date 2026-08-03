# Evidence summary

Status: `BLOCKED` during preflight.

- PR #27 reviewed HEAD: `e7bb7fc45f0c514ccce000d7fff25ee66e967bd6`
- PR #27 squash merge: `e9316ad64fd219a26740b3424f541e00a9409d39`
- Resource profiles: current probes available for `unimol-train-v1` and `reinvent4-cpu-v1`
- Resource authority policy: not configured
- Raw source sufficiency: identity/target scale exists, but source provenance and the BR1 Raw Dataset contract are incomplete
- Condition-aware identity: unresolved; 10,697/13,978 numeric-QY rows are in repeated raw chromophore groups and 2,309 chromophores span multiple solvents
- Scientific scope: unassigned; neither TADF emitter PLQY nor broader organic emitter PLQY is claimed
- Formal acceptance ID/run: not created
- Human confirmation: not run
- Uni-Mol dispatch count: `0`
- REINVENT4 dispatch count: `0`
- Restart/exact replay: not run
- Inspection/Top-N: not created
- OTel: not applicable because execution did not start
- LangSmith: `not_applicable_no_llm_call`
- Canonical blocker roster, in fixed order:
  `REMOTE_RESOURCE_AUTHORITY_POLICY_MISSING`,
  `SOURCE_PROVENANCE_MISSING`, `DATA_CONTRACT_INCOMPLETE`,
  `CONDITION_AWARE_IDENTITY_POLICY_UNRESOLVED`
- Focused tests: `46 passed`
- Local PR Fast after review fixes: `1157 passed, 5473 deselected`
- GitHub PR Fast / CodeQL at exact HEAD
  `7935f22b3a7df2f128e0f8e5e4c24bf02169dbc9`: passed
- Full CI run `30779825797` at that exact HEAD: passed, including static
  validation and all four weighted shards

No old model, prediction, candidate roster, deterministic fake provider, or
`existing_output` was used. This evidence can only establish a fail-closed
preflight blocker.
