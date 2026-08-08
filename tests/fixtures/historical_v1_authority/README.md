# Historical v1 authority fixtures

These artifacts are byte-exact publications generated with the pre-PR #38
code at commit `bf82cfd` (the PR base), using the same writers that ran in
production:

- `publication/` — complete Proposal publication
  (`agent_execution_plan_proposal.v1`, no `authorization_scope_digest`).
- `control/permission_decision/` — PermissionDecision under
  `scientific-agent-permission-policy.v3` (legacy task-authority algorithm).
- `control/authorization/` — Authorization under
  `agent_plan_authorization.v1` (option values part of the digest).
- `control/harness_controller_execution/` — Controller execution tagged
  `scientific-agent-harness-controller-policy.v1` with the legacy digest
  material (per-slot `compiled_options_digest` included).

`manifest.json` records the stable identities and digests asserted by
`tests/test_historical_v1_authority_fixture_replay.py`.

Regeneration (only needed when the compatibility contract itself changes):
check out `bf82cfd`, build one local authority chain
(`_local_authority_chain` in the controller tests), dump the Proposal
publication payloads and the control artifacts with
`ScientificAgentPlanProposalStore._publication_payloads` /
`AgentPlanControlStore._publication_payloads`, and tag the Controller
execution `scientific-agent-harness-controller-policy.v1` before revalidating
with the legacy writer so its digest is computed by the old code.
