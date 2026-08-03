# Private Real-Tool Structured Dataset Canary runbook

This runbook freezes the private BR1 path. It does not claim that a private run has occurred.

## Preconditions

Use a trusted private Molly deployment with server-owned configured resource authority for one Uni-Mol training profile and one REINVENT4 generation profile. Do not place endpoints, hostnames, paths, accounts, SSH arguments, credentials, commands, stdout/stderr or raw exceptions in repository files or public evidence.

Confirm that the private dataset satisfies molecule/InChIKey grouping, paper grouping, an independent/external holdout, deduplicated training size and measurement-condition retention. If TADF sufficiency is not demonstrated, set the scope to broader organic emitter PLQY and do not call outputs TADF candidates.

## Procedure

1. Run Raw Dataset inspection and review snapshot creation through the Molly project/run.
2. Before presenting or accepting the Gate, re-read the exact Registry-bound Raw CSV and deterministically rebuild the v2 review. Require exact equality for molecule identity, normalized condition, observed payload, source context, observation/conflict grouping, duplicate/conflict findings, reason codes, actions and confirmed/excluded rosters; nested digest self-consistency alone is insufficient. Then have a trusted human approve that exact snapshot and verify that the shared GateDecision and receipt bind the current project, run, raw/review digests, row rosters, target, role and condition policy.
3. Publish the Confirmed Dataset from that receipt. Raw CSV is not training authority.
4. Prepare `structured_dataset_private_real_tool_request.v1` with logical profile IDs and public provider/version/config digests only. The request requires fresh Uni-Mol and real REINVENT4 and rejects old model, prediction, candidate roster and `existing_output`.
5. Submit the request through existing Permission evaluation, immutable authorization and approve-and-start. Dispatch only through Harness Controller, `RemoteExecutionService` and `molly-worker`.
6. Require completion receipt + Registry binding + verified publication before consuming the current model package. A checkpoint or `StageState SUCCEEDED` alone is insufficient.
7. Dispatch REINVENT4 once. If the outcome is unknown, stop in recovery-required and reconcile the exact dispatch; do not dispatch again.
8. Predict with the current model/current roster, rank with the frozen config, validate chemistry and publish `Computational Top-N`.
9. Restart the process and exact-replay reads. Confirm no second training or generation dispatch occurred.
10. Read the same run through `GET /api/projects/{project_id}/agent-runs/{run_id}/inspection` and correlate Molly with OTel and actual LLM calls in LangSmith. No extra LLM call is required for BR1.

## Evidence review

Private evidence adds Uni-Mol provider/version, verified training publication, REINVENT4 version/config digest, execution classification, generation/prediction/ranking bindings, OTel correlation and LangSmith correlation only where an LLM was actually invoked. Export only privacy-redacted digests, status classes and safe IDs.

Do not mark `M3H-013 I/T/V / DONE` until fresh Uni-Mol, real REINVENT4, current-run prediction/ranking/validation, restart/replay evidence and repository-owner exact-HEAD review all exist. BR1 alone never closes M3H-GATE-006, M3.5 or Molly v1.

## 2026-08-03 preflight result

The first post-merge private preflight failed closed before an acceptance ID or
run was created. The repository-owned Uni-Mol and REINVENT4 logical profiles
had current available capability probes, but the server-owned remote resource
authority policy was not configured. The supplied source CSV also required an
authoritative BR1 Raw Dataset mapping because it did not contain the required
contract columns as-is.

The uploaded source also lacked dataset name/version, source URL, license and
download date. Those values remain unknown and are recorded as
`SOURCE_PROVENANCE_MISSING`. In the numeric-QY subset, repeated chromophores
across solvent conditions conflict with an InChIKey-only duplicate exclusion;
this is recorded as `CONDITION_AWARE_IDENTITY_POLICY_UNRESOLVED`. Do not map
that subset into BR1 Raw Dataset rows until source provenance, field mapping and
the condition-aware identity policy are authoritative.

No Controller execution, confirmation, training, generation, restart or replay
was attempted. The privacy-safe finding is recorded in
`docs/evidence/br1-private-real-tool-canary-v1/`. After both blockers are
resolved, use a new clean acceptance ID and run ID; do not amend this blocked
preflight into a successful acceptance.
