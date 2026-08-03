# Private Real-Tool Structured Dataset Canary runbook

This runbook freezes the private BR1 path. It does not claim that a private run has occurred.

## Preconditions

Use a trusted private Molly deployment with server-owned configured resource authority for one Uni-Mol training profile and one REINVENT4 generation profile. Do not place endpoints, hostnames, paths, accounts, SSH arguments, credentials, commands, stdout/stderr or raw exceptions in repository files or public evidence.

Confirm that the private dataset satisfies molecule/InChIKey grouping, paper grouping, an independent/external holdout, deduplicated training size and measurement-condition retention. If TADF sufficiency is not demonstrated, set the scope to broader organic emitter PLQY and do not call outputs TADF candidates.

## Procedure

1. Deploy the reviewed repository commit and the matching `molly-worker` package. Confirm the server registry is exactly `br1-private-real-tool-v3` and the Uni-Mol execution profile is exactly `unimol-train-br1-v2`.
2. In the actual configured Uni-Mol Python environment, run the operator-side applicability preflight against the exact Raw CSV, `source_dataset_manifest.v1`, `br1_raw_dataset_mapping_policy.v1`, and the matching source publication/authority artifacts. The command accepts private paths only; publish the complete report privately and export only the summary:

   ```bash
   python scripts/run_br1_unimol_applicability_preflight.py \
     --raw-dataset <private-path> \
     --source-manifest <private-path> \
     --mapping-policy <private-path> \
     --source-authority <private-path> \
     --source-publication <private-path> \
     --source-publication-registry <private-path> \
     --output-report <private-path> \
     --public-summary <private-path> \
     --expected-provider-version <reviewed-unimol-tools-version>
   ```

   Pass the reviewed commit through `--repository-commit` or the private
   `MOLLY_REPOSITORY_COMMIT` environment variable when the installed worker is
   not inside a Git checkout. The expected provider version must come from the
   reviewed worker/capability authority and is exact-compared with the
   installed provider version; omitting it is `BLOCKED` with
   `PROVIDER_VERSION_AUTHORITY_UNAVAILABLE`. The source authority binds the
   source artifact, publication registry, raw digest, canonical source bytes,
   mapping binding, canonical provider-input bytes, row count, provider
   version, and profile digest. The runner rereads the Raw CSV before any
   provider preprocessing and rejects a digest or canonical-byte change.
   Do not put provider paths or probe output in public evidence. The
   project-owned `unimol-tools 0.1.5` adapter uses `DataHub(is_train=False,
   conf_cache_level=0)` in the configured provider Python environment; it
   never constructs `MolTrain`, calls `fit()`, or creates model artifacts.
   A missing or unverified capability contract remains `BLOCKED` with an
   unresolved provider reason; no training attempt may be used as a probe.
   Before exporting or using the summary, verify it against the exact private
   report with the report-bound summary verifier. A schema-valid summary that
   is not the deterministic projection of that report must be rejected.
3. If the summary is `PASS`, freeze the final Raw CSV, source manifest and mapping policy bytes and run Raw Dataset inspection and review snapshot creation through the Molly project/run.
4. Before presenting or accepting the Gate, re-read the exact Registry-bound Raw CSV and deterministically rebuild the v2 review. Require exact equality for molecule identity, normalized condition, observed payload, source context, observation/conflict grouping, duplicate/conflict findings, reason codes, actions and confirmed/excluded rosters; nested digest self-consistency alone is insufficient. Then have a trusted human approve that exact snapshot and verify that the shared GateDecision and receipt bind the current project, run, raw/review digests, row rosters, target, role and condition policy.
5. If the summary is `REVIEW_REQUIRED`, stop and submit only the counts and reason summary to the owner for an explicit exclusion decision. Do not silently project or delete unsupported rows.
6. If the summary is `BLOCKED`, fix the provider/environment authority or revise the data scope. Do not start an acceptance run.
7. Publish the Confirmed Dataset from that receipt. Raw CSV is not training authority.
8. Prepare `structured_dataset_private_real_tool_request.v1` with logical profile IDs and public provider/version/config digests only. The request requires fresh Uni-Mol and real REINVENT4 and rejects old model, prediction, candidate roster and `existing_output`.
9. Submit the request through existing Permission evaluation, immutable authorization and approve-and-start. Dispatch only through Harness Controller, `RemoteExecutionService` and `molly-worker`.
10. Require completion receipt + Registry binding + verified publication before consuming the current model package. A checkpoint or `StageState SUCCEEDED` alone is insufficient.
11. Dispatch REINVENT4 once. If the outcome is unknown, stop in recovery-required and reconcile the exact dispatch; do not dispatch again.
12. Predict with the current model/current roster, rank with the frozen config, validate chemistry and publish `Computational Top-N`.
13. Restart the process and exact-replay reads. Confirm no second training or generation dispatch occurred.
14. Read the same run through `GET /api/projects/{project_id}/agent-runs/{run_id}/inspection` and correlate Molly with OTel and actual LLM calls in LangSmith. No extra LLM call is required for BR1.

## Evidence review

Private evidence adds Uni-Mol provider/version, verified training publication, REINVENT4 version/config digest, execution classification, generation/prediction/ranking bindings, OTel correlation and LangSmith correlation only where an LLM was actually invoked. Export only privacy-redacted digests, status classes and safe IDs.

Do not mark `M3H-013 I/T/V / DONE` until fresh Uni-Mol, real REINVENT4, current-run prediction/ranking/validation, restart/replay evidence and repository-owner exact-HEAD review all exist. BR1 alone never closes M3H-GATE-006, M3.5 or Molly v1.

## 2026-08-03 authoritative remote applicability baseline

The first post-merge Uni-Mol applicability preflight was run remotely against
the private 1,999-row input and failed closed. Its reviewed repository commit
was `19ae10a1c866964d6b993c7dfd100e930127acde`; the provider was
`unimol-tools 0.1.5` under `unimol-train-br1-v2`. The private report contract
and deterministic summary projection verified, but the result was:

```text
status=BLOCKED
input=1999  supported=0  unsupported=0  unresolved=1999
report_digest=sha256:f47b6a50d4fa73f6bfcf2384918e484fc2a46d38c68a93b24d1ea1a66c7f17db
reasons=INPUT_DIGEST_MISMATCH,SOURCE_AUTHORITY_INVALID,
        MAPPING_POLICY_INVALID,PROVIDER_PREFLIGHT_API_UNAVAILABLE,
        PROVIDER_CAPABILITY_UNAVAILABLE
```

`SUPPORTED=0` is not a claim that all molecules are unsupported. Every row
was `UNRESOLVED` because the input/authority chain and provider applicability
capability were not closed. No acceptance ID/run, Controller execution,
training, generation, prediction, ranking, restart, or replay was attempted.

## BR1 preflight remediation v1

The remediation implementation keeps applicability evidence outside the
runtime task graph and uses this explicit identity chain:

```text
authorized source artifact
  -> canonical source reconstruction
  -> server-owned mapping binding
  -> canonical provider-input bytes
  -> remote staged-input digest
  -> provider actual-input digest
```

`br1_preflight_source_authority.v1` binds the source publication registry,
source and mapping digests, row count, canonicalization contract, mapping
binding, provider/version, profile/digest, and canonical source/provider
digests. `br1_unimol_provider_adapter.v1` describes the exact provider
representation, fields, context, missing-value, duplicate, ordering and
no-dispatch rules. The private report records expected/observed/staged/actual
input digests and dispatch assertions; the public summary is still generated
only by exact projection from that report.

The fresh remote run after this remediation must use the reviewed commit and
matching worker, regenerate the authority artifacts from the private source,
verify the actual staged bytes, and then run the report and summary verifiers.
If any authority or provider capability remains incomplete, the result stays
`BLOCKED`; this PR cannot create an acceptance ID/run or launch training,
generation, prediction, or ranking.

## Applicability preflight handoff

This preflight is operator evidence before acceptance, not a second runtime
authority. A development-branch real-environment run can only be called a
smoke test. After this PR is merged, repeat the preflight on the reviewed
commit with the matching worker before freezing data or creating an acceptance
ID/run ID.

- `PASS`: freeze the final Raw CSV, source manifest and mapping policy, then create a new clean BR1 acceptance.
- `REVIEW_REQUIRED`: provide counts and reason summary to the owner and wait for an explicit exclusion decision.
- `BLOCKED`: repair provider/environment authority or select a supported data range; do not start acceptance.

## Development-branch evidence boundary

Real applicability preflight on the remediation HEAD: `NOT EXECUTED` until
the fresh remote run is performed. The previous remote result above is the
only authoritative baseline. A local fake-provider test or a development
branch run is not a private applicability result and cannot be used to freeze
the data or start acceptance.
