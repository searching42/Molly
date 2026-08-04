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

The fresh remote execution of remediation HEAD
`c23b7a0eb8897b0d65bd33b4763eb8f8815f46f7` used the matching worker,
`unimol-tools 0.1.5`, and `unimol-train-br1-v2`. The provider capability
contract was discovered and verified, but the private staging still had no
new source authority/publication registry artifacts, so the result remained:

```text
status=BLOCKED
input=1999  supported=0  unsupported=0  unresolved=1999
report_digest=sha256:e86467ab217f99d8a755b963d1bb46adcb5ddfdb7d736f4819ae818d72e6901e
summary_digest=sha256:57443e3f4c5767ef579af931299d029eb0d69b5ce948b073eb3af785cc17da86
reasons=INPUT_DIGEST_MISMATCH,MAPPING_POLICY_INVALID,
        SOURCE_AUTHORITY_INVALID,SOURCE_PUBLICATION_REGISTRY_INVALID
```

The report contract and report-bound summary projection both passed. The
report recorded `provider_capability_probe_dispatched=true`, while
`provider_preprocessing_dispatched`, training, generation, prediction,
ranking, model artifact, scaler and metrics assertions were all false because
authority validation stopped the run before row-level provider preprocessing.
This is fresh pre-acceptance evidence, not a claim of applicability or owner
acceptance; a new authority-bound run is still required.

## BR1 source authority materialization v1

The remediation writer now materializes a fresh private authority chain from
the actual Raw CSV and the two private legacy provenance inputs. It is a
deterministic operator-side writer, not a runtime task or acceptance writer.
It rereads stable regular-file bytes, validates the exact Raw Dataset columns,
row identity and target contract, computes the physical Raw CSV digest, and
writes immutable `source_dataset_manifest.v1`,
`br1_raw_dataset_mapping_policy.v1`, structured publication, publication
registry, and `br1_preflight_source_authority.v1` artifacts. The writer
computes every digest itself and refuses to overwrite a different existing
artifact.

The source manifest's materialization binding and the private publication,
registry, and authority bind the source artifact identity, source kind, row
count, physical column roster, mapping identity, publication identity,
canonicalization contract, provider/version, profile digest, repository
commit, and worker implementation digest. The mapping binding separately
binds the exact molecule, target, row identity, condition/context, missing,
filter, duplicate and canonical-order rules. Physical Raw CSV order remains
part of the Raw Dataset digest; canonical source/provider digests use the
fixed `row_id` order.

The mapping policy has two explicit semantic layers. `source_to_raw_mapping`
contains the legacy source field origins, fixed values and conversion rules;
its `row_comparable_value` is the literal required in every Raw observation.
The dataset-level `comparability_policy` remains the broader scientific scope
and is not a row literal. `raw_to_provider_mapping_binding` contains the
provider-facing identity, required fields, missing-value, duplicate and
canonical-order contract. The existing `mapping_binding` fields are retained
only as an exact compatibility projection and must equal the provider-facing
binding byte-for-semantic-material. The preflight records privacy-safe,
counted mapping diagnostics such as
`ROW_COMPARABLE_VALUE_MISMATCH`, condition/solvent/temperature mismatches,
role/mechanism mismatches, duplicate Standard InChIKeys and
`SOURCE_TO_RAW_MAPPING_MISMATCH`; it never emits row IDs or molecule values in
the public summary.

In the configured remote private environment, create a fresh
permission-restricted staging directory and
run the materializer with private paths and the exact reviewed implementation
identities:

```bash
python scripts/materialize_br1_preflight_authority.py \
  --raw-dataset <private-raw-csv> \
  --source-manifest-input <private-source-manifest> \
  --mapping-policy-input <private-mapping-policy> \
  --output-source-manifest <fresh-private-source-manifest> \
  --output-mapping-policy <fresh-private-mapping-policy> \
  --output-source-publication <fresh-private-source-publication> \
  --output-registry <fresh-private-source-publication-registry> \
  --output-authority <fresh-private-source-authority> \
  --expected-provider-version 0.1.5 \
  --execution-profile-id unimol-train-br1-v2 \
  --execution-profile-digest <reviewed-profile-digest> \
  --repository-commit <reviewed-commit> \
  --worker-implementation-digest <matching-worker-digest> \
  --publication-identity <fresh-private-publication-id> \
  --registry-id <fresh-private-registry-id>
```

Immediately run the applicability preflight against those newly materialized
artifacts. Verify the private report contract and then verify the public
summary by deterministic exact projection from that report. A mapping-policy
or source-row semantic mismatch remains `BLOCKED`; the materializer never
rewrites the Raw CSV, silently changes comparability, filters rows, or turns a
legacy marker into an implicit alias.

## 2026-08-04 initial source authority and applicability evidence

The initial fresh operator run used PR execution code HEAD
`a81e50505b44840f49400427663644f01c801023` in a new restricted staging
directory, the `unimol-tools 0.1.5` provider, the
`unimol-train-br1-v2` profile, and the matching worker implementation digest
`sha256:e0b9f18a20fb31dc9fd2cde178f8771832e11408c74d16e3b88ab217ed0f397e`.
The materializer succeeded and verified a 1,999-row chain. Its privacy-safe
identity evidence was:

```text
raw_dataset_digest=sha256:755c8bb312c25deffb7bba4a77904e8337646959ecc802575444b2620f848efa
source_manifest_digest=sha256:a2b54d72acaafd565e09fb2dc93c344507c8524186bb8c317572140bcdabeccf
mapping_policy_digest=sha256:68b019534aaa08c93c6c82c8fa25cf5f23ca98d9b0267623068a2f3c3d35a814
canonical_source_dataset_digest=sha256:817d936c343fd63edc50acc85472a2c407df9c60d9d34884bc7f2228b8aab85c
canonical_provider_input_digest=sha256:d8770caf126c68c5b81788d256b985e7d72d60b6bfcbc7b82ca3fc790d8e2da5
source_materialization_binding_digest=sha256:b3ecf2e46167b5ba467c5730ff77149cd0dbbe4e0e64902a5940a24580716ff1
source_publication_digest=sha256:b3e21eb56f372c7842a854391a42786b04ea36feb55f5dc798fda489ba33ee14
registry_digest=sha256:3a24307efcd81d97996713718354a4410837e1baa09c4bbd5edf2f7cef7bfb87
authority_digest=sha256:ae482586d4c13492a371c0b1f1ba1cff1be99ea8eba0a7085d2db938b12b96a1
```

The fresh preflight result was:

```text
status=BLOCKED
input=1999  supported=0  unsupported=0  unresolved=1999
provider=unimol-tools  provider_version=0.1.5
execution_profile=unimol-train-br1-v2
report_digest=sha256:a4a39f95afefd4e9f9dd176223c64b8174ac663f00e04cf70066706d64bbc241
summary_semantic_digest=sha256:2f074b82dbf4ee36e21ebb7b92375b6fd55b5600ffea2f254af6c320b94962d4
reason_counts=MAPPING_POLICY_INVALID:1999
```

The report contract verifier passed and the summary was a canonical exact
projection of that report. A second run with the same inputs, authority,
provider/profile, and frozen timestamp produced byte-identical report and
summary. Capability discovery ran, but provider preprocessing was not
dispatched because the exact mapping contract did not match the source rows;
training, generation, prediction, ranking, model, checkpoint, scaler, and
metrics assertions were all false. The staging directory remained restricted
(`0700`, files `0400/0600`, no symlinks). This is a fresh blocked
pre-acceptance result, not an applicability acceptance or data freeze.

## 2026-08-04 final executable HEAD fresh authority and applicability evidence

After the exact mapping-boundary regression fix, the fresh operator run used
execution code HEAD
`69feba9611635a20411775686e337a17088078ee` in a new restricted staging
directory, the `unimol-tools 0.1.5` provider, the
`unimol-train-br1-v2` profile, and the matching worker implementation digest
`sha256:e0b9f18a20fb31dc9fd2cde178f8771832e11408c74d16e3b88ab217ed0f397e`.
The materializer succeeded and verified a 1,999-row chain. Its privacy-safe
identity evidence was:

```text
raw_dataset_digest=sha256:755c8bb312c25deffb7bba4a77904e8337646959ecc802575444b2620f848efa
source_manifest_digest=sha256:3427295d2ee603b501adc343b3a61484b15bb9c7836fa1ee454665014282d11c
mapping_policy_digest=sha256:68b019534aaa08c93c6c82c8fa25cf5f23ca98d9b0267623068a2f3c3d35a814
canonical_source_dataset_digest=sha256:817d936c343fd63edc50acc85472a2c407df9c60d9d34884bc7f2228b8aab85c
canonical_provider_input_digest=sha256:d8770caf126c68c5b81788d256b985e7d72d60b6bfcbc7b82ca3fc790d8e2da5
source_materialization_binding_digest=sha256:af30c28742e47f0ab4eadc28c22849b80af3f7ae702c38177148c9cc6891eb88
source_publication_digest=sha256:84a97fa169870e453f2ab47f475690b2a8c87e68b4a00452a8439020788ee116
registry_digest=sha256:4e5217e6120b792da25d886cd50badde93ad73e69706e311f451ff7133f8b48a
authority_digest=sha256:e1457a8b489ece721241e76e5d56d429671b7f79ef734b288d923cfd9e7e5090
```

The final executable HEAD preflight result was:

```text
status=BLOCKED
input=1999  supported=0  unsupported=0  unresolved=1999
provider=unimol-tools  provider_version=0.1.5
execution_profile=unimol-train-br1-v2
report_digest=sha256:f14ed01da67d06dfc421b6e92d57cb70fb89b39398f506cf0aa7f12ee67dcab3
summary_semantic_digest=sha256:1eddb45ac9ef4ba67c6e65c11ca21c433d050c39defbd86b28ab06534a2a7c0a
reason_counts=MAPPING_POLICY_INVALID:1999
```

The report contract verifier passed and the summary was a canonical exact
projection of that report. A second run with the same inputs, authority,
provider/profile, and frozen timestamp produced byte-identical report and
summary. Capability discovery ran, but provider preprocessing was not
dispatched because the exact mapping contract did not match the source rows;
training, generation, prediction, ranking, model, checkpoint, scaler, and
metrics assertions were all false. The staging directory remained restricted
(`0700`, files `0400`, no symlinks, nine private artifacts including replay
copies). This is a fresh blocked pre-acceptance result, not an applicability
acceptance or data freeze. The later PR tip is docs-only and does not change
this executable evidence.

## 2026-08-04 mapping-semantic remediation fresh PASS

The mapping-semantic fix was deployed and executed on executable HEAD
`ee4962aa511e8bd4edfe8e4867818f3881d797fa` with the matching worker,
`unimol-tools 0.1.5`, and profile `unimol-train-br1-v2`. This is the first
remote evidence in this runbook that covers the corrected two-layer mapping
contract. The older `c23b7a0...`, `a81e505...`, and `69feba9...` runs remain
historical evidence for the executable HEADs on which they ran; they do not
cover this HEAD.

The fresh restricted staging contained these private semantic artifacts:

- `source_dataset_manifest.v1`
- `br1_raw_dataset_mapping_policy.v1`
- source publication
- source publication registry
- `br1_preflight_source_authority.v1`
- `br1_unimol_applicability_report.v1`
- `br1_unimol_applicability_summary.v1`

The Raw CSV and source provenance inputs remained private inputs and were not
committed or exported. The materializer reread stable regular-file bytes,
computed all digests from the actual inputs, refused replacement/overwrite of
different artifacts, and immediately re-verified the complete chain. The
privacy-safe materialization evidence was:

```text
input_row_count=1999
raw_dataset_digest=sha256:755c8bb312c25deffb7bba4a77904e8337646959ecc802575444b2620f848efa
source_manifest_digest=sha256:8c3958cfc0c5477a49db40ee689607a4b441235ffcfda82ce15bdb06898aef71
mapping_policy_digest=sha256:0f58c28e37f436c77fd167577cdc88e0966fcd816e5afa1e0b94e314b63a6912
canonical_source_dataset_digest=sha256:817d936c343fd63edc50acc85472a2c407df9c60d9d34884bc7f2228b8aab85c
canonical_provider_input_digest=sha256:d8770caf126c68c5b81788d256b985e7d72d60b6bfcbc7b82ca3fc790d8e2da5
source_materialization_binding_digest=sha256:99bdb75c8333183d4fa9331bf31f2f98c564bc0dab8a2a2a69d3818cde02751b
source_publication_digest=sha256:d99a0a973813012b29a7e56384d198afdfb16102c7bb9d20d23d94bd26b60de2
registry_digest=sha256:c94eee683acd81e5a7ea7d0aeecdcdb5cdada73adb11ce2d25fed4a61ce636ef
authority_digest=sha256:9971c2fdd55a09b8320218d0f928298ea1af99fe7f97cbd9d3501e732a5f3555
```

The source-to-Raw layer now binds the legacy source field origins and the
literal `row_comparable_value=true_within_frozen_single_solvent_scope`; the
dataset-level `comparability_policy=partially_comparable_single_solvent`
remains a scientific scope, not a row literal. The Raw-to-provider layer
independently binds `smiles`, `target_value`, `row_id`, condition/context,
missing-value, duplicate, and canonical row-order rules. Mapping diagnostics
are counted and privacy-safe; the fresh run produced no mapping diagnostic
reasons.

The fresh private preflight result was:

```text
status=PASS
provider=unimol-tools  provider_version=0.1.5
expected_provider_version=0.1.5
execution_profile=unimol-train-br1-v2
repository_commit=ee4962aa511e8bd4edfe8e4867818f3881d797fa
input=1999  supported=1999  unsupported=0  unresolved=0
reason_counts={}
mapping_diagnostics={}
report_digest=sha256:c49d61f7f74b14bcdf77c27c307c60859e548f193ba803071f02081a82a44751
summary_semantic_digest=sha256:4bdab1a196f18d030efbf62c7a9c3a03baa98c6d184583fe4135a498bc4da9ba
```

The report contract verifier passed, and the public summary was rebuilt from
that private report and passed canonical exact comparison. Authority
verification passed. The expected Raw digest equaled the observed Raw digest;
the expected canonical provider-input digest equaled the staged and provider
actual input digests. Capability discovery and provider preprocessing were
dispatched. Training, generation, prediction, ranking, model artifacts,
scaler, and training metrics were not dispatched or created. The private
staging directory was mode `0700` with no forbidden artifacts.

This `PASS` closes the applicability preflight engineering blocker for this
reviewed executable HEAD. It does not freeze the final BR1 dataset, create an
acceptance ID/run, or constitute BR1 applicability owner acceptance. An owner
must independently review the privacy-safe evidence before the final Raw CSV,
source manifest, and mapping policy are frozen and a clean BR1 acceptance is
created.

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

The latest remote applicability preflight was executed on the reviewed
executable HEAD `ee4962aa511e8bd4edfe8e4867818f3881d797fa` and returned
`PASS` as recorded above. This closes the source/mapping/provider applicability
preflight authority for that HEAD, but it is not owner acceptance, data
freeze, or a BR1 acceptance run. Historical blocked evidence remains
immutable and is not evidence for this HEAD. A local fake-provider test is
not a private applicability result.
