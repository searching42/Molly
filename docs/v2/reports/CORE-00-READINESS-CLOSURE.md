# Molly Core v2 CORE-00 readiness closure

Status: `SUCCESS`

Date: 2026-08-30

Repository: `searching42/Molly`

Reviewed branch: `codex/molly-core-v2-launcher-1-2`

Draft PR: `#64`

This closure completes the evidence-backed readiness gates C0-C7. It does not
implement production Core v2, start CORE-01 through CORE-08, delete legacy v1,
or authorize default cutover.

## Frozen baseline

```yaml
main_and_v1_freeze_commit: ae7892dbf8a6bfe85dd909056eadc2afecc40d9d
freeze_tag: molly-v1-pre-core-v2-20260829
legacy_branch: legacy/molly-v1
branch_head_at_C2_audit: bc379401fd4d88efc714fb878628cb29e49765d2
matrix_digest_at_owner_approval: 2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d
matrix_digest_after_C2_reconciliation: d26366996db3df2783b3c0fcc8b03981902c2400c1dd6128d436fcdfb2d4fca4
```

The freeze tag and legacy branch were verified locally and remotely before
this closure. Both resolve exactly to the v1 freeze commit. The tag was
created once and was not moved.

## Gate results

| Gate | Status | Evidence | Commit | Remaining risk |
| --- | --- | --- | --- | --- |
| C0 | `PASS` | [`CORE_V2_SCOPE_APPROVAL.md`](../decisions/CORE_V2_SCOPE_APPROVAL.md); spec SHA-256 `0f6c8a0e0c7ef6d1fc19b7c73ed9375f6cc6304f463f42e7bc6175ae6e0a55c7`; matrix approval SHA-256 `2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d`. | `e29705a32b3115240b0d03ff97b3a7f48eefb82a` | Approval covers Core simplification only; error-propagation implementation, default cutover, and legacy deletion remain unauthorized. |
| C1 | `PASS` | [`V1_ROLLBACK_AND_EVIDENCE_INVENTORY.md`](../V1_ROLLBACK_AND_EVIDENCE_INVENTORY.md); freeze commit/tag/branch and remote-ref verification; v1 BR1 and BR2 evidence paths. | `e29705a32b3115240b0d03ff97b3a7f48eefb82a` | v1 evidence retains its documented legacy limitations; no v2 real-run claim is implied. |
| C2 | `PASS` | [`V1_FILE_DISPOSITION_INVENTORY.csv`](../audit/V1_FILE_DISPOSITION_INVENTORY.csv), 453/453 tracked implementation files, 0 unresolved dispositions; inventory SHA-256 `6dc12b0a6d430e9fe6a31c38c3bfde2a12443dbe0dbdafb760859bd596ab83b3`; [`C2_FILE_DISPOSITION_AUDIT.md`](../audit/C2_FILE_DISPOSITION_AUDIT.md). | `e29705a32b3115240b0d03ff97b3a7f48eefb82a` | Dispositions are migration plans, not production implementation; each later milestone must re-check imports and invariants. |
| C3 | `PASS` | [`ACQUISITION_SECURITY_AND_PROVENANCE.md`](../contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md), SHA-256 `69fdb6f100e6b06a2a6d2998e521babaebc96d9f1e69ea0bb67d793801d019d6`; contract/path hygiene tests. | `364c910b912aeafe1f385abc7e8fee8cc34befd0` | No production fetcher or network-live licensing/access validation was performed. |
| C4 | `PASS` | [`C4_CORE_CONTRACT_SPIKE.md`](C4_CORE_CONTRACT_SPIKE.md), SHA-256 `554c4510d951997ed11db2faacce3286efe24812ee7ee345eec1d366e4578efa`; `tests/test_core_v2_contract_spike.py`: 13 passed. | `364c910b912aeafe1f385abc7e8fee8cc34befd0` | The spike is local, single-process, standard-library-only, and not a production persistence or execution engine. |
| C5 | `PASS` | [`PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md`](../contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md), SHA-256 `d1d73b8e495dc8d36bc6ef7609a10dd36d75c2d472c34a110ef8c460f5b059cf`; repaired `uv.lock` SHA-256 `f204dc52afd4d2b50e58651e75bb75a8f6fa0a0192d9f17e03c79891122b30c4`; `UV_CACHE_DIR=<LOCAL_UV_CACHE> uv lock --check`: pass. | `364c910b912aeafe1f385abc7e8fee8cc34befd0` | The current legacy `pyproject.toml` remains unchanged; future dependency separation is conceptual until later milestones. |
| C6 | `PASS` | [`literature_fixture_manifest.json`](../fixtures/literature_fixture_manifest.json), [`oled_gold_fixture.json`](../fixtures/oled_gold_fixture.json), [`br1_parity_manifest.json`](../fixtures/br1_parity_manifest.json); `tests/test_core_v2_fixture_manifests.py`: 3 passed. | `364c910b912aeafe1f385abc7e8fee8cc34befd0` | Fixtures are synthetic/contract-only; no fresh-real BR1, GPU, remote, or network-live acceptance was performed. |
| C7 | `PASS` | [`CODEX_GOAL_EXECUTION_CONTRACT.md`](../CODEX_GOAL_EXECUTION_CONTRACT.md), SHA-256 `e1af801f7ea2636af8fb9521106b8f8d4e1320e606771e15d1d82f1126793bd9`; this closure report; final readiness manifest. | C7 closure commit containing this report | Readiness authorizes later CORE-01 through CORE-07 Goals, not this Goal and not CORE-08. |

## Verification record

The focused readiness batch completed:

```text
git diff --check                         PASS
python -m compileall -q src tests prototypes  PASS
tests/test_core_v2_contract_spike.py       13 passed
tests/test_core_v2_fixture_manifests.py     3 passed
tests/test_core_v2_readiness_contracts.py   4 passed
```

The existing v1 acceptance and repository documentation regression selections
were also run as targeted checks and passed. The repository PR Fast selection
is the final pre-push check for this coherent batch; its result is recorded in
the final Git/PR verification below.

No production Core v2 directory exists under `src/molly/`,
`plugins/br1_inverse_design/`, or `plugins/remote_compute/` on this branch.

The public documentation scan found no absolute user-home paths, credentials,
tokens, private endpoints, or local agent-context paths in the changed CORE-00
authority documents. The documents use portable placeholders where an
execution-context label is necessary.

## B0-B4 state

| Gate | Status | Evidence or required future evidence |
| --- | --- | --- |
| B0 | `PASS` | Immutable v1 freeze and rollback/evidence inventory are inspectable. |
| B1 | `PASS` (contract-only) | BR1 parity stages, invariants, offline runner contract, and claim boundary are frozen. |
| B2 | `PENDING` | Requires fresh Uni-Mol training, real REINVENT4 generation, current-run prediction/evaluation, and exact scientific artifacts. |
| B3 | `PENDING` | Requires a v2 remote-restart canary with durable/idempotent/credential-safe evidence. |
| B4 | `PENDING` | Requires explicit future Owner approval for default cutover. |

## Final decision

```json
{
  "C0": "PASS",
  "C1": "PASS",
  "C2": "PASS",
  "C3": "PASS",
  "C4": "PASS",
  "C5": "PASS",
  "C6": "PASS",
  "C7": "PASS",
  "core_goal_mode_ready": true,
  "core_cutover_ready": false,
  "owner_decision": "APPROVED_FOR_CORE_IMPLEMENTATION_NOT_CUTOVER"
}
```

The next production implementation run must be a separate Owner-reviewed
Goal. CORE-08 remains blocked until B0-B4 are all PASS and the Owner provides
explicit cutover approval.
