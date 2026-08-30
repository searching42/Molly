# Core v2 scope approval

Decision status: `APPROVED`

Decision date: 2026-08-30

This repository decision records the Owner approval supplied with the Core v2
readiness-closure Goal. It approves the research-direction-neutral
simplification scope and the evidence work needed before production
implementation. It does not approve default cutover or legacy deletion.

```yaml
decision: APPROVED
scope: Molly Core v2 simplification refactor
core_spec_path: docs/v2/MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md
core_spec_sha256: 0f6c8a0e0c7ef6d1fc19b7c73ed9375f6cc6304f463f42e7bc6175ae6e0a55c7
matrix_digest_at_owner_approval: 2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d
matrix_digest_after_C2_reconciliation: d26366996db3df2783b3c0fcc8b03981902c2400c1dd6128d436fcdfb2d4fca4
error_propagation_required: false
error_propagation_implementation_authorized: false
production_implementation_authorized_after_C0_C7: true
default_cutover_authorized: false
legacy_deletion_authorized: false
```

The C2 matrix update, if any, is limited to factual path and inventory
reconciliation under this approved architectural disposition. It is not a new
architectural scope decision. The reconciled matrix digest is recorded here
The reconciled matrix digest is
`d26366996db3df2783b3c0fcc8b03981902c2400c1dd6128d436fcdfb2d4fca4`.
It is synchronized into the readiness manifest and final execution contract.

The approval does not authorize CORE-08, fresh-real BR1 claims, remote/GPU
acceptance claims without evidence, experimental scientific claims, or the
error-propagation research runtime.
