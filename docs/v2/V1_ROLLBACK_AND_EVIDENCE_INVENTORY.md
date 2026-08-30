# Molly v1 rollback and evidence inventory

Status: `FROZEN_REFERENCE`

Recorded: 2026-08-30

This document records the immutable v1 reference created before Core v2
readiness closure. It is a rollback and evidence index, not a Core v2
implementation or cutover approval.

## Immutable references

| Item | Value |
| --- | --- |
| Repository | `https://github.com/searching42/Molly.git` |
| Freeze commit | `ae7892dbf8a6bfe85dd909056eadc2afecc40d9d` |
| Freeze tag | `molly-v1-pre-core-v2-20260829` |
| Legacy branch | `legacy/molly-v1` |
| Freeze meaning | v1 pre-Core-v2 implementation baseline |

The freeze tag and legacy branch both point to the freeze commit. The tag is
annotated and must never be force-updated or moved. The legacy branch is a
rollback line and must not be deleted before a separately approved cutover.

## Creation and verification

The target commit was independently confirmed as the exact `main` tip before
creation. Local verification peeled the tag and resolved the legacy branch to
the same commit. The refs were then published with an ordinary non-force push;
the remote creation result reported one new branch and one new tag. A final
remote ref audit is recorded in the accompanying readiness-closure report.

## Rollback commands

Use a fresh checkout or worktree so the active branch is not overwritten:

```text
git fetch --no-tags origin refs/heads/legacy/molly-v1
git worktree add <V1_WORKTREE> legacy/molly-v1
```

The immutable tag can be used when the exact frozen commit is required:

```text
git fetch --tags origin
git worktree add <V1_FREEZE_WORKTREE> molly-v1-pre-core-v2-20260829
```

These commands recover the v1 code without importing, installing, or relying
on any Core v2 package. They do not change `main`, the tag, or the legacy
branch.

## BR1 v1 acceptance evidence

The checked-in, privacy-safe BR1 evidence remains under:

- `docs/evidence/br1-conversation-real-acceptance-v1/README.md`
- `docs/evidence/br1-conversation-real-acceptance-v1/acceptance_manifest.json`
- `docs/evidence/br1-conversation-real-acceptance-v1/result_summary.json`
- `docs/evidence/br1-conversation-real-acceptance-v1/restart_replay_summary.json`

The acceptance manifest records `verification_status: runtime_verified` and
the v1 control-plane commit. Its scientific claim boundary is limited to a
verified deterministic Computational Top-N projection of model-predicted
values; it is not experimental validation or a guarantee.

## BR2 and related evidence inventory

The current repository also contains privacy-safe BR2 implementation/runtime
evidence at:

- `docs/evidence/br2-real-mineru-runtime-v1/README.md`
- `docs/evidence/br2-real-mineru-runtime-v1/acceptance_manifest.json`
- `docs/evidence/br2-real-mineru-runtime-v1/runtime_summary.json`
- `docs/evidence/br2-contextual-mapping-v1/README.md`
- `docs/evidence/br2-contextual-mapping-v1/acceptance_manifest.json`
- `docs/evidence/br2-contextual-mapping-v1/mapping_summary.json`

These records do not close the separate fresh conversation acceptance boundary,
do not constitute Core v2 parity, and do not authorize default cutover.

## Current validation and limitations

The pre-closure branch validation included compile/diff checks, the v1
acceptance test, documentation/path-hygiene tests, and the repository PR Fast
selection. The current PR #64 closure run is recorded in the final readiness
report and remains Draft.

Known limitations are explicit:

- the v1 BR1 evidence covers a reviewed representative runtime, not every
  possible task or failure mode;
- private raw inputs, model weights, worker output, and infrastructure
  identities are intentionally not in Git;
- no fresh-real v2 BR1 parity, remote restart canary for v2, or default cutover
  was performed in this readiness Goal;
- B2, B3, and B4 therefore remain `PENDING`.
