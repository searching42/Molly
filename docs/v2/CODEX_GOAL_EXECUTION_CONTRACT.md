# Molly Core v2 Codex Goal execution contract

Status: `PREPARATION_ONLY`

Date recorded: 2026-08-29 (Asia/Shanghai)

This document records the repository-local execution contract for the uploaded
Molly Core v2 Goal-mode launcher. It is an operational guardrail, not Owner
approval and not an implementation authorization.

## Authoritative inputs

The launcher is the user-provided `<USER_PROVIDED_LAUNCHER>`. Before any
production implementation, the runner must verify these repository files at
the exact implementation HEAD:

- `docs/v2/MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md`
  SHA-256 `0f6c8a0e0c7ef6d1fc19b7c73ed9375f6cc6304f463f42e7bc6175ae6e0a55c7`;
- `docs/v2/MOLLY_CORE_MODULE_DISPOSITION_MATRIX_V1_1_BR1_HARDENED.csv`
  SHA-256 `2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d`;
- `docs/v2/readiness/core_refactor_readiness.json`;
- `docs/v2/MOLLY_REFACTOR_SPLIT_README.md`;
- `docs/roadmap.md`;
- the checked-in v1 BR1 acceptance README and acceptance manifest.

Any missing file, digest mismatch, freeze/tag/legacy-branch inconsistency, or
spec conflict is a preflight blocker. The runner must stop production code
changes and record the blocker in `docs/v2/reports/CORE-00.md`.

## Readiness behavior

The readiness manifest is the only machine-readable gate:

```text
core_goal_mode_ready = true  iff C0-C7 are all PASS
core_cutover_ready   = true  iff C0-C7 and B0-B4 are all PASS
                         and Owner explicitly approves cutover
```

When `core_goal_mode_ready=false`, the runner may perform CORE-00 and safe,
source-backed C0-C7 preparation such as audits, inventories, contract
documentation, and evidence bookkeeping. It must not create or modify
production Core v2 runtime code, and it must not start CORE-01 through CORE-07.
When `core_goal_mode_ready=true`, CORE-01 through CORE-07 are implemented as
independent milestone commits with targeted tests and reports. CORE-08 remains
forbidden until B0-B4 and explicit Owner approval are recorded.

## Core boundaries

Core v2 has one execution path:

```text
RunRequest -> AgentLoop / RunEngine -> ToolRegistry + ToolPolicy
-> scientific tool -> ArtifactStore + RunLedger + ArtifactLineage
-> ValidationResult / ReviewRecord
```

The v2 package must not import `ai4s_agent`, recreate the v1 Controller or
authority graph, or expose credentials, arbitrary hosts, paths, URLs, shell, or
SSH to a model. MinerU, BR1, remote compute, OTel, and LangSmith remain
optional plugins or observer-only integrations. The error-propagation proposal
is non-binding and is not authorized for implementation.

## Required evidence and Git discipline

- Preserve the v1 freeze/tag/legacy rollback line; never force-move a freeze
  tag or delete legacy BR1 before cutover gates pass.
- Keep one report under `docs/v2/reports/` per milestone and retain
  independent, non-squashed milestone commits.
- Run the repository's real compile, diff, targeted, and relevant regression
  commands; do not claim fresh-real BR1, remote, GPU, private-data, or
  experimental validation without inspectable evidence.
- A Draft PR may be opened only for the actual repository and exact reviewed
  branch, with its body stating the readiness state and any blockers. Merge is
  outside this contract.

## Current recorded state

The current preparation evidence is
`docs/v2/reports/CORE-00.md`, recorded from HEAD
`ae7892dbf8a6bfe85dd909056eadc2afecc40d9d`. It found matching authoritative
digests, but readiness remains `core_goal_mode_ready=false`, all C0-C7 and B0-B4
remain `PENDING`, and the v1 freeze metadata is absent. This document therefore
does not authorize production implementation.
