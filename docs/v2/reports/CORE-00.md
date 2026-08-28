# CORE-00 — Molly Core v2 preflight and exact-HEAD audit

Status: `BLOCKED_PRE_IMPLEMENTATION`

Date: 2026-08-29 (Asia/Shanghai)

This report records the read-only preflight required by the Core v2 Goal-mode
launcher. It does not authorize production implementation, v1 cutover, legacy
deletion, or BR1 parity claims.

## Baseline and instruction chain

- Repository: `searching42/Molly`
- Worktree: `/Users/benton/Documents/ChatGPT/Molly v2/core-v2-launcher-1-2`
- Branch: `codex/molly-core-v2-launcher-1-2`
- Exact HEAD: `ae7892dbf8a6bfe85dd909056eadc2afecc40d9d`
- Candidate audit baseline from readiness: `4352f137db3976cff31bf6cb30f543caa38f8013`
- HEAD is based on the candidate plus the merged Core v2 preflight docs; no v1
  freeze commit, tag, or legacy branch is recorded.

Instruction and project guidance loaded for this audit:

- user-provided `AGENTS.md` validation policy in the task context;
- `/Users/benton/openclaw-docker/workspace/AGENTS.md`;
- `/Users/benton/openclaw-docker/workspace/agent/CLAUDE.md`;
- `/Users/benton/.codex/skills/ai4s-project-progress/SKILL.md`;
- `/Users/benton/openclaw-docker/workspace/SOUL.md`;
- the uploaded Goal-mode launcher.

`USER.md`, `MEMORY.md`, and the current/previous daily memory files were not
present under `/Users/benton/openclaw-docker/workspace`.

## Authority and digest verification

The required v2 files are present on the exact HEAD. Their SHA-256 digests are:

| File | Actual SHA-256 | Required/declared SHA-256 | Result |
| --- | --- | --- | --- |
| `docs/v2/MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md` | `0f6c8a0e0c7ef6d1fc19b7c73ed9375f6cc6304f463f42e7bc6175ae6e0a55c7` | launcher/readiness: same | PASS |
| `docs/v2/MOLLY_CORE_MODULE_DISPOSITION_MATRIX_V1_1_BR1_HARDENED.csv` | `2c2eb52e902cdfeac01fa8ac05c6872f4782a0bc8c4d02937bc1338cfef80e3d` | launcher/readiness: same | PASS |
| `docs/v2/readiness/core_refactor_readiness.json` | `729c13dc041a7ec60f579d7fea94b6a87fa550a48eca3dd6d0a253c25b83e254` | self-contained readiness manifest | PASS |
| `docs/v2/MOLLY_REFACTOR_SPLIT_README.md` | `92c92cf172c11a57815aecfda5a29795163553f6f21c22e468d87ddf6ff2783b` | required file | PASS |
| `docs/roadmap.md` | `06ca47cefd64a1ba7bad210b7d5bf324de5bfb92044a8c384e8b13ce4956194b` | required file | PASS |
| `docs/evidence/br1-conversation-real-acceptance-v1/README.md` | `784976fdc739c266c1af47570e943ddd8b59a0e1341d78fb19ccc0fe273874c4` | required file | PASS |
| `docs/evidence/br1-conversation-real-acceptance-v1/acceptance_manifest.json` | `fdbad5cccbd096bba390a5926671ecdab864d50877c24cf1d3eb66acd9c38bd2` | required file | PASS |

The non-binding error-propagation proposal is present only as research context;
its implementation authorization remains `false` and no such runtime was
started.

## Readiness gate

The exact readiness manifest reports:

```text
C0-C7: PENDING
B0-B4: PENDING
v1_freeze_commit: null
v1_freeze_tag: null
legacy_branch: null
core_goal_mode_ready: false
core_cutover_ready: false
owner_decision: PENDING_FINAL_READINESS
```

The spec is still `OWNER_REVIEWED_DRAFT_V1_2`, not an in-repository final
`APPROVED` state. Therefore the launcher permits CORE-00 and safe C0-C7
prework only; it does not permit CORE-01 through CORE-07 production
implementation and does not permit CORE-08.

## Exact-HEAD module audit

The 50 matrix rows were parsed at HEAD
`ae7892dbf8a6bfe85dd909056eadc2afecc40d9d`. Repository-like references were
checked against the repository root, `src/ai4s_agent`, and the checked-in
`docs/evidence` location where the matrix uses a legacy-relative path.

- 35 literal path references were identified;
- 32 literal references resolve at HEAD;
- 3 literal references remain unresolved and require disposition clarification:
  - `src/ai4s_agent/artifact_registry.py` (a related file exists under
    `src/ai4s_agent/provenance/artifact_registry.py`);
  - `src/ai4s_agent/controller.py`;
  - `src/ai4s_agent/replanner.py`;
- 32 additional entries are intentionally semantic/group references rather than
  exact file paths and require owner/maintainer interpretation before migration.

The v2 implementation locations are not present yet, as required by the gate:
`src/molly/`, `plugins/br1_inverse_design/`, `plugins/remote_compute/`,
`tests/molly/`, `scripts/v2/`, and `docs/v2/reports/` were absent before this
report was added. No production runtime code was changed.

## CI and dependency boundary inventory

The current repository provides these real commands and lanes:

- `.github/workflows/ci.yml`: compile, shard validation, and PR Fast pytest
  (`(unit and not slow) or pr_fast`);
- `.github/workflows/full-ci.yml`: compile, shard validation, and four weighted
  Full CI pytest shards;
- `.github/workflows/queued-canary-manual-nightly.yml`: optional queued-canary
  evidence lane;
- `.github/workflows/scheduled-ci.yml`: scheduled checks.

The current `pyproject.toml` still defines the legacy `ai4s-agent` package and
places MinerU in `quickstart`, observability packages in `observability`/`tracing`,
and RDKit/pdf/report tooling in `dev`. This is inventory evidence only; the v2
extras and core dependency boundary remain unfrozen, so C5 stays pending. The
checked-in `uv.lock` is also stale relative to `pyproject.toml`: its package
metadata currently records only Flask/Pydantic as runtime dependencies and
pytest as the dev extra, omitting the declared httpx/jsonschema/keyring/Pillow/
platformdirs and the other declared optional packages. The lock must not be
silently regenerated as a v2 boundary decision before C5 is frozen.

## Verification evidence

Run from the exact worktree:

```text
git diff --check                         PASS
PYTHONPYCACHEPREFIX=<temporary-dir> \\
  python -m compileall -q src tests      PASS
```

The existing legacy acceptance test was not used to claim any v2 readiness or
BR1 parity. No fresh-real BR1 run, private-data access, GPU training, remote
restart canary, or cutover operation was performed.

## Blockers and owner actions

CORE-00 is complete as an audit but the implementation gate remains closed.
Before Goal-mode implementation, the Owner/maintainer must:

1. record final approval of the exact spec and readiness contract;
2. create and verify the immutable v1 freeze commit/tag and legacy rollback
   branch without moving an existing freeze tag;
3. resolve the three unresolved matrix paths and the semantic group entries;
4. freeze provider/security, dependency/CI, licensed sample/gold-set, and BR1
   contract boundaries, then update readiness with source-backed evidence;
5. explicitly provide or approve any private data, GPU/worker, and credential
   prerequisites for B2/B3. Codex must not infer these approvals.

Until those conditions are recorded, keep `core_goal_mode_ready=false`, preserve
the legacy v1 path, and do not create `src/molly` production code.
