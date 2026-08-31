# CORE-08 — Default cutover and legacy runtime closure

## Scope

This report records the CORE-08 default-surface cutover prepared from the
merged CORE-07 mainline. The branch changes the current distribution,
operator surface, dependency closure, CI selection, and documentation so that
Molly Core v2 is the intended default after merge. It does not implement new
scientific features or alter the accepted BR1/remote implementation.

The explicit Owner decision supplied for this Goal is recorded in
[`CORE_V2_CUTOVER_APPROVAL.md`](../decisions/CORE_V2_CUTOVER_APPROVAL.md).

## Base and decision

```text
base_commit: d0182fea0fd33c0b716fd39d89f08cdee4c7791c
b4_decision_commit: cd156d8
cutover_commit: 3b743fe
package_evidence_commit: 16d1c90
ci_repair_commit: 312023d6c0fd3c53d8a14091946e4cea2d7a76bb
current_executable_test_head: 312023d6c0fd3c53d8a14091946e4cea2d7a76bb
b4_decision: APPROVED
b4_status: PASS
readiness_transition: core_cutover_ready false → true
owner_decision: APPROVED_FOR_DEFAULT_CUTOVER
pull_request: #72 (Draft)
```

The decision applies to the accepted CORE-06 BR1 parity and CORE-07 runtime
surface, the default package/runtime transition, removal of obsolete v1
runtime from the current tree, and permanent retention of the immutable v1
rollback refs. It does not approve experimental scientific claims, a
mandatory MinerU route, live telemetry acceptance, UI/API migration, or a
real-paper-to-BR1 integrated workflow.

## Package and runtime surface

```text
distribution: molly
default console entrypoint: molly = molly.cli:main
removed legacy entrypoint: molly-worker
mandatory runtime dependency: jsonschema
```

The optional dependency boundary is kept explicit in `pyproject.toml`:
`pdf`, `mineru`, `observability`, and `dev`. BR1 and remote-compute modules
remain optional plugin seams whose host environments are not mandatory Core
dependencies. MinerU is an optional PDF fallback; it is not a central
architecture dependency.

The current `src/ai4s_agent` runtime tree, legacy-only top-level tests, legacy
acceptance scripts, and queued-canary workflow are removed from the current
mainline. No compatibility shadow package or replacement worker command was
added. The v1 implementation remains recoverable through the immutable refs
below.

The removal is limited to the current mainline. It does not delete or move
the frozen v1 branch/tag, and it does not change the accepted Core v2 BR1 or
remote-compute modules.

## Documentation and CI cutover

The root README, documentation map, roadmap, and security policy now describe
Core v2 as current and v1 as rollback/history only. The roadmap records
real-literature/data-mining integration, structured-source alternatives,
domain expansion, optional UI/API work, and separately approved research
extensions as post-cutover work.

PR Fast and Full CI continue to use the current retained test tree and the
deterministic shard selector. Legacy queued-canary CI was removed. CI installs
the current development/PDF test boundary rather than the old legacy runtime,
and cache keys include both `pyproject.toml` and `uv.lock`.

The repository CodeQL Default Setup was reconciled after the v1 source removal:
it now analyzes the current `actions` and `python` languages only. The removed
JavaScript/TypeScript lane had no source left to analyze and was producing a
configuration failure; no placeholder JavaScript was added.

## Lockfile and clean package checks

The existing lock was regenerated after the package metadata change. The
removed package records are legacy-only dependencies; retained package source
URLs and versions did not change. The lock check is recorded below after the
final tree is verified.

The verification records the editable clean-install/import smoke, wheel and
sdist contents, CLI help commands, and one offline server-owned runtime
profile smoke. These checks did not install or invoke real GPU, network, LLM,
BR1, or remote-compute work.

## Rollback verification

Both permanent refs remain required to resolve exactly to the frozen v1
commit:

```text
tag:    molly-v1-pre-core-v2-20260829
branch: legacy/molly-v1
commit: ae7892dbf8a6bfe85dd909056eadc2afecc40d9d
```

The rollback procedure is Git-based: inspect or check out either immutable ref
in a separate worktree, then run the v1 package/runtime from that ref. The
current Core v2 package does not provide a v1 compatibility alias.

## Verification record

| Check | Result | Evidence |
|---|---|---|
| `git diff --check` | PASS | current executable/test head |
| `python -m compileall -q src tests prototypes` | PASS | current executable/test head |
| `uv lock --check` | PASS on current lock | regenerated `uv.lock` |
| CORE-08 focused tests | 7 passed | `tests/molly/test_core08_cutover.py` |
| retained v2 regression tests | 221 passed | CORE-01 through CORE-07, readiness, privacy |
| isolated `pip install -e .` | PASS | imports and six installed CLI help commands |
| optional heavy modules in clean install | absent | MinerU and RDKit not installed |
| offline runtime/inspection/observation smoke | PASS | `tests/molly/test_core08_cutover.py` |
| PR Fast local selector | 217 passed, 4 deselected | exact selector on `16d1c90`; GitHub rerun also passed on `312023d6` |
| PR Fast GitHub | PASS | run `33392912090`, tested HEAD `312023d6c0fd3c53d8a14091946e4cea2d7a76bb` |
| CodeQL | pending final rerun | Default Setup is now reduced to Actions/Python; the final analysis is triggered by this report-only push |
| Full CI | PASS | run `33393433265`, tested HEAD `312023d6c0fd3c53d8a14091946e4cea2d7a76bb`; compile policy and weighted shards 0–3 passed |

## Final-state claims

After this PR merges, Molly Core v2 is the intended default mainline
runtime/package. Legacy v1 remains permanently available through the immutable
branch and tag. This report does not claim real-paper → reviewed dataset →
BR1 integrated acceptance; that is post-cutover scientific work. It also does
not claim a new BR1 run, remote canary, MinerU run, live telemetry acceptance,
or experimental validation.
