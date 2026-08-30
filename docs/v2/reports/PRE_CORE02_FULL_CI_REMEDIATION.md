# Pre-CORE-02 Full CI Remediation

Status: remediation complete; CORE-02 has not started.

## Scope and baseline

This report covers the two failures from Full CI run `33295179936`, tested at
HEAD `ca12ca92606d933d4006ed8f4b2451d6ef3fa032`:

- Shard 1: `tests/test_repository_privacy.py::test_tracked_repository_has_no_generic_private_infrastructure_markers`
- Shard 3: `tests/test_br1_conversation_runtime_bridge.py::test_br1_conversation_front_door_drives_synthetic_remote_chain`

The remediation branch is based on `origin/main` at
`93fdb08e924f116451bf6472af9bf85a3d473f24`. It is independent of PR #65.

## Root causes and fixes

### Shard 1 — generated `uv.lock` version literal

The repository privacy helper applied the generic IPv4 pattern to every byte
of every tracked file. Four-component dependency versions, including
recognized PyPI artifact filename versions, were
therefore reported as infrastructure IPv4 addresses. The report deliberately
uses a symbolic four-component version description rather than reproducing a
numeric example that would itself be indistinguishable from an IPv4 address
to the repository scanner.

The fix is a narrow path- and syntax-aware exception for recognized generated
`uv.lock` dependency-version fields and canonical PyPI artifact version
fragments. All other `uv.lock` content remains scanned, including source URLs,
registry URLs, arbitrary strings, paths, emails, and credentials. The lock
file itself was not changed.

Regression coverage confirms that a dependency version is ignored, a private
IPv4 in a lockfile source URL is still reported, and the same version-shaped
literal outside a recognized lockfile context remains a finding.

### Shard 3 — human remote approval versus autonomous lease enforcement

The Controller previously sent every remote dispatch/refresh/adopt decision
through the autonomous remote-runtime evidence guard. That guard correctly
fails closed for autonomous work, but it did not distinguish the separate
typed human approval path. Consequently, an exact approved BR1 request was
blocked before its deterministic lifecycle could run.

The Controller now re-reads and verifies the server-owned request, slot
binding, and approval evidence for each remote continuation. The exact
project, run, controller execution, task/slot, request identity and SHA-256,
approval decision, and approval digest must match. Only then may the exact
dispatch/refresh/adopt continuation bypass autonomous lease verification and
accounting. The approval does not mint, extend, renew, or broaden a lease.

The low-level autonomous lease service remains unchanged in behavior:
dispatch/refresh/adopt without trusted remote-runtime evidence still fails
with `AUTONOMY_REMOTE_BUDGET_ENFORCEMENT_UNAVAILABLE`. Resource profiles,
GPU/CPU/walltime bounds, and remote runtime evidence rules were not weakened.

## Files changed

- `src/ai4s_agent/scientific_agent_harness_controller.py`
  - exact persisted human-approval revalidation;
  - separate human-authorized remote continuation path with no autonomous
    lease accounting.
- `src/ai4s_agent/scientific_agent_autonomy_lease.py`
  - clarified that the remote-runtime guard is the autonomous path; its
    fail-closed behavior and API remain unchanged.
- `tests/test_repository_privacy.py`
  - context-aware `uv.lock` version regression tests.
- `tests/test_scientific_agent_harness_controller.py`
  - exact human-approved remote dispatch regression.
- `docs/v2/reports/PRE_CORE02_FULL_CI_REMEDIATION.md`
  - this evidence report.

No `src/molly` production code was changed. No CORE-02 code was started.

## Security and behavior invariants preserved

- No global IPv4 allowlist, privacy-scan disablement, or lockfile rewrite.
- Real infrastructure-like IPv4 addresses remain findings.
- No conversation text or session projection is treated as authority.
- Stale, foreign, changed, or tampered request/slot/approval evidence fails
  closed.
- Explicit human approval is exact-request scoped and does not authorize a
  different request, task, project, or lifecycle.
- Autonomous remote effects still require trusted server-owned runtime
  enforcement and fail closed when it is unavailable.
- Existing remote resource authority and configured bounds remain intact.
- No credentials, network access, runtime fabrication, or authority expansion
  was introduced.

## Verification

Completed focused checks:

- `tests/test_repository_privacy.py`: 40 passed.
- BR1 front-door synthetic remote chain: 1 passed; full bridge file: 6 passed.
- Exact human-approved remote controller regression: 1 passed.
- `tests/test_scientific_agent_autonomy_lease.py`: 19 passed.
- `tests/test_scientific_agent_harness_controller.py`: 47 passed.
- `tests/test_remote_resource_authority.py`: 45 passed.
- `tests/test_controller_remote_successor_crash_windows.py`: 6 passed.
- `python -m compileall -q src tests prototypes`: passed.
- `git diff --check`: passed.
- `uv lock --check`: passed (`185 packages resolved`; no lockfile change).

PR Fast: passed — 1,526 passed, 5,664 deselected in 225.93s
(`PYTHONPATH=src PYTHONHASHSEED=0 python -m pytest -q -m "(unit and not slow) or pr_fast" --durations=20`).

Final Full CI: passed — run `33299345904` on the exact remediation
code/test HEAD `73bab404b927c4d86b5cf8f67e8f93ade5652a8d`.

The compile/shard-policy job and all four weighted pytest shards passed:
compile/shard policy in 7s, shard 0 in 14m1s, shard 1 in 12m21s, shard 2 in
11m51s, and shard 3 in 20m49s. See the
[final Full CI run](https://github.com/searching42/Molly/actions/runs/33299345904).

The preceding documentation-corrected validation run was
`33298428018` on `cabdb9ed01dda65e1d04b0d2a457352358283cdd`; it also passed
the compile/shard-policy job and all four weighted pytest shards. The final
run above verifies the remediation code/test HEAD after the report was
updated to record the final evidence. Any subsequent report-only commit does
not change executable code or test collection.

The first follow-up Full CI attempt was run `33297534349` at
`b3c6cd7a19587d0763bc05bce6931ba1b138c5e2`. Shards 1, 2, and 3 passed. Shard
0 ran 1,704 tests and found one additional repository-privacy finding in this
report: the explanatory numeric four-component version example. The example
has been removed; this is a documentation-only correction and does not alter
the two runtime fixes.

## Freeze references

Verified before remediation:

- `molly-v1-pre-core-v2-20260829` peels to
  `ae7892dbf8a6bfe85dd909056eadc2afecc40d9`.
- `legacy/molly-v1` points to
  `ae7892dbf8a6bfe85dd909056eadc2afecc40d9`.

These references are not modified by this remediation.

## CORE-01 and cutover status

Neither Full CI failure was caused by CORE-01 production `src/molly` code.
The failures were in the legacy repository privacy test and the legacy v1
Controller/remote approval boundary. CORE-01 readiness evidence remains
unchanged; PR #65 is not modified.

B2, B3, and B4 remain unchanged and not PASS. `core_cutover_ready` remains
`false`. This remediation does not authorize or begin CORE-02.
