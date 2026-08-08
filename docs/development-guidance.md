# Molly development guidance

This document contains durable repository-wide guidance for contributors and development agents working in the Molly public repository. Local `CLAUDE.md` may contain machine- or workflow-specific notes, but it is intentionally ignored by Git and is not repository authority.

## Repository role and authority

Molly is a long-horizon AI4S agent for literature-grounded, controlled, and provenance-preserving scientific discovery workflows. It is not limited to an early same-process prototype.

[`roadmap.md`](roadmap.md) is the sole normative source in the public repository for roadmap scope, milestone status, priorities, execution order, and public roadmap decisions. Do not create a second authoritative backlog, status table, or decision log in README files, topic documents, comments, or local agent memory.

## Invariants

- Keep conversation state separate from immutable execution requests.
- Do not bypass `RunPlanExecutor`, execution policy, human gates, or execution snapshot binding. Approval applies only to the exact current snapshot.
- Preserve no-replace publications, exact child-run replay, crash recovery, reconciliation, and artifact provenance guarantees.
- Treat Session revisions, execution records, artifact registry entries, and bound publications as authoritative execution evidence.
- Keep UI state, caches, queue telemetry, SSE events, OpenTelemetry traces, and LangSmith runs non-authoritative. They must not advance work or infer scientific success.
- Keep recommendation, prediction, computational validation, and experimental validation claims distinct.
- Do not add a new state authority or silently weaken failure-closed behavior for compatibility.

Compatibility modules may retain legacy names while they remain imported, tested, or required for exact replay. Do not remove or broadly rename them as part of unrelated documentation cleanup.

## Public-repository boundary

- Never commit credentials, tokens, private papers, user/project data, runtime bundles, known-hosts content, real hostnames or usernames, resolved absolute infrastructure paths, or machine-specific profiles.
- Store private connection, environment, provider, and local working-context configuration outside the tracked tree.
- Use logical resource IDs and synthetic fixtures in public code and docs.
- Treat `legacy-private PR N` and historical SHAs as private audit references unless an explicit public GitHub URL is present.
- Local `CLAUDE.md` and `todo.md` are ignored working-context files. Their presence or contents must never be required by CI, packaging, runtime behavior, or public documentation links.

## Development commands

```bash
# First-time environment
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

# Local application
PYTHONPATH=src .venv/bin/python -m flask \
  --app 'ai4s_agent.app:create_app' run --host 127.0.0.1 --port 8792

# Focused and full tests
PYTHONPATH=src .venv/bin/python -m pytest tests/<test_file>.py -q
PYTHONPATH=src .venv/bin/python -m pytest -q

# Baseline repository checks
.venv/bin/python -m compileall -q src tests
git diff --check
```

## Change discipline

1. Identify the closest public roadmap milestone or state that the work is non-blocking maintenance.
2. Inspect the current implementation and tests before changing a documented contract.
3. Make the smallest change that preserves execution, replay, recovery, gate, and provenance boundaries.
4. Add or update tests for changed behavior and failure boundaries.
5. Keep topic docs focused on durable contracts and operations; put public roadmap status only in [`roadmap.md`](roadmap.md).
6. Keep private scratch plans, machine notes, and local task tracking in ignored working-context files rather than tracked documentation.
7. Run focused checks, then the full suite when practical, and report any check that was not run.
