# Molly — Long-Horizon AI4S Agent

> **Public development repository.** This repository is the canonical source
> for Molly development from 2026-07-27 onward. The complete pre-migration
> history remains in a private audit archive and is intentionally not mirrored
> here.

Molly is a long-horizon AI4S agent for literature-grounded scientific
discovery. It turns conversational intent and reviewed evidence into controlled,
provenance-preserving workflows that can survive restarts, publish replayable
results, and expose an auditable trajectory without treating UI or telemetry as
scientific authority.

Molly is an execution and evidence system, not a claim that generated materials
are experimentally valid. Recommendations, predictions, computational
validation, and experimental validation remain distinct claim levels.

## Capability overview

- Natural-language intent capture, durable project conversations, immutable
  uploads, and frozen execution requests.
- Literature source planning, controlled intake, PDF parsing, extraction,
  review, and provenance-bound dataset preparation.
- `RunPlanExecutor` execution with immutable snapshots, explicit human gates,
  task policy, artifact registration, and fail-closed resume checks.
- Bounded multi-round discovery sessions with deterministic child runs,
  publication replay, crash recovery, and reconciliation.
- Local and remote scientific execution through logical resource and
  environment profiles kept outside the repository.
- Observer-only control-plane projection and SSE, plus trajectory integrity and
  audit work that cannot advance or approve scientific state.
- Evidence-aware modeling, diagnostics, candidate generation and screening,
  review, and controlled asset promotion.

The exact implementation, test, and validation status of these areas is kept in
[`todo.md`](todo.md), not duplicated here.

## Trusted execution boundaries

1. Conversation and planning artifacts are non-executable until an immutable
   execution request and RunPlan are created.
2. A gate approval applies only to the current frozen execution snapshot; it
   cannot authorize changed inputs or future work.
3. Gated scientific adapters run through `RunPlanExecutor`, never through a
   direct-adapter shortcut.
4. Session revisions, artifact registry records, execution records, and bound
   publications establish execution facts. UI state, caches, SSE events, and
   observer projections do not.
5. Recovery may adopt a completed child only after exact publication replay and
   state reconciliation; it must not infer success from mutable telemetry.
6. External literature, LLM, remote compute, and promotion actions require their
   own explicit authorization boundaries.

## Quickstart

Molly requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"

PYTHONPATH=src .venv/bin/python -m flask \
  --app 'ai4s_agent.app:create_app' run --host 127.0.0.1 --port 8792
```

Open `http://127.0.0.1:8792/` or verify the service with:

```bash
curl http://127.0.0.1:8792/healthz
```

Run the test suite with:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

## Local and remote resources

The default local workspace is the repository checkout; ignored `projects/`
and `runs/` directories hold runtime state. Set `MOLLY_WORKSPACE` only when a
different local workspace is required, and never commit its resolved path.

User-specific configuration belongs in the platform configuration directory or
an explicit `MOLLY_CONFIG_DIR`. Keep real hostnames, SSH aliases, usernames,
known-hosts files, interpreter paths, credentials, private papers, and runtime
outputs there—not in Git.

- [Local deployment and private configuration](docs/local_deployment.md)
- [Private remote worker setup](docs/remote_worker_setup.md)
- [Remote execution lifecycle](docs/stage6b-remote-execution-lifecycle.md)

## Documentation

Start with the [documentation map](docs/README.md). Key technical entry points
include:

- [Literature intake](docs/literature-intake.md)
- [Bounded discovery sessions and recovery](docs/oled-bounded-discovery-session.md)
- [Control-plane event projection and SSE](docs/control-plane-event-projection.md)
- [Resume-intent validation](docs/resume-intent-validation-semantics.md)
- [Document parsing providers](docs/document-parsing-providers.md)
- [Security policy](SECURITY.md)

Schemas, sanitized evidence summaries, public examples, and operator runbooks
live under `docs/`. Dated implementation checklists are not maintained as a
second project history.

## Roadmap and status

`todo.md` is the normative source for roadmap, milestone status, priorities,
and execution order. Topic documents explain contracts and operations but must
not maintain a competing status table or decision log.

References written as `legacy-private PR N` identify authorized records in the
pre-migration private audit archive. They are not links to public pull requests.
Public PR numbering restarted with this repository. Unlinked PR numbers in
dated pre-migration technical evidence have the same legacy-private meaning
unless an explicit public GitHub URL says otherwise.

## Public repository boundary

This repository may contain source code, public documentation, synthetic
fixtures, machine-readable schemas, and reviewed sanitized evidence. It must not
contain credentials, real user or project data, private papers, unpublished
full-text material, runtime bundles, private Git history, personal operating
instructions, or concrete infrastructure identities and paths.

Before publishing evidence, replace infrastructure locators with logical IDs,
retain content digests only when they are safe and useful, and verify the claim
boundary. See [SECURITY.md](SECURITY.md) for reporting and privacy guidance.
