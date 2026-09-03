# Molly

> A bounded, provenance-aware execution and evidence system for reproducible
> scientific work.

Molly helps scientific workflows run with explicit tools, immutable outputs,
durable execution records, and reviewable provenance. It is designed for
research environments where a result should be inspectable—not just produced.

Molly Core v2 is the default mainline runtime and architecture. Computational predictions,
rankings, and generated candidates remain computational claims; they are not
experimental validation.

## Why Molly

Scientific automation needs more than a model call. Molly makes the important
boundaries explicit:

- **Bounded execution** — a closed, server-owned `ToolRegistry` and
  `ToolPolicy` define what can run.
- **Immutable content** — `ArtifactStore` addresses exact bytes by SHA-256 and
  rejects conflicting publication.
- **Durable runs** — `RunLedger` records append-only execution occurrences and
  bounded tool observations.
- **Traceable provenance** — `ArtifactLineage` records inputs, outputs, and
  bounded support relationships without pretending provenance is causality.
- **Human review** — `ReviewRecord` binds a decision to the exact artifact
  digest that was reviewed.
- **Safe acquisition** — literature retrieval uses closed provider
  configuration, network policy, cache integrity, and licensing metadata.

## Architecture

```text
RunRequest
    ↓
AgentLoop
    ↓
ToolRegistry + ToolPolicy + exact approvals
    ↓
Scientific tools and optional plugins
    ↓
ArtifactStore + RunLedger + ArtifactLineage
    ↓
ValidationResult + ReviewRecord
```

The scientific path is intentionally modular:

```text
literature acquisition
    → CanonicalDocument
    → evidence / OLED mapping
    → human review
    → reviewed dataset
    → optional BR1 inverse-design plugin
```

## Current capabilities

| Area | What it provides |
| --- | --- |
| Core execution | One bounded `AgentLoop`, typed tool calls, exact approvals, restart-safe state |
| Artifacts | Content-addressed immutable bytes, digest verification, atomic publication |
| Provenance | Append-only run events and lightweight artifact lineage |
| Literature | Policy-controlled metadata, open-access resolution, and full-text acquisition |
| Documents | Deterministic `CanonicalDocument` routing for supported source formats |
| Scientific evidence | Validation, source mapping, reviewed evidence, and OLED dataset contracts |
| BR1 | Optional inverse-design plugin for the reviewed-dataset → model → candidate → prediction → Computational Top-N chain |
| Operations | CLI-first run, inspection, approval, review, and observation surfaces |
| Observability | Observer-only JSON, OpenTelemetry, and LangSmith projections; telemetry cannot advance scientific state |

## Quickstart

Molly requires Python 3.10 or newer.

```bash
git clone https://github.com/searching42/Molly.git
cd Molly

python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

.venv/bin/molly --help
```

The default installation contains the small Core v2 runtime. It does not
install heavy scientific, PDF, telemetry, or legacy runtime dependencies.

### CLI surface

```bash
.venv/bin/molly inspect --help
.venv/bin/molly run --help
.venv/bin/molly approve --help
.venv/bin/molly review --help
.venv/bin/molly observe --help
```

| Command | Purpose |
| --- | --- |
| `molly run` | Start or resume a bounded run using a registered host profile |
| `molly inspect` | Read authoritative run or artifact state without changing it |
| `molly approve` | Record one decision for one exact pending tool call |
| `molly review` | Bind a scientific review to one exact artifact digest |
| `molly observe` | Produce a read-only run trace or observer export |
| `molly web` | Start the local-only browser interface; add `--demo` to try the UI without a registered scientific profile |
| `molly config` | Save or remove provider credentials through a hidden terminal prompt |

Molly does not ship a general-purpose default LLM Agent profile. A scientific
run requires an explicitly registered, server-owned `RuntimeProfile` containing
the appropriate `ToolRegistry`, `ToolPolicy`, and `DecisionProvider`. Inspection
and observation do not require inventing one.

## Optional capabilities

Install only what the host environment needs:

```bash
.venv/bin/python -m pip install -e ".[pdf]"            # lightweight PDF text parsing
.venv/bin/python -m pip install -e ".[mineru]"         # optional MinerU fallback
.venv/bin/python -m pip install -e ".[observability]"  # optional telemetry exporters
.venv/bin/python -m pip install -e ".[dev]"             # test and packaging tools
```

MinerU is a PDF fallback, not the center of the architecture. BR1 and
remote-compute implementations are optional plugin seams whose heavy model,
chemistry, GPU, and transport environments remain server-owned; they are not
required to import or use `molly.core`.

## Data and evidence model

| Component | Authority |
| --- | --- |
| `ArtifactStore` | Exact immutable content and content identity |
| `RunLedger` | Factual append-only execution occurrences |
| `ArtifactLineage` | Bounded input/output/support provenance projection |
| `ValidationResult` | Closed-scope validation of artifacts, relations, or bundles |
| `ReviewRecord` | Human decision bound to one exact artifact SHA-256 |

Occurrence provenance is kept separate from content identity: identical bytes
may be produced by multiple runs or steps without one occurrence overwriting
another. Large scientific payloads belong in `ArtifactStore`; small bounded
tool observations remain in the run ledger.

## Security boundaries

Molly keeps authority on the host side:

- model proposals cannot choose filesystem roots, shell commands, credentials,
  SSH targets, private endpoints, or runtime profiles;
- provider, compute, and runtime configuration is server-owned;
- acquisition enforces allowlists, DNS/IP checks, redirect validation, size and
  content limits, caching, and access/licensing provenance;
- credentials are never placed in prompts, artifacts, ledger events, tool
  observations, or public evidence;
- UI, API, and telemetry surfaces are observers or operators, not scientific
  authority.

See [SECURITY.md](SECURITY.md) for the repository security boundary and
[the documentation map](docs/README.md) for current contracts and evidence.

## Local browser interface

The first browser surface is intentionally small and binds only to
`127.0.0.1`. It is an operator view over the existing Core service, not a
second execution authority:

```bash
PYTHONPATH=src .venv/bin/python -c \
  'from molly.cli import main; raise SystemExit(main())' \
  --state-root .molly web --demo
```

Open <http://127.0.0.1:8765/>. The explicit `--demo` profile lets you try the
new-task → confirmation → continue → completed flow with a deterministic local
operation. Normal mode keeps the existing closed profile registry and shows no
implicit general-purpose model profile.

The browser can save only non-secret model settings. It never accepts an API
key. After saving a provider profile in the UI, configure its key from the
same local checkout:

```bash
PYTHONPATH=src .venv/bin/python -c \
  'from molly.cli import main; raise SystemExit(main())' \
  --state-root .molly config set-key --profile provider-test
```

The prompt is hidden, and the credential is stored in the server-side
`provider_secrets.json` file with owner-only permissions. The browser receives
only whether a key is configured; the key is never written to Core requests,
ledger events, artifacts, or browser storage.

### BR1 real end-to-end run

Normal web mode exposes only worker profiles that are complete and registered
by the server. Configure each worker with its own complete, host-qualified
environment-variable bundle before starting Molly. Replace `<HOST_ID>` with
the exact server-registered host identifier and adjust the paths for that host:

```bash
# Replace <HOST_ID> with the exact registered identifier in each name:
# MOLLY_BR1_<HOST_ID>_SSH_TARGET=worker-alias
# MOLLY_BR1_<HOST_ID>_REMOTE_ROOT=/srv/molly/br1
# MOLLY_BR1_<HOST_ID>_UNIMOL_PYTHON=/opt/unimol/bin/python
# MOLLY_BR1_<HOST_ID>_REINVENT_PYTHON=/opt/reinvent/bin/python
# MOLLY_BR1_<HOST_ID>_REINVENT_REPOSITORY=/opt/REINVENT4

PYTHONPATH=src .venv/bin/python -c \
  'from molly.cli import main; raise SystemExit(main())' \
  --state-root .molly web
```

The unqualified `MOLLY_BR1_SSH_TARGET`, `MOLLY_BR1_REMOTE_ROOT`, and related
variables are ignored. Do not reuse one host's SSH target under another
host's identity; each registered worker must resolve to its own target and
provenance.

In the browser, upload the OE62 JSON/CSV file, enter the target and Top-N
request in natural language, choose the registered CPU/GPU profile, and
approve each displayed external-compute step. Molly then runs cleaning,
applicability preflight, fresh Uni-Mol training, unrestricted REINVENT4
sampling, current-run prediction, and Top-N ranking. The page polls the run,
shows the exact tool lifecycle and failure summary, and exposes immutable
result artifacts for download. The default UI budget is sufficient for the
six-stage BR1 chain; lower limits are intended for smoke runs.

Optional observer exports are enabled by server-side configuration:

```bash
export MOLLY_OTEL_ENDPOINT=https://otel.example/v1/traces
export LANGSMITH_API_KEY= # set outside browser/UI, if used
```

JSON observation is always available. OpenTelemetry and LangSmith remain
observer-only and cannot change the run or its artifacts.

## Development

Install the development extra and run the focused checks:

```bash
.venv/bin/python -m pip install -e ".[dev]"
PYTHONPATH=src PYTHONHASHSEED=0 \
  .venv/bin/python -m pytest -q \
  -m "(unit and not slow) or pr_fast"
```

Current package metadata and optional dependency boundaries are defined in
[`pyproject.toml`](pyproject.toml).

## Further reading

- [Core v2 documentation map](docs/README.md)
- [Core v2 execution contract](docs/v2/CODEX_GOAL_EXECUTION_CONTRACT.md)
- [Core v2 simplification specification](docs/v2/MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md)
- [Package and CI boundary](docs/v2/contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md)
- [Acquisition security contract](docs/v2/contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md)
- [CORE-08 default cutover report](docs/v2/reports/CORE-08.md)
