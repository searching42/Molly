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

## Project status

The defined Core v2 milestones `CORE-00` through `CORE-08` are complete for
their accepted contracts. The current readiness state is recorded in the
[readiness manifest](docs/v2/readiness/core_refactor_readiness.json) and the
[CORE-08 cutover report](docs/v2/reports/CORE-08.md).

The following are deliberately post-cutover work, not hidden promises of the
current runtime:

- validate new real-literature and data-mining pipelines;
- evaluate structured acquisition paths that may replace or bypass MinerU;
- expand scientific domains and acceptance datasets;
- add a UI or general HTTP API only under a separately reviewed scope;
- pursue research trajectory or error-propagation work only with separate
  approval and evidence.

## Development

Install the development extra and run the focused checks:

```bash
.venv/bin/python -m pip install -e ".[dev]"
PYTHONPATH=src PYTHONHASHSEED=0 \
  .venv/bin/python -m pytest -q \
  -m "(unit and not slow) or pr_fast"
```

Current package metadata and optional dependency boundaries are defined in
[`pyproject.toml`](pyproject.toml). The project roadmap is in
[`docs/roadmap.md`](docs/roadmap.md).

## Legacy rollback boundary

The pre-Core-v2 runtime is not the default package or entrypoint. It remains
available as immutable Git history:

```text
tag:    molly-v1-pre-core-v2-20260829
branch: legacy/molly-v1
commit: ae7892dbf8a6bfe85dd909056eadc2afecc40d9
```

Never move or force-update these references. To inspect the frozen source
without changing the current checkout:

```bash
git show molly-v1-pre-core-v2-20260829:pyproject.toml
git ls-tree -r --name-only legacy/molly-v1 | sed -n '/^src\//p' | head
```

Historical v1 documents and acceptance records are retained for audit context;
the current architecture authority is under `docs/v2/`.

## Further reading

- [Core v2 documentation map](docs/README.md)
- [Core v2 roadmap](docs/roadmap.md)
- [Core v2 execution contract](docs/v2/CODEX_GOAL_EXECUTION_CONTRACT.md)
- [Core v2 simplification specification](docs/v2/MOLLY_CORE_SIMPLIFICATION_REFACTOR_SPEC_V1_1_BR1_HARDENED.md)
- [Package and CI boundary](docs/v2/contracts/PACKAGE_DEPENDENCY_AND_CI_BOUNDARY.md)
- [Acquisition security contract](docs/v2/contracts/ACQUISITION_SECURITY_AND_PROVENANCE.md)
- [CORE-08 default cutover report](docs/v2/reports/CORE-08.md)
