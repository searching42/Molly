# Molly — Core v2

Molly Core v2 is the default mainline runtime for bounded, provenance-aware
scientific work. The CORE-00 through CORE-08 milestones are complete in the
current development line. The pre-Core-v2 runtime is retained only through
the immutable rollback references documented below.

Molly is an execution and evidence system. Computational predictions and
candidate rankings are not experimental validation claims.

## Current architecture

```text
RunRequest
    ↓
AgentLoop
    ↓
ToolRegistry + ToolPolicy
    ↓
Scientific tools and optional plugins
    ↓
ArtifactStore + RunLedger + ArtifactLineage
    ↓
ValidationResult / ReviewRecord
```

The scientific path is:

```text
literature acquisition → CanonicalDocument → evidence/OLED mapping
    → review → dataset → optional BR1 inverse-design plugin
```

The default runtime keeps these boundaries explicit:

- `ArtifactStore` owns immutable content identity.
- `RunLedger` records execution occurrences and durable bounded observations.
- `ArtifactLineage` records bounded provenance relations.
- `ToolRegistry`, `ToolPolicy`, and exact approvals are server-owned execution
  boundaries.
- `RuntimeProfile` and provider/compute profiles are server-owned. Inspection,
  review, and observation do not invent a general-purpose Agent profile.
- MinerU is an optional PDF fallback, BR1 is an optional plugin, and
  OpenTelemetry/LangSmith are observer-only integrations.
- UI/API surfaces are not scientific authority. Error-propagation research is
  not implemented or implied by this repository.

## Quickstart

Python 3.10 or newer is required.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

.venv/bin/molly --help
.venv/bin/molly inspect --help
.venv/bin/molly run --help
.venv/bin/molly observe --help
```

The default installation contains the minimal Core v2 runtime. Optional
profiles are available for PDF parsing, MinerU, observability, and development
tools; they are not mandatory for `pip install -e .`.

Molly does not ship a general default LLM Agent profile. A scientific Agent
requires an explicitly registered server-owned `ToolRegistry`, `ToolPolicy`,
and `DecisionProvider` profile. Offline inspection and observation can be used
without creating one.

## Optional scientific profiles

Install only the capability needed by the host environment:

```bash
.venv/bin/python -m pip install -e ".[pdf]"
.venv/bin/python -m pip install -e ".[mineru]"
```

BR1 and remote-compute code are optional plugin seams. Their heavy scientific
or transport environments remain server-owned and are not required by the
minimal Core package. The exact available extras are declared in
`pyproject.toml`.

## Evidence and security

Start with the [documentation map](docs/README.md), the [Core v2 roadmap](docs/roadmap.md),
and the [CORE-08 cutover report](docs/v2/reports/CORE-08.md). Current contracts,
readiness evidence, and milestone reports live under `docs/v2/`.

Do not commit credentials, private datasets, private papers, concrete
infrastructure locators, runtime bundles, or resolved machine paths. See
[SECURITY.md](SECURITY.md) for reporting and privacy boundaries.

## Rollback boundary

The legacy runtime is not the default package or entrypoint on current main.
It remains recoverable without Core v2 through these immutable refs:

```text
tag:    molly-v1-pre-core-v2-20260829
branch: legacy/molly-v1
commit: ae7892dbf8a6bfe85dd909056eadc2afecc40d9
```

To inspect the frozen implementation without changing the current checkout:

```bash
git show molly-v1-pre-core-v2-20260829:pyproject.toml
git ls-tree -r --name-only legacy/molly-v1 | sed -n '/^src\//p' | head
```

Never move or force-update either reference. The historical documents and
acceptance evidence under `docs/` are retained as non-normative context unless
they are explicitly part of a current `docs/v2/` contract.
