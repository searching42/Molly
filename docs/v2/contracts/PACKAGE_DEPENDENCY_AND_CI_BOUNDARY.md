# Package, dependency, and CI boundary

Status: `PASS` (`FROZEN_C5_BOUNDARY`)

This document freezes the dependency and validation boundary for future Core
v2 implementation. It does not restructure the package, create `src/molly/`,
or create the final v2 workflows during CORE-00.

## Current repository lock repair versus future design

At the C5 baseline the repository remains the legacy `ai4s-agent` package. The
declared `pyproject.toml` runtime dependencies are Flask, httpx, jsonschema,
keyring, Pillow, platformdirs, Pydantic, and the Python-version-conditional
tomli dependency. Its optional groups include MinerU under `quickstart`, the
development/document/scientific/test dependencies under `dev`, and
OpenTelemetry/LangSmith under `tracing` and `observability`.

The checked-in `uv.lock` was stale relative to those declarations. The
authorized C5 repair is only:

```text
uv lock
uv lock --check
```

using the current `pyproject.toml`, followed by inspection of the lock diff.
This repair makes the existing legacy package lock-consistent. It is not the
future Core v2 package split and must not be described as one.

The completed repair record is maintained in the C5 section below with the
resulting lock digest, changed-package summary, and `uv lock --check` result.
The large diff is attributable to the previously missing current declared
runtime and optional groups; no pre-existing locked version was changed and
no non-registry package source was introduced.
No unrelated source, package declaration, or package-source change is
accepted as part of this repair.

## Future conceptual installation profiles

The future package design has separate profiles. Names describe dependency
responsibilities; they are not installed by this Goal.

| Profile | Responsibility | Explicit exclusions or constraints |
| --- | --- | --- |
| `core/minimal` | RunRequest, ToolRegistry/ToolPolicy, approvals, artifacts, ledger, lineage, bounded validation, and deterministic core tests. | Must not require MinerU, RDKit, Uni-Mol, REINVENT4, OpenTelemetry exporters, LangSmith, remote SSH, or UI-only dependencies. |
| `pdf` | Optional text-layer PDF parsing behind the document contract. | Not imported by XML/HTML core paths. |
| `mineru` | Optional MinerU PDF fallback and parser-quality metadata. | Never a minimal-install requirement; credentials and endpoints remain server-owned. |
| `observability` | Observer-only OpenTelemetry and LangSmith integrations. | Exporter failure is non-fatal and cannot change ledger facts or tool outcomes. |
| `br1` | Optional BR1 dataset, Uni-Mol, REINVENT4, ranking, and scientific evaluation tools. | Uni-Mol, REINVENT4, and RDKit remain BR1/tool-environment concerns, not Core runtime requirements. |
| `remote` | Optional durable remote submit/inspect/collect and restart-safe worker integration. | SSH/remote stacks are absent from `core/minimal`; credentials never enter model-visible records. |
| `dev` | Test, lint, report, fixture, and local development dependencies. | Development dependencies do not become runtime Core dependencies. |

The dependency direction is one-way: the minimal core owns the data and
authority contracts; optional profiles consume those contracts. Optional
profiles do not add authority layers, hidden network access, or required
runtime imports back into the minimal core.

## Future CI lanes

These lanes are frozen conceptually. The final v2 workflows are not created by
this Goal.

| Lane | Responsibility | Cutover requirement |
| --- | --- | --- |
| `core-fast` | Compile, schema/contract tests, content-addressed artifacts, append-only ledger, and fail-closed policy checks. | Required for every Core change. |
| `document-parser` | Offline XML/JATS/HTML and optional PDF parser contract tests, including locator and content-family checks. | Required before document-parser production milestone. |
| `network-mock` | Deterministic provider mocks for allowlists, SSRF, redirects, rate limits, retries, cache, content type, and provenance. | Required before any acquisition network-live canary. |
| `oled-domain` | OLED ontology, identity, units, duplicate/conflict, leakage, review, and provenance fixtures. | Required before OLED production acceptance. |
| `observability` | Observer-only exporters, privacy filtering, and exporter-failure non-interference. | Required before observability plugin acceptance. |
| `br1-contract` | Offline BR1 artifact schema, current-run binding, stale/foreign rejection, deterministic terminal replay, and claim-boundary tests. | Required before any real BR1 canary. |
| `br1-real-canary` | Fresh Uni-Mol training, real REINVENT4 generation, current-model prediction, deterministic evaluation, and reviewed Computational Top-N projection. | Required for B2 and before default BR1 cutover. |
| `remote-restart-canary` | Durable submit/inspect/collect, idempotency, restart/replay, credential boundary, and no-duplicate-dispatch checks. | Required for B3 and any remote default route. |

No lane can promote its optional dependencies into `core/minimal`. Real
canary lanes do not run as a side effect of the offline contract lane.

## Lock inspection record

```yaml
current_pyproject: pyproject.toml
lock_repair_scope: current legacy pyproject declarations only
future_v2_packaging_scope: not implemented in CORE-00
uv_lock_command: uv lock
uv_lock_check_command: uv lock --check
lock_digest_after_repair: f204dc52afd4d2b50e58651e75bb75a8f6fa0a0192d9f17e03c79891122b30c4
lock_diff_review: PASS; 21 package entries became 177 (156 entries added to represent current declared runtime and optional groups, 0 removed, 0 pre-existing versions changed); only the editable project source and the PyPI registry source are present
uv_lock_check: PASS
unexpected_source_or_dependency_drift: NOT_ACCEPTED
```

## C5 decision

The dependency profiles and CI responsibilities are frozen. C5 is `PASS`: the
existing lock was regenerated against the current `pyproject.toml`, the
resulting diff was reviewed for unrelated drift, and `uv lock --check` passed.
The lock repair and the future v2 boundary remain separate decisions.
