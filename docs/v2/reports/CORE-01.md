# Molly Core v2 CORE-01 production foundation

Status: `PASS`

Date: 2026-08-30

Repository: `searching42/Molly`

Branch: `codex/molly-core-v2-core-01-foundation`

Base commit: `93fdb08e924f116451bf6472af9bf85a3d473f24` (merged CORE-00
PR #64 on `origin/main`)

Implementation commit: `c23063c53287365cdb6845b46716dd8abdecc154`

This milestone implements CORE-01 only. It does not implement CORE-02 or any
later milestone, BR1 v2 parity, remote/GPU execution, acquisition, document
parsing, OLED migration, UI, API routes, or default cutover.

## Requirements implemented

The new production namespace is `src/molly/`. It is standard-library-only and
has no import dependency on `ai4s_agent` or the non-production contract spike.

- `ArtifactStore` publishes exact bytes under a SHA-256 content address with a
  deterministic `objects/<prefix>/<digest>` layout and immutable metadata.
  Publication uses a fsynced temporary file plus an atomic no-replace hard
  link. Existing bytes are verified before reuse; corrupt, incomplete,
  conflicting, traversal, and symlink states fail closed.
- `RunLedger` appends canonical UTF-8 JSONL events. Each event carries the
  run/step/tool/artifact/profile/digest/seed/metadata fields needed by later
  execution work, plus a previous-event hash and its own hash. Reopen and
  inspection validate the complete chain and reject malformed or truncated
  records.
- `ArtifactLineage` records only `CONSUMED_BY`, `PRODUCED_BY`, `DERIVED_FROM`,
  and `SUPPORTED_BY` relations. It supports direct parent/producer/support
  inspection and JSONL restart reconstruction; it is not a causal graph or a
  scheduler.
- `ValidationResult` has the closed scopes `ARTIFACT`, `RELATION`, and
  `BUNDLE`, and the closed statuses `PASS`, `FAIL`, and `REVIEW`, with
  deterministic serialization and evidence/source references.
- `ReviewRecord` is frozen and binds a decision to both the exact
  `sha256:<digest>` artifact identity and its bare SHA-256 digest. Reuse
  against another artifact fails through `assert_matches`.
- Canonical JSON is compact, sorted-key, ASCII-escaped UTF-8 with non-finite
  numbers rejected. Timestamps normalize to UTC RFC 3339 with microseconds.
  Metadata rejects credential-like keys; no network, shell, credential,
  model, or remote-compute authority is present.

## Files

Added production modules:

```text
src/molly/__init__.py
src/molly/core/__init__.py
src/molly/core/_persistence.py
src/molly/core/artifacts.py
src/molly/core/errors.py
src/molly/core/ids.py
src/molly/core/ledger.py
src/molly/core/lineage.py
src/molly/core/reviews.py
src/molly/core/validation.py
```

Added focused tests:

```text
tests/molly/test_core01_foundation.py
```

Added this report:

```text
docs/v2/reports/CORE-01.md
```

No `pyproject.toml` or `uv.lock` change was required: the existing setuptools
package discovery already covers `src/molly/`, and the repaired lock remains
unchanged. Legacy `ai4s_agent` packaging and runtime were not removed or
modified.

## Reused v1 ideas and boundaries

The implementation consulted the C2 disposition inventory and reviewed:

- `src/ai4s_agent/_utils.py` for UTC formatting and canonical/atomic JSON
  patterns;
- `src/ai4s_agent/attempt_publication.py` for fsync plus no-replace
  publication primitives; and
- `src/ai4s_agent/provenance/artifact_registry.py` for content-identity
  reasoning.

Only small, directly required ideas were reimplemented. No v1 Controller,
Replanner, Permission/Authorization/StartIntent, autonomy/authority, lease,
EvidenceGrant, promotion/adoption, or failure-recovery machinery was copied
or renamed into Core.

## Verification

Focused CORE-01 tests:

```text
tests/molly/test_core01_foundation.py: 13 passed
```

Relevant v1 primitive/storage regression tests:

```text
tests/test_attempt_publication.py tests/test_storage.py: 39 passed
```

The final verification record will additionally bind:

```text
git diff --check: PASS
python -m compileall -q src tests prototypes: PASS
uv lock --check: PASS (185 packages resolved)
focused CORE-01 + CORE-00 regression selection: 52 passed in 9.31s
repository PR Fast selection: pending until the final pre-push check
```

The production import boundary is covered by an AST test: no file under
`src/molly/` imports `ai4s_agent`, the C4 spike, `subprocess`, `socket`,
`urllib`, or `httpx`. The focused tests cover content identity, byte/digest
verification, no-replace and restart behavior, corruption/tampering,
traversal/symlink rejection, append ordering and malformed JSONL, bounded
lineage relations, closed validation contracts, and exact review binding.

## Known limitations and next milestone dependencies

CORE-01 intentionally provides local filesystem persistence only. It does not
provide cloud storage, a database, synchronization, garbage collection,
distributed coordination, autonomous recovery, or a review database. A
`ReviewRecord` and `ValidationResult` are immutable contract values; later
milestones may persist them through the same artifact/ledger boundaries.

CORE-02 still needs to define `RunRequest`, `ToolRegistry`, `ToolPolicy`,
`ApprovalRecord`, and the single `AgentLoop`/`RunEngine` that will record
actions through these contracts. No CORE-02 code was started here.

This report does not establish BR1 v2 parity. B2, B3, and B4 remain
unchanged/pending, and `core_cutover_ready` remains `false`. The immutable v1
freeze tag `molly-v1-pre-core-v2-20260829` and branch `legacy/molly-v1` remain
bound to `ae7892dbf8a6bfe85dd909056eadc2afecc40d9d`.
