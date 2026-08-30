# Molly Core v2 CORE-01 production foundation

Status: `PASS — CORE-01A amendment`

Date: 2026-08-30

Repository: `searching42/Molly`

Branch: `codex/molly-core-v2-core-01-foundation`

Draft PR: [#65](https://github.com/searching42/Molly/pull/65)

Base commit: `93fdb08e924f116451bf6472af9bf85a3d473f24` (merged CORE-00
PR #64 on `origin/main`)

Reviewed starting HEAD: `7bf1b44bdc5a506c26db9f9779bbaba60aa220f0`

Implementation commits:

- `c23063c53287365cdb6845b46716dd8abdecc154` — initial CORE-01 foundation
- `dc9dc6facebd41c67bf67aac9a9853750b129838` — CORE-01A content/provenance separation fix

This milestone implements CORE-01 only. It does not implement CORE-02 or any
later milestone, BR1 v2 parity, remote/GPU execution, acquisition, document
parsing, OLED migration, UI, API routes, or default cutover.

## Requirements implemented

The new production namespace is `src/molly/`. It is standard-library-only and
has no import dependency on `ai4s_agent` or the non-production contract spike.

- `ArtifactRecord` is frozen content-level metadata only: exact
  `sha256:<digest>` identity, media type, optional schema metadata, byte size,
  and `stored_at` (the first publication time in this store). It contains no
  producer step, input artifact, or occurrence/source provenance fields.
- `ArtifactStore` publishes exact bytes under a SHA-256 content address with a
  deterministic `objects/<prefix>/<digest>` layout and immutable metadata.
  Publication uses a fsynced temporary file plus an atomic no-replace hard
  link. Existing bytes are verified before reuse; corrupt, incomplete,
  traversal, symlink, and contradictory intrinsic metadata states fail closed.
  Compatible repeated puts return the first content record without changing
  `stored_at` or erasing optional schema metadata.
- `RunLedger` appends canonical UTF-8 JSONL events. Each event carries the
  run/step/tool/artifact/profile/digest/seed/metadata fields needed by later
  execution work, plus a previous-event hash and its own hash. Reopen and
  inspection validate the complete chain and reject malformed or truncated
  records.
- `ArtifactLineage` records only `CONSUMED_BY`, `PRODUCED_BY`, `DERIVED_FROM`,
  and `SUPPORTED_BY` relations. Explicit `record_production()` calls retain
  each run/step occurrence and its input relationships, including when
  multiple occurrences share one content identity. It supports direct
  parent/producer/support inspection and JSONL restart reconstruction; it is
  not a causal graph or a scheduler.
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

## Content identity vs occurrence provenance

### Previous ambiguity

The initial CORE-01 shape placed `producer_step_id`, `input_artifact_ids`,
`provenance`, and `created_at` on `ArtifactRecord`, while storing one record per
content digest. Consequently, a second run that produced the same bytes could
appear to have been produced by the first run's step.

### Revised invariant

```text
CONTENT IDENTITY       = exact immutable bytes
OCCURRENCE PROVENANCE  = this run / step / inputs / production context
```

`artifact_id` remains `sha256:<exact-content-sha256>`. `ArtifactRecord` now
contains only intrinsic metadata and uses `stored_at` for the first publication
time in its local store. `ArtifactStore` never establishes producer identity.
Each occurrence is recorded explicitly through `ArtifactLineage.record_production()`
as `PRODUCED_BY` plus `DERIVED_FROM` relations, with optional non-secret
occurrence metadata. `SUPPORTED_BY` remains a relation rather than a
digest-keyed source field.

`ArtifactRecord` is therefore not authoritative for determining which run or
step produced an artifact occurrence. Future current-run binding must use
`RunLedger` events together with `ArtifactLineage` production relations.

### CORE-01A changes and regression evidence

- Changed `src/molly/core/artifacts.py`: removed occurrence fields from
  `ArtifactRecord` and `ArtifactStore.put()`, renamed `created_at` to
  `stored_at`, and added fail-closed media/schema conflict checks.
- Changed `src/molly/core/lineage.py`: added explicit occurrence recording and
  restart-safe identity hydration; `add_artifact()` now registers identity only
  and never invents production provenance.
- Changed `tests/molly/test_core01_foundation.py`: updated first-writer tests
  and added repeated-content, different-input, conflict, omission, and
  restart-append coverage.
- The mandatory parent_A/step_A and parent_B/step_B regression proves one
  content identity has two correct production relations and does not inherit
  step_A from the first publication.

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

Added/updated focused tests:

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

Focused CORE-01 tests before the CORE-01A fix:

```text
tests/molly/test_core01_foundation.py: 13 passed
```

CORE-01A focused tests:

```text
tests/molly/test_core01_foundation.py: 15 passed
```

Relevant v1 primitive/storage regression tests:

```text
tests/test_attempt_publication.py tests/test_storage.py: 39 passed
```

CORE-01A regression record:

```text
git diff --check: PASS
python -m compileall -q src tests prototypes: PASS
uv lock --check: PASS (185 packages resolved)
focused CORE-00 + CORE-01 regression selection: 54 passed in 1.91s
relevant v1 atomic publication/storage selection: 39 passed in 9.18s
PR Fast command: PYTHONPATH=src PYTHONHASHSEED=0 python -m pytest -q -m "(unit and not slow) or pr_fast" --durations=20
repository PR Fast selection: 1541 passed, 5661 deselected in 197.37s (0:03:17)
PR Fast run commit: `35e86ac` (documentation-bound report commit; no executable changes after the tested CORE-01A commit)
```

Remote verification before final push:

```text
base: main@93fdb08e924f116451bf6472af9bf85a3d473f24
tested head: codex/molly-core-v2-core-01-foundation@35e86ac
Draft PR: #65 (OPEN, Draft)
GitHub checks: to be re-triggered/verified after the final CORE-01A push; local PR Fast passed at the commit above
```

The production import boundary is covered by an AST test: no file under
`src/molly/` imports `ai4s_agent`, the C4 spike, `subprocess`, `socket`,
`urllib`, or `httpx`. The focused tests cover content identity, byte/digest
verification, no-replace and restart behavior, corruption/tampering,
traversal/symlink rejection, append ordering and malformed JSONL, bounded
lineage relations including repeated-content occurrences, closed validation
contracts, and exact review binding. No readiness gate was changed; C0-C7
remain PASS, B2-B4 remain pending/unchanged, and no CORE-02 code was started.

## Final integration after pre-CORE-02 remediation

PR #66 (`https://github.com/searching42/Molly/pull/66`) was merged into
`main` before this integration. The merged main SHA is
`504657a760413386757066071bc0a6f800897238`.

The CORE-01 branch was integrated from pre-integration HEAD
`ca12ca92606d933d4006ed8f4b2451d6ef3fa032`; the resulting implementation
merge commit is `d09911470ae51039e1dc8b6aa2e4bfa7415f92cc`.

The pre-CORE-02 remediation changes legacy/CI behavior: it narrows the
repository privacy scanner's generated-lockfile handling and separates exact
human remote approval from autonomous remote-runtime lease enforcement. It
does not alter the accepted CORE-01 content/provenance model. In particular,
`ArtifactRecord` remains content-intrinsic, while occurrence provenance stays
in `ArtifactLineage` and `RunLedger`.

B2 = `PENDING`; B3 = `PENDING`; B4 = `PENDING`.
`core_cutover_ready` remains `false`.

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
