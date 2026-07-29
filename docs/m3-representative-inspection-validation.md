# M3 representative inspection validation evidence contract

## Scope and claim boundary

PR-BI validates the existing PR-BD → PR-BF → PR-BG → PR-BH observer chain. It
does not add or change a scientific, observer, attribution, or inspection
contract. The evidence runner creates disposable representative Sessions with
production code, publishes the three observer artifacts, and invokes the
project-scoped `scientific_agent_trajectory_inspection.v1` GET route from two
new Python interpreters.

The package is trajectory/audit runtime evidence only. It is not experimental
validation, high-fidelity-computation validation, remote-backend evidence,
attribution-accuracy benchmark evidence, or an M4 result. Representative fault
injection is never described as captured real runtime.

## Evidence version and case roster

The public manifest version is:

```text
m3_representative_inspection_validation.v1
```

The frozen v1 roster is:

| Case | Source class | Required observation |
|---|---|---|
| `single_round_success` | `representative_local_runtime` | production one-round terminal Session, no failure |
| `multi_round_success` | `representative_local_runtime` | production two-round terminal Session, no failure |
| `known_hosts_propagation` | `representative_fault_injection` | transport first cause without infrastructure disclosure |
| `history_truncation` | `representative_fault_injection` | re-signed truncated observer source fails closed |
| `duplicate_dispatch` | `representative_fault_injection` | persisted duplicate dispatch distinguished from replay/adoption |
| `stale_state` | `representative_fault_injection` | telemetry remains non-authoritative and cannot change scientific result |
| `multiple_equal_first_cause_candidates` | `representative_fault_injection` | deterministic ambiguity with no primary cause |
| `causal_link_not_proven` | `representative_fault_injection` | independent later stop remains undetermined without a persisted link |

Every case remains in the manifest even when the current source contract cannot
represent its required evidence. Such a case is `blocked`, never silently
omitted, downgraded, or declared passed from a test-only projection.

## Production and fresh-process protocol

Process A creates isolated inputs, runs the production bounded-session
coordinator, and publishes PR-BD, PR-BF, and PR-BG. It records a complete
scientific and observer source snapshot.

Processes B and C are distinct Python interpreters with different
`PYTHONHASHSEED` values. Each creates a disposable Flask application, registers
the production bounded-session routes, and invokes exactly:

```text
GET /api/projects/<project_id>/oled-bounded-sessions/<session_id>/trajectory-inspect
```

Only the project ID, Session ID, and three publication IDs enter the request.
The response is serialized as canonical JSON and the two response byte streams
must match. A second source snapshot proves that inspection created no durable
file and changed no scientific or observer byte.

The formal runner does not import `tests`, a pytest fixture, or a test case
builder. Tests may invoke and validate the runner, but are not an evidence
dependency.

## Private locator and public package

The fresh-process locator has mode `0600`, lives only in the invocation-owned
private output, is not part of any content identity, and is deleted by default.
Raw Session trees and observer publications remain outside Git. The public
package contains only canonical inspection JSON, fixed error responses, safe
IDs, SHA-256 digests, expected/observed comparisons, and review metadata.

All writes are no-replace at package scope. JSON is UTF-8, sorted-key canonical,
uses fixed separators and a single trailing newline, rejects duplicate keys,
and rejects NaN and Infinity.

## Privacy policy

Public evidence and process output are scanned for absolute paths, home or
workspace locators, hostnames, usernames, email-like accounts, SSH aliases,
known-hosts paths or bytes, remote repository paths, interpreter paths, raw
exceptions, commands, environment variables, credentials, cookies, signed
URLs, and private scientific text. The semantic canary values specified by
PR-BI are rejected even when placed in an otherwise allowed field.

The history-truncation case commits only the fixed 409 response and digests;
it never commits the truncated bytes. Application exception logging is disabled
inside the disposable evidence subprocess so verifier internals cannot enter
public stdout or stderr.

## Human-review and M3 V gate

Machine evidence and owner review are separate states. The generated checklist
is intentionally blank, `human_review_status` is `pending`, and Codex cannot
self-approve. A repository-owner approval must identify the reviewed commit and
decision before any M3 task or gate may change from `I/T/—` to `I/T/V`.

The verifier rejects a human approval or M3 V claim that lacks a matching owner
review record. While any machine case is failed or blocked, or owner review is
pending, PR-BI remains Draft, M3 remains `I/T/—`, and M4 stays locked.

## Current v1 evidence gap

The exact-replayed PR-BD projection currently emits a failed child as only
`failed` or `integrity_failed`. It does not persist the transport reason,
distinct duplicate-dispatch proof, multiple family reasons at one revision, or
a recovered-failure causal link. Consequently those four requested semantics
cannot be produced through the PR-BH GET route without changing PR-BD–PR-BH,
weakening exact replay, or substituting test-only bytes.

PR-BI records these cases as explicit machine blockers. It does not modify the
projection or attribution contracts merely to manufacture acceptance evidence.
Resolving that source-evidence gap requires repository-owner scope direction in
a separate contract PR before PR-BI can become machine-complete.

## Rollback

The runner, verifier, tests, redacted package, and roadmap status can be removed
with an ordinary revert. No Session, storage, Artifact Registry, scientific
publication, observer publication, or API migration is required.
