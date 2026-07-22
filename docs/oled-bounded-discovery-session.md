# OLED bounded discovery session coordinator

PR-AV turns the existing executable OLED tasks into one durable, bounded
session without introducing another scientific decision layer.

The coordinator calls only `RunPlanExecutor`. It never calls a scientific
adapter directly and does not recompute property constraints, Pareto status,
diversity, candidate shortfall, generation authorization, or loop budgets.
Those decisions remain owned by PR-AQ, PR-ARb, PR-AS, PR-AT, PR-ARb v2, and
PR-AU.

## Persistent layout

Each project stores sessions under:

```text
projects/<project-id>/bounded-discovery-sessions/<session-id>/
  session_spec.json
  session_state.json
  state_000000.json               # immutable transition history
  state_000001.json
  session_result.json              # terminal sessions only
  controller_request_01.json       # created when needed
  controller_request_02.json
  generation_roster_02.json
```

Scientific publications remain in deterministic child runs under the normal
project run store. Child IDs include the session ID and action, for example:

```text
<session-id>-screening
<session-id>-initial-decision
<session-id>-generation-01
<session-id>-evaluation-01
<session-id>-candidate-decision-01
<session-id>-controller-01
```

The immutable SessionSpec binds the PR-AO/dataset/Registry anchors, PR-AQ
constraints, the complete PR-ARb selection request, inverse-design transport,
deterministic seed policy, and the original PR-AU limits. Changing any field
creates a different session ID. The identity also includes exact byte/directory
manifests for every external anchor, REINVENT4 config, local generator output,
known-hosts file, and optional cost manifest; reusing a path with changed bytes
cannot silently reuse the session.

## Advancement and gates

`advance_oled_bounded_discovery_session()` requires the caller's expected
revision. The session lock and revision comparison form a compare-and-swap:
two callers holding the same revision cannot both advance the session. One
call commits at most one session-state transition. Every transition is first
written and fsynced to an invocation-owned temporary inode, then atomically
hard-linked under its canonical no-replace revision name and followed by a
parent-directory fsync. A crash before the link leaves the preceding revision
authoritative. `session_state.json` is only a disposable convenience head: a
missing, malformed, or stale copy is atomically rebuilt from the final valid
immutable revision.

The public state contract has six statuses: `ACTIVE`, `WAITING_USER`,
`COMPLETED_TOP_N`, `STOPPED_BOUNDED_NO_SOLUTION`, `RECOVERY_REQUIRED`, and
`FAILED`. `current_step` is one of `screening`, `initial_decision`,
`generation`, `evaluation`, `candidate_decision`, or `controller`. Stage
completion is derived from verified child facts; it is not persisted as a
second family of top-level statuses.

PR-AQ, initial PR-ARb, and every PR-AS child retain their existing
`gate_5_final_threshold` gate. The coordinator returns a waiting child ID and
does not approve it. `approve_oled_bounded_discovery_session_gate()` resumes
the exact existing child RunPlan and gate snapshot with an explicit actor.

The first PR-AS is direct/root. A later PR-AS receives the exact previous
PR-AU controller request, receipt, authorization, and report. Round two uses
the PR-ATb cumulative roster and previous-evaluation binding. The coordinator
copies neither PR-AU routing logic nor its budget arithmetic.

## Recovery and integrity

Every child label is rebound to its deterministic run and task. Waiting
children are checked against their exact `StageState`, RunPlan, execution
snapshot, and later gate decision. Successful children are checked against
their exact registry, file manifest, and publication replay. Terminal state is
accepted only when `session_result.json` can be rebuilt byte-for-byte from its
verified source child and controller. These checks run on inspect and advance,
so a self-consistent re-signed state chain is not a trust anchor.

Recovery rules are deliberately conservative:

- a fully registered child publication is exact-replayed and reused without
  adapter dispatch;
- a waiting child keeps its original RunPlan and gate snapshot;
- if a gated child reached `SUCCEEDED` before its session revision was
  published, the coordinator verifies its execution record, Registry manifest,
  publication replay, original snapshot, and gate decision, then adopts it
  without calling resume or the adapter again;
- a waiting child whose executor fact already reached `FAILED` is reconciled
  to session `FAILED`, while `RUNNING` becomes `RECOVERY_REQUIRED`;
- a child left `RUNNING` without a complete registered publication transitions
  the session to `RECOVERY_REQUIRED` and is never dispatched automatically;
- changed Registry mappings, bytes, source receipts, or replay results fail as
  `FAILED` with an integrity failure code before another child run is created.

This is an at-most-once recovery contract. It does not claim that an
unregistered remote REINVENT4 side effect can be reconciled automatically.

## Terminal results

`session_result.json` is immutable and references, rather than copies, the
final scientific publication.

- Registry-only complete Top-N: `COMPLETED_TOP_N`, source `pr_arb_v1`.
- Generated complete Top-N: `COMPLETED_TOP_N`, source `pr_arb_v2`.
- PR-AU bounded stop or a non-supply selection-policy stop:
  `STOPPED_BOUNDED_NO_SOLUTION` with the exact reason.

Computational validation, Registry mutation, candidate adjudication, and
manual accept/defer/reject are outside this coordinator.

## Public Python entry points

```python
create_oled_bounded_discovery_session(...)
inspect_oled_bounded_discovery_session(...)
advance_oled_bounded_discovery_session(..., expected_revision=revision)
approve_oled_bounded_discovery_session_gate(
    ...,
    expected_revision=revision,
    actor="reviewer",
)
```

## PR-AW control plane and result presentation

PR-AW adds a narrow web/API control plane around the coordinator. It does not
add another scientific task, candidate source, decision policy, or remote
transport branch.

The session endpoints are project scoped:

```text
GET  /api/projects/<project>/oled-bounded-sessions
POST /api/projects/<project>/oled-bounded-sessions
GET  /api/projects/<project>/oled-bounded-sessions/<session>
POST /api/projects/<project>/oled-bounded-sessions/<session>/actions/advance
POST /api/projects/<project>/oled-bounded-sessions/<session>/actions/approve
GET  /api/projects/<project>/oled-bounded-session-actions/<action>
```

Creation and inspection validate the exact PR-AV facts before returning a
path-redacted presentation. Advance and approval requests require the caller's
`expected_revision` and return `202` with a pollable action instead of holding
the HTTP request open while a child task runs. Only the public PR-AV create,
advance, and approval entry points can mutate a session.

Each action directory separates an immutable, no-replace `request.json` from
mutable `action.json` scheduling state. The request freezes project, session,
operation, expected revision, and approval identity at enqueue time. The
same-process worker receives those frozen bytes directly and requires the
on-disk request and initial state to remain byte-for-byte identical before it
calls PR-AV. Mutable state cannot select or replace a scientific operation.
Leading or trailing whitespace in a project ID is rejected before any session
read; the same unmodified ID is used for view validation, action storage,
duplicate detection, execution, and polling.

The action service has one worker. An action left `QUEUED` or `RUNNING` by a
previous process is reported as `RECOVERY_REQUIRED` and is never automatically
replayed. It blocks another transition while its expected session revision
remains current. If independently verified PR-AV recovery later advances
beyond that revision, the obsolete control record no longer locks the session.
Action state is not a scientific result trust anchor: a successful poll always
rebuilds the current session view through PR-AV external-fact validation.

`/oled-bounded-sessions` exposes status, current step, exact gate identity,
PR-AU limits and observed use, child progress, and explainable terminal Top-N.
Candidate strings are escaped before HTML insertion. The page contains no
manual accept/defer/reject action and explicitly states that the output is a
recommendation, not experimental or computational validation.
