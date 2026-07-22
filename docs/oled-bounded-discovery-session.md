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
published as a canonical no-replace revision whose predecessor digest binds
the prior revision; `session_state.json` is only a recoverable convenience
head over that immutable chain.

PR-AQ, initial PR-ARb, and every PR-AS child retain their existing
`gate_5_final_threshold` gate. The coordinator returns a waiting child ID and
does not approve it. `approve_oled_bounded_discovery_session_gate()` resumes
the exact existing child RunPlan and gate snapshot with an explicit actor.

The first PR-AS is direct/root. A later PR-AS receives the exact previous
PR-AU controller request, receipt, authorization, and report. Round two uses
the PR-ATb cumulative roster and previous-evaluation binding. The coordinator
copies neither PR-AU routing logic nor its budget arithmetic.

## Recovery and integrity

Every successful child stores the exact artifact-registry mapping and a digest
over every registered file. Before a child result is committed to session
state, its publication is exact-replayed from its upstream inputs. Later
consumption rechecks the stored registry and byte manifest.

Recovery rules are deliberately conservative:

- a fully registered child publication is exact-replayed and reused without
  adapter dispatch;
- a waiting child keeps its original RunPlan and gate snapshot;
- a child left `RUNNING` without a complete registered publication transitions
  the session to `RECOVERY_REQUIRED` and is never dispatched automatically;
- changed Registry mappings, bytes, source receipts, or replay results fail as
  `FAILED_INTEGRITY` before another child run is created.

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

PR-AW may add the user-facing session controls and result presentation. It
must consume these APIs rather than bypassing the coordinator.
