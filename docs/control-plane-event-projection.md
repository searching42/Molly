# Control-plane event projection

Stage 4 adds a read-only observer channel to the existing bounded-discovery
control plane. It does not add a job manager, executor, queue, recovery state
machine, or scientific source of truth.

## Authority boundary

The following existing records remain authoritative:

- the immutable Session revision chain and mutable replayable Session head;
- validated action telemetry and recovery results;
- child `StageState`, gate decisions, and immutable execution/publication records;
- the artifact registry after the existing external-fact verifier accepts it.

`ControlPlaneEventProjector` first builds the existing exact-replayed Session
view. It then observes committed revisions and validated action telemetry. Its
JSONL journal under `runs/control-plane-event-projections/` is a UI replay log,
not a second lifecycle database. Deleting it loses observer history but cannot
change, retry, approve, complete, or invalidate a Session.

No execution path may consume a projected event as proof of task success. SSE
disconnects do not cancel work, and SSE requests expose no mutation method.
If a child `StageState` is ahead of the immutable Session revision, observation
emits `session.reconciliation_available`; it does not adopt the child or publish
a Session revision. Adoption remains an explicit existing recovery action.
Projection also derives controller requests and cumulative generation rosters
in memory only. Existing helper files are exact-verified, while missing files
are never materialized by snapshot or SSE reads.

## Durable events

Durable observations receive contiguous integer event IDs and may be replayed:

- `session.created`
- `session.stage_changed`
- `session.reconciliation_available`
- `action.queued`, `action.running`, `action.succeeded`, `action.failed`
- `action.recovered`, `action.recovery_required`
- `gate.waiting`, `gate.approved`
- `artifact.registered`
- `session.completed`, `session.failed`, `session.recovery_required`

The journal uses newline as its commit marker. A partial final record is
discarded and regenerated from current authority; a corrupt newline-terminated
record fails closed. Event IDs are observer cursors only and have no Session
revision or approval meaning.

## Ephemeral deltas

`heartbeat` is emitted without an SSE `id`. Future `llm.text_delta`, parser
progress, and bounded log-tail messages must follow the same rule: they serve
the current connection, are not appended to the durable journal, and do not
consume a durable cursor.

## HTTP API

```text
GET /api/projects/<project_id>/oled-bounded-sessions/<session_id>/event-projection
GET /api/projects/<project_id>/oled-bounded-sessions/<session_id>/event-projection/events
```

The JSON endpoint returns the exact current snapshot, durable events after the
requested cursor, the latest cursor, and an explicit authority declaration.
Pass `after=<integer>` or `Last-Event-ID: <integer>` to resume.

The SSE endpoint sends:

1. a snapshot event without an ID;
2. durable events after the cursor, each with its durable ID;
3. heartbeat events without IDs;
4. newly observed durable events until the client disconnects.

The local Flask server must run with request concurrency enabled so a long-lived
SSE observer cannot block approval, cancellation, or other control requests.
SSE remains an observer connection even when it is the longest-lived request.

If the requested cursor is ahead of the available journal, the server rejects
it and the client must reload a fresh snapshot. The journal is not compacted in
Stage 4, so a valid previously issued cursor does not expire during normal use.
