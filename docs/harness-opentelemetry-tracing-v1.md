# Harness OpenTelemetry Tracing v1

Status: PR-BN optional observability design freeze for `M3H-010` prerequisite
seam. Tracing is not execution authority.

## Boundary

The Controller and its local/remote seams call only the internal
`HarnessTracer` interface. Its default implementation is `NoopHarnessTracer`
and has no third-party dependency. OpenTelemetry SDK and OTLP/HTTP exporter
imports are lazy and available only through the optional `tracing` install
extra. No worker-side package or propagation change is part of PR-BN.

Exporter initialization, span creation, attribute/event recording, shutdown,
and export are fail-open. Every exception is contained by the tracing adapter
and may increment a bounded local diagnostic counter; it cannot change a
Controller decision, HTTP status, receipt, StageState, Artifact Registry,
remote lifecycle state, retry/recovery choice, or process exit status.

## Span hierarchy

```text
controller.execution                         (one Controller execution)
  controller.advance                         (one strict advance request)
    controller.action                        (one selected bounded action)
      executor.local_task                    (one local task, when selected)
      remote.prepare | remote.dispatch
      remote.refresh | remote.recover         (one lifecycle call, when selected)
```

Trace and span IDs are excluded from every semantic digest, idempotency key,
immutable request, decision, receipt, Gate decision, remote request,
publication, Registry record, StageState transition, and trajectory identity.
The Controller never reads trace state to make a business decision.

## Safe attributes and events

The adapter accepts a closed allowlist. Values have fixed types, bounded
length/count, and enum validation where applicable.

| Attribute/event | Safety rule |
| --- | --- |
| schema/controller policy version | Fixed public enum/string |
| controller execution/action IDs | Opaque server-generated identifiers |
| run/task/slot IDs | Validated logical identifiers only |
| task index/attempt | Bounded non-negative integers |
| route/action/outcome/status | Fixed enums |
| Gate ID | Registered logical identifier |
| remote task/profile type | Registered logical identifier |
| error family/code | Fixed safe classifier, never exception text |
| duration/retry/output count | Bounded numeric observation |
| authority/decision/receipt digests | Opaque SHA-256 digests only |

Attribute keys not on the allowlist are rejected. Collections are capped;
text is normalized and capped; events use a closed name roster. The adapter
does not accept arbitrary business dictionaries for recursive export.

Forbidden telemetry includes absolute or relative filesystem locations,
host/IP/SSH aliases, usernames/emails, connection locators, commands/argv,
environment names or values, credentials/tokens/keys, headers/cookies,
request or response bodies, raw exception messages, raw stdout/stderr, remote
worker payloads, artifact contents, document text, dataset rows, molecule
strings, model weights, prompts/responses, authorization/Gate notes, and
private reasoning.

## Configuration and dependency policy

The base installation continues to use Noop tracing. Enabling OpenTelemetry
requires the optional, mutually compatible API, SDK, and OTLP/HTTP packages.
Runtime configuration is read by a server-only factory; public Controller
routes cannot choose an exporter, endpoint, headers, resource attributes, or
sampling policy. Endpoint credentials remain in private deployment
configuration and are never reflected into spans or API responses.

If configuration is absent or invalid, imports are unavailable, the exporter
cannot initialize, or export later fails, the factory returns or degrades to a
safe no-op implementation. No buffered span is replay authority after a crash.

## Tests and non-authoritative evidence

Tests cover no-op default behavior, lazy missing dependencies, allowlist and
bounds, forbidden-value rejection, exporter construction/export failures,
span nesting, digest invariance with tracing enabled/disabled, and failure
non-interference with local execution, remote dispatch/recovery, Gate waiting,
receipt publication, and API status.

Operational dashboards may use spans to find latency or exporter health. A
span that says `SUCCEEDED`, or the absence/duplication/reordering of spans,
proves nothing about execution. Inspectors label all tracing facts
`OBSERVATIONAL`; authoritative status remains in immutable controller
artifacts, StageState, Artifact Registry, and exact remote publications.
