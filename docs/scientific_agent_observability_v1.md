# Privacy-safe Scientific Agent Harness observability v1

This document freezes the PR-BQ2 / `M3H-015` observability seam. It does not
define execution or acceptance authority. OpenTelemetry and LangSmith are
non-authoritative telemetry only.

## Authority boundary

Molly's immutable proposal, Permission, authorization, start-intent,
Controller, Gate, StageState, Artifact Registry, verified-publication, and
receipt chains remain authoritative. `AgentRunInspection v1` is the single
current-verified, read-only projection of those sources. Telemetry consumes
privacy-safe bindings from authority objects or that projection; it is never
read back to create, repair, promote, or verify Molly state.

Every correlation and health projection fixes
`telemetry_authoritative=false`. Trace status, span status, vendor run state,
export success, dashboards, and in-memory health are not verifier evidence.
Telemetry identifiers and configuration never enter Molly authority bytes,
digests, idempotency checkpoints, remote requests, or inspection digests.

Telemetry failure does not change execution, authorization,
Gate, StageState, Registry, publication, verifier, recovery,
or scientific-success state.

## Canonical correlation contract

`HarnessTelemetryCorrelationContext` has schema version
`harness_telemetry_correlation.v1`. Both vendor adapters use the shared Harness
tracer validation/export boundary and the same `molly.*` namespace. Present
fields are derived from exact authority or `AgentRunInspection`; absent fields
remain absent and are never inferred from vendor state.

The frozen core attributes are:

- `molly.schema_version`, `molly.project_id`, `molly.run_id`;
- inspection, proposal, semantic-plan, Permission, authorization, and
  start-intent IDs/digests when available;
- Controller execution ID/digest/revision and task, slot, and execution route;
- Execution Agent and Replanner proposal/application/revision bindings;
- Gate and verified-publication bindings;
- `molly.operation`, `molly.component`, `molly.phase`,
  `molly.authority_class`, and `molly.telemetry_authoritative`.

Attributes are capped at 48 entries. Labels are canonical bounded identifiers,
digests are lowercase `sha256:` values, and counters are bounded non-negative
integers. Vendor trace IDs are deliberately unnecessary for correlation.

## OpenTelemetry coverage

The optional adapter extends the existing `HarnessTracer` seam. Stable exported
span names cover plan proposal and LLM calls, Permission evaluation,
authorization/start intent, Controller create/inspect/advance, local execution,
remote prepare/wait-for-approval/dispatch/refresh/recover/adopt, Execution Agent
proposal/LLM/application, Replanner feedback/LLM/revision application, and
unified run-inspection reads.

Only allowlisted IDs, digests, enums, fixed reason codes, bounded counts, and
`telemetry_authoritative=false` are exported. Error events contain a fixed
reason code and sanitized exception type code; adapters never call an API that
records the raw exception object. Batch processing is bounded and has finite
export and shutdown timeouts. Molly does not add a durable telemetry queue.

The tracer provider receives a directly constructed, frozen Resource containing
only `service.name`; it does not run default or environment resource detectors.
A Molly-owned bounded processor sits in front of a privacy-safe exporter
wrapper. Delegate exceptions and failure results become fixed health codes,
queue pressure increments the drop counter, and vendor-created log records are
redacted to a fixed message before handler serialization.

OpenTelemetry mode is private server configuration:

- `disabled` (default);
- `otlp_http`;
- `otlp_grpc`.

The SDK and exporter imports are lazy. Collector endpoint and credentials are
read only by the vendor SDK from private process configuration; Molly never
copies them into project files, responses, logs, attributes, or inspections.

## LangSmith coverage and content modes

LangSmith observes the shared LLM spans for Planner, Execution Agent, and
Replanner. It does not replace their provider checkpoints, and one observation
never causes another provider call or retry. Harness/Controller/Executor
lifecycle telemetry remains available through OpenTelemetry rather than being
represented only as LLM runs.

Supported server modes are `disabled`, `metadata_only`, and an explicitly
policy-gated `structured_content`. The installation default is disabled; when
LangSmith is enabled, `metadata_only` is the normal mode. Metadata-only runs use
empty vendor inputs and export only schema versions, request/response digests,
safe provider classifications when available, latency/count classifications,
fixed outcomes, and canonical correlation fields. Prompt text, response text,
paper text, private feedback, conversation, and reasoning are never accepted by
the adapter.

The adapter generates the UUID before `Client.create_run(id=...)` and uses that
same local UUID for `update_run`, matching the SDK's `create_run() -> None`
contract. The real client is forced to `omit_traced_runtime_info=True`; SDK
input, output, and metadata sanitizers form a second send-boundary allowlist,
and synchronous calls use a finite timeout so adapter-level failures remain
observable and fail open.

`structured_content` requires the server-owned allow flag. Without it, config
deterministically degrades to `metadata_only` without blocking the business
call. In v1 the adapter still has no raw-text ingestion surface: a future typed
structured projection must additionally prove external-LLM authorization and
pass the existing prose/schema privacy validators before any content can be
emitted. Arbitrary strings remain excluded.

## Privacy allowlist

`harness_telemetry_privacy_policy.v1` is shared by both adapters. It rejects
paths, hostnames/IP addresses, user or account locators, SSH material,
endpoints, credentials and authorization headers, command/argv/shell payloads,
stdout/stderr, raw exceptions, artifact or PDF contents, provider prompts and
responses, conversation history, private feedback, and chain-of-thought.

Free text is not a telemetry attribute class. Scientific OLED terms such as
`host material`, `host–dopant`, `host-dopant pair`, `emitter host`, and
`doping host` remain valid scientific inputs; they are not treated as
infrastructure hostnames. They still are not uploaded as metadata-only
telemetry content.

## Fail-open and health semantics

Configuration parse errors, missing optional packages, initialization failure,
span/event failure, exporter timeout or queue pressure, LangSmith create/end
failure, and shutdown/flush failure all degrade independently to no-op or event
drop. Vendor exceptions are absorbed inside the adapter. The original business
return value or exception is preserved, with no retry, dispatch, Gate update,
recovery, cancellation, or authority rollback.

`HarnessTelemetryHealthSnapshot v1` is a process-local diagnostic containing
enabled/available flags, fixed last-result codes, and bounded drop/failure
counters. It is non-authoritative, is not persisted, may reset on restart, and
does not affect API status or run outcome. V1 exposes it only through the
internal app extension; no dashboard or public health API is added.

## Remote execution

Cross-process correlation uses the existing privacy-safe project/run,
proposal/controller, task/slot, request, and publication bindings. No
`traceparent`, span ID, exporter field, or vendor project ID is inserted into a
signed remote request. The fixed `molly-worker` protocol and all remote authority
digests are unchanged. Telemetry availability is never a dispatch prerequisite.

## Optional installation and rollback

The default package remains tracing-disabled and importable without either SDK.
The `observability` optional extra installs both vendor clients; the existing
`tracing` extra remains compatible. Disable both server modes to roll back to
`NoopHarnessTracer` without migrating or rewriting project data. Existing
manual APIs, Controller, local/remote execution, Execution Agent, Replanner,
and inspection contracts remain compatible.

## Non-goals and handoff

This PR does not deploy a real collector or LangSmith project, add a dashboard,
build the BQ3 UI, implement BR1/BR2 canaries, change worker/distributed trace
protocols, add retry/recovery/cancel/Gate automation, or create a second event
ledger. It does not create runtime acceptance or any Gate `V`.

BQ3 may consume the stable inspection boundary but must not treat telemetry as
state. BR3 retains final UI-driven runtime, restart, exact replay, privacy, and
adversarial acceptance evidence. No M3.5 completion or Molly v1 completion is
claimed by this observability seam.
