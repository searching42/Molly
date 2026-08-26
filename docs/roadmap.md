# Molly public roadmap

> Document status: Active public roadmap
>
> Public repository baseline: `main`
>
> Current major milestone: M3.5 — Scientific Agent Harness integration and runtime closure
>
> Current focus: M3.5-AUT-EXECUTION-V2 — Execution Agent v2
>
> Local `todo.md` is intentionally Git-ignored working context. It may be used for private scratch planning, but it is not public repository authority.

`docs/roadmap.md` 是公开仓库中路线图范围、里程碑状态、优先级、验收边界和执行顺序的唯一规范性来源。Topic documents may define durable implementation contracts and acceptance procedures, but they must not maintain a competing current-status table or decision log.

---

## 1. Status and evidence semantics

Public roadmap status separates evidence maturity from work-management state.

Evidence maturity:

- `I`: implemented in the repository.
- `T`: covered by relevant automated tests.
- `V`: validated by a representative runtime, exact replay, benchmark, or reviewed acceptance evidence.

Work state:

- `READY`: prerequisites are satisfied and the work may start.
- `QUEUED`: the work is planned and ordered, but its prerequisites are not yet satisfied.
- `IN_PROGRESS`: the work is actively being implemented or accepted.
- `BLOCKED`: a concrete technical or external blocker prevents progress.
- `DEFERRED`: intentionally postponed; not a blocker.
- `DONE`: the defined scope has met its Definition of Done.

`I/T/—` never implies runtime validation. A schema, unit test, CI pass, or synthetic fixture alone must not be promoted to `V`.

### Roadmap checklist convention

- [x] 表示该 roadmap item 的当前 Definition of Done 已满足。
- [ ] 表示尚未完成。
- 未完成项必须同时标记 work state：`READY / QUEUED / IN_PROGRESS / BLOCKED / DEFERRED`。
- 已完成项标记 `DONE`。
- Evidence maturity 独立记录为 `I/T/V`、`I/T/—` 等。
- Checkbox 只表达 roadmap item 是否完成，不能代替 runtime evidence。

The checklist is a compact maintenance interface for the active queue. It does not turn this document into a daily development log, and no item may use ambiguous states such as `almost done`, `mostly done`, or `90%`.

---

## 2. Trusted execution baseline

The public repository already contains the core execution and audit substrate needed for a bounded scientific Agent:

- immutable execution requests and execution-snapshot binding;
- explicit human gates and exact authorization boundaries;
- `RunPlanExecutor` as the local execution authority;
- `RemoteExecutionService` and the fixed worker protocol for remote dispatch;
- artifact registration, immutable publication, exact replay, recovery, and reconciliation;
- observer-only trajectory projection, deterministic audit metrics, failure attribution, and read-only inspection;
- privacy-safe OpenTelemetry and LangSmith observability as non-authoritative telemetry.

These mechanisms establish execution and provenance guarantees. They do not by themselves prove that generated OLED candidates are scientifically valid or experimentally successful.

M3 trajectory integrity, deterministic auditing, representative failure attribution, and read-only inspection are considered complete. M3.5 is the active public milestone for integration and runtime closure.

---

## 3. M3.5 — Scientific Agent Harness

### Goal

M3.5 connects LLM planning and bounded tool selection to the existing deterministic execution authorities without creating a second state machine.

The authority chain is:

```text
LLM Planner / Execution Agent / Replanner
                  ↓ structured proposal
trusted user review and immutable authorization
                  ↓
Scientific Agent Harness + deterministic Permission Engine
                  ↓ validated dispatch
RunPlanExecutor / RemoteExecutionService / fixed worker protocol
                  ↓
Verifier / output contract / Artifact Registry / exact replay
                  ↓ verified observation
LLM continues, stops, or proposes a new revision
```

### Frozen boundaries

- LLMs may propose plans and select only server-exposed bounded tools; they do not own shell, SSH, arbitrary filesystem paths, worker commands, Gate state, or execution status.
- User authorization and semantic scientific confirmation remain distinct from LLM output and from operational telemetry.
- Controller/Harness decisions must remain bound to current permission, plan, budget, profile, artifact lineage, and Gate state.
- Executor/RemoteExecutionService remain the only execution path; Verifier/publication/Artifact Registry remain the success authority.
- OpenTelemetry and LangSmith are observer-only. Exporter failure or missing telemetry must not alter authoritative bytes or execution outcomes.

### Current execution spine

The post-BR1 execution order is frozen as:

```text
M3.5-BR1 — Conversation-driven real BR1 acceptance                    DONE
        ↓
M3.5-AUT-POLICY — Autonomy action classification
        ↓
M3.5-AUT-L1 — Bounded auto-continuation
        ↓
M3.5-AUT-L2 — Bounded replanning
        ↓
M3.5-AUT-ACCEPT — L1/L2 adversarial and restart acceptance
        ↓
M3.5-BR2-RUNTIME — Real MinerU runtime closure
        ↓
M3.5-BR2-MAPPING — Contextual mapping and evidence binding
        ↓
M3.5-BR2-CONVERSATION-IMPL — Conversation-driven BR2 implementation and crash-safe publication
        ↓
M3.5-AUT-AUTH — Grant-based autonomy authority model
        ↓
M3.5-AUT-AUTH-L2 — Authority-aware L2 provenance and application
        ↓
M3.5-AUT-FASTPATH — Minimal deterministic-successor refactor
        ↓
M3.5-AUT-EXECUTION-V2 — Execution Agent v2
        ↓
M3.5-AUT-FAILURE-RECOVERY — Failure taxonomy and bounded recovery
        ↓
M3.5-AUT-FEEDBACK — Durable structured feedback and EvidenceGrant
        ↓
M3.5-AUT-LEASE — Active-time accounting and autonomy lease
        ↓
M3.5-UI — Minimal unified UI
        ↓
M3.5-V1-ACCEPT — Molly v1 final representative acceptance
```

`M3.5-BR2-ACCEPT` remains a separate `READY` fresh-real-acceptance boundary after
the implementation item above. Its pending rerun does not block the autonomy
refactor queue, but it remains a prerequisite for the later BR2 acceptance gate
and the unified UI.

The new P0 is to give the already-verified BR1 chain bounded autonomy before adding new scientific tools. PR #43 proved that the execution substrate can work on the real BR1 path; the next risk is whether an Agent can progress inside the existing authority envelope. L1/L2 are also the foundation for BR2 and for later long-horizon Agent trajectory/error-propagation research. Autonomy must not introduce a second Controller or state machine.

### Active execution queue

The following queue is the single active M3.5/v1 checklist. Target PR numbers are planning references only; if GitHub numbering conflicts, preserve the stable item ID and update the target PR.

- [x] **M3.5-BR1 — Conversation-driven real BR1 acceptance**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Definition of Done: the sole eligible owner-approved bundle is resolved from the natural-language conversation front door; confirmation, fresh current-run model training, real generation, current-model prediction, deterministic final evaluation, verified Computational Top-N, and `scientific_result.available` complete with privacy-safe projection and exact replay.
  - Evidence: PR #43, merge commit `7264a78eeb320fa00e8d9acbd9e556bfcf3e8360`, [checked-in BR1 acceptance evidence](evidence/br1-conversation-real-acceptance-v1/README.md), [acceptance manifest](evidence/br1-conversation-real-acceptance-v1/acceptance_manifest.json), [result summary](evidence/br1-conversation-real-acceptance-v1/result_summary.json), and [restart/replay summary](evidence/br1-conversation-real-acceptance-v1/restart_replay_summary.json).

- [x] **M3.5-AUT-POLICY — Autonomy action classification**
  - State: `DONE`
  - Evidence: `I/T/—`
  - Dependency: after `M3.5-BR1`.
  - Closed by: PR #45.
  - Definition of Done: the typed `AUTO_CONTINUE` / `REQUIRE_HUMAN` / `PROHIBITED` policy has an explicit exhaustive Controller-action roster, fail-closed unknown-action behavior, exact execution/inspection bindings, non-executable decisions, and a deterministic materiality handoff covered by contract tests. This closes policy implementation and test evidence only; it does not enable runtime continuation.

- [x] **M3.5-AUT-L1 — Bounded auto-continuation runtime**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Dependency: after `M3.5-AUT-POLICY`.
  - Target PR: #46.
  - Closed by: PR #46.
  - Definition of Done: an approved plan may automatically consume only already-valid authority for bounded continuation; every continuation recomputes the current PR #45 policy decision; cumulative transition, LLM-call, dispatch, wall-clock, task-graph, and resource boundaries are enforced from durable evidence; pause/tick continuation remains finite; every human, uncertain, stale, or prohibited condition fails closed before an LLM call or effect. Representative runtime, restart, replay, budget, and adversarial evidence is recorded by PR #48; this remains a control-plane claim and does not rerun BR1 scientific work.
  - Evidence: [Autonomy L1/L2 acceptance manifest](evidence/autonomy-l1-l2-acceptance-v1/acceptance_manifest.json), [scenario matrix](evidence/autonomy-l1-l2-acceptance-v1/scenario_matrix.json), and [restart/replay summary](evidence/autonomy-l1-l2-acceptance-v1/restart_replay_summary.json).

- [x] **M3.5-AUT-L2 — Bounded replanning and materiality boundary**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Dependency: after `M3.5-AUT-L1`.
  - Target PR: #47.
  - Closed by: PR #47.
  - Definition of Done: the explicit server-derived failure replan operation accepts only the exact current typed `FAILED` Controller state; the existing Replanner produces a deterministic canonical diff and non-executable materiality projection; no-change revisions remain stopped, while material revisions publish one review-only successor and clear old authority bindings so exact user approval creates fresh Permission, Authorization, StartIntent, and Controller artifacts. Crash/restart/replay and duplicate-request paths reuse the existing Replanner checkpoints and publication locks. Representative L2 trigger, materiality, fresh-authority, restart, replay, and handoff evidence is recorded by PR #48.
  - Evidence: [Autonomy L1/L2 acceptance manifest](evidence/autonomy-l1-l2-acceptance-v1/acceptance_manifest.json), [scenario matrix](evidence/autonomy-l1-l2-acceptance-v1/scenario_matrix.json), and [authority-boundary summary](evidence/autonomy-l1-l2-acceptance-v1/authority_boundary_summary.json).

- [x] **M3.5-AUT-ACCEPT — L1/L2 adversarial and restart acceptance**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Dependency: after `M3.5-AUT-L2`.
  - Target PR: #48.
  - Definition of Done: representative bounded-autonomy, fail-closed, replay, restart, duplicate-dispatch, concurrent-replan, budget, and adversarial cases pass on exact reviewed code and preserve the existing authority chain. The formal runner passes all 16 versioned scenarios on acceptance-code HEAD `d225b6315dd6ab221c73e5d80f80922ccb62e492`, including real conversation-tick budget stop-before-effect checks, a real fresh-L1 handoff tick, wall-clock exhaustion before effect, task-graph/resource fail-closed checks, one separate-Python-process L2 restart/replay path, and one concurrent `/replan` reconciliation path; checked-in evidence is control-plane-only and privacy-safe. The L1 remote adoption case is explicitly a durable receipt crash-window reconciliation, not a process-boundary restart claim.
  - Closed by: PR #48, [acceptance README](evidence/autonomy-l1-l2-acceptance-v1/README.md), [manifest](evidence/autonomy-l1-l2-acceptance-v1/acceptance_manifest.json), [scenario matrix](evidence/autonomy-l1-l2-acceptance-v1/scenario_matrix.json), [restart/replay summary](evidence/autonomy-l1-l2-acceptance-v1/restart_replay_summary.json), and [authority-boundary summary](evidence/autonomy-l1-l2-acceptance-v1/authority_boundary_summary.json).

- [x] **M3.5-BR2-RUNTIME — Real MinerU runtime closure**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Dependency: after `M3.5-AUT-ACCEPT`.
  - Target PR: #49.
  - Closed by: PR #49.
  - Definition of Done: at least one real OLED PDF completes the real MinerU and parsed-document runtime stages without extending the BR2 terminal boundary.
  - Evidence: [acceptance README](evidence/br2-real-mineru-runtime-v1/README.md), [acceptance manifest](evidence/br2-real-mineru-runtime-v1/acceptance_manifest.json), and [runtime summary](evidence/br2-real-mineru-runtime-v1/runtime_summary.json).

- [x] **M3.5-BR2-MAPPING — Contextual mapping and evidence binding**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Dependency: after `M3.5-BR2-RUNTIME`.
  - Target PR: #50.
  - Definition of Done: deterministic extraction, LLM contextual mapping, schema validation, and evidence binding produce a confirmation-ready candidate raw dataset with privacy-safe provenance.
  - Closed by: PR #50, [acceptance README](evidence/br2-contextual-mapping-v1/README.md), [acceptance manifest](evidence/br2-contextual-mapping-v1/acceptance_manifest.json), and [mapping summary](evidence/br2-contextual-mapping-v1/mapping_summary.json).

- [x] **M3.5-BR2-CONVERSATION-IMPL — Conversation-driven BR2 implementation and crash-safe publication**
  - State: `DONE`
  - Evidence: `I/T/—`
  - Dependency: after `M3.5-BR2-MAPPING`.
  - Closed by: PR #51.
  - Definition of Done: conversation-driven BR2 implementation, confined crash-safe publication, and immutable replay/recovery are implemented and covered by repository tests. This item does not claim fresh real conversation acceptance.

- [ ] **M3.5-BR2-ACCEPT — Conversation-driven BR2 acceptance**
  - State: `READY`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-BR2-MAPPING` and `M3.5-BR2-CONVERSATION-IMPL`.
  - Definition of Done: the real PDF conversation path reaches the human confirmation Gate and `WAITING_USER`; no training, generation, Top-N, or experimental validation is included.
  - Implementation note: PR #51 closed the separate implementation item above; fresh real conversation acceptance has not been rerun.

- [x] **M3.5-AUT-AUTH — Grant-based autonomy authority model**
  - State: `DONE`
  - Evidence: `I/T/—`
  - Dependency: after `M3.5-AUT-L2`.
  - Closed by: PR #54.
  - Definition of Done: bounded autonomy authority is represented by an explicit grant and relation policy; authority does not create a second execution authority, and historical exact-option v1 authorization remains readable and unchanged.

- [x] **M3.5-AUT-AUTH-L2 — Authority-aware L2 provenance and application**
  - State: `DONE`
  - Evidence: `I/T/—`
  - Dependency: after `M3.5-AUT-AUTH`.
  - Closed by: PR #55.
  - Definition of Done: L2 uses `AuthorityRelation` and independent `SemanticBoundary` classification; historical v1 uses exact option authority, scope-aware v2 uses the registered schema range, and authority-aware application preserves `agent_plan_revision_application_receipt.v2` provenance.

- [x] **M3.5-AUT-FASTPATH — Minimal deterministic-successor refactor**
  - State: `DONE`
  - Evidence: `I/T/—`
  - Dependency: after `M3.5-AUT-AUTH-L2` and the minimal golden-path regression guard.
  - Closed by: PR #57.
  - Definition of Done: when the current verified Controller state has exactly one legal, authorized, argument-free deterministic successor with no semantic boundary, the server may skip the Execution Agent LLM call. The path creates no authority, never executes effects directly, never bypasses Controller, does not choose scientific branches or parameters, does not approve Gates, does not handle `UNKNOWN_EFFECT`, retry, or replan, and does not become a second state machine.
  - Implementation note: v1 reviews only `ADOPT_COMPLETED_TASK` after verified local publication; all other and future Controller actions remain on the existing Execution Agent or human/fail-closed paths. The deterministic decision is a recomputed, non-executable projection.

- [x] **M3.5-AUT-EXECUTION-V2 — Execution Agent v2**
  - State: `DONE`
  - Evidence: `I/T/—`
  - Dependency: after `M3.5-AUT-FASTPATH`.
  - Closed by: PR #58.
  - Definition of Done: a strict versioned `TOOL_CALL` / `ASK_USER` / `REPLAN` response and proposal contract is server-validated against a small logical scientific-tool catalog; each logical tool exposes a closed JSON Schema with bounded arguments and no physical adapter, path, host, credential, command, or worker fields; a deterministic non-executable compiler recomputes current Controller evidence, registered options, `AutonomyGrant` relation, and `SemanticBoundary`; only `SUBSET + NONE` with an exact current Controller-compatible operation may continue, and all application remains through the existing Permission / Authorization / Controller chain. v1 proposal/receipt artifacts remain byte-compatible and replayable; v2 makes at most one provider call and does not retry unknown outcomes. The deterministic fast path remains ahead of v2, while `ASK_USER`, semantic boundaries, Gate/remote authority, `REPLAN`, and unsupported/future actions remain non-executable and fail closed. The production Conversation integration is server-opt-in through `AI4S_AGENT_EXECUTION_AGENT_V2_ENABLED` so historical v1 callers retain their exact contract.

- [ ] **M3.5-AUT-FAILURE-RECOVERY — Failure taxonomy and bounded recovery**
  - State: `READY`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-EXECUTION-V2`.
  - Definition of Done: to be defined by a later contract and acceptance PR; this item is not implemented by the regression guard.

- [ ] **M3.5-AUT-FEEDBACK — Durable structured feedback and EvidenceGrant**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-FAILURE-RECOVERY`.
  - Definition of Done: to be defined by a later contract and acceptance PR; this item is not implemented by the regression guard.

- [ ] **M3.5-AUT-LEASE — Active-time accounting and autonomy lease**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-FEEDBACK`.
  - Definition of Done: to be defined by a later contract and acceptance PR; this item is not implemented by the regression guard.

- [ ] **M3.5-UI — Minimal unified UI authority-preservation closure**
  - State: `DEFERRED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-LEASE` and `M3.5-BR2-ACCEPT`.
  - Definition of Done: the minimal UI consumes existing conversation/session, proposal, permission, authorization, Controller state, Gate, result projection, inspection, and trace links without becoming state or execution authority.

- [ ] **M3.5-V1-ACCEPT — Molly v1 final representative acceptance**
  - State: `DEFERRED`
  - Evidence: `—/—/—`
  - Dependency: after BR1, accepted Autonomy L1/L2, accepted BR2 backend, observability acceptance, and minimal UI closure.
  - Definition of Done: repository-owner review binds representative BR1 and BR2 evidence, restart/recovery, exact replay, privacy, authority preservation, and integrated observability to the exact final review HEAD.

### M3.5-AUT-POLICY contract closure

The completed policy is a derived eligibility projection, not a new authority. Its
decision is recomputable from the current verified Controller inspection and binds
the Controller execution ID/digest, inspection digest, exact action, policy
version/digest, canonical reason codes, and `executable: false`.
A serialized decision is only a non-authoritative projection; any consumer must
recompute and exact-verify it against the current typed inspection before using
its eligibility.

The typed Controller action roster is reviewed explicitly. The current v1 policy
allows only the reviewed deterministic action classes to be eligible for
`AUTO_CONTINUE`; Gate approval, remote approval, recovery, and cancel remain
`REQUIRE_HUMAN`. No current typed Controller action is `PROHIBITED`, while an
unknown, untyped, unsupported, or direct-effect bypass request is prohibited or
fails closed. A new Controller action cannot inherit autonomous eligibility
without an explicit policy mapping and test update.

Material changes to dataset, target property, scientific scope, Top-N, thresholds,
task DAG, model/generator strategy, input authority, resource envelope, budget,
or GPU allocation are not ordinary auto-continuation cases. They require a
human/Replanner boundary and are handed off to `M3.5-AUT-L2`; PR #45 froze this
handoff without implementing materiality classification, and PR #46 implements
only the bounded L1 continuation contract.

### M3.5-AUT-L1 implementation closure

PR #46 consumes the PR #45 policy through the existing conversation coordinator;
it does not add a scheduler, Controller, or execution authority. Each mutating
continuation recomputes the current typed Controller inspection and applies the
server-owned L1 runtime policy before an Execution Agent call or Controller
effect. The cumulative finite envelope is scoped to one
`controller_execution_id`: Controller action receipts rebuild transition and
dispatch usage, immutable Execution Agent request checkpoints rebuild LLM-call
usage, `created_at` binds wall-clock usage, and the exact ordered task roster
and resource/budget digests bind task-graph and resource boundaries. A bounded
`tick()` may resume an Execution Agent pause or perform one remote refresh/adopt
path, but Gate approval, remote approval, recovery, cancel, retry, unknown LLM
outcomes, and material changes remain human/L2 boundaries. Read-only session,
SSE, and telemetry projections remain non-authoritative. PR #48 provides
representative runtime, restart, replay, budget, and adversarial evidence for
this implementation, recorded as `I/T/V` in the active queue. This evidence
does not rerun the scientific BR1 path.

### M3.5-AUT-L2 implementation closure

PR #47 freezes and implements the bounded L2 materiality boundary without adding
an execution authority. The versioned policy is
`scientific-agent-autonomy-l2-materiality-policy.v1`; it explicitly reviews the
canonical `task`, `dependency`, `option`, `artifact`,
`route_profile_resource`, `budget`, `gate`, and `semantic` diff dimensions. An
empty current canonical diff is `NON_MATERIAL`; a non-empty diff is `MATERIAL`.
Option changes remain material even when the proposal and authorization carry
the same authorization-scope digest. An unknown dimension fails closed.

The only autonomous L2 entrypoint is the explicit mutating conversation
operation `POST /agent-session/replan`. The server derives the failure trigger,
baseline authority, Controller execution, current inspection, and deterministic
request identity. It permits exactly the current typed `FAILED` Controller
state; it does not automatically replan success, cancellation, recovery, Gate
waiting, remote approval, stale authority, damaged evidence, or unknown LLM
outcomes. The existing Replanner remains review-only and publication-only:
material revisions stop at a fresh-authorization-required pending proposal,
while no-change failures stay stopped. Existing Controller, Permission,
Authorization, Executor, worker, and Replanner publication contracts remain the
only authority paths. PR #48 provides representative adversarial, restart,
replay, exactly-once, and fresh-authority evidence for this boundary, recorded
as `I/T/V` in the active queue.

### M3.5-AUT-ACCEPT closure

PR #48 closes the bounded autonomy acceptance gate with a finite runner over 16
stable scenarios. The evidence covers human Gate and remote-approval stops,
exactly-once remote lifecycle adoption and crash-window reconciliation, invocation and cumulative budgets,
missing evidence and unknown LLM outcome fail-closed behavior, wall-clock/task-
graph/resource boundaries, concurrent ticks, read-only zero-effect surfaces,
the exact-`FAILED` L2 trigger, non-material stop, material successor fresh
authority, process-boundary Replanner replay, successor reconciliation, and the
L1→L2→fresh-L1 epoch handoff through a real B-side tick. `GATE-AUT-L1-L2` is closed by the same reviewed
evidence. This is a control-plane acceptance claim only; PR #48 does not rerun
BR1, implement BR2, or change the scientific claim boundary.

### BR1 acceptance closure

PR #43 closed the Structured Dataset real path:

```text
natural-language conversation front door
→ sole eligible owner-approved BR1 input bundle
→ plan / authorization / Controller
→ dataset confirmation
→ fresh Uni-Mol training
→ real REINVENT4 generation
→ current-run Uni-Mol prediction
→ deterministic final evaluation
→ verified Computational Top-N
→ scientific_result.available
```

The deployment-bound full applicability preflight passed for 1999/1999 supported rows, with 0 unsupported and 0 unresolved rows. The run used exact owner approval, a fresh current-run model, real generation, current-model prediction, the final `evaluate_private_structured_dataset_canary_v1` task, and the `computational-top-n-v1` projection.

The restart/replay canary proves representative BR1 remote continuation and exactly-once adoption for one real training remote stage: after a control-plane restart, the same Controller and remote request were refreshed and adopted with `dispatch_occurred=false`, and dispatch count remained one. It does not prove restart validation for every possible task or failure mode.

---

## 4. Autonomy scope freeze

This section freezes the scope and contracts below. Its definitions do not change
Controller behavior or add execution authority; implementation status is tracked
in the active queue above.

### L0 — Current reference baseline

The system after PR #43 is the L0/reference baseline:

```text
LLM proposes
→ human approves
→ Controller executes
→ explicit boundaries surface to conversation
→ user interaction may be required for continuation
```

L0 does not mean that no Agent exists. The conversation-driven Agent and its bounded proposal path exist; autonomous continuation is not yet a formal accepted capability.

### L1 — Bounded Auto-Continuation

L1 means that, after one explicit approval of a plan, the Agent may automatically execute state transitions that are already inside the current valid authority envelope and do not require a new scientific semantic decision or resource authorization.

L1 must continue to use all of the existing boundaries:

```text
Permission
Authorization
Controller
Gate
Remote Resource Authority
Verifier
Artifact Registry
exact replay
resource budget
```

The invariant is:

> Autonomy does not create authority. Autonomy only consumes already-valid authority.

#### L1 action classes

`AUTO_CONTINUE` may include:

- read verified Controller state;
- inspect verified output;
- refresh an already-dispatched remote task;
- adopt already-completed verified remote output;
- package already-verified outputs;
- continue to the next already-authorized deterministic task;
- retry read-only observation;
- perform an exact replay or read-only recovery operation that creates no new authority; the current typed `RECOVER_REMOTE_TASK` action remains `REQUIRE_HUMAN` until a later reviewed contract changes it.

`REQUIRE_HUMAN` includes:

- scientific dataset confirmation;
- semantic Gate approval;
- new or expanded remote resource authority;
- a material threshold change;
- a Top-N change;
- a scientific target change;
- a new task graph;
- a new model or generator strategy;
- material replanning.

`PROHIBITED` includes:

- arbitrary shell;
- arbitrary SSH;
- unbounded filesystem access;
- direct Gate mutation;
- direct StageState mutation;
- fabricated execution success;
- bypassing Controller or Executor;
- self-authorizing new scientific scope.

#### L1 safety envelope

L1 is bounded by the existing authority and explicit continuation budgets. The contract must account for:

- maximum autonomous transitions;
- maximum autonomous LLM calls;
- maximum remote dispatches allowed by the current authorization;
- maximum wall-clock continuation window;
- existing resource authority and resource budget;
- the existing task graph.

PR #46 fixes the named, server-owned finite L1 runtime bounds as 128 cumulative
Controller transitions, 64 Execution Agent LLM calls, 32 steps per invocation,
and 86,400 seconds from execution creation; remote dispatches remain limited to
the exact authorized remote task-slot roster. These limits are not client
configurable and are a restrictive runtime envelope, not new authority. If an
action cannot be proven to belong to the current authority envelope, the system
must fail closed and surface a human boundary. L1 must not add a second
Controller, state machine, or execution path.

### L2 — Bounded Replanning

L2 means that, when the original plan fails or an observation changes, the Agent may propose the next plan, but only a non-material change may continue to reuse current authority. A material change must produce a successor proposal and fresh authorization.

The frozen decision flow is:

```text
verified failure / observation
        ↓
Execution Agent / Replanner proposal
        ↓
deterministic materiality classification
        ↓
NON_MATERIAL
    → existing authority may continue

MATERIAL
    → successor proposal
    → new digest
    → fresh human authorization
```

Non-material changes may include:

- polling again;
- reading current verified state;
- refreshing the same remote request;
- adopting completed output;
- repeating deterministic read-only inspection;
- recovering the exact same request;
- retrying explicitly idempotent observation.

Material changes include:

- changing the model;
- changing the dataset;
- changing the target property;
- changing Top-N;
- changing the ranking threshold;
- changing the task DAG;
- changing the generation policy;
- changing the resource envelope or adding another GPU;
- changing scientific scope;
- replacing input authority.

Materiality classification should be as deterministic as possible. The LLM must not be the sole authority deciding whether its own proposed change requires fresh authorization.

---

## 5. Molly v1 scope and dependency order

### Structured-data workflow

```text
Raw / Candidate Dataset
→ inspect and clean
→ human confirmation
→ Confirmed Dataset
→ fresh model training
→ current-run Model Package
→ real candidate generation
→ current-model prediction and ranking
→ chemical/applicability checks
→ Computational Top-N
```

The formal real-tool path must not substitute old checkpoints, old predictions, or `existing_output` for the current run's model, generation, or prediction stages.

### BR2 — PDF / MinerU / LLM

BR2 remains deliberately bounded:

```text
real OLED PDF
→ MinerU
→ ParsedDocument
→ deterministic extraction
→ LLM contextual mapping
→ schema validation
→ evidence-bound candidate raw dataset
→ human confirmation Gate
→ WAITING_USER
```

BR2 v1 does not enter training, generation, Top-N, or experimental validation. Autonomy L1/L2 sequencing does not expand the BR2 scientific scope.

### Minimal unified UI

The UI remains deferred until BR1 is complete, Autonomy L1/L2 is accepted, and the BR2 backend is accepted. It may consume:

```text
conversation/session
proposal
permission
authorization
Controller state
Gate
result projection
inspection
trace links
```

It must not become a new state or execution authority, and this roadmap does not start a large UI rewrite.

### Observability

Molly's immutable execution/evidence records are authoritative. OpenTelemetry records system, Controller, Executor, remote lifecycle, and tool spans; LangSmith records bounded LLM runs such as Planner, Execution Agent, Replanner, and contextual mapping. Shared correlation may connect telemetry to a Molly run, but telemetry identifiers and vendor status never become execution or scientific authority. Integrated final acceptance remains pending.

---

## 6. Scientific claim boundary

The current BR1 evidence supports only:

- `Model-ranked Computational Candidates`;
- `Computational Top-N`.

These are model-predicted values from a verified deterministic evaluation and are not experimental discoveries or measurements. Molly must not claim experimental PLQY, synthesis confirmation or guarantee, device performance, or applicability outside the verified scope. The BR1 result also does not claim that Molly v1 is complete.

After the full M3.5/v1 acceptance gates below are satisfied, Molly v1 may additionally claim that the structured-data workflow is reproducible from reviewed input to computational candidates, the PDF workflow converts a real paper into confirmation-ready evidence-bound candidate data, and planning/authorization/execution/verification/replanning are auditable. Those claims still do not imply experimental validation or superiority to established baselines.

---

## 7. Acceptance gates

The following gates are separate from the active execution queue, but use the same checklist convention. A checked gate closes only the stated scope.

- [x] **GATE-BR1-REAL — BR1 structured real-tool path**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Completion criterion: PR #43 evidence covers conversation front door, exact owner approval, confirmation, fresh model, real generation, current-model prediction, deterministic evaluation, and verified Computational Top-N.

- [x] **GATE-BR1-RECOVERY — Representative restart/recovery and exactly-once remote adoption**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Completion criterion: one real training remote stage survives control-plane restart using the same Controller/request with no duplicate dispatch; this is not a claim about every task or failure mode.

- [x] **GATE-AUT-L1-L2 — Bounded autonomy acceptance**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Dependency: after `M3.5-AUT-ACCEPT`.
  - Completion criterion: L1/L2 action classification, budgets, fail-closed boundaries, materiality handling, adversarial cases, restart, replay, duplicate-dispatch behavior, and fresh-authority handoff pass on exact reviewed code. Closed by PR #48 [acceptance evidence](evidence/autonomy-l1-l2-acceptance-v1/README.md).

- [ ] **GATE-BR2 — PDF–MinerU–LLM real acceptance**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-BR2-ACCEPT`.
  - Completion criterion: a real PDF reaches evidence-bound candidate raw data, human confirmation, and `WAITING_USER`, without entering downstream structured-data execution.

- [ ] **GATE-OBSERVABILITY — OTel/LangSmith integrated final acceptance**
  - State: `QUEUED`
  - Evidence: `I/T(partial)/—`
  - Dependency: after representative BR1/BR2 acceptance.
  - Completion criterion: telemetry is privacy-safe and observer-only, and exporter availability cannot affect authoritative execution bytes or outcomes.

- [ ] **GATE-UI-AUTHORITY — Minimal unified UI authority preservation**
  - State: `DEFERRED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-UI`.
  - Completion criterion: the UI consumes existing authority projections and creates no second task, Gate, StageState, publication, or authorization authority.

- [ ] **GATE-V1-OWNER — Repository-owner final v1 representative acceptance**
  - State: `DEFERRED`
  - Evidence: `—/—/—`
  - Dependency: after all preceding gates.
  - Completion criterion: the exact final review HEAD, representative runtime evidence, restart/recovery, exact replay, privacy, authority preservation, and claim boundaries are reviewed and accepted by the repository owner.

No individual CI reference canary, preflight PASS, schema freeze, or observability trace closes these gates by itself.

---

## 8. Later research milestones

Later research remains outside the active v1 execution queue and must not be pulled forward by the autonomy scope freeze. After M3.5/v1 runtime closure, the following directions may proceed subject to explicit evidence:

- trajectory audit benchmark with independent reviewed labels;
- long-horizon trajectory perturbation;
- error propagation and failure amplification/attenuation;
- compensatory errors;
- reward-hacking or spurious-improvement analysis;
- evidence-bound Critic evaluation;
- benchmark-driven adaptive policy;
- Agentic RL only after the evaluation and safety envelopes are mature.

Representative Autonomy L1/L2, BR1, and BR2 trajectories can provide an experimental substrate for later long-horizon scientific-Agent error-propagation research. This PR implements none of those research programs: no benchmark implementation, trajectory perturbation, Critic, or RL is started here.

---

## 9. Public repository and history policy

The public repository contains source code, durable public documentation, synthetic fixtures, schemas, and sanitized reviewed evidence. Real credentials, private papers, user/project data, runtime bundles, concrete infrastructure identities, and machine-specific working context remain outside Git.

`CLAUDE.md` and `todo.md` are intentionally ignored local working-context filenames. Public CI and documentation must not depend on them. They must not be re-tracked and must not compete with this roadmap.

References written as `legacy-private PR N` identify authorized pre-migration audit records. They are not public GitHub links. GitHub PR 编号从公开仓库重新开始。 Deleting or ignoring a file in the current tree does not erase historical public Git objects; history rewriting is a separate security operation.

---

## 10. Roadmap update rule

A public roadmap change should update this file when it changes externally meaningful scope, milestone state, acceptance requirements, claim boundaries, or execution order. Topic documents should change only when their durable technical contract changes.

When an active item is completed, update its checkbox, `State`, evidence maturity, Definition of Done evidence, and dependencies in this file in the same PR that closes the item. Private scratch planning may live in ignored `todo.md`; local agent-specific instructions may live in ignored `CLAUDE.md`. Neither may override the tracked public contracts in this repository.
