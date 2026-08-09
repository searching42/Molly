# Molly public roadmap

> Document status: Active public roadmap
>
> Public repository baseline: `main`
>
> Current major milestone: M3.5 — Scientific Agent Harness integration and runtime closure
>
> Current focus: M3.5-AUT-POLICY — Autonomy action classification
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
M3.5-BR2-ACCEPT — Conversation-driven BR2 acceptance
        ↓
M3.5-UI — Minimal unified UI
        ↓
M3.5-V1-ACCEPT — Molly v1 final representative acceptance
```

The new P0 is to give the already-verified BR1 chain bounded autonomy before adding new scientific tools. PR #43 proved that the execution substrate can work on the real BR1 path; the next risk is whether an Agent can progress inside the existing authority envelope. L1/L2 are also the foundation for BR2 and for later long-horizon Agent trajectory/error-propagation research. Autonomy must not introduce a second Controller or state machine.

### Active execution queue

The following queue is the single active M3.5/v1 checklist. Target PR numbers are planning references only; if GitHub numbering conflicts, preserve the stable item ID and update the target PR.

- [x] **M3.5-BR1 — Conversation-driven real BR1 acceptance**
  - State: `DONE`
  - Evidence: `I/T/V`
  - Definition of Done: the sole eligible owner-approved bundle is resolved from the natural-language conversation front door; confirmation, fresh current-run model training, real generation, current-model prediction, deterministic final evaluation, verified Computational Top-N, and `scientific_result.available` complete with privacy-safe projection and exact replay.
  - Evidence: PR #43, merge commit `7264a78eeb320fa00e8d9acbd9e556bfcf3e8360`, [checked-in BR1 acceptance evidence](evidence/br1-conversation-real-acceptance-v1/README.md), [acceptance manifest](evidence/br1-conversation-real-acceptance-v1/acceptance_manifest.json), [result summary](evidence/br1-conversation-real-acceptance-v1/result_summary.json), and [restart/replay summary](evidence/br1-conversation-real-acceptance-v1/restart_replay_summary.json).

- [ ] **M3.5-AUT-POLICY — Autonomy action classification**
  - State: `READY`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-BR1`.
  - Target PR: #45.
  - Definition of Done: the `AUTO_CONTINUE` / `REQUIRE_HUMAN` / `PROHIBITED` policy contract, fail-closed rule, authority inputs, and materiality handoff are deterministic, documented, and covered by contract tests.

- [ ] **M3.5-AUT-L1 — Bounded auto-continuation runtime**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-POLICY`.
  - Target PR: #46.
  - Definition of Done: an approved plan may automatically consume only already-valid authority for bounded continuation; transition, LLM-call, dispatch, wall-clock, task-graph, and resource budgets are enforced; every boundary remains visible and all uncertain actions fail closed to a human boundary.

- [ ] **M3.5-AUT-L2 — Bounded replanning and materiality boundary**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-L1`.
  - Target PR: #47.
  - Definition of Done: verified failure or changed observation produces a successor proposal; deterministic materiality classification permits only non-material reuse of current authority, while every material change creates a new digest and requires fresh human authorization.

- [ ] **M3.5-AUT-ACCEPT — L1/L2 adversarial and restart acceptance**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-L2`.
  - Target PR: #48.
  - Definition of Done: representative bounded-autonomy, fail-closed, replay, restart, duplicate-dispatch, budget, and adversarial cases pass on exact reviewed code and preserve the existing authority chain.

- [ ] **M3.5-BR2-RUNTIME — Real MinerU runtime closure**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-ACCEPT`.
  - Target PR: #49.
  - Definition of Done: at least one real OLED PDF completes the real MinerU and parsed-document runtime stages without extending the BR2 terminal boundary.

- [ ] **M3.5-BR2-MAPPING — Contextual mapping and evidence binding**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-BR2-RUNTIME`.
  - Target PR: #50.
  - Definition of Done: deterministic extraction, LLM contextual mapping, schema validation, and evidence binding produce a confirmation-ready candidate raw dataset with privacy-safe provenance.

- [ ] **M3.5-BR2-ACCEPT — Conversation-driven BR2 acceptance**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-BR2-MAPPING`.
  - Target PR: #51.
  - Definition of Done: the real PDF conversation path reaches the human confirmation Gate and `WAITING_USER`; no training, generation, Top-N, or experimental validation is included.

- [ ] **M3.5-UI — Minimal unified UI authority-preservation closure**
  - State: `DEFERRED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-ACCEPT` and `M3.5-BR2-ACCEPT`.
  - Target PR: #52.
  - Definition of Done: the minimal UI consumes existing conversation/session, proposal, permission, authorization, Controller state, Gate, result projection, inspection, and trace links without becoming state or execution authority.

- [ ] **M3.5-V1-ACCEPT — Molly v1 final representative acceptance**
  - State: `DEFERRED`
  - Evidence: `—/—/—`
  - Dependency: after BR1, accepted Autonomy L1/L2, accepted BR2 backend, observability acceptance, and minimal UI closure.
  - Target PR: #53.
  - Definition of Done: repository-owner review binds representative BR1 and BR2 evidence, restart/recovery, exact replay, privacy, authority preservation, and integrated observability to the exact final review HEAD.

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

This roadmap freezes the scope and contracts below. It does not implement runtime autonomy, change Controller behavior, or add execution authority.

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
- perform exact replay or recovery that creates no new authority.

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

This roadmap does not choose new default numeric values unless an existing deterministic contract already fixes them. If an action cannot be proven to belong to the current authority envelope, the system must fail closed and surface a human boundary. L1 must not add a second Controller, state machine, or execution path.

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

- [ ] **GATE-AUT-L1-L2 — Bounded autonomy acceptance**
  - State: `QUEUED`
  - Evidence: `—/—/—`
  - Dependency: after `M3.5-AUT-ACCEPT`.
  - Completion criterion: L1/L2 action classification, budgets, fail-closed boundaries, materiality handling, adversarial cases, restart, replay, and duplicate-dispatch behavior pass on exact reviewed code.

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
