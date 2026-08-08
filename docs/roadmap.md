# Molly public roadmap

> Document status: Active public roadmap
>
> Public repository baseline: `main`
>
> Current major milestone: M3.5 — Scientific Agent Harness integration and runtime closure
>
> Local `todo.md` is intentionally Git-ignored working context. It may be used for private scratch planning, but it is not public repository authority.

`docs/roadmap.md` 是公开仓库中路线图范围、里程碑状态、优先级、验收边界和执行顺序的唯一规范性来源。Topic documents may define durable implementation contracts and acceptance procedures, but they must not maintain a competing current-status table or decision log.

---

## 1. Status and evidence semantics

Public roadmap status separates implementation maturity from work-management state.

Evidence maturity:

- `I`: implemented in the repository.
- `T`: covered by relevant automated tests.
- `V`: validated by a representative runtime, exact replay, benchmark, or reviewed acceptance evidence.

Work state:

- `READY`: prerequisites are satisfied and the work may start.
- `IN_PROGRESS`: the work is actively being implemented or accepted.
- `BLOCKED`: a concrete technical or external blocker prevents progress.
- `DEFERRED`: intentionally postponed; not a blocker.
- `DONE`: the defined scope has met its Definition of Done.

`I/T/—` never implies runtime validation. A schema, unit test, CI pass, or synthetic fixture alone must not be promoted to `V`.

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

M3 trajectory integrity, deterministic auditing, representative failure attribution, and read-only inspection are considered complete. The active public milestone is M3.5 integration/runtime closure.

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

### Current public status

| Area | Public status | Claim boundary |
|---|---|---|
| Planner / tool catalog / proposal contracts | `I/T/—` | implemented and tested; not runtime `V` by contract tests alone |
| Permission / immutable authorization / resource authority | `I/T/—` | deterministic authority exists; real-tool acceptance remains separate |
| Harness Controller | `I/T/—` | integrated with existing execution authorities; no second execution authority |
| Execution Agent | `I/T/—` | bounded tool-call proposal path exists |
| Replanner / plan revision | `I/T/—` | material changes create successor proposal/digest and require fresh authorization |
| Unified run inspection | `I/T/—` | read-only current/historical projection exists |
| OTel / LangSmith observability | `I/T(partial)/—` | non-authoritative tracing exists; final integrated acceptance is still pending |
| Structured Dataset real-tool canary (BR1) | `IN_PROGRESS` | public contracts/preflight exist; clean owner-reviewed end-to-end acceptance is not yet complete |
| PDF–MinerU–LLM canary (BR2) | `READY` | must stop at human confirmation with an evidence-bound candidate raw dataset |
| Unified Harness UI | `DEFERRED` until backend canaries stabilize | UI must consume existing strict APIs and must not become state authority |
| Final v1 acceptance | `DEFERRED` | requires representative runtime, restart/recovery, exact replay, privacy, and adversarial evidence |

The current P0 direction is M3.5 runtime closure: finish BR1 real-tool acceptance, complete BR2, then stabilize the minimal unified UI and run final acceptance. New infrastructure or authority contracts should not displace these acceptance tasks unless they close a demonstrated correctness or security blocker.

---

## 4. Molly v1 scope

Molly v1 is a reproducible and auditable scientific Agent for OLED organic small-molecule emitter workflows. TADF emitters are the preferred scientific target when the available data support defensible molecule/paper splits and condition-aware comparison.

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

The formal real-tool path must not substitute old checkpoints, old predictions, or `existing_output` for the current run's model/generation/prediction stages.

### PDF workflow

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

The PDF workflow ends at the confirmation Gate in v1 acceptance. It does not imply a confirmed dataset, model training, candidate generation, or Top-N result.

### Observability

- Molly's immutable execution/evidence records are authoritative.
- OpenTelemetry records system, Controller, Executor, remote lifecycle, and tool spans.
- LangSmith records bounded LLM runs such as Planner, Execution Agent, Replanner, and contextual mapping.
- Shared correlation may connect the two telemetry systems to a Molly run, but telemetry identifiers and vendor status never become execution or scientific authority.

### UI

The minimal v1 UI should expose plan review, permission results, approve-and-start, revision/rejection, Gate decisions, recovery-required state, artifact/model/candidate results, and trace links. A large front-end rewrite is not required.

---

## 5. Scientific claim boundary

After the required acceptance evidence exists, Molly v1 may claim that:

- the structured-data workflow is reproducible from reviewed input to computational candidates;
- the PDF workflow reproducibly converts a real paper into confirmation-ready candidate data with evidence anchors;
- training and generation execute from the current run's authorized inputs;
- Agent planning, authorization, execution, verification, and replanning are auditable;
- final outputs are model-ranked computational candidates with basic chemistry/applicability evidence.

Molly v1 must not claim, without separate evidence, that:

- candidates are experimentally validated or discovered materials;
- predicted PLQY/EQE/lifetime/device performance is guaranteed in experiment;
- a candidate is necessarily synthesizable or necessarily a high-performance TADF material;
- DFT, TD-DFT, MD, QM/MM, KMC, wet-lab validation, or autonomous-scientist capability has been completed;
- Molly outperforms established molecular-optimization baselines.

The preferred result names remain `Model-ranked Computational Candidates` or `Computational Top-N`.

---

## 6. Acceptance gates

M3.5/v1 closure requires evidence for all of the following classes:

1. Planner, permission, authorization, Controller, Execution Agent, and Replanner operate through the frozen authority chain in representative runtime conditions.
2. Structured Dataset Canary executes a current-run confirmed-dataset → fresh-model → real-generation → prediction/ranking path and produces `Computational Top-N` with provenance.
3. PDF–MinerU–LLM Canary uses at least one real OLED/emitter paper, produces evidence-bound candidate raw data, and stops before human confirmation.
4. Restart/recovery and exact replay do not repeat completed scientific work or remote dispatch.
5. OTel/LangSmith availability does not affect authoritative execution bytes or outcomes, and exported metadata satisfies the public privacy boundary.
6. The unified UI does not create a second task, Gate, StageState, publication, or authorization authority.
7. Repository-owner review binds final runtime evidence to the exact reviewed code and inputs.

No individual CI Reference Canary, preflight PASS, schema freeze, or observability trace closes these gates by itself.

---

## 7. Later research milestones

After M3.5/v1 runtime closure, later work may resume in this order subject to explicit evidence:

- trajectory-audit benchmark and independent reviewed labels;
- narrow OLED scientific optimization benchmark with fair baselines and leakage-resistant splits;
- evidence-bound Critic evaluation in offline/shadow mode;
- benchmark-driven adaptive scientific policy;
- stronger external/prospective scientific validation;
- Agentic RL only after the evaluation and safety envelopes above are mature.

These are research milestones, not current v1 product claims.

---

## 8. Public repository and history policy

The public repository contains source code, durable public documentation, synthetic fixtures, schemas, and sanitized reviewed evidence. Real credentials, private papers, user/project data, runtime bundles, concrete infrastructure identities, and machine-specific working context remain outside Git.

`CLAUDE.md` and `todo.md` are intentionally ignored local working-context filenames. Public CI and documentation must not depend on them.

References written as `legacy-private PR N` identify authorized pre-migration audit records. They are not public GitHub links. GitHub PR 编号从公开仓库重新开始。Deleting or ignoring a file in the current tree does not erase historical public Git objects; history rewriting is a separate security operation.

---

## 9. Roadmap update rule

A public roadmap change should update this file when it changes externally meaningful scope, milestone state, acceptance requirements, claim boundaries, or execution order. Topic documents should change only when their durable technical contract changes.

Private scratch planning may live in ignored `todo.md`; local agent-specific instructions may live in ignored `CLAUDE.md`. Neither may override the tracked public contracts in this repository.
