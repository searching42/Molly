# Authorization scope refactor change list

> Status: active working change list for the authorization-scope refactor.
> This is a task-tracking document for the refactor only; `../todo.md` remains
> the sole normative source for roadmap scope and status.

> 2026-08-06 progress: Steps 1-4 complete (see checkboxes). Full suite
> `6770 passed, 5 skipped`. One follow-up is explicitly deferred and requires
> a product decision: an argument-carrying Execution Agent tool (dispatch-time
> option-schema validation plus new Controller actions) is the natural next
> step after the option-schema projection added in #10.

> 2026-08-06 review follow-up (PR #38): the original change mutated v1 digest
> semantics in place. The follow-up makes the refactor a strict superset:
>
> * `agent_execution_plan_proposal.v1` keeps the legacy field set and digest
>   algorithm; new proposals are written as `.v2` with
>   `authorization_scope_digest`.
> * `agent_plan_authorization.v1` keeps option values in its digest and stays
>   byte-reproducible; new authorizations are written as `.v2` with the
>   scope identity and option values excluded from the digest.
> * `agent-task-authority-binding.v1/.v2` keep the legacy option-value
>   material; the new `.v3` binds the option *policy* digest.
> * `scientific-agent-permission-policy.v6` is the new write policy using the
>   v3 binding; policies v1-v5 replay with their legacy algorithms.
> * Controller executions dispatch `semantic_material()` on
>   `controller_policy_version`: v1 keeps the per-slot
>   `compiled_options_digest` and the concrete `compiled_task_options_digest`;
>   v2 binds the new `task_authority_roster_digest` and records the concrete
>   options digest for audit only.
> * Historical fixture replay tests
>   (`tests/test_historical_v1_authority_fixture_replay.py`) prove old
>   Proposal/Authorization/PermissionDecision/ControllerExecution artifacts
>   remain read-only verifiable, and
>   `tests/test_controller_remote_successor_crash_windows.py` pins the remote
>   successor crash-window contract.
>
> Positioning stays "scope-identity groundwork": current v1/v2 execution
> still requires an exact proposal and authorization binding. Bounded
> in-workflow option revision (a real `AgentTaskOptionSelectionReceipt`
> object) remains a separate design decision and is not half-implemented.

## Goal

Decouple LLM decision content from authorization binding. Authorizations bind
the *scope* (task roster, per-task policy, profiles, budgets, gates, stop
conditions) while LLM-chosen *values* are validated against schemas at dispatch
time instead of being pre-hashed. Audit digests that record what actually
happened remain exact.

## Guiding rule

- Checks that require "this execution is identical to the last one" are
  replay/identity checks: keep them.
- Checks that require "the LLM-chosen content equals a pre-hashed value" are
  conflict sources: change them to "the LLM-chosen content must satisfy the
  authorized scope".
- Human gate approvals, immutable scientific artifacts, remote dispatch
  requests, and publications keep exact binding.

## Change order

### Step 1: Proposal / authorization scope digest (#1-#5)

- [x] #1 `schemas.py` `AgentExecutionPlanProposal`:
  added `authorization_scope_material()` and `authorization_scope_digest`
  (policy fields only); `proposal_digest` stays the full-content audit digest.
- [x] #2 Plan digest stays for audit; proposal scope digest excludes task
  option values; semantic-plan digest excludes the derived scope digest.
- [x] #3 `scientific_agent_permissions.py`: `evaluate` gains
  `expected_authorization_scope_digest`; mismatch is a deterministic DENY
  (`AUTHORIZATION_SCOPE_MISMATCH`); exact `proposal_digest` binding remains
  for the immutable review/authorization of the published proposal.
- [x] #4 `scientific_agent_authorization.py`: authorization binds
  `authorization_scope_digest`; `verify_authorization` cross-checks it;
  authorization digest no longer covers option values (recorded for audit).
- [x] #5 `scientific_agent_permissions.py` `_task_authority_digest`: hashes the
  task option *policy* (`_option_policy_digest`: option schema, server
  defaults, review-required IDs, compiler version) instead of option values.

### Step 2: Controller checks from content equality to scope validation (#6-#9)

- [x] #6 `scientific_agent_harness_controller.py` `_build_execution` +
  `schemas.py`: execution identity binds the task-authority roster digest
  (`agent_harness_controller_task_authority_roster.v1`) and strips per-slot
  option values from the execution digest material; per-attempt option values
  remain recorded for dispatch audit.
- [x] #7 `_inspect`: verified no code change needed. The authorized-input
  content check covers only `authorization.artifact_bindings` (pre-existing
  approved inputs). Workflow-produced artifacts registered during execution
  are not content-gated; existing controller tests prove outputs advance.
- [x] #8 `scientific_agent_harness_controller.py` `_decision_is_fresh`:
  evaluated; kept as-is. The check guards concurrent state mutation between
  decision creation and execution (explicit actions like cancel/recover are
  allowed to differ from `next_action` by design). It is a concurrency/recovery
  invariant, not an LLM-content gate; a source-binding-only variant made the
  remote cancel race fail and was reverted.
- [x] #9 `_verify_post_start_sources`: evaluated; the digest-equality check
  against the latest receipt is a crash-recovery invariant, not an LLM-choice
  gate. Legitimate widening (adopted remote outputs) already merged in the
  `adopted_remote_exact_successor` path; no further change needed.

### Step 3: Execution Agent decision space (#10-#11)

- [x] #10 `execution_agent.py` `build_execution_tool_catalog` +
  `schemas.py` + controller: tool specs now carry the pending task's registered
  `option_schema` (projected through `ControllerAdvanceResult`), so the agent
  sees the parameter space of the step it is about to take. The operation
  remains argument-free; a parameter-adjustment tool with dispatch-time schema
  validation is a separate design decision (new controller actions + argument
  plumbing) and is explicitly deferred, not half-implemented.
- [x] #11 `execution_agent.py` system prompt: updated to state that the
  option schema is selection context only and authorized values cannot be
  changed in this version; inventing tools/arguments/tasks/resources remains
  forbidden.

### Step 4: Resume / execution snapshot semantics (#12-#13)

- [x] #12 `run_plan_state_fingerprint.py` +
  `run_plan_resume_intent_validation.py`: evaluated; no change needed. The
  `run_plan_fingerprint` already covers only the structural task graph and
  artifact roster (options are not part of `RunPlan`), and `stage_state
  _fingerprint` binds stage identity + the executed execution snapshot. Both
  are identity checks, not content pre-hashes of LLM choices.
- [x] #13 `executor.py` `_execution_snapshot`: evaluated; already behaves as
  "validated choice -> new snapshot". Each dispatch validates/creates its own
  execution snapshot; a changed plan or input is a fail-closed identity
  mismatch, which is correct. No code change needed.

## Completion criteria per item

- Code change applied with a one-line summary here.
- Focused tests for the touched module(s) pass.
- Existing fail-closed/replay tests for unchanged boundaries still pass.
