# Scientific Agent failure recovery v1

This document defines the independent `M3.5-AUT-FAILURE-RECOVERY` contract.
It does not replace the Controller, the v2 Execution Agent, L2, Feedback, or
Lease contracts.

## Authority and evidence

`AgentTaskFailureEvidence` is the smallest server-owned typed boundary record.
`AgentFailureObservation` (`agent_failure_observation.v1`) is an immutable,
non-executable projection bound to the current Controller execution,
inspection, source receipt(s), logical tool and closed-schema catalog digest,
input/argument/authority digests, stable session/authority-epoch anchors,
policy, and durable retry/replan counts. Recovery never parses exception text.

| Authoritative source | Typed mapping | Effect certainty | Automatic recovery |
| --- | --- | --- | --- |
| Controller failed before dispatch | `controller_pre_effect_failure` | `NO_EFFECT_CONFIRMED` | bounded policy may retry |
| Controller receipt with dispatch and terminal failure | `controller_effect_failed` | `EFFECT_FAILED_CONFIRMED` | only a separately authorized recovery attempt |
| Controller/remote recovery-required or ambiguous dispatch | `controller_effect_unknown` | `EFFECT_UNKNOWN` | human/reconcile only |
| `ExecutionAgentV2LLMOutcomeUnknown` | `provider_outcome_unknown` | `EFFECT_UNKNOWN` | no recovery provider/effect call |
| typed executor evidence | `AgentTaskFailureEvidence` as published | server evidence | policy-specific |

The complete audited failure-surface matrix is:

| Failure surface | Server evidence used | Class/certainty projection | Recovery boundary |
| --- | --- | --- | --- |
| local terminal failure before dispatch (including duplicate rejection) | Controller/executor receipt proves no dispatch | `NONRECOVERABLE` + `NO_EFFECT_CONFIRMED` | stop; no effect |
| local terminal failure after a committed dispatch | immutable dispatch and terminal result receipt | `NONRECOVERABLE` + `EFFECT_FAILED_CONFIRMED` | only a fresh typed recovery attempt |
| local adapter/process outcome without a committed result | Controller recovery-required receipt | `UNKNOWN_EFFECT` + `EFFECT_UNKNOWN` | reconcile or ask user |
| remote `FAILED` after dispatch | remote lifecycle receipt with dispatch binding | `NONRECOVERABLE` + `EFFECT_FAILED_CONFIRMED` | fresh recovery only |
| remote `RECOVERY_REQUIRED`, transport ambiguity, or ambiguous worker dispatch | remote recovery/lifecycle state | `UNKNOWN_EFFECT` + `EFFECT_UNKNOWN` | existing Controller reconciliation only |
| provider unavailable before an external call | typed unavailable exception before request start | `NONRECOVERABLE` + `NO_EFFECT_CONFIRMED` | stop without provider/effect |
| provider response/outcome unknown | typed `ExecutionAgent*LLMOutcomeUnknown` | `UNKNOWN_EFFECT` + `EFFECT_UNKNOWN` | zero recovery provider/effect calls |
| proposal, closed-schema, or logical-tool validation failure | typed server validation error | `NONRECOVERABLE` + `NO_EFFECT_CONFIRMED` | stop |
| authority expansion | server `AutonomyGrant` comparison | `AUTHORITY_EXPANSION_REQUIRED` + `NO_EFFECT_CONFIRMED` | ask user/fresh grant |
| semantic boundary | server `SemanticBoundary` evaluation | `SEMANTIC_REVIEW_REQUIRED` + `NO_EFFECT_CONFIRMED` | ask user; no Gate bypass |
| missing or stale input evidence | current Controller/input digest verification | `INPUT_EVIDENCE_INSUFFICIENT` + `NO_EFFECT_CONFIRMED` | ask user or one bounded replan |
| existing L2 exact-FAILED replan entrypoint | fresh Controller/replanner receipt | typed replan result; unknown outcome is `EFFECT_UNKNOWN` | one existing Replanner call |

The remaining audited surfaces are intentionally fail-closed: remote
`FAILED` with a committed dispatch uses `EFFECT_FAILED_CONFIRMED`, remote
transport/lifecycle ambiguity uses `EFFECT_UNKNOWN`, provider unavailable
before a call is a typed no-effect `NONRECOVERABLE` boundary and stops without
an effect, proposal/schema validation failure uses `STOP`, authority expansion maps to `ASK_USER`/fresh grant,
semantic-boundary changes map to `ASK_USER`, and stale or missing input
evidence maps to `INPUT_EVIDENCE_INSUFFICIENT`/`ASK_USER`. No surface is
classified by a raw exception string or regular expression.

Unknown or future classes/certainties/actions fail closed. Raw traceback,
stdout/stderr, paths, hosts, commands, credentials, and raw provider payloads
are not recovery inputs or persisted recovery fields.

## Policy

The closed failure classes are `TRANSIENT`, `PARAMETER_RECOVERABLE`,
`ALTERNATIVE_TOOL_AVAILABLE`, `INPUT_EVIDENCE_INSUFFICIENT`,
`AUTHORITY_EXPANSION_REQUIRED`, `SEMANTIC_REVIEW_REQUIRED`, `UNKNOWN_EFFECT`,
and `NONRECOVERABLE`. Effect certainty is independently one of
`NO_EFFECT_CONFIRMED`, `EFFECT_COMMITTED`, `EFFECT_FAILED_CONFIRMED`, or
`EFFECT_UNKNOWN`.

The independent response (`agent_recovery_llm_response.v1`) chooses exactly one
of `RETRY_EXACT`, `TOOL_CALL`, `REPLAN`, `ASK_USER`, or `STOP`. The server
recomputes authority and semantic boundary; confidence is observational only.

* `RETRY_EXACT` copies the logical tool, closed arguments, input digests,
  resource scope, authority, and semantic scope. It requires
  `TRANSIENT + NO_EFFECT_CONFIRMED` and remaining retry budget. It has no LLM
  arguments and is applied only as a new exact successor through
  Permission → Authorization → StartIntent → Controller.
* `TOOL_CALL` validates the server-owned logical-tool roster and closed schema.
  Parameter changes require `SUBSET + NONE`; a tool alternative must already
  be in the server roster and grant. Arguments are never clamped.
* `REPLAN` calls the existing one-shot Replanner/L2 entrypoint once and never
  mutates a DAG directly. `max_replans` is counted from durable receipts.
* `INPUT_EVIDENCE_INSUFFICIENT` may ask the user or use a safe evidence replan;
  evidence is never fabricated. Authority expansion and semantic boundaries
  are `ASK_USER`/`REQUIRE_HUMAN` with zero effect. `NONRECOVERABLE` is `STOP`.
* `UNKNOWN_EFFECT` is always fail closed: no retry, revised tool, alternative,
  automatic replan, provider call, or effect. Only read-only reconciliation or
  explicit human recovery may proceed.

## Durability

`AgentRecoveryAttemptReceipt` (`agent_recovery_attempt_receipt.v1`) is
no-replace and binds the failure and decision digests, action and ordinals,
baseline authority, stable session/grant/authority epoch, successor
provenance, effect receipt (when known), and outcome. `RecoveryBudgetEvidence`
is derived by rescanning unique receipts; serialized counters are not trusted.
The aggregate therefore survives successor Controller IDs and cannot be reset
by a restart or replan. A failure has one recovery-attempt identity, so a
concurrent duplicate receives the same receipt or a deterministic conflict.
Provider-start, provider-response, decision, effect-start, effect-result, and
receipt checkpoints make crash windows replayable. An effect-started window is
reconciled by an authoritative callback and is never dispatched a second time.

## Scope exclusions

This phase adds no Feedback/EvidenceGrant redesign, no active-time Lease, no
endless autonomous loop, and no new LLM effect authority. It bypasses no
Permission, Authorization, Gate, or Controller and invokes no adapter or
worker directly. Historical Execution Agent v1/v2 and Controller receipt
artifacts remain versioned and byte-compatible. `M3.5-BR2-ACCEPT` remains a
separate `READY` boundary and no `V` evidence is claimed here.
