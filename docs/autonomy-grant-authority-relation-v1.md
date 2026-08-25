# AutonomyGrant, AuthorityRelation, and SemanticBoundary v1

This document defines the Phase 2 authority-model foundation. It is a
non-executable contract: it does not select tools, create a Controller, call an
LLM, dispatch work, or bypass the existing Permission/Controller/Executor
chain.

## Grant scope

`AutonomyGrant` is the user's durable authority envelope. Its scope is bound by
an immutable digest and includes:

- allowed logical task IDs and scientific effect classes;
- closed parameter intervals or enumerated values;
- exact server-owned resource profile IDs and external-I/O scope tokens;
- aggregate and per-task budget caps;
- cumulative retry and replan caps;
- a validity window ending at the required `valid_until` timestamp.

The grant contains no adapter, path, shell command, credential, or provider
instruction. It is an authority description, not an execution request.

## Authority relation

`classify_authority_relation(grant, candidate)` compares complete canonical
scopes in both directions:

| Relation | Meaning |
| --- | --- |
| `SUBSET` | The candidate only narrows tasks/effects/parameters/resources/I/O/budgets/retries/replans or lease duration. |
| `EQUIVALENT` | Both scopes contain exactly the same authority. |
| `EXPANSION` | The candidate is a strict superset, such as adding a task or increasing a budget. |
| `INCOMPARABLE` | One dimension narrows while another expands, or the scopes use incompatible values. |

The comparison is fail-closed. A stale or forged grant digest is rejected
before relation evaluation. Parameter bounds use set containment; task-scoped
bounds and budgets belonging only to a deleted task do not turn task deletion
into an expansion.

## Semantic boundary

`SemanticBoundary` is deliberately independent of resource authority. The
current vocabulary is:

`NONE`, `SCIENTIFIC_CONFIRMATION`, `GOAL_CHANGE`, `DATASET_CHANGE`,
`EXTERNAL_SHARING_CHANGE`, `PUBLICATION`, `PROMOTION`, and
`IRREVERSIBLE_EFFECT`.

The classifier accepts explicit boundary values or canonical change evidence.
Unknown explicit boundary values fail closed. A candidate can therefore be
inside the grant while still requiring a human scientific decision, for
example when a confirmed dataset replaces a raw dataset or a publication step
is introduced.

## Automatic-application rule

`AuthorityEvaluation.auto_apply` is derived, never user- or LLM-supplied:

```text
(relation == SUBSET) AND (semantic_boundary == NONE)
```

`EQUIVALENT` covers an idempotent/no-op re-evaluation and is not a new
autonomous action. `SUBSET` covers safe reductions such as a smaller batch,
fewer records, fewer retries, or deleting a task. Any `EXPANSION`,
`INCOMPARABLE` relation, or non-`NONE` semantic boundary remains outside
automatic application and requires the next authority or human boundary
defined by the runtime phase that consumes this primitive.

## Scope boundary for this phase

This phase only adds the typed models, canonical digests, deterministic
comparison, semantic-boundary classification, frozen JSON schemas, and
contract tests. Existing L1/L2 runtime behavior remains unchanged until a
separate integration phase wires this primitive into the coordinator.
