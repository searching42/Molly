# L2 authority-aware revision integration

The serialized L2 authority projection is `agent_autonomy_l2_materiality_decision.v2`.

The L2 replan boundary keeps the canonical server diff as an integrity check,
but does not use “non-empty diff” as a proxy for fresh user authority.

For each verified revision the server projects:

1. the authorized baseline proposal into an `AutonomyGrant`, using the
   registered task catalog and each option schema's closed bounds for
   scope-aware v2 proposals; historical v1 proposals instead use the exact
   approved option values because v1 has no separate scope identity;
2. the validated successor into a candidate grant, using exact proposed
   option values and exact route/profile/resource bindings; and
3. canonical diff evidence into `evaluate_authority`, which independently
   returns `AuthorityRelation` and `SemanticBoundary`.

Canonical goal/constraint paths map to `GOAL_CHANGE`, source-artifact and
missing-artifact paths map to `DATASET_CHANGE`, and other semantic/dependency/
gate changes conservatively map to `SCIENTIFIC_CONFIRMATION`.

The L2 decision is authority-safe only when the frozen policy returns:

```text
AuthorityRelation.SUBSET
SemanticBoundary.NONE
```

That path publishes the immutable successor and runs it through the existing
Permission → authorization → Controller chain with a durable authority-reuse
receipt. It does not bypass exact proposal binding, Gate handling, resource
authority, or Controller verification. A relation expansion, incomparable
scope, equivalent action, or any semantic boundary remains review-only and
requires a new user authorization.

The L2 decision stores the relation, boundary, evaluation identity/digest, and
fresh-authority flags. Coordinators recompute this projection from the current
verified baseline and revision before using it; serialized L2 decisions are
not execution authority.

An L2 application emits
`agent_plan_revision_application_receipt.v2`. The receipt binds the verified
decision/evaluation digests and baseline authorization identity, and its
freshness flags are derived from `authority_auto_apply`. The historical
`agent_plan_revision_application_receipt.v1` contract remains unchanged and
is used only by the generic application endpoint. A successor publication
without its receipt therefore cannot be recovered under a different authority
interpretation: the application request digest and receipt both carry the
verified v2 binding.

The checked-in `autonomy-l1-l2-acceptance-v1` directory is an immutable
historical run and remains bound to its original L2 v1 policy. A future
authority-aware acceptance run must create a new evidence directory and
acceptance identity rather than rewriting that manifest.
