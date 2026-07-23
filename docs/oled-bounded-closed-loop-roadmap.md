# OLED bounded closed-loop roadmap

This roadmap constrains the executable loop after PR-AT. It is intentionally narrow:
the immediate goal is an executable end-to-end candidate-discovery flow, not a
general candidate ontology or an autonomous-scientist claim.

## PR-ATb: cumulative generated-candidate evaluation

Before a session coordinator can execute a second generation round, PR-AT must
retain the prior generated pool. PR-ATb therefore adds an ordered, exact-bound
PR-AS roster and a previous-evaluation binding. It continues to support only
`registry` and `generated` sources, preserves the single-round v1 contract, and
recomputes one global ranking over Registry plus every accepted generation
publication. It adds no new prediction, selection, or budget policy.

## PR-ARb v2: candidate decision successor

PR-ARb v2 consumes the exact PR-AT `oled_candidate_evaluation` publication and
emits an explainable Top-N dossier. Its candidate source contract supports only:

- `registry`;
- `generated`.

The first version must not add variants for literature, external databases,
simulation, or human-submitted candidates. A future source requires a concrete
execution need and a separate contract change.

PR-ARb v2 remains a recommendation artifact. It does not assign generated
structures a Registry material identity and does not include manual
accept/defer/reject adjudication.

It inherits the exact first PR-ARb target, property constraints, budget, and
diversity request. It does not create a second configurable selection schema.
The original rank-anchored greedy max-min Tanimoto policy remains unchanged,
and an incomplete Top-N produces no final selected candidates.

## PR-AU: bounded closed-loop discovery controller

The first controller should be described as a **bounded closed-loop discovery
controller**, or more generally an **agentic workflow controller**. It should
not be described as a fully autonomous loop or autonomous scientist.

Its action space is fixed to the implemented route:

1. evaluate current candidate supply;
2. trigger gated inverse design only for a genuine property-qualified supply
   shortfall;
3. run controlled prediction and global ranking;
4. produce the explainable Top-N dossier;
5. stop when the target is met or any budget is exhausted.

The v1 request contract must enforce these hard upper bounds:

```yaml
max_iterations: 3
max_generation_rounds: 2
max_generated_candidates: 512
```

`max_generated_candidates` is cumulative across unique PR-AS publications,
using accepted/source candidates before PR-AT exclusion rather than only later
prediction successes. One iteration is one supply-evaluation and candidate-
decision cycle; a generation round is only an iteration that actually invokes
the generator. Callers may request smaller positive limits, but v1 rejects any
value above these ceilings. The controller stops before dispatching an action
that would exceed any limit. Reaching a limit is a normal bounded outcome
recorded in the final receipt, not permission to silently increase the budget.

Human approval remains attached to the gated generation action; it is not a
manual accept/defer/reject step for each candidate. Computational or molecular
dynamics validation is optional future capability and is not an acceptance
criterion for closing this first end-to-end workflow.

PR-AU implements this as a control-decision publication. It never executes a
generation action itself: `request_generation_approval` emits an exact-bound
authorization for the existing gated PR-AS task, while `stop` is terminal for
the supplied bounded history. All rounds share a loop fingerprint over the
scientific target and upstream model/dataset/Registry context, and a chemical
identity ledger prevents publication-scoped generated IDs from hiding repeated
molecules across rounds.

## PR-AV: bounded discovery session coordinator

PR-AV coordinates the existing PR-AQ, PR-ARb, PR-AS, PR-AT, PR-ARb v2, and
PR-AU tasks through deterministic child runs. It uses the existing
RunPlanExecutor and gate snapshots, advances at most one durable state
transition at a time, and never calls scientific adapters directly or
maintains a second copy of PR-AU's budget decisions.

The implementation is a project-level persistent session rather than another
AtomicTask. It provides immutable SessionSpec/result artifacts, revision-CAS
advancement, exact child-publication replay, restart-safe waiting gates, and a
fail-closed `RECOVERY_REQUIRED` boundary for an interrupted unregistered
remote execution. See `docs/oled-bounded-discovery-session.md`.

The paper018 local `existing_output` canary is complete; see
`docs/evidence/oled-paper018-existing-output-session-canary-20260722.md`. It
formed an explainable mixed-source Top-4 and a durable `COMPLETED_TOP_N`
session result after one generation round.

## PR-AW: bounded session control plane

PR-AW exposes PR-AV through project-scoped create, inspect, revision-CAS
advance, and exact gate-approval APIs plus a dedicated result page. Long child
transitions run outside the HTTP request thread and leave pollable action
metadata. Interrupted actions fail closed as `RECOVERY_REQUIRED`; they are not
automatically replayed. Successful action reads revalidate the authoritative
session and child publications before displaying an explainable Top-N.

This control plane does not change scientific selection semantics and does not
claim experimental or computational validation.

## PR-AX: local control-plane acceptance

The paper018 `existing_output` flow has also completed the PR-AW HTTP
control-plane canary; see
`docs/evidence/oled-paper018-pr-aw-control-plane-canary-20260723.md`. The run
created and drove the session through project-scoped asynchronous APIs,
survived an application/service restart at a waiting gate, approved three
exact-bound gates, and reproduced the same explainable chemical Top-4 as the
direct PR-AV canary.

PR-AX adds runtime evidence only. It introduces no new schema, scientific
policy, generator, or validation claim.

## PR-AY/PR-AZ: remote transport and session acceptance

The executable CPU-only REINVENT4 transport is deployed on node221 through the
`workstation1-node221-reinvent4-v1` profile. The real paper018 remote session
canary is complete; see
`docs/evidence/oled-paper018-node221-remote-session-canary-20260723.md`.

That run exposed and closed one PR-AV provenance-propagation defect: the pinned
remote known-hosts anchor now remains available to every downstream round child
that exact-replays PR-AS. A fresh project then completed remote generation,
single-round Registry-plus-generated prediction, explainable Top-4 selection,
and bounded controller termination. The result remains recommendation-only and
adds no computational or experimental validation claim.
