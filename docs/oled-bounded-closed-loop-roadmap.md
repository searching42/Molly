# OLED bounded closed-loop roadmap

This roadmap constrains the two steps after PR-AT. It is intentionally narrow:
the immediate goal is an executable end-to-end candidate-discovery flow, not a
general candidate ontology or an autonomous-scientist claim.

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

`max_generated_candidates` is cumulative across the controller invocation, not
per generation round. One iteration is one supply-evaluation and candidate-
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
generation action itself: `request_generation_approval` routes the existing
gated PR-AS task, while `stop` is terminal for the supplied bounded history.
