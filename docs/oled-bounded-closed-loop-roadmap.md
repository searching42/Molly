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
per generation round. The controller stops before dispatching an action that
would exceed any limit. Reaching a limit is a normal bounded outcome recorded
in the final receipt, not permission to silently increase the budget.

Human approval remains attached to the gated generation action; it is not a
manual accept/defer/reject step for each candidate. Computational or molecular
dynamics validation is optional future capability and is not an acceptance
criterion for closing this first end-to-end workflow.
