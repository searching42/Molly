# Molly Long-Horizon Scientific Error Propagation Research Extension

Status: `RESEARCH_PROPOSAL`
Advisor approval: `PENDING`
Implementation authorization: `false`
Required for Core v2 refactor: `false`
Repository baseline candidate: `main@4352f137db3976cff31bf6cb30f543caa38f8013`

## 1. Purpose

This document preserves a possible research extension for studying error generation and propagation in long-horizon scientific-agent workflows. It is intentionally separated from the Molly Core v2 simplification refactor because the research direction has not yet been approved by the advisor.

Nothing in this document authorizes implementation. Core v2 MUST remain usable without this extension.

## 2. Research premise

The central hypothesis is:

> Error semantics are domain-specific, while propagation mechanics may be studied with domain-general trajectory and artifact-dependency methods.

For OLED/material discovery, a scientific error can involve molecule identity, measurement condition, evidence binding, units, unsupported inference, dataset leakage, or downstream modeling decisions. Another scientific domain may require different semantic validators.

The extension therefore separates:

- domain-specific error semantics and validators;
- domain-general run/artifact dependencies, controlled interventions, replay, and statistical comparison.

## 3. Definitions

The proposal distinguishes:

- `Error`: a confirmed violation of a declared scientific, evidence, or execution constraint;
- `Uncertainty`: evidence is insufficient to establish validity or invalidity;
- `Failure`: the task does not satisfy its acceptance goal; failure is not automatically an error;
- `Propagation`: a validated upstream error produces a measurable downstream effect under a controlled intervention/replay design.

Validation should avoid forced binary labels where scientific evidence is ambiguous. A future research extension may use states such as:

- `VALID`
- `INVALID`
- `INDETERMINATE`
- `DISPUTED`

## 4. Potential validation scopes

The Core v2 engineering scopes are `ARTIFACT`, `RELATION`, and `BUNDLE`. If this research extension is activated, a richer research vocabulary may distinguish:

- `NODE`: validity of one artifact/value;
- `EDGE`: validity of a relation or binding between two objects;
- `SUBGRAPH`: consistency of a local scientific bundle;
- `TRAJECTORY`: workflow-level properties such as stale state or leakage;
- `OUTCOME`: final utility/performance evaluation.

These scopes are research semantics, not current Core runtime requirements.

## 5. Candidate OLED error classes

A future advisor-approved taxonomy may include:

- omission;
- substitution;
- molecule/entity misassociation;
- condition mismatch such as solution vs film;
- unit or scale error;
- unsupported inference;
- scientific constraint violation;
- dataset or scaffold leakage;
- objective/evaluation misalignment.

Error origin should be tracked separately from semantic class. Candidate origins include source, acquisition, parser, LLM, deterministic tool, human review, model, evaluator, and environment.

## 6. Controlled intervention track

A possible causal experiment would start from a reviewed control artifact and create a minimal, pre-declared intervention artifact differing in one target error.

Example:

```text
Control:
PLQY = 0.82
phase = solution

Intervention:
PLQY = 0.82
phase = film
```

Only downstream stages causally dependent on the changed artifact would be rerun. The original run must remain immutable.

Potential future schemas include:

- `ErrorInstance`
- `InterventionSpec`
- `RandomnessManifest`
- `PairedRunGroup`
- `PropagationOutcome`

None of these belong to the current Core v2 implementation.

## 7. Natural failure track

Controlled errors have strong causal interpretability but may be artificial. A complementary natural-failure corpus could collect errors produced by real agent runs and use expert annotation to test ecological validity.

Natural failures should not be mixed statistically with controlled interventions without an explicit analysis plan.

## 8. Randomness and repeated runs

Repeated runs do not eliminate randomness. They estimate the distribution of outcomes under a condition.

A future experiment should compare paired distributions such as:

```text
P(Y | do(control))
versus
P(Y | do(intervention))
```

Repeated runs of one error instance are not independent scientific samples. The design should distinguish:

- number of independent papers/tasks/error instances;
- repetitions within each instance;
- model/training/generation random seeds;
- provider/model versions and run batch;
- within-instance and between-instance variance.

Paired seeds and matched execution conditions should be used where meaningful.

## 9. Possible propagation outcomes

A future analyzer may classify effects such as:

- direct transmission;
- transformation;
- amplification;
- attenuation;
- masking;
- compensation;
- recovery;
- delayed manifestation;
- no measurable downstream effect.

An invalid intervention that improves one downstream metric remains scientifically invalid. Such a result may indicate metric gaming, compensation, evaluator misalignment, or another system bias rather than successful scientific reasoning.

## 10. Human annotation boundary

Human experts would be needed to define and calibrate scientific ground truth, but they should not need to intervene in every repeated run.

A potential workflow is:

```text
deterministic validators
-> evidence-aware automatic triage
-> independent human review for gold labels
-> adjudication of disagreement
-> automated paired runs and scoring
-> sampled post-run audit
```

LLM judges may assist triage but should not be the sole authority for domain-scientific correctness.

## 11. Relationship to Molly Core v2

Core v2 may retain low-cost metadata useful to this extension without implementing it:

- run_id / step_id;
- immutable artifact identity and SHA-256;
- input/output artifact references;
- tool and model versions;
- prompt/config digests;
- source locators;
- random-seed metadata where available;
- environment/version metadata.

These fields primarily serve provenance, reproducibility, stale-artifact detection, partial reruns, and BR1 current-run binding.

The dependency direction must be:

```text
error-propagation research extension
        -> Molly Core v2
```

Core v2 MUST NOT depend on research-extension schemas.

## 12. Activation gates

Implementation of this extension requires a separate owner/advisor decision. A future activation process should at minimum require:

- `R0`: advisor agrees the research question is worth pursuing;
- `R1`: scope and publication claim are frozen;
- `R2`: domain error taxonomy and annotation protocol are approved;
- `R3`: intervention and replay semantics are approved;
- `R4`: statistical analysis plan is frozen before large-scale experiments;
- `R5`: representative gold annotations and agreement evidence exist;
- `R6`: explicit Owner authorization to implement the extension.

Until all required research gates are satisfied, this proposal remains non-executable.

## 13. Explicit non-authorization

The presence of this document MUST NOT cause Codex or another coding agent to implement:

- InterventionEngine;
- ErrorInstance runtime;
- PairedRunGroup runtime;
- descendant-only counterfactual replay;
- propagation statistics;
- natural-failure benchmark infrastructure;
- cross-domain error-propagation abstractions.

Those capabilities require a separate research PR and explicit approval.
