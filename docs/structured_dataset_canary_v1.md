# Structured Dataset Canary v1

Structured Dataset Canary v1 implements the BR1 engineering canary for broader organic emitter PLQY data. Public CI does not establish a TADF discovery claim. Its final artifact is named `Computational Top-N` and represents model-ranked computational candidates only.

## Authority chain

The canary uses the existing Molly run directory, GateDecision store, `StageState`, Artifact Registry, verified immutable publications, Harness tracing seam and `AgentRunInspection v1`. It does not create a second Gate, registry, status, recovery ledger, model registry or telemetry authority.

`Raw Dataset is never training authority.` Raw CSV publication status is always `candidate_unconfirmed`. Review produces an immutable row roster and proposed actions. A trusted actor then approves the exact review snapshot through the shared training-config Gate; the confirmation receipt binds project, run, raw digest, review digest, included/excluded rows, target, role, condition policy, actor/source and decision digest.

`Training requires an exact-bound confirmation receipt and Confirmed Dataset publication.` Missing, stale, replaced or cross-scope authority fails closed. Client flags and LLM text cannot create a receipt.

## CI Reference Canary

The public path performs a real deterministic ridge fit on RDKit molecular descriptors from the current run's Confirmed Dataset. It records molecule, paper and external-holdout split assignments. Candidate generation executes a seed-bound deterministic molecule construction algorithm; it never reads a preset Top-N. Prediction consumes the current model checkpoint and current generated roster. Ranking is deterministic, displays OOD findings and excludes OOD candidates from Top-N according to an explicit rule.

Chemical validation records RDKit validity, canonical SMILES, Standard InChI/InChIKey, duplicates, training-set exact matches, nearest-neighbor similarity, scaffold novelty, AD/OOD status and a no-silent-loss summary.

The current run does not reuse an old model, prediction, generated candidate roster, or `existing_output`.

## Recovery and replay

Every request, checkpoint and publication has a semantic digest. Restart reads and verifies exact existing authority. A model checkpoint is adopted without refitting. A verified model or generation publication missing only its Controller completion receipt is reconciled by exact digest without re-execution. An unknown generation dispatch outcome enters recovery-required and is never dispatched again automatically. Replaced source bytes, registry conflicts and cross-run packages fail closed.

Exact replay uses canonical JSON and semantic digests that exclude observation timestamps and telemetry. Telemetry is non-authoritative and fail-open.

## Scientific and claim boundary

The frozen engineering scope is organic small-molecule emitters and PLQY. Metal complexes, polymers, exciplex systems, host generation and transport-material generation are excluded. The raw contract retains emission mechanism, medium, host, doping ratio, temperature, measurement condition, paper evidence and comparability. Values from different conditions are never silently merged.

The final output is `Computational Top-N`, not experimental validation or material discovery. It does not assert synthesis feasibility, EQE, lifetime, TADF identity, DFT validation or a closed materials-discovery loop.

No M3H Gate V, M3.5 completion, or Molly v1 completion is claimed unless the corresponding owner-reviewed runtime evidence exists.
