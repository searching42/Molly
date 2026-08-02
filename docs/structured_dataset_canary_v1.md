# Structured Dataset Canary v1

Structured Dataset Canary v1 implements the BR1 engineering canary for broader organic emitter PLQY data. Public CI does not establish a TADF discovery claim. Its final artifact is named `Computational Top-N` and represents model-ranked computational candidates only.

## Authority chain

The canary is compiled into five planner-visible `ScientificToolSpec` / RunPlan tasks: prepare, confirm, train, generate and evaluate. The CI fixture first creates a proposal, receives Permission, calls approve-and-start and then lets the Harness Controller dispatch one task at a time through `RunPlanExecutor`. Task backends cannot register artifacts, write `StageState`, append Gate decisions, create Controller receipts or reconcile recovery. Those facts remain owned by Molly's existing authority chain.

Every backend input is the exact immutable path supplied by `RunPlanExecutor` from the current Artifact Registry/input manifest. Raw and Confirmed CSV content are separate registered artifacts (`raw_dataset_csv` and `confirmed_training_dataset_csv`), while their JSON publications bind the corresponding content digests. Evaluation also consumes the registered current checkpoint and generated candidate roster. No stage may rediscover an input through a fixed canary directory or an unregistered sibling file; path substitution, stale self-consistent siblings and publication/content mismatches fail closed.

`Raw Dataset is never training authority.` Raw CSV publication status is always `candidate_unconfirmed`. Review produces an immutable row roster and proposed actions. A trusted actor then approves the exact review snapshot through the shared training-config Gate; the confirmation receipt binds project, run, raw digest, review digest, included/excluded rows, target, role, condition policy, actor/source and decision digest.

`Training requires an exact-bound confirmation receipt and Confirmed Dataset publication.` Missing, stale, replaced or cross-scope authority fails closed. Client flags and LLM text cannot create a receipt.

## CI Reference Canary

The public path performs a real deterministic ridge fit on RDKit molecular descriptors from the current run's Confirmed Dataset. Split assignment operates on connected components of the InChIKey–paper bipartite graph; records sharing either a molecule or paper never cross train, test or external holdout, and insufficient independent components fail closed. Candidate generation executes a seed-bound deterministic molecule construction algorithm; it never reads a preset Top-N. Prediction consumes the current model checkpoint and current generated roster. Ranking is deterministic, displays OOD findings and excludes OOD candidates from Top-N according to an explicit rule.

Chemical validation records RDKit validity, canonical SMILES, Standard InChI/InChIKey, duplicates, training-set exact matches, nearest-neighbor similarity, scaffold novelty, AD/OOD status and a no-silent-loss summary.

The current run does not reuse an old model, prediction, generated candidate roster, or `existing_output`.

## Recovery and replay

Every request, checkpoint and publication has a semantic digest. Training request, current Controller dispatch receipt, checkpoint, Registry outputs, verified local publication and Controller action receipt form one exact chain. A pre-existing/copied checkpoint or request is rejected by the backend. If a dispatch occurred but the Controller publication/receipt is missing, only the existing Controller recovery verifier may reconcile it; the backend never adopts or redispatches it. Generation follows the same Controller-owned unknown-outcome rule and has no canary-specific dispatch ledger. Replaced source bytes, registry conflicts and cross-run packages fail closed.

Exact replay verifies the succeeded current Controller execution and its published artifacts without invoking training or generation again. Canonical JSON and semantic digests exclude telemetry. Telemetry is non-authoritative and fail-open.

## Scientific and claim boundary

The frozen engineering scope is organic small-molecule emitters and PLQY. Metal complexes, polymers, exciplex systems, host generation and transport-material generation are excluded. The raw contract retains emission mechanism, medium, host, doping ratio, temperature, measurement condition, paper evidence and comparability. Values from different conditions are never silently merged.

The final output is `Computational Top-N`, not experimental validation or material discovery. It does not assert synthesis feasibility, EQE, lifetime, TADF identity, DFT validation or a closed materials-discovery loop.

No M3H Gate V, M3.5 completion, or Molly v1 completion is claimed unless the corresponding owner-reviewed runtime evidence exists.
