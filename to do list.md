# AI4S Agent Roadmap And TODO

Generated after brainstorming on 2026-05-27.

This document captures the agreed direction for `workspace/agent` as the final mainline. `workspace/claude` and `workspace/oled-agent` are retained as legacy/reference folders. Do not merge either legacy folder wholesale into `workspace/agent`; migrate selected modules only after adapting contracts and removing hardcoded assumptions.

## 0. Final Positioning

- Build toward a production-oriented AI4S agent, not a narrow OLED demo.
- Primary near-term positioning is a data-centric AI4S agent.
- Engineering usability is the first priority; auditability and paper potential are secondary design constraints.
- Phase 1 focuses on a local single-user `localhost` app.
- Phase 1 main loop is data cleaning, model training, candidate prediction, filtering/ranking, and reporting.
- Phase 1 supports both end-to-end workflows and composable atomic tasks.
- Phase 1 reserves interfaces for generation and literature mining, but does not implement those loops.
- Phase 2 adds candidate generation and inverse design loops.
- Phase 3 adds literature mining and structured data extraction.
- Phase 4 moves from fixed workflow orchestration toward an autonomous research assistant / AI co-scientist style workflow.
- Keep the current workflow engine as the safe execution substrate; add LLM-driven planning, observation, reflection, and replanning above it.
- The agent should not replace gates, artifacts, or adapters with opaque free-form actions.
- RAG, MCP, skills, and other agent technologies are extension layers, not Phase 1 hard dependencies.
- Future RAG work should support evidence-grounded literature-to-dataset workflows, not generic chat-only RAG.
- `workspace/agent` is the canonical mainline for new design, code, docs, and tests.
- `workspace/claude` remains as a legacy backend and regression reference.
- `workspace/oled-agent` remains as a legacy/reference implementation; reuse selected modules by copying and adapting them into `workspace/agent`.

## 1. Workspace Migration TODO

- Keep `workspace/agent` as the final mainline.
- Keep `workspace/claude` as legacy backend and regression baseline.
- Keep `workspace/oled-agent` as legacy/reference, similar to `claude`.
- Do not directly move `oled-agent` into `agent`.
- Copy selected `oled-agent` files into `agent` only when needed.
- Modify copied `oled-agent` code before or during migration to remove OLED-specific assumptions.
- Avoid importing `oled-agent` as a runtime dependency for the final mainline.

### 1.1 Reuse From `workspace/claude`

- Reuse real working data cleaning and alignment logic.
- Reuse `clean_dataset.py` behavior as the first implementation reference.
- Reuse `prepare_training_entry_from_prompt.py` behavior as the first data confirmation reference.
- Reuse `run_mvp_flow.py` only as a temporary legacy full-flow fallback.
- Reuse job status and log ideas from `job_manager.py`.
- Reuse NL parsing improvements from `nl_task_parser.py`, especially percent normalization.
- Reuse `generate_statistics_and_summary.py` report generation ideas.
- Reuse current remote Uni-Mol training path until `workspace/agent` has native adapters.
- Keep `claude/ui` as a UI reference only, not the long-term UI mainline.

### 1.2 Reuse From `workspace/oled-agent`

- Reuse adapter JSON-in/JSON-out contract ideas.
- Reuse adapter validation ideas from `scripts/adapters/validate_adapter_contract.py`.
- Reuse runtime helper patterns from `scripts/adapters/runtime_helpers.py`.
- Reuse model catalog concepts from `src/oled_agent/agent/model_catalog.py`.
- Reuse planner provider split, especially rule-based provider plus optional LLM provider.
- Reuse execution budget and resume concepts from `executor.py`, `session.py`, and `step_runner.py`.
- Reuse diagnostics concepts from `diagnostics.py`.
- Reuse evaluation, guardrails, experiment trace, and memory context concepts.
- Reuse UI ideas for task timeline, artifact preview, compare, bundle export, retry, and snapshot.
- Do not reuse OLED-specific schemas directly without modification.
- Remove hardcoded `lambda_em`, `plqy`, and `stability` enum restrictions.
- Replace generation-first workflow assumptions with data-confirmation-first workflow assumptions.
- Change fallback/stub semantics to fit the Phase 1 strict confirmation policy.

## 2. Phase 1 Product Scope

- Build a local single-user `localhost` app.
- Pre-design APIs so a later phase can add a remote worker.
- Use a new lightweight UI in `workspace/agent`.
- Use `claude/ui` only as a reference for features and behavior.
- Main UI pattern is Chat + Wizard Cards + Stage Timeline.
- Chat handles intent, explanation, and follow-up questions.
- Wizard Cards handle structured confirmation.
- Stage Timeline shows progress, logs, failure location, retry state, and artifacts.

### 2.1 Phase 1 Main Workflow

- User creates or selects a project.
- User uploads `train_dataset`.
- User optionally uploads `evaluation_dataset`.
- User uploads `candidate_dataset` or derives candidates from the cleaned master dataset.
- User describes requirements in natural language.
- Agent parses intent into `TaskSpec`.
- Agent validates uploads.
- Agent drafts cleaning rules.
- User confirms data rules and training readiness.
- Agent executes cleaning.
- Agent builds property catalog.
- Agent resolves requested target properties from natural language against the property catalog and alias rules, instead of assuming a fixed metric list.
- Agent prepares a `TargetModelingBrief` for each requested trainable target before deciding preprocessing and hyperparameters.
- The brief uses project memory, previous run diagnostics, built-in domain rules, and optional user-approved web/literature search to identify target-specific risks such as solvent dependence, bounded targets, noisy labels, unit conventions, split strategy, recommended target transforms, and backend/hyperparameter defaults.
- Agent checks trainability.
- Agent explains which requested properties are trainable, blocked, missing, or need more data, and only trains/evaluates the feasible subset after confirmation.
- Agent runs lightweight baseline.
- Agent recommends backend.
- User confirms backend and run plan.
- Agent trains model or models.
- Agent diagnoses training results against baselines, target-specific expectations, split leakage checks, prediction distribution, fold stability, and known target risks.
- If training quality is weak or degraded, Agent proposes a reviewable `RerunProposal` before prediction, such as more data, revised cleaning, target transform, solvent/context conditioning, different split, seed ensemble, backend switch, or low-confidence model use.
- Agent predicts candidate properties.
- Agent applies hard constraints, soft constraints, weighted score, and ranking.
- Agent renders reports.
- User optionally saves confirmed rules and models into project memory.

### 2.2 Atomic Task Support

End-to-end workflows are predefined compositions of atomic tasks.

Phase 1 atomic tasks:

- `inspect_dataset`
- `clean_dataset`
- `prepare_modeling_brief`
- `check_trainability`
- `run_baseline`
- `train_model`
- `diagnose_model`
- `predict_candidates`
- `filter_rank`
- `render_report`

Atomic task rules:

- Users may request a full workflow or a single atomic task.
- The natural-language entry and advanced toolbox both produce a `RunPlan`.
- If a requested atomic task lacks required artifacts, planner generates a dependency expansion plan.
- Dependency expansion must show missing artifacts and reasons before execution.
- Existing project assets should be recommended for reuse before rerunning dependencies.
- High-risk dependency tasks still require user confirmation.
- Full workflows may pause at key gates and continue later from saved artifacts.
- At each high-risk gate users can continue, save and stop, modify the plan, or cancel.
- Phase 1 uses linear workflow execution with dependency expansion, not a full DAG engine.
- Keep `task_id` and `depends_on` fields available for future DAG upgrade.

### 2.3 Out Of Scope For Phase 1

- Automatic literature crawling.
- Automatic public dataset ingestion.
- Full REINVENT4 generation loop.
- Full multi-task model training.
- Classification or ranking task training.
- Multi-user server deployment.
- Fully remote worker orchestration.
- Automatic complete CSV upload to external LLM.
- Automatic proxy property substitution.

## 3. Phase 1 Storage Structure

Use project-based storage:

```text
workspace/agent/projects/<project_id>/
  project_memory.json
  assets/
    datasets/
    models/
    rules/
    reports/
  runs/<run_id>/
    stage.json
    task_info.json
    01_intake/
    02_data/
    03_training/
    04_screening/
    05_report/
    logs/
```

- Keep `workspace/agent/runs/<run_id>` only as a compatibility index or transition path.
- Stage directories are the primary run organization.
- Each stage can contain JSON, Markdown, HTML, CSV, and logs.
- Markdown/HTML reports are the main human-facing artifacts.
- JSON artifacts are the machine-readable source of truth for recovery, tests, and memory.

Project assets:

```text
projects/<project_id>/assets/
  datasets/
    raw/<dataset_id>/v001/
    cleaned/<scope>/v001/
    candidates/<dataset_id>/v001/
  models/
    <property_name>/<backend>/v001/
  rules/
    cleaning/v001/
    property_alias/v001/
    unit_mapping/v001/
  reports/
    <run_id>/
```

Asset versioning:

- Use `v001`, `v002`, `v003` versions.
- Run artifacts are always kept in the run directory.
- Project assets are confirmed, reusable versions of run artifacts.
- New confirmations create new versions; do not overwrite old versions.
- Prefer `deprecated` status over physical deletion.
- Physical deletion requires explicit user confirmation.

Every project asset must include `asset_manifest.json` with:

- `asset_id`
- `asset_type`
- `version`
- `status`: `candidate`, `confirmed`, or `deprecated`
- `created_from_run_id`
- `source_artifacts`
- `content_hash`
- `schema_version`

Asset promotion:

- Low-risk run artifacts are saved automatically.
- Scientific decision artifacts require user confirmation before becoming project assets.
- Use `run artifact -> user review -> project asset` promotion.
- Examples: `cleaned_train.csv`, trained model directories, property aliases, unit mappings, and confirmed cleaning rules.

## 4. Phase 1 State Machine

Use this default stage sequence:

```text
parse
upload_validate
cleaning_draft
user_confirm_data
clean_execute
target_modeling_brief
trainability
user_confirm_run
train
model_diagnostics
user_confirm_rerun
predict
filter_rank
report
memory_update
done/error
```

Each `stage.json` update must include:

- `stage`
- `next_stage`
- `status`
- `started_at`
- `ended_at`
- `updated_at`
- `error`
- `details`
- `artifacts`
- `history`

Status values should include:

- `PENDING`
- `RUNNING`
- `WAITING_USER`
- `PAUSED_BY_USER`
- `SUCCEEDED`
- `DEGRADED`
- `FAILED`
- `SKIPPED`
- `CANCELLED`
- `DONE`

`user_confirm_rerun` is skipped when `model_diagnostics` accepts the trained
model for the requested use case. It becomes `WAITING_USER` when the diagnostics
show weak metrics, collapsed prediction range, unstable folds, leakage risk,
target-specific failure modes, or a proposed rerun that changes preprocessing,
hyperparameters, backend, data requirements, or confidence policy.

Phase 1 execution model:

- Maintain an atomic task registry.
- Each atomic task declares required artifacts, output artifacts, risk level, gates, and default adapter.
- Planner expands missing dependencies into a linear plan.
- Executor runs the expanded plan stage by stage.
- Full DAG scheduling is out of scope for Phase 1.
- Keep enough dependency metadata to migrate to a DAG engine later.

Atomic task spec minimum fields:

- `task_name`
- `task_id`
- `depends_on`
- `required_artifacts`
- `output_artifacts`
- `risk_level`
- `gates`
- `default_adapter`

Run plan modification:

- Users may modify constraints, TopN, target properties, backend, input artifacts, and whether to continue later stages.
- Full DAG visual editing is deferred.
- Every modification regenerates `RunPlan`, shows a diff, and requires confirmation before execution.

## 5. Phase 1 Core Schemas

Implement Python internal models with Pydantic and export or maintain JSON Schema for API, artifact, LLM, and adapter contracts.

- `TaskSpec`
- `DatasetSpec`
- `CleaningRuleSet`
- `PropertyCatalog`
- `TrainabilityReport`
- `CandidateReadinessReport`
- `BaselineReport`
- `ReadinessAssessment`
- `ModelBackendSpec`
- `BackendRecommendation`
- `RunPlan`
- `AtomicTaskSpec`
- `StageState`
- `RunReport`
- `AssetManifest`
- `AssetPromotion`
- Future `EvidenceRecord`
- Future `SourceSpan`
- Future `ConditionContext`
- Future `ConflictReport`
- Future `LLMDataPolicy`

Schema requirements:

- Do not hardcode OLED-only property enums.
- Allow arbitrary property names discovered from uploaded data.
- Represent property aliases, source columns, units, scale, value ranges, and trainability status.
- Represent hard constraints, soft constraints, ranking objectives, weights, and TopN.
- Represent dataset role as `train`, `evaluation`, or `candidate`.
- Represent candidate source as `uploaded`, `derived_from_master`, or future `generator`.
- Represent backend as `unimol`, `baseline_rf`, `baseline_xgboost`, or future backends.
- Represent training mode as `single_task` or future `multi_task`.
- Represent task type, with Phase 1 supporting only numeric regression.
- Represent atomic task mode: `full_screening`, `inspect_only`, `clean_only`, `trainability_only`, `baseline_only`, `train_only`, `predict_only`, `rank_only`, and `report_only`.
- Represent artifact dependencies, dependency expansion reason, and asset promotion decision.
- Reserve evidence and source-span schemas for Phase 3 literature extraction.

## 6. Phase 1 Adapter Contracts

Minimum adapter/tool contracts:

- `parse_task`
- `inspect_dataset`
- `draft_cleaning_rules`
- `execute_cleaning`
- `check_trainability`
- `run_baseline`
- `recommend_backend`
- `train_model`
- `predict_candidates`
- `filter_rank`
- `render_report`

Phase 3 reserved adapter contracts:

- `parse_document`
- `index_corpus`
- `retrieve_evidence`
- `extract_records`
- `normalize_extractions`
- `detect_literature_conflicts`

Adapter rules:

- Use JSON-in/JSON-out.
- Write logs to stderr or log files, not stdout.
- Return structured errors.
- Return explicit artifact paths.
- Preserve original inputs and normalized outputs.
- Validate adapter input/output with contract tests.
- Keep a temporary `legacy_full_flow_adapter` around for `claude/scripts/run_mvp_flow.py` fallback.

Native vs wrapper boundary:

- Native first: `inspect_dataset`, `check_trainability`, `run_baseline`, `recommend_backend`, `render_report`, state machine, artifact store, API/schema, UI planning/status.
- Wrapper first: `execute_cleaning`, `train_unimol`, remote training, and legacy full-flow fallback.
- Legacy wrapper outputs must be converted into standard `workspace/agent` artifacts.
- Wrapper code must not leak hardcoded `lambda_em`, `plqy`, or `mw` assumptions into core schemas.
- Target-property selection must stay dynamic: the user states goals in natural language, the agent maps them onto the available property catalog, and trainability plus backend recommendation decide what can actually be trained.

## 7. Data Upload And Dataset Roles

Phase 1 supports three dataset roles:

- `train_dataset`: required for training.
- `evaluation_dataset`: optional; if absent, train/valid/test split is generated from training data.
- `candidate_dataset`: used for final prediction and screening.

UI defaults:

- Show train and candidate uploads prominently.
- Put evaluation dataset under advanced options.
- Remove absolute path inputs from main UI where possible; prefer file upload/selection.
- Allow provenance fields for URL, DOI, PDF path, and source notes.
- Do not automatically crawl or ingest source URLs in Phase 1.

Candidate dataset policy:

- Phase 1 supports uploaded candidate sets and candidates derived from cleaned master dataset.
- Phase 1 does not run generators.
- Future generator candidates must pass through the same prediction and filtering chain.

## 8. Data Confirmation Policy

Every new training dataset requires user confirmation before training.

The data confirmation card has five sections:

- Data overview: row count, column count, SMILES column, duplicate count, invalid count.
- Property catalog: label count, missing rate, unit, scale, value range for every property.
- Cleaning rule draft: unit conversion, duplicate aggregation, outlier handling, split strategy.
- Trainability: status and reason for every requested property.
- Confirmation actions: execute cleaning, edit mapping, save project rule, cancel.

Candidate dataset confirmation:

- Phase 1 performs automatic validation.
- User confirmation is required if warnings or blockers exist.
- Full product should always confirm Candidate Readiness Report.

## 9. Cleaning Rules

Phase 1 cleaning is human-in-the-loop.

- Agent drafts `CleaningRuleSet`.
- User confirms before execution.
- Confirmed rules apply to current run by default.
- User can choose to save confirmed rules into project memory.
- Saved project rules must display source and scope when reused.

Unit and scale:

- Do not auto-convert labels before confirmation.
- Suggest common conversions such as `80% -> 0.8`.
- Support Chinese percent expressions such as `百分之八十`.
- Record every unit/scale conversion in `CleaningRuleSet`.

Duplicate samples:

- Same SMILES and same property with small label variance: suggest median aggregation.
- Same SMILES and same property with large label variance: mark label conflict and require confirmation.
- Same SMILES with different properties: allow merge into multi-property row.
- Report original rows, deduped rows, aggregation count, conflict count, and rejected rows.

Outliers:

- Generate handling suggestions.
- Do not delete silently.
- User can choose keep, delete, clip, or winsorize.
- Known properties use physical or empirical ranges first.
- Unknown properties use IQR or quantile detection as warning only.

Splits:

- Use provided split if valid.
- If no split exists, allow deterministic hash split.
- Record split strategy and counts.

## 10. Trainability Rules

Phase 1 supports numeric regression only.

Trainability statuses:

- `TRAIN_READY`: effective labels `>= 100`.
- `TRAIN_WITH_WARNING`: effective labels `30-99`.
- `INSUFFICIENT_LABELS`: effective labels `< 30`.
- `NEEDS_MAPPING`: property mapping is missing or uncertain.
- `INVALID_LABELS`: labels cannot be parsed reliably.
- `UNSUPPORTED_TASK_TYPE`: non-numeric/classification/ranking in Phase 1.
- `UNSUPPORTED_BACKEND`: backend cannot train this target.
- `BLOCKED_ENV`: environment or remote runtime is unavailable.

Training behavior:

- Training always requires user review before submission.
- `TRAIN_WITH_WARNING` may proceed if user clearly says this is a test or smoke run.
- If the user asks for formal, strict, or production-quality results, warning properties require explicit confirmation.
- If user intent is unclear, warning properties require explicit confirmation.
- If any requested property is blocked, default behavior is to stop and ask user how to revise the task.

Long-term note:

- `100` labels is only a Phase 1 workflow threshold.
- For convincing scientific models, more labels are strongly preferred.
- Even `300` labels can be low depending on property complexity and data noise.

## 11. Baseline And Backend Recommendation

Phase 1 does not automatically decide that Uni-Mol should or should not be used.

Backend decision flow:

```text
Property Catalog
-> TargetModelingBrief
-> Trainability Report
-> Morgan/ECFP + XGBoost/RF baseline
-> 3D relevance and feasibility check
-> Backend Recommendation
-> User confirmation
-> Training submission
-> ModelDiagnosticsReport
-> Accept model or propose RerunProposal
```

Target modeling brief:

- Build one brief per requested target property after property resolution and before backend recommendation.
- Read confirmed project memory first, including property aliases, accepted cleaning rules, prior model diagnostics, known dataset caveats, and user-approved backend preferences.
- Use built-in target heuristics for common modeling risks, for example bounded targets, log-scale targets, unit conversions, solvent/context dependence, device-structure dependence, class imbalance, censored values, high replicate variance, or target leakage.
- If local memory and built-in rules are insufficient, propose a web/literature search plan and ask for confirmation before external acquisition or sending private data outside the machine.
- The brief should recommend preprocessing, split strategy, label aggregation policy, target transform, baseline family, backend candidates, hyperparameter ranges, expected metric scale, and acceptance thresholds.
- Record why each recommendation was made and which source produced it: memory, built-in rule, current dataset statistics, prior run, or external evidence.

Baseline:

- Use Morgan/ECFP fingerprints.
- Prefer XGBoost when available.
- Use RandomForest fallback when XGBoost is unavailable.
- Use scaffold split by default when SMILES and RDKit are available.
- Fall back to random split when scaffold split is unavailable or unstable.
- Record `split_strategy`, fallback reason, train/valid size, and random seed.
- Baseline unavailable should not block the main workflow; record the reason.
- Baseline metrics are used for sanity checking and recommendation only.
- Baseline can be saved as a project model only if user confirms.
- Baseline report must record backend, parameters, split, metrics, fallback reason, and model artifact paths.

Baseline artifacts:

- `baseline_report.json`
- `baseline_report.md`
- `model_metrics.json`
- `predictions_val.csv`

Readiness assessment:

```text
ReadinessAssessment
  data_readiness: READY | WARNING | BLOCKED
  model_readiness: PROMISING | UNCERTAIN | WEAK | NOT_EVALUATED
  recommendation: train_baseline | train_unimol | collect_more_data | review_data
```

- Data readiness comes from trainability and cleaning results.
- Model readiness comes from baseline metrics and backend feasibility.
- Data-level `BLOCKED` prevents training.
- Weak baseline metrics are advisory and do not hard-block training.
- Users may proceed through warnings after explicit confirmation.

Post-training model diagnostics:

- Compare trained model metrics against scaffold/random mean baselines, Morgan/ECFP baseline, and any target-specific expected metric scale from `TargetModelingBrief`.
- Inspect per-fold metrics, prediction-vs-target distribution, prediction range collapse, target-bucket bias, high-error samples, split leakage checks, conformer/featurization failure rates, and train/validation learning curves.
- Produce an explicit use-case judgment: `ACCEPT`, `ACCEPT_LOW_CONFIDENCE`, `RERUN_RECOMMENDED`, or `BLOCKED`.
- For weak or degraded models, emit a `RerunProposal` with concrete candidate changes, expected rationale, extra cost, required approvals, and a fallback policy if rerun still fails.
- Never silently rerun expensive training, switch backend, relax constraints, use external search, or promote a weak model without user confirmation.

3D relevance:

- Do not call it a conformer variance label test unless conformer-level labels or computations exist.
- Use it as 3D relevance and feasibility evidence.
- Check whether conformers can be generated.
- Check flexibility, stereochemistry, and property type.
- Treat HOMO/LUMO, dipole, conformer energy, binding-like, steric, or electrostatic properties as more 3D-relevant.
- Treat simple topology-dominated properties as potentially baseline-sufficient.
- Use a lightweight property ontology and property name, alias, description, and unit parsing.
- Output `high`, `medium`, `low`, or `unknown` with confidence and evidence.
- Allow user override for warnings and recommendations.
- Do not allow user override for data-layer blockers such as missing labels, invalid labels, missing structure column, unconfirmed cleaning rules, or missing runtime credentials.

Backend recommendation:

- Generate recommendations per property.
- Show available backends, recommended backend, evidence, metrics, and risks.
- Phase 1 does not allow backend mixing inside one run.
- If different properties recommend different backends, UI asks user to select one backend route for this run.
- Default backend flow is baseline sanity check first, then user confirmation.
- Use data size, data quality, property type, 3D relevance, baseline metrics, runtime cost, and user goal.
- Implement backend recommendation as transparent rules in Phase 1, not a black-box model.
- If user says test or smoke, recommend baseline.
- If user says formal, reliable, 3D, or Uni-Mol, recommend Uni-Mol.
- If user intent is unclear, recommend Uni-Mol but show baseline and data risks first.

Backend recommendation artifacts:

- `backend_recommendation.json`
- `backend_recommendation.md`
- Include recommended backend, alternatives, reasons, warnings, required confirmations, estimated cost, and expected outputs.

## 12. Model Training And Prediction

Phase 1 training:

- Use single-task training only.
- Train one model per property.
- Reserve `training_mode=multi_task` for future.
- Do not overwrite existing models.
- Every training run writes to a new model directory.
- User can manually register a trained model as a project model.
- Overwriting baseline, deleting models, or replacing default project model requires confirm-each-time.

Backends:

- Uni-Mol can attempt arbitrary numeric regression properties, but new properties must be marked as unvalidated.
- Baseline RF/XGBoost can train and predict candidate properties.
- Same run uses one backend route, not mixed backend per property.

Prediction:

- Candidate prediction must record model id, backend, training run id, property, prediction column, and score column.
- Generated candidates in future phases must be predicted by the same prediction chain before filtering.

## 13. Constraints, Ranking, And Reports

Constraint types:

- Hard constraints filter out candidates.
- Soft constraints affect score or warnings, not direct filtering.
- Ranking objectives participate in weighted score.

NL semantics:

- "must", "cannot", "strict", "at least", "not above", "必须", "不能", "不得", "严格", "不超过", "至少" imply hard constraints.
- "hope", "prefer", "better", "try to", "希望", "最好", "尽量", "优先", "偏好" imply soft constraints or ranking objectives.
- "maximize", "higher is better", "越高越好", "最大化", "尽量高" imply ranking objective.
- "near", "around", "between X and Y", "接近", "附近", "在 X 到 Y" imply target window or range, confirmed by user.
- Percent expressions must be normalized and shown with unit/scale.

Ranking:

- Use weighted score as the primary TopN ordering in Phase 1.
- Add Pareto/trade-off explanation in the report.
- Do not use Pareto as the primary ranking in Phase 1.

Reports:

- Markdown/HTML are primary human-facing reports.
- JSON is generated in parallel for machine use.
- UI renders artifacts and should not contain hidden report logic.
- Each atomic task emits JSON as source of truth and a Markdown summary for human review.
- HTML is rendered on demand or aggregated in the full workflow report.

Required Phase 1 reports:

- `dataset_overview.md/html`
- `cleaning_rule_draft.md/html`
- `target_modeling_brief.md/html`
- `trainability_report.md/html`
- `candidate_readiness_report.md/html`
- `baseline_report.md/html`
- `backend_recommendation.md/html`
- `training_report.md/html`
- `model_diagnostics_report.md/html`
- `rerun_proposal.md/html` when applicable
- `screening_report.md/html`
- `final_summary.md/html`

Atomic report examples:

```text
inspect_dataset/
  dataset_profile.json
  dataset_profile.md

check_trainability/
  trainability_report.json
  trainability_report.md

prepare_modeling_brief/
  target_modeling_brief.json
  target_modeling_brief.md

train_model/
  training_report.json
  training_report.md
  model_metrics.json

diagnose_model/
  model_diagnostics_report.json
  model_diagnostics_report.md
  rerun_proposal.json
  rerun_proposal.md
```

## 14. Memory Policy

Phase 1 memory:

- Always save run memory.
- Save confirmed project memory only after user confirmation.
- Do not save raw data in project memory.
- Do not automatically learn preferences in Phase 1.

Run memory stores:

- Original prompt.
- Final structured task.
- Uploaded file references.
- Cleaning rules.
- Confirmation decisions.
- Stage history.
- Target modeling briefs and their sources.
- Training reports.
- Model diagnostics and rerun decisions.
- Screening reports.
- Error and retry records.

Project memory may store:

- Confirmed data column mapping rules.
- Confirmed property aliases.
- Confirmed unit and scale rules.
- Confirmed target-specific modeling notes, such as solvent/context dependence, target transforms, aggregation policy, split strategy, and acceptance thresholds.
- Default model choices.
- Confirmed backend preferences.
- Confirmed rerun policies and prior model diagnostic summaries, including metric baselines and known failure modes.
- Confirmed property 3D relevance overrides.
- Remote runtime preferences.
- Default strictness preferences.
- Common output preferences.
- Source run id, confirmer, confirmation time, and last-used time for saved preferences.

Project memory reuse:

- Saved project memory is shown as a recommendation or default draft in future runs.
- Saved preferences do not bypass confirmation gates.
- Users can use, change, or forget saved preferences.
- Do not implement automatic cross-project memory reuse.
- Phase 1 does not implement memory package import/export.
- Future versions may support manual memory package export/import with explicit diff and confirmation.

Future note:

- Expert mode is not implemented in Phase 1.
- Future expert mode may reuse confirmed rules more aggressively.
- Low-risk preference auto-learning can be added later, but must be visible, editable, and deletable.

## 15. Permissions

Use three permission levels:

- `auto`
- `project-approved`
- `confirm-each-time`

Phase 1 `auto` operations:

- Read local files.
- Inspect headers.
- Generate cleaning drafts.
- Run local lightweight analysis.
- Execute local cleaning after user confirmed rules.
- Validate SMILES.
- Deduplicate.
- Generate statistics and charts.
- Generate trainability/readiness reports.
- Read historical artifacts.

Phase 1 `project-approved` operations:

- Reuse a fixed remote runtime configuration.
- Reuse a fixed model output base directory.
- Reuse confirmed non-sensitive project preferences.

Always confirm each time:

- Submit training.
- Run remote GPU jobs.
- Use external LLM.
- Use external web/MCP tools.
- Upload raw data to external services.
- Overwrite baseline.
- Delete model or artifact files.
- Replace default model.
- Relax hard constraints.
- Save project memory rules.

Complete-agent technology policy:

- RAG, MCP, and skill-like procedural modules are valid future extension layers.
- Phase 1 should keep adapter and permission boundaries compatible with these technologies.
- Product runtime should use internal `ToolAdapter`, `ModelBackend`, and recipe contracts instead of depending on Codex skill files.

## 16. External LLM And Public Data Policy

Phase 1 default:

- External LLM is off by default.
- Rules-based parser runs first.
- Low-confidence or complex parsing can prompt user to enable LLM assist.

Dataset privacy:

- Add `dataset_public` switch.
- Default is `dataset_public=false`.
- If false, external LLM can receive prompt, schema, headers, and aggregate statistics only.
- If true, external LLM can receive richer summaries and small sample rows.
- Full raw CSV upload to external LLM still requires separate confirmation.
- Long-term versions may consider full public CSV upload with license and audit checks.

LLM role:

- LLM can suggest mappings, unit interpretations, and task structure.
- LLM cannot be the final authority for counts, cleaning execution, training readiness, or model metrics.
- Record what context was sent to LLM and what it returned.

Phase 3 literature extraction default:

- External LLM may be enabled by default for literature extraction.
- Uploading PDF, full text, tables, or evidence chunks to external LLM requires user confirmation.
- Default external LLM payload should be limited to evidence chunks unless the user explicitly approves full text or PDF upload.
- Track `LLMDataPolicy` with payload level and confirmation record.

LLM payload levels:

- `metadata_only`
- `schema_and_query`
- `evidence_chunks`
- `tables`
- `full_text`
- `full_pdf`

Literature extraction audit trail:

- Save query.
- Save retrieval hits.
- Save prompt template id or prompt summary.
- Save payload policy.
- Save LLM output.
- Save parser output.
- Save parser errors.
- Save human corrections.
- Save final promotion decision.
- For sensitive payloads, save hash or source reference instead of copying full text.

## 17. Error Taxonomy

Use these categories:

- `VALIDATION`
- `DATA`
- `TRAINABILITY`
- `MODEL`
- `REMOTE`
- `RESOURCE`
- `PERMISSION`
- `ARTIFACT`
- `EXTERNAL`
- `UNKNOWN`

Every failure record must include:

- Error category.
- User-readable reason.
- Technical details.
- Retryability.
- Suggested next action.
- Related artifact paths.
- Related log paths.

## 18. Job Execution

Phase 1 uses a local background job manager.

- Submit returns `project_id`, `run_id`, and `job_id`.
- UI polls job state, `stage.json`, and recent logs.
- Cleaning and reporting can run concurrently.
- Training and remote GPU tasks are serialized by a simple resource lock.
- Multiple jobs may exist, but only one active training/remote GPU job runs at a time.
- Future remote worker or formal queue should keep the same public API shape.

Recovery:

- Phase 1 supports retry from the latest failed stage or explicitly retryable stage.
- Phase 1 does not support arbitrary stage rerun.
- Full product can support arbitrary stage rerun with artifact version and dependency checks.

## 19. UI TODO

- Build new lightweight UI inside `workspace/agent`.
- Keep `claude/ui` only as reference.
- Use Chat + Wizard Cards + Stage Timeline.
- Add Advanced Toolbox for atomic tasks.
- Default to simple interface.
- Keep advanced options collapsible.
- Show uploaded train and candidate files at the top.
- Put optional evaluation dataset under advanced options.
- Remove absolute path inputs from the main UI.
- Show task summary by default.
- Allow expanding structured `TaskSpec`.
- Allow editing structured task and regenerating plan.
- Show detected task mode and atomic task expansion.
- Show dependency plan when required artifacts are missing.
- Show `RunPlan` diff after user edits constraints, backend, target properties, input artifacts, or continuation mode.
- Show data confirmation card with the five fixed sections.
- Show trainability and backend recommendation before training.
- Show warning and blocker states clearly.
- Support continue, save and stop, modify plan, and cancel at high-risk gates.
- Show asset promotion actions for cleaned datasets, rules, models, and reports.
- Show latest logs and refresh in real time or via refresh button.
- Show artifact links and report previews.
- Show retry button only for retryable failures.

Atomic task toolbox entries:

- Inspect Dataset
- Clean Dataset
- Check Trainability
- Run Baseline
- Train Model
- Predict Candidates
- Filter And Rank
- Render Report

## 20. Testing And Acceptance

Default CI or fast local tests:

- Schema validation.
- Adapter contract validation.
- Dry-run workflow.
- Baseline workflow on small data.
- Atomic task dependency expansion.
- Asset manifest validation.
- Asset promotion decision validation.
- UI smoke.
- Stage history validation.
- Report artifact presence validation.
- Error taxonomy validation.

Manual or release acceptance:

- At least one real remote Uni-Mol training run.
- Candidate prediction from trained model.
- Filtering/ranking output.
- Markdown/HTML final report.
- Full stage history.
- Retry/failure evidence where feasible.

Phase 1 completion gate:

- At least one manually successful remote Uni-Mol closed-loop run is required before Phase 2.
- Default CI does not run remote training.
- Dry-run/baseline/workflow stability must be tested with repeated runs.

Milestone 1 acceptance:

- User can upload a training dataset.
- System can inspect the dataset natively.
- System can generate `PropertyCatalog`.
- System can produce a cleaning draft or passthrough decision.
- System can generate `TrainabilityReport`.
- System can run Morgan/ECFP + XGBoost baseline or RF fallback.
- System can generate `BackendRecommendation`.
- System can render Markdown report.
- UI can show run plan, warnings, blockers, and report links.

Phase 1 full acceptance:

- User can confirm training.
- System can train baseline model or call Uni-Mol wrapper.
- System can predict candidate dataset.
- System can filter, rank, and export TopN.
- System can render final Markdown/HTML report.
- System can optionally promote confirmed datasets, models, and rules to project assets.
- UI supports atomic task toolbox.

Suggested stability check:

- Repeat same dry-run or baseline input three times.
- Confirm key artifacts are generated each time.
- Confirm deterministic TopN overlap meets threshold.
- Confirm stage history has no unexpected jumps.
- Confirm all JSON schemas validate.
- Confirm report paths are correct.

## 21. Phase 1 Detailed Implementation TODO

### 21.1 Project Skeleton And Contracts

- [x] Define Pydantic models for all core schemas.
- [x] Generate or maintain JSON Schema files.
- [x] Add contract tests for schema roundtrip.
- [x] Create project storage abstraction for `projects/<project_id>/runs/<run_id>`.
- [x] Create project asset abstraction for `projects/<project_id>/assets/...`.
- [x] Implement `AssetManifest`.
- [x] Implement asset version allocator with `v001` style versions.
- [x] Implement asset promotion records.
- [x] Implement `stage.json` writer and reader.
- [x] Implement artifact path registry.
- [x] Implement error taxonomy module.
- [x] Implement atomic task registry.
- [x] Implement linear dependency expansion from atomic task requirements.
- [x] Implement `RunPlan` diff representation after plan edits.

### 21.2 Adapter Layer

- [x] Implement JSON-in/JSON-out adapter runner.
- [x] Port/adapt adapter validation from `oled-agent`.
- [x] Add `legacy_full_flow_adapter` for temporary `claude/run_mvp_flow.py`.
- [x] Add `parse_task` adapter using current `claude` NL parser initially.
- [x] Add native `inspect_dataset` service.
- [x] Add `draft_cleaning_rules` adapter around `claude` cleaning logic.
- [x] Add `execute_cleaning` adapter around adapted cleaning script.
- [x] Add native `check_trainability` service.
- [x] Add native `run_baseline` service.
- [x] Add native `recommend_backend` service.
- [x] Add `train_model` adapter for baseline.
- [x] Add `train_model` adapter for Uni-Mol legacy remote training.
- [x] Add `predict_candidates` adapter for baseline.
- [x] Add `predict_candidates` adapter for Uni-Mol legacy predictor path.
- [x] Add `filter_rank` adapter.
- [x] Add `render_report` adapter.

### 21.3 Data Layer

- [x] Implement dataset upload and registration.
- [x] Implement dataset role handling.
- [x] Implement header and delimiter detection.
- [x] Implement SMILES column detection.
- [x] Implement property candidate detection.
- [x] Implement property alias mapping.
- [x] Implement unit and scale draft detection.
- [x] Implement duplicate conflict detection.
- [x] Implement outlier detection.
- [x] Implement split detection and fallback split proposal.
- [x] Implement property catalog generation.
- [x] Implement data leakage check for canonical SMILES overlap.

### 21.4 Trainability And Baseline

- [x] Implement trainability thresholds.
- [x] Implement task type detection for numeric regression.
- [x] Implement baseline feature generation with Morgan/ECFP.
- [x] Implement XGBoost baseline if dependency exists.
- [x] Implement RandomForest fallback.
- [x] Implement scaffold split default.
- [x] Implement random split fallback.
- [x] Record split fallback reason.
- [x] Implement baseline metric report.
- [x] Implement `ReadinessAssessment`.
- [x] Implement property 3D relevance heuristic table.
- [x] Implement user override storage for backend and 3D relevance recommendations.
- [x] Implement backend recommendation.
- [x] Implement per-property recommendation while requiring one backend per run.

### 21.5 Training And Prediction

- [x] Implement baseline training and model persistence.
- [x] Implement baseline candidate prediction.
- [x] Wrap existing remote Uni-Mol training.
- [x] Wrap Uni-Mol candidate prediction.
- [x] Write model metadata for every trained model.
- [x] Register model only after user confirmation.
- [x] Prevent automatic overwrite.

### 21.6 Screening And Reporting

- [x] Implement hard constraint filtering.
- [x] Implement soft constraint scoring.
- [x] Implement weighted score ranking.
- [x] Implement TopN export.
- [x] Add Pareto/trade-off explanation.
- [x] Render atomic task Markdown reports.
- [x] Render full workflow Markdown and HTML reports.
- [x] Generate matching JSON artifacts.

### 21.7 UI And Job Manager

- [x] Implement Flask or equivalent local app in `workspace/agent`.
- [x] Implement project creation and selection.
- [x] Implement upload endpoints.
- [x] Implement parse/regenerate plan endpoint.
- [x] Implement atomic task toolbox.
- [x] Implement dependency plan preview.
- [x] Implement RunPlan diff preview.
- [x] Implement data confirmation card.
- [x] Implement run confirmation card.
- [x] Implement pause/save-and-stop at high-risk gates.
- [x] Implement asset promotion UI.
- [x] Implement local job manager.
- [x] Implement status endpoint.
- [x] Implement recent logs endpoint.
- [x] Implement stage timeline component.
- [x] Implement report preview component.
- [x] Implement retry failed stage endpoint.

### 21.8 Memory And Permissions

- [x] Implement run memory artifact collection.
- [x] Implement project memory schema.
- [x] Implement project memory save confirmation.
- [x] Implement permission policy module.
- [x] Implement `auto`, `project-approved`, and `confirm-each-time` gates.
- [x] Implement external LLM context policy with `dataset_public`.

### 21.9 Acceptance

- [x] Add unit tests for schemas.
- [x] Add adapter contract tests.
- [x] Add dry-run workflow test.
- [x] Add baseline workflow test.
- [x] Add UI smoke test.
- [x] Add stage history validation.
- [x] Add report artifact validation.
- [x] Add manual real Uni-Mol acceptance script or checklist.

## 22. Phase 2 Roadmap

- [x] Add generator candidate source implementation.
- [x] Add REINVENT4 or another molecular generator as a backend.
- [x] Ensure generated candidates pass through prediction before ranking.
- [x] Add generation run artifacts and model provenance.
- [x] Add iterative generate-predict-filter loop.
- [x] Add diversity and novelty checks.
- [x] Add optional Pareto/frontier-driven generation targets.
- [x] Add user confirmation before expensive generation runs.

## 23. Phase 3 Roadmap

- [x] Add evidence-grounded literature-to-dataset workflow.
- [x] Support uploaded PDF folder as the first Phase 3 corpus input.
- [x] Support search query, URL, DOI, dataset registry, and external database inputs as auditable source manifest entries.
- [x] Add network/database acquisition adapters that turn non-local source manifest entries into local PDFs or structured datasets.
- [x] Use MinerU as the default `DocumentParserAdapter`.
- [x] Decide optional PyMuPDF/pdfplumber and GROBID parser adapters are not Phase 3 blockers while MinerU remains the default parser.
- [x] Add optional PyMuPDF/pdfplumber/GROBID fallback parser adapters for concrete MinerU failure cases or CPU-only/license-specific deployments.
- [x] Build table-aware parsing and chunking.
- [x] Build multi-index retrieval.
- [x] Build BM25 sparse retrieval with table-aware retrieval channel.
- [x] Build extraction confidence reports.
- [x] Add citation, license, and provenance tracking.
- [x] Add human confirmation for extracted data before confirmed training dataset promotion.
- [x] Add data merge and conflict resolution across sources.
- [x] Add benchmark contamination and train/test leakage checks for public datasets.
- [x] Add literature extraction audit trail.
- [x] Add extraction benchmark metrics: retrieval recall, extraction precision, conflict rate, confirmation workload, trainable labels gained, and downstream model performance impact.

Phase 3 corpus sources:

- [x] `uploaded_pdf_folder`: Phase 3 first implementation target.
- [x] `search_query`: supported as source manifest input.
- [x] `url`: supported as source manifest input.
- [x] `doi`: supported as source manifest input.
- [x] `dataset_registry`: supported as source manifest input.
- [x] `external_database`: supported as source manifest input.

Document parsing:

- [x] Default adapter: `MinerUDocumentParser`.
- [x] Optional fallback adapters: `pdfplumber` for local table-oriented extraction, `PyMuPDF` for fast page/text extraction, and `GROBID` for scholarly TEI metadata/body/table extraction.
- [x] Adapter contract: `parse_pdf(input_pdf) -> ParsedDocument`.
- [x] Preserve page, element type, bounding box, source hash, and table structure.
- [x] Preserve tables as headers, rows, footnotes, caption, page, Markdown, and source bbox.

Parsed document schema:

```text
ParsedDocument
  paper_id
  metadata
  pages[]
  elements[]
    element_id
    page
    type: title | abstract | paragraph | table | figure | caption | formula | reference
    text
    markdown
    bbox
    source_hash
```

Table schema:

```text
ParsedTable
  table_id
  caption
  headers
  rows
  footnotes
  page
  markdown
  source_bbox
```

Retrieval strategy:

- Full target: BM25 sparse retrieval, dense embedding retrieval, table-aware structured retrieval, chemical/entity index, and reranker.
- [x] Phase 3 first version: BM25 sparse retrieval + table-aware chunking + basic rerank.
- [x] Add deterministic multi-index retrieval over text, table, property, and chemical channels.
- [x] Add dense retrieval.
- [x] Add production `sentence-transformers` dense embedding backend; keep deterministic hash dense fallback for offline CI and dependency-free runs.
- Reserve chemical/entity index for RDKit, canonical SMILES, InChIKey, fingerprint similarity, and substructure search.
- [x] Do not rely on dense vector retrieval alone for SMILES, property names, units, or abbreviations.

Evidence hit schema:

```text
EvidenceHit
  source_id
  page
  element_id
  element_type
  retrieval_channel: bm25 | dense | table | chemical
  score
  text_or_table_ref
  citation_context
```

Extraction flow:

```text
PDF corpus
  -> layout parsing
  -> text/table/figure/caption elements
  -> multi-index storage
  -> property-specific retrieval
  -> schema extraction
  -> evidence-linked records
  -> condition/unit normalization
  -> conflict detection
  -> human confirmation
  -> confirmed training dataset
```

Literature extraction artifacts:

- [x] `retrieval_log.jsonl`
- [x] `corpus_source_manifest.json`
- [x] `corpus_source_manifest.md`
- [x] `literature_acquisition_manifest.json`
- [x] `literature_acquisition_plan.md`
- [x] `multi_index.json`
- [x] `multi_index_summary.md`
- [x] `dense_index.json`
- [x] `dense_index_summary.md`
- [x] `extraction_attempts.jsonl`
- [x] `extracted_records.jsonl`
- [x] `rejected_records.jsonl`
- [x] `candidate_training_dataset.csv`
- [x] `conflict_report.json`
- [x] `conflict_report.md`
- [x] `confirmed_training_dataset.csv`
- [x] `extraction_confirmation_record.json`
- [x] `literature_workflow_report.json`
- [x] `literature_workflow_summary.md`
- [x] `benchmark_contamination_report.json`
- [x] `benchmark_contamination_report.md`
- [x] `unit_normalization_report.json`
- [x] `unit_normalization_report.md`
- [x] `extraction_benchmark_report.json`
- [x] `extraction_benchmark_report.md`
- [x] `extraction_summary.md`
- [x] `audit_summary.md`

Training dataset promotion:

- High-confidence extracted records may enter candidate training datasets.
- Candidate training datasets cannot become confirmed training datasets without user confirmation.
- Training before confirmation is not allowed.
- Show extraction summary, conflict report, unit normalization report, and trainability report before promotion.
- Keep rejected and low-confidence records for audit and future correction.

Human confirmation:

- Confirm stable rules, aliases, units, and high-confidence groups in batch.
- Review low-confidence records, conflict groups, outliers, high-impact records, new properties, new units, and new aliases individually.
- High-impact records include records that strongly affect label distribution, sparse properties, target-boundary samples, or conflicts with confirmed data.

RAG role:

- RAG is evidence infrastructure, not the final authority.
- RAG retrieves evidence and candidate records.
- Structured extraction, normalization, conflict detection, and human confirmation decide whether data enters training.
- Do not silently aggregate or delete literature-derived data based only on RAG output.

## 24. Phase 4 Roadmap

Phase 4 reframes the system from a mostly fixed workflow orchestrator into an agentic decision layer over the existing audited workflow substrate.

Core principle:

- The workflow/adapters/gates remain the deterministic execution layer.
- LLM agents choose, explain, monitor, and revise plans.
- High-risk actions still require explicit confirmation.
- Every autonomous decision must leave a structured trace.

### 24.1 Agentic Planning Layer

- [x] Add `PlannerAgent` that converts a natural-language research goal into a `RunPlan`.
- [x] Let `PlannerAgent` choose atomic tasks dynamically from the task registry instead of relying on only fixed workflow presets.
- [x] Require `PlannerAgent` to explain selected tasks, skipped tasks, missing artifacts, assumptions, risks, and required gates.
- [x] Add `PlanRationale` artifact with task-level reasoning and user-facing summary.
- [x] Add `PlanQuestion` / clarification flow when the goal is underspecified or risk is high.
- [x] Add a dry-run plan preview that never executes adapters until the user confirms the plan or allowed subset.
- [x] Keep rule-based planning as deterministic fallback when no LLM provider is configured.

### 24.2 LLM Provider And Tool Calling

- [x] Add an LLM provider abstraction for planner/verifier agents.
- [x] Support local/stub provider for tests and deterministic CI.
- [x] Support configurable OpenAI-compatible endpoint from runtime config.
- [x] Define JSON-only tool contracts for selecting tasks, requesting artifacts, and proposing replans.
- [x] Validate every LLM output with Pydantic schemas before using it.
- [x] Reject free-form LLM tool calls that do not map to registered adapters or approved actions.
- [x] Log prompt version, model name, response id if available, and parsed structured output.

### 24.3 Observation And Critic/Verifier Layer

- [x] Add `ObserverAgent` that reads stage outputs, logs, reports, and artifact manifests after each step.
- [x] Add `CriticAgent` / `VerifierAgent` that checks whether artifacts are usable before advancing.
- [x] Detect common failure modes: empty extraction, low confidence, high conflict rate, leakage, invalid units, poor trainability, abnormal model metrics, missing provenance, and stale approvals.
- [x] Produce `verification_report.json` and `verification_report.md` for every major run.
- [x] Convert verifier findings into either continue, retry, replan, ask-user, or abort decisions.
- [x] Keep verifier checks partly rule-based so safety-critical checks do not depend only on LLM judgment.

### 24.3.1 Target-Aware Modeling And Rerun Loop

- [x] Add `TargetModelingBrief` as a first-class planning artifact for every requested training target.
- [ ] Require `ModelingAgent` to consult project memory, previous run diagnostics, built-in domain rules, and optional user-approved web/literature search before proposing preprocessing, target transforms, split strategy, backend, and hyperparameters.
  - MVP status: project memory, previous diagnostics, trainability context, built-in OLED rules, `user_approved_external_search` policy labels, and structured `TargetEvidenceItem` injection are wired. Remaining TODO: automatically hand off cited ResearchAgent/web/literature evidence into the brief after explicit user approval.
- [x] Record target-specific cautions such as solvent dependence, device/context dependence, bounded labels, log-scale labels, replicate variance, class imbalance, censored values, target leakage risks, and expected metric ranges.
- [x] Add `ModelDiagnosticsReport` after training that compares the trained model to baselines, target-specific expectations, prediction range, bucket bias, and high-value compression risk.
- [x] Add `RerunProposal` as a structured artifact with candidate fixes, rationale, expected impact, cost, approval requirements, and fallback policy.
- [x] Let the agent recommend rerun actions such as revised cleaning, low-noise filtering, target transform, solvent/context conditioning, split correction, seed ensemble, hyperparameter sweep, backend switch, more data, or low-confidence model use.
- [x] Keep rerun execution gated: the agent may propose and explain reruns, but expensive training, external search, backend switches, data relaxation, or model promotion still require user confirmation.
- [x] Add `DomainModelRegistry` and OLED historical modeling priors for emission, scalar PLQY, and high-PLQY screening. These records are agent memory for future training decisions, not default prediction weights.
- [x] Add `PredictionPreparation` as a non-executable pre-prediction artifact that turns a natural-language target and available input columns into historical prior selection, missing-input questions, required gates, training-required warnings, and draft adapter payload only when a promoted asset or explicitly approved historical reuse exists.
- [x] Implement `predict_candidates_domain_model_adapter` bridge for solvent-aware OLED models and wire it into adapter export, API permission policy, and RunPlan adapter override allowlist.
- [x] Add `score_domain_model_candidates.py` runtime package for domain model manifests, including input-column validation, precomputed prediction CSV merge, external-command dispatch, and standardized stdout JSON for adapter results.
- [x] Prevent historical training results such as `plqy_solvent_pca64_seed42` and `plqy_manual_weight3_ensemble` from being used as default MVP prediction assets; fresh target-specific training remains the default for new user requests.
- [x] Add a `PromotedModelAsset` path for future models trained against a specific confirmed request, with explicit user approval, applicability limits, manifest/runtime, and rollback metadata before prediction reuse.
  - MVP status: schema, JSON Schema export, project storage promotion/read path, and `PredictionPreparationAgent` selection path are implemented. Confirmed promoted assets can produce draft prediction payloads; candidate/deprecated assets and historical priors still require fresh training or explicit reuse approval.
  - API/UI status: `/models/promote` exposes the confirmed model asset promotion path, and `/models/promote/draft` plus the local console draft button prefill promotion fields from registered model metadata/manifests before human confirmation.
  - Training package status: baseline training now writes `model_metadata.json`, `model_manifest.json`, and `domain_model_manifest.json`; `RunPlanExecutor` registers those artifacts so a confirmed model registration can immediately produce a promotion draft.
  - Agent review status: `ModelPackageReview` and `/api/agent/model-package-review` now classify a trained package as `promote_candidate`, `rerun_recommended`, `memory_only`, or `blocked` from manifests, metrics, diagnostics, required inputs, and known risks; `RunPlanExecutor` automatically writes `ModelDiagnosticsReport` and `ModelPackageReview` artifacts after baseline training with manifests; promote remains gated by `promote_asset`.
  - Acceptance status: local pytest now covers train -> auto diagnostics/review -> register model asset -> confirmed promotion -> `PredictionPreparationAgent.prepare_prediction_for_project()` selecting the confirmed asset without requiring fresh training.
  - Remaining polish: expand metadata coverage from model diagnostics/rerun reports and add richer UI review cards for applicability, limitations, and rollback comparison.

### 24.4 Recovery And Replanning

- [x] Add `ReplanRequest` schema for failures, degraded outputs, new user constraints, or changed artifacts.
- [x] Add `RunPlanRevision` artifact that records previous plan, revised plan, diff, reason, and approvals required.
- [x] Implement recovery policies for parser fallback: MinerU -> pdfplumber/PyMuPDF/GROBID when evidence quality is poor or parsing fails.
- [x] Implement recovery policies for data mining: expand query, add DOI/URL sources, retry acquisition, or request user-provided PDFs.
- [x] Implement recovery policies for modeling: switch backend, adjust split, reduce target properties, request more data, or run only baseline.
- [x] Link modeling recovery policies to `ModelDiagnosticsReport` and `RerunProposal` rather than treating weak metrics as a generic failure.
- [x] Never silently downgrade a high-risk action into a lower-risk action.
- [x] Require user approval before executing a revised plan that adds high-risk or external-network actions.

### 24.5 Memory-Driven Planning

- [x] Add project memory records for confirmed user preferences, backend choices, remote hosts, parser choices, property aliases, and accepted risk policies.
- [x] Separate project memory from raw data; store references, summaries, hashes, and decisions rather than full datasets.
- [x] Use memory to prefill future plans while still showing assumptions to the user.
- [x] Add memory governance: inspect, edit, delete, export, and disable memory per project.
- [x] Record why memory was used in a plan, not just the resulting decision.
- [x] Keep sensitive credentials and private raw data out of memory.

### 24.6 Research Agent Capabilities

- [x] Add `ResearchAgent` for source discovery, query expansion, DOI/URL ranking, and evidence quality assessment.
- [x] Add `ModelingAgent` for backend recommendation, experiment design, metric interpretation, and retry proposals.
- [ ] Let `ResearchAgent` provide target-specific evidence only through cited summaries and structured source manifests; do not inject web/literature claims directly into preprocessing or hyperparameters without a reviewable `TargetModelingBrief`.
  - MVP status: `TargetModelingBrief` now accepts structured `TargetEvidenceItem` records and exposes them in artifacts/review cards; direct ResearchAgent-to-ModelingAgent handoff is still TODO.
- [x] Add `GenerationAgent` for candidate-generation strategy, frontier/constraint selection, and diversity/novelty tradeoff proposals.
- [x] Add `ReportAgent` for run synthesis, limitations, next-step recommendations, and paper-style audit summaries.
- [x] Keep these agents coordinated through shared schemas and artifacts, not hidden conversation state.

### 24.7 Agentic UI

- [x] Show "why this plan" alongside the task timeline.
- [x] Show agent assumptions and missing information before execution.
- [x] Show verifier findings and proposed replans as reviewable cards.
- [x] Show `TargetModelingBrief`, `ModelDiagnosticsReport`, `RerunProposal`, and `ModelPackageReview` as reviewable cards with source labels and approval controls.
- [x] Add compare view for original plan vs revised plan.
- [x] Add approval controls for plan confirmation, replan confirmation, memory use, external acquisition, training, and generation.
- [x] Avoid making autonomous actions look like chat suggestions; every executable action should map to a visible task/gate.
- [x] Add a conservative `RunPlan` execution bridge that runs confirmed low-risk tasks and pauses at explicit gates.
- [x] Add gate-approved `RunPlan` resume so confirmed tasks continue from the current waiting stage.

### 24.8 Evaluation And Acceptance

- [x] Add unit tests for planner output validation, invalid LLM output rejection, and rule-based fallback planning.
- [x] Add integration tests for observe -> verify -> replan loops.
- [x] Add acceptance scenario: user gives a broad research goal and the agent proposes a safe plan without executing it.
- [x] Add acceptance scenario: parsing fails and agent proposes a parser fallback replan.
- [x] Add acceptance scenario: extraction has high conflict rate and agent asks for human review before promotion.
- [x] Add acceptance scenario: model metrics are weak and agent proposes more data or a different backend.
- [x] Add acceptance scenario: a trained model package is reviewed, registered, explicitly promoted, and then reused by prediction preparation only as a confirmed model asset.
- [x] Track agent autonomy metrics: tasks selected by agent, replans proposed, user confirmations required, verifier catches, and failed autonomous decisions.
- [x] Add integration tests for the conservative `RunPlan` executor and invalid execution payloads.
- [x] Add integration tests for gate-approved resume through training, generation, prediction, ranking, and reporting.

### 24.9 Later Deployment

- [x] Add remote worker abstraction after agentic planning is stable.
- [x] Add multi-user deployment only after permission, memory, and audit boundaries are tested.
- [x] Add background/long-running research jobs with resumable state and explicit budget limits.

## 25. Notes To Preserve

- Phase 1 deliberately favors correctness, traceability, and user confirmation over full automation.
- The agent should not silently change user intent, relax constraints, replace backend, or delete data.
- Raw data should not be stored in project memory.
- Public data reduces privacy risk but does not remove license, citation, and audit requirements.
- Backend recommendation is advisory; final training backend is user-confirmed.
- The final product should support more advanced modes later, but Phase 1 should stay small enough to complete and verify.
- Phase 1 first implementation milestone is `inspect_dataset -> property_catalog -> trainability -> baseline -> backend recommendation -> Markdown report`.
- Phase 1 full acceptance adds training, prediction, filtering/ranking, HTML report, and asset promotion.
- Complete workflows and atomic tasks must share the same artifact and confirmation model.
- RAG/MCP/skill technologies are useful long-term, but they should plug into adapter, permission, and artifact contracts rather than define the Phase 1 core.
