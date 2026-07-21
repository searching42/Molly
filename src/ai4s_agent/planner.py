from __future__ import annotations

from collections.abc import Iterable

from ai4s_agent.schemas import (
    AtomicTaskSpec,
    GateName,
    PlanModel,
    PlannedTask,
    PlanStep,
    RiskLevel,
    RunPlan,
    RunPlanDiff,
)


DEFAULT_ATOMIC_TASKS: tuple[AtomicTaskSpec, ...] = (
    AtomicTaskSpec(
        task_id="inspect_dataset",
        required_artifacts=[],
        output_artifacts=["dataset_profile", "property_catalog"],
        risk_level=RiskLevel.LOW,
        default_adapter="inspect_dataset_service",
    ),
    AtomicTaskSpec(
        task_id="clean_dataset",
        required_artifacts=["dataset_profile"],
        output_artifacts=["cleaned_train_dataset", "cleaning_rules"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="execute_cleaning_adapter",
    ),
    AtomicTaskSpec(
        task_id="check_trainability",
        required_artifacts=["cleaned_train_dataset", "property_catalog"],
        output_artifacts=["trainability_report"],
        risk_level=RiskLevel.LOW,
        default_adapter="check_trainability_service",
    ),
    AtomicTaskSpec(
        task_id="run_baseline",
        required_artifacts=["trainability_report"],
        output_artifacts=["baseline_report", "backend_recommendation"],
        risk_level=RiskLevel.LOW,
        default_adapter="run_baseline_service",
    ),
    AtomicTaskSpec(
        task_id="train_model",
        required_artifacts=["cleaned_train_dataset", "trainability_report"],
        output_artifacts=[
            "trained_model",
            "model_metadata",
            "model_manifest",
            "domain_model_manifest",
            "model_diagnostics_report",
            "model_package_review",
        ],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.TRAIN_CONFIG.value],
        default_adapter="train_model_baseline_adapter",
    ),
    AtomicTaskSpec(
        task_id="generate_candidates",
        required_artifacts=["trained_model", "model_metadata"],
        output_artifacts=["candidate_dataset", "generation_report"],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.FINAL_THRESHOLD.value],
        default_adapter="generate_candidates_stub_adapter",
    ),
    AtomicTaskSpec(
        task_id="predict_candidates",
        required_artifacts=["trained_model", "candidate_dataset"],
        output_artifacts=["candidate_predictions"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="predict_candidates_baseline_adapter",
    ),
    AtomicTaskSpec(
        task_id="filter_rank",
        required_artifacts=["candidate_predictions"],
        output_artifacts=["ranked_candidates", "topn_export"],
        risk_level=RiskLevel.LOW,
        default_adapter="filter_rank_adapter",
    ),
    AtomicTaskSpec(
        task_id="render_report",
        required_artifacts=["ranked_candidates"],
        output_artifacts=["report_markdown", "report_html"],
        risk_level=RiskLevel.LOW,
        default_adapter="render_report_adapter",
    ),
    AtomicTaskSpec(
        task_id="parse_document",
        required_artifacts=["pdf_corpus"],
        output_artifacts=["parsed_document", "parsed_tables", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.DATA_MINING.value],
        default_adapter="parse_document_mineru_adapter",
    ),
    AtomicTaskSpec(
        task_id="parse_document_pdfplumber",
        required_artifacts=["pdf_corpus"],
        output_artifacts=["parsed_document", "parsed_tables", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="parse_document_pdfplumber_adapter",
    ),
    AtomicTaskSpec(
        task_id="parse_document_pymupdf",
        required_artifacts=["pdf_corpus"],
        output_artifacts=["parsed_document", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="parse_document_pymupdf_adapter",
    ),
    AtomicTaskSpec(
        task_id="parse_document_grobid",
        required_artifacts=["pdf_corpus"],
        output_artifacts=["parsed_document", "parsed_tables", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.DATA_MINING.value],
        default_adapter="parse_document_grobid_adapter",
    ),
    AtomicTaskSpec(
        task_id="prepare_literature_corpus_sources",
        required_artifacts=[],
        output_artifacts=["corpus_source_manifest"],
        risk_level=RiskLevel.LOW,
        default_adapter="prepare_literature_corpus_sources_adapter",
    ),
    AtomicTaskSpec(
        task_id="acquire_literature_sources",
        required_artifacts=["corpus_source_manifest"],
        output_artifacts=["pdf_corpus", "structured_datasets", "acquisition_manifest"],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.DATA_MINING.value],
        default_adapter="acquire_literature_sources_adapter",
    ),
    AtomicTaskSpec(
        task_id="index_corpus",
        required_artifacts=["parsed_document"],
        output_artifacts=["corpus_index", "evidence_chunks"],
        risk_level=RiskLevel.LOW,
        default_adapter="index_corpus_adapter",
    ),
    AtomicTaskSpec(
        task_id="build_multi_index",
        required_artifacts=["evidence_chunks"],
        output_artifacts=["multi_index"],
        risk_level=RiskLevel.LOW,
        default_adapter="build_multi_index_adapter",
    ),
    AtomicTaskSpec(
        task_id="build_dense_index",
        required_artifacts=["evidence_chunks"],
        output_artifacts=["dense_index"],
        risk_level=RiskLevel.LOW,
        default_adapter="build_dense_index_adapter",
    ),
    AtomicTaskSpec(
        task_id="retrieve_evidence",
        required_artifacts=["corpus_index"],
        output_artifacts=["evidence_hits", "retrieval_log"],
        risk_level=RiskLevel.LOW,
        default_adapter="retrieve_evidence_adapter",
    ),
    AtomicTaskSpec(
        task_id="extract_records",
        required_artifacts=["evidence_hits", "evidence_chunks"],
        output_artifacts=[
            "extracted_records",
            "rejected_records",
            "extraction_confidence_report",
            "candidate_training_dataset",
        ],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="extract_records_adapter",
    ),
    AtomicTaskSpec(
        task_id="normalize_extracted_units",
        required_artifacts=["extracted_records"],
        output_artifacts=[
            "normalized_extracted_records",
            "candidate_training_dataset",
            "unit_normalization_report",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="normalize_extracted_units_adapter",
    ),
    AtomicTaskSpec(
        task_id="track_citation_provenance",
        required_artifacts=["parsed_document", "evidence_hits", "extracted_records"],
        output_artifacts=["citation_provenance_report", "audit_summary"],
        risk_level=RiskLevel.LOW,
        default_adapter="track_citation_provenance_adapter",
    ),
    AtomicTaskSpec(
        task_id="merge_extracted_records",
        required_artifacts=["normalized_extracted_records", "citation_provenance_report"],
        output_artifacts=["merged_records", "conflict_report", "candidate_training_dataset"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="merge_extracted_records_adapter",
    ),
    AtomicTaskSpec(
        task_id="evaluate_extraction_benchmark",
        required_artifacts=["evidence_hits", "normalized_extracted_records", "conflict_report"],
        output_artifacts=["extraction_benchmark_report"],
        risk_level=RiskLevel.LOW,
        default_adapter="evaluate_extraction_benchmark_adapter",
    ),
    AtomicTaskSpec(
        task_id="confirm_extracted_dataset",
        required_artifacts=[
            "candidate_training_dataset",
            "conflict_report",
            "citation_provenance_report",
        ],
        output_artifacts=["confirmed_training_dataset", "extraction_confirmation_record"],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.DATA_MINING.value],
        default_adapter="confirm_extracted_dataset_adapter",
    ),
    AtomicTaskSpec(
        task_id="literature_to_dataset_workflow",
        required_artifacts=["pdf_corpus"],
        output_artifacts=[
            "corpus_manifest",
            "corpus_index",
            "evidence_hits",
            "extracted_records",
            "unit_normalization_report",
            "citation_provenance_report",
            "conflict_report",
            "extraction_benchmark_report",
            "candidate_training_dataset",
            "workflow_report",
        ],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.DATA_MINING.value],
        default_adapter="literature_to_dataset_workflow_adapter",
    ),
    AtomicTaskSpec(
        task_id="check_public_dataset_leakage",
        required_artifacts=["candidate_training_dataset"],
        output_artifacts=["benchmark_contamination_report"],
        risk_level=RiskLevel.LOW,
        default_adapter="check_public_dataset_leakage_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_local_demo",
        required_artifacts=[],
        output_artifacts=[
            "oled_demo_bundle_report",
            "oled_demo_bundle_markdown",
            "oled_local_demo_execution_manifest",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="execute_oled_local_demo_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_registry_candidate_screening",
        required_artifacts=[
            "oled_phase1_execution_dir",
            "oled_dataset_snapshot",
            "oled_registry_snapshot",
        ],
        output_artifacts=[
            "oled_registry_screening_receipt",
            "oled_registry_screening_shortlist",
            "oled_registry_screening_predictions",
            "oled_registry_screening_exclusions",
            "oled_registry_screening_eligible_candidates",
            "oled_registry_screening_report",
            "oled_registry_screening_execution_record",
        ],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.FINAL_THRESHOLD.value],
        default_adapter="execute_oled_registry_candidate_screening_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_experiment_batch_selection",
        required_artifacts=[
            "oled_registry_screening_receipt",
            "oled_registry_screening_shortlist",
            "oled_phase1_execution_dir",
            "oled_dataset_snapshot",
            "oled_registry_snapshot",
        ],
        output_artifacts=[
            "oled_experiment_batch_receipt",
            "oled_experiment_batch_handoff",
            "oled_candidate_decision_dossier",
            "oled_experiment_batch_report",
            "oled_experiment_batch_execution_record",
        ],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.FINAL_THRESHOLD.value],
        default_adapter="execute_oled_experiment_batch_selection_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_inverse_design",
        required_artifacts=[
            "oled_experiment_batch_receipt",
            "oled_registry_screening_receipt",
            "oled_registry_screening_shortlist",
            "oled_phase1_execution_dir",
            "oled_dataset_snapshot",
            "oled_registry_snapshot",
            "oled_inverse_design_reinvent4_config",
        ],
        output_artifacts=[
            "oled_inverse_design_receipt",
            "oled_inverse_design_candidates",
            "oled_inverse_design_exclusions",
            "oled_inverse_design_report",
            "oled_inverse_design_execution_record",
        ],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.FINAL_THRESHOLD.value],
        default_adapter="execute_oled_inverse_design_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_generated_candidate_evaluation",
        required_artifacts=[
            "oled_inverse_design_receipt",
            "oled_experiment_batch_receipt",
            "oled_registry_screening_receipt",
            "oled_registry_screening_shortlist",
            "oled_phase1_execution_dir",
            "oled_dataset_snapshot",
            "oled_registry_snapshot",
        ],
        output_artifacts=[
            "oled_candidate_evaluation_receipt",
            "oled_candidate_evaluation_predictions",
            "oled_candidate_evaluation_shortlist",
            "oled_candidate_evaluation_exclusions",
            "oled_candidate_evaluation_report",
            "oled_candidate_evaluation_execution_record",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="execute_oled_generated_candidate_evaluation_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_candidate_decision",
        required_artifacts=[
            "oled_candidate_evaluation_receipt",
            "oled_inverse_design_receipt",
            "oled_experiment_batch_receipt",
            "oled_registry_screening_receipt",
            "oled_registry_screening_shortlist",
            "oled_phase1_execution_dir",
            "oled_dataset_snapshot",
            "oled_registry_snapshot",
        ],
        output_artifacts=[
            "oled_final_candidate_decision_receipt",
            "oled_final_candidate_decision_top_n",
            "oled_final_candidate_decision_dossier",
            "oled_final_candidate_decision_report",
            "oled_final_candidate_decision_execution_record",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="execute_oled_candidate_decision_adapter",
    ),
    AtomicTaskSpec(
        task_id="execute_oled_bounded_discovery_controller",
        required_artifacts=["oled_bounded_controller_request"],
        output_artifacts=[
            "oled_bounded_controller_receipt",
            "oled_bounded_controller_request_snapshot",
            "oled_bounded_controller_generation_authorization",
            "oled_bounded_controller_report",
            "oled_bounded_controller_execution_record",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="execute_oled_bounded_discovery_controller_adapter",
    ),
)


class AtomicTaskRegistry:
    def __init__(self, tasks: Iterable[AtomicTaskSpec] | None = None) -> None:
        source = list(tasks or DEFAULT_ATOMIC_TASKS)
        self._validate_tasks(source)
        self._tasks = {task.task_id: task for task in source}
        self._artifact_producers: dict[str, str] = {}
        for task in source:
            for artifact in task.output_artifacts:
                self._artifact_producers.setdefault(artifact, task.task_id)

    @staticmethod
    def _validate_tasks(tasks: list[AtomicTaskSpec]) -> None:
        valid_gates = {gate.value for gate in GateName}
        for task in tasks:
            if task.risk_level == RiskLevel.HIGH and not task.gates:
                raise ValueError(f"high-risk task requires gate: {task.task_id}")
            unknown_gates = [gate for gate in task.gates if gate not in valid_gates]
            if unknown_gates:
                raise ValueError(f"unknown gate on task {task.task_id}: {', '.join(unknown_gates)}")

    def list_tasks(self) -> list[AtomicTaskSpec]:
        return [self._tasks[k] for k in sorted(self._tasks)]

    def get(self, task_id: str) -> AtomicTaskSpec:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise ValueError(f"unknown atomic task: {task_id}") from exc

    def producer_for(self, artifact_id: str) -> str | None:
        return self._artifact_producers.get(artifact_id)


def build_plan(run_id: str, prompt: str) -> PlanModel:
    steps = [
        PlanStep(
            name="parse_task",
            agent="PlannerAgent",
            action="parse_task",
            inputs={"prompt": prompt},
        ),
        PlanStep(
            name="inspect_dataset",
            agent="DataAgent",
            action="inspect_dataset",
            inputs={},
        ),
        PlanStep(
            name="clean_dataset",
            agent="DataAgent",
            action="clean_dataset",
            inputs={},
        ),
        PlanStep(
            name="check_trainability",
            agent="TrainabilityAgent",
            action="check_trainability",
            inputs={},
        ),
        PlanStep(
            name="run_baseline",
            agent="TrainerAgent",
            action="run_baseline",
            inputs={},
        ),
        PlanStep(
            name="train_model",
            agent="TrainerAgent",
            action="train_model",
            inputs={},
        ),
        PlanStep(
            name="generate_candidates",
            agent="GeneratorAgent",
            action="generate_candidates",
            inputs={},
        ),
        PlanStep(
            name="predict_candidates",
            agent="PredictorAgent",
            action="predict_candidates",
            inputs={},
        ),
        PlanStep(
            name="filter_rank",
            agent="ScreenerAgent",
            action="filter_rank",
            inputs={},
        ),
        PlanStep(
            name="render_report",
            agent="ReportAgent",
            action="render_report",
            inputs={},
        ),
    ]
    return PlanModel(run_id=run_id, steps=steps, gates=[gate.value for gate in GateName])


def expand_run_plan(
    run_id: str,
    requested_tasks: list[str],
    available_artifacts: list[str] | None = None,
    registry: AtomicTaskRegistry | None = None,
) -> RunPlan:
    task_registry = registry or AtomicTaskRegistry()
    pre_existing_artifacts = set(available_artifacts or [])
    available = set(pre_existing_artifacts)
    missing_artifacts: set[str] = set()
    resolved: set[str] = set()
    resolving: set[str] = set()
    ordered_task_ids: list[str] = []
    unresolved_by_task: dict[str, list[str]] = {}
    dependencies_by_task: dict[str, list[str]] = {}
    dedup_requested: list[str] = []
    for requested in requested_tasks:
        if requested not in dedup_requested:
            dedup_requested.append(requested)

    preferred_producers: dict[str, str] = {}
    for task_id in dedup_requested:
        spec = task_registry.get(task_id)
        for artifact in spec.output_artifacts:
            preferred_producers.setdefault(artifact, task_id)

    def resolve_task(task_id: str) -> None:
        if task_id in resolved:
            return
        if task_id in resolving:
            raise ValueError(f"cyclic dependency detected: {task_id}")
        spec = task_registry.get(task_id)
        resolving.add(task_id)

        dependencies = list(spec.depends_on)
        unresolved_requirements: list[str] = []

        for required in spec.required_artifacts:
            if required in pre_existing_artifacts:
                continue
            producer = preferred_producers.get(required) or task_registry.producer_for(required)
            if producer == task_id:
                raise ValueError(
                    f"self-referencing artifact dependency in task {task_id}: {required}"
                )
            if producer:
                dependencies.append(producer)
                continue
            unresolved_requirements.append(required)
            missing_artifacts.add(required)

        dedup_dependencies: list[str] = []
        for dep in dependencies:
            if dep not in dedup_dependencies:
                dedup_dependencies.append(dep)

        for dep in dedup_dependencies:
            resolve_task(dep)

        resolving.remove(task_id)
        resolved.add(task_id)
        ordered_task_ids.append(task_id)
        unresolved_by_task[task_id] = unresolved_requirements
        dependencies_by_task[task_id] = dedup_dependencies
        available.update(spec.output_artifacts)

    for requested in requested_tasks:
        resolve_task(requested)

    tasks: list[PlannedTask] = []
    for task_id in ordered_task_ids:
        spec = task_registry.get(task_id)
        depends_on = [dep for dep in dependencies_by_task.get(task_id, []) if dep in ordered_task_ids]
        tasks.append(
            PlannedTask(
                task_id=task_id,
                depends_on=depends_on,
                required_artifacts=list(spec.required_artifacts),
                output_artifacts=list(spec.output_artifacts),
                unresolved_requirements=list(unresolved_by_task.get(task_id, [])),
            )
        )

    return RunPlan(
        run_id=run_id,
        requested_tasks=dedup_requested,
        tasks=tasks,
        available_artifacts=sorted(available),
        missing_artifacts=sorted(missing_artifacts),
    )


def diff_run_plans(before: RunPlan, after: RunPlan) -> RunPlanDiff:
    before_ids = [task.task_id for task in before.tasks]
    after_ids = [task.task_id for task in after.tasks]
    before_set = set(before_ids)
    after_set = set(after_ids)

    added_tasks = [task_id for task_id in after_ids if task_id not in before_set]
    removed_tasks = [task_id for task_id in before_ids if task_id not in after_set]
    unchanged_tasks = [task_id for task_id in after_ids if task_id in before_set]

    before_dep_map = {task.task_id: sorted(task.depends_on) for task in before.tasks}
    after_dep_map = {task.task_id: sorted(task.depends_on) for task in after.tasks}
    changed_dependencies: dict[str, dict[str, list[str]]] = {}
    for task_id in unchanged_tasks:
        if before_dep_map.get(task_id, []) != after_dep_map.get(task_id, []):
            changed_dependencies[task_id] = {
                "before": before_dep_map.get(task_id, []),
                "after": after_dep_map.get(task_id, []),
            }

    return RunPlanDiff(
        added_tasks=added_tasks,
        removed_tasks=removed_tasks,
        unchanged_tasks=unchanged_tasks,
        changed_dependencies=changed_dependencies,
    )
