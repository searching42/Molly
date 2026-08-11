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


def _closed_option_schema(
    properties: dict[str, dict[str, object]] | None = None,
    *,
    required: list[str] | None = None,
) -> dict[str, object]:
    """Return an explicit, closed high-level planning option contract."""

    return {
        "type": "object",
        "properties": properties or {},
        "required": required or [],
        "additionalProperties": False,
    }


def _uniform_input_trust(
    artifact_ids: list[str],
    trust_classes: list[str],
) -> dict[str, list[str]]:
    return {artifact_id: list(trust_classes) for artifact_id in artifact_ids}


DEFAULT_ATOMIC_TASKS: tuple[AtomicTaskSpec, ...] = (
    AtomicTaskSpec(
        task_id="inspect_dataset",
        required_artifacts=[],
        optional_input_artifacts=["uploaded_dataset", "confirmed_training_dataset"],
        input_artifact_alternatives=[["uploaded_dataset", "confirmed_training_dataset"]],
        output_artifacts=["dataset_profile", "property_catalog"],
        risk_level=RiskLevel.LOW,
        default_adapter="inspect_dataset_service",
        scientific_tool_id="inspect_dataset",
        label="Inspect dataset",
        description="Inspect a content-bound dataset and derive its logical profile.",
        effect_class="observe",
        required_permissions=["read_content_bound_input", "derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "uploaded_dataset": ["content_bound_input", "registered_intermediate"],
            "confirmed_training_dataset": ["confirmed_scientific_input"],
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="clean_dataset",
        required_artifacts=["uploaded_dataset"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["cleaned_train_dataset", "cleaning_rules", "property_catalog"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="execute_cleaning_adapter",
        depends_on=["inspect_dataset"],
        scientific_tool_id="clean_dataset",
        label="Clean dataset",
        description="Apply the registered data-cleaning workflow to a logical dataset profile.",
        effect_class="derive_local",
        required_permissions=["read_content_bound_input", "derive_project_artifact"],
        option_schema=_closed_option_schema(
            {
                "min_numeric_ratio": {"type": "number", "minimum": 0, "maximum": 1},
                "min_nonempty": {"type": "integer", "minimum": 1, "maximum": 1000000000},
                "drop_empty_target_rows": {"type": "boolean"},
                "strict_smiles_cleaning": {"type": "boolean"},
            }
        ),
        default_planner_options={
            "drop_empty_target_rows": False,
            "min_nonempty": 1,
            "min_numeric_ratio": 0.5,
            "strict_smiles_cleaning": True,
        },
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-clean-dataset.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "uploaded_dataset": ["content_bound_input", "registered_intermediate"],
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="check_trainability",
        required_artifacts=["property_catalog"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["trainability_report"],
        risk_level=RiskLevel.LOW,
        default_adapter="check_trainability_service",
        scientific_tool_id="check_trainability",
        label="Check trainability",
        description="Evaluate whether the cleaned dataset supports the registered modeling workflow.",
        effect_class="observe",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["property_catalog"],
            ["registered_intermediate", "verified_output", "confirmed_scientific_input"],
        ),
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="run_baseline",
        required_artifacts=["trainability_report"],
        optional_input_artifacts=["cleaned_train_dataset", "confirmed_training_dataset"],
        input_artifact_alternatives=[["cleaned_train_dataset", "confirmed_training_dataset"]],
        output_artifacts=["baseline_report", "backend_recommendation"],
        risk_level=RiskLevel.LOW,
        default_adapter="run_baseline_service",
        scientific_tool_id="run_baseline",
        label="Run baseline assessment",
        description="Produce the registered baseline assessment and backend recommendation.",
        effect_class="observe",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "trainability_report": ["registered_intermediate", "verified_output"],
            "cleaned_train_dataset": ["registered_intermediate", "verified_output"],
            "confirmed_training_dataset": ["confirmed_scientific_input"],
        },
        budget_dimensions=["max_records", "max_runtime_sec"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="train_model",
        required_artifacts=["trainability_report"],
        optional_input_artifacts=["cleaned_train_dataset", "confirmed_training_dataset"],
        input_artifact_alternatives=[["cleaned_train_dataset", "confirmed_training_dataset"]],
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
        scientific_tool_id="train_model",
        label="Train model",
        description="Train the registered model from a reviewed training dataset and trainability report.",
        effect_class="compute",
        required_permissions=["derive_project_artifact", "model_training_compute"],
        option_schema=_closed_option_schema(
            {
                "backend": {"type": "string", "enum": ["baseline", "unimol"]},
                "property_id": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "maxLength": 96,
                },
                "n_bits": {"type": "integer", "minimum": 32, "maximum": 8192},
            },
            required=["backend", "property_id"],
        ),
        default_planner_options={"backend": "baseline", "property_id": None},
        backend_default_planner_options={
            "baseline": {"n_bits": 256},
            "unimol": {},
        },
        review_required_option_ids=["property_id"],
        option_compiler_version="scientific-planner-option-train-model.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={"baseline": [], "unimol": ["model_training"]},
        default_planner_backend="baseline",
        execution_route=None,
        remote_task_type=None,
        backend_execution_routes={
            "baseline": "local_executor",
            "unimol": "remote_execution_service",
        },
        backend_remote_task_types={"baseline": None, "unimol": "model_training"},
        accepted_input_trust_classes_by_artifact={
            "trainability_report": ["registered_intermediate", "verified_output"],
            "cleaned_train_dataset": ["registered_intermediate", "verified_output"],
            "confirmed_training_dataset": ["confirmed_scientific_input"],
        },
        budget_dimensions=["max_runtime_sec", "max_gpu_hours"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="generate_candidates",
        required_artifacts=[],
        optional_input_artifacts=["cleaned_train_dataset", "confirmed_training_dataset"],
        input_artifact_alternatives=[],
        output_artifacts=["candidate_dataset", "generation_report", "generation_publication"],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.FINAL_THRESHOLD.value],
        default_adapter="generate_candidates_stub_adapter",
        scientific_tool_id="generate_candidates",
        label="Generate candidate molecules",
        description="Generate candidate molecules using the registered logical generation workflow.",
        effect_class="compute",
        required_permissions=["derive_project_artifact", "candidate_generation_compute"],
        option_schema=_closed_option_schema(
            {
                "backend": {"type": "string", "enum": ["deterministic_stub", "reinvent4"]},
                "count": {"type": "integer", "minimum": 1, "maximum": 100000},
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
            },
            required=["backend", "count"],
        ),
        default_planner_options={
            "backend": "deterministic_stub",
            "count": 32,
            "seed": 0,
        },
        backend_default_planner_options={
            "deterministic_stub": {},
            "reinvent4": {},
        },
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-generate-candidates.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={
            "deterministic_stub": [],
            "reinvent4": ["molecular_generation"],
        },
        default_planner_backend="deterministic_stub",
        execution_route=None,
        remote_task_type=None,
        backend_execution_routes={
            "deterministic_stub": "local_executor",
            "reinvent4": "remote_execution_service",
        },
        backend_remote_task_types={
            "deterministic_stub": None,
            "reinvent4": "molecular_generation",
        },
        accepted_input_trust_classes_by_artifact={
            "cleaned_train_dataset": ["registered_intermediate", "verified_output"],
            "confirmed_training_dataset": ["confirmed_scientific_input"],
        },
        budget_dimensions=["max_runtime_sec", "max_steps"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="predict_candidates",
        required_artifacts=["model_metadata", "candidate_dataset"],
        optional_input_artifacts=["trained_model"],
        input_artifact_alternatives=[],
        output_artifacts=["candidate_predictions"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="predict_candidates_baseline_adapter",
        scientific_tool_id="predict_candidates",
        label="Predict candidate properties",
        description="Score generated candidates with the registered prediction workflow.",
        effect_class="compute",
        required_permissions=["derive_project_artifact", "model_inference_compute"],
        option_schema=_closed_option_schema(
            {
                "property_id": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "maxLength": 96,
                }
            }
        ),
        default_planner_options={"property_id": None},
        backend_default_planner_options={},
        review_required_option_ids=["property_id"],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["candidate_dataset", "model_metadata", "trained_model"], ["verified_output"]
        ),
        budget_dimensions=["max_runtime_sec", "max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="filter_rank",
        required_artifacts=["candidate_predictions"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["ranked_candidates", "topn_export"],
        risk_level=RiskLevel.LOW,
        default_adapter="filter_rank_adapter",
        scientific_tool_id="filter_rank",
        label="Filter and rank candidates",
        description="Filter and rank candidate predictions with registered high-level criteria.",
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(
            {
                "top_n": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": 10000,
                },
                "objectives": {
                    "type": ["array", "null"],
                    "minItems": 1,
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "minLength": 1, "maxLength": 96},
                            "direction": {"type": "string", "enum": ["maximize", "minimize"]},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1000000},
                        },
                        "required": ["column", "direction", "weight"],
                        "additionalProperties": False,
                    },
                },
                "constraints": {
                    "type": "array",
                    "maxItems": 64,
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "minLength": 1, "maxLength": 96},
                            "minimum": {"type": "number", "minimum": -1000000, "maximum": 1000000},
                            "maximum": {"type": "number", "minimum": -1000000, "maximum": 1000000},
                        },
                        "required": ["column"],
                        "additionalProperties": False,
                    },
                },
            },
            required=["top_n", "objectives"],
        ),
        default_planner_options={
            "constraints": [],
            "objectives": None,
            "top_n": None,
        },
        backend_default_planner_options={},
        review_required_option_ids=["objectives", "top_n"],
        option_compiler_version="scientific-planner-option-filter-rank.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "candidate_predictions": ["registered_intermediate", "verified_output"]
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="render_report",
        required_artifacts=["ranked_candidates"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["report_markdown", "report_html"],
        risk_level=RiskLevel.LOW,
        default_adapter="render_report_adapter",
        scientific_tool_id="render_report",
        label="Render scientific report",
        description="Render a reviewable report from the registered ranked-candidate artifact.",
        effect_class="mutate_artifacts",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={"ranked_candidates": ["verified_output"]},
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="parse_document",
        required_artifacts=["pdf_corpus"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["parsed_document", "parsed_tables", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.DATA_MINING.value],
        default_adapter="parse_document_mineru_adapter",
        scientific_tool_id="parse_document",
        label="Parse PDF corpus",
        description="Parse a content-bound PDF corpus through the registered document-parsing workflow.",
        effect_class="compute",
        required_permissions=[
            "read_content_bound_input",
            "derive_project_artifact",
            "external_document_processing",
        ],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=["document_parsing"],
        backend_profile_requirements={},
        execution_route="remote_execution_service",
        remote_task_type="document_parsing",
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "pdf_corpus": ["content_bound_input", "registered_intermediate", "verified_output"]
        },
        budget_dimensions=["max_runtime_sec", "max_gpu_hours"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="parse_document_pdfplumber",
        required_artifacts=["pdf_corpus"],
        output_artifacts=["parsed_document", "parsed_tables", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="parse_document_pdfplumber_adapter",
    ),
    AtomicTaskSpec(
        task_id="parse_pdf_corpus_pdfplumber",
        required_artifacts=["pdf_corpus"],
        output_artifacts=["parsed_corpus_manifest", "parser_audit"],
        risk_level=RiskLevel.MEDIUM,
        gates=[GateName.DATA_MINING.value],
        default_adapter="parse_pdf_corpus_pdfplumber_adapter",
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
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["corpus_index", "evidence_chunks"],
        risk_level=RiskLevel.LOW,
        default_adapter="index_corpus_adapter",
        scientific_tool_id="index_corpus",
        label="Index parsed corpus",
        description="Build the registered logical index from parsed document artifacts.",
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "parsed_document": ["registered_intermediate", "verified_output"]
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
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
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["evidence_hits", "retrieval_log"],
        risk_level=RiskLevel.LOW,
        default_adapter="retrieve_evidence_adapter",
        scientific_tool_id="retrieve_evidence",
        label="Retrieve evidence",
        description="Retrieve bounded evidence from the registered corpus index.",
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(
            {
                "query": {
                    "type": ["string", "null"],
                    "minLength": 1,
                    "maxLength": 2000,
                },
                "topk": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            required=["query"],
        ),
        default_planner_options={"query": None, "topk": 10},
        backend_default_planner_options={},
        review_required_option_ids=["query"],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "corpus_index": ["registered_intermediate", "verified_output"]
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="extract_records",
        required_artifacts=["evidence_hits", "evidence_chunks"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=[
            "extracted_records",
            "rejected_records",
            "extraction_confidence_report",
            "candidate_training_dataset",
        ],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="extract_records_adapter",
        scientific_tool_id="extract_records",
        label="Extract scientific records",
        description="Extract structured records from registered evidence artifacts.",
        effect_class="compute",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["evidence_hits", "evidence_chunks"], ["registered_intermediate", "verified_output"]
        ),
        budget_dimensions=["max_records", "max_runtime_sec"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="normalize_extracted_units",
        required_artifacts=["extracted_records"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=[
            "normalized_extracted_records",
            "candidate_training_dataset",
            "unit_normalization_report",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="normalize_extracted_units_adapter",
        scientific_tool_id="normalize_extracted_units",
        label="Normalize extracted units",
        description="Normalize units in registered extracted scientific records.",
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "extracted_records": ["registered_intermediate", "verified_output"]
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="track_citation_provenance",
        required_artifacts=["parsed_document", "evidence_hits", "extracted_records"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["citation_provenance_report", "audit_summary"],
        risk_level=RiskLevel.LOW,
        default_adapter="track_citation_provenance_adapter",
        scientific_tool_id="track_citation_provenance",
        label="Track citation provenance",
        description="Build the registered provenance report for parsed and extracted records.",
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["parsed_document", "evidence_hits", "extracted_records"],
            ["registered_intermediate", "verified_output"],
        ),
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="merge_extracted_records",
        required_artifacts=["normalized_extracted_records"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["merged_records", "conflict_report", "candidate_training_dataset"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="merge_extracted_records_adapter",
        scientific_tool_id="merge_extracted_records",
        label="Merge extracted records",
        description="Merge normalized records using registered provenance and conflict rules.",
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["normalized_extracted_records"],
            ["registered_intermediate", "verified_output"],
        ),
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="evaluate_extraction_benchmark",
        required_artifacts=["evidence_hits", "normalized_extracted_records", "conflict_report"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["extraction_benchmark_report"],
        risk_level=RiskLevel.LOW,
        default_adapter="evaluate_extraction_benchmark_adapter",
        scientific_tool_id="evaluate_extraction_benchmark",
        label="Evaluate extraction benchmark",
        description="Evaluate registered extraction quality evidence without changing scientific confirmation.",
        effect_class="observe",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["evidence_hits", "normalized_extracted_records", "conflict_report"],
            ["registered_intermediate", "verified_output"],
        ),
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="confirm_extracted_dataset",
        required_artifacts=[
            "candidate_training_dataset",
            "conflict_report",
            "citation_provenance_report",
        ],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["confirmed_training_dataset", "extraction_confirmation_record"],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.DATA_MINING.value],
        default_adapter="confirm_extracted_dataset_adapter",
        scientific_tool_id="confirm_extracted_dataset",
        label="Confirm extracted dataset",
        description="Prepare the registered dataset-confirmation task for review; it does not grant confirmation authority.",
        effect_class="scientific_confirm",
        required_permissions=[
            "derive_project_artifact",
            "scientific_dataset_confirmation",
        ],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=_uniform_input_trust(
            ["candidate_training_dataset", "conflict_report", "citation_provenance_report"],
            ["content_bound_input", "registered_intermediate", "verified_output"],
        ),
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
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
    AtomicTaskSpec(
        task_id="prepare_structured_dataset_canary",
        required_artifacts=["uploaded_dataset"],
        output_artifacts=["raw_dataset", "raw_dataset_csv", "review_snapshot"],
        risk_level=RiskLevel.LOW,
        default_adapter="prepare_structured_dataset_canary_adapter",
        scientific_tool_id="prepare_structured_dataset_canary",
        label="Prepare structured dataset canary",
        description="Publish a candidate Raw Dataset and immutable review snapshot.",
        effect_class="derive_local",
        required_permissions=["read_content_bound_input", "derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        accepted_input_trust_classes_by_artifact={
            "uploaded_dataset": ["content_bound_input"]
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="confirm_structured_dataset_canary",
        required_artifacts=["raw_dataset", "raw_dataset_csv", "review_snapshot"],
        output_artifacts=[
            "confirmation_receipt",
            "confirmed_training_dataset",
            "confirmed_training_dataset_csv",
        ],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.TRAIN_CONFIG.value],
        default_adapter="confirm_structured_dataset_canary_adapter",
        depends_on=["prepare_structured_dataset_canary"],
        scientific_tool_id="confirm_structured_dataset_canary",
        label="Confirm structured dataset canary",
        description="Consume an exact Controller GateDecision and publish the Confirmed Dataset.",
        effect_class="scientific_confirm",
        required_permissions=["scientific_dataset_confirmation", "derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        accepted_input_trust_classes_by_artifact={
            "raw_dataset": ["registered_intermediate", "verified_output"],
            "raw_dataset_csv": ["registered_intermediate", "verified_output"],
            "review_snapshot": ["registered_intermediate", "verified_output"],
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="controller_gate_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="train_structured_dataset_canary",
        required_artifacts=[
            "confirmation_receipt",
            "confirmed_training_dataset",
            "confirmed_training_dataset_csv",
        ],
        output_artifacts=["training_request", "trained_model", "model_package"],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.TRAIN_CONFIG.value],
        default_adapter="train_structured_dataset_canary_adapter",
        depends_on=["confirm_structured_dataset_canary"],
        scientific_tool_id="train_structured_dataset_canary",
        label="Train structured dataset canary model",
        description="Fresh-fit the CI baseline from the exact Confirmed Dataset.",
        effect_class="compute",
        required_permissions=["model_training_compute", "derive_project_artifact"],
        option_schema=_closed_option_schema(
            {"seed": {"type": "integer", "minimum": 0, "maximum": 2147483647}},
            required=["seed"],
        ),
        default_planner_options={"seed": 1729},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        accepted_input_trust_classes_by_artifact={
            "confirmation_receipt": ["verified_output"],
            "confirmed_training_dataset": ["confirmed_scientific_input", "verified_output"],
            "confirmed_training_dataset_csv": ["confirmed_scientific_input", "verified_output"],
        },
        budget_dimensions=["max_records", "max_runtime_sec"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="controller_dispatch_publication_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="generate_structured_dataset_canary",
        required_artifacts=["confirmed_training_dataset", "model_package"],
        output_artifacts=[
            "generation_request", "candidate_dataset", "generation_publication"
        ],
        risk_level=RiskLevel.HIGH,
        gates=[GateName.FINAL_THRESHOLD.value],
        default_adapter="generate_structured_dataset_canary_adapter",
        depends_on=["train_structured_dataset_canary"],
        scientific_tool_id="generate_structured_dataset_canary",
        label="Generate structured dataset canary candidates",
        description="Generate a current-run deterministic candidate roster from the current model package.",
        effect_class="compute",
        required_permissions=["candidate_generation_compute", "derive_project_artifact"],
        option_schema=_closed_option_schema(
            {"seed": {"type": "integer", "minimum": 0, "maximum": 2147483647}},
            required=["seed"],
        ),
        default_planner_options={"seed": 1729},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        accepted_input_trust_classes_by_artifact={
            "confirmed_training_dataset": ["confirmed_scientific_input", "verified_output"],
            "model_package": ["verified_output"],
        },
        budget_dimensions=["max_records", "max_runtime_sec"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="controller_dispatch_publication_registry_and_stage_verifier",
        planner_visible=True,
    ),
    AtomicTaskSpec(
        task_id="evaluate_structured_dataset_canary",
        required_artifacts=[
            "raw_dataset", "raw_dataset_csv", "review_snapshot", "confirmation_receipt",
            "confirmed_training_dataset", "confirmed_training_dataset_csv", "trained_model",
            "model_package", "generation_publication", "candidate_dataset",
        ],
        output_artifacts=[
            "prediction_publication", "candidate_validation", "ranking_publication",
            "computational_top_n", "structured_dataset_canary_evidence",
        ],
        risk_level=RiskLevel.LOW,
        default_adapter="evaluate_structured_dataset_canary_adapter",
        depends_on=["generate_structured_dataset_canary"],
        scientific_tool_id="evaluate_structured_dataset_canary",
        label="Evaluate structured dataset canary candidates",
        description="Predict, validate, rank, and publish Computational Top-N.",
        effect_class="compute",
        required_permissions=["model_inference_compute", "derive_project_artifact"],
        option_schema=_closed_option_schema(
            {
                "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            required=["seed", "top_n"],
        ),
        default_planner_options={"seed": 1729, "top_n": 5},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        accepted_input_trust_classes_by_artifact={
            artifact_id: ["registered_intermediate", "verified_output", "confirmed_scientific_input"]
            for artifact_id in [
                "raw_dataset", "raw_dataset_csv", "review_snapshot", "confirmation_receipt",
                "confirmed_training_dataset", "confirmed_training_dataset_csv", "trained_model",
                "model_package", "generation_publication", "candidate_dataset",
            ]
        },
        budget_dimensions=["max_records", "max_runtime_sec"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    ),
)


class AtomicTaskRegistry:
    def __init__(self, tasks: Iterable[AtomicTaskSpec] | None = None) -> None:
        source = list(tasks or DEFAULT_ATOMIC_TASKS)
        self._validate_tasks(source)
        self._tasks = {task.task_id: task for task in source}
        self._artifact_producers: dict[str, list[str]] = {}
        for task in source:
            for artifact in task.output_artifacts:
                self._artifact_producers.setdefault(artifact, []).append(task.task_id)

    @staticmethod
    def _validate_tasks(tasks: list[AtomicTaskSpec]) -> None:
        valid_gates = {gate.value for gate in GateName}
        task_ids = [task.task_id for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            duplicates = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
            raise ValueError(f"duplicate atomic task ID: {', '.join(duplicates)}")
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
        producers = self._artifact_producers.get(artifact_id, [])
        return producers[0] if producers else None

    def producers_for(self, artifact_id: str) -> list[str]:
        return list(self._artifact_producers.get(artifact_id, []))


def private_structured_dataset_task_registry_v2() -> AtomicTaskRegistry:
    """Build the explicitly selected private BR1 v2 catalog boundary.

    The default registry remains byte-for-byte frozen for PR-BM v1 replay.
    Trusted private server bootstrap must inject this registry consistently
    into planning, permission, Controller, and execution services.
    """

    prepare_v2 = AtomicTaskSpec(
        task_id="prepare_private_structured_dataset_canary_v2",
        required_artifacts=[
            "uploaded_dataset",
            "source_dataset_manifest",
            "br1_mapping_policy",
        ],
        output_artifacts=["raw_dataset", "raw_dataset_csv", "review_snapshot"],
        risk_level=RiskLevel.LOW,
        default_adapter="prepare_private_structured_dataset_canary_v2_adapter",
        scientific_tool_id="prepare_private_structured_dataset_canary_v2",
        label="Prepare private structured dataset canary v2",
        description=(
            "Publish a private-source Raw Dataset and condition-aware immutable "
            "review snapshot from exact source and mapping authority inputs."
        ),
        effect_class="derive_local",
        required_permissions=["read_content_bound_input", "derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        accepted_input_trust_classes_by_artifact={
            "uploaded_dataset": ["content_bound_input"],
            "source_dataset_manifest": ["content_bound_input"],
            "br1_mapping_policy": ["content_bound_input"],
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier.v2",
        planner_visible=True,
    )
    tasks: list[AtomicTaskSpec] = []
    for task in DEFAULT_ATOMIC_TASKS:
        if task.task_id == "prepare_structured_dataset_canary":
            tasks.append(prepare_v2)
        elif task.task_id == "confirm_structured_dataset_canary":
            tasks.append(
                task.model_copy(
                    update={
                        "depends_on": [
                            "prepare_private_structured_dataset_canary_v2"
                        ]
                    }
                )
            )
        else:
            tasks.append(task)
    return AtomicTaskRegistry(tasks)


def br2_contextual_mapping_task_registry_v1() -> AtomicTaskRegistry:
    """Add the BR2 mapping projection to the existing task catalog.

    The document parser remains the default ``parse_document`` task and is
    deliberately not overridden here.  These three local tasks only consume
    its verified output, build the existing deterministic OLED evidence, call
    the configured contextual mapper, and publish a review-only package.
    """

    trust = {
        "parsed_document": ["registered_intermediate", "verified_output"],
    }
    extract_evidence = AtomicTaskSpec(
        task_id="extract_oled_evidence",
        required_artifacts=["parsed_document"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["oled_mapping_evidence"],
        risk_level=RiskLevel.LOW,
        default_adapter="extract_oled_evidence_adapter",
        scientific_tool_id="extract_oled_evidence",
        label="Extract deterministic OLED evidence",
        description=(
            "Build evidence-bound OLED MinerU candidates, semantic packets, and "
            "deterministic schema candidates from a verified ParsedDocument."
        ),
        effect_class="derive_local",
        required_permissions=["read_content_bound_input", "derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact=trust,
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    )
    contextual_mapping = AtomicTaskSpec(
        task_id="map_oled_contextual_semantics",
        required_artifacts=["parsed_document", "oled_mapping_evidence"],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["contextual_mapping_result"],
        risk_level=RiskLevel.MEDIUM,
        default_adapter="map_oled_contextual_semantics_adapter",
        scientific_tool_id="map_oled_contextual_semantics",
        label="Map OLED contextual semantics",
        description=(
            "Call the configured structured-output LLM mapper on full document "
            "context and keep proposals review-only."
        ),
        effect_class="external_io",
        required_permissions=["external_document_processing", "derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "parsed_document": ["registered_intermediate", "verified_output"],
            "oled_mapping_evidence": ["registered_intermediate", "verified_output"],
        },
        budget_dimensions=["max_runtime_sec", "max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    )
    candidate_dataset = AtomicTaskSpec(
        task_id="prepare_oled_candidate_raw_dataset",
        required_artifacts=[
            "parsed_document",
            "oled_mapping_evidence",
            "contextual_mapping_result",
        ],
        optional_input_artifacts=[],
        input_artifact_alternatives=[],
        output_artifacts=["candidate_raw_dataset", "candidate_raw_dataset_review"],
        risk_level=RiskLevel.LOW,
        default_adapter="prepare_oled_candidate_raw_dataset_adapter",
        scientific_tool_id="prepare_oled_candidate_raw_dataset",
        label="Prepare OLED candidate raw dataset",
        description=(
            "Compile evidence-bound layered OLED candidates into a review-only "
            "package without confirmation or downstream execution."
        ),
        effect_class="derive_local",
        required_permissions=["derive_project_artifact"],
        option_schema=_closed_option_schema(),
        default_planner_options={},
        backend_default_planner_options={},
        review_required_option_ids=[],
        option_compiler_version="scientific-planner-option-identity.v1",
        logical_profile_requirements=[],
        backend_profile_requirements={},
        execution_route="local_executor",
        remote_task_type=None,
        backend_execution_routes={},
        backend_remote_task_types={},
        accepted_input_trust_classes_by_artifact={
            "parsed_document": ["registered_intermediate", "verified_output"],
            "oled_mapping_evidence": ["registered_intermediate", "verified_output"],
            "contextual_mapping_result": ["registered_intermediate", "verified_output"],
        },
        budget_dimensions=["max_records"],
        supports_plan_preapproval=False,
        idempotency_policy="server_checked",
        verification_policy="artifact_registry_and_stage_verifier",
        planner_visible=True,
    )
    return AtomicTaskRegistry(
        [*DEFAULT_ATOMIC_TASKS, extract_evidence, contextual_mapping, candidate_dataset]
    )


def private_structured_dataset_real_tool_task_registry_v3() -> AtomicTaskRegistry:
    """Build the explicitly selected BR1 real-tool runtime catalog.

    The default v1 and private preparation v2 catalogs remain frozen replay
    boundaries.  This v3 catalog decomposes request construction, remote tool
    execution, and verified scientific packaging so a remote output cannot be
    mistaken for the final Model Package or generation publication.
    """

    private_v2 = private_structured_dataset_task_registry_v2()
    retained = [
        task
        for task in private_v2.list_tasks()
        if task.task_id
        not in {
            "train_structured_dataset_canary",
            "generate_structured_dataset_canary",
            "evaluate_structured_dataset_canary",
        }
    ]

    def trust(*artifact_ids: str) -> dict[str, list[str]]:
        return {
            artifact_id: [
                "confirmed_scientific_input",
                "registered_intermediate",
                "verified_output",
            ]
            for artifact_id in artifact_ids
        }

    tasks = [
        AtomicTaskSpec(
            task_id="prepare_private_unimol_training_v1",
            required_artifacts=[
                "confirmation_receipt",
                "confirmed_training_dataset",
                "confirmed_training_dataset_csv",
            ],
            output_artifacts=[
                "unimol_split_manifest",
                "unimol_training_dataset_csv",
                "unimol_training_request",
                "unimol_training_config",
            ],
            risk_level=RiskLevel.LOW,
            default_adapter="prepare_private_unimol_training_v1_adapter",
            depends_on=["confirm_structured_dataset_canary"],
            scientific_tool_id="prepare_private_unimol_training_v1",
            label="Freeze private Uni-Mol training request",
            description="Bind the exact Confirmed Dataset to a closed Uni-Mol training configuration.",
            effect_class="derive_local",
            required_permissions=["derive_project_artifact"],
            option_schema=_closed_option_schema(
                {
                    "batch_size": {"type": "integer", "minimum": 1, "maximum": 4096},
                    "early_stopping": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "epochs": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "gpu_device": {"type": "integer", "minimum": 0, "maximum": 64},
                    "learning_rate": {"type": "number", "minimum": 0.000000000001, "maximum": 1},
                    "seed": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                },
                required=[
                    "batch_size",
                    "early_stopping",
                    "epochs",
                    "gpu_device",
                    "learning_rate",
                    "seed",
                ],
            ),
            default_planner_options={
                "batch_size": 8,
                "early_stopping": 3,
                "epochs": 6,
                "gpu_device": 0,
                "learning_rate": 0.0001,
                "seed": 1729,
            },
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-unimol-training-options.v1",
            logical_profile_requirements=[],
            backend_profile_requirements={},
            execution_route="local_executor",
            remote_task_type=None,
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "confirmation_receipt",
                "confirmed_training_dataset",
                "confirmed_training_dataset_csv",
            ),
            budget_dimensions=["max_records"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="br1_private_request_registry_verifier.v1",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="train_private_unimol_v1",
            required_artifacts=[
                "unimol_training_dataset_csv",
                "unimol_training_config",
            ],
            output_artifacts=[
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_target_scaler",
                "unimol_training_audit",
                "unimol_training_metrics",
            ],
            risk_level=RiskLevel.HIGH,
            gates=[GateName.TRAIN_CONFIG.value],
            default_adapter=None,
            depends_on=["prepare_private_unimol_training_v1"],
            scientific_tool_id="train_private_unimol_v1",
            label="Train private Uni-Mol model",
            description="Train a fresh prediction-capable Uni-Mol model through the server-owned remote profile.",
            effect_class="compute",
            required_permissions=["model_training_compute"],
            option_schema=_closed_option_schema(),
            default_planner_options={},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-remote-task-options.v1",
            logical_profile_requirements=["model_training"],
            backend_profile_requirements={},
            execution_route="remote_execution_service",
            remote_task_type="model_training",
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "unimol_training_dataset_csv", "unimol_training_config"
            ),
            budget_dimensions=["max_gpu_hours", "max_runtime_sec"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="remote_publication_and_registry_exact.v2",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="package_private_unimol_model_v1",
            required_artifacts=[
                "confirmation_receipt",
                "confirmed_training_dataset",
                "unimol_training_request",
                "unimol_split_manifest",
                "unimol_training_dataset_csv",
                "unimol_training_config",
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_target_scaler",
                "unimol_training_audit",
                "unimol_training_metrics",
            ],
            output_artifacts=["training_request", "trained_model", "model_package"],
            risk_level=RiskLevel.LOW,
            default_adapter="package_private_unimol_model_v1_adapter",
            depends_on=["train_private_unimol_v1"],
            scientific_tool_id="package_private_unimol_model_v1",
            label="Package current Uni-Mol model",
            description="Publish a current-run Model Package bound to the verified remote Uni-Mol outputs.",
            effect_class="derive_local",
            required_permissions=["derive_project_artifact"],
            option_schema=_closed_option_schema(),
            default_planner_options={},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-local-package-options.v1",
            logical_profile_requirements=[],
            backend_profile_requirements={},
            execution_route="local_executor",
            remote_task_type=None,
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "confirmation_receipt",
                "confirmed_training_dataset",
                "unimol_training_request",
                "unimol_split_manifest",
                "unimol_training_dataset_csv",
                "unimol_training_config",
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_target_scaler",
                "unimol_training_audit",
                "unimol_training_metrics",
            ),
            budget_dimensions=["max_records"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="br1_private_model_package_verifier.v1",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="prepare_private_reinvent4_generation_v1",
            required_artifacts=[
                "confirmed_training_dataset",
                "model_package",
                "reinvent4_config_template",
            ],
            output_artifacts=[
                "generation_request",
                "reinvent4_bound_config",
                "reinvent4_execution_request",
            ],
            risk_level=RiskLevel.LOW,
            default_adapter="prepare_private_reinvent4_generation_v1_adapter",
            depends_on=["package_private_unimol_model_v1"],
            scientific_tool_id="prepare_private_reinvent4_generation_v1",
            label="Freeze private REINVENT4 request",
            description="Bind the current model and Confirmed Dataset to a frozen REINVENT4 configuration.",
            effect_class="derive_local",
            required_permissions=["derive_project_artifact"],
            option_schema=_closed_option_schema(
                {"seed": {"type": "integer", "minimum": 0, "maximum": 2147483647}},
                required=["seed"],
            ),
            default_planner_options={"seed": 1729},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-reinvent4-options.v1",
            logical_profile_requirements=[],
            backend_profile_requirements={},
            execution_route="local_executor",
            remote_task_type=None,
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact={
                **trust("confirmed_training_dataset", "model_package"),
                "reinvent4_config_template": [
                    "content_bound_input",
                    "confirmed_scientific_input",
                    "registered_intermediate",
                    "verified_output",
                ],
            },
            budget_dimensions=["max_records"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="br1_private_request_registry_verifier.v1",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="generate_private_reinvent4_v1",
            required_artifacts=[
                "reinvent4_bound_config",
                "reinvent4_execution_request",
            ],
            output_artifacts=[
                "reinvent4_candidates",
                "reinvent4_generation_audit",
            ],
            risk_level=RiskLevel.HIGH,
            gates=[GateName.FINAL_THRESHOLD.value],
            default_adapter=None,
            depends_on=["prepare_private_reinvent4_generation_v1"],
            scientific_tool_id="generate_private_reinvent4_v1",
            label="Generate private REINVENT4 candidates",
            description="Execute REINVENT4 for real through the server-owned remote profile.",
            effect_class="compute",
            required_permissions=["candidate_generation_compute"],
            option_schema=_closed_option_schema(),
            default_planner_options={},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-remote-task-options.v1",
            logical_profile_requirements=["molecular_generation"],
            backend_profile_requirements={},
            execution_route="remote_execution_service",
            remote_task_type="molecular_generation",
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "reinvent4_bound_config", "reinvent4_execution_request"
            ),
            budget_dimensions=["max_runtime_sec", "max_steps"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="remote_publication_and_registry_exact.v1",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="package_private_reinvent4_generation_v1",
            required_artifacts=[
                "confirmed_training_dataset",
                "model_package",
                "generation_request",
                "reinvent4_bound_config",
                "reinvent4_execution_request",
                "reinvent4_candidates",
                "reinvent4_generation_audit",
            ],
            output_artifacts=[
                "candidate_dataset",
                "candidate_dataset_csv",
                "generation_publication",
                "unimol_prediction_config",
            ],
            risk_level=RiskLevel.LOW,
            default_adapter="package_private_reinvent4_generation_v1_adapter",
            depends_on=["generate_private_reinvent4_v1"],
            scientific_tool_id="package_private_reinvent4_generation_v1",
            label="Package current REINVENT4 generation",
            description="Publish the current-run candidate roster and its exact real-generation bindings.",
            effect_class="derive_local",
            required_permissions=["derive_project_artifact"],
            option_schema=_closed_option_schema(),
            default_planner_options={},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-local-package-options.v1",
            logical_profile_requirements=[],
            backend_profile_requirements={},
            execution_route="local_executor",
            remote_task_type=None,
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "confirmed_training_dataset",
                "model_package",
                "generation_request",
                "reinvent4_bound_config",
                "reinvent4_execution_request",
                "reinvent4_candidates",
                "reinvent4_generation_audit",
            ),
            budget_dimensions=["max_records"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="br1_private_generation_package_verifier.v1",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="predict_private_unimol_v1",
            required_artifacts=[
                "candidate_dataset_csv",
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_prediction_config",
                "unimol_target_scaler",
            ],
            output_artifacts=["unimol_prediction_audit", "unimol_predictions"],
            risk_level=RiskLevel.MEDIUM,
            default_adapter=None,
            depends_on=["package_private_reinvent4_generation_v1"],
            scientific_tool_id="predict_private_unimol_v1",
            label="Predict current candidates with current Uni-Mol model",
            description="Run prediction remotely using only the current model package and current candidate roster.",
            effect_class="compute",
            required_permissions=["model_inference_compute"],
            option_schema=_closed_option_schema(),
            default_planner_options={},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-remote-task-options.v1",
            logical_profile_requirements=["model_inference"],
            backend_profile_requirements={},
            execution_route="remote_execution_service",
            remote_task_type="model_inference",
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "candidate_dataset_csv",
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_prediction_config",
                "unimol_target_scaler",
            ),
            budget_dimensions=["max_gpu_hours", "max_runtime_sec"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="remote_publication_and_registry_exact.v1",
            planner_visible=True,
        ),
        AtomicTaskSpec(
            task_id="evaluate_private_structured_dataset_canary_v1",
            required_artifacts=[
                "raw_dataset",
                "raw_dataset_csv",
                "review_snapshot",
                "confirmation_receipt",
                "confirmed_training_dataset",
                "confirmed_training_dataset_csv",
                "unimol_split_manifest",
                "model_package",
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_target_scaler",
                "generation_publication",
                "candidate_dataset",
                "candidate_dataset_csv",
                "unimol_prediction_config",
                "unimol_predictions",
                "unimol_prediction_audit",
            ],
            output_artifacts=[
                "prediction_publication",
                "candidate_validation",
                "ranking_publication",
                "computational_top_n",
                "structured_dataset_canary_evidence",
            ],
            risk_level=RiskLevel.LOW,
            default_adapter="evaluate_private_structured_dataset_canary_v1_adapter",
            depends_on=["predict_private_unimol_v1"],
            scientific_tool_id="evaluate_private_structured_dataset_canary_v1",
            label="Validate and rank private BR1 candidates",
            description="Bind current-model predictions, chemical validation, deterministic ranking, and Computational Top-N.",
            effect_class="derive_local",
            required_permissions=["derive_project_artifact"],
            option_schema=_closed_option_schema(
                {
                    "top_n": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "validation_seed": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2147483647,
                    },
                },
                required=["top_n", "validation_seed"],
            ),
            default_planner_options={"top_n": 5, "validation_seed": 1729},
            backend_default_planner_options={},
            review_required_option_ids=[],
            option_compiler_version="br1-private-evaluation-options.v1",
            logical_profile_requirements=[],
            backend_profile_requirements={},
            execution_route="local_executor",
            remote_task_type=None,
            backend_execution_routes={},
            backend_remote_task_types={},
            optional_input_artifacts=[],
            input_artifact_alternatives=[],
            accepted_input_trust_classes_by_artifact=trust(
                "raw_dataset",
                "raw_dataset_csv",
                "review_snapshot",
                "confirmation_receipt",
                "confirmed_training_dataset",
                "confirmed_training_dataset_csv",
                "unimol_split_manifest",
                "model_package",
                "unimol_model_config",
                "unimol_model_weights",
                "unimol_target_scaler",
                "generation_publication",
                "candidate_dataset",
                "candidate_dataset_csv",
                "unimol_prediction_config",
                "unimol_predictions",
                "unimol_prediction_audit",
            ),
            budget_dimensions=["max_records"],
            supports_plan_preapproval=False,
            idempotency_policy="server_checked",
            verification_policy="br1_private_scientific_chain_verifier.v1",
            planner_visible=True,
        ),
    ]
    return AtomicTaskRegistry([*retained, *tasks])


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
    required_by_task: dict[str, list[str]] = {}
    dedup_requested: list[str] = []
    for requested in requested_tasks:
        if requested not in dedup_requested:
            dedup_requested.append(requested)

    requested_producers: dict[str, list[str]] = {}
    for task_id in dedup_requested:
        spec = task_registry.get(task_id)
        for artifact in spec.output_artifacts:
            requested_producers.setdefault(artifact, []).append(task_id)
    requested_output_artifacts = set(requested_producers)

    def direct_inputs_match_snapshot(task_id: str) -> bool:
        candidate = task_registry.get(task_id)
        if any(
            artifact not in pre_existing_artifacts
            and artifact not in requested_output_artifacts
            for artifact in candidate.required_artifacts
        ):
            return False
        return all(
            any(
                artifact in pre_existing_artifacts
                or artifact in requested_output_artifacts
                for artifact in alternatives
            )
            for alternatives in candidate.input_artifact_alternatives
        )

    def depends_transitively(
        task_id: str,
        dependency_id: str,
        seen: set[str] | None = None,
    ) -> bool:
        if task_id == dependency_id:
            return False
        visited = set(seen or set())
        if task_id in visited:
            return False
        visited.add(task_id)
        task = task_registry.get(task_id)
        if dependency_id in task.depends_on:
            return True
        return any(
            depends_transitively(item, dependency_id, visited)
            for item in task.depends_on
        )

    def select_producer(artifact_id: str) -> str | None:
        producers = task_registry.producers_for(artifact_id)
        if not producers:
            return None
        compatible = [
            producer for producer in producers if direct_inputs_match_snapshot(producer)
        ]
        candidates = compatible or producers
        candidate_set = set(candidates)
        requested = set(requested_producers.get(artifact_id, []))
        source_order = {task_id: index for index, task_id in enumerate(producers)}
        return max(
            candidates,
            key=lambda producer: (
                sum(
                    1
                    for other in candidate_set
                    if depends_transitively(producer, other)
                ),
                int(producer in requested),
                -source_order[producer],
            ),
        )

    def resolve_task(task_id: str) -> None:
        if task_id in resolved:
            return
        if task_id in resolving:
            raise ValueError(f"cyclic dependency detected: {task_id}")
        spec = task_registry.get(task_id)
        resolving.add(task_id)

        dependencies: list[str] = []
        unresolved_requirements: list[str] = []
        selected_requirements = list(spec.required_artifacts)

        for alternatives in spec.input_artifact_alternatives:
            selected = next(
                (
                    artifact
                    for artifact in alternatives
                    if artifact in pre_existing_artifacts
                ),
                None,
            )
            if selected is None:
                selected = next(
                    (
                        artifact
                        for artifact in alternatives
                        if artifact in requested_producers
                    ),
                    None,
                )
            if selected is None:
                first_alternative = alternatives[0]
                producer = select_producer(first_alternative)
                if producer is not None:
                    selected = first_alternative
            if selected is None:
                selected = alternatives[0]
            selected_requirements.append(selected)
            if selected in pre_existing_artifacts:
                continue
            producer = select_producer(selected)
            if producer is None:
                unresolved_requirements.append(selected)
                missing_artifacts.add(selected)
                continue
            if producer == task_id:
                raise ValueError(
                    f"self-referencing alternative artifact dependency in task {task_id}: {selected}"
                )
            dependencies.append(producer)

        for required in spec.required_artifacts:
            if required in pre_existing_artifacts:
                continue
            producer = select_producer(required)
            if producer == task_id:
                raise ValueError(
                    f"self-referencing artifact dependency in task {task_id}: {required}"
                )
            if producer:
                dependencies.append(producer)
                continue
            unresolved_requirements.append(required)
            missing_artifacts.add(required)

        dependencies.extend(spec.depends_on)

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
        required_by_task[task_id] = list(dict.fromkeys(selected_requirements))
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
                required_artifacts=list(required_by_task.get(task_id, spec.required_artifacts)),
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
