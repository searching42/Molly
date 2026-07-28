import pytest

import ai4s_agent.planner as planner
import ai4s_agent.adapters as adapters
from ai4s_agent.planner import AtomicTaskRegistry, build_plan, diff_run_plans, expand_run_plan
from ai4s_agent.schemas import AtomicTaskSpec, GateName, RiskLevel


def test_build_plan_has_phase1_steps() -> None:
    plan = build_plan(run_id="r1", prompt="maximize target properties")
    names = [s.name for s in plan.steps]
    assert "inspect_dataset" in names
    assert "clean_dataset" in names
    assert "check_trainability" in names
    assert "run_baseline" in names
    assert "train_model" in names
    assert "generate_candidates" in names
    assert "predict_candidates" in names
    assert "filter_rank" in names
    assert "render_report" in names


def test_expand_run_plan_adds_required_dependencies() -> None:
    plan = expand_run_plan(
        run_id="r1",
        requested_tasks=["render_report"],
        available_artifacts=["candidate_dataset"],
    )
    ordered_ids = [task.task_id for task in plan.tasks]
    assert ordered_ids == [
        "inspect_dataset",
        "clean_dataset",
        "check_trainability",
        "train_model",
        "predict_candidates",
        "filter_rank",
        "render_report",
    ]
    assert plan.missing_artifacts == []


def test_train_model_task_declares_promotable_model_package_outputs() -> None:
    spec = AtomicTaskRegistry().get("train_model")

    assert "trained_model" in spec.output_artifacts
    assert "model_metadata" in spec.output_artifacts
    assert "model_manifest" in spec.output_artifacts
    assert "domain_model_manifest" in spec.output_artifacts
    assert "model_diagnostics_report" in spec.output_artifacts
    assert "model_package_review" in spec.output_artifacts


def test_expand_run_plan_tracks_missing_artifacts_without_producers() -> None:
    registry = AtomicTaskRegistry(
        [
            AtomicTaskSpec(
                task_id="need_ghost_artifact",
                required_artifacts=["ghost_artifact"],
                output_artifacts=["ghost_result"],
            )
        ]
    )
    plan = expand_run_plan(
        run_id="r1",
        requested_tasks=["need_ghost_artifact"],
        available_artifacts=[],
        registry=registry,
    )
    assert "ghost_artifact" in plan.missing_artifacts
    task = [task for task in plan.tasks if task.task_id == "need_ghost_artifact"][0]
    assert task.unresolved_requirements == ["ghost_artifact"]


def test_diff_run_plans_reports_added_and_removed_tasks() -> None:
    before = expand_run_plan(
        run_id="r1",
        requested_tasks=["run_baseline"],
        available_artifacts=[],
    )
    after = expand_run_plan(
        run_id="r1",
        requested_tasks=["render_report"],
        available_artifacts=["candidate_dataset"],
    )
    diff = diff_run_plans(before, after)
    assert "run_baseline" in diff.removed_tasks
    assert "predict_candidates" in diff.added_tasks


def test_expand_run_plan_reuses_dependencies_computed_during_resolution(monkeypatch) -> None:
    registry = AtomicTaskRegistry(
        [
            AtomicTaskSpec(task_id="produce_a", output_artifacts=["artifact_a"]),
            AtomicTaskSpec(task_id="consume_a", required_artifacts=["artifact_a"]),
        ]
    )

    assert not hasattr(planner, "_expanded_dependencies")
    plan = planner.expand_run_plan(
        run_id="r1",
        requested_tasks=["consume_a"],
        registry=registry,
    )
    consume = [task for task in plan.tasks if task.task_id == "consume_a"][0]
    assert consume.depends_on == ["produce_a"]


def test_expand_run_plan_rejects_self_referencing_artifacts() -> None:
    registry = AtomicTaskRegistry(
        [
            AtomicTaskSpec(
                task_id="bad_task",
                required_artifacts=["artifact_a"],
                output_artifacts=["artifact_a"],
            )
        ]
    )
    with pytest.raises(ValueError, match="self-referencing artifact"):
        expand_run_plan(run_id="r1", requested_tasks=["bad_task"], registry=registry)


def test_atomic_task_registry_rejects_high_risk_tasks_without_gates() -> None:
    with pytest.raises(ValueError, match="high-risk task requires gate: unsafe_task"):
        AtomicTaskRegistry([AtomicTaskSpec(task_id="unsafe_task", risk_level=RiskLevel.HIGH)])


def test_default_high_risk_tasks_declare_gates() -> None:
    registry = AtomicTaskRegistry()

    for task in registry.list_tasks():
        if task.risk_level == RiskLevel.HIGH:
            assert task.gates


def test_expand_run_plan_can_generate_missing_candidate_dataset() -> None:
    plan = expand_run_plan(
        run_id="r1",
        requested_tasks=["render_report"],
        available_artifacts=[],
    )
    ordered_ids = [task.task_id for task in plan.tasks]
    assert "generate_candidates" in ordered_ids
    assert ordered_ids.index("generate_candidates") < ordered_ids.index("predict_candidates")
    assert "candidate_dataset" not in plan.missing_artifacts


def test_phase3_parse_document_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("parse_document")

    assert task.required_artifacts == ["pdf_corpus"]
    assert task.output_artifacts == ["parsed_document", "parsed_tables", "parser_audit"]
    assert task.default_adapter == "parse_document_mineru_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_prepare_literature_corpus_sources_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("prepare_literature_corpus_sources")

    assert task.required_artifacts == []
    assert task.output_artifacts == ["corpus_source_manifest"]
    assert task.default_adapter == "prepare_literature_corpus_sources_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_acquire_literature_sources_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("acquire_literature_sources")

    assert task.required_artifacts == ["corpus_source_manifest"]
    assert task.output_artifacts == ["pdf_corpus", "structured_datasets", "acquisition_manifest"]
    assert task.risk_level == RiskLevel.HIGH
    assert task.gates == [GateName.DATA_MINING.value]
    assert task.default_adapter == "acquire_literature_sources_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_retrieval_atomic_tasks_are_registered() -> None:
    registry = AtomicTaskRegistry()
    index_task = registry.get("index_corpus")
    multi_index_task = registry.get("build_multi_index")
    dense_index_task = registry.get("build_dense_index")
    retrieve_task = registry.get("retrieve_evidence")

    assert index_task.required_artifacts == ["parsed_document"]
    assert index_task.output_artifacts == ["corpus_index", "evidence_chunks"]
    assert index_task.default_adapter == "index_corpus_adapter"
    assert multi_index_task.required_artifacts == ["evidence_chunks"]
    assert multi_index_task.output_artifacts == ["multi_index"]
    assert multi_index_task.default_adapter == "build_multi_index_adapter"
    assert dense_index_task.required_artifacts == ["evidence_chunks"]
    assert dense_index_task.output_artifacts == ["dense_index"]
    assert dense_index_task.default_adapter == "build_dense_index_adapter"
    assert retrieve_task.required_artifacts == ["corpus_index"]
    assert retrieve_task.output_artifacts == ["evidence_hits", "retrieval_log"]
    assert retrieve_task.default_adapter == "retrieve_evidence_adapter"
    assert callable(getattr(adapters, index_task.default_adapter))
    assert callable(getattr(adapters, multi_index_task.default_adapter))
    assert callable(getattr(adapters, dense_index_task.default_adapter))
    assert callable(getattr(adapters, retrieve_task.default_adapter))


def test_phase3_extraction_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("extract_records")

    assert task.required_artifacts == ["evidence_hits", "evidence_chunks"]
    assert task.output_artifacts == [
        "extracted_records",
        "rejected_records",
        "extraction_confidence_report",
        "candidate_training_dataset",
    ]
    assert task.default_adapter == "extract_records_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_citation_provenance_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("track_citation_provenance")

    assert task.required_artifacts == ["parsed_document", "evidence_hits", "extracted_records"]
    assert task.output_artifacts == ["citation_provenance_report", "audit_summary"]
    assert task.default_adapter == "track_citation_provenance_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_merge_conflict_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("merge_extracted_records")

    assert task.required_artifacts == ["normalized_extracted_records", "citation_provenance_report"]
    assert task.output_artifacts == ["merged_records", "conflict_report", "candidate_training_dataset"]
    assert task.default_adapter == "merge_extracted_records_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_confirmation_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("confirm_extracted_dataset")

    assert task.required_artifacts == [
        "candidate_training_dataset",
        "conflict_report",
        "citation_provenance_report",
    ]
    assert task.output_artifacts == ["confirmed_training_dataset", "extraction_confirmation_record"]
    assert task.gates == [GateName.DATA_MINING.value]
    assert task.default_adapter == "confirm_extracted_dataset_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_literature_to_dataset_workflow_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("literature_to_dataset_workflow")

    assert task.required_artifacts == ["pdf_corpus"]
    assert task.output_artifacts == [
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
    ]
    assert task.gates == [GateName.DATA_MINING.value]
    assert task.default_adapter == "literature_to_dataset_workflow_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_public_leakage_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("check_public_dataset_leakage")

    assert task.required_artifacts == ["candidate_training_dataset"]
    assert task.output_artifacts == ["benchmark_contamination_report"]
    assert task.default_adapter == "check_public_dataset_leakage_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_unit_normalization_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("normalize_extracted_units")

    assert task.required_artifacts == ["extracted_records"]
    assert task.output_artifacts == [
        "normalized_extracted_records",
        "candidate_training_dataset",
        "unit_normalization_report",
    ]
    assert task.default_adapter == "normalize_extracted_units_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_phase3_extraction_benchmark_atomic_task_is_registered() -> None:
    registry = AtomicTaskRegistry()
    task = registry.get("evaluate_extraction_benchmark")

    assert task.required_artifacts == ["evidence_hits", "normalized_extracted_records", "conflict_report"]
    assert task.output_artifacts == ["extraction_benchmark_report"]
    assert task.default_adapter == "evaluate_extraction_benchmark_adapter"
    assert callable(getattr(adapters, task.default_adapter))


def test_default_atomic_task_adapters_resolve_to_exported_callables() -> None:
    registry = AtomicTaskRegistry()

    for task in registry.list_tasks():
        assert task.default_adapter
        assert callable(getattr(adapters, task.default_adapter))
