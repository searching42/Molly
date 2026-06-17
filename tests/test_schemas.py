from pathlib import Path

import pytest
from pydantic import ValidationError

from ai4s_agent.schemas import (
    AssetManifest,
    AssetPromotionRecord,
    AssetStatus,
    BackgroundJobBudget,
    BackgroundJobCheckpoint,
    BackgroundJobState,
    CandidateSourceType,
    CorpusMultiIndex,
    DenseRetrievalIndex,
    CorpusChunk,
    EvidenceHit,
    ExtractionBenchmarkReport,
    ExtractedRecord,
    ExtractionConfirmationRecord,
    ExtractionConfidenceReport,
    CitationLicenseReport,
    ConflictGroup,
    ConflictReport,
    GenerationBackend,
    GenerationCandidate,
    GenerationConstraint,
    GenerationFrontierTarget,
    GenerationReport,
    GenerationStrategyProposal,
    GenerationTradeoff,
    LiteratureAcquisitionItem,
    LiteratureAcquisitionManifest,
    LiteratureCorpusManifest,
    LiteratureCorpusSource,
    ParsedDocument,
    ParsedDocumentElement,
    ParsedTable,
    PlanStep,
    LiteratureSourceProvenance,
    MergedRecord,
    CORE_SCHEMA_MODELS,
    PromotedModelAsset,
    RunPlan,
    RunPlanDiff,
    RunStatus,
    ResearchEvidenceQuality,
    ModelingBackendRecommendation,
    ModelingExperimentDesign,
    ModelingMetricInterpretation,
    ModelingPlanProposal,
    ModelingRetryProposal,
    MultiUserBoundaryCheck,
    MultiUserDeploymentReadiness,
    PredictionPreparation,
    ReportNextStep,
    ReportSection,
    ReportSynthesisProposal,
    RemoteWorkerAssignment,
    RemoteWorkerConfig,
    RemoteWorkerRequest,
    ResearchQueryExpansion,
    ResearchSourceCandidate,
    ResearchSourceProposal,
    StageState,
    UnitNormalizationReport,
    export_json_schemas,
)


def test_package_importable() -> None:
    import ai4s_agent  # noqa: F401


def test_core_schema_roundtrip() -> None:
    stage = StageState(
        stage="train",
        next_stage="predict",
        status=RunStatus.RUNNING,
        started_at="2026-05-28T10:00:00Z",
        updated_at="2026-05-28T10:10:00Z",
    )
    manifest = AssetManifest(
        asset_id="model-1",
        asset_type="model",
        version="v001",
        status=AssetStatus.CANDIDATE,
        created_from_run_id="run-1",
        source_artifacts=["03_training/model.bin"],
        content_hash="sha256:model",
    )
    record = AssetPromotionRecord(
        run_id="run-1",
        asset_id="model-1",
        asset_type="model",
        version="v001",
        source_artifacts=["03_training/model.bin"],
        approved_by="user",
        approved_at="2026-05-28T10:11:00Z",
    )
    plan = RunPlan(
        run_id="run-1",
        requested_tasks=["train_model"],
        tasks=[],
        available_artifacts=["trainability_report"],
    )
    diff = RunPlanDiff(added_tasks=["train_model"])

    for model in [stage, manifest, record, plan, diff]:
        payload = model.model_dump_json()
        restored = model.__class__.model_validate_json(payload)
        assert restored.model_dump(mode="json") == model.model_dump(mode="json")


def test_export_json_schemas(tmp_path: Path) -> None:
    exported = export_json_schemas(tmp_path)
    names = {path.name for path in exported}
    assert "asset_manifest.schema.json" in names
    assert "promoted_model_asset.schema.json" in names
    assert "prediction_preparation.schema.json" in names
    assert "model_package_review.schema.json" in names
    assert "run_plan.schema.json" in names
    assert "run_plan_diff.schema.json" in names


def test_docs_schema_files_include_every_core_schema() -> None:
    docs_dir = Path(__file__).resolve().parents[1] / "docs" / "schemas"
    missing = [
        f"{name}.schema.json"
        for name in CORE_SCHEMA_MODELS
        if not (docs_dir / f"{name}.schema.json").exists()
    ]
    assert missing == []


def test_promoted_model_asset_schema_roundtrip_and_metrics_validation() -> None:
    asset = PromotedModelAsset(
        asset_id="model/unimol_with_solvent_pca64/plqy/v007",
        model_id="plqy_request_specific_v007",
        domain="oled",
        property_id="plqy",
        use_case="scalar_prediction",
        backend="unimol_with_solvent_pca64",
        model_dir="projects/proj-oled/assets/models/plqy/v007/model",
        created_from_run_id="run-train-plqy-v007",
        source_artifacts=["03_training/domain_model_manifest.json"],
        approved_by="user",
        approved_at="2026-06-17T08:30:00Z",
        metrics={"mae": 0.171, "r2": 0.41},
        applicability={"domain": "OLED", "split": "scaffold"},
        feature_requirements=["canonical_smiles", "solvent", "solvent"],
        input_columns={"canonical_smiles": "SMILES", "solvent": "solvent"},
        limitations=["not a universal molecular design model"],
        rollback_asset_id="model/unimol_with_solvent_pca64/plqy/v006",
    )

    assert asset.feature_requirements == ["canonical_smiles", "solvent"]
    restored = PromotedModelAsset.model_validate_json(asset.model_dump_json())
    assert restored.model_dump(mode="json") == asset.model_dump(mode="json")

    with pytest.raises(ValidationError, match="must be a number, got bool"):
        PromotedModelAsset(
            asset_id="bad",
            model_id="bad",
            domain="oled",
            property_id="plqy",
            use_case="scalar_prediction",
            backend="baseline",
            model_dir="assets/models/bad",
            created_from_run_id="run-bad",
            approved_by="user",
            approved_at="2026-06-17T08:35:00Z",
            metrics={"r2": True},
        )


def test_generation_report_schema_roundtrip() -> None:
    report = GenerationReport(
        run_id="run-gen",
        backend=GenerationBackend.DETERMINISTIC_STUB,
        source_type=CandidateSourceType.GENERATOR,
        requested_count=5,
        generated_count=2,
        candidate_csv="04_generation/generated_candidates.csv",
        rescore_with_screener=True,
        candidates=[
            GenerationCandidate(candidate_id="gen_0001", smiles="CCO", source="deterministic_stub"),
            GenerationCandidate(candidate_id="gen_0002", smiles="CCN", source="deterministic_stub"),
        ],
        diversity={"unique_smiles_ratio": 1.0},
        novelty={"novel_smiles_ratio": 0.5},
        frontier_targets=[
            GenerationFrontierTarget(property_id="plqy", direction="maximize", weight=0.7),
            GenerationFrontierTarget(property_id="lambda_em", direction="target", target_value=520.0, weight=0.3),
        ],
        frontier_strategy="pareto_hint",
        frontier_summary={"target_count": 2, "note": "adapter guidance only"},
        provenance={"seed": 7, "backend": "deterministic_stub"},
    )

    restored = GenerationReport.model_validate_json(report.model_dump_json())
    assert restored.model_dump(mode="json") == report.model_dump(mode="json")
    assert restored.source_type == CandidateSourceType.GENERATOR
    assert restored.rescore_with_screener is True
    assert restored.frontier_targets[1].direction == "target"
    assert restored.frontier_summary["target_count"] == 2


def test_generation_strategy_proposal_schema_roundtrip() -> None:
    proposal = GenerationStrategyProposal(
        run_id="run-generation-proposal",
        goal="Generate diverse candidates.",
        backend=GenerationBackend.DETERMINISTIC_STUB,
        requested_count=32,
        strategy="deterministic_diversity_seed",
        frontier_targets=[GenerationFrontierTarget(property_id="plqy", direction="maximize", weight=1.0)],
        constraints=[
            GenerationConstraint(
                constraint_id="mw_limit",
                property_id="mw",
                operator="<=",
                value=700,
                hard=True,
            )
        ],
        tradeoffs=[
            GenerationTradeoff(
                name="diversity_novelty",
                recommendation="Prioritize diversity before expensive generation.",
                diversity_weight=0.5,
                novelty_weight=0.4,
                exploitation_weight=0.1,
            )
        ],
        required_gates=["gate_5_final_threshold"],
        adapter_payload={"backend": "deterministic_stub", "count": 32},
    )

    restored = GenerationStrategyProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")
    assert restored.constraints[0].operator == "<="


def test_literature_corpus_manifest_schema_roundtrip() -> None:
    source = LiteratureCorpusSource(
        source_id="doi_abc123",
        source_type="doi",
        value="10.1000/example",
        doi="10.1000/example",
        title="Example paper",
        status="pending_acquisition",
        metadata={"priority": 1},
    )
    manifest = LiteratureCorpusManifest(
        run_id="run-lit",
        source_count=1,
        source_type_counts={"doi": 1},
        sources=[source],
        created_at="2026-05-29T00:00:00Z",
        notes=["metadata-only source manifest"],
    )

    restored = LiteratureCorpusManifest.model_validate_json(manifest.model_dump_json())
    assert restored.model_dump(mode="json") == manifest.model_dump(mode="json")


def test_literature_corpus_source_rejects_unknown_source_type() -> None:
    with pytest.raises(ValidationError):
        LiteratureCorpusSource(
            source_id="bad",
            source_type="rss_feed",
            value="https://example.org/feed",
        )


def test_research_source_proposal_schema_roundtrip() -> None:
    proposal = ResearchSourceProposal(
        run_id="run-research-schema",
        goal="Find OLED PLQY papers.",
        query_expansion=ResearchQueryExpansion(
            original_goal="Find OLED PLQY papers.",
            expanded_queries=["OLED photoluminescence quantum yield"],
            rationale=["Expanded PLQY into its full property name."],
        ),
        source_candidates=[
            ResearchSourceCandidate(
                source_id="doi_abc123",
                source_type="doi",
                value="10.1000/schema",
                doi="10.1000/schema",
                score=0.95,
                rationale="Explicit DOI supplied by user.",
            )
        ],
        selected_sources=[
            LiteratureCorpusSource(
                source_id="doi_abc123",
                source_type="doi",
                value="10.1000/schema",
                doi="10.1000/schema",
            )
        ],
        evidence_quality=ResearchEvidenceQuality(
            source_count=1,
            ranked_source_count=1,
            doi_count=1,
            quality_score=0.75,
            quality_level="usable",
        ),
    )

    restored = ResearchSourceProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_modeling_plan_proposal_schema_roundtrip() -> None:
    proposal = ModelingPlanProposal(
        run_id="run-modeling-schema",
        goal="Train PLQY model.",
        backend_recommendations=[
            ModelingBackendRecommendation(
                property_id="plqy",
                backend="unimol",
                confidence=0.8,
                reason="3D-sensitive property.",
                requirements=["confirmed training dataset"],
            )
        ],
        experiment_design=ModelingExperimentDesign(
            backend="unimol",
            target_properties=["plqy"],
            split_strategy="scaffold_split_then_random_fallback",
            validation_strategy="holdout_with_baseline_comparison",
            required_artifacts=["cleaned_train_dataset", "trainability_report"],
            required_gates=["gate_3_train_config"],
        ),
        metric_interpretations=[
            ModelingMetricInterpretation(
                property_id="plqy",
                metrics={"r2": -0.1},
                status="weak",
                decision="replan",
                message="Negative R2 indicates weak generalization.",
            )
        ],
        retry_proposals=[
            ModelingRetryProposal(
                action="adjust_split",
                reason="Metrics are weak under the current split.",
                target_tasks=["run_baseline", "train_model"],
                requires_user_approval=True,
            )
        ],
    )

    restored = ModelingPlanProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_report_synthesis_proposal_schema_roundtrip() -> None:
    proposal = ReportSynthesisProposal(
        run_id="run-report-schema",
        goal="Summarize run.",
        executive_summary="The run needs review before promotion.",
        sections=[
            ReportSection(
                title="Verification",
                summary="One warning finding requires replanning.",
                evidence_refs=["verification_report.json"],
                risk_flags=["abnormal_model_metrics"],
            )
        ],
        limitations=["Model metrics are weak."],
        next_steps=[
            ReportNextStep(
                action="propose_replan",
                reason="Verifier requested replan.",
                priority="high",
                required_approval=True,
                related_artifacts=["verification_report_json"],
            )
        ],
        paper_audit_outline=["Methods", "Artifacts", "Limitations"],
    )

    restored = ReportSynthesisProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")
    assert restored.next_steps[0].priority == "high"


def test_remote_worker_schema_roundtrip_and_secret_rejection() -> None:
    worker = RemoteWorkerConfig(
        worker_id="workstation2-mineru",
        transport="ssh",
        host="workstation2",
        display_name="Workstation2 MinerU",
        capabilities=["gpu", "mineru_parse", "unimol_train"],
        work_dir="/remote/work/ai4s",
        environment="MinerU",
        max_concurrent_jobs=1,
        default_timeout_sec=3600,
    )
    request = RemoteWorkerRequest(
        project_id="proj-remote-worker",
        run_id="run-remote-worker",
        task_id="parse_document",
        required_capabilities=["mineru_parse", "gpu"],
        preferred_worker_id="workstation2-mineru",
        budget_limit_sec=1800,
        payload_ref="projects/proj-remote-worker/runs/run-remote-worker/input.pdf",
    )
    assignment = RemoteWorkerAssignment(
        assignment_id="assign-run-remote-worker-parse_document",
        project_id="proj-remote-worker",
        run_id="run-remote-worker",
        task_id="parse_document",
        worker_id="workstation2-mineru",
        transport="ssh",
        host="workstation2",
        matched_capabilities=["gpu", "mineru_parse"],
        status="needs_confirmation",
        requires_confirmation=True,
        required_permissions=["remote_worker:workstation2-mineru", "external_network:ssh"],
        budget_limit_sec=1800,
    )

    assert RemoteWorkerConfig.model_validate_json(worker.model_dump_json()).model_dump(mode="json") == worker.model_dump(mode="json")
    assert RemoteWorkerRequest.model_validate_json(request.model_dump_json()).model_dump(mode="json") == request.model_dump(mode="json")
    assert RemoteWorkerAssignment.model_validate_json(assignment.model_dump_json()).model_dump(mode="json") == assignment.model_dump(mode="json")

    with pytest.raises(ValidationError):
        RemoteWorkerConfig(
            worker_id="bad",
            transport="ssh",
            host="workstation2",
            capabilities=["mineru_parse"],
            metadata={"api_key": "secret"},
        )


def test_background_job_schema_roundtrip() -> None:
    state = BackgroundJobState(
        job_id="bg-run",
        project_id="proj-bg",
        run_id="run-bg",
        task_id="retrieve_evidence",
        status=RunStatus.RUNNING,
        budget=BackgroundJobBudget(max_runtime_sec=3600, max_steps=20),
        checkpoints=[
            BackgroundJobCheckpoint(
                checkpoint_id="ckpt-run-bg-001",
                stage="retrieve_evidence",
                cursor={"query_index": 2},
                completed_units=8,
                artifact_refs=["evidence_hits_partial.json"],
            )
        ],
    )

    restored = BackgroundJobState.model_validate_json(state.model_dump_json())
    assert restored.model_dump(mode="json") == state.model_dump(mode="json")
    assert restored.executable is False


def test_background_job_numeric_counters_reject_bool() -> None:
    with pytest.raises(ValueError):
        BackgroundJobCheckpoint(
            checkpoint_id="ckpt-run-bg-001",
            stage="retrieve_evidence",
            completed_units=True,
        )
    with pytest.raises(ValueError):
        BackgroundJobState(
            job_id="bg-run-1",
            run_id="run-1",
            task_id="retrieve_evidence",
            budget=BackgroundJobBudget(max_steps=1),
            consumed_steps=True,
        )
    with pytest.raises(ValueError):
        BackgroundJobState(
            job_id="bg-run-2",
            run_id="run-2",
            task_id="retrieve_evidence",
            budget=BackgroundJobBudget(max_cost_usd=1.0),
            consumed_cost_usd=True,
        )


def test_multi_user_deployment_readiness_schema_roundtrip() -> None:
    report = MultiUserDeploymentReadiness(
        status="ready",
        checks=[
            MultiUserBoundaryCheck(
                name="permission_actor_boundary",
                status="pass",
                message="Strict permission policy requires actor attribution.",
                evidence={"checked_actions": ["train_model", "predict_candidates"]},
            )
        ],
    )

    restored = MultiUserDeploymentReadiness.model_validate_json(report.model_dump_json())
    assert restored.model_dump(mode="json") == report.model_dump(mode="json")
    assert restored.executable is False


def test_literature_acquisition_manifest_schema_roundtrip() -> None:
    item = LiteratureAcquisitionItem(
        source_id="doi_abc123",
        source_type="doi",
        value="10.1000/example",
        status="acquired",
        acquisition_type="pdf",
        strategy="local_mirror",
        output_path="acquired/pdfs/doi_abc123.pdf",
    )
    manifest = LiteratureAcquisitionManifest(
        run_id="run-acq",
        source_count=1,
        acquired_count=1,
        planned_count=0,
        failed_count=0,
        acquired_pdf_dir="acquired/pdfs",
        acquired_dataset_dir="acquired/datasets",
        items=[item],
        created_at="2026-05-29T00:00:00Z",
    )

    restored = LiteratureAcquisitionManifest.model_validate_json(manifest.model_dump_json())
    assert restored.model_dump(mode="json") == manifest.model_dump(mode="json")


def test_corpus_multi_index_schema_roundtrip() -> None:
    index = CorpusMultiIndex(
        run_id="run-index",
        chunk_count=1,
        chunks_jsonl="chunks.jsonl",
        indices={
            "chemical": {"cco": ["paper:table_1"]},
            "property": {"plqy": ["paper:table_1"]},
            "table": {"table_1": ["paper:table_1"]},
            "text": {"oled": ["paper:table_1"]},
        },
        channel_counts={"chemical": 1, "property": 1, "table": 1, "text": 1},
        created_at="2026-05-29T00:00:00Z",
    )

    restored = CorpusMultiIndex.model_validate_json(index.model_dump_json())
    assert restored.model_dump(mode="json") == index.model_dump(mode="json")


def test_dense_retrieval_index_schema_roundtrip() -> None:
    index = DenseRetrievalIndex(
        run_id="run-dense",
        chunk_count=1,
        chunks_jsonl="chunks.jsonl",
        dimension=4,
        embedding_backend="deterministic_hash_embedding",
        vectors={"paper:el_1": [0.5, 0.5, 0.0, 0.0]},
        metadata={"paper:el_1": {"source_id": "paper", "element_type": "paragraph"}},
        created_at="2026-05-29T00:00:00Z",
    )

    restored = DenseRetrievalIndex.model_validate_json(index.model_dump_json())
    assert restored.model_dump(mode="json") == index.model_dump(mode="json")


def test_parsed_document_schema_roundtrip() -> None:
    parsed = ParsedDocument(
        paper_id="paper-1",
        source_path="papers/paper-1.pdf",
        parser_backend="mineru_remote_cli",
        metadata={"title": "Test Paper", "source_hash": "sha256:abc"},
        pages=[{"page": 1, "width": 595.0, "height": 842.0}],
        elements=[
            ParsedDocumentElement(
                element_id="el_0001",
                page=1,
                type="title",
                text="Test Paper",
                markdown="# Test Paper",
                source_hash="sha256:abc",
            )
        ],
        tables=[
            ParsedTable(
                table_id="table_0001",
                caption="Table 1",
                headers=["SMILES", "PLQY"],
                rows=[{"SMILES": "CCO", "PLQY": "0.8"}],
                page=1,
                markdown="| SMILES | PLQY |\n| --- | --- |\n| CCO | 0.8 |",
                source_bbox={"x0": 1.0, "y0": 2.0, "x1": 3.0, "y1": 4.0},
            )
        ],
    )

    restored = ParsedDocument.model_validate_json(parsed.model_dump_json())
    assert restored.model_dump(mode="json") == parsed.model_dump(mode="json")
    assert restored.tables[0].rows[0]["PLQY"] == "0.8"


def test_evidence_retrieval_schema_roundtrip() -> None:
    chunk = CorpusChunk(
        chunk_id="paper-1:table_0001",
        source_id="paper-1",
        paper_id="paper-1",
        page=1,
        element_id="table_0001",
        element_type="table",
        text="SMILES CCO PLQY 0.8",
        markdown="| SMILES | PLQY |\n| --- | --- |\n| CCO | 0.8 |",
        table_id="table_0001",
        retrieval_channels=["bm25", "table"],
        citation_context="paper-1 p.1 table_0001",
    )
    hit = EvidenceHit(
        source_id="paper-1",
        page=1,
        element_id="table_0001",
        element_type="table",
        retrieval_channel="table",
        score=3.2,
        text_or_table_ref="paper-1:table_0001",
        citation_context="paper-1 p.1 table_0001",
        metadata={"chunk_id": chunk.chunk_id},
    )

    assert CorpusChunk.model_validate_json(chunk.model_dump_json()).model_dump(mode="json") == chunk.model_dump(mode="json")
    assert EvidenceHit.model_validate_json(hit.model_dump_json()).model_dump(mode="json") == hit.model_dump(mode="json")


def test_extracted_record_schema_roundtrip() -> None:
    record = ExtractedRecord(
        record_id="rec_000001",
        smiles="CCO",
        properties={"plqy": 0.8, "lambda_em": 520.0},
        source_id="paper-1",
        paper_id="paper-1",
        page=1,
        table_id="table_0001",
        row_index=0,
        evidence_ref="paper-1:table_0001",
        citation_context="paper-1 p.1 table_0001",
        confidence=0.95,
        confidence_factors={"has_smiles": True, "numeric_property_count": 2},
        raw_values={"SMILES": "CCO", "PLQY": "0.8", "lambda_em": "520"},
    )
    report = ExtractionConfidenceReport(
        run_id="run-extract",
        attempted_hit_count=1,
        extracted_record_count=1,
        rejected_record_count=0,
        high_confidence_count=1,
        medium_confidence_count=0,
        low_confidence_count=0,
        confidence_threshold=0.7,
        generated_at="2026-05-29T00:00:00Z",
    )

    assert ExtractedRecord.model_validate_json(record.model_dump_json()).model_dump(mode="json") == record.model_dump(mode="json")
    assert ExtractionConfidenceReport.model_validate_json(report.model_dump_json()).model_dump(mode="json") == report.model_dump(mode="json")


def test_citation_license_report_schema_roundtrip() -> None:
    source = LiteratureSourceProvenance(
        source_id="paper-1",
        paper_id="paper-1",
        title="OLED paper",
        source_path="papers/paper-1.pdf",
        source_hash="sha256:abc",
        parser_backend="mineru_remote_cli",
        citation="Doe et al. OLED paper, 2026",
        doi="10.1000/example",
        license="CC-BY-4.0",
        license_requires_review=False,
        evidence_count=2,
        extracted_record_count=1,
    )
    report = CitationLicenseReport(
        run_id="run-prov",
        source_count=1,
        evidence_count=2,
        extracted_record_count=1,
        unknown_license_count=0,
        sources=[source],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert LiteratureSourceProvenance.model_validate_json(source.model_dump_json()).model_dump(mode="json") == source.model_dump(mode="json")
    assert CitationLicenseReport.model_validate_json(report.model_dump_json()).model_dump(mode="json") == report.model_dump(mode="json")


def test_unit_normalization_report_schema_roundtrip() -> None:
    report = UnitNormalizationReport(
        run_id="run-normalize",
        input_record_count=2,
        normalized_record_count=2,
        conversion_count=1,
        warning_count=0,
        conversions=[
            {
                "record_id": "rec_000001",
                "property_id": "plqy",
                "source_unit": "%",
                "canonical_unit": "fraction",
                "source_value": 80.0,
                "canonical_value": 0.8,
            }
        ],
        warnings=[],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert UnitNormalizationReport.model_validate_json(report.model_dump_json()).model_dump(mode="json") == report.model_dump(mode="json")


def test_extraction_benchmark_report_schema_roundtrip() -> None:
    report = ExtractionBenchmarkReport(
        run_id="run-benchmark",
        retrieval_recall=0.75,
        extraction_precision=0.5,
        conflict_rate=0.25,
        confirmation_workload_count=3,
        trainable_labels_gained=8,
        downstream_model_performance_delta={"plqy.r2": 0.12, "plqy.mae": -0.02},
        metric_statuses={"retrieval_recall": "computed", "extraction_precision": "computed"},
        counts={"evidence_hits": 4, "gold_evidence": 4},
        generated_at="2026-05-29T00:00:00Z",
    )

    assert ExtractionBenchmarkReport.model_validate_json(report.model_dump_json()).model_dump(mode="json") == report.model_dump(mode="json")


def test_data_merge_conflict_schema_roundtrip() -> None:
    merged = MergedRecord(
        merge_id="merge_000001",
        smiles="CCO",
        properties={"plqy": 0.81},
        property_status={"plqy": "merged"},
        source_record_ids=["rec_000001", "rec_000002"],
        source_ids=["paper-a", "paper-b"],
        citations=["paper-a p.1 table_1", "paper-b p.2 table_2"],
        confidence=0.925,
        conflict_ids=[],
        status="merged",
    )
    conflict = ConflictGroup(
        conflict_id="conflict_000001",
        smiles="CCN",
        property_id="plqy",
        min_value=0.2,
        max_value=0.9,
        tolerance=0.05,
        observations=[
            {"record_id": "rec_000003", "value": 0.2, "source_id": "paper-a"},
            {"record_id": "rec_000004", "value": 0.9, "source_id": "paper-c"},
        ],
    )
    report = ConflictReport(
        run_id="run-merge",
        input_record_count=4,
        merged_record_count=2,
        conflict_count=1,
        non_conflicting_record_count=1,
        conflicts=[conflict],
        generated_at="2026-05-29T00:00:00Z",
    )

    assert MergedRecord.model_validate_json(merged.model_dump_json()).model_dump(mode="json") == merged.model_dump(mode="json")
    assert ConflictReport.model_validate_json(report.model_dump_json()).model_dump(mode="json") == report.model_dump(mode="json")


def test_extraction_confirmation_record_schema_roundtrip() -> None:
    record = ExtractionConfirmationRecord(
        run_id="run-confirm",
        dataset_id="confirmed_lit_dataset",
        source_dataset_path="merge/merged_candidate_training_dataset.csv",
        confirmed_dataset_path="confirmed/confirmed_lit_dataset.csv",
        confirmed_by="user",
        confirmed_at="2026-05-29T00:00:00Z",
        record_count=12,
        conflict_count=0,
        unknown_license_count=0,
        source_reports={
            "conflict_report_json": "merge/conflict_report.json",
            "citation_provenance_report_json": "audit/citation_provenance_report.json",
        },
        status="confirmed",
    )

    assert ExtractionConfirmationRecord.model_validate_json(record.model_dump_json()).model_dump(mode="json") == record.model_dump(mode="json")


def test_plan_step_inputs_reject_non_json_safe_values() -> None:
    with pytest.raises(ValidationError):
        PlanStep(name="bad", agent="agent", action="act", inputs={"bad": object()})
