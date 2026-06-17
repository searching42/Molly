from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent.schemas import ModelDiagnosticsReport, ModelPackageReview, ModelingPlanProposal, TargetModelingBrief
from ai4s_agent.storage import ProjectStorage


def _trainability_report(status: str = "READY") -> dict:
    return {
        "overall_status": status,
        "properties": [
            {
                "property_id": "plqy",
                "effective_labels": 120 if status == "READY" else 12,
                "numeric_ratio": 1.0,
                "task_type": "numeric_regression",
                "status": "TRAIN_READY" if status == "READY" else "INSUFFICIENT_LABELS",
                "reason": "TRAIN_READY" if status == "READY" else "INSUFFICIENT_LABELS",
            }
        ],
    }


def _backend_recommendation() -> dict:
    return {
        "selected_backend": "unimol",
        "per_property": [
            {
                "property_id": "plqy",
                "recommended_backend": "unimol",
                "recommendation": "train_unimol",
                "reason": "3d_relevance_or_user_intent",
                "trainability_status": "TRAIN_READY",
                "three_d_relevance": "high",
                "baseline_metrics": {"r2": -0.15, "mae": 0.18, "rmse": 0.22},
            }
        ],
        "mixed_backend_warning": False,
        "warnings": [],
    }


def test_modeling_agent_recommends_backend_design_and_metric_retry() -> None:
    proposal = ModelingAgent().propose_modeling_plan(
        run_id="run-modeling",
        goal="Train a reliable 3D model for PLQY.",
        trainability_report=_trainability_report(),
        backend_recommendation=_backend_recommendation(),
        model_metrics={
            "properties": [
                {
                    "property_id": "plqy",
                    "metrics": {"r2": -0.1, "mae": 0.2, "rmse": 0.3},
                }
            ]
        },
    )

    assert proposal.run_id == "run-modeling"
    assert proposal.status == "needs_confirmation"
    assert proposal.executable is False
    assert proposal.backend_recommendations[0].backend == "unimol"
    assert proposal.experiment_design.backend == "unimol"
    assert proposal.experiment_design.split_strategy == "scaffold_split_then_random_fallback"
    assert proposal.metric_interpretations[0].status == "weak"
    assert proposal.retry_proposals
    assert {item.action for item in proposal.retry_proposals} >= {"adjust_split", "switch_backend_or_features"}

    restored = ModelingPlanProposal.model_validate_json(proposal.model_dump_json())
    assert restored.model_dump(mode="json") == proposal.model_dump(mode="json")


def test_modeling_agent_blocks_training_when_trainability_is_blocked() -> None:
    proposal = ModelingAgent().propose_modeling_plan(
        run_id="run-modeling-blocked",
        goal="Train a model for PLQY.",
        trainability_report=_trainability_report(status="BLOCKED"),
    )

    assert proposal.status == "needs_clarification"
    assert proposal.experiment_design.backend == "none"
    assert proposal.backend_recommendations[0].backend == "none"
    assert proposal.questions
    assert proposal.questions[0].question_id == "q_modeling_data_readiness"
    assert any(item.action == "request_more_data" for item in proposal.retry_proposals)


def test_modeling_agent_ignores_non_finite_and_boolean_metrics() -> None:
    proposal = ModelingAgent().propose_modeling_plan(
        run_id="run-modeling-bad-metrics",
        goal="Review model metrics.",
        trainability_report=_trainability_report(),
        model_metrics={
            "properties": [
                {
                    "property_id": "plqy",
                    "metrics": {"r2": "nan", "rmse": "inf", "mae": True},
                }
            ]
        },
    )

    interpretation = proposal.metric_interpretations[0]
    assert interpretation.metrics == {}
    assert interpretation.status == "not_evaluated"
    assert interpretation.decision == "continue"


def test_modeling_agent_writes_modeling_plan_artifact(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = ModelingAgent()
    proposal = agent.propose_modeling_plan(
        run_id="run-modeling-artifact",
        goal="Train baseline model.",
        trainability_report=_trainability_report(),
    )

    json_path, md_path = agent.write_proposal(storage, "proj-modeling", "run-modeling-artifact", proposal)

    assert json_path.name == "modeling_plan_proposal.json"
    assert md_path.name == "modeling_plan_proposal.md"
    assert proposal.experiment_design.backend == "baseline"
    assert "gate_3_train_config" in proposal.experiment_design.required_gates
    registry = storage.read_artifact_registry("proj-modeling", "run-modeling-artifact")
    assert registry["modeling_plan_proposal_json"] == "modeling_plan_proposal.json"
    assert registry["modeling_plan_proposal_md"] == "modeling_plan_proposal.md"


def test_modeling_agent_prepares_oled_plqy_target_modeling_brief() -> None:
    brief = ModelingAgent().prepare_target_modeling_brief(
        run_id="run-target-brief",
        goal="Screen OLED emitters for high PLQY in common solvents.",
        property_id="plqy",
        trainability_report=_trainability_report(),
        project_memory={
            "default_models": {
                "plqy_scalar": "solvent_pca64_seed42",
                "plqy_high_screening": "manual_weight3_ensemble",
            },
            "previous_diagnostics": [
                "PLQY solvent-aware models improve overall R2 but still compress high-QY values."
            ],
        },
        previous_diagnostics=[
            {
                "model_id": "manual_weight3_ensemble",
                "metrics": {"mae": 0.1741, "r2": 0.3754, "pearson": 0.6496},
                "high_qy_bias": -0.216,
            }
        ],
    )

    assert brief.run_id == "run-target-brief"
    assert brief.property_id == "plqy"
    assert brief.domain == "oled"
    assert brief.recommended_backend == "unimol_with_solvent_features"
    assert brief.split_strategy == "scaffold_split_grouped_by_canonical_smiles"
    assert "bounded_target" in brief.risk_flags
    assert "solvent_context_dependence" in brief.risk_flags
    assert "high_value_compression_risk" in brief.risk_flags
    assert "preserve_solvent_conditioned_rows" in brief.preprocessing_steps
    assert "add_solvent_descriptors_or_embeddings" in brief.preprocessing_steps
    assert brief.target_transform == "bounded_logit_or_calibrated_regression"
    assert brief.evidence_sources == [
        "project_memory",
        "previous_run_diagnostics",
        "trainability_report",
        "built_in_domain_rules",
    ]
    assert brief.external_search_policy == "not_used"
    assert brief.acceptance_criteria["review_high_value_bucket_bias"] is True

    restored = TargetModelingBrief.model_validate_json(brief.model_dump_json())
    assert restored.model_dump(mode="json") == brief.model_dump(mode="json")


def test_modeling_agent_brief_includes_registry_model_selection_and_missing_inputs() -> None:
    brief = ModelingAgent().prepare_target_modeling_brief(
        run_id="run-target-model-selection",
        goal="Prioritize OLED candidates with high PLQY.",
        property_id="quantum_yield",
        trainability_report=_trainability_report(),
        available_inputs={"canonical_smiles"},
    )

    assert brief.property_id == "quantum_yield"
    assert brief.model_selection is not None
    assert brief.model_selection.selected_model_id == "plqy_manual_weight3_ensemble"
    assert brief.model_selection.normalized_property_id == "plqy"
    assert brief.model_selection.missing_required_inputs == ["solvent"]
    assert "missing_required_input:solvent" in brief.model_selection.warnings
    assert brief.status == "needs_clarification"
    assert any(question.question_id == "q_target_plqy_missing_inputs" for question in brief.questions)


def test_modeling_agent_diagnoses_high_plqy_compression_with_gated_rerun() -> None:
    agent = ModelingAgent()
    brief = agent.prepare_target_modeling_brief(
        run_id="run-plqy-diagnostics",
        goal="Prioritize OLED candidates with high PLQY.",
        property_id="plqy",
        trainability_report=_trainability_report(),
    )

    report = agent.diagnose_model(
        run_id="run-plqy-diagnostics",
        goal="Prioritize OLED candidates with high PLQY.",
        property_id="plqy",
        model_id="manual_weight3_ensemble",
        metrics={"mae": 0.1741, "r2": 0.3754, "pearson": 0.6496},
        baseline_metrics={"mean": {"mae": 0.246, "r2": 0.0}},
        distribution_diagnostics={
            "true_p95": 0.887,
            "pred_p95": 0.827,
            "high_qy_threshold": 0.7,
            "high_qy_mae": 0.233,
            "high_qy_bias": -0.216,
        },
        modeling_brief=brief,
    )

    assert report.run_id == "run-plqy-diagnostics"
    assert report.property_id == "plqy"
    assert report.model_id == "manual_weight3_ensemble"
    assert report.readiness == "promising"
    assert report.decision == "rerun_recommended"
    assert "high_value_underprediction" in report.risk_flags
    assert report.baseline_comparison["mean_mae_improvement"] == 0.0719
    assert report.rerun_proposal is not None
    assert report.rerun_proposal.property_id == "plqy"
    assert report.rerun_proposal.requires_user_approval is True
    assert "gate_3_train_config" in report.rerun_proposal.required_approvals
    assert {
        "posthoc_calibration",
        "high_value_weighting_or_two_stage_model",
        "seed_ensemble",
    }.issubset(set(report.rerun_proposal.candidate_changes))

    restored = ModelDiagnosticsReport.model_validate_json(report.model_dump_json())
    assert restored.model_dump(mode="json") == report.model_dump(mode="json")


def test_modeling_agent_reviews_model_package_before_promotion() -> None:
    agent = ModelingAgent()
    diagnostics = agent.diagnose_model(
        run_id="run-package-review",
        goal="Prioritize OLED candidates with high PLQY.",
        property_id="plqy",
        model_id="manual_weight3_ensemble",
        metrics={"mae": 0.1741, "r2": 0.3754, "pearson": 0.6496},
        distribution_diagnostics={"high_qy_bias": -0.216, "high_qy_threshold": 0.7},
        modeling_brief=agent.prepare_target_modeling_brief(
            run_id="run-package-review",
            goal="Prioritize OLED candidates with high PLQY.",
            property_id="plqy",
            trainability_report=_trainability_report(),
        ),
    )

    review = agent.review_model_package(
        run_id="run-package-review",
        goal="Prioritize OLED candidates with high PLQY.",
        model_manifest={
            "model_id": "manual_weight3_ensemble",
            "property_id": "plqy",
            "model_backend": "unimol_with_manual_solvent",
            "metrics": {"mae": 0.1741, "r2": 0.3754},
            "split_strategy": "scaffold_split_grouped_by_canonical_smiles",
            "feature_type": "unimol+manual_solvent",
        },
        domain_model_manifest={
            "domain": "oled",
            "use_case": "high_plqy_screening",
            "feature_requirements": ["canonical_smiles", "solvent"],
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
            "applicability": {"objective_type": "regression"},
            "limitations": ["high PLQY compression remains monitored"],
        },
        diagnostics_report=diagnostics,
    )

    assert review.run_id == "run-package-review"
    assert review.model_id == "manual_weight3_ensemble"
    assert review.decision == "rerun_recommended"
    assert review.status == "needs_confirmation"
    assert "high_value_underprediction" in review.risk_flags
    assert "gate_3_train_config" in review.required_gates
    assert "promote_asset" not in review.required_permissions
    assert review.rerun_proposal is not None
    assert review.promotion_draft == {}
    assert review.memory_updates[0]["kind"] == "modeling_lesson"

    restored = ModelPackageReview.model_validate_json(review.model_dump_json())
    assert restored.model_dump(mode="json") == review.model_dump(mode="json")


def test_modeling_agent_recommends_promote_candidate_for_strong_clean_package(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = ModelingAgent()
    review = agent.review_model_package(
        run_id="run-package-promote",
        goal="Predict OLED emission wavelength.",
        model_manifest={
            "model_id": "emission_unimol_v001",
            "property_id": "emission_max_nm",
            "model_backend": "unimol",
            "metrics": {"mae": 28.5, "r2": 0.84, "pearson": 0.91},
            "split_strategy": "scaffold_split_grouped_by_canonical_smiles",
            "feature_type": "unimol_3d",
        },
        domain_model_manifest={
            "domain": "oled",
            "use_case": "scalar_prediction",
            "feature_requirements": ["canonical_smiles"],
            "input_columns": {"canonical_smiles": "SMILES"},
            "applicability": {"target_range_nm": "visible"},
            "limitations": [],
        },
    )

    assert review.decision == "promote_candidate"
    assert review.status == "needs_confirmation"
    assert review.required_gates == []
    assert review.required_permissions == ["promote_asset"]
    assert review.promotion_draft["model_id"] == "emission_unimol_v001"
    assert review.promotion_draft["input_columns"] == {"canonical_smiles": "SMILES"}

    json_path, md_path = agent.write_model_package_review(storage, "proj-modeling", "run-package-promote", review)
    assert json_path.name == "model_package_review_emission_max_nm.json"
    assert md_path.name == "model_package_review_emission_max_nm.md"
    registry = storage.read_artifact_registry("proj-modeling", "run-package-promote")
    assert registry["model_package_review_emission_max_nm_json"] == "model_package_review_emission_max_nm.json"
    assert registry["model_package_review_emission_max_nm_md"] == "model_package_review_emission_max_nm.md"


def test_modeling_agent_blocks_promotion_when_package_lacks_input_contract() -> None:
    review = ModelingAgent().review_model_package(
        run_id="run-package-incomplete",
        goal="Predict OLED emission wavelength.",
        model_manifest={
            "model_id": "emission_unimol_v001",
            "property_id": "emission_max_nm",
            "model_backend": "unimol",
            "metrics": {"mae": 28.5, "r2": 0.84},
        },
        domain_model_manifest={"domain": "oled", "use_case": "scalar_prediction"},
    )

    assert review.decision == "blocked"
    assert review.status == "blocked"
    assert "missing_feature_requirements" in review.risk_flags
    assert "missing_input_columns" in review.risk_flags
    assert review.required_permissions == []
    assert review.promotion_draft == {}


def test_modeling_agent_uses_domain_manifest_metrics_for_package_review() -> None:
    review = ModelingAgent().review_model_package(
        run_id="run-package-domain-metrics",
        goal="Predict OLED emission wavelength.",
        model_manifest={
            "model_id": "emission_unimol_v001",
            "property_id": "emission_max_nm",
            "model_backend": "unimol",
        },
        domain_model_manifest={
            "domain": "oled",
            "use_case": "scalar_prediction",
            "metrics": {"mae": 28.5, "r2": 0.84},
            "feature_requirements": ["canonical_smiles"],
            "input_columns": {"canonical_smiles": "SMILES"},
        },
    )

    assert review.decision == "promote_candidate"
    assert review.metrics["r2"] == 0.84


def test_modeling_agent_blocks_package_review_when_diagnostics_do_not_match_manifest() -> None:
    review = ModelingAgent().review_model_package(
        run_id="run-package-mismatch",
        goal="Predict OLED PLQY.",
        model_manifest={
            "model_id": "weak_plqy_v001",
            "property_id": "plqy",
            "model_backend": "unimol",
            "metrics": {"r2": 0.1},
        },
        domain_model_manifest={
            "domain": "oled",
            "use_case": "scalar_prediction",
            "feature_requirements": ["canonical_smiles", "solvent"],
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        },
        diagnostics_report={
            "run_id": "run-package-mismatch",
            "property_id": "emission_max_nm",
            "model_id": "strong_emission_v001",
            "readiness": "strong",
            "decision": "accept",
            "metrics": {"r2": 0.84, "mae": 28.5},
        },
    )

    assert review.decision == "blocked"
    assert "diagnostics_model_id_mismatch" in review.risk_flags
    assert "diagnostics_property_id_mismatch" in review.risk_flags
    assert review.metrics == {"r2": 0.1}
    assert review.promotion_draft == {}


def test_modeling_agent_applies_target_acceptance_criteria_before_promotion() -> None:
    review = ModelingAgent().review_model_package(
        run_id="run-package-plqy-criteria",
        goal="Predict OLED PLQY.",
        model_manifest={
            "model_id": "plqy_v001",
            "property_id": "plqy",
            "model_backend": "unimol",
            "metrics": {"r2": 0.6, "mae": 0.35},
        },
        domain_model_manifest={
            "domain": "oled",
            "use_case": "scalar_prediction",
            "feature_requirements": ["canonical_smiles", "solvent"],
            "input_columns": {"canonical_smiles": "SMILES", "solvent": "solvent"},
        },
    )

    assert review.decision == "rerun_recommended"
    assert review.required_gates == ["gate_3_train_config"]
    assert "target_acceptance_criteria_failed:max_mae" in review.risk_flags
    assert review.promotion_draft == {}


def test_modeling_agent_writes_target_brief_and_diagnostics_artifacts(tmp_path) -> None:
    storage = ProjectStorage(tmp_path)
    agent = ModelingAgent()
    brief = agent.prepare_target_modeling_brief(
        run_id="run-modeling-artifacts",
        goal="Train OLED PLQY model.",
        property_id="plqy",
        trainability_report=_trainability_report(),
    )
    report = agent.diagnose_model(
        run_id="run-modeling-artifacts",
        goal="Train OLED PLQY model.",
        property_id="plqy",
        model_id="solvent_pca64_seed42",
        metrics={"mae": 0.1737, "r2": 0.3883, "pearson": 0.6446},
        distribution_diagnostics={"high_qy_bias": -0.262, "high_qy_threshold": 0.7},
        modeling_brief=brief,
    )

    brief_json, brief_md = agent.write_target_modeling_brief(storage, "proj-modeling", "run-modeling-artifacts", brief)
    report_json, report_md = agent.write_model_diagnostics_report(
        storage,
        "proj-modeling",
        "run-modeling-artifacts",
        report,
    )

    assert brief_json.name == "target_modeling_brief_plqy.json"
    assert brief_md.name == "target_modeling_brief_plqy.md"
    assert report_json.name == "model_diagnostics_report_plqy.json"
    assert report_md.name == "model_diagnostics_report_plqy.md"
    registry = storage.read_artifact_registry("proj-modeling", "run-modeling-artifacts")
    assert registry["target_modeling_brief_plqy_json"] == "target_modeling_brief_plqy.json"
    assert registry["target_modeling_brief_plqy_md"] == "target_modeling_brief_plqy.md"
    assert registry["model_diagnostics_report_plqy_json"] == "model_diagnostics_report_plqy.json"
    assert registry["model_diagnostics_report_plqy_md"] == "model_diagnostics_report_plqy.md"
