from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent.schemas import ModelDiagnosticsReport, ModelingPlanProposal, TargetModelingBrief
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
