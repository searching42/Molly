from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.domains.oled import OLED_MODEL_REGISTRY
from ai4s_agent.schemas import (
    GateName,
    ModelDiagnosticsReport,
    ModelPackageReview,
    ModelingBackendRecommendation,
    ModelingExperimentDesign,
    ModelingMetricInterpretation,
    ModelingPlanProposal,
    ModelingRetryProposal,
    PlanQuestion,
    RerunProposal,
    TargetModelingBrief,
)
from ai4s_agent.storage import ProjectStorage


class ModelingAgent:
    """Dry-run modeling advisor for backend choice, experiment design, and metric review."""

    def prepare_target_modeling_brief(
        self,
        *,
        run_id: str,
        goal: str,
        property_id: str,
        trainability_report: dict[str, Any] | None = None,
        project_memory: dict[str, Any] | None = None,
        previous_diagnostics: list[dict[str, Any]] | None = None,
        allow_external_search: bool = False,
        available_inputs: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> TargetModelingBrief:
        clean_goal = str(goal or "").strip()
        clean_property = str(property_id or "default").strip() or "default"
        trainability = trainability_report if isinstance(trainability_report, dict) else {}
        memory = project_memory if isinstance(project_memory, dict) else {}
        diagnostics = previous_diagnostics if isinstance(previous_diagnostics, list) else []
        profile = self._target_rule_profile(clean_goal, clean_property)
        trainability_item = self._trainability_item(trainability, clean_property)
        status = "ready_for_confirmation"
        questions: list[PlanQuestion] = []
        if str(trainability_item.get("status") or "").upper() in {"INSUFFICIENT_LABELS", "INVALID_LABELS"}:
            status = "needs_clarification"
            questions.append(
                PlanQuestion(
                    question_id=f"q_target_{clean_property}_labels",
                    prompt=f"Should `{clean_property}` be delayed, reduced, or trained as a low-confidence baseline only?",
                    reason="The target does not have enough valid labels for confident model training.",
                    choices=["delay_target", "train_baseline_only", "request_more_data"],
                    blocks_execution=True,
                )
            )

        evidence_sources: list[str] = []
        if memory:
            evidence_sources.append("project_memory")
        if diagnostics:
            evidence_sources.append("previous_run_diagnostics")
        if trainability:
            evidence_sources.append("trainability_report")
        evidence_sources.append("built_in_domain_rules")
        if allow_external_search:
            evidence_sources.append("user_approved_external_search")

        model_selection = self._select_domain_model(
            goal=clean_goal,
            property_id=clean_property,
            profile=profile,
            available_inputs=available_inputs,
        )
        if model_selection and model_selection.requires_user_input:
            status = "needs_clarification"
            questions.append(
                PlanQuestion(
                    question_id=f"q_target_{model_selection.normalized_property_id}_missing_inputs",
                    prompt=f"Provide required model inputs for `{model_selection.normalized_property_id}`: {', '.join(model_selection.missing_required_inputs)}.",
                    reason="The selected reviewed model requires inputs that are not present in the current request context.",
                    choices=["provide_missing_inputs", "use_default_context", "choose_different_model"],
                    blocks_execution=True,
                )
            )

        dataset_context = {
            "trainability_status": str(trainability_item.get("status") or ""),
            "effective_labels": self._safe_int(trainability_item.get("effective_labels")),
            "previous_diagnostics_count": len(diagnostics),
            "has_project_memory": bool(memory),
        }
        assumptions = [
            "Brief is advisory and must be reviewed before expensive training.",
            "Execution adapters remain the authority for actual preprocessing and training.",
        ]
        if profile["domain"] == "oled":
            assumptions.append("OLED-first defaults are used, while core schemas remain target-agnostic.")

        return TargetModelingBrief(
            run_id=str(run_id or "").strip(),
            goal=clean_goal,
            property_id=clean_property,
            domain=profile["domain"],
            status=status,
            evidence_sources=evidence_sources,
            external_search_policy="allowed_with_user_approval" if allow_external_search else "not_used",
            risk_flags=profile["risk_flags"],
            preprocessing_steps=profile["preprocessing_steps"],
            split_strategy=profile["split_strategy"],
            target_transform=profile["target_transform"],
            recommended_backend=profile["recommended_backend"],
            hyperparameters=profile["hyperparameters"],
            acceptance_criteria=profile["acceptance_criteria"],
            dataset_context=dataset_context,
            model_selection=model_selection,
            assumptions=assumptions,
            questions=questions,
            executable=False,
        )

    def diagnose_model(
        self,
        *,
        run_id: str,
        property_id: str,
        metrics: dict[str, Any],
        goal: str = "",
        model_id: str = "",
        baseline_metrics: dict[str, Any] | None = None,
        distribution_diagnostics: dict[str, Any] | None = None,
        fold_metrics: dict[str, Any] | None = None,
        modeling_brief: TargetModelingBrief | None = None,
    ) -> ModelDiagnosticsReport:
        clean_property = str(property_id or "default").strip() or "default"
        numeric_metrics = self._numeric_metrics(metrics)
        readiness = self._diagnostic_readiness(numeric_metrics)
        decision = self._diagnostic_decision(readiness)
        distribution = distribution_diagnostics if isinstance(distribution_diagnostics, dict) else {}
        folds = fold_metrics if isinstance(fold_metrics, dict) else {}
        baseline_comparison = self._baseline_comparison(numeric_metrics, baseline_metrics or {})
        risk_flags: list[str] = []
        messages: list[str] = []

        if readiness == "weak":
            risk_flags.append("weak_generalization")
            messages.append("Model metrics are weak for downstream screening.")
        if readiness == "not_evaluated":
            messages.append("No usable numeric metrics were supplied.")

        high_value_bias = self._safe_float(distribution.get("high_qy_bias"))
        max_bias = 0.2
        if modeling_brief and isinstance(modeling_brief.acceptance_criteria.get("max_abs_high_value_bias"), int | float):
            max_bias = float(modeling_brief.acceptance_criteria["max_abs_high_value_bias"])
        high_value_goal = self._is_high_value_goal(goal) or "high_value_compression_risk" in (
            modeling_brief.risk_flags if modeling_brief else []
        )
        if high_value_goal and high_value_bias is not None and high_value_bias < -max_bias:
            risk_flags.append("high_value_underprediction")
            messages.append("High-value samples are systematically underpredicted; use caution for high-PLQY screening.")
            decision = "rerun_recommended"

        true_p95 = self._safe_float(distribution.get("true_p95"))
        pred_p95 = self._safe_float(distribution.get("pred_p95"))
        if true_p95 and pred_p95 is not None and pred_p95 < true_p95 * 0.9:
            risk_flags.append("prediction_range_compression")
            messages.append("Prediction upper tail is compressed relative to observed targets.")
            if decision == "accept":
                decision = "low_confidence_accept"

        rerun_proposal = None
        if decision in {"rerun_recommended", "blocked"}:
            rerun_proposal = self._build_rerun_proposal(
                property_id=clean_property,
                readiness=readiness,
                risk_flags=risk_flags,
                modeling_brief=modeling_brief,
            )

        return ModelDiagnosticsReport(
            run_id=str(run_id or "").strip(),
            goal=str(goal or "").strip(),
            property_id=clean_property,
            model_id=str(model_id or "").strip(),
            readiness=readiness,
            decision=decision,
            metrics=numeric_metrics,
            baseline_comparison=baseline_comparison,
            distribution_diagnostics=distribution,
            fold_diagnostics=folds,
            risk_flags=self._dedup_strings(risk_flags),
            messages=messages,
            rerun_proposal=rerun_proposal,
            executable=False,
        )

    def review_model_package(
        self,
        *,
        run_id: str,
        goal: str = "",
        model_manifest: dict[str, Any] | None = None,
        domain_model_manifest: dict[str, Any] | None = None,
        diagnostics_report: ModelDiagnosticsReport | dict[str, Any] | None = None,
    ) -> ModelPackageReview:
        model_manifest = model_manifest if isinstance(model_manifest, dict) else {}
        domain_manifest = domain_model_manifest if isinstance(domain_model_manifest, dict) else {}
        diagnostics = self._coerce_diagnostics_report(diagnostics_report)
        metrics = self._numeric_metrics(domain_manifest.get("metrics", {}))
        metrics.update(self._numeric_metrics(model_manifest.get("metrics", {})))
        if diagnostics is not None and diagnostics.metrics:
            metrics.update(diagnostics.metrics)

        clean_run_id = str(run_id or "").strip()
        clean_goal = str(goal or "").strip()
        model_id = self._first_nonempty(
            model_manifest.get("model_id"),
            domain_manifest.get("model_id"),
            diagnostics.model_id if diagnostics else "",
        )
        property_id = self._first_nonempty(
            model_manifest.get("property_id"),
            domain_manifest.get("property_id"),
            diagnostics.property_id if diagnostics else "",
        )
        backend = self._first_nonempty(
            model_manifest.get("model_backend"),
            model_manifest.get("backend"),
            domain_manifest.get("model_backend"),
            domain_manifest.get("backend"),
        )
        domain = self._first_nonempty(domain_manifest.get("domain"), model_manifest.get("domain"), "general")
        use_case = self._first_nonempty(domain_manifest.get("use_case"), model_manifest.get("use_case"), "scalar_prediction")
        applicability = self._json_object(domain_manifest.get("applicability"))
        feature_requirements = self._string_list(domain_manifest.get("feature_requirements"))
        input_columns = self._string_map(domain_manifest.get("input_columns"))
        limitations = self._string_list(domain_manifest.get("limitations"))

        risk_flags: list[str] = []
        rationale: list[str] = []
        required_gates: list[str] = []
        required_permissions: list[str] = []
        rerun_proposal: RerunProposal | None = None

        missing = [
            label
            for label, value in (
                ("model_id", model_id),
                ("property_id", property_id),
                ("backend", backend),
            )
            if not value
        ]
        if missing:
            risk_flags.extend(f"missing_package_field:{item}" for item in missing)
            rationale.append("Model package is missing required manifest metadata.")

        if not metrics:
            risk_flags.append("missing_numeric_metrics")
            rationale.append("No numeric validation metrics are available for promotion review.")

        if not feature_requirements:
            risk_flags.append("missing_feature_requirements")
            rationale.append("Domain manifest does not declare required model inputs.")
        if not input_columns:
            risk_flags.append("missing_input_columns")
            rationale.append("Domain manifest does not map required model inputs to dataset columns.")

        readiness = self._diagnostic_readiness(metrics)
        if diagnostics is not None:
            risk_flags.extend(diagnostics.risk_flags)
            if diagnostics.rerun_proposal is not None:
                rerun_proposal = diagnostics.rerun_proposal
            if diagnostics.decision in {"blocked", "rerun_recommended"}:
                rationale.append(f"Model diagnostics decision is `{diagnostics.decision}`.")
                readiness = "weak" if diagnostics.decision == "rerun_recommended" else "blocked"
            elif diagnostics.decision == "low_confidence_accept":
                risk_flags.append("low_confidence_diagnostics_accept")
                rationale.append("Model diagnostics only supports low-confidence acceptance.")

        critical_risks = {
            "high_value_underprediction",
            "prediction_range_compression",
            "weak_generalization",
            "low_confidence_diagnostics_accept",
        }
        has_critical_risk = any(flag in critical_risks for flag in risk_flags)
        package_incomplete = not feature_requirements or not input_columns
        blocked = bool(missing) or package_incomplete

        if blocked:
            decision = "blocked"
            status = "blocked"
            rationale.append("Keep the package out of reusable assets until the manifest is repaired.")
        elif readiness == "strong" and not has_critical_risk:
            decision = "promote_candidate"
            status = "needs_confirmation"
            required_permissions.append("promote_asset")
            rationale.append("Validation metrics are strong and no blocking package risks were detected.")
        elif readiness in {"weak", "blocked"} or has_critical_risk:
            decision = "rerun_recommended"
            status = "needs_confirmation"
            required_gates.append(GateName.TRAIN_CONFIG.value)
            rationale.append("Training should be reviewed for rerun before any model promotion.")
            if rerun_proposal is None:
                rerun_proposal = self._build_rerun_proposal(
                    property_id=property_id or "default",
                    readiness="weak",
                    risk_flags=risk_flags,
                    modeling_brief=None,
                )
        else:
            decision = "memory_only"
            status = "memory_only"
            rationale.append("Model package is useful as training memory, but is not strong enough for reusable prediction.")

        promotion_draft = {}
        if decision == "promote_candidate":
            promotion_draft = {
                "model_id": model_id,
                "domain": domain,
                "property_id": property_id,
                "use_case": use_case,
                "backend": backend,
                "metrics": metrics,
                "applicability": applicability,
                "feature_requirements": feature_requirements,
                "input_columns": input_columns,
                "limitations": limitations,
            }

        memory_updates = [
            {
                "kind": "modeling_lesson",
                "model_id": model_id,
                "property_id": property_id,
                "decision": decision,
                "metrics": metrics,
                "risk_flags": self._dedup_strings(risk_flags),
                "rationale": self._dedup_strings(rationale),
            }
        ]

        return ModelPackageReview(
            run_id=clean_run_id,
            goal=clean_goal,
            model_id=model_id or "unknown_model",
            domain=domain,
            property_id=property_id or "unknown_property",
            use_case=use_case,
            backend=backend or "unknown_backend",
            status=status,
            decision=decision,
            metrics=metrics,
            applicability=applicability,
            feature_requirements=feature_requirements,
            input_columns=input_columns,
            limitations=limitations,
            risk_flags=self._dedup_strings(risk_flags),
            rationale=self._dedup_strings(rationale),
            required_gates=self._dedup_strings(required_gates),
            required_permissions=self._dedup_strings(required_permissions),
            promotion_draft=promotion_draft,
            rerun_proposal=rerun_proposal,
            memory_updates=memory_updates,
            executable=False,
        )

    def propose_modeling_plan(
        self,
        *,
        run_id: str,
        goal: str,
        trainability_report: dict[str, Any] | None = None,
        backend_recommendation: dict[str, Any] | None = None,
        model_metrics: dict[str, Any] | None = None,
    ) -> ModelingPlanProposal:
        clean_goal = str(goal or "").strip()
        trainability = trainability_report if isinstance(trainability_report, dict) else {}
        backend_summary = backend_recommendation if isinstance(backend_recommendation, dict) else {}
        metrics = model_metrics if isinstance(model_metrics, dict) else {}

        recommendations = self._backend_recommendations(clean_goal, trainability, backend_summary)
        design = self._experiment_design(recommendations, trainability, clean_goal)
        interpretations = self._metric_interpretations(metrics, backend_summary)
        retry_proposals = self._retry_proposals(trainability, interpretations, design)
        questions = self._questions(trainability, design)
        status = "needs_clarification" if any(question.blocks_execution for question in questions) else "needs_confirmation"

        return ModelingPlanProposal(
            run_id=str(run_id or "").strip(),
            goal=clean_goal,
            status=status,
            backend_recommendations=recommendations,
            experiment_design=design,
            metric_interpretations=interpretations,
            retry_proposals=retry_proposals,
            assumptions=[
                "ModelingAgent does not train models or mutate run state.",
                "High-risk training still requires explicit gate approval.",
                "Metric interpretation is advisory and should be reviewed before retrying expensive jobs.",
            ],
            questions=questions,
            executable=False,
        )

    def write_proposal(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        proposal: ModelingPlanProposal,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "modeling_plan_proposal.json", proposal.model_dump(mode="json"))
        md_path = run_dir / "modeling_plan_proposal.md"
        md_path.write_text(self._render_markdown(proposal), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "modeling_plan_proposal_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "modeling_plan_proposal_md", md_path.name)
        return json_path, md_path

    def write_target_modeling_brief(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        brief: TargetModelingBrief,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        safe_property = self._safe_property_stem(brief.property_id)
        json_name = f"target_modeling_brief_{safe_property}.json"
        md_name = f"target_modeling_brief_{safe_property}.md"
        json_path = write_json(run_dir / json_name, brief.model_dump(mode="json"))
        md_path = run_dir / md_name
        md_path.write_text(self._render_target_brief_markdown(brief), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, f"target_modeling_brief_{safe_property}_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, f"target_modeling_brief_{safe_property}_md", md_path.name)
        return json_path, md_path

    def write_model_diagnostics_report(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        report: ModelDiagnosticsReport,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        safe_property = self._safe_property_stem(report.property_id)
        json_name = f"model_diagnostics_report_{safe_property}.json"
        md_name = f"model_diagnostics_report_{safe_property}.md"
        json_path = write_json(run_dir / json_name, report.model_dump(mode="json"))
        md_path = run_dir / md_name
        md_path.write_text(self._render_diagnostics_markdown(report), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, f"model_diagnostics_report_{safe_property}_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, f"model_diagnostics_report_{safe_property}_md", md_path.name)
        return json_path, md_path

    def write_model_package_review(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        review: ModelPackageReview,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        safe_property = self._safe_property_stem(review.property_id)
        json_name = f"model_package_review_{safe_property}.json"
        md_name = f"model_package_review_{safe_property}.md"
        json_path = write_json(run_dir / json_name, review.model_dump(mode="json"))
        md_path = run_dir / md_name
        md_path.write_text(self._render_model_package_review_markdown(review), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, f"model_package_review_{safe_property}_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, f"model_package_review_{safe_property}_md", md_path.name)
        return json_path, md_path

    def _backend_recommendations(
        self,
        goal: str,
        trainability: dict[str, Any],
        backend_summary: dict[str, Any],
    ) -> list[ModelingBackendRecommendation]:
        properties = self._trainability_properties(trainability)
        per_property = backend_summary.get("per_property", [])
        backend_by_property = {
            str(item.get("property_id") or ""): item
            for item in per_property
            if isinstance(item, dict) and str(item.get("property_id") or "").strip()
        }
        if not properties and backend_by_property:
            properties = [{"property_id": key, "status": "TRAIN_READY", "effective_labels": 0} for key in backend_by_property]
        if not properties:
            properties = [{"property_id": "default", "status": "INSUFFICIENT_LABELS", "effective_labels": 0}]

        recommendations: list[ModelingBackendRecommendation] = []
        for prop in properties:
            property_id = str(prop.get("property_id") or "default").strip()
            status = str(prop.get("status") or "").upper()
            item = backend_by_property.get(property_id, {})
            backend = str(item.get("recommended_backend") or backend_summary.get("selected_backend") or "").strip()
            if not backend:
                backend = self._infer_backend(goal, property_id, status)
            if status in {"INSUFFICIENT_LABELS", "INVALID_LABELS", "UNSUPPORTED_TASK_TYPE"}:
                backend = "none"
            confidence = self._backend_confidence(backend, prop, item)
            recommendations.append(
                ModelingBackendRecommendation(
                    property_id=property_id,
                    backend=backend,
                    confidence=confidence,
                    reason=str(item.get("reason") or self._backend_reason(backend, status)),
                    requirements=self._backend_requirements(backend),
                    risk_flags=self._backend_risk_flags(backend, status, backend_summary),
                )
            )
        return recommendations

    @staticmethod
    def _experiment_design(
        recommendations: list[ModelingBackendRecommendation],
        trainability: dict[str, Any],
        goal: str,
    ) -> ModelingExperimentDesign:
        trainable = [item for item in recommendations if item.backend != "none"]
        backend = "none"
        if trainable:
            unimol_count = sum(1 for item in trainable if item.backend == "unimol")
            baseline_count = sum(1 for item in trainable if item.backend in {"baseline", "random_forest", "xgboost"})
            backend = "unimol" if unimol_count >= baseline_count and unimol_count > 0 else trainable[0].backend
        required_gates = [GateName.TRAIN_CONFIG.value] if backend != "none" else []
        if backend == "none":
            split_strategy = "not_applicable"
            validation_strategy = "review_data_before_training"
        else:
            split_strategy = "scaffold_split_then_random_fallback"
            validation_strategy = "holdout_with_baseline_comparison"
        budget_notes = ["Use dry-run planning until the user approves training."]
        if backend == "unimol":
            budget_notes.append("Uni-Mol training may require remote GPU scheduling.")
        if str(trainability.get("overall_status") or "").upper() == "WARNING":
            budget_notes.append("Low-label properties should be treated as exploratory.")
        if "quick" in goal.lower() or "smoke" in goal.lower():
            budget_notes.append("User goal suggests limiting to baseline or smoke-test training.")
        return ModelingExperimentDesign(
            backend=backend,
            target_properties=[item.property_id for item in trainable],
            split_strategy=split_strategy,
            validation_strategy=validation_strategy,
            required_artifacts=["cleaned_train_dataset", "trainability_report"] if backend != "none" else ["trainability_report"],
            required_gates=required_gates,
            budget_notes=budget_notes,
        )

    def _metric_interpretations(
        self,
        model_metrics: dict[str, Any],
        backend_summary: dict[str, Any],
    ) -> list[ModelingMetricInterpretation]:
        properties = model_metrics.get("properties", [])
        if not isinstance(properties, list):
            properties = []
        if not properties:
            properties = [
                {"property_id": item.get("property_id"), "metrics": item.get("baseline_metrics", {})}
                for item in backend_summary.get("per_property", [])
                if isinstance(item, dict)
            ]
        interpretations: list[ModelingMetricInterpretation] = []
        for item in properties:
            if not isinstance(item, dict):
                continue
            property_id = str(item.get("property_id") or "default")
            metrics = self._numeric_metrics(item.get("metrics", {}))
            status, decision, message = self._interpret_metrics(metrics)
            interpretations.append(
                ModelingMetricInterpretation(
                    property_id=property_id,
                    metrics=metrics,
                    status=status,
                    decision=decision,
                    message=message,
                )
            )
        if not interpretations:
            interpretations.append(
                ModelingMetricInterpretation(
                    property_id="all",
                    metrics={},
                    status="not_evaluated",
                    decision="continue",
                    message="No model metrics were supplied; run baseline before interpreting model quality.",
                )
            )
        return interpretations

    @staticmethod
    def _retry_proposals(
        trainability: dict[str, Any],
        interpretations: list[ModelingMetricInterpretation],
        design: ModelingExperimentDesign,
    ) -> list[ModelingRetryProposal]:
        proposals: list[ModelingRetryProposal] = []
        overall = str(trainability.get("overall_status") or "").upper()
        if overall == "BLOCKED" or design.backend == "none":
            proposals.extend(
                [
                    ModelingRetryProposal(
                        action="request_more_data",
                        reason="Training data is blocked or insufficient for model training.",
                        target_tasks=["inspect_dataset", "check_trainability"],
                        requires_user_approval=False,
                    ),
                    ModelingRetryProposal(
                        action="run_baseline_only",
                        reason="Avoid high-risk training until labels are sufficient.",
                        target_tasks=["run_baseline"],
                        requires_user_approval=True,
                    ),
                ]
            )
        weak = [item for item in interpretations if item.status in {"weak", "invalid"}]
        if weak:
            proposals.extend(
                [
                    ModelingRetryProposal(
                        action="adjust_split",
                        reason="Weak or invalid metrics may reflect an unstable train/validation split.",
                        target_tasks=["run_baseline", "train_model"],
                        requires_user_approval=True,
                    ),
                    ModelingRetryProposal(
                        action="switch_backend_or_features",
                        reason="Weak metrics justify trying a different backend or feature representation.",
                        target_tasks=["recommend_backend", "train_model"],
                        requires_user_approval=True,
                    ),
                ]
            )
        return ModelingAgent._dedup_retry_proposals(proposals)

    @staticmethod
    def _questions(trainability: dict[str, Any], design: ModelingExperimentDesign) -> list[PlanQuestion]:
        if str(trainability.get("overall_status") or "").upper() != "BLOCKED" and design.backend != "none":
            return []
        return [
            PlanQuestion(
                question_id="q_modeling_data_readiness",
                prompt="Should the agent request more labeled data, reduce target properties, or run baseline only?",
                reason="The available training data is not ready for the requested modeling action.",
                choices=["request_more_data", "reduce_properties", "run_baseline_only"],
                blocks_execution=True,
            )
        ]

    @staticmethod
    def _trainability_properties(trainability: dict[str, Any]) -> list[dict[str, Any]]:
        properties = trainability.get("properties", [])
        return [item for item in properties if isinstance(item, dict)] if isinstance(properties, list) else []

    @staticmethod
    def _infer_backend(goal: str, property_id: str, status: str) -> str:
        normalized = f"{goal} {property_id}".lower()
        if status in {"INSUFFICIENT_LABELS", "INVALID_LABELS", "UNSUPPORTED_TASK_TYPE"}:
            return "none"
        if any(token in normalized for token in ["unimol", "3d", "lambda", "emission", "spectrum"]):
            return "unimol"
        if any(token in normalized for token in ["quick", "smoke", "baseline"]):
            return "baseline"
        return "baseline"

    @staticmethod
    def _backend_confidence(backend: str, prop: dict[str, Any], backend_item: dict[str, Any]) -> float:
        if backend == "none":
            return 0.95
        labels = ModelingAgent._safe_int(prop.get("effective_labels"))
        confidence = 0.55
        if backend_item:
            confidence += 0.15
        if labels >= 100:
            confidence += 0.2
        elif labels >= 30:
            confidence += 0.1
        if backend == "unimol":
            confidence += 0.05
        return min(0.95, round(confidence, 3))

    @staticmethod
    def _backend_reason(backend: str, status: str) -> str:
        if backend == "none":
            return status or "data_not_trainable"
        if backend == "unimol":
            return "3d_sensitive_or_user_requested"
        return "baseline_sufficient_initial_route"

    @staticmethod
    def _backend_requirements(backend: str) -> list[str]:
        if backend == "none":
            return ["resolved trainability issues"]
        if backend == "unimol":
            return ["confirmed training dataset", "remote GPU environment", "gate_3_train_config approval"]
        return ["confirmed training dataset", "baseline feature generation"]

    @staticmethod
    def _backend_risk_flags(backend: str, status: str, backend_summary: dict[str, Any]) -> list[str]:
        flags: list[str] = []
        if backend == "unimol":
            flags.append("high_cost_training")
        if status == "TRAIN_WITH_WARNING":
            flags.append("low_label_count")
        if bool(backend_summary.get("mixed_backend_warning")):
            flags.append("mixed_backend_policy")
        return flags

    @staticmethod
    def _numeric_metrics(raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            return {}
        metrics: dict[str, float] = {}
        for key, value in raw.items():
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            metrics[str(key)] = number
        return metrics

    @staticmethod
    def _coerce_diagnostics_report(raw: ModelDiagnosticsReport | dict[str, Any] | None) -> ModelDiagnosticsReport | None:
        if raw is None:
            return None
        if isinstance(raw, ModelDiagnosticsReport):
            return raw
        if isinstance(raw, dict):
            return ModelDiagnosticsReport.model_validate(raw)
        raise ValueError("diagnostics_report must be an object")

    @staticmethod
    def _first_nonempty(*values: Any) -> str:
        for value in values:
            clean = str(value or "").strip()
            if clean:
                return clean
        return ""

    @staticmethod
    def _json_object(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return ModelingAgent._dedup_strings([str(item or "").strip() for item in value])

    @staticmethod
    def _string_map(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        result: dict[str, str] = {}
        for key, raw in value.items():
            clean_key = str(key or "").strip()
            clean_value = str(raw or "").strip()
            if clean_key and clean_value:
                result[clean_key] = clean_value
        return result

    @staticmethod
    def _safe_int(value: Any) -> int:
        if isinstance(value, bool):
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _interpret_metrics(metrics: dict[str, float]) -> tuple[str, str, str]:
        if not metrics:
            return "not_evaluated", "continue", "No numeric metrics were supplied."
        r2 = metrics.get("r2")
        if r2 is not None:
            if r2 < 0:
                return "weak", "replan", "Negative R2 indicates weak generalization."
            if r2 < 0.2:
                return "weak", "retry", "Low R2 indicates the model is not reliable yet."
            if r2 < 0.5:
                return "promising", "continue", "R2 is modest; continue only with review."
            return "strong", "continue", "R2 is strong enough for cautious downstream screening."
        rmse = metrics.get("rmse")
        if rmse is not None and rmse > 0:
            return "promising", "continue", "RMSE is available; compare against property scale before promotion."
        return "not_evaluated", "continue", "Metrics do not include R2 or RMSE."

    @staticmethod
    def _dedup_retry_proposals(proposals: list[ModelingRetryProposal]) -> list[ModelingRetryProposal]:
        seen: set[str] = set()
        result: list[ModelingRetryProposal] = []
        for proposal in proposals:
            if proposal.action in seen:
                continue
            seen.add(proposal.action)
            result.append(proposal)
        return result

    @staticmethod
    def _select_domain_model(
        *,
        goal: str,
        property_id: str,
        profile: dict[str, Any],
        available_inputs: set[str] | list[str] | tuple[str, ...] | None,
    ):
        if profile.get("domain") != "oled":
            return None
        normalized = f"{goal} {property_id}".lower()
        use_case = "scalar_prediction"
        if ModelingAgent._is_plqy_target(normalized) and ModelingAgent._is_high_value_goal(goal):
            use_case = "high_plqy_screening"
        inputs = available_inputs
        if inputs is None:
            inputs = {"canonical_smiles", "solvent"}
        try:
            return OLED_MODEL_REGISTRY.select(
                domain="oled",
                property_id=property_id,
                use_case=use_case,
                available_inputs=inputs,
            )
        except ValueError:
            return None

    @staticmethod
    def _target_rule_profile(goal: str, property_id: str) -> dict[str, Any]:
        normalized = f"{goal} {property_id}".lower()
        if ModelingAgent._is_plqy_target(normalized):
            return {
                "domain": "oled" if "oled" in normalized else "organic_optoelectronics",
                "risk_flags": [
                    "bounded_target",
                    "solvent_context_dependence",
                    "high_value_compression_risk",
                    "noisy_literature_labels",
                ],
                "preprocessing_steps": [
                    "canonicalize_smiles_with_rdkit",
                    "preserve_solvent_conditioned_rows",
                    "add_solvent_descriptors_or_embeddings",
                    "keep_reference_and_condition_metadata",
                    "flag_high_replicate_variance",
                ],
                "split_strategy": "scaffold_split_grouped_by_canonical_smiles",
                "target_transform": "bounded_logit_or_calibrated_regression",
                "recommended_backend": "unimol_with_solvent_features",
                "hyperparameters": {
                    "epochs": 30,
                    "patience": 8,
                    "seed_policy": "multi_seed_if_metrics_unstable",
                    "high_value_threshold": 0.7,
                    "solvent_feature_policy": "manual_descriptors_or_pca64",
                },
                "acceptance_criteria": {
                    "min_r2": 0.35,
                    "max_mae": 0.18,
                    "review_high_value_bucket_bias": True,
                    "max_abs_high_value_bias": 0.20,
                },
            }
        if ModelingAgent._is_emission_target(normalized):
            return {
                "domain": "oled" if "oled" in normalized else "organic_optoelectronics",
                "risk_flags": [
                    "solvent_context_dependence",
                    "unit_nm_required",
                    "conformer_failure_sensitivity",
                ],
                "preprocessing_steps": [
                    "canonicalize_smiles_with_rdkit",
                    "standardize_emission_units_to_nm",
                    "flag_large_replicate_disagreement",
                    "track_3d_conformer_failures",
                ],
                "split_strategy": "scaffold_split_grouped_by_canonical_smiles",
                "target_transform": "none",
                "recommended_backend": "unimol",
                "hyperparameters": {
                    "epochs": 30,
                    "patience": 8,
                    "seed_policy": "single_seed_then_ensemble_if_unstable",
                },
                "acceptance_criteria": {
                    "min_r2": 0.75,
                    "max_mae_nm": 35,
                    "compare_against_mean_baseline": True,
                },
            }
        return {
            "domain": "general",
            "risk_flags": ["target_specific_rules_not_configured"],
            "preprocessing_steps": ["canonicalize_smiles_with_rdkit", "preserve_property_units_and_metadata"],
            "split_strategy": "scaffold_split_then_random_fallback",
            "target_transform": "decide_from_target_distribution",
            "recommended_backend": "baseline_then_unimol_if_3d_relevant",
            "hyperparameters": {"seed_policy": "single_seed_smoke_then_expand"},
            "acceptance_criteria": {"compare_against_mean_and_fingerprint_baselines": True},
        }

    @staticmethod
    def _trainability_item(trainability: dict[str, Any], property_id: str) -> dict[str, Any]:
        for item in ModelingAgent._trainability_properties(trainability):
            if str(item.get("property_id") or "").strip() == property_id:
                return item
        return {}

    @staticmethod
    def _diagnostic_readiness(metrics: dict[str, float]) -> str:
        if not metrics:
            return "not_evaluated"
        r2 = metrics.get("r2")
        if r2 is not None:
            if r2 < 0.2:
                return "weak"
            if r2 < 0.5:
                return "promising"
            return "strong"
        return "promising" if metrics else "not_evaluated"

    @staticmethod
    def _diagnostic_decision(readiness: str) -> str:
        if readiness == "strong":
            return "accept"
        if readiness == "promising":
            return "low_confidence_accept"
        if readiness == "weak":
            return "rerun_recommended"
        return "not_evaluated"

    @staticmethod
    def _baseline_comparison(metrics: dict[str, float], baseline_metrics: dict[str, Any]) -> dict[str, float]:
        comparison: dict[str, float] = {}
        mae = metrics.get("mae")
        r2 = metrics.get("r2")
        for baseline_name, raw_metrics in baseline_metrics.items():
            if not isinstance(raw_metrics, dict):
                continue
            clean_name = str(baseline_name or "baseline").strip() or "baseline"
            baseline_mae = ModelingAgent._safe_float(raw_metrics.get("mae"))
            baseline_r2 = ModelingAgent._safe_float(raw_metrics.get("r2"))
            if mae is not None and baseline_mae is not None:
                comparison[f"{clean_name}_mae_improvement"] = round(baseline_mae - mae, 4)
            if r2 is not None and baseline_r2 is not None:
                comparison[f"{clean_name}_r2_delta"] = round(r2 - baseline_r2, 4)
        return comparison

    @staticmethod
    def _build_rerun_proposal(
        *,
        property_id: str,
        readiness: str,
        risk_flags: list[str],
        modeling_brief: TargetModelingBrief | None,
    ) -> RerunProposal:
        candidate_changes: list[str] = []
        rationale: list[str] = []
        if "high_value_underprediction" in risk_flags:
            candidate_changes.extend(
                [
                    "posthoc_calibration",
                    "high_value_weighting_or_two_stage_model",
                    "seed_ensemble",
                ]
            )
            rationale.append("High target values are underpredicted, which hurts candidate screening recall.")
        if "prediction_range_compression" in risk_flags:
            candidate_changes.append("target_distribution_calibration")
            rationale.append("Prediction upper tail is compressed relative to the observed target distribution.")
        if readiness == "weak":
            candidate_changes.extend(["adjust_split", "switch_backend_or_features", "request_more_data"])
            rationale.append("Weak validation metrics justify changing split, features, backend, or data scope.")
        if modeling_brief and "solvent_context_dependence" in modeling_brief.risk_flags:
            candidate_changes.append("solvent_feature_sweep")
        candidate_changes = ModelingAgent._dedup_strings(candidate_changes)
        return RerunProposal(
            property_id=property_id,
            trigger="; ".join(risk_flags) if risk_flags else readiness,
            candidate_changes=candidate_changes,
            rationale=rationale or ["Model diagnostics indicate the current model should be reviewed before promotion."],
            expected_impact="Improve validation reliability and reduce target-specific prediction bias.",
            estimated_cost="medium",
            required_approvals=[GateName.TRAIN_CONFIG.value],
            fallback_policy="If rerun still fails, keep the model as a low-confidence signal and request more data.",
            requires_user_approval=True,
            executable=False,
        )

    @staticmethod
    def _is_plqy_target(normalized: str) -> bool:
        return any(
            token in normalized
            for token in [
                "plqy",
                "quantum_yield",
                "quantum yield",
                "photoluminescence quantum yield",
                "fluorescence quantum yield",
            ]
        )

    @staticmethod
    def _is_emission_target(normalized: str) -> bool:
        return any(token in normalized for token in ["lambda_em", "emission", "emissive", "fluorescence maximum"])

    @staticmethod
    def _is_high_value_goal(goal: str) -> bool:
        normalized = str(goal or "").lower()
        return any(token in normalized for token in ["high", "maximize", "maximise", "top", "prioritize", "screen"])

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _dedup_strings(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = str(value or "").strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _safe_property_stem(property_id: str) -> str:
        stem = "".join(char if char.isalnum() else "_" for char in str(property_id or "property").strip().lower())
        stem = "_".join(part for part in stem.split("_") if part)
        return stem or "property"

    @staticmethod
    def _render_markdown(proposal: ModelingPlanProposal) -> str:
        lines = [
            "# Modeling Plan Proposal",
            "",
            f"- Run: `{proposal.run_id}`",
            f"- Status: `{proposal.status}`",
            f"- Backend: `{proposal.experiment_design.backend}`",
            "",
            "## Backend Recommendations",
        ]
        for item in proposal.backend_recommendations:
            lines.append(f"- `{item.property_id}` -> `{item.backend}` ({item.confidence:.2f}): {item.reason}")
        lines.extend(["", "## Metric Interpretation"])
        for item in proposal.metric_interpretations:
            lines.append(f"- `{item.property_id}`: `{item.status}` / `{item.decision}` - {item.message}")
        lines.extend(["", "## Retry Proposals"])
        lines.extend(f"- `{item.action}`: {item.reason}" for item in proposal.retry_proposals)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_target_brief_markdown(brief: TargetModelingBrief) -> str:
        lines = [
            "# Target Modeling Brief",
            "",
            f"- Run: `{brief.run_id}`",
            f"- Property: `{brief.property_id}`",
            f"- Domain: `{brief.domain}`",
            f"- Backend: `{brief.recommended_backend}`",
            f"- Split: `{brief.split_strategy}`",
            f"- Target transform: `{brief.target_transform}`",
            "",
            "## Risks",
        ]
        lines.extend(f"- `{flag}`" for flag in brief.risk_flags)
        lines.extend(["", "## Preprocessing"])
        lines.extend(f"- `{step}`" for step in brief.preprocessing_steps)
        lines.extend(["", "## Evidence Sources"])
        lines.extend(f"- `{source}`" for source in brief.evidence_sources)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_diagnostics_markdown(report: ModelDiagnosticsReport) -> str:
        lines = [
            "# Model Diagnostics Report",
            "",
            f"- Run: `{report.run_id}`",
            f"- Property: `{report.property_id}`",
            f"- Model: `{report.model_id}`",
            f"- Readiness: `{report.readiness}`",
            f"- Decision: `{report.decision}`",
            "",
            "## Risk Flags",
        ]
        lines.extend(f"- `{flag}`" for flag in report.risk_flags)
        lines.extend(["", "## Messages"])
        lines.extend(f"- {message}" for message in report.messages)
        if report.rerun_proposal:
            lines.extend(["", "## Rerun Proposal"])
            lines.extend(f"- `{change}`" for change in report.rerun_proposal.candidate_changes)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _render_model_package_review_markdown(review: ModelPackageReview) -> str:
        lines = [
            "# Model Package Review",
            "",
            f"- Run: `{review.run_id}`",
            f"- Model: `{review.model_id}`",
            f"- Property: `{review.property_id}`",
            f"- Backend: `{review.backend}`",
            f"- Decision: `{review.decision}`",
            f"- Status: `{review.status}`",
            "",
            "## Rationale",
        ]
        lines.extend(f"- {item}" for item in review.rationale)
        lines.extend(["", "## Risk Flags"])
        lines.extend(f"- `{flag}`" for flag in review.risk_flags)
        if review.required_gates or review.required_permissions:
            lines.extend(["", "## Required Approvals"])
            lines.extend(f"- `{gate}`" for gate in review.required_gates)
            lines.extend(f"- `{permission}`" for permission in review.required_permissions)
        return "\n".join(lines) + "\n"
