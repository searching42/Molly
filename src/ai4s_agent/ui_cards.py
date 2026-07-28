from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.agents.evaluation import compute_autonomy_metrics
from ai4s_agent.data_layer import generate_property_catalog, inspect_dataset
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import (
    ModelDiagnosticsReport,
    ModelPackageReview,
    RerunProposal,
    RunPlan,
    StageState,
    TargetModelingBrief,
)
from ai4s_agent.trainability import assess_trainability


def build_data_confirmation_card(payload: dict[str, Any], *, base: Path) -> dict[str, Any]:
    dataset_path = resolve_payload_path(str(payload.get("dataset_path") or ""), base=base)
    if not dataset_path.exists() or not dataset_path.is_file():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    min_numeric_ratio = parse_float(payload.get("min_numeric_ratio", 0.6), field="min_numeric_ratio")
    min_nonempty = parse_positive_int(payload.get("min_nonempty", 1), field="min_nonempty")
    inspection = inspect_dataset(
        dataset_path,
        min_numeric_ratio=min_numeric_ratio,
        min_nonempty=min_nonempty,
    )
    catalog = generate_property_catalog(inspection)
    trainability = assess_trainability(
        [
            {
                "property_id": item["property_id"],
                "effective_labels": item["label_count"],
                "numeric_ratio": 1.0,
            }
            for item in catalog.get("properties", [])
            if isinstance(item, dict)
        ]
    ).model_dump(mode="json")
    duplicate_count = sum(max(0, item.row_count - 1) for item in inspection.duplicate_conflicts)
    invalid_count = inspection.structure.row_count if not inspection.smiles_column else 0

    return {
        "run_id": str(payload.get("run_id") or ""),
        "dataset_path": str(dataset_path),
        "generated_at": now_iso(),
        "requires_confirmation": True,
        "sections": {
            "data_overview": {
                "row_count": inspection.structure.row_count,
                "column_count": inspection.structure.column_count,
                "smiles_column": inspection.smiles_column,
                "duplicate_count": duplicate_count,
                "invalid_count": invalid_count,
                "warnings": inspection.warnings,
            },
            "property_catalog": catalog.get("properties", []),
            "cleaning_rule_draft": {
                "unit_conversions": [
                    {
                        "property_id": item["property_id"],
                        "source_column": item["source_column"],
                        "conversion_note": item.get("conversion_note", ""),
                        "scale": item.get("scale", 1.0),
                    }
                    for item in catalog.get("properties", [])
                    if isinstance(item, dict) and item.get("conversion_note")
                ],
                "duplicate_aggregation": "review_conflicts" if inspection.duplicate_conflicts else "none",
                "outlier_handling": "review_warnings" if inspection.outlier_warnings else "none",
                "split_strategy": inspection.split_assessment.fallback_strategy
                or inspection.split_assessment.status,
                "split_reason": inspection.split_assessment.reason,
            },
            "trainability": trainability,
            "confirmation_actions": [
                "execute_cleaning",
                "edit_mapping",
                "save_project_rule",
                "cancel",
            ],
        },
    }


def build_run_confirmation_card(payload: dict[str, Any]) -> dict[str, Any]:
    run_plan = expand_plan_from_payload(payload)
    registry = AtomicTaskRegistry()
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    required_gates: list[str] = []
    task_risks: list[dict[str, object]] = []
    generation_tasks: list[str] = []
    generation_requires_confirmation = False
    generation_backend = str(payload.get("generation_backend") or "deterministic_stub").strip().lower()
    generation_count = parse_positive_int(payload.get("generation_count", 32), field="generation_count")
    for task in run_plan.tasks:
        spec = registry.get(task.task_id)
        risk = spec.risk_level.value
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        for gate in spec.gates:
            if gate not in required_gates:
                required_gates.append(gate)
        if task.task_id == "generate_candidates":
            generation_tasks.append(task.task_id)
            generation_requires_confirmation = (
                generation_backend != "deterministic_stub" or generation_count >= 128
            )
            if generation_requires_confirmation and "generate_candidates_expensive" not in required_gates:
                required_gates.append("generate_candidates_expensive")
        task_risks.append(
            {
                "task_id": task.task_id,
                "risk_level": risk,
                "gates": list(spec.gates),
                "depends_on": list(task.depends_on),
            }
        )

    sections: dict[str, Any] = {
        "run_summary": {
            "requested_tasks": run_plan.requested_tasks,
            "expanded_task_count": len(run_plan.tasks),
            "missing_artifacts": run_plan.missing_artifacts,
        },
        "dependency_plan": run_plan.model_dump(mode="json"),
        "risk_gates": {
            "risk_counts": risk_counts,
            "required_gates": required_gates,
            "tasks": task_risks,
        },
        "confirmation_actions": [
            "continue",
            "save_and_stop",
            "modify_plan",
            "cancel",
        ],
    }
    if generation_tasks:
        sections["generation_confirmation"] = {
            "generation_tasks": generation_tasks,
            "requires_confirmation": generation_requires_confirmation,
            "generation_backend": generation_backend,
            "generation_count": generation_count,
            "thresholds": {
                "expensive_count_threshold": 128,
            },
        }

    return {
        "run_id": run_plan.run_id,
        "generated_at": now_iso(),
        "requires_confirmation": bool(required_gates or run_plan.missing_artifacts),
        "sections": sections,
    }


def build_stage_timeline(state: StageState) -> dict[str, Any]:
    error = state.error if isinstance(state.error, dict) else {}
    return {
        "current_stage": state.stage,
        "next_stage": state.next_stage,
        "status": state.status.value,
        "started_at": state.started_at,
        "updated_at": state.updated_at,
        "ended_at": state.ended_at,
        "retryable": bool(error.get("retryable")),
        "error": error,
        "events": [item.model_dump(mode="json") for item in state.history],
        "artifacts": [item.model_dump(mode="json") for item in state.artifacts],
        "details": state.details,
    }


def build_report_preview(
    *,
    run_dir: Path,
    artifact_id: str,
    relative_path: str,
    max_chars: int = 12000,
) -> dict[str, Any]:
    clean_artifact_id = str(artifact_id or "").strip()
    clean_relative = str(relative_path or "").strip()
    if not clean_artifact_id or not clean_relative:
        raise ValueError("artifact_id and relative_path required")

    base = run_dir.resolve()
    path = (base / clean_relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError("report path escapes run directory")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"report artifact not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        fmt = "markdown"
    elif suffix == ".html":
        fmt = "html"
    elif suffix == ".json":
        fmt = "json"
    else:
        fmt = "text"
    content = path.read_text(encoding="utf-8", errors="replace")
    truncated = len(content) > max_chars
    return {
        "artifact_id": clean_artifact_id,
        "relative_path": clean_relative,
        "format": fmt,
        "content": content[:max_chars],
        "truncated": truncated,
    }


def build_agent_review_card(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a UI-facing review surface from structured agent artifacts."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    plan = optional_object_payload(payload, "plan_proposal")
    verification = optional_object_payload(payload, "verification_report")
    revision = optional_object_payload(payload, "run_plan_revision")
    research = optional_object_payload(payload, "research_proposal")
    modeling = optional_object_payload(payload, "modeling_proposal")
    generation = optional_object_payload(payload, "generation_proposal")
    report = optional_object_payload(payload, "report_proposal")
    target_brief = optional_validated_model_payload(payload, "target_modeling_brief", TargetModelingBrief)
    diagnostics = optional_validated_model_payload(payload, "model_diagnostics_report", ModelDiagnosticsReport)
    rerun = optional_validated_model_payload(payload, "rerun_proposal", RerunProposal)
    package_review = optional_validated_model_payload(payload, "model_package_review", ModelPackageReview)
    if not rerun and object_payload(diagnostics.get("rerun_proposal")):
        rerun = validate_model_payload(
            object_payload(diagnostics.get("rerun_proposal")),
            "model_diagnostics_report.rerun_proposal",
            RerunProposal,
        )

    run_id = first_nonempty(
        plan.get("run_id"),
        verification.get("run_id"),
        revision.get("run_id"),
        research.get("run_id"),
        modeling.get("run_id"),
        generation.get("run_id"),
        report.get("run_id"),
        target_brief.get("run_id"),
        diagnostics.get("run_id"),
        package_review.get("run_id"),
    )
    run_plan = object_payload(plan.get("run_plan"))
    approval_controls = build_agent_approval_controls(
        run_id=run_id,
        plan=plan,
        revision=revision,
        generation=generation,
        modeling=modeling,
        research=research,
        target_brief=target_brief,
        diagnostics=diagnostics,
        rerun=rerun,
        package_review=package_review,
    )
    metric_payload = dict(payload)
    metric_payload["agent_review_card"] = {"approval_controls": approval_controls}
    sections = {
        "plan_explanation": build_plan_explanation_section(plan),
        "task_timeline": build_task_timeline_section(run_plan, plan),
        "missing_information": build_missing_information_section(plan, run_plan),
        "memory_use": list_payload(plan.get("memory_references")),
        "verifier_findings": build_verifier_findings_section(verification),
        "replan_compare": build_replan_compare_section(revision),
        "target_modeling_brief": build_target_modeling_brief_section(target_brief),
        "model_diagnostics": build_model_diagnostics_section(diagnostics),
        "rerun_proposal": build_rerun_proposal_section(rerun),
        "model_package_review": build_model_package_review_section(package_review),
        "agent_proposals": build_agent_proposals_section(
            research=research,
            modeling=modeling,
            generation=generation,
            report=report,
        ),
        "autonomy_metrics": compute_autonomy_metrics(metric_payload),
    }
    return {
        "run_id": run_id,
        "generated_at": now_iso(),
        "requires_confirmation": bool(approval_controls or sections["missing_information"]),
        "sections": sections,
        "approval_controls": approval_controls,
    }


def build_plan_explanation_section(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "planner_backend": str(plan.get("planner_backend") or ""),
        "status": str(plan.get("status") or ""),
        "goal": str(plan.get("goal") or ""),
        "rationales": list_payload(plan.get("rationales")),
        "assumptions": string_list(plan.get("assumptions")),
        "required_gates": string_list(plan.get("required_gates")),
    }


def build_task_timeline_section(run_plan: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    rationale_by_task = {
        str(item.get("task_id") or ""): item
        for item in list_payload(plan.get("rationales"))
        if isinstance(item, dict)
    }
    timeline: list[dict[str, Any]] = []
    for index, task in enumerate(list_payload(run_plan.get("tasks")), start=1):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "")
        rationale = object_payload(rationale_by_task.get(task_id))
        timeline.append(
            {
                "order": index,
                "task_id": task_id,
                "depends_on": string_list(task.get("depends_on")),
                "required_artifacts": string_list(task.get("required_artifacts")),
                "output_artifacts": string_list(task.get("output_artifacts")),
                "unresolved_requirements": string_list(task.get("unresolved_requirements")),
                "risk_level": str(rationale.get("risk_level") or ""),
                "required_gates": string_list(rationale.get("required_gates")),
                "reason": str(rationale.get("reason") or ""),
            }
        )
    return timeline


def build_missing_information_section(plan: dict[str, Any], run_plan: dict[str, Any]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for artifact in string_list(run_plan.get("missing_artifacts")):
        missing.append(
            {
                "kind": "artifact",
                "id": artifact,
                "message": f"Artifact `{artifact}` is required before execution.",
                "blocks_execution": True,
            }
        )
    for question in list_payload(plan.get("questions")):
        if not isinstance(question, dict):
            continue
        if not bool(question.get("blocks_execution")):
            continue
        missing.append(
            {
                "kind": "question",
                "id": str(question.get("question_id") or ""),
                "message": str(question.get("prompt") or ""),
                "reason": str(question.get("reason") or ""),
                "choices": string_list(question.get("choices")),
                "blocks_execution": True,
            }
        )
    return missing


def build_verifier_findings_section(verification: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for finding in list_payload(verification.get("findings")):
        if not isinstance(finding, dict):
            continue
        findings.append(
            {
                "finding_id": str(finding.get("finding_id") or ""),
                "category": str(finding.get("category") or ""),
                "severity": str(finding.get("severity") or ""),
                "decision": str(finding.get("decision") or ""),
                "message": str(finding.get("message") or ""),
                "evidence": object_payload(finding.get("evidence")),
            }
        )
    return findings


def build_replan_compare_section(revision: dict[str, Any]) -> dict[str, Any]:
    diff = object_payload(revision.get("diff"))
    previous = object_payload(revision.get("previous_plan"))
    revised = object_payload(revision.get("revised_plan"))
    return {
        "revision_id": str(revision.get("revision_id") or ""),
        "reason": str(revision.get("reason") or ""),
        "recovery_actions": string_list(revision.get("recovery_actions")),
        "added_tasks": string_list(diff.get("added_tasks")),
        "removed_tasks": string_list(diff.get("removed_tasks")),
        "unchanged_tasks": string_list(diff.get("unchanged_tasks")),
        "changed_dependencies": object_payload(diff.get("changed_dependencies")),
        "previous_requested_tasks": string_list(previous.get("requested_tasks")),
        "revised_requested_tasks": string_list(revised.get("requested_tasks")),
        "approvals_required": string_list(revision.get("approvals_required")),
        "user_approval_required": bool(revision.get("user_approval_required")),
    }


def build_target_modeling_brief_section(brief: dict[str, Any]) -> dict[str, Any]:
    if not brief:
        return {}
    evidence_sources = string_list(brief.get("evidence_sources"))
    evidence_items = list_payload(brief.get("evidence_items"))
    external_policy = str(brief.get("external_search_policy") or "").strip()
    source_labels = list(evidence_sources)
    if external_policy:
        source_labels.append(f"external_search:{external_policy}")
    for item in evidence_items:
        if isinstance(item, dict):
            source_type = str(item.get("source_type") or "").strip()
            if source_type:
                source_labels.append(source_type)
    return {
        "run_id": str(brief.get("run_id") or ""),
        "goal": str(brief.get("goal") or ""),
        "property_id": str(brief.get("property_id") or ""),
        "domain": str(brief.get("domain") or ""),
        "status": str(brief.get("status") or ""),
        "recommended_backend": str(brief.get("recommended_backend") or ""),
        "split_strategy": str(brief.get("split_strategy") or ""),
        "target_transform": str(brief.get("target_transform") or ""),
        "risk_flags": string_list(brief.get("risk_flags")),
        "preprocessing_steps": string_list(brief.get("preprocessing_steps")),
        "evidence_sources": evidence_sources,
        "evidence_items": evidence_items,
        "external_search_policy": external_policy,
        "source_labels": dedup_strings(source_labels),
        "acceptance_criteria": object_payload(brief.get("acceptance_criteria")),
        "dataset_context": object_payload(brief.get("dataset_context")),
        "model_selection": object_payload(brief.get("model_selection")),
        "assumptions": string_list(brief.get("assumptions")),
        "questions": list_payload(brief.get("questions")),
        "executable": bool(brief.get("executable")),
    }


def build_model_diagnostics_section(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if not diagnostics:
        return {}
    return {
        "run_id": str(diagnostics.get("run_id") or ""),
        "goal": str(diagnostics.get("goal") or ""),
        "property_id": str(diagnostics.get("property_id") or ""),
        "model_id": str(diagnostics.get("model_id") or ""),
        "readiness": str(diagnostics.get("readiness") or ""),
        "decision": str(diagnostics.get("decision") or ""),
        "metrics": object_payload(diagnostics.get("metrics")),
        "baseline_comparison": object_payload(diagnostics.get("baseline_comparison")),
        "distribution_diagnostics": object_payload(diagnostics.get("distribution_diagnostics")),
        "fold_diagnostics": object_payload(diagnostics.get("fold_diagnostics")),
        "risk_flags": string_list(diagnostics.get("risk_flags")),
        "messages": string_list(diagnostics.get("messages")),
        "rerun_proposal_present": bool(object_payload(diagnostics.get("rerun_proposal"))),
        "source_labels": dedup_strings(
            [
                "model_diagnostics_report",
                *[f"risk:{flag}" for flag in string_list(diagnostics.get("risk_flags"))],
            ]
        ),
        "executable": bool(diagnostics.get("executable")),
    }


def build_rerun_proposal_section(rerun: dict[str, Any]) -> dict[str, Any]:
    if not rerun:
        return {}
    return {
        "property_id": str(rerun.get("property_id") or ""),
        "trigger": str(rerun.get("trigger") or ""),
        "candidate_changes": string_list(rerun.get("candidate_changes")),
        "rationale": string_list(rerun.get("rationale")),
        "expected_impact": str(rerun.get("expected_impact") or ""),
        "estimated_cost": str(rerun.get("estimated_cost") or ""),
        "required_approvals": string_list(rerun.get("required_approvals")),
        "fallback_policy": str(rerun.get("fallback_policy") or ""),
        "requires_user_approval": bool(rerun.get("requires_user_approval")),
        "source_labels": dedup_strings(
            ["rerun_proposal", *[f"approval:{item}" for item in string_list(rerun.get("required_approvals"))]]
        ),
        "executable": bool(rerun.get("executable")),
    }


def build_model_package_review_section(review: dict[str, Any]) -> dict[str, Any]:
    if not review:
        return {}
    return {
        "run_id": str(review.get("run_id") or ""),
        "goal": str(review.get("goal") or ""),
        "model_id": str(review.get("model_id") or ""),
        "domain": str(review.get("domain") or ""),
        "property_id": str(review.get("property_id") or ""),
        "use_case": str(review.get("use_case") or ""),
        "backend": str(review.get("backend") or ""),
        "status": str(review.get("status") or ""),
        "decision": str(review.get("decision") or ""),
        "metrics": object_payload(review.get("metrics")),
        "applicability": object_payload(review.get("applicability")),
        "feature_requirements": string_list(review.get("feature_requirements")),
        "input_columns": object_payload(review.get("input_columns")),
        "limitations": string_list(review.get("limitations")),
        "risk_flags": string_list(review.get("risk_flags")),
        "rationale": string_list(review.get("rationale")),
        "required_gates": string_list(review.get("required_gates")),
        "required_permissions": string_list(review.get("required_permissions")),
        "promotion_draft": object_payload(review.get("promotion_draft")),
        "memory_updates": list_payload(review.get("memory_updates")),
        "source_labels": dedup_strings(
            [
                "model_package_review",
                f"decision:{review.get('decision')}",
                *[f"risk:{flag}" for flag in string_list(review.get("risk_flags"))],
            ]
        ),
        "executable": bool(review.get("executable")),
    }


def build_agent_proposals_section(
    *,
    research: dict[str, Any],
    modeling: dict[str, Any],
    generation: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "research": compact_proposal(research, extra_keys=["evidence_quality"]),
        "modeling": compact_proposal(modeling, extra_keys=["retry_proposals", "experiment_design"]),
        "generation": compact_proposal(generation, extra_keys=["backend", "requested_count", "required_permissions"]),
        "report": compact_proposal(report, extra_keys=["next_steps", "limitations"]),
    }


def compact_proposal(payload: dict[str, Any], *, extra_keys: list[str]) -> dict[str, Any]:
    if not payload:
        return {}
    result = {
        "status": str(payload.get("status") or ""),
        "questions": list_payload(payload.get("questions")),
        "assumptions": string_list(payload.get("assumptions")),
        "executable": bool(payload.get("executable")),
    }
    for key in extra_keys:
        value = payload.get(key)
        if value not in (None, "", [], {}):
            result[key] = value
    return result


def build_agent_approval_controls(
    *,
    run_id: str,
    plan: dict[str, Any],
    revision: dict[str, Any],
    generation: dict[str, Any],
    modeling: dict[str, Any],
    research: dict[str, Any],
    target_brief: dict[str, Any],
    diagnostics: dict[str, Any],
    rerun: dict[str, Any],
    package_review: dict[str, Any],
) -> list[dict[str, str]]:
    controls: list[dict[str, str]] = []
    if plan:
        controls.append(
            approval_control(
                "confirm_plan",
                target_type="plan",
                target_id=run_id,
                action="confirm_plan",
                label="Confirm plan before execution",
            )
        )
    if target_brief:
        controls.append(
            approval_control(
                "confirm_modeling_brief",
                target_type="modeling_brief",
                target_id=artifact_target_id(run_id, target_brief.get("property_id")),
                action="confirm_modeling_brief",
                label="Confirm target modeling brief",
            )
        )
    if diagnostics:
        controls.append(
            approval_control(
                "confirm_model_diagnostics",
                target_type="model_diagnostics",
                target_id=artifact_target_id(run_id, diagnostics.get("property_id"), diagnostics.get("model_id")),
                action="confirm_model_diagnostics",
                label="Confirm model diagnostics",
            )
        )
    if rerun and bool(rerun.get("requires_user_approval")):
        controls.append(
            approval_control(
                "confirm_rerun_proposal",
                target_type="rerun_proposal",
                target_id=artifact_target_id(run_id, rerun.get("property_id")),
                action="confirm_rerun_proposal",
                label="Confirm model rerun proposal",
            )
        )
    if package_review:
        controls.append(
            approval_control(
                "confirm_model_package_review",
                target_type="model_package_review",
                target_id=artifact_target_id(run_id, package_review.get("property_id"), package_review.get("model_id")),
                action="confirm_model_package_review",
                label="Confirm model package review",
            )
        )
    for gate in collect_required_gates(plan, revision, rerun, package_review):
        controls.append(
            approval_control(
                f"approve_{gate}",
                target_type="gate",
                target_id=gate,
                action="approve_gate",
                label=f"Approve gate {gate}",
            )
        )
    if list_payload(plan.get("memory_references")):
        controls.append(
            approval_control(
                "approve_memory_use",
                target_type="memory",
                target_id=run_id,
                action="approve_memory_use",
                label="Approve visible project memory use",
            )
        )
    if revision:
        controls.append(
            approval_control(
                "confirm_replan",
                target_type="replan",
                target_id=str(revision.get("revision_id") or run_id),
                action="confirm_replan",
                label="Confirm revised plan",
            )
        )
    for permission in collect_required_permissions(revision, generation, modeling, research, rerun, package_review):
        controls.append(
            approval_control(
                f"approve_permission_{permission}",
                target_type="permission",
                target_id=permission,
                action="approve_permission",
                label=f"Approve permission {permission}",
            )
        )
    return dedup_controls(controls)


def collect_required_gates(
    plan: dict[str, Any],
    revision: dict[str, Any],
    rerun: dict[str, Any],
    package_review: dict[str, Any],
) -> list[str]:
    gates = string_list(plan.get("required_gates"))
    for rationale in list_payload(plan.get("rationales")):
        if isinstance(rationale, dict):
            gates.extend(string_list(rationale.get("required_gates")))
    for approval in string_list(revision.get("approvals_required")):
        if approval.startswith("gate_"):
            gates.append(approval)
    for approval in string_list(rerun.get("required_approvals")):
        if approval.startswith("gate_"):
            gates.append(approval)
    for gate in string_list(package_review.get("required_gates")):
        gates.append(gate)
    return dedup_strings(gates)


def collect_required_permissions(
    revision: dict[str, Any],
    generation: dict[str, Any],
    modeling: dict[str, Any],
    research: dict[str, Any],
    rerun: dict[str, Any],
    package_review: dict[str, Any],
) -> list[str]:
    permissions: list[str] = []
    for approval in string_list(revision.get("approvals_required")):
        if not approval.startswith("gate_"):
            permissions.append(approval)
    permissions.extend(string_list(generation.get("required_permissions")))
    for proposal in list_payload(modeling.get("retry_proposals")):
        if isinstance(proposal, dict) and bool(proposal.get("requires_user_approval")):
            permissions.append(f"modeling_retry:{proposal.get('action')}")
    if research.get("status") == "needs_clarification":
        permissions.append("external_acquisition_scope")
    for approval in string_list(rerun.get("required_approvals")):
        if not approval.startswith("gate_"):
            permissions.append(approval)
    permissions.extend(string_list(package_review.get("required_permissions")))
    return dedup_strings(permissions)


def approval_control(control_id: str, *, target_type: str, target_id: str, action: str, label: str) -> dict[str, str]:
    return {
        "control_id": control_id,
        "target_type": target_type,
        "target_id": target_id,
        "action": action,
        "label": label,
    }


def artifact_target_id(*parts: object) -> str:
    clean_parts = [str(part or "").strip() for part in parts if str(part or "").strip()]
    return ":".join(clean_parts)


def dedup_controls(controls: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []
    for control in controls:
        key = (control["target_type"], control["target_id"], control["action"])
        if key in seen:
            continue
        seen.add(key)
        result.append(control)
    return result


def object_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def optional_object_payload(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload or payload.get(key) is None:
        return {}
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def optional_validated_model_payload(payload: dict[str, Any], key: str, model_cls: Any) -> dict[str, Any]:
    raw = optional_object_payload(payload, key)
    if not raw:
        return {}
    return validate_model_payload(raw, key, model_cls)


def validate_model_payload(raw: dict[str, Any], key: str, model_cls: Any) -> dict[str, Any]:
    try:
        return model_cls.model_validate(raw).model_dump(mode="json")
    except ValidationError as exc:
        raise ValueError(f"{key} is invalid: {exc}") from exc


def list_payload(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def first_nonempty(*values: object) -> str:
    for value in values:
        clean = str(value or "").strip()
        if clean:
            return clean
    return ""


def expand_plan_from_payload(payload: dict[str, Any]) -> RunPlan:
    run_id = str(payload.get("run_id") or "preview").strip() or "preview"
    requested_tasks = string_list(payload.get("requested_tasks"))
    if not requested_tasks:
        raise ValueError("requested_tasks required")
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=requested_tasks,
        available_artifacts=string_list(payload.get("available_artifacts")),
    )


def string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def dedup_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def resolve_payload_path(path_like: str, *, base: Path) -> Path:
    workspace = base.resolve()
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = workspace / path
    resolved = path.resolve()
    if not resolved.is_relative_to(workspace):
        raise ValueError("path must stay within workspace")
    return resolved


def parse_positive_int(value: object, *, field: str) -> int:
    if value in (None, ""):
        raise ValueError(f"{field} must be a positive integer")
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def parse_float(value: object, *, field: str) -> float:
    if value in (None, ""):
        raise ValueError(f"{field} must be numeric")
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
