from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.planner import AtomicTaskRegistry, diff_run_plans, expand_run_plan
from ai4s_agent.schemas import (
    PlanQuestion,
    ReplanRequest,
    RiskLevel,
    RunPlan,
    RunPlanRevision,
    VerificationReport,
)
from ai4s_agent.storage import ProjectStorage


FALLBACK_PARSER_TASKS = ["parse_document_pdfplumber"]
GRANULAR_LITERATURE_WORKFLOW_TASKS = [
    "parse_document",
    "index_corpus",
    "retrieve_evidence",
    "extract_records",
    "normalize_extracted_units",
    "track_citation_provenance",
    "merge_extracted_records",
    "evaluate_extraction_benchmark",
    "confirm_extracted_dataset",
]


class RecoveryAgent:
    """Builds audited dry-run recovery plans from verifier findings."""

    def __init__(self, registry: AtomicTaskRegistry | None = None) -> None:
        self.registry = registry or AtomicTaskRegistry()

    def propose_revision(
        self,
        *,
        request: ReplanRequest,
        previous_plan: RunPlan,
        verification_report: VerificationReport | None = None,
    ) -> RunPlanRevision:
        categories = self._finding_categories(verification_report)
        trusted_available = self._trusted_available_artifacts(request, previous_plan)
        requested_tasks = list(previous_plan.requested_tasks)
        requested_tasks = [
            task_id
            for task_id in requested_tasks
            if task_id not in self._trusted_upstream_tasks(previous_plan, trusted_available)
        ]
        recovery_actions: list[str] = []
        approvals_required: list[str] = []
        questions: list[PlanQuestion] = []
        external_network_added = False

        if self._needs_parser_fallback(request, categories):
            requested_tasks = self._expand_monolithic_literature_workflow(requested_tasks)
            requested_tasks = self._replace_requested(
                requested_tasks,
                remove={"parse_document"},
                prepend=FALLBACK_PARSER_TASKS,
            )
            recovery_actions.append("parser_fallback")

        if self._needs_data_mining_recovery(request, categories):
            requested_tasks = self._replace_requested(
                requested_tasks,
                remove=set(),
                prepend=["prepare_literature_corpus_sources", "acquire_literature_sources"],
            )
            recovery_actions.extend(["expand_literature_query", "retry_acquisition", "request_user_provided_pdfs"])
            approvals_required.append("external_network_action")
            external_network_added = True
            questions.append(
                PlanQuestion(
                    question_id="q_literature_sources",
                    prompt="Which additional DOI, URL, or PDF sources should the agent use for recovery?",
                    reason="The current evidence set is insufficient, so expanding acquisition requires explicit source scope.",
                    choices=["add_doi_or_url_sources", "upload_pdfs", "retry_with_expanded_query"],
                    blocks_execution=True,
                )
            )

        if self._needs_modeling_baseline_only(request, categories):
            requested_tasks = self._replace_requested(
                requested_tasks,
                remove={"train_model", "generate_candidates", "predict_candidates", "filter_rank", "render_report"},
                prepend=["check_trainability", "run_baseline"],
            )
            recovery_actions.extend(["reduce_target_properties", "request_more_data", "run_baseline_only"])
        elif self._needs_model_retrain(request, categories):
            requested_tasks = self._append_unique(requested_tasks, ["train_model"])
            recovery_actions.extend(["switch_backend", "adjust_split", "retrain_model"])

        revised_plan = expand_run_plan(
            run_id=request.run_id,
            requested_tasks=requested_tasks,
            available_artifacts=trusted_available,
            registry=self.registry,
        )
        diff = diff_run_plans(previous_plan, revised_plan)
        added_high_risk = self._high_risk_tasks(diff.added_tasks)
        removed_high_risk = self._high_risk_tasks(diff.removed_tasks)
        for task_id in added_high_risk:
            approvals_required.extend(self._gates_for_task(task_id))
        if removed_high_risk:
            approvals_required.append("high_risk_downgrade_review")

        approvals_required = self._dedup(approvals_required)
        revision_id = self._revision_id(request, previous_plan, revised_plan)
        user_approval_required = bool(approvals_required or added_high_risk or external_network_added)
        return RunPlanRevision(
            revision_id=revision_id,
            project_id=request.project_id,
            run_id=request.run_id,
            created_at=now_iso(),
            previous_plan=previous_plan,
            revised_plan=revised_plan,
            diff=diff,
            reason=request.reason,
            recovery_actions=self._dedup(recovery_actions) or ["retry_same_plan"],
            approvals_required=approvals_required,
            questions=questions,
            user_approval_required=user_approval_required,
            high_risk_added=bool(added_high_risk),
            external_network_added=external_network_added,
            removed_high_risk_tasks=removed_high_risk,
            executable=False,
        )

    def write_revision(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        revision: RunPlanRevision,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        revision_stem = self._safe_revision_stem(revision.revision_id)
        json_path = write_json(run_dir / f"{revision_stem}_run_plan_revision.json", revision.model_dump(mode="json"))
        md_path = run_dir / f"{revision_stem}_run_plan_revision.md"
        md_path.write_text(self._render_markdown(revision), encoding="utf-8")
        index_path = self._write_revision_index(run_dir, revision, json_path.name, md_path.name)
        storage.register_artifact_path(project_id, run_id, "run_plan_revision_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "run_plan_revision_md", md_path.name)
        storage.register_artifact_path(project_id, run_id, "run_plan_revisions_index", index_path.name)
        return json_path, md_path

    @staticmethod
    def _finding_categories(report: VerificationReport | None) -> set[str]:
        if report is None:
            return set()
        return {finding.category for finding in report.findings}

    @staticmethod
    def _trusted_available_artifacts(request: ReplanRequest, previous_plan: RunPlan) -> list[str]:
        if request.available_artifacts:
            return RecoveryAgent._dedup(request.available_artifacts)
        planned_outputs: set[str] = set()
        for task in previous_plan.tasks:
            planned_outputs.update(task.output_artifacts)
        return sorted(set(previous_plan.available_artifacts) - planned_outputs)

    @staticmethod
    def _trusted_upstream_tasks(previous_plan: RunPlan, trusted_available: list[str]) -> set[str]:
        output_to_task: dict[str, str] = {}
        dependencies_by_task: dict[str, list[str]] = {}
        for task in previous_plan.tasks:
            dependencies_by_task[task.task_id] = list(task.depends_on)
            for artifact in task.output_artifacts:
                output_to_task.setdefault(artifact, task.task_id)

        removable: set[str] = set()

        def collect(task_id: str) -> None:
            if task_id in removable:
                return
            removable.add(task_id)
            for dependency in dependencies_by_task.get(task_id, []):
                collect(dependency)

        for artifact in trusted_available:
            producer = output_to_task.get(artifact)
            if producer:
                collect(producer)
        return removable

    @staticmethod
    def _needs_parser_fallback(request: ReplanRequest, categories: set[str]) -> bool:
        text = " ".join([request.failed_stage, request.failure_category, request.reason]).lower()
        if request.failed_stage in {"parse_document", "parse_document_mineru"}:
            return True
        parser_terms = r"\b(parse|parser|mineru|pdf|table|tables)\b"
        if request.failed_stage in {"", "extract_records"} and re.search(parser_terms, text):
            return True
        return request.failed_stage == "extract_records" and bool(
            categories & {"empty_extraction", "low_confidence", "malformed_report"}
        )

    @staticmethod
    def _needs_data_mining_recovery(request: ReplanRequest, categories: set[str]) -> bool:
        text = " ".join([request.failed_stage, request.failure_category, request.reason]).lower()
        return (
            request.failed_stage in {"retrieve_evidence", "acquire_literature_sources"}
            or "no evidence" in text
            or "doi" in text
            or "url" in text
            or bool(categories & {"missing_provenance"})
        )

    @staticmethod
    def _needs_modeling_baseline_only(request: ReplanRequest, categories: set[str]) -> bool:
        text = " ".join([request.failed_stage, request.failure_category, request.reason]).lower()
        return "poor_trainability" in categories or "not ready" in text or "more data" in text

    @staticmethod
    def _needs_model_retrain(request: ReplanRequest, categories: set[str]) -> bool:
        constraints = " ".join(request.new_constraints).lower()
        text = " ".join([request.failed_stage, request.failure_category, request.reason, constraints]).lower()
        return "abnormal_model_metrics" in categories and ("retrain" in text or "backend" in text)

    def _gates_for_task(self, task_id: str) -> list[str]:
        spec = self.registry.get(task_id)
        gates = list(spec.gates)
        if spec.risk_level == RiskLevel.HIGH and not gates:
            gates.append("gate_3_train_config")
        return gates

    def _high_risk_tasks(self, task_ids: list[str]) -> list[str]:
        high_risk: list[str] = []
        for task_id in task_ids:
            try:
                spec = self.registry.get(task_id)
            except ValueError:
                continue
            if spec.risk_level == RiskLevel.HIGH:
                high_risk.append(task_id)
        return high_risk

    @staticmethod
    def _replace_requested(
        requested_tasks: list[str],
        *,
        remove: set[str],
        prepend: list[str],
    ) -> list[str]:
        return RecoveryAgent._dedup(prepend + [task for task in requested_tasks if task not in remove])

    @staticmethod
    def _expand_monolithic_literature_workflow(requested_tasks: list[str]) -> list[str]:
        expanded: list[str] = []
        for task_id in requested_tasks:
            if task_id == "literature_to_dataset_workflow":
                expanded.extend(GRANULAR_LITERATURE_WORKFLOW_TASKS)
            else:
                expanded.append(task_id)
        return RecoveryAgent._dedup(expanded)

    @staticmethod
    def _append_unique(requested_tasks: list[str], extra: list[str]) -> list[str]:
        return RecoveryAgent._dedup(requested_tasks + extra)

    @staticmethod
    def _dedup(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = str(value).strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _revision_id(request: ReplanRequest, previous_plan: RunPlan, revised_plan: RunPlan) -> str:
        digest = hashlib.sha1(
            (
                request.model_dump_json()
                + previous_plan.model_dump_json()
                + revised_plan.model_dump_json()
            ).encode("utf-8")
        ).hexdigest()[:10]
        return f"rev-{request.run_id}-{digest}"

    @staticmethod
    def _safe_revision_stem(revision_id: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(revision_id or "").strip()).strip("._-")
        return clean or "revision"

    @staticmethod
    def _write_revision_index(
        run_dir: Path,
        revision: RunPlanRevision,
        json_name: str,
        md_name: str,
    ) -> Path:
        index_path = run_dir / "run_plan_revisions.json"
        payload = RecoveryAgent._read_json_object(index_path)
        records_raw = payload.get("revisions", [])
        records = [record for record in records_raw if isinstance(record, dict)]
        records = [record for record in records if str(record.get("revision_id") or "") != revision.revision_id]
        records.append(
            {
                "revision_id": revision.revision_id,
                "created_at": revision.created_at,
                "reason": revision.reason,
                "json": json_name,
                "markdown": md_name,
                "approvals_required": revision.approvals_required,
                "user_approval_required": revision.user_approval_required,
            }
        )
        return write_json(
            index_path,
            {
                "run_id": revision.run_id,
                "latest_revision_id": revision.revision_id,
                "revisions": records,
            },
        )

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    @staticmethod
    def _render_markdown(revision: RunPlanRevision) -> str:
        lines = [
            "# Run Plan Revision",
            "",
            f"- Run: `{revision.run_id}`",
            f"- Project: `{revision.project_id}`",
            f"- User approval required: `{revision.user_approval_required}`",
            f"- Reason: {revision.reason}",
            "",
            "## Recovery Actions",
        ]
        lines.extend(f"- {action}" for action in revision.recovery_actions)
        lines.extend(["", "## Plan Diff", ""])
        lines.append(f"- Added tasks: {', '.join(revision.diff.added_tasks) or 'none'}")
        lines.append(f"- Removed tasks: {', '.join(revision.diff.removed_tasks) or 'none'}")
        lines.append(f"- Approvals required: {', '.join(revision.approvals_required) or 'none'}")
        return "\n".join(lines) + "\n"
