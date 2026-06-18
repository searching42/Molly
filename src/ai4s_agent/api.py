from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from pydantic import ValidationError
from werkzeug.utils import secure_filename

import ai4s_agent.adapters as adapter_exports
from ai4s_agent._utils import PROTECTED_PAYLOAD_KEYS, now_iso, strict_bool, write_json
from ai4s_agent.agents.conversation import ConversationAgent
from ai4s_agent.agents.generation import GenerationAgent
from ai4s_agent.agents.modeling import ModelingAgent
from ai4s_agent.agents.observer import ObserverAgent
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.agents.recovery import RecoveryAgent
from ai4s_agent.agents.report import ReportAgent
from ai4s_agent.agents.research import ResearchAgent
from ai4s_agent.agents.verifier import VerifierAgent
from ai4s_agent.deployment import assess_multi_user_deployment
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.job_manager import JobManager
from ai4s_agent.llm_provider import LLMProvider, LLMProviderError, create_llm_provider
from ai4s_agent.memory import PermissionPolicy, ProjectMemory
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.planner import AtomicTaskRegistry, build_plan, diff_run_plans, expand_run_plan
from ai4s_agent.remote_worker import RemoteWorkerRegistry
from ai4s_agent.schemas import (
    AssetPromotionRecord,
    BackgroundJobBudget,
    GateName,
    LLMProviderConfig,
    LiteratureCorpusSource,
    ProjectMemoryRecord,
    RemoteWorkerConfig,
    RemoteWorkerRequest,
    ReplanRequest,
    ResearchSourceProposal,
    RunPlan,
    RunStatus,
    StageHistoryItem,
    StageState,
    VerificationReport,
)
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.ui_cards import (
    build_agent_review_card,
    build_data_confirmation_card,
    build_report_preview,
    build_run_confirmation_card,
    build_stage_timeline,
)


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]
ALLOWED_EXTENSIONS = {"csv", "json", "sdf", "mol", "smi"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _copy_upload_stream(src: Any, dest: Any, *, max_bytes: int) -> None:
    total = 0
    while True:
        chunk = src.read(UPLOAD_COPY_CHUNK_BYTES)
        if not chunk:
            return
        total += len(chunk)
        if max_bytes > 0 and total > max_bytes:
            raise ValueError(f"upload exceeds size limit: {max_bytes} bytes")
        dest.write(chunk)


def _request_json_object() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    return payload


def _workspace_from_config(base_runs_dir: Path | None, workspace_dir: Path | None) -> Path:
    if workspace_dir is not None:
        return Path(workspace_dir).resolve()
    if base_runs_dir is None:
        return DEFAULT_WORKSPACE.resolve()
    runs_path = Path(base_runs_dir).resolve()
    if runs_path.name == "runs":
        return runs_path.parent.resolve()
    return runs_path


def register_routes(app: Flask, base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> None:
    runs = Path(base_runs_dir or DEFAULT_RUNS_DIR).resolve()
    workspace = _workspace_from_config(base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
    orch = Orchestrator(base_runs_dir=runs)
    jobs = JobManager(runs_dir=runs)
    projects = ProjectStorage(workspace_dir=workspace)
    project_memory = ProjectMemory(workspace_dir=workspace)
    remote_workers = RemoteWorkerRegistry(workspace_dir=workspace)
    permissions = PermissionPolicy()

    @app.get("/")
    def index():
        return render_template("index.html", gate_names=[gate.value for gate in GateName])

    @app.get("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    # --- Plan ---

    @app.post("/api/plan")
    def create_plan():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        if jobs.get_job(run_id):
            return jsonify({"ok": False, "error": f"job already active: {run_id}"}), 409
        try:
            status = orch.start_run(run_id=run_id, prompt=prompt)
            jobs.start_job(run_id, details={"gate": status.get("gate")})
        except ValueError as exc:
            message = str(exc)
            status_code = 409 if "already active" in message or "already exists" in message else 400
            return jsonify({"ok": False, "error": message}), status_code
        except KeyError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, **status})

    @app.post("/api/run-plan/expand")
    def expand_plan_preview():
        payload = request.get_json(silent=True) or {}
        try:
            run_plan = _expand_plan_from_payload(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "run_plan": run_plan.model_dump(mode="json")})

    @app.post("/api/run-plan/diff")
    def diff_plan_preview():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "preview").strip() or "preview"
        try:
            before = _run_plan_from_payload(payload.get("before"), run_id=run_id)
            after = _run_plan_from_payload(payload.get("after"), run_id=run_id)
            diff = diff_run_plans(before, after)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "before": before.model_dump(mode="json"),
                "after": after.model_dump(mode="json"),
                "diff": diff.model_dump(mode="json"),
            }
        )

    @app.post("/api/run-plan/regenerate")
    def regenerate_plan_preview():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        if not run_id or not prompt:
            return jsonify({"ok": False, "error": "run_id and prompt required"}), 400
        plan = build_plan(run_id=run_id, prompt=prompt)
        requested = payload.get("requested_tasks")
        if not isinstance(requested, list) or not requested:
            return jsonify({"ok": False, "error": "requested_tasks required"}), 400
        try:
            run_plan = expand_run_plan(
                run_id=run_id,
                requested_tasks=[str(task) for task in requested if str(task).strip()],
                available_artifacts=_string_list(payload.get("available_artifacts")),
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "plan": plan.model_dump(mode="json"),
                "run_plan": run_plan.model_dump(mode="json"),
            }
        )

    @app.post("/api/run-plan/execute")
    def execute_run_plan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        run_plan_payload = payload.get("run_plan")
        if not isinstance(run_plan_payload, dict):
            return jsonify({"ok": False, "error": "run_plan object required"}), 400
        input_artifacts = payload.get("input_artifacts", {})
        if input_artifacts is None:
            input_artifacts = {}
        if not isinstance(input_artifacts, dict):
            return jsonify({"ok": False, "error": "input_artifacts must be an object"}), 400
        task_options = payload.get("task_options", {})
        if task_options is None:
            task_options = {}
        if not isinstance(task_options, dict):
            return jsonify({"ok": False, "error": "task_options must be an object"}), 400
        run_plan: RunPlan | None = None
        try:
            run_plan = RunPlan.model_validate(run_plan_payload)
            jobs.add_log(run_plan.run_id, "INFO", "run_plan", "RunPlan execution started")
            execution = RunPlanExecutor(storage=projects).execute(
                project_id=project_id,
                run_plan=run_plan,
                input_artifacts={str(k): str(v) for k, v in input_artifacts.items()},
                task_options=_task_options(task_options),
            )
            _log_run_plan_execution_result(jobs, run_plan.run_id, execution)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            if run_plan is not None:
                jobs.add_log(run_plan.run_id, "ERROR", "run_plan", f"RunPlan execution failed: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "execution": execution})

    @app.post("/api/run-plan/resume")
    def resume_run_plan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        actor = str(payload.get("actor") or payload.get("approved_by") or "").strip()
        if not actor:
            return jsonify({"ok": False, "error": "actor required"}), 400
        run_plan_payload = payload.get("run_plan")
        if not isinstance(run_plan_payload, dict):
            return jsonify({"ok": False, "error": "run_plan object required"}), 400
        approved_gates = payload.get("approved_gates", [])
        if approved_gates is None:
            approved_gates = []
        if not isinstance(approved_gates, list):
            return jsonify({"ok": False, "error": "approved_gates must be a list"}), 400
        input_artifacts = payload.get("input_artifacts", {})
        if input_artifacts is None:
            input_artifacts = {}
        if not isinstance(input_artifacts, dict):
            return jsonify({"ok": False, "error": "input_artifacts must be an object"}), 400
        task_options = payload.get("task_options", {})
        if task_options is None:
            task_options = {}
        if not isinstance(task_options, dict):
            return jsonify({"ok": False, "error": "task_options must be an object"}), 400
        run_plan: RunPlan | None = None
        try:
            run_plan = RunPlan.model_validate(run_plan_payload)
            jobs.add_log(run_plan.run_id, "INFO", "run_plan", "RunPlan resume requested")
            execution = RunPlanExecutor(storage=projects).resume_after_gate(
                project_id=project_id,
                run_plan=run_plan,
                approved_gates=[str(gate) for gate in approved_gates],
                actor=actor,
                note=str(payload.get("note") or ""),
                input_artifacts={str(k): str(v) for k, v in input_artifacts.items()},
                task_options=_task_options(task_options),
            )
            _log_run_plan_execution_result(jobs, run_plan.run_id, execution)
        except (ValidationError, ValueError, FileNotFoundError) as exc:
            if run_plan is not None:
                jobs.add_log(run_plan.run_id, "ERROR", "run_plan", f"RunPlan resume failed: {exc}")
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "execution": execution})

    @app.post("/api/agent/plan-proposal")
    def agent_plan_proposal():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
        if not run_id or not goal:
            return jsonify({"ok": False, "error": "run_id and goal required"}), 400
        try:
            provider = _llm_provider_from_payload(payload)
        except (LLMProviderError, ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        try:
            memory_records = project_memory.list_project_records(project_id) if project_id else []
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        proposal = PlannerAgent(provider=provider, memory_records=memory_records).propose_plan(
            run_id=run_id,
            goal=goal,
            available_artifacts=_string_list(payload.get("available_artifacts")),
        )
        return jsonify({"ok": True, "proposal": proposal.model_dump(mode="json")})

    @app.post("/api/agent/replan")
    def agent_replan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            return jsonify({"ok": False, "error": "request object required"}), 400
        try:
            replan_request = ReplanRequest.model_validate(request_payload)
            previous_plan = _run_plan_from_payload(payload.get("previous_plan"), run_id=replan_request.run_id)
            report_payload = payload.get("verification_report")
            if report_payload is not None and not isinstance(report_payload, dict):
                return jsonify({"ok": False, "error": "verification_report must be an object"}), 400
            verification_report = (
                VerificationReport.model_validate(report_payload)
                if isinstance(report_payload, dict)
                else None
            )
            recovery = RecoveryAgent()
            revision = recovery.propose_revision(
                request=replan_request,
                previous_plan=previous_plan,
                verification_report=verification_report,
            )
            outputs: dict[str, str] = {}
            if replan_request.project_id:
                revision_json, revision_md = recovery.write_revision(
                    projects,
                    replan_request.project_id,
                    replan_request.run_id,
                    revision,
                )
                outputs = {
                    "run_plan_revision_json": str(revision_json),
                    "run_plan_revision_md": str(revision_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "revision": revision.model_dump(mode="json"), "outputs": outputs})

    @app.post("/api/agent/research-sources")
    def agent_research_sources():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not goal:
            return jsonify({"ok": False, "error": "run_id and goal required"}), 400
        seed_sources_raw = payload.get("seed_sources", [])
        if not isinstance(seed_sources_raw, list):
            return jsonify({"ok": False, "error": "seed_sources must be a list"}), 400
        if any(not isinstance(item, dict) for item in seed_sources_raw):
            return jsonify({"ok": False, "error": "seed_sources entries must be objects"}), 400
        try:
            seed_sources = [
                LiteratureCorpusSource.model_validate(item)
                for item in seed_sources_raw
            ]
            research = ResearchAgent()
            proposal = research.propose_sources(run_id=run_id, goal=goal, seed_sources=seed_sources)
            outputs: dict[str, str] = {}
            if project_id:
                proposal_json, proposal_md = research.write_proposal(projects, project_id, run_id, proposal)
                outputs = {
                    "research_source_proposal_json": str(proposal_json),
                    "research_source_proposal_md": str(proposal_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "proposal": proposal.model_dump(mode="json"), "outputs": outputs})

    @app.post("/api/agent/conversation/research-sources")
    def agent_conversation_research_sources():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        messages = payload.get("messages")
        if not run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            research_payload = ConversationAgent().prepare_research_source_payload(
                run_id=run_id,
                project_id=project_id or None,
                messages=messages,
            )
            goal = str(research_payload.get("goal") or "").strip()
            if not goal:
                return jsonify({"ok": False, "error": "conversation research goal required"}), 400
            seed_sources = [
                LiteratureCorpusSource.model_validate(item)
                for item in research_payload.get("seed_sources", [])
            ]
            research = ResearchAgent()
            proposal = research.propose_sources(run_id=run_id, goal=goal, seed_sources=seed_sources)
            outputs: dict[str, str] = {}
            if project_id:
                proposal_json, proposal_md = research.write_proposal(projects, project_id, run_id, proposal)
                outputs = {
                    "research_source_proposal_json": str(proposal_json),
                    "research_source_proposal_md": str(proposal_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "research_source_payload": research_payload,
                "proposal": proposal.model_dump(mode="json"),
                "outputs": outputs,
            }
        )

    @app.post("/api/agent/research-acquisition/prepare")
    def agent_research_acquisition_prepare():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        proposal_raw = payload.get("proposal")
        selected_sources_raw = payload.get("selected_sources")
        if proposal_raw is not None and not isinstance(proposal_raw, dict):
            return jsonify({"ok": False, "error": "proposal must be an object"}), 400
        if selected_sources_raw is not None and not isinstance(selected_sources_raw, list):
            return jsonify({"ok": False, "error": "selected_sources must be a list"}), 400
        output_dir = str(payload.get("output_dir") or "").strip()
        if not output_dir and project_id:
            output_dir = str(projects.run_dir(project_id, run_id) / "research_acquisition")
        try:
            proposal = ResearchSourceProposal.model_validate(proposal_raw) if proposal_raw is not None else None
            selected_sources = [
                LiteratureCorpusSource.model_validate(item)
                for item in (selected_sources_raw or [])
            ]
            preparation = ResearchAgent().prepare_acquisition(
                run_id=run_id,
                proposal=proposal,
                selected_sources=selected_sources,
                goal=str(payload.get("goal") or "").strip(),
                output_dir=output_dir,
                local_mirror=payload.get("local_mirror") if isinstance(payload.get("local_mirror"), dict) else None,
                user_confirmed_external_acquisition=_as_bool(payload.get("user_confirmed_external_acquisition")),
            )
            outputs: dict[str, str] = {}
            if project_id:
                preparation_json, preparation_md = ResearchAgent().write_acquisition_preparation(
                    projects,
                    project_id,
                    run_id,
                    preparation,
                )
                outputs = {
                    "research_acquisition_preparation_json": str(preparation_json),
                    "research_acquisition_preparation_md": str(preparation_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "preparation": preparation.model_dump(mode="json"), "outputs": outputs})

    @app.post("/api/agent/conversation/modeling-payload")
    def agent_conversation_modeling_payload():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        messages = payload.get("messages")
        if not run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            modeling_payload = ConversationAgent().prepare_modeling_plan_payload(
                run_id=run_id,
                project_id=project_id or None,
                messages=messages,
                available_inputs=payload.get("available_inputs"),
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "modeling_plan_payload": modeling_payload})

    @app.post("/api/agent/conversation/next-turn")
    def agent_conversation_next_turn():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        messages = payload.get("messages")
        if not run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            decision = ConversationAgent().decide_next_turn(
                run_id=run_id,
                project_id=project_id or None,
                messages=messages,
                project_memory=payload.get("project_memory"),
                previous_diagnostics=payload.get("previous_diagnostics"),
                available_inputs=payload.get("available_inputs"),
            )
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "decision": decision.model_dump(mode="json")})

    @app.post("/api/agent/modeling-plan")
    def agent_modeling_plan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not goal:
            return jsonify({"ok": False, "error": "run_id and goal required"}), 400
        trainability_report = payload.get("trainability_report")
        backend_recommendation = payload.get("backend_recommendation")
        model_metrics = payload.get("model_metrics")
        property_id = str(payload.get("property_id") or "").strip()
        cited_target_evidence = payload.get("cited_target_evidence", [])
        if cited_target_evidence is None:
            cited_target_evidence = []
        if not isinstance(cited_target_evidence, list):
            return jsonify({"ok": False, "error": "cited_target_evidence must be a list"}), 400
        if any(not isinstance(item, dict) for item in cited_target_evidence):
            return jsonify({"ok": False, "error": "cited_target_evidence entries must be objects"}), 400
        project_memory = payload.get("project_memory")
        previous_diagnostics = payload.get("previous_diagnostics", [])
        if project_memory is not None and not isinstance(project_memory, dict):
            return jsonify({"ok": False, "error": "project_memory must be an object"}), 400
        if previous_diagnostics is None:
            previous_diagnostics = []
        if not isinstance(previous_diagnostics, list):
            return jsonify({"ok": False, "error": "previous_diagnostics must be a list"}), 400
        if any(not isinstance(item, dict) for item in previous_diagnostics):
            return jsonify({"ok": False, "error": "previous_diagnostics entries must be objects"}), 400
        for label, value in (
            ("trainability_report", trainability_report),
            ("backend_recommendation", backend_recommendation),
            ("model_metrics", model_metrics),
        ):
            if value is not None and not isinstance(value, dict):
                return jsonify({"ok": False, "error": f"{label} must be an object"}), 400
        target_brief = None
        try:
            modeling = ModelingAgent()
            proposal = modeling.propose_modeling_plan(
                run_id=run_id,
                goal=goal,
                trainability_report=trainability_report,
                backend_recommendation=backend_recommendation,
                model_metrics=model_metrics,
            )
            if property_id or cited_target_evidence:
                if not property_id:
                    property_id = _first_trainability_property(trainability_report) or "default"
                target_evidence = ResearchAgent().prepare_target_evidence_items(
                    goal=goal,
                    property_id=property_id,
                    cited_summaries=cited_target_evidence,
                    user_approved_external_search=_as_bool(payload.get("user_approved_external_search")),
                )
                available_inputs_raw = payload.get("available_inputs")
                available_inputs = (
                    set(_string_list(available_inputs_raw))
                    if available_inputs_raw is not None
                    else None
                )
                target_brief = modeling.prepare_target_modeling_brief(
                    run_id=run_id,
                    goal=goal,
                    property_id=property_id,
                    trainability_report=trainability_report,
                    project_memory=project_memory,
                    previous_diagnostics=previous_diagnostics,
                    allow_external_search=_as_bool(payload.get("user_approved_external_search")),
                    available_inputs=available_inputs,
                    target_evidence=target_evidence,
                )
            outputs: dict[str, str] = {}
            if project_id:
                proposal_json, proposal_md = modeling.write_proposal(projects, project_id, run_id, proposal)
                outputs = {
                    "modeling_plan_proposal_json": str(proposal_json),
                    "modeling_plan_proposal_md": str(proposal_md),
                }
                if target_brief is not None:
                    brief_json, brief_md = modeling.write_target_modeling_brief(
                        projects, project_id, run_id, target_brief
                    )
                    safe_property = ModelingAgent._safe_property_stem(target_brief.property_id)
                    outputs.update(
                        {
                            f"target_modeling_brief_{safe_property}_json": str(brief_json),
                            f"target_modeling_brief_{safe_property}_md": str(brief_md),
                        }
                    )
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        response = {"ok": True, "proposal": proposal.model_dump(mode="json"), "outputs": outputs}
        if target_brief is not None:
            response["target_modeling_brief"] = target_brief.model_dump(mode="json")
        return jsonify(response)

    @app.post("/api/agent/model-package-review")
    def agent_model_package_review():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
        model_manifest = payload.get("model_manifest")
        domain_model_manifest = payload.get("domain_model_manifest")
        diagnostics_report = payload.get("model_diagnostics_report")
        if not run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        for label, value in (
            ("model_manifest", model_manifest),
            ("domain_model_manifest", domain_model_manifest),
            ("model_diagnostics_report", diagnostics_report),
        ):
            if value is not None and not isinstance(value, dict):
                return jsonify({"ok": False, "error": f"{label} must be an object"}), 400
        try:
            modeling = ModelingAgent()
            review = modeling.review_model_package(
                run_id=run_id,
                goal=goal,
                model_manifest=model_manifest,
                domain_model_manifest=domain_model_manifest,
                diagnostics_report=diagnostics_report,
            )
            outputs: dict[str, str] = {}
            if project_id:
                review_json, review_md = modeling.write_model_package_review(projects, project_id, run_id, review)
                outputs = {
                    "model_package_review_json": str(review_json),
                    "model_package_review_md": str(review_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "review": review.model_dump(mode="json"), "outputs": outputs})

    @app.post("/api/agent/generation-plan")
    def agent_generation_plan():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not goal:
            return jsonify({"ok": False, "error": "run_id and goal required"}), 400
        generation_request = payload.get("generation_request")
        if generation_request is not None and not isinstance(generation_request, dict):
            return jsonify({"ok": False, "error": "generation_request must be an object"}), 400
        try:
            generation = GenerationAgent()
            proposal = generation.propose_generation_plan(
                run_id=run_id,
                goal=goal,
                generation_request=generation_request,
            )
            outputs: dict[str, str] = {}
            if project_id:
                proposal_json, proposal_md = generation.write_proposal(projects, project_id, run_id, proposal)
                outputs = {
                    "generation_strategy_proposal_json": str(proposal_json),
                    "generation_strategy_proposal_md": str(proposal_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "proposal": proposal.model_dump(mode="json"), "outputs": outputs})

    @app.post("/api/agent/report-summary")
    def agent_report_summary():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        run_id = str(payload.get("run_id") or "").strip()
        goal = str(payload.get("goal") or payload.get("prompt") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        if not run_id or not goal:
            return jsonify({"ok": False, "error": "run_id and goal required"}), 400
        object_fields = (
            "observation",
            "verification_report",
            "run_plan_revision",
            "research_proposal",
            "modeling_proposal",
            "generation_proposal",
        )
        for field in object_fields:
            value = payload.get(field)
            if value is not None and not isinstance(value, dict):
                return jsonify({"ok": False, "error": f"{field} must be an object"}), 400
        try:
            reporter = ReportAgent()
            proposal = reporter.synthesize_run(
                run_id=run_id,
                goal=goal,
                observation=payload.get("observation"),
                verification_report=payload.get("verification_report"),
                run_plan_revision=payload.get("run_plan_revision"),
                research_proposal=payload.get("research_proposal"),
                modeling_proposal=payload.get("modeling_proposal"),
                generation_proposal=payload.get("generation_proposal"),
            )
            outputs: dict[str, str] = {}
            if project_id:
                proposal_json, proposal_md = reporter.write_proposal(projects, project_id, run_id, proposal)
                outputs = {
                    "report_synthesis_proposal_json": str(proposal_json),
                    "report_synthesis_proposal_md": str(proposal_md),
                }
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "proposal": proposal.model_dump(mode="json"), "outputs": outputs})

    @app.post("/api/agent/review-card")
    def agent_review_card():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        try:
            card = build_agent_review_card(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "card": card})

    @app.post("/api/agent/decision-card")
    def agent_decision_card():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        run_id = str(payload.get("run_id") or "").strip()
        question = str(payload.get("question") or "").strip()
        if not project_id or not run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        try:
            state = projects.read_stage_state(project_id, run_id)
            log_tail = jobs.get_logs(run_id, limit=20)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        card = _build_agent_decision_card(
            project_id=project_id,
            run_id=run_id,
            state=state,
            log_tail=log_tail,
            question=question,
        )
        return jsonify({"ok": True, "card": card, "log_tail": log_tail})

    @app.get("/api/workers")
    def list_remote_workers():
        include_disabled = _as_bool(request.args.get("include_disabled"))
        workers = remote_workers.list_workers(include_disabled=include_disabled)
        return jsonify({"ok": True, "workers": [worker.model_dump(mode="json") for worker in workers]})

    @app.post("/api/workers")
    def save_remote_worker():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            worker = remote_workers.save_worker(RemoteWorkerConfig.model_validate(payload))
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "worker": worker.model_dump(mode="json")})

    @app.post("/api/workers/assignment")
    def plan_remote_worker_assignment():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            assignment = remote_workers.plan_assignment(RemoteWorkerRequest.model_validate(payload))
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "assignment": assignment.model_dump(mode="json")})

    @app.get("/api/deployment/multi-user-readiness")
    def multi_user_readiness():
        readiness = assess_multi_user_deployment(workspace_dir=workspace, runs_dir=runs)
        return jsonify({"ok": True, "readiness": readiness.model_dump(mode="json")})

    @app.get("/api/atomic-tasks")
    def list_atomic_tasks():
        registry = AtomicTaskRegistry()
        return jsonify(
            {
                "ok": True,
                "tasks": [task.model_dump(mode="json") for task in registry.list_tasks()],
            }
        )

    @app.post("/api/data-confirmation-card")
    def data_confirmation_card():
        payload = request.get_json(silent=True) or {}
        dataset_path_raw = str(payload.get("dataset_path") or "").strip()
        if not dataset_path_raw:
            return jsonify({"ok": False, "error": "dataset_path required"}), 400
        try:
            card = build_data_confirmation_card(payload, base=workspace)
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "card": card})

    @app.post("/api/run-confirmation-card")
    def run_confirmation_card():
        payload = request.get_json(silent=True) or {}
        try:
            card = build_run_confirmation_card(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "card": card})

    @app.post("/api/permissions/resolve")
    def resolve_permission():
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action") or "").strip()
        if not action:
            return jsonify({"ok": False, "error": "action required"}), 400
        decision = permissions.decide(
            action,
            project_id=str(payload.get("project_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            project_approved=_as_bool(payload.get("project_approved")),
            confirmed=_as_bool(payload.get("confirmed")),
            actor=str(payload.get("actor") or payload.get("approved_by") or ""),
        )
        return jsonify({"ok": True, **decision.model_dump(mode="json")})

    @app.post("/api/gates/approve")
    def approve_gate():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        gate_raw = str(payload.get("gate") or "").strip()
        actor = str(payload.get("actor") or "").strip()
        note = str(payload.get("note") or "").strip()
        if not run_id or not gate_raw or not actor:
            return jsonify({"ok": False, "error": "run_id, gate, and actor required"}), 400
        try:
            gate = GateName(gate_raw)
        except ValueError:
            return jsonify({"ok": False, "error": f"unknown gate: {gate_raw}"}), 400
        try:
            status = orch.approve_gate(run_id=run_id, gate=gate, actor=actor, note=note)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_log(run_id, "INFO", "gate", f"Gate {gate_raw} approved by {actor}")
        return jsonify({"ok": True, **status})

    @app.get("/api/runs/<run_id>")
    def run_status(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        project_id = str(request.args.get("project_id") or "").strip()
        try:
            status = _read_run_status(orch, projects, clean_run_id, project_id=project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        job = jobs.get_job(clean_run_id)
        return jsonify({"ok": True, "job": job, **status})

    @app.post("/api/adapters/execute")
    def execute_adapter():
        payload = request.get_json(silent=True) or {}
        run_id = str(payload.get("run_id") or "").strip()
        adapter_name = str(payload.get("adapter") or "").strip()
        adapter_payload = payload.get("payload")
        if not run_id or not adapter_name:
            return jsonify({"ok": False, "error": "run_id and adapter required"}), 400
        if not isinstance(adapter_payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        adapter = getattr(adapter_exports, adapter_name, None)
        if not callable(adapter):
            return jsonify({"ok": False, "error": f"unknown adapter: {adapter_name}"}), 400
        try:
            policy = _adapter_execution_policy(adapter_name, adapter_payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if policy is None:
            return jsonify({"ok": False, "error": f"adapter is not registered for direct execution: {adapter_name}"}), 400
        action, required_gates = policy
        project_id = str(payload.get("project_id") or adapter_payload.get("project_id") or "")
        actor = str(
            payload.get("actor")
            or payload.get("approved_by")
            or adapter_payload.get("actor")
            or adapter_payload.get("approved_by")
            or ""
        )
        decision = permissions.decide(
            action,
            project_id=project_id,
            run_id=run_id,
            project_approved=_as_bool(payload.get("project_approved"))
            or _as_bool(adapter_payload.get("project_approved")),
            confirmed=_as_bool(payload.get("confirmed")) or _as_bool(adapter_payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "adapter execution requires permission",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403
        if required_gates:
            return jsonify(
                {
                    "ok": False,
                    "error": "gated adapter execution requires run-plan snapshot approval",
                    "required_gates": required_gates,
                    "permission": decision.model_dump(mode="json"),
                }
            ), 400
        try:
            snapshot_required = _adapter_requires_snapshot_for_execute(adapter_name, adapter_payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if snapshot_required:
            return jsonify(
                {
                    "ok": False,
                    "error": "this adapter requires run-plan snapshot approval when execute=true",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 400
        try:
            status = _read_run_status(orch, projects, run_id, project_id=project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        missing_gates = [
            gate
            for gate in required_gates
            if not _gate_approved(status, gate)
        ]
        if missing_gates:
            return jsonify(
                {
                    "ok": False,
                    "error": "gate approval required before adapter execution",
                    "missing_gates": missing_gates,
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403
        jobs.add_log(run_id, "INFO", "adapter", f"Starting adapter: {adapter_name}")
        try:
            result = adapter(adapter_payload)
        except Exception as exc:
            jobs.add_log(run_id, "ERROR", "adapter", f"Adapter {adapter_name} raised: {exc}")
            return jsonify({"ok": False, "error": str(exc), "adapter": adapter_name}), 500
        status = str(result.get("status") or "") if isinstance(result, dict) else ""
        level = "INFO" if status in {"success", "planned"} else "ERROR"
        jobs.add_log(run_id, level, "adapter", f"Adapter {adapter_name} finished: {status or 'unknown'}")
        return jsonify({"ok": True, "adapter": adapter_name, "result": result})

    # --- Projects ---

    @app.post("/api/projects")
    def create_project():
        payload = request.get_json(silent=True) or {}
        project_id = str(payload.get("project_id") or uuid.uuid4().hex[:8]).strip()
        name = str(payload.get("name") or project_id).strip()
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        try:
            project_dir = projects.project_dir(project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        write_json(project_dir / "project.json", {
            "project_id": project_id,
            "name": name,
            "created_at": now_iso(),
        })
        return jsonify({"ok": True, "project_id": project_id, "name": name})

    @app.get("/api/projects")
    def list_projects():
        projects_root = projects.projects_root
        result = []
        if projects_root.exists():
            for child in sorted(projects_root.iterdir()):
                if not child.is_dir():
                    continue
                meta = _read_json(child / "project.json")
                result.append({
                    "project_id": child.name,
                    "name": meta.get("name", child.name),
                    "created_at": meta.get("created_at", ""),
                })
        return jsonify({"ok": True, "projects": result})

    @app.get("/api/projects/<project_id>/memory")
    def list_project_memory(project_id: str):
        try:
            records = project_memory.list_project_records(str(project_id or "").strip())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "project_id": project_id,
                "enabled": project_memory.project_memory_enabled(str(project_id or "").strip()),
                "records": [record.model_dump(mode="json") for record in records],
            }
        )

    @app.post("/api/projects/<project_id>/memory/records")
    def create_project_memory_record(project_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            record = ProjectMemoryRecord.model_validate(payload)
            saved = project_memory.save_project_record(str(project_id or "").strip(), record)
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "record": saved.model_dump(mode="json")})

    @app.patch("/api/projects/<project_id>/memory/records/<record_id>")
    def update_project_memory_record(project_id: str, record_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            updated = project_memory.update_project_record(str(project_id or "").strip(), record_id, payload)
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if updated is None:
            return jsonify({"ok": False, "error": "memory record not found"}), 404
        return jsonify({"ok": True, "record": updated.model_dump(mode="json")})

    @app.delete("/api/projects/<project_id>/memory/records/<record_id>")
    def delete_project_memory_record(project_id: str, record_id: str):
        try:
            deleted = project_memory.delete_project_record(str(project_id or "").strip(), record_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "deleted": deleted})

    @app.get("/api/projects/<project_id>/memory/export")
    def export_project_memory(project_id: str):
        try:
            exported = project_memory.export_project_records(str(project_id or "").strip())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "export": exported})

    @app.post("/api/projects/<project_id>/memory/enabled")
    def set_project_memory_enabled(project_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        if not isinstance(payload.get("enabled"), bool):
            return jsonify({"ok": False, "error": "enabled boolean required"}), 400
        enabled = payload["enabled"]
        try:
            project_memory.set_project_memory_enabled(str(project_id or "").strip(), enabled)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "project_id": project_id, "enabled": enabled})

    # --- Upload ---

    @app.post("/api/projects/<project_id>/upload")
    def upload_file(project_id: str):
        clean_id = str(project_id or "").strip()
        if not clean_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        decision = permissions.decide(
            "upload_dataset",
            project_id=clean_id,
            project_approved=_as_bool(request.form.get("project_approved"))
            or _as_bool(request.headers.get("X-Project-Approved")),
            actor=str(request.form.get("actor") or request.headers.get("X-Actor") or ""),
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "project approval required for dataset upload",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403
        max_upload_bytes = int(app.config.get("AI4S_MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES) or MAX_UPLOAD_BYTES)
        content_length = request.content_length
        if max_upload_bytes > 0 and content_length is not None and content_length > max_upload_bytes:
            return jsonify(
                {
                    "ok": False,
                    "error": f"upload exceeds size limit: {max_upload_bytes} bytes",
                    "max_upload_bytes": max_upload_bytes,
                }
            ), 413
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "no file part"}), 400
        f = request.files["file"]
        if not f.filename or not _allowed_file(f.filename):
            return jsonify({"ok": False, "error": "unsupported file type"}), 400
        filename = secure_filename(f.filename)
        if not filename or not _allowed_file(filename):
            return jsonify({"ok": False, "error": "invalid filename after sanitization"}), 400
        try:
            project_dir = projects.project_dir(clean_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        upload_dir = project_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / filename
        if dest.exists() and dest.is_dir():
            return jsonify({"ok": False, "error": "invalid filename target"}), 400
        if dest.exists():
            return jsonify({"ok": False, "error": f"upload filename already exists: {filename}"}), 409
        try:
            with dest.open("xb") as out:
                _copy_upload_stream(f.stream, out, max_bytes=max_upload_bytes)
        except FileExistsError:
            return jsonify({"ok": False, "error": f"upload filename already exists: {filename}"}), 409
        except ValueError as exc:
            dest.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": str(exc), "max_upload_bytes": max_upload_bytes}), 413
        return jsonify({"ok": True, "path": str(dest), "filename": filename})

    @app.post("/api/projects/<project_id>/runs/<run_id>/models/register")
    def register_model(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        actor = str(payload.get("approved_by") or payload.get("actor") or "").strip()
        decision = permissions.decide(
            "register_model",
            project_id=clean_project_id,
            run_id=clean_run_id,
            confirmed=_as_bool(payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "model registration requires per-action confirmation",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403

        model_dir_raw = str(payload.get("model_dir") or "").strip()
        property_id = str(payload.get("property_id") or "").strip()
        backend = str(payload.get("backend") or "").strip()
        content_hash = str(payload.get("content_hash") or "").strip()
        if not model_dir_raw or not property_id or not backend or not content_hash:
            return jsonify(
                {
                    "ok": False,
                    "error": "model_dir, property_id, backend, and content_hash required",
                }
            ), 400
        try:
            manifest, version_dir = projects.register_model_asset(
                clean_project_id,
                clean_run_id,
                Path(model_dir_raw),
                property_id=property_id,
                backend=backend,
                content_hash=content_hash,
                approved_by=actor,
                approval_note=str(payload.get("approval_note") or ""),
            )
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "manifest": manifest.model_dump(mode="json"),
                "version_dir": str(version_dir),
                "permission": decision.model_dump(mode="json"),
            }
        )

    @app.post("/api/projects/<project_id>/runs/<run_id>/models/promote/draft")
    def draft_promoted_model_asset(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        version_dir_raw = str(payload.get("version_dir") or "").strip()
        if not version_dir_raw:
            return jsonify({"ok": False, "error": "version_dir required"}), 400
        try:
            draft = projects.build_promoted_model_asset_draft(
                clean_project_id,
                Path(version_dir_raw),
                overrides=_promotion_draft_overrides(payload),
            )
        except (ValueError, FileNotFoundError, ValidationError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "draft": draft})

    @app.post("/api/projects/<project_id>/runs/<run_id>/models/promote")
    def promote_model_asset(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        actor = str(payload.get("approved_by") or payload.get("actor") or "").strip()
        decision = permissions.decide(
            "promote_asset",
            project_id=clean_project_id,
            run_id=clean_run_id,
            confirmed=_as_bool(payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "model asset promotion requires per-action confirmation",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403

        version_dir_raw = str(payload.get("version_dir") or "").strip()
        model_id = str(payload.get("model_id") or "").strip()
        domain = str(payload.get("domain") or "").strip()
        property_id = str(payload.get("property_id") or "").strip()
        use_case = str(payload.get("use_case") or "").strip()
        backend = str(payload.get("backend") or "").strip()
        if not version_dir_raw or not model_id or not domain or not property_id or not use_case or not backend:
            return jsonify(
                {
                    "ok": False,
                    "error": "version_dir, model_id, domain, property_id, use_case, and backend required",
                }
            ), 400
        try:
            metrics = _object_field(payload.get("metrics"), "metrics")
            applicability = _object_field(payload.get("applicability"), "applicability")
            input_columns = _string_dict_field(payload.get("input_columns"), "input_columns")
            promoted, promoted_path = projects.promote_registered_model_asset(
                clean_project_id,
                clean_run_id,
                Path(version_dir_raw),
                model_id=model_id,
                domain=domain,
                property_id=property_id,
                use_case=use_case,
                backend=backend,
                approved_by=actor,
                metrics=metrics,
                applicability=applicability,
                feature_requirements=_string_list(payload.get("feature_requirements")),
                input_columns=input_columns,
                limitations=_string_list(payload.get("limitations")),
                rollback_asset_id=str(payload.get("rollback_asset_id") or "").strip(),
                note=str(payload.get("note") or ""),
            )
        except (ValueError, FileNotFoundError, ValidationError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "promoted_model_asset": promoted.model_dump(mode="json"),
                "promoted_model_asset_path": str(promoted_path),
                "permission": decision.model_dump(mode="json"),
            }
        )

    @app.post("/api/projects/<project_id>/runs/<run_id>/assets/promote")
    def promote_asset(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        actor = str(payload.get("approved_by") or payload.get("actor") or "").strip()
        decision = permissions.decide(
            "promote_asset",
            project_id=clean_project_id,
            run_id=clean_run_id,
            confirmed=_as_bool(payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "asset promotion requires per-action confirmation",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403

        source_artifacts = _string_list(payload.get("source_artifacts"))
        asset_id = str(payload.get("asset_id") or "").strip()
        asset_type = str(payload.get("asset_type") or "").strip()
        version = str(payload.get("version") or "").strip()
        if not asset_id or not asset_type or not version:
            return jsonify({"ok": False, "error": "asset_id, asset_type, and version required"}), 400
        record = AssetPromotionRecord(
            run_id=clean_run_id,
            asset_id=asset_id,
            asset_type=asset_type,
            version=version,
            source_artifacts=source_artifacts,
            approved_by=actor,
            approved_at=now_iso(),
            note=str(payload.get("note") or ""),
        )
        try:
            path = projects.append_asset_promotion_record(clean_project_id, clean_run_id, record)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "record": record.model_dump(mode="json"),
                "record_path": str(path),
                "permission": decision.model_dump(mode="json"),
            }
        )

    @app.get("/api/projects/<project_id>/runs/<run_id>/stage-timeline")
    def stage_timeline(project_id: str, run_id: str):
        try:
            state = projects.read_stage_state(project_id, run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if state is None:
            return jsonify({"ok": False, "error": "no stage state found for run"}), 404
        return jsonify({"ok": True, "timeline": build_stage_timeline(state)})

    @app.get("/api/projects/<project_id>/runs/<run_id>/report-preview")
    def report_preview(project_id: str, run_id: str):
        artifact_id = str(request.args.get("artifact_id") or "").strip()
        if not artifact_id:
            return jsonify({"ok": False, "error": "artifact_id required"}), 400
        try:
            registry = projects.read_artifact_registry(project_id, run_id)
            relative_path = registry.get(artifact_id, "")
            if not relative_path:
                return jsonify({"ok": False, "error": "report artifact not found in registry"}), 404
            preview = build_report_preview(
                run_dir=projects.run_dir(project_id, run_id),
                artifact_id=artifact_id,
                relative_path=relative_path,
            )
        except FileNotFoundError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "preview": preview})

    @app.post("/api/projects/<project_id>/runs/<run_id>/verify")
    def verify_project_run(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        try:
            observer = ObserverAgent(storage=projects, jobs=jobs)
            observation = observer.observe_run(clean_project_id, clean_run_id)
            verifier = VerifierAgent()
            report = verifier.verify(observation)
            report_json, report_md = verifier.write_reports(projects, clean_project_id, clean_run_id, report)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_log(clean_run_id, "INFO", "verifier", f"Verifier decision: {report.overall_decision}")
        return jsonify(
            {
                "ok": True,
                "report": report.model_dump(mode="json"),
                "outputs": {
                    "verification_report_json": str(report_json),
                    "verification_report_md": str(report_md),
                },
            }
        )

    # --- Logs ---

    @app.get("/api/runs/<run_id>/logs")
    def run_logs(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        limit = int(request.args.get("limit", 50))
        entries = jobs.get_logs(clean_run_id, limit=limit)
        return jsonify({"ok": True, "run_id": clean_run_id, "logs": entries})

    # --- Job control ---

    @app.post("/api/runs/<run_id>/pause")
    def pause_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.pause_job(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/runs/<run_id>/resume")
    def resume_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.resume_job(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/runs/<run_id>/stop")
    def stop_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.stop_job(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no active job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/background-jobs")
    def create_background_job():
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        project_id = str(payload.get("project_id") or "").strip()
        run_id = str(payload.get("run_id") or "").strip()
        task_id = str(payload.get("task_id") or "").strip()
        budget_payload = payload.get("budget")
        details_payload = payload.get("details")
        if not run_id or not task_id:
            return jsonify({"ok": False, "error": "run_id and task_id required"}), 400
        if not isinstance(budget_payload, dict):
            return jsonify({"ok": False, "error": "budget object required"}), 400
        if details_payload is not None and not isinstance(details_payload, dict):
            return jsonify({"ok": False, "error": "details must be an object"}), 400
        try:
            budget = BackgroundJobBudget.model_validate(budget_payload)
            job = jobs.start_background_job(
                run_id,
                project_id=project_id,
                task_id=task_id,
                budget=budget,
                details=details_payload if isinstance(details_payload, dict) else None,
            )
        except ValidationError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        except ValueError as exc:
            status_code = 409 if "already active" in str(exc) else 400
            return jsonify({"ok": False, "error": str(exc)}), status_code
        return jsonify({"ok": True, "job": job})

    @app.get("/api/background-jobs/<run_id>")
    def get_background_job(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            job = jobs.get_background_job(clean_run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if job is None:
            return jsonify({"ok": False, "error": "no background job"}), 404
        return jsonify({"ok": True, "job": job})

    @app.post("/api/background-jobs/<run_id>/checkpoints")
    def record_background_checkpoint(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            payload = _request_json_object()
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        stage = str(payload.get("stage") or "").strip()
        cursor = payload.get("cursor")
        artifact_refs = payload.get("artifact_refs")
        if not stage:
            return jsonify({"ok": False, "error": "stage required"}), 400
        if cursor is not None and not isinstance(cursor, dict):
            return jsonify({"ok": False, "error": "cursor must be an object"}), 400
        if artifact_refs is not None and not isinstance(artifact_refs, list):
            return jsonify({"ok": False, "error": "artifact_refs must be a list"}), 400
        try:
            completed_units = payload.get("completed_units", 0)
            runtime_sec = payload.get("runtime_sec", 0)
            cost_usd = payload.get("cost_usd", 0.0)
            checkpoint = jobs.record_background_checkpoint(
                clean_run_id,
                stage=stage,
                cursor=cursor if isinstance(cursor, dict) else None,
                completed_units=completed_units,
                runtime_sec=runtime_sec,
                cost_usd=cost_usd,
                artifact_refs=[str(item) for item in artifact_refs] if isinstance(artifact_refs, list) else None,
            )
        except KeyError:
            return jsonify({"ok": False, "error": "no background job"}), 404
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "checkpoint": checkpoint})

    @app.get("/api/background-jobs/<run_id>/resume-plan")
    def background_resume_plan(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        try:
            resume_plan = jobs.background_resume_plan(clean_run_id)
        except KeyError:
            return jsonify({"ok": False, "error": "no background job"}), 404
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "resume_plan": resume_plan})

    @app.post("/api/runs/<run_id>/retry")
    def retry_run(run_id: str):
        clean_run_id = str(run_id or "").strip()
        if not clean_run_id:
            return jsonify({"ok": False, "error": "run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        stage = str(payload.get("stage") or "").strip()
        project_id = str(payload.get("project_id") or "").strip()
        status = orch.read_status(clean_run_id)
        if not status.get("plan_exists"):
            return jsonify({"ok": False, "error": "no plan found for run"}), 404
        if jobs.get_job(clean_run_id):
            return jsonify({"ok": False, "error": "run is active; pause or stop before retry"}), 409
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required for failed-stage retry"}), 400
        try:
            state = projects.read_stage_state(project_id, clean_run_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if state is None:
            return jsonify({"ok": False, "error": "no stage state found for run"}), 404
        if state.status != RunStatus.FAILED:
            return jsonify({"ok": False, "error": "latest stage has not failed"}), 409

        error = state.error if isinstance(state.error, dict) else {}
        retryable_stages = error.get("retryable_stages", [])
        if not isinstance(retryable_stages, list):
            retryable_stages = []
        requested_stage = stage or state.stage
        explicitly_retryable = requested_stage in {str(item) for item in retryable_stages}
        if requested_stage != state.stage and not explicitly_retryable:
            return jsonify(
                {
                    "ok": False,
                    "error": "retry is limited to latest failed stage or explicitly retryable stage",
                    "latest_failed_stage": state.stage,
                }
            ), 400
        if not bool(error.get("retryable")) and not explicitly_retryable:
            return jsonify({"ok": False, "error": "latest failed stage is not retryable"}), 409

        now = now_iso()
        details = dict(state.details)
        details["retry_requested_at"] = now
        details["retry_stage"] = requested_stage
        details["retry_count"] = int(details.get("retry_count") or 0) + 1
        state.status = RunStatus.PENDING
        state.started_at = now
        state.updated_at = now
        state.ended_at = None
        state.details = details
        state.history.append(
            StageHistoryItem(
                stage=requested_stage,
                status=RunStatus.PENDING,
                updated_at=now,
                note="retry requested",
            )
        )
        projects.write_stage_state(project_id, clean_run_id, state)
        try:
            job = jobs.start_job(
                clean_run_id,
                details={"retry": True, "retry_stage": requested_stage, "project_id": project_id},
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        jobs.add_log(clean_run_id, "INFO", "retry", f"Retry requested for stage: {requested_stage}")
        return jsonify({"ok": True, "run_id": clean_run_id, "retry_stage": requested_stage, "job": job})

    # --- Job list ---

    @app.get("/api/jobs")
    def list_jobs():
        return jsonify({"ok": True, "jobs": jobs.list_jobs()})


def _read_json(path: Path) -> dict:
    import json
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_trainability_property(trainability_report: object) -> str:
    if not isinstance(trainability_report, dict):
        return ""
    properties = trainability_report.get("properties")
    if not isinstance(properties, list):
        return ""
    for item in properties:
        if not isinstance(item, dict):
            continue
        property_id = str(item.get("property_id") or "").strip()
        if property_id:
            return property_id
    return ""


def _object_field(value: object, field_name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_dict_field(value: object, field_name: str) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()
        clean_value = str(raw or "").strip()
        if clean_key and clean_value:
            result[clean_key] = clean_value
    return result


def _promotion_draft_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("model_id", "domain", "use_case", "rollback_asset_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            overrides[key] = value
    for key in ("metrics", "applicability", "input_columns"):
        value = payload.get(key)
        if isinstance(value, dict):
            overrides[key] = value
    for key in ("feature_requirements", "limitations"):
        value = _string_list(payload.get(key))
        if value:
            overrides[key] = value
    return overrides


def _task_options(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for task_id, options in value.items():
        if isinstance(options, dict):
            normalized[str(task_id)] = {str(key): option_value for key, option_value in options.items()}
    return normalized


def _log_run_plan_execution_result(jobs: JobManager, run_id: str, execution: dict[str, Any]) -> None:
    status = str(execution.get("status") or "")
    if status == RunStatus.WAITING_USER.value:
        task = str(execution.get("waiting_task") or execution.get("planned_task") or "").strip()
        suffix = f" at {task}" if task else ""
        jobs.add_log(run_id, "INFO", "run_plan", f"RunPlan waiting for user{suffix}")
    elif status == RunStatus.FAILED.value:
        task = str(execution.get("failed_task") or "").strip()
        suffix = f" at {task}" if task else ""
        jobs.add_log(run_id, "ERROR", "run_plan", f"RunPlan execution failed{suffix}")
    elif status in {RunStatus.SUCCEEDED.value, RunStatus.DONE.value}:
        jobs.add_log(run_id, "INFO", "run_plan", f"RunPlan execution completed: {status}")
    else:
        jobs.add_log(run_id, "INFO", "run_plan", f"RunPlan execution status: {status or 'unknown'}")


_STAGE_LABELS = {
    "inspect_dataset": "数据检查",
    "clean_dataset": "数据清洗",
    "check_trainability": "可训练性评估",
    "run_baseline": "基线评估",
    "train_model": "模型训练",
    "generate_candidates": "候选分子生成",
    "predict_candidates": "候选性质预测",
    "filter_rank": "筛选排序",
    "render_report": "报告生成",
}

_GATE_EXPLANATIONS = {
    GateName.TRAIN_CONFIG.value: "模型训练会消耗算力并产生可复用模型资产，批准前应确认数据列、目标属性、后端和远程执行策略。",
    GateName.FINAL_THRESHOLD.value: "候选生成可能消耗较多算力或调用外部生成后端，批准前应确认生成数量、约束和后续筛选标准。",
    GateName.DATA_MINING.value: "数据挖掘可能访问外部来源，批准前应确认来源范围、许可和引用要求。",
    GateName.TASK_PARSE.value: "任务解析决定后续执行范围，批准前应确认目标、输入和输出是否符合预期。",
    GateName.POST_INFER_STATS.value: "推理后统计会影响筛选阈值，批准前应确认指标和异常处理策略。",
}


def _build_agent_decision_card(
    *,
    project_id: str,
    run_id: str,
    state: StageState | None,
    log_tail: list[dict[str, str]],
    question: str = "",
) -> dict[str, object]:
    if state is None:
        card = {
            "project_id": project_id,
            "run_id": run_id,
            "stage": "",
            "stage_label": "尚未开始",
            "status": RunStatus.PENDING.value,
            "title": "等待提交任务",
            "summary": "Agent 还没有可观察的运行阶段。请先提交任务，随后我会解释每一步需要你决策的内容。",
            "decision_required": False,
            "primary_action": "submit_task",
            "recommendation": "先提交任务并进入监控；提交后会自动生成下一张决策卡。",
            "required_gates": [],
            "evidence": [],
            "risks": [],
            "next_step": "提交任务",
            "answer": _answer_agent_decision_question(question, None, [], []),
        }
        return card

    details = state.details if isinstance(state.details, dict) else {}
    required_gates = _string_list(details.get("required_gates"))
    stage_label = _STAGE_LABELS.get(state.stage, state.stage or "当前阶段")
    evidence = _decision_evidence(state, log_tail)
    risks = _decision_risks(state, required_gates)
    status = state.status.value

    if state.status == RunStatus.WAITING_USER and required_gates:
        primary_action = "approve_gate"
        title = f"需要你确认：{stage_label}"
        recommendation = f"建议在确认证据和风险后批准 {required_gates[0]}，让 Agent 继续执行 {stage_label}。"
        next_step = "批准并继续"
        decision_required = True
    elif state.status == RunStatus.WAITING_USER:
        primary_action = "review_plan"
        title = f"需要你审阅：{stage_label}"
        recommendation = f"{stage_label} 已生成可审阅计划；如果计划符合预期，再提供执行确认或必要输入。"
        next_step = "审阅计划"
        decision_required = True
    elif state.status == RunStatus.FAILED:
        primary_action = "review_failure"
        title = f"运行失败：{stage_label}"
        recommendation = "先查看错误和日志；如果错误可恢复，再选择重试或修改计划。"
        next_step = "查看失败原因"
        decision_required = True
    elif state.status in {RunStatus.SUCCEEDED, RunStatus.DONE}:
        primary_action = "continue"
        title = f"阶段完成：{stage_label}"
        recommendation = "当前阶段已完成，可以继续查看报告、提升资产，或根据结果发起下一轮计划。"
        next_step = "继续到报告"
        decision_required = False
    else:
        primary_action = "refresh"
        title = f"运行中：{stage_label}"
        recommendation = "当前阶段仍在执行；建议先查看 tail-20 日志，等待下一张需要决策的卡片。"
        next_step = "刷新状态"
        decision_required = False

    return {
        "project_id": project_id,
        "run_id": run_id,
        "stage": state.stage,
        "stage_label": stage_label,
        "status": status,
        "title": title,
        "summary": _decision_summary(state, stage_label),
        "decision_required": decision_required,
        "primary_action": primary_action,
        "recommendation": recommendation,
        "required_gates": required_gates,
        "evidence": evidence,
        "risks": risks,
        "next_step": next_step,
        "answer": _answer_agent_decision_question(question, state, required_gates, risks),
    }


def _decision_summary(state: StageState, stage_label: str) -> str:
    if state.status == RunStatus.WAITING_USER:
        return f"Agent 已暂停在 {stage_label}，需要你根据证据和风险做出下一步决策。"
    if state.status == RunStatus.FAILED:
        message = ""
        if isinstance(state.error, dict):
            message = str(state.error.get("message") or state.error.get("reason") or "")
        return f"{stage_label} 未成功完成。{message}".strip()
    if state.status in {RunStatus.SUCCEEDED, RunStatus.DONE}:
        return f"{stage_label} 已完成，Agent 可以继续后续审阅或报告步骤。"
    return f"{stage_label} 当前状态为 {state.status.value}。"


def _decision_evidence(state: StageState, log_tail: list[dict[str, str]]) -> list[str]:
    details = state.details if isinstance(state.details, dict) else {}
    evidence: list[str] = []
    executed = _string_list(details.get("executed_tasks"))
    if executed:
        evidence.append("已完成任务：" + ", ".join(executed))
    if state.next_stage:
        evidence.append(f"下一阶段：{state.next_stage}")
    if state.artifacts:
        evidence.append(f"当前阶段产物数量：{len(state.artifacts)}")
    if isinstance(state.error, dict) and state.error:
        evidence.append("错误信息：" + str(state.error.get("message") or state.error.get("reason") or state.error))
    if log_tail:
        last = log_tail[-1]
        evidence.append(f"最新日志：[{last.get('level', '')}] {last.get('message', '')}".strip())
    return evidence or ["暂无额外证据；请先刷新状态或查看日志。"]


def _decision_risks(state: StageState, required_gates: list[str]) -> list[str]:
    risks = [_GATE_EXPLANATIONS[gate] for gate in required_gates if gate in _GATE_EXPLANATIONS]
    if state.stage == "train_model" and not risks:
        risks.append("模型训练可能消耗算力，并会影响后续预测和报告质量。")
    if state.stage == "generate_candidates" and not risks:
        risks.append("候选生成可能引入外部后端、较高成本或无效候选，需要确认生成策略。")
    if state.status == RunStatus.FAILED and isinstance(state.error, dict):
        risks.append("失败后直接重试可能重复同一错误；建议先确认错误是否可恢复。")
    return risks


def _answer_agent_decision_question(
    question: str,
    state: StageState | None,
    required_gates: list[str],
    risks: list[str],
) -> str:
    clean_question = question.strip()
    if not clean_question:
        return ""
    stage = state.stage if state is not None else ""
    stage_label = _STAGE_LABELS.get(stage, stage or "当前阶段")
    lowered = clean_question.lower()
    if "远程" in clean_question or "unimol" in lowered or "reinvent" in lowered:
        return (
            f"{stage_label} 是否会执行远程后端取决于当前后端选项和 execute 标记。"
            "默认策略是 plan-only：没有显式 execute=true 时，只生成执行计划并等待你再次确认。"
        )
    if "为什么" in clean_question or "批准" in clean_question:
        gate_text = "、".join(required_gates) if required_gates else "当前决策"
        risk_text = "；".join(risks) if risks else "该步骤会影响后续结果或资源消耗"
        return f"需要批准 {gate_text}，因为 {stage_label} 是高影响步骤。主要风险：{risk_text}。"
    if "日志" in clean_question or "进度" in clean_question:
        return "请查看下方实时日志 tail-20；每次提交、刷新或批准后都会重新加载最近 20 条运行日志。"
    return f"我会基于当前阶段 {stage_label} 的状态、门控、证据和日志给出建议；如果信息不足，建议先刷新日志和阶段时间线。"


def _llm_provider_from_payload(payload: dict) -> LLMProvider | None:
    raw = payload.get("llm_provider")
    if raw in (None, "", False):
        return None
    if not isinstance(raw, dict):
        raise ValueError("llm_provider must be an object when provided")
    config = LLMProviderConfig.model_validate(raw)
    return create_llm_provider(config)


def _expand_plan_from_payload(payload: dict) -> RunPlan:
    run_id = str(payload.get("run_id") or "preview").strip() or "preview"
    requested_tasks = _string_list(payload.get("requested_tasks"))
    if not requested_tasks:
        raise ValueError("requested_tasks required")
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=requested_tasks,
        available_artifacts=_string_list(payload.get("available_artifacts")),
    )


def _run_plan_from_payload(value: object, *, run_id: str) -> RunPlan:
    if not isinstance(value, dict):
        raise ValueError("before and after plan payloads are required")
    if "tasks" in value and "requested_tasks" in value:
        return RunPlan.model_validate(value | {"run_id": str(value.get("run_id") or run_id)})
    payload = dict(value)
    payload.setdefault("run_id", run_id)
    return _expand_plan_from_payload(payload)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}


def _adapter_execution_policy(adapter_name: str, adapter_payload: dict) -> tuple[str, list[str]] | None:
    registry = AtomicTaskRegistry()
    by_adapter = {
        task.default_adapter: task
        for task in registry.list_tasks()
        if task.default_adapter
    }
    adapter_aliases = {
        "draft_cleaning_rules_adapter": "clean_dataset",
        "train_model_unimol_legacy_adapter": "train_model",
        "predict_candidates_domain_model_adapter": "predict_candidates",
        "predict_candidates_unimol_legacy_adapter": "predict_candidates",
        "parse_pdf_folder_mineru_adapter": "parse_document",
        "parse_document_pdfplumber_adapter": "parse_document",
        "parse_document_pymupdf_adapter": "parse_document",
        "parse_document_grobid_adapter": "parse_document",
    }
    task = by_adapter.get(adapter_name)
    if task is None and adapter_name in adapter_aliases:
        task = registry.get(adapter_aliases[adapter_name])
    if task is None:
        return None

    action = task.task_id
    if task.task_id == "generate_candidates":
        backend = str(adapter_payload.get("backend") or "deterministic_stub").strip().lower()
        try:
            count = int(adapter_payload.get("count") or adapter_payload.get("num_candidates") or 32)
        except (TypeError, ValueError) as exc:
            raise ValueError("generation count must be a positive integer") from exc
        if count <= 0:
            raise ValueError("generation count must be a positive integer")
        if backend != "deterministic_stub" or count >= 128:
            action = "generate_candidates_expensive"
    return action, list(task.gates)


_CANNOT_DIRECT_EXECUTE = frozenset(
    {
        "predict_candidates_unimol_legacy_adapter",
        "predict_candidates_domain_model_adapter",
        "train_model_unimol_legacy_adapter",
    }
)


def _adapter_requires_snapshot_for_execute(adapter_name: str, adapter_payload: dict) -> bool:
    """Return True when a plan-capable adapter requests execute=true directly.

    Remote / heavyweight adapters that accept ``execute`` must go through
    the RunPlan executor so that every execution records a snapshot,
    approval audit, and content-hash verification.

    Only accepts Python ``bool`` for ``execute``.
    String ``\"false\"``, ``\"0\"``, and missing keys are treated as ``False``.
    """
    if adapter_name not in _CANNOT_DIRECT_EXECUTE:
        return False
    execute_raw = adapter_payload.get("execute")
    if execute_raw is None:
        return False
    return strict_bool(execute_raw, key="execute")


def _read_run_status(
    orch: Orchestrator,
    projects: ProjectStorage,
    run_id: str,
    *,
    project_id: str = "",
) -> dict[str, object]:
    legacy_status = dict(orch.read_status(run_id))
    clean_project_id = str(project_id or "").strip()
    if not clean_project_id:
        return {**legacy_status, "state_source": "legacy"}

    project_status = _read_project_run_status(projects, clean_project_id, run_id)
    if not project_status:
        return {
            **legacy_status,
            "project_id": clean_project_id,
            "state_source": "legacy",
        }
    return {
        **legacy_status,
        **project_status,
        "legacy_plan_exists": bool(legacy_status.get("plan_exists")),
    }


def _read_project_run_status(projects: ProjectStorage, project_id: str, run_id: str) -> dict[str, object]:
    run_path = _project_run_dir_if_exists(projects, project_id, run_id)
    if run_path is None:
        return {}

    stage = _read_json(run_path / "stage.json")
    gate_payload = _read_json(run_path / "gate_decisions.json")
    artifact_payload = _read_json(run_path / "artifact_registry.json")
    decisions = gate_payload.get("decisions", [])
    if not isinstance(decisions, list):
        decisions = []
    artifacts = artifact_payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    status: dict[str, object] = {
        "run_id": run_id,
        "project_id": project_id,
        "state_source": "project",
        "plan_exists": bool(_read_json(run_path / "run_plan.json") or _read_json(run_path / "plan.json")),
        "gate_decisions": [decision for decision in decisions if isinstance(decision, dict)],
        "artifacts": {str(key): str(value) for key, value in artifacts.items()},
    }
    if stage:
        status["stage"] = stage
        status["stage_status"] = str(stage.get("status") or "")
    return status


def _project_run_dir_if_exists(projects: ProjectStorage, project_id: str, run_id: str) -> Path | None:
    clean_project_id = str(project_id or "").strip()
    clean_run_id = str(run_id or "").strip()
    if not clean_project_id or not clean_run_id:
        return None

    project_path = (projects.projects_root / clean_project_id).resolve()
    if not project_path.is_relative_to(projects.projects_root):
        raise ValueError("project_id escapes base directory")
    runs_base = (project_path / "runs").resolve()
    run_path = (runs_base / clean_run_id).resolve()
    if not run_path.is_relative_to(runs_base):
        raise ValueError("run_id escapes base directory")
    return run_path if run_path.exists() else None


def _gate_approved(status: dict[str, object], gate: str) -> bool:
    decisions = status.get("gate_decisions", [])
    if not isinstance(decisions, list):
        return False
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("gate") or "") == gate and bool(decision.get("approved")):
            return True
    return False
