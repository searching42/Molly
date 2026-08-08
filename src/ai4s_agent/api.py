from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from ai4s_agent.conversation_store import ConversationStore
from ai4s_agent.control_plane_events import ControlPlaneEventProjector
from ai4s_agent.dataset_workflow import DatasetWorkflowService
from ai4s_agent.execution_agent import ExecutionAgentService
from ai4s_agent.execution_agent_store import ExecutionAgentStore
from ai4s_agent.agent_run_inspection import AgentRunInspectionService
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.harness_tracing import build_harness_observability
from ai4s_agent.job_manager import JobManager
from ai4s_agent.llm_provider import LLMProviderManager
from ai4s_agent.llm_settings import LLMSettingsStore
from ai4s_agent.literature_intake import LiteratureIntakeService
from ai4s_agent.memory import PermissionPolicy, ProjectMemory
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.planner import AtomicTaskRegistry
from ai4s_agent.remote_execution_lifecycle import RemoteExecutionLifecycleService
from ai4s_agent.remote_resource_authority import RemoteResourceAuthorityPolicyStore
from ai4s_agent.resource_profiles import (
    BR1_REAL_TOOL_EXECUTION_PROFILE_IDS,
    ResourceProfileStore,
)
from ai4s_agent.routes import run_control as run_control_routes
from ai4s_agent.routes.agents import _as_bool, register_agent_routes
from ai4s_agent.routes.conversations import register_conversation_routes
from ai4s_agent.routes.control_plane_events import register_control_plane_event_routes
from ai4s_agent.routes.datasets import register_dataset_routes
from ai4s_agent.routes.execution_agent import register_execution_agent_routes
from ai4s_agent.routes.agent_run_inspection import register_agent_run_inspection_routes
from ai4s_agent.routes.core import register_core_routes
from ai4s_agent.routes.internal_run_plan_queue import register_internal_run_plan_queue_routes
from ai4s_agent.routes.jobs import register_job_routes
from ai4s_agent.routes.legacy_plan import register_legacy_plan_routes
from ai4s_agent.routes.llm_settings import register_llm_settings_routes
from ai4s_agent.routes.literature_intakes import register_literature_intake_routes
from ai4s_agent.routes.oled_bounded_sessions import register_oled_bounded_session_routes
from ai4s_agent.routes.project_assets import register_project_asset_routes
from ai4s_agent.routes.project_runs import register_project_run_routes
from ai4s_agent.routes.remote_executions import register_remote_execution_routes
from ai4s_agent.routes.projects import register_project_routes
from ai4s_agent.routes.review import register_review_routes
from ai4s_agent.routes.run_plans import register_run_plan_routes
from ai4s_agent.routes.scientific_agent_plans import register_scientific_agent_plan_routes
from ai4s_agent.routes.scientific_agent_replanner import (
    register_scientific_agent_replanner_routes,
)
from ai4s_agent.routes.scientific_agent_permissions import (
    register_scientific_agent_permission_routes,
)
from ai4s_agent.routes.scientific_agent_harness_controller import (
    register_scientific_agent_harness_controller_routes,
)
from ai4s_agent.scientific_agent_harness_controller import (
    ScientificAgentHarnessController,
)
from ai4s_agent.scientific_agent_replanner import ScientificAgentReplannerService
from ai4s_agent.scientific_agent_conversation import (
    ScientificAgentConversationSessionService,
)
from ai4s_agent.scientific_agent_run_input_binding import (
    ScientificAgentRunInputBindingService,
)
from ai4s_agent.routes.scientific_agent_conversation import (
    register_scientific_agent_conversation_routes,
)
from ai4s_agent.scientific_agent_plan import ScientificAgentPlanService
from ai4s_agent.routes.worker_deployment import register_worker_deployment_routes
from ai4s_agent.storage import ProjectStorage


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]
ALLOWED_EXTENSIONS = {"csv", "json", "sdf", "mol", "smi"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_CONVERSATION_ATTACHMENT_BYTES = 100 * 1024 * 1024
UPLOAD_COPY_CHUNK_BYTES = 1024 * 1024

_adapter_execution_policy = run_control_routes._adapter_execution_policy
_adapter_requires_snapshot_for_execute = run_control_routes._adapter_requires_snapshot_for_execute


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


def _workspace_from_config(base_runs_dir: Path | None, workspace_dir: Path | None) -> Path:
    if workspace_dir is not None:
        return Path(workspace_dir).resolve()
    if base_runs_dir is None:
        return DEFAULT_WORKSPACE.resolve()
    runs_path = Path(base_runs_dir).resolve()
    if runs_path.name == "runs":
        return runs_path.parent.resolve()
    return runs_path


def register_routes(
    app: Flask,
    base_runs_dir: Path | None = None,
    workspace_dir: Path | None = None,
    user_config_dir: Path | None = None,
    scientific_task_registry: AtomicTaskRegistry | None = None,
) -> None:
    runs = Path(base_runs_dir or DEFAULT_RUNS_DIR).resolve()
    workspace = _workspace_from_config(base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
    orch = Orchestrator(base_runs_dir=runs)
    jobs = JobManager(runs_dir=runs)
    projects = ProjectStorage(workspace_dir=workspace)
    conversations = ConversationStore(projects=projects)
    datasets = DatasetWorkflowService(projects=projects, conversations=conversations)
    literature_intakes = LiteratureIntakeService(
        projects=projects,
        conversations=conversations,
    )
    br1_real_tool_registry = False
    if scientific_task_registry is not None:
        try:
            scientific_task_registry.get("predict_private_unimol_v1")
        except ValueError:
            pass
        else:
            br1_real_tool_registry = True
    resource_profiles = ResourceProfileStore(
        workspace_dir=workspace,
        config_dir=user_config_dir,
        execution_profile_ids=(
            BR1_REAL_TOOL_EXECUTION_PROFILE_IDS
            if br1_real_tool_registry
            else None
        ),
    )
    resource_authority_policies = RemoteResourceAuthorityPolicyStore(
        config_dir=user_config_dir,
    )
    remote_executions = RemoteExecutionLifecycleService(
        projects=projects,
        profiles=resource_profiles,
    )
    project_memory = ProjectMemory(workspace_dir=workspace)
    llm_settings = LLMSettingsStore(workspace_dir=workspace, config_dir=user_config_dir)
    llm_providers = LLMProviderManager()
    permissions = PermissionPolicy()
    bounded_session_actions = OledBoundedDiscoverySessionActionService(
        storage=projects,
        actions_root=runs / "oled-bounded-session-actions",
    )
    control_plane_events = ControlPlaneEventProjector(
        storage=projects,
        actions=bounded_session_actions,
        events_root=runs / "control-plane-event-projections",
    )
    app.extensions["oled_bounded_session_actions"] = bounded_session_actions
    app.extensions["control_plane_event_projector"] = control_plane_events
    app.extensions["llm_provider_manager"] = llm_providers
    app.extensions["conversation_store"] = conversations
    app.extensions["dataset_workflow_service"] = datasets
    app.extensions["literature_intake_service"] = literature_intakes
    app.extensions["remote_execution_lifecycle"] = remote_executions
    harness_tracer, telemetry_health = build_harness_observability()
    app.extensions["harness_tracer"] = harness_tracer
    app.extensions["harness_telemetry_health"] = telemetry_health

    register_core_routes(app)
    register_legacy_plan_routes(app, orch=orch, jobs=jobs)
    register_run_plan_routes(app, projects=projects, jobs=jobs)
    register_internal_run_plan_queue_routes(app, projects=projects)
    register_agent_routes(
        app,
        projects=projects,
        project_memory=project_memory,
        jobs=jobs,
        llm_settings=llm_settings,
        llm_providers=llm_providers,
    )
    register_scientific_agent_plan_routes(
        app,
        projects=projects,
        resource_profiles=resource_profiles,
        llm_settings=llm_settings,
        llm_providers=llm_providers,
        registry=scientific_task_registry,
        tracer=harness_tracer,
    )
    register_scientific_agent_permission_routes(
        app,
        projects=projects,
        proposal_store=app.extensions["scientific_agent_plan_proposal_store"],
        resource_profiles=resource_profiles,
        resource_authority_policy_store=resource_authority_policies,
        registry=scientific_task_registry,
        tracer=harness_tracer,
    )
    harness_controller = ScientificAgentHarnessController(
        storage=projects,
        proposal_store=app.extensions["scientific_agent_plan_proposal_store"],
        authorization_service=app.extensions["scientific_agent_authorization_service"],
        control_store=app.extensions["scientific_agent_plan_control_store"],
        resource_authority_service=app.extensions["remote_resource_authority_service"],
        executor=RunPlanExecutor(storage=projects, registry=scientific_task_registry),
        remote_executions=remote_executions,
        tracer=harness_tracer,
    )
    register_scientific_agent_harness_controller_routes(
        app,
        controller=harness_controller,
    )
    execution_agent_store = ExecutionAgentStore(storage=projects)
    execution_agent = ExecutionAgentService(
        controller=harness_controller,
        store=execution_agent_store,
        tracer=harness_tracer,
    )
    register_execution_agent_routes(
        app,
        service=execution_agent,
        llm_settings=llm_settings,
        llm_providers=llm_providers,
    )
    conversation_plan_service = ScientificAgentPlanService(
        storage=projects,
        registry=app.extensions["scientific_agent_plan_proposal_store"].registry,
        observation_builder=app.extensions["scientific_agent_plan_observation_builder"],
        proposal_store=app.extensions["scientific_agent_plan_proposal_store"],
        tracer=harness_tracer,
    )
    input_binding_service = ScientificAgentRunInputBindingService(
        storage=projects,
        require_reinvent4_template=br1_real_tool_registry,
    )
    app.extensions["scientific_agent_run_input_binding_service"] = input_binding_service
    conversation_session_service = ScientificAgentConversationSessionService(
        projects=projects,
        conversations=conversations,
        plan_service=conversation_plan_service,
        proposal_store=app.extensions["scientific_agent_plan_proposal_store"],
        authorization_service=app.extensions["scientific_agent_authorization_service"],
        controller=harness_controller,
        execution_agent=execution_agent,
        input_binding_service=input_binding_service,
    )
    register_scientific_agent_conversation_routes(
        app,
        service=conversation_session_service,
        llm_settings=llm_settings,
        llm_providers=llm_providers,
    )
    replanner = ScientificAgentReplannerService(
        storage=projects,
        proposal_store=app.extensions["scientific_agent_plan_proposal_store"],
        observation_builder=app.extensions["scientific_agent_plan_observation_builder"],
        authorization_service=app.extensions["scientific_agent_authorization_service"],
        control_store=app.extensions["scientific_agent_plan_control_store"],
        controller=harness_controller,
        execution_agent_store=execution_agent_store,
        tracer=harness_tracer,
    )
    register_scientific_agent_replanner_routes(
        app,
        service=replanner,
        llm_settings=llm_settings,
        llm_providers=llm_providers,
    )
    register_agent_run_inspection_routes(
        app,
        service=AgentRunInspectionService(
            storage=projects,
            proposal_store=app.extensions["scientific_agent_plan_proposal_store"],
            authorization_service=app.extensions["scientific_agent_authorization_service"],
            control_store=app.extensions["scientific_agent_plan_control_store"],
            controller=harness_controller,
            execution_agent_store=execution_agent_store,
            tracer=harness_tracer,
        ),
    )
    register_llm_settings_routes(
        app,
        settings=llm_settings,
        providers=llm_providers,
        on_change=llm_providers.invalidate,
    )
    register_conversation_routes(
        app,
        conversations=conversations,
        max_attachment_bytes_default=MAX_CONVERSATION_ATTACHMENT_BYTES,
    )
    register_dataset_routes(app, datasets=datasets)
    register_literature_intake_routes(app, intakes=literature_intakes)
    register_worker_deployment_routes(
        app,
        workspace=workspace,
        runs=runs,
        user_config_dir=user_config_dir,
        resource_profiles=resource_profiles,
    )
    register_remote_execution_routes(app, executions=remote_executions)
    register_review_routes(app, workspace=workspace, permissions=permissions)
    run_control_routes.register_run_control_routes(
        app,
        orch=orch,
        jobs=jobs,
        projects=projects,
        permissions=permissions,
    )
    register_project_routes(
        app,
        projects=projects,
        project_memory=project_memory,
        permissions=permissions,
        allowed_file=_allowed_file,
        copy_upload_stream=_copy_upload_stream,
        max_upload_bytes_default=MAX_UPLOAD_BYTES,
    )
    register_project_asset_routes(app, projects=projects, permissions=permissions)
    register_project_run_routes(app, projects=projects, jobs=jobs)
    register_job_routes(app, jobs=jobs, orch=orch, projects=projects)
    register_oled_bounded_session_routes(
        app,
        projects=projects,
        actions=bounded_session_actions,
    )
    register_control_plane_event_routes(
        app,
        projector=control_plane_events,
    )
