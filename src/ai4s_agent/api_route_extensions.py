from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


RouteInstaller = Callable[[], None]


@dataclass(frozen=True)
class RouteExtensionSpec:
    extension_id: str
    installer_name: str
    module: str
    summary: str
    mechanism: str
    depends_on: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "extension_id": self.extension_id,
            "installer_name": self.installer_name,
            "module": self.module,
            "summary": self.summary,
            "mechanism": self.mechanism,
            "depends_on": list(self.depends_on),
        }


ROUTE_EXTENSION_SPECS: tuple[RouteExtensionSpec, ...] = (
    RouteExtensionSpec(
        extension_id="run_plan_executor_snapshot_builder",
        installer_name="install_run_plan_executor_snapshot_builder",
        module="ai4s_agent.snapshot_material",
        summary="Bind RunPlanExecutor gate pauses to snapshot material builders.",
        mechanism="method_patch",
    ),
    RouteExtensionSpec(
        extension_id="execution_confirmation_audit",
        installer_name="install_execution_confirmation_audit",
        module="ai4s_agent.execution_confirmation",
        summary="Record execution confirmations after approved run-plan resumes.",
        mechanism="method_patch",
        depends_on=("run_plan_executor_snapshot_builder",),
    ),
    RouteExtensionSpec(
        extension_id="chat_project_context",
        installer_name="install_chat_project_context",
        module="ai4s_agent.chat_context",
        summary="Infer conversation available inputs from project artifact registries.",
        mechanism="method_patch",
    ),
    RouteExtensionSpec(
        extension_id="chat_run_plan_routes",
        installer_name="install_chat_run_plan_routes",
        module="ai4s_agent.chat_runplan",
        summary="Add conversation run-plan preview and execution-feedback routes.",
        mechanism="register_routes_wrapper",
        depends_on=("chat_project_context",),
    ),
    RouteExtensionSpec(
        extension_id="phase3_executor_support",
        installer_name="install_phase3_executor_support",
        module="ai4s_agent.phase3_executor",
        summary="Add Phase 3 literature task payload builders to RunPlanExecutor.",
        mechanism="method_patch",
    ),
    RouteExtensionSpec(
        extension_id="execution_policy_registry",
        installer_name="install_execution_policy_registry",
        module="ai4s_agent.execution_policy",
        summary="Install centralized adapter execution policy hooks.",
        mechanism="method_patch",
    ),
    RouteExtensionSpec(
        extension_id="external_approval_split",
        installer_name="install_external_approval_split",
        module="ai4s_agent.external_approvals",
        summary="Split external target-evidence approval from search/acquisition scope.",
        mechanism="register_routes_wrapper",
    ),
    RouteExtensionSpec(
        extension_id="external_approval_error_compat",
        installer_name="install_external_approval_error_compat",
        module="ai4s_agent.external_approval_compat",
        summary="Normalize legacy external-approval error text.",
        mechanism="method_patch",
        depends_on=("external_approval_split",),
    ),
    RouteExtensionSpec(
        extension_id="durable_job_control",
        installer_name="install_durable_job_control",
        module="ai4s_agent.durable_job_control",
        summary="Install durable job lease, heartbeat, and cancellation control-plane methods.",
        mechanism="class_patch",
    ),
    RouteExtensionSpec(
        extension_id="project_scoped_jobs",
        installer_name="install_project_scoped_jobs",
        module="ai4s_agent.project_scoped_jobs",
        summary="Install project-scoped job state, log, and background-job methods.",
        mechanism="class_patch",
    ),
    RouteExtensionSpec(
        extension_id="json_rmw_locks",
        installer_name="install_json_rmw_locks",
        module="ai4s_agent.json_rmw_lock",
        summary="Install locked read-modify-write storage hot paths.",
        mechanism="method_patch",
    ),
    RouteExtensionSpec(
        extension_id="project_plan_route_guard",
        installer_name="install_project_plan_route_guard",
        module="ai4s_agent.project_plan_guard",
        summary="Guard project-scoped plan state before legacy route writes.",
        mechanism="method_patch",
    ),
    RouteExtensionSpec(
        extension_id="project_scoped_job_routes",
        installer_name="install_project_scoped_job_routes",
        module="ai4s_agent.project_job_routes",
        summary="Route job APIs through project-scoped job keys when project_id is present.",
        mechanism="view_function_override",
        depends_on=("project_plan_route_guard", "project_scoped_jobs"),
    ),
    RouteExtensionSpec(
        extension_id="project_scoped_plan_routes",
        installer_name="install_project_scoped_plan_routes",
        module="ai4s_agent.project_plan_routes",
        summary="Finalize project-scoped plan, gate approval, status, and retry routes.",
        mechanism="view_function_override",
        depends_on=("project_scoped_job_routes",),
    ),
    RouteExtensionSpec(
        extension_id="immutable_upload_assets",
        installer_name="install_immutable_upload_assets",
        module="ai4s_agent.upload_assets",
        summary="Route project uploads through immutable versioned upload assets.",
        mechanism="view_function_override",
    ),
    RouteExtensionSpec(
        extension_id="server_permission_routes",
        installer_name="install_server_permission_routes",
        module="ai4s_agent.server_permissions",
        summary="Add server-side permission grant and audit routes.",
        mechanism="register_routes_wrapper",
        depends_on=("immutable_upload_assets",),
    ),
    RouteExtensionSpec(
        extension_id="project_memory_permission_routes",
        installer_name="install_project_memory_permission_routes",
        module="ai4s_agent.project_memory_permissions",
        summary="Protect project memory writes with server-side grants.",
        mechanism="view_function_override",
        depends_on=("server_permission_routes",),
    ),
)


INSTALLER_NAMES: tuple[str, ...] = tuple(spec.installer_name for spec in ROUTE_EXTENSION_SPECS)


def api_route_extension_specs() -> tuple[RouteExtensionSpec, ...]:
    """Return observable route extension metadata in installation order."""

    return ROUTE_EXTENSION_SPECS


def api_route_installers() -> tuple[RouteInstaller, ...]:
    """Return route extension installers in dependency order.

    This keeps package initialization from becoming the place where route-layer
    monkeypatches are manually coordinated. `api.py` owns runtime dependency
    assembly and base route-module registration, while feature-specific route
    extensions live in separate modules and are installed through this registry.
    """

    from ai4s_agent.chat_context import install_chat_project_context
    from ai4s_agent.chat_runplan import install_chat_run_plan_routes
    from ai4s_agent.durable_job_control import install_durable_job_control
    from ai4s_agent.execution_confirmation import install_execution_confirmation_audit
    from ai4s_agent.execution_policy import install_execution_policy_registry
    from ai4s_agent.external_approval_compat import install_external_approval_error_compat
    from ai4s_agent.external_approvals import install_external_approval_split
    from ai4s_agent.json_rmw_lock import install_json_rmw_locks
    from ai4s_agent.phase3_executor import install_phase3_executor_support
    from ai4s_agent.project_job_routes import install_project_scoped_job_routes
    from ai4s_agent.project_memory_permissions import install_project_memory_permission_routes
    from ai4s_agent.project_plan_guard import install_project_plan_route_guard
    from ai4s_agent.project_plan_routes import install_project_scoped_plan_routes
    from ai4s_agent.project_scoped_jobs import install_project_scoped_jobs
    from ai4s_agent.server_permissions import install_server_permission_routes
    from ai4s_agent.snapshot_material import install_run_plan_executor_snapshot_builder
    from ai4s_agent.upload_assets import install_immutable_upload_assets

    installers_by_name: dict[str, RouteInstaller] = {
        "install_run_plan_executor_snapshot_builder": install_run_plan_executor_snapshot_builder,
        "install_execution_confirmation_audit": install_execution_confirmation_audit,
        "install_chat_project_context": install_chat_project_context,
        "install_chat_run_plan_routes": install_chat_run_plan_routes,
        "install_phase3_executor_support": install_phase3_executor_support,
        "install_execution_policy_registry": install_execution_policy_registry,
        "install_external_approval_split": install_external_approval_split,
        "install_external_approval_error_compat": install_external_approval_error_compat,
        "install_durable_job_control": install_durable_job_control,
        "install_project_scoped_jobs": install_project_scoped_jobs,
        "install_json_rmw_locks": install_json_rmw_locks,
        "install_project_plan_route_guard": install_project_plan_route_guard,
        "install_project_scoped_job_routes": install_project_scoped_job_routes,
        "install_project_scoped_plan_routes": install_project_scoped_plan_routes,
        "install_immutable_upload_assets": install_immutable_upload_assets,
        "install_server_permission_routes": install_server_permission_routes,
        "install_project_memory_permission_routes": install_project_memory_permission_routes,
    }
    return tuple(installers_by_name[spec.installer_name] for spec in ROUTE_EXTENSION_SPECS)


def install_api_route_extensions() -> None:
    """Install all route extensions exactly once in registry order."""

    for installer in api_route_installers():
        installer()
