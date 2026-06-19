from __future__ import annotations

from collections.abc import Callable


RouteInstaller = Callable[[], None]


INSTALLER_NAMES: tuple[str, ...] = (
    "install_run_plan_executor_snapshot_builder",
    "install_execution_confirmation_audit",
    "install_chat_project_context",
    "install_chat_run_plan_routes",
    "install_phase3_executor_support",
    "install_execution_policy_registry",
    "install_external_approval_split",
    "install_external_approval_error_compat",
    "install_durable_job_control",
    "install_project_scoped_jobs",
    "install_json_rmw_locks",
    "install_project_plan_route_guard",
    "install_project_scoped_job_routes",
    "install_project_scoped_plan_routes",
    "install_immutable_upload_assets",
    "install_server_permission_routes",
    "install_project_memory_permission_routes",
)


def api_route_installers() -> tuple[RouteInstaller, ...]:
    """Return route extension installers in dependency order.

    This keeps package initialization from becoming the place where route-layer
    monkeypatches are manually coordinated. `api.py` still owns the legacy base
    routes, while feature-specific route extensions live in separate modules and
    are installed through this registry.
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

    return (
        install_run_plan_executor_snapshot_builder,
        install_execution_confirmation_audit,
        install_chat_project_context,
        install_chat_run_plan_routes,
        install_phase3_executor_support,
        install_execution_policy_registry,
        install_external_approval_split,
        install_external_approval_error_compat,
        install_durable_job_control,
        install_project_scoped_jobs,
        install_json_rmw_locks,
        install_project_plan_route_guard,
        install_project_scoped_job_routes,
        install_project_scoped_plan_routes,
        install_immutable_upload_assets,
        install_server_permission_routes,
        install_project_memory_permission_routes,
    )


def install_api_route_extensions() -> None:
    """Install all route extensions exactly once in registry order."""

    for installer in api_route_installers():
        installer()
