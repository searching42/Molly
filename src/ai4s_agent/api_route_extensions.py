from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RouteInstaller = Callable[[], None]


@dataclass(frozen=True)
class RouteOverrideDeclaration:
    endpoint: str

    def as_dict(self) -> dict[str, object]:
        return {"endpoint": self.endpoint}


@dataclass(frozen=True)
class RouteRegistrationDeclaration:
    endpoint: str
    rule: str
    methods: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "rule": self.rule,
            "methods": [method.upper() for method in self.methods],
        }


@dataclass(frozen=True)
class RouteExtensionSpec:
    extension_id: str
    installer_name: str
    module: str
    summary: str
    mechanism: str
    depends_on: tuple[str, ...] = ()
    explicit_hook_active: bool = False
    declared_route_overrides: tuple[RouteOverrideDeclaration, ...] = ()
    declared_new_routes: tuple[RouteRegistrationDeclaration, ...] = ()

    @property
    def explicit_hook_capable(self) -> bool:
        return bool(self.declared_route_overrides or self.declared_new_routes)

    def as_dict(self) -> dict[str, object]:
        return {
            "extension_id": self.extension_id,
            "installer_name": self.installer_name,
            "module": self.module,
            "summary": self.summary,
            "mechanism": self.mechanism,
            "depends_on": list(self.depends_on),
            "explicit_hook_capable": self.explicit_hook_capable,
            "explicit_hook_active": self.explicit_hook_active,
            "declared_route_overrides": [
                item.as_dict()
                for item in self.declared_route_overrides
            ],
            "declared_new_routes": [
                item.as_dict()
                for item in self.declared_new_routes
            ],
        }


@dataclass(frozen=True)
class RouteExtensionContext:
    app: Any
    base_runs_dir: Path | None
    workspace_dir: Path | None
    route_overrides: "RouteOverrideRegistry"


class RouteOverrideRegistry:
    """Track declared and applied explicit route extension hooks."""

    def __init__(self) -> None:
        self._route_overrides: list[dict[str, object]] = []
        self._new_routes: list[dict[str, object]] = []
        self._applied_route_overrides: list[dict[str, object]] = []
        self._applied_new_routes: list[dict[str, object]] = []

    @classmethod
    def from_specs(cls, specs: tuple[RouteExtensionSpec, ...]) -> "RouteOverrideRegistry":
        registry = cls()
        for spec in specs:
            for declaration in spec.declared_route_overrides:
                registry.declare_route_override(
                    spec.extension_id,
                    endpoint=declaration.endpoint,
                    explicit_hook_active=spec.explicit_hook_active,
                )
            for declaration in spec.declared_new_routes:
                registry.declare_new_route(
                    spec.extension_id,
                    endpoint=declaration.endpoint,
                    rule=declaration.rule,
                    methods=declaration.methods,
                    explicit_hook_active=spec.explicit_hook_active,
                )
        return registry

    def declare_route_override(
        self,
        extension_id: str,
        *,
        endpoint: str,
        explicit_hook_active: bool = False,
    ) -> None:
        self._route_overrides.append(
            {
                "extension_id": extension_id,
                "endpoint": endpoint,
                "explicit_hook_active": bool(explicit_hook_active),
            }
        )

    def declare_new_route(
        self,
        extension_id: str,
        *,
        endpoint: str,
        rule: str,
        methods: tuple[str, ...],
        explicit_hook_active: bool = False,
    ) -> None:
        self._new_routes.append(
            {
                "extension_id": extension_id,
                "endpoint": endpoint,
                "rule": rule,
                "methods": [method.upper() for method in methods],
                "explicit_hook_active": bool(explicit_hook_active),
            }
        )

    def apply_route_override(
        self,
        app: Any,
        *,
        extension_id: str,
        endpoint: str,
        view_func: Any,
    ) -> None:
        if endpoint not in app.view_functions:
            raise RuntimeError(f"route endpoint not registered before override: {endpoint}")
        app.view_functions[endpoint] = view_func
        self._applied_route_overrides.append(
            {
                "extension_id": extension_id,
                "endpoint": endpoint,
            }
        )

    def apply_new_route(
        self,
        app: Any,
        *,
        extension_id: str,
        endpoint: str,
        rule: str,
        view_func: Any,
        methods: tuple[str, ...],
    ) -> None:
        if endpoint in app.view_functions:
            raise RuntimeError(
                f"route endpoint already registered before explicit hook: {endpoint}"
            )
        normalized_methods = tuple(method.upper() for method in methods)
        app.add_url_rule(
            rule,
            endpoint=endpoint,
            view_func=view_func,
            methods=list(normalized_methods),
        )
        self._applied_new_routes.append(
            {
                "extension_id": extension_id,
                "endpoint": endpoint,
                "rule": rule,
                "methods": list(normalized_methods),
            }
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "route_overrides": [dict(item) for item in self._route_overrides],
            "new_routes": [dict(item) for item in self._new_routes],
            "applied_route_overrides": [dict(item) for item in self._applied_route_overrides],
            "applied_new_routes": [dict(item) for item in self._applied_new_routes],
        }


def route_extension_context(
    *,
    app: Any,
    base_runs_dir: Path | None,
    workspace_dir: Path | None,
    specs: tuple[RouteExtensionSpec, ...] | None = None,
) -> RouteExtensionContext:
    selected_specs = specs if specs is not None else api_route_extension_specs()
    return RouteExtensionContext(
        app=app,
        base_runs_dir=base_runs_dir,
        workspace_dir=workspace_dir,
        route_overrides=RouteOverrideRegistry.from_specs(selected_specs),
    )


def apply_explicit_route_hooks(context: RouteExtensionContext) -> None:
    """Apply route extensions that have migrated from monkeypatches to hooks."""

    from ai4s_agent.project_job_routes import apply_project_scoped_job_routes
    from ai4s_agent.project_memory_permissions import apply_project_memory_permission_routes
    from ai4s_agent.project_plan_routes import apply_project_scoped_plan_routes
    from ai4s_agent.server_permissions import apply_server_permission_routes
    from ai4s_agent.upload_assets import apply_immutable_upload_assets_route_override

    hooks: dict[str, Callable[[RouteExtensionContext], None]] = {
        "project_scoped_job_routes": apply_project_scoped_job_routes,
        "project_scoped_plan_routes": apply_project_scoped_plan_routes,
        "immutable_upload_assets": apply_immutable_upload_assets_route_override,
        "server_permission_routes": apply_server_permission_routes,
        "project_memory_permission_routes": apply_project_memory_permission_routes,
    }
    for spec in api_route_extension_specs():
        if not spec.explicit_hook_active:
            continue
        hook = hooks.get(spec.extension_id)
        if hook is None:
            raise RuntimeError(f"missing explicit route hook for extension: {spec.extension_id}")
        hook(context)


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
        summary="Provide project-scoped plan key validation and rollback helpers.",
        mechanism="helper_module",
    ),
    RouteExtensionSpec(
        extension_id="project_scoped_job_routes",
        installer_name="install_project_scoped_job_routes",
        module="ai4s_agent.project_job_routes",
        summary="Route job APIs through project-scoped job keys when project_id is present.",
        mechanism="explicit_route_override",
        explicit_hook_active=True,
        depends_on=("project_plan_route_guard", "project_scoped_jobs"),
        declared_route_overrides=(
            RouteOverrideDeclaration(endpoint="run_logs"),
            RouteOverrideDeclaration(endpoint="pause_run"),
            RouteOverrideDeclaration(endpoint="resume_run"),
            RouteOverrideDeclaration(endpoint="stop_run"),
            RouteOverrideDeclaration(endpoint="create_background_job"),
            RouteOverrideDeclaration(endpoint="get_background_job"),
            RouteOverrideDeclaration(endpoint="record_background_checkpoint"),
            RouteOverrideDeclaration(endpoint="background_resume_plan"),
            RouteOverrideDeclaration(endpoint="retry_run"),
            RouteOverrideDeclaration(endpoint="list_jobs"),
        ),
        declared_new_routes=(
            RouteRegistrationDeclaration(
                endpoint="project_run_logs",
                rule="/api/projects/<project_id>/runs/<run_id>/logs",
                methods=("GET",),
            ),
            RouteRegistrationDeclaration(
                endpoint="project_pause_run",
                rule="/api/projects/<project_id>/runs/<run_id>/pause",
                methods=("POST",),
            ),
            RouteRegistrationDeclaration(
                endpoint="project_resume_run",
                rule="/api/projects/<project_id>/runs/<run_id>/resume",
                methods=("POST",),
            ),
            RouteRegistrationDeclaration(
                endpoint="project_stop_run",
                rule="/api/projects/<project_id>/runs/<run_id>/stop",
                methods=("POST",),
            ),
        ),
    ),
    RouteExtensionSpec(
        extension_id="project_scoped_plan_routes",
        installer_name="install_project_scoped_plan_routes",
        module="ai4s_agent.project_plan_routes",
        summary="Finalize project-scoped plan, gate approval, and status routes.",
        mechanism="explicit_route_override",
        explicit_hook_active=True,
        depends_on=("project_scoped_job_routes",),
        declared_route_overrides=(
            RouteOverrideDeclaration(endpoint="create_plan"),
            RouteOverrideDeclaration(endpoint="approve_gate"),
        ),
        declared_new_routes=(
            RouteRegistrationDeclaration(
                endpoint="project_run_status",
                rule="/api/projects/<project_id>/runs/<run_id>/status",
                methods=("GET",),
            ),
        ),
    ),
    RouteExtensionSpec(
        extension_id="immutable_upload_assets",
        installer_name="install_immutable_upload_assets",
        module="ai4s_agent.upload_assets",
        summary="Route project uploads through immutable versioned upload assets.",
        mechanism="explicit_route_override",
        explicit_hook_active=True,
        declared_route_overrides=(RouteOverrideDeclaration(endpoint="upload_file"),),
    ),
    RouteExtensionSpec(
        extension_id="server_permission_routes",
        installer_name="install_server_permission_routes",
        module="ai4s_agent.server_permissions",
        summary="Add server-side permission grant and audit routes.",
        mechanism="explicit_route_registration",
        explicit_hook_active=True,
        depends_on=("immutable_upload_assets",),
        declared_new_routes=(
            RouteRegistrationDeclaration(
                endpoint="create_permission_grant",
                rule="/api/projects/<project_id>/permissions/grants",
                methods=("POST",),
            ),
            RouteRegistrationDeclaration(
                endpoint="list_permission_grants",
                rule="/api/projects/<project_id>/permissions/grants",
                methods=("GET",),
            ),
            RouteRegistrationDeclaration(
                endpoint="list_permission_audit",
                rule="/api/projects/<project_id>/permissions/audit",
                methods=("GET",),
            ),
        ),
    ),
    RouteExtensionSpec(
        extension_id="project_memory_permission_routes",
        installer_name="install_project_memory_permission_routes",
        module="ai4s_agent.project_memory_permissions",
        summary="Protect project memory writes with server-side grants.",
        mechanism="explicit_route_override",
        explicit_hook_active=True,
        depends_on=("server_permission_routes",),
        declared_route_overrides=(
            RouteOverrideDeclaration(endpoint="create_project_memory_record"),
            RouteOverrideDeclaration(endpoint="update_project_memory_record"),
            RouteOverrideDeclaration(endpoint="delete_project_memory_record"),
            RouteOverrideDeclaration(endpoint="set_project_memory_enabled"),
        ),
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
