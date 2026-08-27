from __future__ import annotations

import copy
import os
import weakref
from pathlib import Path
from typing import Any

from flask import Flask, jsonify

from ai4s_agent.api_route_extensions import (
    apply_explicit_route_hooks,
    api_route_extension_specs,
    route_extension_context,
)
from ai4s_agent.api import register_routes
from ai4s_agent.local_security import install_local_request_protection
from ai4s_agent.planner import (
    AtomicTaskRegistry,
    br2_contextual_mapping_task_registry_v1,
    private_structured_dataset_real_tool_task_registry_v3,
)
from ai4s_agent.profiles import route_extension_inspection_enabled, selected_profile


def create_app(
    base_runs_dir: Path | None = None,
    workspace_dir: Path | None = None,
    user_config_dir: Path | None = None,
    scientific_task_registry: AtomicTaskRegistry | None = None,
) -> Flask:
    app = Flask(__name__)
    app.config.setdefault("AI4S_PROFILE", selected_profile())
    app.config.setdefault(
        "AI4S_AGENT_EXECUTION_AGENT_V2_ENABLED",
        os.environ.get("AI4S_AGENT_EXECUTION_AGENT_V2_ENABLED", "false"),
    )
    app.config.setdefault(
        "AI4S_AGENT_FAILURE_RECOVERY_ENABLED",
        os.environ.get("AI4S_AGENT_FAILURE_RECOVERY_ENABLED", "false"),
    )
    # Recovery retry/replan limits remain independent from the shared
    # server-issued AutonomyGrant and autonomy lease.  A zero recovery budget
    # disables recovery continuation; it must not disable lease enforcement
    # for ordinary bounded automatic effects.
    app.config.setdefault(
        "AI4S_AGENT_FAILURE_RECOVERY_MAX_RETRIES",
        os.environ.get("AI4S_AGENT_FAILURE_RECOVERY_MAX_RETRIES", "1"),
    )
    app.config.setdefault(
        "AI4S_AGENT_FAILURE_RECOVERY_MAX_REPLANS",
        os.environ.get("AI4S_AGENT_FAILURE_RECOVERY_MAX_REPLANS", "1"),
    )
    app.config.setdefault(
        "AI4S_AGENT_FAILURE_RECOVERY_GRANT_TTL_SECONDS",
        os.environ.get("AI4S_AGENT_FAILURE_RECOVERY_GRANT_TTL_SECONDS", "86400"),
    )
    app.config.setdefault(
        "AI4S_AGENT_AUTONOMY_LEASE_ENABLED",
        os.environ.get("AI4S_AGENT_AUTONOMY_LEASE_ENABLED", "true"),
    )
    app.config.setdefault(
        "AI4S_AGENT_AUTONOMY_LEASE_TTL_SECONDS",
        os.environ.get("AI4S_AGENT_AUTONOMY_LEASE_TTL_SECONDS", "3600"),
    )
    app.config.setdefault(
        "AI4S_AGENT_AUTONOMY_MAX_ACTIVE_EXECUTION_SECONDS",
        os.environ.get("AI4S_AGENT_AUTONOMY_MAX_ACTIVE_EXECUTION_SECONDS", "900"),
    )
    app.config.setdefault(
        "AI4S_AGENT_AUTONOMY_MAX_REMOTE_RUNTIME_SECONDS",
        os.environ.get("AI4S_AGENT_AUTONOMY_MAX_REMOTE_RUNTIME_SECONDS", "900"),
    )
    app.config.setdefault(
        "AI4S_AGENT_AUTONOMY_OPERATION_RESERVATION_SECONDS",
        os.environ.get("AI4S_AGENT_AUTONOMY_OPERATION_RESERVATION_SECONDS", "300"),
    )
    app.config.setdefault(
        "AI4S_AGENT_AUTONOMY_REMOTE_RESERVATION_SECONDS",
        os.environ.get("AI4S_AGENT_AUTONOMY_REMOTE_RESERVATION_SECONDS", "300"),
    )
    if scientific_task_registry is None:
        registry_id = os.environ.get("AI4S_SCIENTIFIC_TASK_REGISTRY", "").strip()
        if registry_id == "br1-private-real-tool-v3":
            scientific_task_registry = (
                private_structured_dataset_real_tool_task_registry_v3()
            )
        elif registry_id == "br2-contextual-mapping-v1":
            scientific_task_registry = br2_contextual_mapping_task_registry_v1()
        elif registry_id:
            raise ValueError("unknown server-owned scientific task registry")
    install_local_request_protection(app)
    register_routes(
        app,
        base_runs_dir=base_runs_dir,
        workspace_dir=workspace_dir,
        user_config_dir=user_config_dir,
        scientific_task_registry=scientific_task_registry,
    )
    llm_provider_manager = app.extensions.get("llm_provider_manager")
    if llm_provider_manager is not None:
        app.extensions["llm_provider_finalizer"] = weakref.finalize(
            app,
            llm_provider_manager.close,
        )
    harness_tracer = app.extensions.get("harness_tracer")
    if harness_tracer is not None:
        app.extensions["harness_tracer_finalizer"] = weakref.finalize(
            app,
            harness_tracer.shutdown,
        )
    extension_context = route_extension_context(
        app=app,
        base_runs_dir=base_runs_dir,
        workspace_dir=workspace_dir,
    )
    apply_explicit_route_hooks(extension_context)
    app.config["AI4S_ROUTE_EXTENSIONS"] = tuple(spec.as_dict() for spec in api_route_extension_specs())
    app.config["AI4S_ROUTE_OVERRIDE_REGISTRY"] = extension_context.route_overrides.as_dict()
    register_route_inspection(app)
    return app


def installed_route_extensions(app: Flask) -> tuple[dict[str, Any], ...]:
    """Return JSON-safe installed route extension metadata for this app."""

    raw = app.config.get("AI4S_ROUTE_EXTENSIONS", ())
    if not isinstance(raw, tuple):
        return ()
    copied: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        record = copy.deepcopy(item)
        depends_on = record.get("depends_on")
        record["depends_on"] = list(depends_on) if isinstance(depends_on, list) else []
        copied.append(record)
    return tuple(copied)


def route_override_registry(app: Flask) -> dict[str, Any]:
    """Return JSON-safe explicit route hook declaration metadata."""

    raw = app.config.get("AI4S_ROUTE_OVERRIDE_REGISTRY", {})
    if not isinstance(raw, dict):
        return {"route_overrides": [], "new_routes": []}
    registry = copy.deepcopy(raw)
    route_overrides = registry.get("route_overrides")
    new_routes = registry.get("new_routes")
    applied_route_overrides = registry.get("applied_route_overrides")
    applied_new_routes = registry.get("applied_new_routes")
    registry["route_overrides"] = list(route_overrides) if isinstance(route_overrides, list) else []
    registry["new_routes"] = list(new_routes) if isinstance(new_routes, list) else []
    registry["applied_route_overrides"] = (
        list(applied_route_overrides)
        if isinstance(applied_route_overrides, list)
        else []
    )
    registry["applied_new_routes"] = (
        list(applied_new_routes)
        if isinstance(applied_new_routes, list)
        else []
    )
    return registry


def route_ownership(app: Flask) -> tuple[dict[str, Any], ...]:
    """Return read-only route ownership metadata for this app."""

    extension_by_module = {
        str(item.get("module") or ""): str(item.get("extension_id") or "")
        for item in installed_route_extensions(app)
    }
    routes: list[dict[str, Any]] = []
    for rule in sorted(app.url_map.iter_rules(), key=lambda item: (item.rule, item.endpoint)):
        view = app.view_functions.get(rule.endpoint)
        owner_module = str(getattr(view, "__module__", "") or "")
        owner_qualname = str(getattr(view, "__qualname__", getattr(view, "__name__", "")) or "")
        owner_extension_id = extension_by_module.get(owner_module, "")
        routes.append(
            {
                "rule": rule.rule,
                "endpoint": rule.endpoint,
                "methods": sorted(method for method in rule.methods if method not in {"HEAD", "OPTIONS"}),
                "owner_module": owner_module,
                "owner_qualname": owner_qualname,
                "owner_extension_id": owner_extension_id,
                "owner_kind": "extension" if owner_extension_id else "base",
            }
        )
    return tuple(routes)


def register_route_inspection(app: Flask) -> None:
    @app.get("/api/system/route-extensions")
    def inspect_route_extensions():
        if not route_extension_inspection_enabled(app):
            return jsonify({"ok": False, "error": "route extension inspection disabled"}), 404
        return jsonify(
            {
                "ok": True,
                "extensions": list(installed_route_extensions(app)),
                "route_override_registry": route_override_registry(app),
                "routes": list(route_ownership(app)),
            }
        )
