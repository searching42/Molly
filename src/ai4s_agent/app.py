from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from flask import Flask, jsonify

from ai4s_agent.api_route_extensions import (
    apply_explicit_route_hooks,
    api_route_extension_specs,
    route_extension_context,
)
from ai4s_agent.api import register_routes
from ai4s_agent.profiles import route_extension_inspection_enabled, selected_profile


def create_app(base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config.setdefault("AI4S_PROFILE", selected_profile())
    register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
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
