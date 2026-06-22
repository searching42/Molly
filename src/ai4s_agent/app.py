from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify

from ai4s_agent.api_route_extensions import api_route_extension_specs
from ai4s_agent.api import register_routes


def create_app(base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
    app.config["AI4S_ROUTE_EXTENSIONS"] = tuple(spec.as_dict() for spec in api_route_extension_specs())
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
        record = dict(item)
        depends_on = record.get("depends_on")
        record["depends_on"] = list(depends_on) if isinstance(depends_on, list) else []
        copied.append(record)
    return tuple(copied)


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
        return jsonify(
            {
                "ok": True,
                "extensions": list(installed_route_extensions(app)),
                "routes": list(route_ownership(app)),
            }
        )
