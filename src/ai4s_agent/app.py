from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from ai4s_agent.api_route_extensions import api_route_extension_specs
from ai4s_agent.api import register_routes


def create_app(base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
    app.config["AI4S_ROUTE_EXTENSIONS"] = tuple(spec.as_dict() for spec in api_route_extension_specs())
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
