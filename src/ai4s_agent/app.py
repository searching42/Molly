from __future__ import annotations

from pathlib import Path

from flask import Flask

from ai4s_agent.api import register_routes


def create_app(base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> Flask:
    app = Flask(__name__)
    register_routes(app, base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
    return app
