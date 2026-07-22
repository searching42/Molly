from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask

from ai4s_agent.job_manager import JobManager
from ai4s_agent.memory import PermissionPolicy, ProjectMemory
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.routes import run_control as run_control_routes
from ai4s_agent.routes.agents import _as_bool, register_agent_routes
from ai4s_agent.routes.core import register_core_routes
from ai4s_agent.routes.internal_run_plan_queue import register_internal_run_plan_queue_routes
from ai4s_agent.routes.jobs import register_job_routes
from ai4s_agent.routes.legacy_plan import register_legacy_plan_routes
from ai4s_agent.routes.oled_bounded_sessions import register_oled_bounded_session_routes
from ai4s_agent.routes.project_assets import register_project_asset_routes
from ai4s_agent.routes.project_runs import register_project_run_routes
from ai4s_agent.routes.projects import register_project_routes
from ai4s_agent.routes.review import register_review_routes
from ai4s_agent.routes.run_plans import register_run_plan_routes
from ai4s_agent.routes.worker_deployment import register_worker_deployment_routes
from ai4s_agent.storage import ProjectStorage


DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
DEFAULT_WORKSPACE = Path(__file__).resolve().parents[2]
ALLOWED_EXTENSIONS = {"csv", "json", "sdf", "mol", "smi"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
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


def register_routes(app: Flask, base_runs_dir: Path | None = None, workspace_dir: Path | None = None) -> None:
    runs = Path(base_runs_dir or DEFAULT_RUNS_DIR).resolve()
    workspace = _workspace_from_config(base_runs_dir=base_runs_dir, workspace_dir=workspace_dir)
    orch = Orchestrator(base_runs_dir=runs)
    jobs = JobManager(runs_dir=runs)
    projects = ProjectStorage(workspace_dir=workspace)
    project_memory = ProjectMemory(workspace_dir=workspace)
    permissions = PermissionPolicy()
    bounded_session_actions = OledBoundedDiscoverySessionActionService(
        storage=projects,
        actions_root=runs / "oled-bounded-session-actions",
    )
    app.extensions["oled_bounded_session_actions"] = bounded_session_actions

    register_core_routes(app)
    register_legacy_plan_routes(app, orch=orch, jobs=jobs)
    register_run_plan_routes(app, projects=projects, jobs=jobs)
    register_internal_run_plan_queue_routes(app, projects=projects)
    register_agent_routes(app, projects=projects, project_memory=project_memory, jobs=jobs)
    register_worker_deployment_routes(app, workspace=workspace, runs=runs)
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
