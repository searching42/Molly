from __future__ import annotations

from ai4s_agent.api_route_extensions import INSTALLER_NAMES, api_route_installers
from ai4s_agent.app import create_app
from ai4s_agent.routes.core import register_core_routes
from ai4s_agent.routes.jobs import register_job_routes
from ai4s_agent.routes.legacy_plan import register_legacy_plan_routes
from ai4s_agent.routes.project_assets import register_project_asset_routes
from ai4s_agent.routes.project_runs import register_project_run_routes
from ai4s_agent.routes.projects import register_project_routes
from ai4s_agent.routes.review import register_review_routes
from ai4s_agent.routes.run_control import register_run_control_routes
from ai4s_agent.routes.run_plans import register_run_plan_routes
from ai4s_agent.routes.worker_deployment import register_worker_deployment_routes


def test_api_route_extension_registry_preserves_order() -> None:
    installers = api_route_installers()
    assert tuple(installer.__name__ for installer in installers) == INSTALLER_NAMES
    assert INSTALLER_NAMES.index("install_project_plan_route_guard") < INSTALLER_NAMES.index("install_project_scoped_job_routes")
    assert INSTALLER_NAMES.index("install_project_scoped_job_routes") < INSTALLER_NAMES.index("install_project_scoped_plan_routes")
    assert INSTALLER_NAMES.index("install_immutable_upload_assets") < INSTALLER_NAMES.index("install_server_permission_routes")
    assert INSTALLER_NAMES.index("install_server_permission_routes") < INSTALLER_NAMES.index("install_project_memory_permission_routes")


def test_low_coupling_base_routes_are_registered_from_route_module(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)

    assert callable(register_core_routes)
    assert callable(register_job_routes)
    assert callable(register_legacy_plan_routes)
    assert callable(register_project_asset_routes)
    assert callable(register_project_run_routes)
    assert callable(register_project_routes)
    assert callable(register_review_routes)
    assert callable(register_run_control_routes)
    assert callable(register_run_plan_routes)
    assert callable(register_worker_deployment_routes)
    assert "index" in app.view_functions
    assert "healthz" in app.view_functions
    assert "data_confirmation_card" in app.view_functions
    assert "run_confirmation_card" in app.view_functions
    assert "resolve_permission" in app.view_functions
    assert "create_plan" in app.view_functions
    assert "create_project" in app.view_functions
    assert "list_projects" in app.view_functions
    assert "list_project_memory" in app.view_functions
    assert "create_project_memory_record" in app.view_functions
    assert "update_project_memory_record" in app.view_functions
    assert "delete_project_memory_record" in app.view_functions
    assert "export_project_memory" in app.view_functions
    assert "set_project_memory_enabled" in app.view_functions
    assert "upload_file" in app.view_functions
    assert "register_model" in app.view_functions
    assert "draft_promoted_model_asset" in app.view_functions
    assert "promote_model_asset" in app.view_functions
    assert "promote_asset" in app.view_functions
    assert "expand_plan_preview" in app.view_functions
    assert "diff_plan_preview" in app.view_functions
    assert "regenerate_plan_preview" in app.view_functions
    assert "execute_run_plan" in app.view_functions
    assert "resume_run_plan" in app.view_functions
    assert "approve_gate" in app.view_functions
    assert "run_status" in app.view_functions
    assert "execute_adapter" in app.view_functions
    assert "stage_timeline" in app.view_functions
    assert "report_preview" in app.view_functions
    assert "verify_project_run" in app.view_functions
    assert "run_logs" in app.view_functions
    assert "pause_run" in app.view_functions
    assert "resume_run" in app.view_functions
    assert "stop_run" in app.view_functions
    assert "create_background_job" in app.view_functions
    assert "get_background_job" in app.view_functions
    assert "record_background_checkpoint" in app.view_functions
    assert "background_resume_plan" in app.view_functions
    assert "retry_run" in app.view_functions
    assert "list_jobs" in app.view_functions
    assert "list_remote_workers" in app.view_functions
    assert "save_remote_worker" in app.view_functions
    assert "plan_remote_worker_assignment" in app.view_functions
    assert "multi_user_readiness" in app.view_functions
    assert "list_atomic_tasks" in app.view_functions
