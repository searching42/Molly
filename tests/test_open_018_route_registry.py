from __future__ import annotations

import json

from ai4s_agent.api_route_extensions import INSTALLER_NAMES, api_route_installers
import ai4s_agent.api_route_extensions as route_extensions
import ai4s_agent.app as app_module
from ai4s_agent.app import create_app
from ai4s_agent.routes.agents import register_agent_routes
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


def test_api_route_extension_metadata_is_observable_and_ordered() -> None:
    specs = route_extensions.api_route_extension_specs()

    assert tuple(spec.installer_name for spec in specs) == INSTALLER_NAMES
    assert len({spec.extension_id for spec in specs}) == len(specs)
    assert len({spec.installer_name for spec in specs}) == len(specs)
    assert all(spec.module.startswith("ai4s_agent.") for spec in specs)
    assert all(spec.summary for spec in specs)
    assert {spec.mechanism for spec in specs} <= {
        "class_patch",
        "method_patch",
        "register_routes_wrapper",
        "view_function_override",
    }

    seen: set[str] = set()
    for spec in specs:
        assert set(spec.depends_on) <= seen
        seen.add(spec.extension_id)

    payload = [spec.as_dict() for spec in specs]
    assert json.loads(json.dumps(payload))[0]["extension_id"] == specs[0].extension_id


def test_route_extension_metadata_declares_explicit_hook_skeleton() -> None:
    specs = route_extensions.api_route_extension_specs()
    by_id = {spec.extension_id: spec.as_dict() for spec in specs}

    upload = by_id["immutable_upload_assets"]
    assert upload["explicit_hook_capable"] is True
    assert upload["explicit_hook_active"] is False
    assert upload["declared_route_overrides"] == [{"endpoint": "upload_file"}]
    assert upload["declared_new_routes"] == []

    permissions = by_id["server_permission_routes"]
    assert permissions["explicit_hook_capable"] is True
    assert permissions["explicit_hook_active"] is False
    assert permissions["declared_route_overrides"] == []
    assert permissions["declared_new_routes"] == [
        {
            "endpoint": "create_permission_grant",
            "rule": "/api/projects/<project_id>/permissions/grants",
            "methods": ["POST"],
        },
        {
            "endpoint": "list_permission_grants",
            "rule": "/api/projects/<project_id>/permissions/grants",
            "methods": ["GET"],
        },
        {
            "endpoint": "list_permission_audit",
            "rule": "/api/projects/<project_id>/permissions/audit",
            "methods": ["GET"],
        },
    ]


def test_create_app_exposes_installed_route_extension_metadata(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)

    configured = app.config["AI4S_ROUTE_EXTENSIONS"]
    installed = app_module.installed_route_extensions(app)

    assert isinstance(configured, tuple)
    assert installed == configured
    assert tuple(item["installer_name"] for item in installed) == INSTALLER_NAMES
    assert json.loads(json.dumps(installed))[0]["extension_id"] == installed[0]["extension_id"]

    installed[0]["extension_id"] = "mutated"
    installed[0]["depends_on"].append("mutated")
    fresh = app_module.installed_route_extensions(app)
    assert fresh[0]["extension_id"] != "mutated"
    assert "mutated" not in fresh[0]["depends_on"]


def test_create_app_exposes_route_override_registry_metadata(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)

    configured = app.config["AI4S_ROUTE_OVERRIDE_REGISTRY"]
    registry = app_module.route_override_registry(app)

    assert registry == configured
    assert json.loads(json.dumps(registry))["route_overrides"][0]["endpoint"]
    overrides = {
        (item["extension_id"], item["endpoint"])
        for item in registry["route_overrides"]
    }
    new_routes = {
        (item["extension_id"], item["endpoint"], item["rule"], tuple(item["methods"]))
        for item in registry["new_routes"]
    }

    assert ("immutable_upload_assets", "upload_file") in overrides
    assert (
        "server_permission_routes",
        "create_permission_grant",
        "/api/projects/<project_id>/permissions/grants",
        ("POST",),
    ) in new_routes


def test_route_extension_inspection_endpoint_reports_route_ownership(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.get("/api/system/route-extensions")

    assert response.status_code == 200
    body = response.json
    assert body["ok"] is True
    assert tuple(item["installer_name"] for item in body["extensions"]) == INSTALLER_NAMES
    assert body["route_override_registry"] == app_module.route_override_registry(app)
    assert json.loads(json.dumps(body["routes"]))[0]["rule"]

    by_rule = {item["rule"]: item for item in body["routes"]}
    assert by_rule["/api/plan"]["endpoint"] == "create_plan"
    assert by_rule["/api/plan"]["owner_extension_id"] == "project_scoped_plan_routes"
    assert by_rule["/api/plan"]["owner_module"] == "ai4s_agent.project_plan_routes"
    assert by_rule["/api/projects/<project_id>/upload"]["owner_extension_id"] == "immutable_upload_assets"
    assert by_rule["/api/agent/modeling-plan"]["owner_extension_id"] == "external_approval_split"
    assert by_rule["/api/system/route-extensions"]["owner_extension_id"] == ""
    assert by_rule["/api/system/route-extensions"]["owner_module"] == "ai4s_agent.app"
    assert by_rule["/api/system/route-extensions"]["methods"] == ["GET"]

    upload = next(item for item in body["extensions"] if item["extension_id"] == "immutable_upload_assets")
    assert upload["explicit_hook_capable"] is True
    assert upload["explicit_hook_active"] is False


def test_low_coupling_base_routes_are_registered_from_route_module(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)

    assert callable(register_agent_routes)
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
    assert "agent_plan_proposal" in app.view_functions
    assert "agent_replan" in app.view_functions
    assert "agent_research_sources" in app.view_functions
    assert "agent_conversation_research_sources" in app.view_functions
    assert "agent_research_acquisition_prepare" in app.view_functions
    assert "agent_conversation_modeling_payload" in app.view_functions
    assert "agent_conversation_next_turn" in app.view_functions
    assert "agent_modeling_plan" in app.view_functions
    assert "agent_model_package_review" in app.view_functions
    assert "agent_generation_plan" in app.view_functions
    assert "agent_report_summary" in app.view_functions
    assert "agent_review_card" in app.view_functions
    assert "agent_decision_card" in app.view_functions
