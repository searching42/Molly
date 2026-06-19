from __future__ import annotations

from ai4s_agent.api_route_extensions import INSTALLER_NAMES, api_route_installers


def test_api_route_extension_registry_preserves_order() -> None:
    installers = api_route_installers()
    assert tuple(installer.__name__ for installer in installers) == INSTALLER_NAMES
    assert INSTALLER_NAMES.index("install_project_plan_route_guard") < INSTALLER_NAMES.index("install_project_scoped_job_routes")
    assert INSTALLER_NAMES.index("install_project_scoped_job_routes") < INSTALLER_NAMES.index("install_project_scoped_plan_routes")
    assert INSTALLER_NAMES.index("install_immutable_upload_assets") < INSTALLER_NAMES.index("install_server_permission_routes")
    assert INSTALLER_NAMES.index("install_server_permission_routes") < INSTALLER_NAMES.index("install_project_memory_permission_routes")
