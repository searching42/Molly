from __future__ import annotations

from typing import Any

import pytest


def _project_id_from_memory_path(path: str) -> str:
    marker = "/api/projects/"
    if marker not in path:
        return ""
    rest = path.split(marker, 1)[1]
    return rest.split("/", 1)[0].strip()


@pytest.fixture(autouse=True)
def _grant_project_memory_for_legacy_api_smoke(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the broad API smoke test aligned with the server grant boundary.

    The smoke test predates OPEN-016 and exercises project memory together with
    planner prefill. Rather than weakening production defaults, this fixture
    grants only that legacy smoke path the same project_memory_write permission a
    real caller must create explicitly.
    """

    if request.node.name != "test_project_memory_governance_endpoints_and_plan_prefill":
        return
    module = request.module
    original_create_app = getattr(module, "create_app", None)
    if not callable(original_create_app):
        return

    def create_app_with_memory_grant(*args: Any, **kwargs: Any):
        app = original_create_app(*args, **kwargs)
        original_test_client = app.test_client

        def test_client_with_memory_grant(*client_args: Any, **client_kwargs: Any):
            client = original_test_client(*client_args, **client_kwargs)
            original_post = client.post
            original_delete = client.delete
            granted_projects: set[str] = set()

            def ensure_grant(path: str) -> None:
                project_id = _project_id_from_memory_path(path)
                if not project_id or project_id in granted_projects:
                    return
                response = original_post(
                    f"/api/projects/{project_id}/permissions/grants",
                    json={"action": "project_memory_write", "actor": "api-smoke", "confirmed": True},
                )
                if response.status_code == 200:
                    granted_projects.add(project_id)

            def post(path: str, *post_args: Any, **post_kwargs: Any):
                if isinstance(path, str) and ("/memory/records" in path or path.endswith("/memory/enabled")):
                    ensure_grant(path)
                return original_post(path, *post_args, **post_kwargs)

            def delete(path: str, *delete_args: Any, **delete_kwargs: Any):
                if isinstance(path, str) and "/memory/records/" in path:
                    ensure_grant(path)
                return original_delete(path, *delete_args, **delete_kwargs)

            client.post = post  # type: ignore[method-assign]
            client.delete = delete  # type: ignore[method-assign]
            return client

        app.test_client = test_client_with_memory_grant  # type: ignore[method-assign]
        return app

    monkeypatch.setattr(module, "create_app", create_app_with_memory_grant)
