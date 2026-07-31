from __future__ import annotations

import hashlib

import pytest

from ai4s_agent.app import create_app
from ai4s_agent.storage import ProjectStorage


INVALID_PROJECT_IDS = (
    "a/b",
    "a/../b",
    "../b",
    ".",
    "..",
    "/absolute/path",
    " project-a",
    "project-a ",
    "a" * 97,
    "a\\b",
    "project\x00id",
)


def _app(tmp_path):
    return create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path,
        user_config_dir=tmp_path / "config",
    )


@pytest.mark.pr_fast
@pytest.mark.integration
def test_project_delete_archives_data_and_prevents_id_reuse(tmp_path) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    storage = ProjectStorage(tmp_path)

    assert client.post(
        "/api/projects",
        json={"project_id": "project-a", "name": "Project A"},
    ).status_code == 200
    assert client.post(
        "/api/projects",
        json={"project_id": "project-b", "name": "Project B"},
    ).status_code == 200
    retained = storage.project_dir("project-a") / "retained.txt"
    retained.write_text("recoverable project data", encoding="utf-8")

    deleted = client.delete("/api/projects/project-a")

    assert deleted.status_code == 200
    assert deleted.json == {
        "ok": True,
        "deleted": True,
        "project_id": "project-a",
        "name": "Project A",
    }
    listing = client.get("/api/projects").json["projects"]
    assert len(listing) == 1
    assert listing[0]["project_id"] == "project-b"
    assert listing[0]["name"] == "Project B"
    assert listing[0]["created_at"]
    archive = (
        tmp_path / ".deleted-projects" / hashlib.sha256(b"project-a").hexdigest()
    )
    assert (archive / "retained.txt").read_text(encoding="utf-8") == "recoverable project data"
    assert not (tmp_path / "projects" / "project-a").exists()
    assert client.delete("/api/projects/project-a").status_code == 404
    recreate = client.post(
        "/api/projects",
        json={"project_id": "project-a", "name": "Replacement"},
    )
    assert recreate.status_code == 400
    assert recreate.json["error"] == "deleted project_id cannot be reused"
    with pytest.raises(ValueError, match="deleted project_id cannot be reused"):
        storage.project_dir("project-a")


@pytest.mark.parametrize(
    ("deleted_id", "allowed_id"),
    [("a.b", "a"), ("a", "a.b")],
)
def test_project_tombstone_uses_exact_identity(tmp_path, deleted_id: str, allowed_id: str) -> None:
    client = _app(tmp_path).test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": deleted_id},
    ).status_code == 200
    assert client.delete(f"/api/projects/{deleted_id}").status_code == 200

    allowed = client.post(
        "/api/projects",
        json={"project_id": allowed_id},
    )

    assert allowed.status_code == 200
    assert allowed.json["project_id"] == allowed_id


def test_project_delete_rejects_symlink_alias(tmp_path) -> None:
    client = _app(tmp_path).test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": "project-source"},
    ).status_code == 200
    (tmp_path / "projects" / "project-alias").symlink_to(
        tmp_path / "projects" / "project-source",
        target_is_directory=True,
    )

    response = client.delete("/api/projects/project-alias")

    assert response.status_code == 400
    assert response.json == {"ok": False, "error": "invalid project identifier"}
    assert (tmp_path / "projects" / "project-source" / "project.json").is_file()


@pytest.mark.parametrize("project_id", INVALID_PROJECT_IDS)
def test_project_storage_rejects_noncanonical_project_ids_before_path_use(
    tmp_path,
    project_id: str,
) -> None:
    storage = ProjectStorage(tmp_path)

    with pytest.raises(
        ValueError,
        match="project_id must be a canonical single-component identifier",
    ):
        storage.project_dir(project_id)

    assert not [item for item in storage.projects_root.iterdir() if item.is_dir()]


@pytest.mark.parametrize("project_id", INVALID_PROJECT_IDS)
def test_create_project_api_rejects_noncanonical_project_ids(
    tmp_path,
    project_id: str,
) -> None:
    response = _app(tmp_path).test_client().post(
        "/api/projects",
        json={"project_id": project_id},
    )

    assert response.status_code == 400
    assert response.json == {
        "ok": False,
        "error": "project_id must be a canonical single-component identifier",
    }


def test_delete_project_api_does_not_trim_project_id(tmp_path) -> None:
    client = _app(tmp_path).test_client()
    assert client.post(
        "/api/projects",
        json={"project_id": "project-a"},
    ).status_code == 200

    response = client.delete("/api/projects/%20project-a%20")

    assert response.status_code == 400
    assert response.json == {"ok": False, "error": "invalid project identifier"}
    assert (tmp_path / "projects" / "project-a" / "project.json").is_file()


@pytest.mark.pr_fast
def test_project_delete_ui_is_context_safe_and_uses_server_route(tmp_path) -> None:
    html = _app(tmp_path).test_client().get("/").get_data(as_text=True)

    assert 'remove.className = "project-delete";' in html
    assert "async function deleteProject(projectId, name)" in html
    assert "await deleteJSON(`/api/projects/${encodeURIComponent(projectId)}`);" in html
    assert "if (activeProject.project_id !== projectId)" in html
    assert "conversationLoadGeneration += 1;" in html
    assert "deletedProjectIds.has(projectId)" in html
