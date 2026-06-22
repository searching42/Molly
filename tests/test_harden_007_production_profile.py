from __future__ import annotations

from io import BytesIO

from ai4s_agent.app import create_app
from ai4s_agent.profiles import app_profile, production_profile_enabled


def test_profile_resolves_from_ai4s_profile_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    monkeypatch.setenv("AI4S_ENV", "local")

    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)

    assert app_profile(app) == "production"
    assert production_profile_enabled(app) is True


def test_profile_resolves_from_ai4s_env_when_profile_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AI4S_PROFILE", raising=False)
    monkeypatch.setenv("AI4S_ENV", "prod")

    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)

    assert app_profile(app) == "production"
    assert production_profile_enabled(app) is True


def test_local_profile_preserves_upload_legacy_permission_flag_default(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    uploaded = client.post(
        "/api/projects/proj-a/upload",
        data={
            "file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"),
            "project_approved": "true",
            "actor": "legacy-user",
        },
        content_type="multipart/form-data",
    )

    assert uploaded.status_code == 200
    assert uploaded.json["permission"]["reason"] == "LEGACY_CLIENT_PROJECT_APPROVED"


def test_production_profile_rejects_upload_legacy_permission_flag_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    denied = client.post(
        "/api/projects/proj-a/upload",
        data={
            "file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"),
            "project_approved": "true",
            "actor": "legacy-user",
        },
        content_type="multipart/form-data",
    )

    assert denied.status_code == 403
    assert denied.json["permission"]["reason"] == "SERVER_GRANT_REQUIRED"
    assert denied.json["permission"]["legacy_client_flag"] is False


def test_production_profile_ignores_explicit_legacy_permission_flag_enable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ALLOW_CLIENT_PERMISSION_FLAGS"] = True
    app.config["AI4S_ALLOW_MEMORY_CLIENT_PERMISSION_FLAGS"] = True
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    upload_denied = client.post(
        "/api/projects/proj-a/upload",
        data={
            "file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"),
            "project_approved": "true",
            "actor": "legacy-user",
        },
        content_type="multipart/form-data",
    )
    memory_denied = client.post(
        "/api/projects/proj-a/memory/records",
        json={
            "record_id": "parser-choice",
            "category": "parser_choice",
            "summary": "Prefer MinerU.",
            "value": {"preferred_parser": "mineru"},
            "decision": "confirmed_parser_policy",
            "source_refs": ["run:run-1:parser_eval"],
            "source_hashes": ["sha256:abc"],
            "confirmed_by": "alice",
            "project_approved": True,
        },
    )

    assert upload_denied.status_code == 403
    assert upload_denied.json["permission"]["legacy_client_flag"] is False
    assert memory_denied.status_code == 403
    assert memory_denied.json["permission"]["legacy_client_flag"] is False


def test_production_profile_upload_accepts_server_grant(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})
    grant = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True},
    )

    uploaded = client.post(
        "/api/projects/proj-a/upload",
        headers={"X-Actor": "alice"},
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv")},
        content_type="multipart/form-data",
    )

    assert grant.status_code == 200
    assert uploaded.status_code == 200
    assert uploaded.json["permission"]["reason"] == "SERVER_GRANT"
    assert uploaded.json["permission"]["grant_id"] == grant.json["grant"]["grant_id"]


def test_production_profile_keeps_memory_legacy_permission_flag_disabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    denied = client.post(
        "/api/projects/proj-a/memory/records",
        json={
            "record_id": "parser-choice",
            "category": "parser_choice",
            "summary": "Prefer MinerU.",
            "value": {"preferred_parser": "mineru"},
            "decision": "confirmed_parser_policy",
            "source_refs": ["run:run-1:parser_eval"],
            "source_hashes": ["sha256:abc"],
            "confirmed_by": "alice",
            "project_approved": True,
        },
    )

    assert denied.status_code == 403
    assert denied.json["permission"]["reason"] == "SERVER_GRANT_REQUIRED"
    assert denied.json["permission"]["legacy_client_flag"] is False


def test_production_profile_disables_route_extension_inspection_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.get("/api/system/route-extensions")

    assert response.status_code == 404


def test_production_profile_allows_route_extension_inspection_when_debug_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ENABLE_ROUTE_EXTENSION_INSPECTION"] = True
    client = app.test_client()

    response = client.get("/api/system/route-extensions")

    assert response.status_code == 200
    assert response.json["ok"] is True


def test_production_profile_allows_route_extension_inspection_when_env_enabled(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AI4S_PROFILE", "production")
    monkeypatch.setenv("AI4S_ENABLE_ROUTE_EXTENSION_INSPECTION", "true")
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()

    response = client.get("/api/system/route-extensions")

    assert response.status_code == 200
    assert response.json["ok"] is True
