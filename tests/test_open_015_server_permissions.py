from __future__ import annotations

from io import BytesIO

from ai4s_agent.app import create_app


def test_upload_requires_server_permission_grant_when_legacy_flags_disabled(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ALLOW_CLIENT_PERMISSION_FLAGS"] = False
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    denied = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"), "project_approved": "true"},
        content_type="multipart/form-data",
    )
    assert denied.status_code == 403
    assert denied.json["permission"]["reason"] == "SERVER_GRANT_REQUIRED"
    assert denied.json["permission"]["legacy_client_flag"] is False

    grant = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True, "reason": "approved dataset upload"},
    )
    assert grant.status_code == 200
    assert grant.json["grant"]["action"] == "upload_dataset"

    uploaded = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv")},
        headers={"X-Actor": "alice"},
        content_type="multipart/form-data",
    )
    assert uploaded.status_code == 200
    assert uploaded.json["permission"]["reason"] == "SERVER_GRANT"
    assert uploaded.json["permission"]["server_authorized"] is True
    assert uploaded.json["permission"]["grant_id"] == grant.json["grant"]["grant_id"]

    audit = client.get("/api/projects/proj-a/permissions/audit")
    assert audit.status_code == 200
    reasons = [item["reason"] for item in audit.json["audit"]]
    assert "SERVER_GRANT_REQUIRED" in reasons
    assert "SERVER_GRANT_CREATED" in reasons
    assert "SERVER_GRANT" in reasons


def test_legacy_project_approval_flag_is_audited_as_legacy_fallback(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    uploaded = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"), "project_approved": "true", "actor": "legacy-user"},
        content_type="multipart/form-data",
    )

    assert uploaded.status_code == 200
    assert uploaded.json["permission"]["reason"] == "LEGACY_CLIENT_PROJECT_APPROVED"
    assert uploaded.json["permission"]["legacy_client_flag"] is True
    assert uploaded.json["permission"]["server_authorized"] is False
    audit = client.get("/api/projects/proj-a/permissions/audit")
    assert audit.status_code == 200
    assert any(item["legacy_client_flag"] is True and item["reason"] == "LEGACY_CLIENT_PROJECT_APPROVED" for item in audit.json["audit"])


def test_permission_grant_requires_actor_and_confirmation(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    missing_actor = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "confirmed": True},
    )
    assert missing_actor.status_code == 403

    missing_confirmation = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice"},
    )
    assert missing_confirmation.status_code == 403
