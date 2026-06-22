from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.memory import PermissionPolicy
from ai4s_agent.server_permissions import ServerPermissionStore, decide_server_permission


def _iso_at(delta: timedelta) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat().replace("+00:00", "Z")


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


def test_permission_grant_expires_at_is_persisted_and_authorizes_until_expiry(tmp_path) -> None:
    store = ServerPermissionStore(workspace_dir=tmp_path)
    project_dir = store.projects.project_dir("proj-a")
    write_json(project_dir / "project.json", {"project_id": "proj-a", "name": "proj-a", "created_at": now_iso()})
    expires_at = _iso_at(timedelta(hours=1))

    grant = store.create_grant("proj-a", "upload_dataset", actor="alice", expires_at=expires_at)
    decision = decide_server_permission(
        store,
        PermissionPolicy(),
        "upload_dataset",
        project_id="proj-a",
        actor="alice",
    )

    assert grant["expires_at"] == expires_at
    assert decision["allowed"] is True
    assert decision["reason"] == "SERVER_GRANT"
    assert decision["grant_id"] == grant["grant_id"]
    assert decision["server_authorized"] is True


def test_expired_permission_grant_is_denied_and_audited(tmp_path) -> None:
    store = ServerPermissionStore(workspace_dir=tmp_path)
    project_dir = store.projects.project_dir("proj-a")
    write_json(project_dir / "project.json", {"project_id": "proj-a", "name": "proj-a", "created_at": now_iso()})
    grant = store.create_grant("proj-a", "upload_dataset", actor="alice", expires_at=_iso_at(timedelta(hours=1)))
    grant["expires_at"] = _iso_at(timedelta(minutes=-1))
    write_json(
        store._grants_path("proj-a"),
        {"project_id": "proj-a", "updated_at": now_iso(), "grants": [grant]},
    )

    decision = decide_server_permission(
        store,
        PermissionPolicy(),
        "upload_dataset",
        project_id="proj-a",
        actor="alice",
    )

    assert decision["allowed"] is False
    assert decision["reason"] == "EXPIRED_GRANT"
    assert decision["grant_id"] == grant["grant_id"]
    assert decision["server_authorized"] is False
    audit = store.read_audit("proj-a")
    assert audit[-1]["reason"] == "EXPIRED_GRANT"
    assert audit[-1]["allowed"] is False
    assert audit[-1]["grant_id"] == grant["grant_id"]


def test_permission_grant_rejects_invalid_or_past_expires_at(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    invalid = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True, "expires_at": "not-a-date"},
    )
    expired = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True, "expires_at": _iso_at(timedelta(minutes=-1))},
    )
    valid = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True, "expires_at": _iso_at(timedelta(hours=1))},
    )

    assert invalid.status_code == 400
    assert "expires_at" in invalid.json["error"]
    assert expired.status_code == 400
    assert "future" in expired.json["error"]
    assert valid.status_code == 200
    assert valid.json["grant"]["expires_at"]


def test_string_false_disables_legacy_client_permission_flags(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ALLOW_CLIENT_PERMISSION_FLAGS"] = "false"
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


def test_run_scoped_permission_grant_does_not_authorize_project_level_request(tmp_path) -> None:
    store = ServerPermissionStore(workspace_dir=tmp_path)
    project_dir = store.projects.project_dir("proj-a")
    write_json(project_dir / "project.json", {"project_id": "proj-a", "name": "proj-a", "created_at": now_iso()})
    grant = store.create_grant("proj-a", "upload_dataset", actor="alice", run_id="run-1")

    project_level = decide_server_permission(
        store,
        PermissionPolicy(),
        "upload_dataset",
        project_id="proj-a",
        actor="alice",
    )
    run_level = decide_server_permission(
        store,
        PermissionPolicy(),
        "upload_dataset",
        project_id="proj-a",
        run_id="run-1",
        actor="alice",
    )

    assert project_level["allowed"] is False
    assert project_level["reason"] == "SERVER_GRANT_REQUIRED"
    assert project_level["grant_id"] == ""
    assert run_level["allowed"] is True
    assert run_level["reason"] == "SERVER_GRANT"
    assert run_level["grant_id"] == grant["grant_id"]


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


def test_permission_grant_requires_literal_json_true_confirmation(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    for value in ["false", "0", "no", "true", 1, False, None]:
        response = client.post(
            "/api/projects/proj-a/permissions/grants",
            json={"action": "upload_dataset", "actor": "alice", "confirmed": value},
        )
        assert response.status_code == 403

    grants = client.get("/api/projects/proj-a/permissions/grants")
    assert grants.status_code == 200
    assert grants.json["grants"] == []

    created = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True},
    )
    assert created.status_code == 200
    grants_after = client.get("/api/projects/proj-a/permissions/grants")
    assert len(grants_after.json["grants"]) == 1


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
