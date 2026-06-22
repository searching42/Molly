from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.memory import PermissionLevel, PermissionPolicy
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


def test_permission_grant_uses_x_actor_before_body_actor_and_audits_actor_source(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    grant = client.post(
        "/api/projects/proj-a/permissions/grants",
        headers={"X-Actor": "header-admin"},
        json={"action": "upload_dataset", "actor": "body-admin", "confirmed": True},
    )

    assert grant.status_code == 200
    assert grant.json["grant"]["actor"] == "header-admin"
    audit = client.get("/api/projects/proj-a/permissions/audit")
    created = [item for item in audit.json["audit"] if item["reason"] == "SERVER_GRANT_CREATED"][-1]
    assert created["actor"] == "header-admin"
    assert created["actor_source"] == "header:X-Actor"


def test_upload_permission_uses_x_actor_before_form_actor_and_audits_actor_source(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ALLOW_CLIENT_PERMISSION_FLAGS"] = False
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})
    client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True},
    )

    uploaded = client.post(
        "/api/projects/proj-a/upload",
        data={"file": (BytesIO(b"SMILES,value\nCCO,1.0\n"), "dataset.csv"), "actor": "form-user"},
        headers={"X-Actor": "header-user"},
        content_type="multipart/form-data",
    )

    assert uploaded.status_code == 200
    assert uploaded.json["permission"]["actor"] == "header-user"
    assert uploaded.json["permission"]["actor_source"] == "header:X-Actor"
    audit = client.get("/api/projects/proj-a/permissions/audit")
    allowed = [item for item in audit.json["audit"] if item["reason"] == "SERVER_GRANT"][-1]
    assert allowed["actor"] == "header-user"
    assert allowed["actor_source"] == "header:X-Actor"


def test_revoke_permission_grant_accepts_json_revoked_by_and_audits_actor_source(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})
    grant = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True},
    )

    revoked = client.delete(
        f"/api/projects/proj-a/permissions/grants/{grant.json['grant']['grant_id']}",
        json={"revoked_by": "security-admin"},
    )

    assert revoked.status_code == 200
    assert revoked.json["grant"]["revoked_by"] == "security-admin"
    audit = client.get("/api/projects/proj-a/permissions/audit")
    revoked_record = [item for item in audit.json["audit"] if item["reason"] == "SERVER_GRANT_REVOKED"][-1]
    assert revoked_record["actor"] == "security-admin"
    assert revoked_record["actor_source"] == "json:revoked_by"


def test_required_actor_missing_returns_403_for_permission_grant_and_revoke(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    missing_grant_actor = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "confirmed": True},
    )
    grant = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True},
    )
    missing_revoke_actor = client.delete(
        f"/api/projects/proj-a/permissions/grants/{grant.json['grant']['grant_id']}",
        json={},
    )

    assert missing_grant_actor.status_code == 403
    assert missing_grant_actor.json["actor_source"] == "missing"
    assert missing_revoke_actor.status_code == 403
    assert missing_revoke_actor.json["actor_source"] == "missing"


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


def test_revoke_grant_disables_active_grant_and_denies_permission(tmp_path) -> None:
    store = ServerPermissionStore(workspace_dir=tmp_path)
    policy = PermissionPolicy()
    policy.set_policy("upload_dataset", PermissionLevel.PROJECT_APPROVED)

    grant = store.create_grant("proj-revoke", "upload_dataset", actor="alice", reason="test revoke")
    assert grant["active"] is True

    # Active grant authorizes
    decision = decide_server_permission(store, policy, "upload_dataset", project_id="proj-revoke", actor="alice")
    assert decision["allowed"] is True
    assert decision["reason"] == "SERVER_GRANT"
    assert decision["grant_id"] == grant["grant_id"]

    # Revoke
    revoked = store.revoke_grant("proj-revoke", grant["grant_id"], revoked_by="admin", revoke_reason="mistaken grant")
    assert revoked["active"] is False
    assert revoked["revoked_at"]
    assert revoked["revoked_by"] == "admin"
    assert revoked["revoke_reason"] == "mistaken grant"

    # Revoked grant no longer authorizes
    decision = decide_server_permission(store, policy, "upload_dataset", project_id="proj-revoke", actor="alice")
    assert decision["allowed"] is False
    assert decision["reason"] == "REVOKED_GRANT"
    assert decision["grant_id"] == grant["grant_id"]

    # Audit trail includes SERVER_GRANT_REVOKED
    audit = store.read_audit("proj-revoke")
    audit_reasons = [entry["reason"] for entry in audit]
    assert "SERVER_GRANT_CREATED" in audit_reasons
    assert "SERVER_GRANT_REVOKED" in audit_reasons


def test_revoked_grant_blocks_upload_and_memory_write(tmp_path) -> None:
    store = ServerPermissionStore(workspace_dir=tmp_path)
    policy = PermissionPolicy()
    for action in ("upload_dataset", "project_memory_write"):
        policy.set_policy(action, PermissionLevel.PROJECT_APPROVED)

    # Create grants
    upload_grant = store.create_grant("proj-block", "upload_dataset", actor="alice")
    mem_grant = store.create_grant("proj-block", "project_memory_write", actor="alice")

    # Revoke upload grant only
    store.revoke_grant("proj-block", upload_grant["grant_id"], revoked_by="admin")

    # Upload is denied with REVOKED_GRANT
    decision = decide_server_permission(store, policy, "upload_dataset", project_id="proj-block", actor="alice")
    assert decision["allowed"] is False
    assert decision["reason"] == "REVOKED_GRANT"

    # Memory write still active (not revoked)
    decision = decide_server_permission(store, policy, "project_memory_write", project_id="proj-block", actor="alice")
    assert decision["allowed"] is True
    assert decision["reason"] == "SERVER_GRANT"
    assert decision["grant_id"] == mem_grant["grant_id"]


def test_revoke_endpoint_returns_400_for_missing_grant(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    resp = client.delete(
        "/api/projects/proj-revoke/permissions/grants/nonexistent-grant",
        json={"actor": "admin"},
    )
    assert resp.status_code == 400


def test_recreate_after_revoke_authorizes_new_grant(tmp_path) -> None:
    store = ServerPermissionStore(workspace_dir=tmp_path)
    policy = PermissionPolicy()
    policy.set_policy("upload_dataset", PermissionLevel.PROJECT_APPROVED)

    # Grant A → revoke → Grant B for same scope
    grant_a = store.create_grant("proj-recreate", "upload_dataset", actor="alice", reason="first")
    store.revoke_grant("proj-recreate", grant_a["grant_id"], revoked_by="admin")
    grant_b = store.create_grant("proj-recreate", "upload_dataset", actor="alice", reason="second")

    decision = decide_server_permission(store, policy, "upload_dataset", project_id="proj-recreate", actor="alice")
    assert decision["allowed"] is True
    assert decision["reason"] == "SERVER_GRANT"
    assert decision["grant_id"] == grant_b["grant_id"]
    # Revoked grant A does not shadow active grant B
    assert decision["grant_id"] != grant_a["grant_id"]


def test_revoke_endpoint_requires_actor(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    # First create a grant
    resp = client.post(
        "/api/projects/proj-revoke/permissions/grants",
        json={"action": "upload_dataset", "actor": "alice", "confirmed": True},
    )
    grant_id = resp.json["grant"]["grant_id"]

    # Try to revoke without actor
    resp = client.delete(
        f"/api/projects/proj-revoke/permissions/grants/{grant_id}",
        json={},
    )
    assert resp.status_code == 403
    assert "actor required" in resp.json["error"].lower() or "required" in resp.json["error"].lower()
