from __future__ import annotations

from ai4s_agent.app import create_app


def _record_payload(**extra):
    payload = {
        "record_id": "parser-choice",
        "category": "parser_choice",
        "summary": "Prefer MinerU unless table extraction is poor.",
        "value": {"preferred_parser": "mineru"},
        "decision": "confirmed_parser_policy",
        "source_refs": ["run:run-1:parser_eval"],
        "source_hashes": ["sha256:abc"],
        "confirmed_by": "alice",
    }
    payload.update(extra)
    return payload


def test_project_memory_write_requires_server_grant_by_default(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    denied = client.post("/api/projects/proj-a/memory/records", json=_record_payload(project_approved=True, actor="alice"))

    assert denied.status_code == 403
    assert denied.json["permission"]["action"] == "project_memory_write"
    assert denied.json["permission"]["reason"] == "SERVER_GRANT_REQUIRED"
    assert denied.json["permission"]["legacy_client_flag"] is False
    listed = client.get("/api/projects/proj-a/memory")
    assert listed.status_code == 200
    assert listed.json["records"] == []


def test_project_memory_record_create_update_delete_with_server_grant(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})
    grant = client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "project_memory_write", "actor": "alice", "confirmed": True, "reason": "approve memory edits"},
    )
    assert grant.status_code == 200

    created = client.post("/api/projects/proj-a/memory/records", json=_record_payload(actor="alice"))
    assert created.status_code == 200
    assert created.json["permission"]["reason"] == "SERVER_GRANT"
    assert created.json["record"]["record_id"] == "parser-choice"

    updated = client.patch(
        "/api/projects/proj-a/memory/records/parser-choice",
        json={"summary": "Prefer MinerU; use pdfplumber for poor tables.", "actor": "alice"},
    )
    assert updated.status_code == 200
    assert "poor tables" in updated.json["record"]["summary"]
    assert updated.json["permission"]["grant_id"] == grant.json["grant"]["grant_id"]

    disabled = client.post("/api/projects/proj-a/memory/enabled", json={"enabled": False, "actor": "alice"})
    assert disabled.status_code == 200
    assert disabled.json["enabled"] is False

    deleted = client.delete("/api/projects/proj-a/memory/records/parser-choice", json={"actor": "alice"})
    assert deleted.status_code == 200
    assert deleted.json["deleted"] is True

    audit = client.get("/api/projects/proj-a/permissions/audit")
    reasons = [item["reason"] for item in audit.json["audit"] if item["action"] == "project_memory_write"]
    assert reasons.count("SERVER_GRANT") >= 4


def test_project_memory_legacy_flag_requires_explicit_opt_in_and_is_audited(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    app.config["AI4S_ALLOW_MEMORY_CLIENT_PERMISSION_FLAGS"] = "true"
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})

    created = client.post("/api/projects/proj-a/memory/records", json=_record_payload(project_approved=True, actor="legacy-user"))

    assert created.status_code == 200
    assert created.json["permission"]["reason"] == "LEGACY_CLIENT_PROJECT_APPROVED"
    assert created.json["permission"]["legacy_client_flag"] is True
    audit = client.get("/api/projects/proj-a/permissions/audit")
    assert any(
        item["action"] == "project_memory_write"
        and item["legacy_client_flag"] is True
        and item["reason"] == "LEGACY_CLIENT_PROJECT_APPROVED"
        for item in audit.json["audit"]
    )


def test_project_memory_enabled_requires_literal_boolean_after_permission(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    client = app.test_client()
    client.post("/api/projects", json={"project_id": "proj-a"})
    client.post(
        "/api/projects/proj-a/permissions/grants",
        json={"action": "project_memory_write", "actor": "alice", "confirmed": True},
    )

    response = client.post("/api/projects/proj-a/memory/enabled", json={"enabled": "false", "actor": "alice"})

    assert response.status_code == 400
    assert "enabled boolean required" in response.json["error"]
