"""Focused tests for the small local Molly browser surface."""

from __future__ import annotations

import base64
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from molly.web import MollyHTTPRequestHandler, ProviderConfigStore, create_application


pytestmark = pytest.mark.acceptance


def test_demo_web_flow_exposes_core_state_and_exact_confirmation(tmp_path: Path) -> None:
    app = create_application(tmp_path / "runtime", demo=True)

    status, bootstrap = app.dispatch("GET", "/api/bootstrap")
    assert status == 200
    assert bootstrap["runtime_profiles"][0]["name"] == "本地演示"
    assert "provider_secrets" not in json.dumps(bootstrap, ensure_ascii=False)

    status, started = app.dispatch(
        "POST",
        "/api/runs",
        {
            "profile_id": "profile:web-demo",
            "goal": "体验一遍本地任务流程",
            "budget": {"max_decisions": 4, "max_tool_calls": 2, "max_steps": 2},
        },
    )
    assert status == 201
    assert started["status"] == "WAITING_APPROVAL"
    pending = started["inspection"]["pending_call"]
    assert pending["tool_name"] == "create_demo_result"

    status, approved = app.dispatch(
        "POST",
        f"/api/runs/{started['run_id']}/approval",
        {
            "decision": "APPROVED",
            "reviewer_ref": "local-user",
            "call_id": pending["call_id"],
        },
    )
    assert status == 200
    assert approved["approval"]["decision"] == "APPROVED"
    assert approved["result"]["status"] == "ACTIVE"

    status, finished = app.dispatch(
        "POST",
        f"/api/runs/{started['run_id']}/resume",
        {},
    )
    assert status == 200
    assert finished["status"] == "STOPPED"
    assert finished["inspection"]["status_label"] == "已完成"
    assert finished["inspection"]["final_artifact_ids"]

    status, observed = app.dispatch(
        "POST",
        f"/api/runs/{started['run_id']}/observe",
        {"exporter": "json"},
    )
    assert status == 200
    assert observed["status"] == "EXPORTED"
    assert observed["trace"]["run_id"] == started["run_id"]

    status, runs = app.dispatch("GET", "/api/runs")
    assert status == 200
    assert len(runs["runs"]) == 1
    assert runs["runs"][0]["status_label"] == "已完成"


def test_provider_settings_are_non_secret_in_browser_and_secret_is_server_only(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runtime"
    app = create_application(root)
    payload = {
        "profile_ref": "provider:test",
        "display_name": "测试模型",
        "endpoint": "https://models.example.test/v1",
        "model_identifier": "test-model",
        "model_version": "1",
        "timeout_seconds": 20,
        "max_response_bytes": 262144,
    }

    status, saved = app.dispatch("POST", "/api/model-profiles", payload)
    assert status == 201
    assert saved["profile"]["credential_status"] == "未配置"
    assert "secret" not in json.dumps(saved, ensure_ascii=False).lower()

    app.provider_store.set_secret("provider:test", "server-only-secret")
    status, profiles = app.dispatch("GET", "/api/model-profiles")
    assert status == 200
    assert profiles["profiles"][0]["credential_configured"] is True
    assert "server-only-secret" not in json.dumps(profiles, ensure_ascii=False)

    status, checked = app.dispatch("POST", "/api/model-profiles/provider:test/check", {})
    assert status == 200
    assert checked["ready"] is True
    assert "server-only-secret" not in json.dumps(checked, ensure_ascii=False)

    status, rejected = app.dispatch(
        "POST",
        "/api/model-profiles",
        {**payload, "api_key": "must-not-be-accepted"},
    )
    assert status == 400
    assert "must-not-be-accepted" not in json.dumps(rejected, ensure_ascii=False)

    secret_file = root / "provider_secrets.json"
    assert secret_file.stat().st_mode & 0o077 == 0


def test_upload_publishes_verified_local_file_without_path_input(tmp_path: Path) -> None:
    app = create_application(tmp_path / "runtime")
    status, uploaded = app.dispatch(
        "POST",
        "/api/artifacts",
        {
            "file_name": "notes.txt",
            "media_type": "text/plain",
            "content_base64": base64.b64encode(b"local notes").decode("ascii"),
        },
    )
    assert status == 201
    assert uploaded["name"] == "notes.txt"
    assert uploaded["size_bytes"] == len(b"local notes")
    artifact_status, artifact = app.dispatch(
        "GET", f"/api/artifacts/{uploaded['artifact_id']}"
    )
    assert artifact_status == 200
    assert artifact["artifact_id"] == uploaded["artifact_id"]
    assert artifact["size_bytes"] == len(b"local notes")
    status, content, media_type, download_name = app.artifact_content(uploaded["artifact_id"])
    assert status == 200
    assert content == b"local notes"
    assert media_type == "text/plain"
    assert download_name.endswith(".bin")
    assert app.dispatch("GET", f"/api/artifacts/{uploaded['artifact_id']}/content")[0] == 200


def test_static_surface_is_available_without_a_frontend_dependency(tmp_path: Path) -> None:
    app = create_application(tmp_path / "runtime")
    result = app.static_file("/")
    assert result is not None
    status, content, media_type = result
    assert status == 200
    assert media_type.startswith("text/html")
    assert "科学任务工作台" in content.decode("utf-8")
    assert app.local_session_token in content.decode("utf-8")


def test_http_write_surface_requires_loopback_origin_token_and_json(tmp_path: Path) -> None:
    app = create_application(tmp_path / "runtime")
    handler = type(
        "TestMollyHTTPRequestHandler",
        (MollyHTTPRequestHandler,),
        {"application": app},
    )
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except PermissionError:
        app.close()
        pytest.skip("the test environment does not permit local socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    host = f"127.0.0.1:{port}"
    origin = f"http://127.0.0.1:{port}"
    body = json.dumps({"file_name": "notes.txt", "content_base64": "bG9jYWw="}).encode()

    def post(**overrides: str | None) -> tuple[int, dict[str, object]]:
        headers = {
            "Host": host,
            "Origin": origin,
            "X-Molly-Local-Token": app.local_session_token,
            "Content-Type": "application/json",
        }
        headers.update({key: value for key, value in overrides.items() if value is not None})
        connection = HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("POST", "/api/artifacts", body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    try:
        status, accepted = post()
        assert status == 201
        assert "artifact_id" in accepted

        status, rejected = post(Host="attacker.example")
        assert status == 403
        assert rejected["error_type"] == "LOCAL_HOST_REQUIRED"

        status, rejected = post(Origin="https://attacker.example")
        assert status == 403
        assert rejected["error_type"] == "LOCAL_ORIGIN_REQUIRED"

        status, rejected = post(**{"X-Molly-Local-Token": "wrong-token"})
        assert status == 403
        assert rejected["error_type"] == "LOCAL_SESSION_REQUIRED"

        status, rejected = post(**{"Content-Type": "text/plain"})
        assert status == 415
        assert rejected["error_type"] == "JSON_CONTENT_TYPE_REQUIRED"
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
        app.close()
