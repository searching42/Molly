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

    status, credential = app.dispatch(
        "POST",
        "/api/model-profiles/provider:test/credential",
        {"api_key": "browser-supplied-secret"},
    )
    assert status == 200
    assert credential["credential_configured"] is True
    assert "browser-supplied-secret" not in json.dumps(credential, ensure_ascii=False)

    status, profiles = app.dispatch("GET", "/api/model-profiles")
    assert status == 200
    assert profiles["profiles"][0]["credential_configured"] is True
    assert "browser-supplied-secret" not in json.dumps(profiles, ensure_ascii=False)

    status, checked = app.dispatch("POST", "/api/model-profiles/provider:test/check", {})
    assert status == 200
    assert checked["ready"] is True
    assert "browser-supplied-secret" not in json.dumps(checked, ensure_ascii=False)

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
    assert "Molly" not in content.decode("utf-8")
    assert "demo" not in content.decode("utf-8").casefold()
    assert "max-decisions" not in content.decode("utf-8")
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
            "X-Local-Session-Token": app.local_session_token,
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

        status, rejected = post(**{"X-Local-Session-Token": "wrong-token"})
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
