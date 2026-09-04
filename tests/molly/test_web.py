"""Focused tests for the small local Molly browser surface."""

from __future__ import annotations

import base64
import json
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading

import pytest

from molly.core import StopAction, ToolPolicy, ToolRegistry
from molly.core.artifacts import ArtifactStore
from molly.web import MollyHTTPRequestHandler, ProviderConfigStore, create_application
from molly.web.environments import EnvironmentDetector, EnvironmentManager
from molly.plugins.br1_inverse_design.dataset import validate_raw_dataset_source
from molly.plugins.br1_inverse_design.errors import Br1IntegrityError
from molly.plugins.br1_inverse_design.workflow import br1_profile
from molly.runtime import RuntimeProfile, RuntimeProfileRegistry, RuntimeService


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


def test_provider_secret_is_bound_to_the_profile_endpoint_digest(tmp_path: Path) -> None:
    store = ProviderConfigStore(tmp_path / "runtime")
    payload = {
        "profile_ref": "provider:test",
        "display_name": "测试模型",
        "endpoint": "https://models.example.test/v1",
        "model_identifier": "test-model",
        "model_version": "1",
        "timeout_seconds": 20,
        "max_response_bytes": 262144,
    }

    original = store.upsert_profile(payload)
    store.set_secret("provider:test", "old-endpoint-secret")
    assert store.get_profile("provider:test").credential_configured is True
    persisted = json.loads((tmp_path / "runtime" / "provider_secrets.json").read_text())
    assert persisted["version"] == 2
    assert persisted["secrets"]["provider:test"]["profile_digest"] == original.profile.digest

    changed = store.upsert_profile(
        {**payload, "endpoint": "https://models.example.test/v2"}
    )
    assert changed.profile.digest != original.profile.digest
    assert changed.credential_configured is False
    assert store.resolve_secret(changed.profile) is None

    store.set_secret("provider:test", "new-endpoint-secret")
    assert store.get_profile("provider:test").credential_configured is True
    assert store.resolve_secret(changed.profile) == "new-endpoint-secret"


def test_environment_api_reports_read_only_detection_and_install_preview(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    calls: list[tuple[str, ...]] = []
    raw_report = {
        "system": {"os": "Linux", "release": "6.8", "architecture": "x86_64"},
        "disk": {
            "path": "/srv/molly/runtimes",
            "exists": True,
            "writable": True,
            "parent_writable": True,
            "total_bytes": 10_000_000,
            "available_bytes": 9_000_000,
        },
        "gpu": {
            "available": True,
            "devices": [{"name": "A10", "memory_mib": 24_576, "driver_version": "550"}],
            "cuda": {"available": True, "version": "12.4"},
        },
        "python": {
            "executable": "/opt/python",
            "version": "3.11.9",
            "implementation": "CPython",
            "managers": {"conda": {"available": True, "version": "conda 24", "path": "/opt/conda"}},
        },
        "unimol": {"installed": True, "importable": True, "package": "unimol-tools", "version": "0.1.5"},
        "reinvent4": {
            "installed": True,
            "importable": True,
            "package": "reinvent4",
            "version": "4.7.15",
            "repositories": [{"path": "/opt/REINVENT4", "exists": True, "git": True, "config": True}],
            "license_present": True,
        },
        "weights": {"entries": [{"name": "unimolv1.pt", "path": "/opt/unimolv1.pt", "size_bytes": 1}], "total_bytes": 1},
    }

    def runner(argv, _input, _timeout):
        calls.append(tuple(argv))
        return 0, json.dumps(raw_report).encode("utf-8")

    manager = EnvironmentManager(
        root,
        detector=EnvironmentDetector(runner=runner, local_run_directory=root / "runtimes"),
    )
    app = create_application(root, environment_manager=manager)
    try:
        status, saved = app.dispatch(
            "POST",
            "/api/environments",
            {
                "mode": "local",
                "display_name": "本地探测",
            },
        )
        assert status == 201
        environment_ref = saved["environment"]["environment_ref"]

        status, listed = app.dispatch("GET", "/api/environments")
        assert status == 200
        assert listed["environments"][0]["target_label"] == "本地"

        status, detected = app.dispatch(
            "POST", f"/api/environments/{environment_ref}/detect", {}
        )
        assert status == 200
        assert detected["read_only"] is True
        assert detected["match"]["status"] == "PLAN_REQUIRED"
        assert detected["match"]["plan"]["will_execute"] is False
        assert detected["report"]["system"]["architecture"] == "x86_64"
        assert detected["report"]["weights"]["verification_status"] == "pending"
        assert calls
        assert (root / "environment_reports.json").is_file()
    finally:
        app.close()


def test_legacy_unbound_provider_secret_is_not_used_for_a_profile(tmp_path: Path) -> None:
    store = ProviderConfigStore(tmp_path / "runtime")
    profile = store.upsert_profile(
        {
            "profile_ref": "provider:test",
            "display_name": "测试模型",
            "endpoint": "https://models.example.test/v1",
            "model_identifier": "test-model",
        }
    )
    store.secrets_path.write_text(
        json.dumps({"version": 1, "secrets": {"provider:test": "legacy-secret"}})
    )

    assert store.get_profile("provider:test").credential_configured is False
    assert store.resolve_secret(profile.profile) is None


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
    assert download_name == "notes.txt"
    assert app.dispatch("GET", f"/api/artifacts/{uploaded['artifact_id']}/content")[0] == 200


def test_workflow_requires_one_valid_dataset_file(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    profile = br1_profile(root, profile_id="profile:workflow-input")
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = create_application(root, service=service)
    valid = json.dumps(
        {
            "columns": ["canonical_smiles", "quantum_yield"],
            "data": [["CCO", 0.4]],
        }
    ).encode()
    try:
        status, missing = app.dispatch(
            "POST",
            "/api/runs",
            {"profile_id": profile.profile_id, "goal": "rank molecules"},
        )
        assert status == 400
        assert missing["error_type"] == "WORKFLOW_FILE_REQUIRED"

        uploaded_ids = []
        for name, content in (("one.json", valid), ("two.json", valid + b" ")):
            status, uploaded = app.dispatch(
                "POST",
                "/api/artifacts",
                {
                    "file_name": name,
                    "media_type": "application/json",
                    "content_base64": base64.b64encode(content).decode("ascii"),
                    "workflow": "br1",
                },
            )
            assert status == 201
            uploaded_ids.append(uploaded["artifact_id"])
        status, multiple = app.dispatch(
            "POST",
            "/api/runs",
            {
                "profile_id": profile.profile_id,
                "goal": "rank molecules",
                "input_artifact_ids": uploaded_ids,
            },
        )
        assert status == 400
        assert multiple["error_type"] == "WORKFLOW_SINGLE_FILE_REQUIRED"

        status, invalid = app.dispatch(
            "POST",
            "/api/artifacts",
            {
                "file_name": "invalid.json",
                "media_type": "application/json",
                "content_base64": base64.b64encode(b'{"wrong": []}').decode("ascii"),
                "workflow": "br1",
            },
        )
        assert status == 400
        assert invalid["error_type"] == "WORKFLOW_FILE_INVALID"

        with pytest.raises(Br1IntegrityError):
            validate_raw_dataset_source(
                json.dumps(
                    {
                        "columns": ["canonical_smiles", "energies_occ_pbe0_vac_tier2"],
                        "data": [["CCO", [-5.0]]],
                    }
                ).encode(),
                target_property="homo_lumo_gap",
            )
    finally:
        app.close()


def test_non_br1_run_ignores_residual_unconfigured_model_reference(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    provider_store = ProviderConfigStore(root)
    provider_store.upsert_profile(
        {
            "profile_ref": "provider:unused",
            "display_name": "未配置模型",
            "endpoint": "https://models.example.test/v1/chat/completions",
            "model_identifier": "unused-model",
        }
    )

    class StopProvider:
        def next_action(self, context: object, model_visible_tools: object) -> StopAction:
            return StopAction("core workflow complete")

    profile = RuntimeProfile(
        profile_id="profile:core-web",
        tool_registry_factory=ToolRegistry,
        tool_policy_factory=ToolPolicy,
        decision_provider_factory=StopProvider,
        config={"display_name": "核心工作流", "workflow": "core"},
    )
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = create_application(root, service=service, provider_store=provider_store)
    payload = {
        "profile_id": profile.profile_id,
        "goal": "run the core workflow",
        "llm_profile_ref": "provider:unused",
    }
    try:
        status, preview = app.dispatch("POST", "/api/workflows/preview", payload)
        assert status == 200
        assert preview["workflow"] == "core"

        status, started = app.dispatch("POST", "/api/runs", payload)
        assert status == 201
        assert started["status"] == "STOPPED"
    finally:
        app.close()


def test_workflow_preview_binds_the_frozen_plan_to_start(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "runtime"
    provider_store = ProviderConfigStore(root)
    provider_store.upsert_profile(
        {
            "profile_ref": "provider:test",
            "display_name": "测试模型",
            "endpoint": "https://models.example.test/v1/chat/completions",
            "model_identifier": "structured-test",
        }
    )
    provider_store.set_secret("provider:test", "test-api-key")
    profile = br1_profile(
        root,
        profile_id="profile:workflow-preview",
        intent_provider_resolver=provider_store.create_intent_provider,
    )
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = create_application(root, service=service, provider_store=provider_store)
    intent_payload = {
        "target_property": "quantum_yield",
        "direction": "MAX",
        "candidate_count": 8,
        "top_n": 2,
        "scaffold_constraint": "NONE",
        "seed": 11,
        "host_preference": "auto",
        "cpu_threads": 8,
        "gpu_count": 0,
        "walltime_sec": 3600,
    }
    requests: list[dict[str, object]] = []

    def fake_transport(profile):
        def send(endpoint, *, headers, json_body, timeout_seconds):
            requests.append({"endpoint": endpoint, "headers": dict(headers), "body": dict(json_body)})
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps(intent_payload)}}]}
            ).encode()

        return send

    monkeypatch.setattr(ProviderConfigStore, "_transport", staticmethod(fake_transport))
    try:
        _, uploaded = app.dispatch(
            "POST",
            "/api/artifacts",
            {
                "file_name": "dataset.json",
                "media_type": "application/json",
                "content_base64": base64.b64encode(
                    json.dumps(
                        {"columns": ["canonical_smiles", "quantum_yield"], "data": [["CCO", 0.4]]}
                    ).encode()
                ).decode("ascii"),
            },
        )
        status, preview = app.dispatch(
            "POST",
            "/api/workflows/preview",
            {
                "profile_id": profile.profile_id,
                "goal": "maximize quantum yield",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
            },
        )
        assert status == 200
        assert preview["intent"]["spec"]["candidate_count"] == 8
        assert preview["preview_token"]

        status, started = app.dispatch(
            "POST",
            "/api/runs",
            {
                "profile_id": profile.profile_id,
                "goal": "maximize quantum yield",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
                "workflow_intent_preview_token": preview["preview_token"],
            },
        )
        assert status == 201
        assert started["status"] == "WAITING_APPROVAL"
        assert started["inspection"]["frozen_intent"]["spec"]["seed"] == 11
        assert len(requests) == 1

        status, tested = app.dispatch(
            "POST", "/api/model-profiles/provider:test/test", {}
        )
        assert status == 200
        assert tested["ready"] is True

        intent_payload["target_property"] = "homo_lumo_gap"
        status, mismatch = app.dispatch(
            "POST",
            "/api/workflows/preview",
            {
                "profile_id": profile.profile_id,
                "goal": "maximize quantum yield",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
            },
        )
        assert status == 400
        assert mismatch["error_type"] == "WORKFLOW_FILE_INVALID"
        status, mismatch_start = app.dispatch(
            "POST",
            "/api/runs",
            {
                "profile_id": profile.profile_id,
                "goal": "maximize quantum yield",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
            },
        )
        assert status == 400
        assert mismatch_start["error_type"] == "WORKFLOW_FILE_INVALID"

        def fail_on_body_read(*_args: object, **_kwargs: object) -> bytes:
            raise AssertionError("run-detail polling must not read artifact bodies")

        monkeypatch.setattr(
            ArtifactStore,
            "_read_verified_bytes",
            staticmethod(fail_on_body_read),
        )
        status, detail = app.dispatch("GET", f"/api/runs/{started['run_id']}")
        assert status == 200
        assert detail["status"] == "WAITING_APPROVAL"
    finally:
        app.close()


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
    script = app.static_file("/static/app.js")
    assert script is not None
    _, script_content, _ = script
    script_text = script_content.decode("utf-8")
    assert "网页不会接收" not in script_text
    assert "只发送到本机服务端" in script_text
    assert "运行环境" in script_text
    assert "检测环境" in script_text
    assert "apply_install_plan" not in script_text


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


def test_http_run_response_serializes_nested_frozen_intent(tmp_path: Path) -> None:
    root = tmp_path / "runtime"
    provider_store = ProviderConfigStore(root)
    provider_store.upsert_profile(
        {
            "profile_ref": "provider:test",
            "display_name": "测试模型",
            "endpoint": "https://models.example.test/v1/chat/completions",
            "model_identifier": "structured-test",
        }
    )
    provider_store.set_secret("provider:test", "test-api-key")
    provider_profile = provider_store.get_profile("provider:test").profile
    intent_payload = {
        "target_property": "quantum_yield",
        "direction": "MAX",
        "candidate_count": 8,
        "top_n": 2,
        "scaffold_constraint": "NONE",
        "seed": 11,
        "host_preference": "auto",
        "cpu_threads": 8,
        "gpu_count": 0,
        "walltime_sec": 3600,
    }

    class IntentProvider:
        profile = provider_profile

        def parse_br1_intent(self, goal: str, *, allowed_target_properties):
            return dict(intent_payload)

    profile = br1_profile(
        root,
        profile_id="profile:http-run",
        intent_provider_resolver=lambda _profile_ref: IntentProvider(),
    )
    service = RuntimeService(root, profiles=RuntimeProfileRegistry((profile,)))
    app = create_application(root, service=service, provider_store=provider_store)
    handler = type(
        "TestHttpRunHandler",
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

    def send(method: str, path: str, value: object | None = None) -> tuple[int, dict[str, object]]:
        body = json.dumps(value).encode() if value is not None else None
        headers = {"Host": host, "Content-Type": "application/json"}
        if method == "POST":
            headers.update(
                {
                    "Origin": origin,
                    "X-Local-Session-Token": app.local_session_token,
                }
            )
        connection = HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()

    try:
        status, uploaded = send(
            "POST",
            "/api/artifacts",
            {
                "file_name": "dataset.json",
                "media_type": "application/json",
                "content_base64": base64.b64encode(
                    json.dumps(
                        {"columns": ["canonical_smiles", "quantum_yield"], "data": [["CCO", 0.4]]}
                    ).encode()
                ).decode("ascii"),
            },
        )
        assert status == 201
        status, started = send(
            "POST",
            "/api/runs",
            {
                "profile_id": profile.profile_id,
                "goal": "maximize quantum yield",
                "input_artifact_ids": [uploaded["artifact_id"]],
                "llm_profile_ref": "provider:test",
            },
        )
        assert status == 201
        assert started["inspection"]["frozen_intent"]["spec"]["target_property"] == "quantum_yield"
        status, detail = send("GET", f"/api/runs/{started['run_id']}")
        assert status == 200
        assert detail["frozen_intent"]["spec"]["seed"] == 11
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
        app.close()
