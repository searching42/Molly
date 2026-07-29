from __future__ import annotations

import json
import os
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

from ai4s_agent.app import create_app
from ai4s_agent.llm_settings import LLMSettingsStore


class PasswordDeleteError(Exception):
    pass


class _FakeKeyring:
    def __init__(self) -> None:
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, ref: str) -> str | None:
        return self.passwords.get((service, ref))

    def set_password(self, service: str, ref: str, secret: str) -> None:
        self.passwords[(service, ref)] = secret

    def delete_password(self, service: str, ref: str) -> None:
        try:
            del self.passwords[(service, ref)]
        except KeyError as exc:
            raise PasswordDeleteError(str(exc)) from exc


def _app(tmp_path: Path):
    return create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "user-config",
    )


def test_settings_are_user_scoped_and_profile_never_contains_secret(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    initial = client.get("/api/settings/llm")
    assert initial.status_code == 200
    assert initial.json == {"ok": True, "configured": False, "config": None}

    saved = client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1/",
            "model": "decision-model",
            "timeout_sec": 75,
            "api_key_source": "file",
            "api_key": "secret-token",
        },
    )
    assert saved.status_code == 200
    assert saved.json["config"]["api_key_configured"] is True
    assert saved.json["config"]["resolved_api_key_source"] == "file"
    assert "api_key" not in saved.json["config"]

    profile_path = tmp_path / "user-config" / "llm_profiles.json"
    secret_path = tmp_path / "user-config" / "secrets" / "default.key"
    assert "secret-token" not in profile_path.read_text(encoding="utf-8")
    assert secret_path.read_text(encoding="utf-8") == "secret-token"
    assert stat.S_IMODE(profile_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    fetched = client.get("/api/settings/llm")
    assert fetched.headers["Cache-Control"] == "no-store"
    assert "secret-token" not in fetched.get_data(as_text=True)


def test_patch_omission_retains_key_and_delete_is_explicit(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    payload = {
        "endpoint": "http://127.0.0.1:8000/v1",
        "model": "model-a",
        "api_key_source": "file",
        "api_key": "keep-me",
    }
    assert client.patch("/api/settings/llm", json=payload).status_code == 200
    assert client.patch("/api/settings/llm", json={"model": "model-b"}).status_code == 200
    assert client.patch("/api/settings/llm", json={"api_key": ""}).status_code == 400

    secret_path = tmp_path / "user-config" / "secrets" / "default.key"
    assert secret_path.read_text(encoding="utf-8") == "keep-me"
    assert client.delete("/api/settings/llm/api-key").status_code == 200
    assert not secret_path.exists()


def test_saved_llm_settings_can_be_verified_with_minimal_safe_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    assert client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1",
            "model": "decision-model",
            "api_key_source": "file",
            "api_key": "server-only-secret",
        },
    ).status_code == 200
    calls: list[tuple[list[dict[str, str]], str]] = []

    class ProbeProvider:
        def complete_text(self, *, messages, prompt_version):
            calls.append((messages, prompt_version))
            return "OK"

    @contextmanager
    def fake_lease(config):
        assert config.api_key == "server-only-secret"
        yield ProbeProvider()

    monkeypatch.setattr(app.extensions["llm_provider_manager"], "lease", fake_lease)
    response = client.post("/api/settings/llm/probe", json={})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json["probe"] == {
        "model": "decision-model",
        "provider": "openai_compatible",
        "request_kind": "minimal_chat_completion",
        "status": "available",
    }
    assert calls == [
        (
            [{"role": "user", "content": "Reply only with OK."}],
            "llm-settings-connection-probe.v1",
        )
    ]
    assert "server-only-secret" not in response.get_data(as_text=True)


def test_llm_probe_failure_is_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    assert client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1",
            "model": "decision-model",
            "api_key_source": "file",
            "api_key": "server-only-secret",
        },
    ).status_code == 200

    @contextmanager
    def failing_lease(_config):
        raise RuntimeError("/private/secret/path server-only-secret")
        yield  # pragma: no cover

    monkeypatch.setattr(app.extensions["llm_provider_manager"], "lease", failing_lease)
    response = client.post("/api/settings/llm/probe", json={})
    assert response.status_code == 409
    assert response.json["error_code"] == "llm_connection_failed"
    assert "/private/secret/path" not in response.get_data(as_text=True)
    assert "server-only-secret" not in response.get_data(as_text=True)


def test_environment_source_is_explicit(tmp_path: Path) -> None:
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        environ={"SELECTED_MOLLY_KEY": "env-secret"},
    )
    config = store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "environment",
            "api_key_env": "SELECTED_MOLLY_KEY",
        }
    )
    assert config.api_key == "env-secret"
    assert store.public_state()["config"]["api_key_source"] == "environment"
    assert store.public_state()["config"]["resolved_api_key_source"] == "environment"
    assert "env-secret" not in store.path.read_text(encoding="utf-8")


def test_legacy_workspace_secret_is_migrated_and_redacted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    legacy = workspace / ".ai4s" / "llm_provider.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(
        json.dumps(
            {
                "provider": "openai_compatible",
                "endpoint": "https://llm.example.test/v1",
                "model": "legacy-model",
                "api_key": "legacy-secret",
                "timeout_sec": 60,
            }
        ),
        encoding="utf-8",
    )
    store = LLMSettingsStore(workspace, config_dir=tmp_path / "config")
    assert store.read().api_key == "legacy-secret"
    assert "legacy-secret" not in legacy.read_text(encoding="utf-8")
    assert json.loads(legacy.read_text(encoding="utf-8"))["migrated_to_user_config"] is True


def test_endpoint_policy_rejects_remote_http_and_url_credentials(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    remote_http = client.patch(
        "/api/settings/llm",
        json={"endpoint": "http://llm.example.test/v1", "model": "m"},
    )
    credentials = client.patch(
        "/api/settings/llm",
        json={"endpoint": "https://user:pass@llm.example.test/v1", "model": "m"},
    )
    assert remote_http.status_code == 400
    assert "must use https" in remote_http.json["error"]
    assert credentials.status_code == 400
    assert "must not contain credentials" in credentials.json["error"]


def test_browser_origin_writes_require_startup_token(tmp_path: Path) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    page = client.get("/")
    assert page.headers["Cache-Control"] == "no-store"
    assert app.config["MOLLY_LOCAL_SESSION_TOKEN"] in page.get_data(as_text=True)
    payload = {
        "endpoint": "https://llm.example.test/v1",
        "model": "model-a",
        "api_key_source": "file",
    }
    missing = client.patch(
        "/api/settings/llm",
        json=payload,
        headers={"Origin": "http://127.0.0.1:8792", "Sec-Fetch-Site": "same-origin"},
    )
    allowed = client.patch(
        "/api/settings/llm",
        json=payload,
        headers={
            "Origin": "http://127.0.0.1:8792",
            "Sec-Fetch-Site": "same-origin",
            "X-Molly-Local-Token": app.config["MOLLY_LOCAL_SESSION_TOKEN"],
        },
    )
    cross_site = client.patch(
        "/api/settings/llm",
        json=payload,
        headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
    )
    assert missing.status_code == 403
    assert allowed.status_code == 200
    assert cross_site.status_code == 403


def test_remote_peer_cannot_spoof_loopback_host_header(tmp_path: Path) -> None:
    client = _app(tmp_path).test_client()
    response = client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "file",
        },
        headers={"Host": "127.0.0.1:8792"},
        environ_overrides={"REMOTE_ADDR": "192.0.2.20"},
    )
    assert response.status_code == 403
    assert response.json["error"] == "remote client is not loopback"


def test_managed_secret_moves_without_leaving_old_sources(tmp_path: Path) -> None:
    keyring = _FakeKeyring()
    environment = {"SELECTED_KEY": "environment-secret"}
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        environ=environment,
        keyring_backend=keyring,
    )
    common = {"endpoint": "https://llm.example.test/v1", "model": "model-a"}
    store.patch({**common, "api_key_source": "file", "api_key": "managed-secret"})
    default_file = tmp_path / "config" / "secrets" / "default.key"

    store.patch({"api_key_source": "keyring"})
    assert not default_file.exists()
    assert keyring.get_password("Molly", "default") == "managed-secret"

    store.patch({"api_key_source": "file", "api_key_ref": "rotated"})
    rotated_file = tmp_path / "config" / "secrets" / "rotated.key"
    assert keyring.get_password("Molly", "default") is None
    assert rotated_file.read_text(encoding="utf-8") == "managed-secret"

    store.patch({"profile_id": "profile-two", "api_key_ref": "profile-two"})
    profile_file = tmp_path / "config" / "secrets" / "profile-two.key"
    assert not rotated_file.exists()
    assert profile_file.read_text(encoding="utf-8") == "managed-secret"

    store.patch({"api_key_source": "environment", "api_key_env": "SELECTED_KEY"})
    assert not profile_file.exists()
    assert store.read().api_key == "environment-secret"


def test_auto_is_read_only_and_reports_effective_source(tmp_path: Path) -> None:
    keyring = _FakeKeyring()
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        environ={"AUTO_KEY": "environment-secret"},
        keyring_backend=keyring,
    )
    store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "file",
            "api_key": "file-secret",
        }
    )
    store.patch({"api_key_source": "auto", "api_key_env": "AUTO_KEY"})
    state = store.public_state()["config"]
    assert state["api_key_source"] == "auto"
    assert state["resolved_api_key_source"] == "environment"
    assert (tmp_path / "config" / "secrets" / "default.key").read_text(
        encoding="utf-8"
    ) == "file-secret"

    with pytest.raises(ValueError, match="select keyring or file"):
        store.patch({"api_key": "ambiguous-secret"})
    with pytest.raises(ValueError, match="auto is read-only discovery"):
        store.delete_api_key()
    assert store.read().api_key == "environment-secret"

    fallback_keyring = _FakeKeyring()
    fallback_store = LLMSettingsStore(
        workspace_dir=tmp_path / "fallback-workspace",
        config_dir=tmp_path / "fallback-config",
        environ={},
        keyring_backend=fallback_keyring,
    )
    fallback_store._write_managed_secret(("file", "default"), "file-fallback")
    fallback_keyring.set_password("Molly", "default", "keyring-fallback")
    fallback_store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "auto",
        }
    )
    assert fallback_store.public_state()["config"]["resolved_api_key_source"] == "keyring"
    fallback_keyring.delete_password("Molly", "default")
    assert fallback_store.public_state()["config"]["resolved_api_key_source"] == "file"


def test_auto_reference_change_never_deletes_discovered_secrets(tmp_path: Path) -> None:
    keyring = _FakeKeyring()
    keyring.set_password("Molly", "default", "discovered-keyring-secret")
    keyring_store = LLMSettingsStore(
        workspace_dir=tmp_path / "keyring-workspace",
        config_dir=tmp_path / "keyring-config",
        environ={},
        keyring_backend=keyring,
    )
    keyring_store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "auto",
            "api_key_ref": "default",
        }
    )
    assert keyring_store.public_state()["config"]["resolved_api_key_source"] == "keyring"
    keyring_store.patch({"api_key_ref": "other"})
    assert keyring.get_password("Molly", "default") == "discovered-keyring-secret"
    assert keyring_store.public_state()["config"]["resolved_api_key_source"] == "unavailable"

    file_store = LLMSettingsStore(
        workspace_dir=tmp_path / "file-workspace",
        config_dir=tmp_path / "file-config",
        environ={},
        keyring_backend=_FakeKeyring(),
    )
    file_store._write_managed_secret(("file", "default"), "discovered-file-secret")
    file_store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "auto",
            "api_key_ref": "default",
        }
    )
    old_file = tmp_path / "file-config" / "secrets" / "default.key"
    file_store.patch({"api_key_ref": "other"})
    assert old_file.read_text(encoding="utf-8") == "discovered-file-secret"
    assert file_store.public_state()["config"]["resolved_api_key_source"] == "unavailable"


def test_auto_to_explicit_storage_requires_reentered_key_and_preserves_discovery_source(
    tmp_path: Path,
) -> None:
    keyring = _FakeKeyring()
    keyring.set_password("Molly", "default", "discovered-keyring-secret")
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        environ={},
        keyring_backend=keyring,
    )
    store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "auto",
            "api_key_ref": "default",
        }
    )

    with pytest.raises(ValueError, match="requires a newly supplied api_key"):
        store.patch({"api_key_source": "file", "api_key_ref": "explicit"})
    assert keyring.get_password("Molly", "default") == "discovered-keyring-secret"
    assert not (tmp_path / "config" / "secrets" / "explicit.key").exists()
    assert store.public_state()["config"]["api_key_source"] == "auto"

    store.patch(
        {
            "api_key_source": "file",
            "api_key_ref": "explicit",
            "api_key": "reentered-explicit-secret",
        }
    )
    assert keyring.get_password("Molly", "default") == "discovered-keyring-secret"
    assert (tmp_path / "config" / "secrets" / "explicit.key").read_text(
        encoding="utf-8"
    ) == "reentered-explicit-secret"


def test_profile_write_failure_rolls_back_secret_migration(tmp_path: Path, monkeypatch) -> None:
    keyring = _FakeKeyring()
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        keyring_backend=keyring,
    )
    store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "file",
            "api_key": "original-secret",
        }
    )
    original_write_profile = store._write_profile

    def fail_after_new_profile(profile):
        original_write_profile(profile)
        if profile["api_key_source"] == "keyring":
            raise OSError("simulated profile failure")

    monkeypatch.setattr(store, "_write_profile", fail_after_new_profile)
    with pytest.raises(ValueError, match="simulated profile failure"):
        store.patch({"api_key_source": "keyring"})

    assert store.public_state()["config"]["api_key_source"] == "file"
    assert (tmp_path / "config" / "secrets" / "default.key").read_text(
        encoding="utf-8"
    ) == "original-secret"
    assert keyring.get_password("Molly", "default") is None


def test_old_secret_cleanup_failure_rolls_back_profile_and_target(tmp_path: Path, monkeypatch) -> None:
    keyring = _FakeKeyring()
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
        keyring_backend=keyring,
    )
    store.patch(
        {
            "endpoint": "https://llm.example.test/v1",
            "model": "model-a",
            "api_key_source": "file",
            "api_key": "original-secret",
        }
    )
    original_delete = store._delete_managed_secret
    failed = False

    def fail_old_file_once(identity):
        nonlocal failed
        if identity == ("file", "default") and not failed:
            failed = True
            raise OSError("simulated cleanup failure")
        original_delete(identity)

    monkeypatch.setattr(store, "_delete_managed_secret", fail_old_file_once)
    with pytest.raises(ValueError, match="simulated cleanup failure"):
        store.patch({"api_key_source": "keyring"})

    assert store.public_state()["config"]["api_key_source"] == "file"
    assert (tmp_path / "config" / "secrets" / "default.key").read_text(
        encoding="utf-8"
    ) == "original-secret"
    assert keyring.get_password("Molly", "default") is None


def test_file_secret_uses_atomic_0600_replace(tmp_path: Path, monkeypatch) -> None:
    store = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    store._secure_directory(store.secrets_dir)
    real_open = os.open
    created_modes: list[int] = []

    def tracked_open(path, flags, mode=0o777, *args, **kwargs):
        if flags & os.O_CREAT and str(path).endswith(".tmp"):
            created_modes.append(mode)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr("ai4s_agent.llm_settings.os.open", tracked_open)
    target = store._secret_file_for_ref("atomic")
    store._atomic_write_secret(target, "first-secret")
    assert created_modes == [0o600]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(target.parent.glob(".atomic.key.*.tmp"))

    def fail_replace(_source, _target):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("ai4s_agent.llm_settings.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        store._atomic_write_secret(target, "second-secret")
    assert target.read_text(encoding="utf-8") == "first-secret"
    assert not list(target.parent.glob(".atomic.key.*.tmp"))


def test_ui_exposes_explicit_secret_source_and_delete_controls(tmp_path: Path) -> None:
    html = _app(tmp_path).test_client().get("/").get_data(as_text=True)
    assert 'id="llm-settings-modal"' in html
    assert 'id="llm-settings-button"' in html
    assert 'id="llm-api-key-source"' in html
    assert 'id="llm-api-key-delete"' in html
    assert 'id="llm-resolved-source"' in html
    assert 'type="password"' in html
    assert 'patchJSON("/api/settings/llm", payload)' in html
    assert 'deleteJSON("/api/settings/llm/api-key")' in html
    assert 'postJSON("/api/settings/llm/probe", {})' in html
    assert "保存并测试 API 连接" in html
