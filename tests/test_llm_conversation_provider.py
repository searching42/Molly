from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

from ai4s_agent.app import create_app


pytestmark = pytest.mark.integration


def _app(tmp_path: Path):
    return create_app(
        base_runs_dir=tmp_path / "runs",
        workspace_dir=tmp_path / "workspace",
        user_config_dir=tmp_path / "user-config",
    )


def _configure_external_llm(client) -> None:
    response = client.patch(
        "/api/settings/llm",
        json={
            "endpoint": "https://llm.example.test/v1",
            "model": "conversation-model",
            "api_key_source": "file",
            "api_key": "server-only-secret",
        },
    )
    assert response.status_code == 200


@pytest.mark.pr_fast
def test_saved_llm_settings_can_be_verified_with_minimal_safe_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    _configure_external_llm(client)
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
        "model": "conversation-model",
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


@pytest.mark.pr_fast
def test_external_llm_conversation_requires_consent_then_returns_provider_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    _configure_external_llm(client)
    calls: list[tuple[list[dict[str, str]], str]] = []

    class ConversationProvider:
        def complete_text(self, *, messages, prompt_version):
            calls.append((messages, prompt_version))
            return "请明确希望训练的目标属性，例如 PLQY。"

    @contextmanager
    def fake_lease(config):
        assert config.model == "conversation-model"
        yield ConversationProvider()

    monkeypatch.setattr(app.extensions["llm_provider_manager"], "lease", fake_lease)
    payload = {
        "project_id": "proj-llm-chat",
        "run_id": "run-llm-chat",
        "messages": [{"role": "user", "content": "我想研究发光材料。"}],
    }

    blocked = client.post("/api/agent/conversation/next-turn", json=payload)
    allowed = client.post(
        "/api/agent/conversation/next-turn",
        json={**payload, "external_llm_approved": True},
    )

    assert blocked.status_code == 400
    assert blocked.json["error_code"] == "external_llm_approval_required"
    assert allowed.status_code == 200
    assert allowed.json["llm_used"] is True
    assert allowed.json["assistant_source"] == "configured_llm"
    assert allowed.json["assistant_message"] == "请明确希望训练的目标属性，例如 PLQY。"
    assert allowed.json["decision"]["status"] == "needs_clarification"
    assert len(calls) == 1
    messages, prompt_version = calls[0]
    assert prompt_version == "conversation-assistant-response.v1"
    assert messages[-1] == {"role": "user", "content": "我想研究发光材料。"}
    assert "deterministic decision" in messages[0]["content"].lower()


def test_conversation_without_configured_llm_is_explicitly_deterministic(
    tmp_path: Path,
) -> None:
    response = _app(tmp_path).test_client().post(
        "/api/agent/conversation/next-turn",
        json={
            "project_id": "proj-rule-chat",
            "run_id": "run-rule-chat",
            "messages": [{"role": "user", "content": "Train OLED PLQY."}],
        },
    )

    assert response.status_code == 200
    assert response.json["llm_used"] is False
    assert response.json["assistant_source"] == "deterministic_rules"
    assert response.json["assistant_message"] == response.json["decision"]["summary"]


@pytest.mark.pr_fast
def test_llm_conversation_and_probe_failures_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    _configure_external_llm(client)

    class FailingProvider:
        def complete_text(self, *, messages, prompt_version):
            del messages, prompt_version
            raise RuntimeError("/private/secret/path server-only-secret")

    @contextmanager
    def failing_lease(_config):
        yield FailingProvider()

    monkeypatch.setattr(app.extensions["llm_provider_manager"], "lease", failing_lease)
    with caplog.at_level(logging.WARNING):
        probe = client.post("/api/settings/llm/probe", json={})
        conversation = client.post(
            "/api/agent/conversation/next-turn",
            json={
                "project_id": "proj-failed-chat",
                "run_id": "run-failed-chat",
                "messages": [{"role": "user", "content": "Train OLED PLQY."}],
                "external_llm_approved": True,
            },
        )

    assert probe.status_code == 409
    assert probe.json["error_code"] == "llm_connection_failed"
    assert conversation.status_code == 409
    assert conversation.json["error_code"] == "llm_conversation_failed"
    combined = probe.get_data(as_text=True) + conversation.get_data(as_text=True)
    assert "/private/secret/path" not in combined
    assert "server-only-secret" not in combined
    assert "/private/secret/path" not in caplog.text
    assert "server-only-secret" not in caplog.text
    assert "llm_connection_failed" in caplog.text
    assert "llm_conversation_failed" in caplog.text


@pytest.mark.parametrize("secret_source", ["file", "environment", "keyring"])
@pytest.mark.pr_fast
def test_configured_llm_with_unavailable_secret_fails_closed(
    tmp_path: Path,
    secret_source: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _app(tmp_path)
    client = app.test_client()
    _configure_external_llm(client)
    profile_path = tmp_path / "user-config" / "llm_profiles.json"
    profile_document = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = profile_document["active_profile"]
    profile["api_key_source"] = secret_source
    if secret_source == "environment":
        profile["api_key_env"] = "MOLLY_TEST_KEY_THAT_IS_NOT_SET"
    elif secret_source == "keyring":
        profile["api_key_ref"] = "missing-test-key"
    profile_path.write_text(json.dumps(profile_document), encoding="utf-8")
    (tmp_path / "user-config" / "secrets" / "default.key").unlink()

    with caplog.at_level(logging.WARNING):
        response = client.post(
            "/api/agent/conversation/next-turn",
            json={
                "project_id": "proj-unavailable-llm",
                "run_id": "run-unavailable-llm",
                "messages": [{"role": "user", "content": "Train OLED PLQY."}],
                "external_llm_approved": True,
            },
        )

    assert response.status_code == 409
    assert response.json == {
        "ok": False,
        "error_code": "llm_conversation_unavailable",
        "error": "The configured LLM could not be prepared for this conversation.",
    }
    assert "assistant_source" not in response.json
    assert "llm_conversation_unavailable" in caplog.text


@pytest.mark.pr_fast
def test_ui_probes_llm_and_requires_explicit_external_conversation_consent(
    tmp_path: Path,
) -> None:
    html = _app(tmp_path).test_client().get("/").get_data(as_text=True)

    assert 'id="conversation-external-llm-approved"' in html
    assert "external_llm_approved: document.getElementById" in html
    assert "response.assistant_message || currentConversationDecision?.summary" in html
    assert 'postJSON("/api/settings/llm/probe", {})' in html
    assert "保存并测试 API 连接" in html
    assert "未配置可用 LLM，已使用确定性决策摘要" in html
    assert "function resetExternalLLMApproval()" in html
    assert "function selectProject(project) {\n        resetExternalLLMApproval();" in html
    assert "if (nextConversationId !== currentConversationId) resetExternalLLMApproval();" in html
