from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.llm_provider import LLMProviderManager, StubLLMProvider
from ai4s_agent.llm_provider_resolution import (
    CONTROL_PLANE_ROLE,
    SCIENTIFIC_MAPPING_ROLE,
    resolve_llm_provider_payload,
)
from ai4s_agent.llm_settings import (
    LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE,
    LLMSettingsStore,
)


def _profile(
    profile_id: str,
    *,
    model: str,
    api_key_env: str,
    capabilities: dict[str, object] | None = None,
) -> dict[str, object]:
    profile: dict[str, object] = {
        "profile_id": profile_id,
        "provider": "openai_compatible",
        "endpoint": "http://127.0.0.1:8000/v1",
        "model": model,
        "timeout_sec": 60,
        "api_key_source": "environment",
        "api_key_ref": profile_id,
        "api_key_env": api_key_env,
    }
    if capabilities is not None:
        profile["capabilities"] = capabilities
    return profile


def _settings(tmp_path: Path, *, control_plane_eligible: bool = True) -> LLMSettingsStore:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    profiles = {
        "control-plane": _profile(
            "control-plane",
            model="authoritative-model",
            api_key_env="CONTROL_PLANE_KEY",
            capabilities={
                "structured_output_mode": "native_json_schema",
                "control_plane_eligible": control_plane_eligible,
            },
        ),
        "deepseek": _profile(
            "deepseek",
            model="mapping-model",
            api_key_env="SCIENTIFIC_MAPPING_KEY",
            capabilities={
                "structured_output_mode": "json_object_local_validation",
                "control_plane_eligible": False,
                "scientific_mapping_eligible": True,
            },
        ),
    }
    (config_dir / "llm_profiles.json").write_text(
        json.dumps(
            {
                "version": 3,
                "preferences": {"external_llm_data_sharing_enabled": False},
                "active_profile": profiles["control-plane"],
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    (config_dir / "llm_role_bindings.json").write_text(
        json.dumps(
            {
                "schema_version": "llm_role_bindings.v1",
                "bindings": {
                    "control_plane": "control-plane",
                    "scientific_mapping": "deepseek",
                },
            }
        ),
        encoding="utf-8",
    )
    return LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=config_dir,
        environ={
            "CONTROL_PLANE_KEY": "control-secret",
            "SCIENTIFIC_MAPPING_KEY": "mapping-secret",
        },
    )


def test_server_owned_role_binding_ignores_request_provider(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    resolution = resolve_llm_provider_payload(
        {
            "llm_provider": {
                "provider": "openai_compatible",
                "endpoint": "http://127.0.0.1:8999/v1",
                "model": "request-selected-attacker-model",
                "api_key": "request-secret",
            }
        },
        settings=settings,
        providers=providers,
        role=CONTROL_PLANE_ROLE,
    )

    assert resolution.server_owned is True
    assert resolution.role == CONTROL_PLANE_ROLE
    assert resolution.config is not None
    assert resolution.config.model == "authoritative-model"
    assert resolution.config.api_key == "control-secret"
    with resolution.provider_context as provider:
        assert isinstance(provider, StubLLMProvider)
    providers.close()


def test_role_binding_selects_mapping_profile_and_capabilities(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    resolution = resolve_llm_provider_payload(
        {},
        settings=settings,
        providers=providers,
        role=SCIENTIFIC_MAPPING_ROLE,
    )

    assert resolution.config is not None
    assert resolution.config.model == "mapping-model"
    assert resolution.config.capabilities.structured_output_mode == (
        "json_object_local_validation"
    )
    assert resolution.config.capabilities.scientific_mapping_eligible is True
    assert resolution.config.capabilities.control_plane_eligible is False
    providers.close()


def test_ineligible_control_plane_profile_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path, control_plane_eligible=False)
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    with pytest.raises(ValueError, match="not eligible for control_plane"):
        resolve_llm_provider_payload(
            {},
            settings=settings,
            providers=providers,
            role=CONTROL_PLANE_ROLE,
        )
    providers.close()


def test_role_bound_profile_missing_capabilities_fails_closed(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    document = json.loads(settings.path.read_text(encoding="utf-8"))
    del document["profiles"]["control-plane"]["capabilities"]
    settings.path.write_text(json.dumps(document), encoding="utf-8")
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    status, config = settings.resolve_role(CONTROL_PLANE_ROLE)
    assert status == LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE
    assert config is None
    with pytest.raises(ValueError, match="configured LLM settings are unavailable"):
        resolve_llm_provider_payload(
            {},
            settings=settings,
            providers=providers,
            role=CONTROL_PLANE_ROLE,
        )
    providers.close()


def test_role_bound_profile_missing_explicit_eligibility_fails_closed(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    document = json.loads(settings.path.read_text(encoding="utf-8"))
    document["profiles"]["control-plane"]["capabilities"] = {
        "structured_output_mode": "native_json_schema",
    }
    settings.path.write_text(json.dumps(document), encoding="utf-8")
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    with pytest.raises(ValueError, match="configured LLM settings are unavailable"):
        resolve_llm_provider_payload(
            {},
            settings=settings,
            providers=providers,
            role=CONTROL_PLANE_ROLE,
        )
    providers.close()


def test_missing_bound_profile_is_configured_but_unavailable(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.role_bindings_path.write_text(
        json.dumps(
            {
                "schema_version": "llm_role_bindings.v1",
                "bindings": {"control_plane": "missing-profile"},
            }
        ),
        encoding="utf-8",
    )
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    status, config = settings.resolve_role(CONTROL_PLANE_ROLE)
    assert status == LLM_SETTINGS_CONFIGURED_BUT_UNAVAILABLE
    assert config is None
    with pytest.raises(ValueError, match="configured LLM settings are unavailable"):
        resolve_llm_provider_payload(
            {},
            settings=settings,
            providers=providers,
            role=CONTROL_PLANE_ROLE,
        )
    providers.close()


def test_role_argument_keeps_legacy_request_resolution_without_role_file(
    tmp_path: Path,
) -> None:
    settings = LLMSettingsStore(
        workspace_dir=tmp_path / "workspace",
        config_dir=tmp_path / "config",
    )
    providers = LLMProviderManager(
        provider_factory=lambda _config: StubLLMProvider(response={"ok": True})
    )

    resolution = resolve_llm_provider_payload(
        {
            "llm_provider": {
                "provider": "stub",
                "model": "request-model",
                "capabilities": {"control_plane_eligible": False},
            }
        },
        settings=settings,
        providers=providers,
        role=CONTROL_PLANE_ROLE,
    )

    assert resolution.server_owned is False
    assert resolution.config is not None
    assert resolution.config.provider == "stub"
    assert resolution.config.model == "request-model"
    assert resolution.config.capabilities.control_plane_eligible is False
    providers.close()
