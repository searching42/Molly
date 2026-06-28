from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent.mineru_endpoint_profiles import (
    MinerUEndpointProfileConfigError,
    load_mineru_endpoint_profile_config,
    resolve_mineru_endpoint_profile,
)


EXAMPLE_PROFILE = Path("docs/examples/mineru-endpoint-profiles.example.json")


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _minimal_config(**overrides: object) -> dict:
    profile = {
        "name": "node45-loopback",
        "api_url": "http://127.0.0.1:18000",
        "endpoint_kind": "mineru-api",
    }
    profile.update(overrides.pop("profile_overrides", {}))
    payload = {
        "schema_version": "mineru_endpoint_profiles.v1",
        "profiles": [profile],
        "routing_policies": [
            {
                "name": "manual-primary",
                "default_profile": "node45-loopback",
                "fallback_profiles": ["node45-backup"],
                "mode": "manual",
            }
        ],
    }
    backup = {
        "name": "node45-backup",
        "api_url": "http://127.0.0.1:18001",
        "endpoint_kind": "mineru-api",
    }
    payload["profiles"].append(backup)
    payload.update(overrides)
    return payload


def test_example_profile_loads_and_resolves_default_policy() -> None:
    config = load_mineru_endpoint_profile_config(EXAMPLE_PROFILE)
    resolved = resolve_mineru_endpoint_profile(config, profile_name=None, policy_name=None)

    assert resolved.profile.name == "node45-loopback"
    assert resolved.profile.api_url == "http://127.0.0.1:18000"
    assert resolved.profile.endpoint_kind == "mineru_api"
    assert resolved.profile.backend == "hybrid-engine"
    assert resolved.profile.effort == "medium"
    assert resolved.profile.parse_method == "auto"
    assert resolved.profile.allow_remote_upload is True
    assert resolved.profile.compare_pdfplumber is True
    assert resolved.profile.expected_protocol_version == "2"
    assert resolved.routing_policy_name == "manual-primary"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["profiles"].append(dict(payload["profiles"][0])), "duplicate profile"),
        (lambda payload: payload["routing_policies"].append(dict(payload["routing_policies"][0])), "duplicate routing policy"),
        (lambda payload: payload["routing_policies"][0].update(default_profile="missing"), "default profile"),
        (lambda payload: payload["routing_policies"][0].update(fallback_profiles=["missing"]), "fallback profile"),
        (lambda payload: payload["profiles"][0].update(api_url="ftp://127.0.0.1:18000"), "http or https"),
        (lambda payload: payload["profiles"][0].update(api_url="http://user:pass@127.0.0.1:18000"), "userinfo"),
        (lambda payload: payload["profiles"][0].update(api_url="http://127.0.0.1:18000?token=abc"), "query"),
        (lambda payload: payload["profiles"][0].update(api_url="http://127.0.0.1:18000#token"), "fragment"),
        (lambda payload: payload["profiles"][0].update(api_token="actual-secret-value"), "credential"),
    ],
)
def test_profile_config_validation_errors_are_safe(tmp_path: Path, mutate: object, message: str) -> None:
    payload = _minimal_config()
    mutate(payload)
    config_path = _write_config(tmp_path / "profiles.json", payload)

    with pytest.raises(MinerUEndpointProfileConfigError) as excinfo:
        load_mineru_endpoint_profile_config(config_path)

    error_text = str(excinfo.value)
    assert message in error_text
    assert "actual-secret-value" not in error_text
    assert "abc" not in error_text


def test_cli_overrides_take_precedence_and_summary_is_redacted(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "profiles.json",
        _minimal_config(
            profile_overrides={
                "backend": "vlm-transformers",
                "effort": "high",
                "parse_method": "ocr",
                "allow_remote_upload": False,
                "compare_pdfplumber": False,
                "http_timeout_sec": 10,
            }
        ),
    )
    config = load_mineru_endpoint_profile_config(config_path)

    resolved = resolve_mineru_endpoint_profile(
        config,
        profile_name="node45-loopback",
        policy_name="manual-primary",
        profile_source_path=config_path,
        cli_overrides={
            "api_url": "http://127.0.0.1:19000/path",
            "backend": "hybrid-engine",
            "effort": "medium",
            "allow_remote_upload": True,
            "compare_pdfplumber": True,
            "http_timeout_sec": 60.0,
        },
    )
    summary = resolved.redacted_summary(base_dir=tmp_path)

    assert resolved.profile.api_url == "http://127.0.0.1:19000/path"
    assert resolved.profile.backend == "hybrid-engine"
    assert resolved.profile.effort == "medium"
    assert resolved.profile.allow_remote_upload is True
    assert resolved.profile.compare_pdfplumber is True
    assert resolved.profile.http_timeout_sec == 60.0
    assert summary["endpoint_profile_name"] == "node45-loopback"
    assert summary["routing_policy_name"] == "manual-primary"
    assert summary["redacted_api_origin"] == "http://127.0.0.1:19000"
    assert summary["routing_fallback_profile_names"] == ["node45-backup"]
    assert "19000/path" not in json.dumps(summary)
    assert "token" not in json.dumps(summary).lower()


def test_routing_policy_default_is_deterministic_and_no_fallback_is_attempted(tmp_path: Path) -> None:
    config = load_mineru_endpoint_profile_config(_write_config(tmp_path / "profiles.json", _minimal_config()))
    resolved = resolve_mineru_endpoint_profile(config, profile_name=None, policy_name="manual-primary")

    assert resolved.profile.name == "node45-loopback"
    assert resolved.routing_policy_name == "manual-primary"
    assert resolved.routing_fallback_profile_names == ["node45-backup"]
