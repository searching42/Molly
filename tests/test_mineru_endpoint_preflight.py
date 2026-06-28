from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from ai4s_agent.mineru_endpoint_preflight import (
    MinerUEndpointPreflightReport,
    main,
    run_mineru_endpoint_preflight,
)


def _profile_config(tmp_path: Path, *, api_url: str = "http://127.0.0.1:18000/path") -> Path:
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "mineru_endpoint_profiles.v1",
                "profiles": [
                    {
                        "name": "node45-loopback",
                        "api_url": api_url,
                        "endpoint_kind": "mineru-api",
                        "backend": "hybrid-engine",
                        "effort": "medium",
                        "expected_protocol_version": "2",
                        "health_path": "/health",
                    }
                ],
                "routing_policies": [
                    {
                        "name": "manual-primary",
                        "default_profile": "node45-loopback",
                        "fallback_profiles": [],
                        "mode": "manual",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_preflight_passes_with_mock_health_profile_and_redacted_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _profile_config(tmp_path)
    seen_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_requests.append(request)
        return httpx.Response(
            200,
            json={
                "status": "healthy",
                "version": "3.4.0",
                "protocol_version": 2,
            },
        )

    _patch_environment(monkeypatch)
    report = run_mineru_endpoint_preflight(
        output_dir=tmp_path / "out",
        run_id="preflight-ok",
        profile_config=config_path,
        policy_name="manual-primary",
        transport=httpx.MockTransport(handler),
        generated_at="2026-06-28T00:00:00Z",
    )
    run_root = tmp_path / "out" / "preflight-ok"
    persisted = MinerUEndpointPreflightReport.model_validate_json(
        (run_root / "preflight_report.json").read_text(encoding="utf-8")
    )
    raw = json.dumps(persisted.model_dump(mode="json"))

    assert report.decision == "passed"
    assert report.health.ok is True
    assert report.health.http_status_code == 200
    assert report.health.status == "healthy"
    assert report.health.mineru_version == "3.4.0"
    assert report.health.protocol_version == "2"
    assert report.profile.redacted_api_origin == "http://127.0.0.1:18000"
    assert report.profile.health_path == "/health"
    assert report.environment.cuda_home == "/usr/local/cuda-12.8"
    assert report.environment.torch_cuda_version == "13.0"
    assert report.environment.torch_cuda_available is True
    assert report.environment.nvidia_smi_driver_version == "580.126.20"
    assert report.environment.diagnostics["VLLM_USE_FLASHINFER_SAMPLER"]["recommendation"] == "set_to_0"
    assert report.outputs["preflight_report"] == "preflight_report.json"
    assert (run_root / report.outputs["preflight_summary"]).exists()
    assert seen_requests[0].url.path == "/health"
    assert "path" not in report.profile.redacted_api_origin
    assert "Authorization" not in raw
    assert "secret" not in raw.lower()


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        ({"status": "healthy", "version": "3.4.0", "protocol_version": 1}, "unsupported_protocol_version"),
        ({"status": "down", "version": "3.4.0", "protocol_version": 2}, "unhealthy_status"),
        ({"status": "healthy", "version": "3.4.0"}, "missing_protocol_version"),
        ({"version": "3.4.0", "protocol_version": 2}, "missing_status"),
        (["not", "object"], "invalid_health_schema"),
    ],
)
def test_preflight_fails_schema_or_protocol_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    expected_code: str,
) -> None:
    _patch_environment(monkeypatch)
    report = run_mineru_endpoint_preflight(
        output_dir=tmp_path / "out",
        run_id=f"preflight-{expected_code}",
        profile_config=_profile_config(tmp_path),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
    )

    assert report.decision == "failed"
    assert any(error["code"] == expected_code for error in report.errors)


def test_preflight_fails_when_http_unreachable_without_leaking_url_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = _profile_config(tmp_path, api_url="http://127.0.0.1:18000/path")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed with secret-token", request=request)

    _patch_environment(monkeypatch)
    report = run_mineru_endpoint_preflight(
        output_dir=tmp_path / "out",
        run_id="preflight-unreachable",
        profile_config=config_path,
        transport=httpx.MockTransport(handler),
        api_token="secret-token",
    )
    raw = report.model_dump_json()

    assert report.decision == "failed"
    assert report.health.ok is False
    assert any(error["code"] == "health_unreachable" for error in report.errors)
    assert report.profile.redacted_api_origin == "http://127.0.0.1:18000"
    assert "secret-token" not in raw
    assert "Authorization" not in raw


def test_preflight_rejects_profile_config_with_sensitive_api_url(tmp_path: Path) -> None:
    report = run_mineru_endpoint_preflight(
        output_dir=tmp_path / "out",
        run_id="preflight-sensitive-url",
        profile_config=_profile_config(tmp_path, api_url="http://user:secret@127.0.0.1:18000?token=abc"),
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "healthy"})),
    )
    raw = report.model_dump_json()

    assert report.decision == "failed"
    assert any(error["code"] == "profile_config_error" for error in report.errors)
    assert "secret" not in raw.lower()
    assert "abc" not in raw


def test_preflight_rejects_profile_config_with_sensitive_health_path(tmp_path: Path) -> None:
    config_path = _profile_config(tmp_path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["profiles"][0]["health_path"] = "/health?token=abc"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    report = run_mineru_endpoint_preflight(
        output_dir=tmp_path / "out",
        run_id="preflight-sensitive-health-path",
        profile_config=config_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "healthy"})),
    )
    raw = report.model_dump_json()

    assert report.decision == "failed"
    assert any(error["code"] == "profile_config_error" for error in report.errors)
    assert "abc" not in raw
    assert "token=abc" not in raw


def test_preflight_cli_rejects_sensitive_health_path_without_leaking_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_environment(monkeypatch)
    exit_code = main(
        [
            "--api-url",
            "http://127.0.0.1:18000",
            "--endpoint-kind",
            "mineru-api",
            "--health-path",
            "/health?token=abc",
            "--output",
            str(tmp_path / "out"),
            "--run-id",
            "preflight-cli-sensitive-health-path",
        ],
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"status": "healthy", "protocol_version": 2})),
    )
    stdout = capsys.readouterr().out

    assert exit_code == 1
    payload = json.loads(stdout)
    assert payload["decision"] == "failed"
    assert "abc" not in stdout
    assert "token=abc" not in stdout


def test_preflight_cli_writes_report_and_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_environment(monkeypatch)
    config_path = _profile_config(tmp_path)
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"status": "healthy", "version": "3.4.0", "protocol_version": 2})
    )

    exit_code = main(
        [
            "--profile-config",
            str(config_path),
            "--policy-name",
            "manual-primary",
            "--output",
            str(tmp_path / "out"),
            "--run-id",
            "preflight-cli",
        ],
        transport=transport,
    )
    stdout = capsys.readouterr().out

    assert exit_code == 0
    assert json.loads(stdout)["decision"] == "passed"
    assert (tmp_path / "out" / "preflight-cli" / "preflight_report.json").exists()
    assert (tmp_path / "out" / "preflight-cli" / "preflight_summary.md").exists()


def _patch_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda-12.8")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda-12.4/lib64:/home/lbh/miniconda3/envs/mineru34/lib")

    import ai4s_agent.mineru_endpoint_preflight as module

    monkeypatch.setattr(
        module,
        "_torch_diagnostics",
        lambda: {
            "torch_version": "2.11.0+cu130",
            "torch_cuda_version": "13.0",
            "torch_cuda_available": True,
            "torch_cuda_device_name": "NVIDIA GeForce RTX 5090",
            "torch_cuda_device_capability": "12.0",
        },
    )
    monkeypatch.setattr(
        module,
        "_nvidia_smi_diagnostics",
        lambda: {
            "available": True,
            "driver_version": "580.126.20",
            "reported_cuda_version": "13.0",
            "raw": "NVIDIA-SMI 580.126.20 Driver Version: 580.126.20 CUDA Version: 13.0",
        },
    )
