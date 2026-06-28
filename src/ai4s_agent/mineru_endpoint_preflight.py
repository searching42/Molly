from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.mineru_endpoint_profiles import (
    MinerUEndpointProfileConfigError,
    MinerUEndpointProfileReportSummary,
    ResolvedMinerUEndpointProfile,
    load_mineru_endpoint_profile_config,
    resolve_mineru_endpoint_profile,
)


_SCHEMA_VERSION = "mineru_endpoint_preflight.v1"


class MinerUEndpointHealthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    http_status_code: int = 0
    status: str = ""
    mineru_version: str = ""
    protocol_version: str = ""
    response_schema_valid: bool = False
    elapsed_seconds: float = 0.0
    health_path: str = "/health"


class MinerUEndpointEnvironmentDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vllm_use_flashinfer_sampler: str = ""
    cuda_home: str = ""
    ld_library_path_entries: list[str] = Field(default_factory=list)
    torch_version: str = ""
    torch_cuda_version: str = ""
    torch_cuda_available: bool | None = None
    torch_cuda_device_name: str = ""
    torch_cuda_device_capability: str = ""
    nvidia_smi_available: bool = False
    nvidia_smi_driver_version: str = ""
    nvidia_smi_reported_cuda_version: str = ""
    diagnostics: dict[str, dict[str, Any]] = Field(default_factory=dict)


class MinerUEndpointPreflightReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = _SCHEMA_VERSION
    run_id: str
    generated_at: str
    decision: str
    profile: MinerUEndpointProfileReportSummary
    health: MinerUEndpointHealthSummary
    environment: MinerUEndpointEnvironmentDiagnostics
    warnings: list[str] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
    outputs: dict[str, str] = Field(default_factory=dict)


def run_mineru_endpoint_preflight(
    *,
    output_dir: str | Path,
    run_id: str,
    profile_config: str | Path | None = None,
    profile_name: str | None = None,
    policy_name: str | None = None,
    api_url: str | None = None,
    endpoint_kind: str | None = None,
    expected_protocol_version: str | None = None,
    health_path: str | None = None,
    http_timeout_sec: float | None = None,
    api_token: str = "",
    transport: httpx.BaseTransport | None = None,
    generated_at: str | None = None,
) -> MinerUEndpointPreflightReport:
    generated = generated_at or now_iso()
    clean_run_id = str(run_id or "").strip()
    root = Path(output_dir).expanduser().resolve()
    run_root = root / clean_run_id if _safe_run_id(clean_run_id) else root
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    if not _safe_run_id(clean_run_id):
        errors.append(_error("invalid_run_id", "run_id must be a single safe path segment"))
        report = _failed_report(
            run_id=clean_run_id,
            generated_at=generated,
            errors=errors,
        )
        _persist_report(report, root)
        return report
    if run_root.exists() and any(run_root.iterdir()):
        errors.append(_error("output_directory_not_empty", "run-specific preflight output directory must be empty"))
        report = _failed_report(run_id=clean_run_id, generated_at=generated, errors=errors)
        _persist_report(report, run_root)
        return report

    resolved: ResolvedMinerUEndpointProfile | None = None
    try:
        resolved = _resolve_profile(
            profile_config=profile_config,
            profile_name=profile_name,
            policy_name=policy_name,
            api_url=api_url,
            endpoint_kind=endpoint_kind,
            expected_protocol_version=expected_protocol_version,
            health_path=health_path,
            http_timeout_sec=http_timeout_sec,
        )
        profile_summary = MinerUEndpointProfileReportSummary.model_validate(
            resolved.redacted_summary(base_dir=Path(profile_config).expanduser().resolve().parent if profile_config else None)
        )
    except Exception as exc:
        errors.append(_error("profile_config_error", _safe_error(str(exc))))
        report = _failed_report(
            run_id=clean_run_id,
            generated_at=generated,
            errors=errors,
            environment=collect_environment_diagnostics(),
        )
        _persist_report(report, run_root)
        return report

    environment = collect_environment_diagnostics()
    health = _check_health(
        resolved=resolved,
        api_token=api_token,
        transport=transport,
    )
    errors.extend(_health_errors(health=health, expected_protocol_version=resolved.profile.expected_protocol_version))
    if environment.diagnostics["VLLM_USE_FLASHINFER_SAMPLER"]["status"] != "ok":
        warnings.append("node45_hint_vllm_use_flashinfer_sampler")
    if environment.diagnostics["LD_LIBRARY_PATH"]["status"] != "ok":
        warnings.append("node45_hint_ld_library_path_cuda_cudnn_ordering")
    if environment.torch_cuda_available is False:
        warnings.append("torch_cuda_not_available")

    decision = "failed" if errors else "passed"
    report = MinerUEndpointPreflightReport(
        run_id=clean_run_id,
        generated_at=generated,
        decision=decision,
        profile=profile_summary,
        health=health,
        environment=environment,
        warnings=warnings,
        errors=errors,
    )
    _persist_report(report, run_root)
    return report


def collect_environment_diagnostics() -> MinerUEndpointEnvironmentDiagnostics:
    torch_info = _torch_diagnostics()
    nvidia_info = _nvidia_smi_diagnostics()
    vllm_value = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER", "")
    cuda_home = os.environ.get("CUDA_HOME", "")
    ld_entries = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    diagnostics = {
        "VLLM_USE_FLASHINFER_SAMPLER": {
            "value": vllm_value,
            "status": "ok" if vllm_value == "0" else "warning",
            "recommendation": "set_to_0",
            "reason": "node45 RTX 5090 live MinerU startup required VLLM_USE_FLASHINFER_SAMPLER=0",
        },
        "LD_LIBRARY_PATH": {
            "entries": ld_entries,
            "status": "ok" if _conda_runtime_precedes_system_cuda(ld_entries) else "warning",
            "recommendation": "put_conda_nvidia_runtime_libraries_before_system_cuda_lib64",
            "reason": "avoid loading stale system CUDA/cuDNN libraries before the mineru environment runtime",
        },
        "CUDA_HOME": {
            "value": cuda_home,
            "status": "present" if cuda_home else "missing",
            "recommendation": "verify CUDA_HOME matches the active torch CUDA runtime when set",
        },
    }
    return MinerUEndpointEnvironmentDiagnostics(
        vllm_use_flashinfer_sampler=vllm_value,
        cuda_home=cuda_home,
        ld_library_path_entries=ld_entries,
        torch_version=str(torch_info.get("torch_version") or ""),
        torch_cuda_version=str(torch_info.get("torch_cuda_version") or ""),
        torch_cuda_available=torch_info.get("torch_cuda_available") if isinstance(torch_info.get("torch_cuda_available"), bool) else None,
        torch_cuda_device_name=str(torch_info.get("torch_cuda_device_name") or ""),
        torch_cuda_device_capability=str(torch_info.get("torch_cuda_device_capability") or ""),
        nvidia_smi_available=bool(nvidia_info.get("available")),
        nvidia_smi_driver_version=str(nvidia_info.get("driver_version") or ""),
        nvidia_smi_reported_cuda_version=str(nvidia_info.get("reported_cuda_version") or ""),
        diagnostics=diagnostics,
    )


def _resolve_profile(
    *,
    profile_config: str | Path | None,
    profile_name: str | None,
    policy_name: str | None,
    api_url: str | None,
    endpoint_kind: str | None,
    expected_protocol_version: str | None,
    health_path: str | None,
    http_timeout_sec: float | None,
) -> ResolvedMinerUEndpointProfile:
    if profile_config is None:
        if api_url is None:
            raise MinerUEndpointProfileConfigError("profile_config or api_url is required")
        payload = {
            "schema_version": "mineru_endpoint_profiles.v1",
            "profiles": [
                {
                    "name": profile_name or "cli-endpoint",
                    "api_url": api_url,
                    "endpoint_kind": endpoint_kind or "mineru-api",
                    "expected_protocol_version": expected_protocol_version or "2",
                    "health_path": health_path or "/health",
                    **({"http_timeout_sec": http_timeout_sec} if http_timeout_sec is not None else {}),
                }
            ],
            "routing_policies": [],
        }
        from ai4s_agent.mineru_endpoint_profiles import MinerUEndpointProfileConfig

        config = MinerUEndpointProfileConfig.model_validate(payload)
        return resolve_mineru_endpoint_profile(config, profile_name=profile_name or "cli-endpoint", policy_name=None)
    config = load_mineru_endpoint_profile_config(profile_config)
    overrides = {
        "api_url": api_url,
        "endpoint_kind": endpoint_kind,
        "expected_protocol_version": expected_protocol_version,
        "health_path": health_path,
        "http_timeout_sec": http_timeout_sec,
    }
    return resolve_mineru_endpoint_profile(
        config,
        profile_name=profile_name,
        policy_name=policy_name,
        cli_overrides=overrides,
        profile_source_path=profile_config,
    )


def _check_health(
    *,
    resolved: ResolvedMinerUEndpointProfile,
    api_token: str,
    transport: httpx.BaseTransport | None,
) -> MinerUEndpointHealthSummary:
    profile = resolved.profile
    health_url = _health_url(profile.api_url, profile.health_path)
    headers = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    start = None
    try:
        import time

        start = time.monotonic()
        with httpx.Client(transport=transport, timeout=profile.http_timeout_sec, headers=headers) as client:
            response = client.get(health_url)
        elapsed = max(0.0, time.monotonic() - start) if start is not None else 0.0
    except Exception as exc:
        return MinerUEndpointHealthSummary(
            ok=False,
            response_schema_valid=False,
            elapsed_seconds=0.0,
            health_path=profile.health_path,
            status="",
            mineru_version="",
            protocol_version="",
        ).model_copy(update={"ok": False})

    payload: Any
    try:
        payload = response.json()
    except Exception:
        payload = None
    parsed = _parse_health_payload(payload)
    healthy_status = str(parsed["status"]).strip().lower() in {"healthy", "ok"}
    return MinerUEndpointHealthSummary(
        ok=response.status_code == 200 and parsed["schema_valid"] and healthy_status,
        http_status_code=response.status_code,
        status=parsed["status"],
        mineru_version=parsed["mineru_version"],
        protocol_version=parsed["protocol_version"],
        response_schema_valid=parsed["schema_valid"],
        elapsed_seconds=elapsed,
        health_path=profile.health_path,
    )


def _parse_health_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"schema_valid": False, "status": "", "mineru_version": "", "protocol_version": ""}
    status = str(payload.get("status") or "").strip()
    version = str(payload.get("version") or payload.get("version_name") or payload.get("_version_name") or "").strip()
    protocol = str(payload.get("protocol_version") or "").strip()
    schema_valid = bool(status and protocol)
    return {
        "schema_valid": schema_valid,
        "status": status,
        "mineru_version": version,
        "protocol_version": protocol,
    }


def _health_errors(*, health: MinerUEndpointHealthSummary, expected_protocol_version: str) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    if health.http_status_code == 0 and not health.ok:
        errors.append(_error("health_unreachable", "MinerU health endpoint is unreachable"))
        return errors
    if health.http_status_code != 200:
        errors.append(_error("health_http_error", "MinerU health endpoint returned a non-200 response", {"http_status_code": health.http_status_code}))
    if not health.response_schema_valid:
        errors.append(_error("invalid_health_schema", "MinerU health response must be a JSON object with status and protocol_version"))
    if not health.status:
        errors.append(_error("missing_status", "MinerU health response is missing status"))
    elif health.status.strip().lower() not in {"healthy", "ok"}:
        errors.append(
            _error(
                "unhealthy_status",
                "MinerU health status is not healthy",
                {"observed": health.status, "accepted": ["healthy", "ok"]},
            )
        )
    if not health.protocol_version:
        errors.append(_error("missing_protocol_version", "MinerU health response is missing protocol_version"))
    elif health.protocol_version != str(expected_protocol_version):
        errors.append(
            _error(
                "unsupported_protocol_version",
                f"MinerU protocol_version must be {expected_protocol_version}",
                {"observed": health.protocol_version, "expected": str(expected_protocol_version)},
            )
        )
    return _dedupe_errors(errors)


def _torch_diagnostics() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        available = bool(torch.cuda.is_available())
        capability = ""
        device_name = ""
        if available:
            device_name = str(torch.cuda.get_device_name(0))
            capability_tuple = torch.cuda.get_device_capability(0)
            capability = ".".join(str(part) for part in capability_tuple)
        return {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda or ""),
            "torch_cuda_available": available,
            "torch_cuda_device_name": device_name,
            "torch_cuda_device_capability": capability,
        }
    except Exception as exc:
        return {"error": exc.__class__.__name__}


def _nvidia_smi_diagnostics() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return {"available": False}
    if completed.returncode != 0:
        return {"available": False}
    raw = completed.stdout.strip()
    return {
        "available": True,
        "driver_version": _regex_group(raw, r"Driver Version:\s*([0-9.]+)"),
        "reported_cuda_version": _regex_group(raw, r"CUDA Version:\s*([0-9.]+)"),
        "raw": raw[:1000],
    }


def _persist_report(report: MinerUEndpointPreflightReport, run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = write_json(run_root / "preflight_report.json", report.model_dump(mode="json"))
    summary_path = run_root / "preflight_summary.md"
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    report.outputs["preflight_report"] = _rel(report_path, run_root)
    report.outputs["preflight_summary"] = _rel(summary_path, run_root)
    write_json(report_path, report.model_dump(mode="json"))


def _summary_markdown(report: MinerUEndpointPreflightReport) -> str:
    lines = [
        f"# MinerU Endpoint Preflight: {report.run_id}",
        "",
        f"- decision: {report.decision}",
        f"- endpoint profile: {report.profile.endpoint_profile_name}",
        f"- redacted API origin: {report.profile.redacted_api_origin}",
        f"- health status: {report.health.status or 'unknown'}",
        f"- protocol version: {report.health.protocol_version or 'unknown'}",
        f"- MinerU version: {report.health.mineru_version or 'unknown'}",
        f"- torch CUDA: {report.environment.torch_cuda_version or 'unknown'}",
        f"- torch cuda available: {report.environment.torch_cuda_available}",
        "",
        "This preflight records endpoint and environment diagnostics only. It does not parse PDFs, train models, or change routing.",
    ]
    if report.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in report.warnings)
    if report.errors:
        lines.extend(["", "## Errors"])
        lines.extend(f"- {error['code']}: {error['message']}" for error in report.errors)
    return "\n".join(lines) + "\n"


def _failed_report(
    *,
    run_id: str,
    generated_at: str,
    errors: list[dict[str, Any]],
    environment: MinerUEndpointEnvironmentDiagnostics | None = None,
) -> MinerUEndpointPreflightReport:
    return MinerUEndpointPreflightReport(
        run_id=run_id,
        generated_at=generated_at,
        decision="failed",
        profile=MinerUEndpointProfileReportSummary(),
        health=MinerUEndpointHealthSummary(ok=False),
        environment=environment or MinerUEndpointEnvironmentDiagnostics(),
        errors=errors,
    )


def _health_url(api_url: str, health_path: str) -> str:
    parsed = urlparse(str(api_url or "").strip())
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return str(httpx.URL(origin.rstrip("/") + "/" + str(health_path).lstrip("/")))


def _conda_runtime_precedes_system_cuda(entries: list[str]) -> bool:
    conda_indexes = [idx for idx, item in enumerate(entries) if "miniconda" in item or "conda" in item]
    system_cuda_indexes = [idx for idx, item in enumerate(entries) if item.startswith("/usr/local/cuda")]
    if not conda_indexes or not system_cuda_indexes:
        return False
    return min(conda_indexes) < min(system_cuda_indexes)


def _safe_run_id(value: str) -> bool:
    clean = str(value or "").strip()
    return bool(clean) and clean not in {".", ".."} and "/" not in clean and "\\" not in clean and Path(clean).name == clean


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "code": str(code or "error").strip(),
        "message": _safe_error(str(message or "").strip()),
        "details": _redact_details(details or {}),
    }


def _safe_error(message: str) -> str:
    safe = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted]", message, flags=re.IGNORECASE)
    safe = re.sub(r"(token|secret|authorization|password)[A-Za-z0-9._~+/=-]*", "[redacted]", safe, flags=re.IGNORECASE)
    return safe


def _redact_details(details: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in details.items():
        lowered = str(key).lower()
        if any(token in lowered for token in ["token", "authorization", "secret", "password"]):
            redacted[str(key)] = "[redacted]"
        else:
            redacted[str(key)] = value
    return redacted


def _dedupe_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for error in errors:
        code = str(error.get("code") or "")
        if code in seen:
            continue
        seen.add(code)
        deduped.append(error)
    return deduped


def _regex_group(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.expanduser().resolve().relative_to(root.expanduser().resolve()))
    except Exception:
        return str(path)


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    transport: httpx.BaseTransport | None = None,
) -> int:
    output = stdout or sys.stdout
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    token = os.environ.get("MINERU_API_TOKEN") or os.environ.get("AI4S_MINERU_API_TOKEN") or ""
    report = run_mineru_endpoint_preflight(
        output_dir=args.output,
        run_id=args.run_id,
        profile_config=args.profile_config,
        profile_name=args.profile_name,
        policy_name=args.policy_name,
        api_url=args.api_url,
        endpoint_kind=str(args.endpoint_kind).replace("-", "_") if args.endpoint_kind else None,
        expected_protocol_version=args.expected_protocol_version,
        health_path=args.health_path,
        http_timeout_sec=args.http_timeout_sec,
        api_token=token,
        transport=transport,
    )
    output.write(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, sort_keys=True))
    output.write("\n")
    return 0 if report.decision == "passed" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.mineru_endpoint_preflight",
        description="Manual MinerU endpoint preflight health and environment diagnostics.",
    )
    parser.add_argument("--profile-config")
    parser.add_argument("--profile-name")
    parser.add_argument("--policy-name")
    parser.add_argument("--api-url")
    parser.add_argument("--endpoint-kind", choices=["mineru-api", "mineru-router"])
    parser.add_argument("--expected-protocol-version")
    parser.add_argument("--health-path")
    parser.add_argument("--http-timeout-sec", type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
