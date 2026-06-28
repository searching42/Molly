from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ai4s_agent.document_parse_live_acceptance import _redact_details
from ai4s_agent.mineru_endpoint_profiles import MinerUEndpointProfileReportSummary
from ai4s_agent.mineru_endpoint_preflight import MinerUEndpointPreflightReport


_EXPECTED_PROTOCOL_VERSION = "2"
_CREDENTIAL_MARKERS = ("token", "secret", "authorization", "password", "bearer", "cookie", "x-api-key")


class PreflightBindingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preflight_report_path: str = ""
    preflight_run_id: str = ""
    preflight_decision: str = ""
    preflight_health_status: str = ""
    preflight_protocol_version: str = ""
    preflight_redacted_api_origin: str = ""
    preflight_endpoint_profile_name: str = ""
    preflight_routing_policy_name: str = ""
    preflight_artifact_sha256: str = ""
    require_preflight_match: bool = False
    matched: bool = False
    mismatches: list[str] = Field(default_factory=list)


def load_and_bind_preflight_report(
    *,
    preflight_report_path: str | Path,
    preflight_artifact_sha256: str = "",
    require_preflight_match: bool = False,
    expected_origin: str,
    endpoint_profile_summary: dict[str, Any] | MinerUEndpointProfileReportSummary | None = None,
    expected_protocol_version: str = _EXPECTED_PROTOCOL_VERSION,
    failure_message: str = "preflight report does not match this live acceptance endpoint",
) -> tuple[PreflightBindingSummary, list[str], list[dict[str, Any]]]:
    path = Path(preflight_report_path).expanduser()
    binding = PreflightBindingSummary(
        preflight_report_path=safe_path_label(path),
        preflight_artifact_sha256=safe_sha256(preflight_artifact_sha256),
        require_preflight_match=bool(require_preflight_match),
    )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = MinerUEndpointPreflightReport.model_validate(payload)
    except Exception:
        return binding, [], [_error("invalid_preflight_report", "preflight report is missing, unreadable, or invalid")]

    profile = report.profile
    health = report.health
    binding = binding.model_copy(
        update={
            "preflight_run_id": report.run_id,
            "preflight_decision": report.decision,
            "preflight_health_status": health.status,
            "preflight_protocol_version": health.protocol_version,
            "preflight_redacted_api_origin": profile.redacted_api_origin,
            "preflight_endpoint_profile_name": profile.endpoint_profile_name,
            "preflight_routing_policy_name": profile.routing_policy_name,
        }
    )
    expected_profile = _endpoint_profile_summary(endpoint_profile_summary)
    mismatches: list[str] = []
    if report.decision != "passed":
        mismatches.append("preflight_decision_not_passed")
    if str(health.status or "").strip().lower() not in {"healthy", "ok"}:
        mismatches.append("preflight_health_status_unhealthy")
    if str(health.protocol_version or "").strip() != str(expected_protocol_version):
        mismatches.append("preflight_protocol_version_mismatch")
    if str(profile.redacted_api_origin or "").strip() != str(expected_origin or "").strip():
        mismatches.append("redacted_api_origin_mismatch")
    if expected_profile.endpoint_profile_name and profile.endpoint_profile_name != expected_profile.endpoint_profile_name:
        mismatches.append("endpoint_profile_name_mismatch")
    if expected_profile.routing_policy_name and profile.routing_policy_name != expected_profile.routing_policy_name:
        mismatches.append("routing_policy_name_mismatch")

    binding = binding.model_copy(update={"matched": not mismatches, "mismatches": mismatches})
    if mismatches and require_preflight_match:
        return binding, [], [
            _error(
                "preflight_match_failed",
                failure_message,
                {"mismatches": mismatches},
            )
        ]
    return binding, [f"preflight_binding_warning:{item}" for item in mismatches], []


def safe_path_label(path: Path) -> str:
    name = path.name or "preflight_report.json"
    if contains_credential_marker(name):
        return "[redacted-preflight-report-path]"
    return name


def safe_sha256(value: str) -> str:
    clean = str(value or "").strip().lower()
    if not clean:
        return ""
    if re.fullmatch(r"(sha256:)?[0-9a-f]{64}", clean):
        return clean if clean.startswith("sha256:") else f"sha256:{clean}"
    return "[invalid-sha256-redacted]"


def contains_credential_marker(value: str) -> bool:
    lowered = str(value or "").lower()
    return any(marker in lowered for marker in _CREDENTIAL_MARKERS)


def _endpoint_profile_summary(
    value: dict[str, Any] | MinerUEndpointProfileReportSummary | None,
) -> MinerUEndpointProfileReportSummary:
    if value is None:
        return MinerUEndpointProfileReportSummary()
    if isinstance(value, MinerUEndpointProfileReportSummary):
        return value
    return MinerUEndpointProfileReportSummary.model_validate(value)


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "code": str(code or "error").strip(),
        "message": str(message or "").strip(),
        "details": _redact_details(details or {}),
    }
