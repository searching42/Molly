"""Canonical privacy policy and correlation builder for Harness telemetry."""

from __future__ import annotations

import re
from typing import Any, Mapping

from ai4s_agent.schemas import (
    AgentRunInspection,
    HarnessTelemetryCorrelationContext,
)


TELEMETRY_PRIVACY_POLICY_VERSION = "harness_telemetry_privacy_policy.v1"
MAX_TELEMETRY_ATTRIBUTES = 48
MAX_TELEMETRY_ATTRIBUTE_LENGTH = 256

_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SAFE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_FIXED_REASON_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


def build_harness_telemetry_correlation(
    *,
    operation: str,
    component: str,
    phase: str,
    inspection: AgentRunInspection | None = None,
    **bindings: Any,
) -> HarnessTelemetryCorrelationContext:
    """Build one vendor-neutral context from authority or inspection bindings."""

    payload: dict[str, Any] = {
        "operation": operation,
        "component": component,
        "phase": phase,
        **bindings,
    }
    if inspection is not None:
        defaults: dict[str, Any] = {
            "project_id": inspection.project_id,
            "run_id": inspection.run_id,
            "inspection_id": inspection.inspection_id,
            "inspection_digest": inspection.inspection_digest,
            "proposal_id": inspection.plan.proposal.object_id,
            "proposal_digest": inspection.plan.proposal.object_digest,
            "semantic_plan_id": inspection.plan.semantic_plan.object_id,
            "semantic_plan_digest": inspection.plan.semantic_plan.object_digest,
            "permission_decision_id": (
                inspection.plan.permission_decision.object_id
                if inspection.plan.permission_decision
                else ""
            ),
            "authorization_id": (
                inspection.plan.authorization.object_id
                if inspection.plan.authorization
                else ""
            ),
            "start_intent_id": (
                inspection.plan.start_intent.object_id
                if inspection.plan.start_intent
                else ""
            ),
            "controller_execution_id": (
                inspection.controller.execution.object_id
                if inspection.controller
                else ""
            ),
            "controller_execution_digest": (
                inspection.controller.execution.object_digest
                if inspection.controller
                else ""
            ),
            "controller_revision": (
                inspection.controller.controller_revision
                if inspection.controller
                else None
            ),
        }
        for key, verified_value in defaults.items():
            supplied_value = payload.get(key)
            if (
                supplied_value not in {None, ""}
                and verified_value not in {None, ""}
                and supplied_value != verified_value
            ):
                raise ValueError(
                    f"telemetry {key} conflicts with the verified inspection"
                )
            if verified_value not in {None, ""}:
                payload[key] = verified_value
    return HarnessTelemetryCorrelationContext.model_validate(payload)


def privacy_safe_telemetry_attributes(
    correlation: HarnessTelemetryCorrelationContext,
) -> dict[str, str | int | bool]:
    """Return the only correlation attributes vendors may receive."""

    attributes = correlation.telemetry_attributes()
    if len(attributes) > MAX_TELEMETRY_ATTRIBUTES:
        raise ValueError("telemetry correlation exceeds the bounded allowlist")
    for key, value in attributes.items():
        if not key.startswith("molly."):
            raise ValueError("telemetry attribute namespace is invalid")
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            if value < 0 or value > 2**63 - 1:
                raise ValueError("telemetry integer is invalid")
            continue
        if not isinstance(value, str) or len(value) > MAX_TELEMETRY_ATTRIBUTE_LENGTH:
            raise ValueError("telemetry string is invalid")
        if key.endswith("_digest"):
            if _SAFE_DIGEST.fullmatch(value) is None:
                raise ValueError("telemetry digest is invalid")
        elif _SAFE_LABEL.fullmatch(value) is None:
            raise ValueError("telemetry label is invalid")
    return attributes


def safe_reason_code(value: object, *, fallback: str) -> str:
    candidate = str(value or "")
    return candidate if _FIXED_REASON_CODE.fullmatch(candidate) else fallback


def safe_exception_type_code(exc_type: type[BaseException] | None) -> str:
    if exc_type is None:
        return "NO_EXCEPTION"
    name = re.sub(r"[^A-Za-z0-9]", "_", exc_type.__name__).upper()
    candidate = f"EXCEPTION_{name}"[:128]
    return safe_reason_code(candidate, fallback="EXCEPTION_UNKNOWN")


def merge_telemetry_attributes(
    *mappings: Mapping[str, str | int | bool],
) -> dict[str, str | int | bool]:
    merged: dict[str, str | int | bool] = {}
    for mapping in mappings:
        merged.update(mapping)
    if len(merged) > MAX_TELEMETRY_ATTRIBUTES:
        raise ValueError("telemetry attributes exceed the bounded allowlist")
    return merged


__all__ = [
    "MAX_TELEMETRY_ATTRIBUTES",
    "TELEMETRY_PRIVACY_POLICY_VERSION",
    "build_harness_telemetry_correlation",
    "merge_telemetry_attributes",
    "privacy_safe_telemetry_attributes",
    "safe_exception_type_code",
    "safe_reason_code",
]
