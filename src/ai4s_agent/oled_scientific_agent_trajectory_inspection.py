"""Ephemeral read-only inspection over exact-replayed observer publications."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ai4s_agent.oled_scientific_agent_trajectory_failure_attribution import (
    _BoundFailureAttribution,
    _FINDING_CODE_SET,
    _TAXONOMY_BY_ID,
)
from ai4s_agent.oled_scientific_agent_trajectory_projection import _EVENT_ORDER


INSPECTION_VERSION = "scientific_agent_trajectory_inspection.v1"
DEFAULT_LIMIT = 200
MAX_LIMIT = 500

_FILTER_KEYS = frozenset(
    {
        "event_kind",
        "taxonomy_family",
        "finding_code",
        "attribution_role",
        "attribution_status",
        "source_artifact",
        "limit",
    }
)
_ATTRIBUTION_ROLES = frozenset({"first_cause", "downstream_symptom"})
_ATTRIBUTION_STATUSES = frozenset({"determined", "undetermined", "no_failure"})
_OBSERVER_ARTIFACTS = frozenset(
    {
        "events.jsonl",
        "source_bindings.json",
        "telemetry_findings.jsonl",
        "trajectory.json",
        "audit_findings.jsonl",
        "audit_manifest.json",
        "audit_metrics.json",
        "source_binding.json",
        "attribution_manifest.json",
        "failure_attributions.jsonl",
        "failure_taxonomy.json",
        "report.md",
    }
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_OUTCOME_FIELDS = frozenset(
    {
        "status",
        "current_step",
        "approved",
        "gate",
        "task_id",
        "has_complete_top_n",
        "stop_reason",
        "selected_candidate_count",
    }
)


class InspectionRequestError(ValueError):
    """A fixed-contract query value is invalid."""


class InspectionLimitError(InspectionRequestError):
    """The requested bounded response exceeds the frozen maximum."""


@dataclass(frozen=True)
class InspectionFilters:
    event_kind: str | None = None
    taxonomy_family: str | None = None
    finding_code: str | None = None
    attribution_role: str | None = None
    attribution_status: str | None = None
    source_artifact: str | None = None
    limit: int = DEFAULT_LIMIT

    def public_payload(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in (
                ("event_kind", self.event_kind),
                ("taxonomy_family", self.taxonomy_family),
                ("finding_code", self.finding_code),
                ("attribution_role", self.attribution_role),
                ("attribution_status", self.attribution_status),
                ("source_artifact", self.source_artifact),
                ("limit", self.limit),
            )
            if value is not None
        }


def parse_inspection_filters(values: Mapping[str, str]) -> InspectionFilters:
    unknown = set(values) - _FILTER_KEYS
    if unknown:
        raise InspectionRequestError("unknown inspection filter")

    event_kind = _optional(values.get("event_kind"))
    taxonomy_family = _optional(values.get("taxonomy_family"))
    finding_code = _optional(values.get("finding_code"))
    attribution_role = _optional(values.get("attribution_role"))
    attribution_status = _optional(values.get("attribution_status"))
    source_artifact = _optional(values.get("source_artifact"))

    _require_allowed(event_kind, frozenset(_EVENT_ORDER), "event_kind")
    _require_allowed(taxonomy_family, frozenset(_TAXONOMY_BY_ID), "taxonomy_family")
    _require_allowed(finding_code, _FINDING_CODE_SET, "finding_code")
    _require_allowed(attribution_role, _ATTRIBUTION_ROLES, "attribution_role")
    _require_allowed(attribution_status, _ATTRIBUTION_STATUSES, "attribution_status")
    _require_allowed(source_artifact, _OBSERVER_ARTIFACTS, "source_artifact")

    raw_limit = _optional(values.get("limit"))
    if raw_limit is None:
        limit = DEFAULT_LIMIT
    else:
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise InspectionRequestError("limit must be an integer") from exc
        if str(limit) != raw_limit or limit < 1:
            raise InspectionRequestError("limit is outside the allowed range")
        if limit > MAX_LIMIT:
            raise InspectionLimitError("limit exceeds the maximum response size")
    return InspectionFilters(
        event_kind=event_kind,
        taxonomy_family=taxonomy_family,
        finding_code=finding_code,
        attribution_role=attribution_role,
        attribution_status=attribution_status,
        source_artifact=source_artifact,
        limit=limit,
    )


def build_oled_scientific_agent_trajectory_inspection(
    *,
    project_id: str,
    session_id: str,
    bound: _BoundFailureAttribution,
    filters: InspectionFilters | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe response without opening a source path or writing state."""

    selected = filters or InspectionFilters()
    trajectory = _object(bound.trajectory_payloads["trajectory.json"], "trajectory")
    audit_manifest = _object(bound.audit_payloads["audit_manifest.json"], "audit manifest")
    attribution_manifest = _object(
        bound.attribution_payloads["attribution_manifest.json"],
        "attribution manifest",
    )
    attribution_binding = _object(
        bound.attribution_payloads["source_binding.json"],
        "attribution source binding",
    )
    events = _jsonl(bound.trajectory_payloads["events.jsonl"], "events")
    audit_findings = _jsonl(
        bound.audit_payloads["audit_findings.jsonl"], "audit findings"
    )
    attributions = _jsonl(
        bound.attribution_payloads["failure_attributions.jsonl"],
        "failure attributions",
    )
    telemetry = _jsonl(
        bound.trajectory_payloads["telemetry_findings.jsonl"],
        "telemetry findings",
    )

    _require_chain(
        session_id=session_id,
        trajectory=trajectory,
        audit_manifest=audit_manifest,
        attribution_manifest=attribution_manifest,
        attribution_binding=attribution_binding,
        bound=bound,
    )
    timeline, unattached = _join_timeline(
        events=events,
        audit_findings=audit_findings,
        attributions=attributions,
        telemetry=telemetry,
        events_sha256=_digest(bound.trajectory_payloads["events.jsonl"]),
    )
    matching_timeline = [item for item in timeline if _timeline_matches(item, selected)]
    matching_unattached = [
        item for item in unattached if _finding_matches(item, selected)
    ]
    combined: list[tuple[str, dict[str, Any]]] = [
        ("timeline", item) for item in matching_timeline
    ] + [("unattached", item) for item in matching_unattached]
    returned = combined[: selected.limit]
    returned_timeline = [item for kind, item in returned if kind == "timeline"]
    returned_unattached = [item for kind, item in returned if kind == "unattached"]

    attribution_result = attribution_manifest.get("result")
    if not isinstance(attribution_result, dict):
        raise ValueError("inspection attribution result is invalid")
    counts = attribution_manifest.get("counts")
    if not isinstance(counts, dict):
        raise ValueError("inspection attribution counts are invalid")
    primary_id = _safe_optional_id(attribution_result.get("primary_first_cause_id"))
    ambiguity = _safe_optional_id(attribution_result.get("ambiguity_reason"))

    return {
        "ok": True,
        "inspection_version": INSPECTION_VERSION,
        "project_id": _safe_required_id(project_id),
        "session_id": _safe_required_id(session_id),
        "verified_chain": {
            "trajectory_id": _safe_required_id(trajectory.get("trajectory_id")),
            "trajectory_publication_id": _safe_required_id(
                trajectory.get("publication_id")
            ),
            "audit_id": _safe_required_id(audit_manifest.get("audit_id")),
            "audit_publication_id": _safe_required_id(
                audit_manifest.get("publication_id")
            ),
            "attribution_id": _safe_required_id(
                attribution_manifest.get("attribution_id")
            ),
            "attribution_publication_id": _safe_required_id(
                attribution_manifest.get("publication_id")
            ),
            "exact_replay": True,
            "observer_only": True,
        },
        "summary": {
            "terminal_status": _safe_scalar(trajectory.get("terminal_status")),
            "terminal_revision": _nonnegative_int(trajectory.get("terminal_revision")),
            "event_count": len(events),
            "audit_finding_count": len(audit_findings),
            "attribution_count": len(attributions),
            "first_cause_count": _nonnegative_int(counts.get("first_cause_count")),
            "downstream_symptom_count": _nonnegative_int(
                counts.get("downstream_symptom_count")
            ),
            "attribution_status": _safe_required_id(
                attribution_result.get("attribution_status")
            ),
            "primary_first_cause_id": primary_id,
            "ambiguity_reason": ambiguity,
            "total_timeline_count": len(timeline),
            "total_unattached_finding_count": len(unattached),
        },
        "applied_filters": selected.public_payload(),
        "page": {
            "limit": selected.limit,
            "truncated": len(combined) > selected.limit,
            "returned_count": len(returned),
            "total_matching_count": len(combined),
        },
        "timeline": returned_timeline,
        "unattached_findings": returned_unattached,
        "alternatives": {
            "available": False,
            "items": [],
            "reason": "source_observer_publications_do_not_persist_alternatives",
        },
        "claims": {
            "read_only": True,
            "observer_publications_only": True,
            "scientific_execution_modified": False,
            "control_action_available": False,
            "scientific_validation_claimed": False,
            "private_chain_of_thought_included": False,
        },
    }


def _join_timeline(
    *,
    events: Sequence[dict[str, Any]],
    audit_findings: Sequence[dict[str, Any]],
    attributions: Sequence[dict[str, Any]],
    telemetry: Sequence[dict[str, Any]],
    events_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    timeline: list[dict[str, Any]] = []
    by_event: dict[str, dict[str, Any]] = {}
    by_action: dict[str, dict[str, Any]] = {}
    for expected_index, event in enumerate(events):
        sequence = _nonnegative_int(event.get("sequence_index"))
        if sequence != expected_index:
            raise ValueError("inspection event order is not canonical")
        event_id = _safe_required_id(event.get("event_id"))
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        logical_role = _safe_optional_id(source.get("logical_role"))
        source_id = _safe_optional_id(source.get("source_artifact_id"))
        action_id = source_id if logical_role == "action_request" else None
        evidence = {
            "artifact_name": "events.jsonl",
            "sha256": events_sha256,
            "record_id": event_id,
            "record_digest": _digest(_canonical(event)),
            "source_binding_sha256": _digest(_canonical(source)),
            "logical_role": logical_role,
        }
        item = {
            "sequence_index": sequence,
            "session_revision": _nonnegative_int(event.get("session_revision")),
            "event_id": event_id,
            "event_kind": _safe_required_id(event.get("event_kind")),
            "task_id": _safe_optional_id(event.get("task_id")),
            "action_id": action_id,
            "child_run_id": _safe_optional_id(event.get("child_run_id")),
            "outcome": _safe_outcome(event.get("outcome")),
            "reason_codes": _safe_reason_codes(event.get("reason_codes")),
            "source_references": [evidence],
            "audit_findings": [],
            "failure_attributions": [],
            "telemetry_findings": [],
            "authority": "authoritative_observer_projection",
        }
        timeline.append(item)
        by_event[event_id] = item
        if action_id is not None:
            by_action[action_id] = item

    unattached: list[dict[str, Any]] = []
    for finding in audit_findings:
        safe = _safe_audit_finding(finding)
        target = _target_by_refs(finding.get("source_refs"), by_event)
        if target is None:
            unattached.append({**safe, "finding_layer": "audit"})
        else:
            target["audit_findings"].append(safe)
    for finding in attributions:
        safe = _safe_attribution(finding)
        affected = finding.get("affected") if isinstance(finding.get("affected"), dict) else {}
        target = by_event.get(str(affected.get("event_id") or ""))
        if target is None:
            target = _target_by_refs(finding.get("source_refs"), by_event)
        if target is None:
            unattached.append({**safe, "finding_layer": "attribution"})
        else:
            target["failure_attributions"].append(safe)
    for finding in telemetry:
        safe = _safe_telemetry_finding(finding)
        target = by_action.get(str(finding.get("action_id") or ""))
        if target is None:
            unattached.append({**safe, "finding_layer": "telemetry"})
        else:
            target["telemetry_findings"].append(safe)
    for item in timeline:
        item["audit_findings"].sort(key=lambda value: value["finding_id"])
        item["failure_attributions"].sort(key=lambda value: value["finding_id"])
        item["telemetry_findings"].sort(key=lambda value: value["finding_id"])
    unattached.sort(key=lambda value: (value["finding_layer"], value["finding_id"]))
    return timeline, unattached


def _safe_audit_finding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": _safe_required_id(value.get("finding_id")),
        "reason_code": _safe_required_id(value.get("reason_code")),
        "source_references": _safe_refs(value.get("source_refs")),
        "authority": "audit_metrics_only",
    }


def _safe_attribution(value: Mapping[str, Any]) -> dict[str, Any]:
    family = _safe_required_id(value.get("taxonomy_family"))
    code = _safe_required_id(value.get("finding_code"))
    role = _safe_required_id(value.get("attribution_role"))
    status = _safe_required_id(value.get("attribution_status"))
    if family not in _TAXONOMY_BY_ID or code not in _FINDING_CODE_SET:
        raise ValueError("inspection attribution enum is invalid")
    if role not in _ATTRIBUTION_ROLES or status not in {"determined", "undetermined"}:
        raise ValueError("inspection attribution role or status is invalid")
    return {
        "finding_id": _safe_required_id(value.get("finding_id")),
        "taxonomy_family": family,
        "attribution_role": role,
        "finding_code": code,
        "attribution_status": status,
        "evidence_sufficiency": _safe_required_id(value.get("evidence_sufficiency")),
        "deterministic_reason_code": _safe_required_id(
            value.get("deterministic_reason_code")
        ),
        "source_references": _safe_refs(value.get("source_refs")),
        "authority": "source_backed_failure_attribution",
    }


def _safe_telemetry_finding(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "finding_id": _safe_required_id(value.get("finding_id")),
        "reason_code": _safe_required_id(value.get("reason_code")),
        "action_id": _safe_required_id(value.get("action_id")),
        "source_references": [],
        "authority": "non_authoritative_telemetry",
    }


def _safe_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("inspection source references are invalid")
    result: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise ValueError("inspection source reference is invalid")
        artifact = str(raw.get("artifact_name") or "")
        sha256 = str(raw.get("sha256") or "")
        if artifact not in _OBSERVER_ARTIFACTS or not _DIGEST.fullmatch(sha256):
            raise ValueError("inspection source reference binding is invalid")
        ref: dict[str, Any] = {"artifact_name": artifact, "sha256": sha256}
        for key in ("record_id", "logical_role"):
            clean = _safe_optional_id(raw.get(key))
            if clean is not None:
                ref[key] = clean
        for key in ("record_digest", "source_binding_sha256"):
            clean = str(raw.get(key) or "")
            if clean:
                if not _DIGEST.fullmatch(clean):
                    raise ValueError("inspection source digest is invalid")
                ref[key] = clean
        result.append(ref)
    return sorted(result, key=lambda item: (item["artifact_name"], item.get("record_id", "")))


def _target_by_refs(value: Any, by_event: Mapping[str, dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(value, list):
        return None
    candidates = {
        str(ref.get("record_id"))
        for ref in value
        if isinstance(ref, dict) and str(ref.get("record_id") or "") in by_event
    }
    if len(candidates) != 1:
        return None
    return by_event[next(iter(candidates))]


def _timeline_matches(item: Mapping[str, Any], selected: InspectionFilters) -> bool:
    if selected.event_kind is not None and item.get("event_kind") != selected.event_kind:
        return False
    findings = [
        *item.get("audit_findings", []),
        *item.get("failure_attributions", []),
        *item.get("telemetry_findings", []),
    ]
    has_finding_filter = any(
        value is not None
        for value in (
            selected.taxonomy_family,
            selected.finding_code,
            selected.attribution_role,
            selected.attribution_status,
        )
    )
    if selected.attribution_status == "no_failure":
        return not item.get("failure_attributions")
    if has_finding_filter and not any(_finding_matches(value, selected) for value in findings):
        return False
    if selected.source_artifact is not None:
        refs = [*item.get("source_references", [])]
        refs.extend(ref for finding in findings for ref in finding.get("source_references", []))
        if not any(ref.get("artifact_name") == selected.source_artifact for ref in refs):
            return False
    return True


def _finding_matches(item: Mapping[str, Any], selected: InspectionFilters) -> bool:
    checks = (
        ("taxonomy_family", selected.taxonomy_family),
        ("finding_code", selected.finding_code),
        ("attribution_role", selected.attribution_role),
        ("attribution_status", selected.attribution_status),
    )
    for key, expected in checks:
        if expected is not None and item.get(key) != expected:
            return False
    if selected.source_artifact is not None and not any(
        ref.get("artifact_name") == selected.source_artifact
        for ref in item.get("source_references", [])
    ):
        return False
    return True


def _require_chain(
    *,
    session_id: str,
    trajectory: Mapping[str, Any],
    audit_manifest: Mapping[str, Any],
    attribution_manifest: Mapping[str, Any],
    attribution_binding: Mapping[str, Any],
    bound: _BoundFailureAttribution,
) -> None:
    if trajectory.get("session_id") != session_id:
        raise ValueError("inspection observer publication Session mismatch")
    expected = {
        "trajectory_id": bound.result.source_trajectory_id,
        "audit_id": bound.result.source_audit_id,
        "attribution_id": bound.result.attribution_id,
        "attribution_publication_id": bound.result.publication_id,
    }
    observed = {
        "trajectory_id": trajectory.get("trajectory_id"),
        "audit_id": audit_manifest.get("audit_id"),
        "attribution_id": attribution_manifest.get("attribution_id"),
        "attribution_publication_id": attribution_manifest.get("publication_id"),
    }
    if observed != expected:
        raise ValueError("inspection observer publication identity mismatch")
    if audit_manifest.get("source_trajectory_id") != observed["trajectory_id"]:
        raise ValueError("inspection trajectory/audit chain mismatch")
    if attribution_manifest.get("source_trajectory_id") != observed["trajectory_id"]:
        raise ValueError("inspection trajectory/attribution chain mismatch")
    if attribution_manifest.get("source_audit_id") != observed["audit_id"]:
        raise ValueError("inspection audit/attribution chain mismatch")
    if attribution_binding.get("source_trajectory_publication_id") != trajectory.get(
        "publication_id"
    ):
        raise ValueError("inspection trajectory publication binding mismatch")
    if attribution_binding.get("source_audit_publication_id") != audit_manifest.get(
        "publication_id"
    ):
        raise ValueError("inspection audit publication binding mismatch")


def _safe_outcome(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        key: _safe_scalar(value[key])
        for key in sorted(_SAFE_OUTCOME_FIELDS & set(value))
        if _safe_scalar(value[key]) is not None
    }


def _safe_reason_codes(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted(
        {
            clean
            for raw in value
            if (clean := _safe_optional_id(raw)) is not None
        }
    )


def _safe_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == value and value not in {float("inf"), float("-inf")}:
        return value
    if isinstance(value, str) and _SAFE_ID.fullmatch(value):
        return value
    return None


def _safe_required_id(value: Any) -> str:
    clean = _safe_optional_id(value)
    if clean is None:
        raise ValueError("inspection stable identifier is invalid")
    return clean


def _safe_optional_id(value: Any) -> str | None:
    if value is None:
        return None
    clean = str(value)
    if not _SAFE_ID.fullmatch(clean):
        return None
    return clean


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("inspection numeric field is invalid")
    return value


def _object(payload: bytes, label: str) -> dict[str, Any]:
    value = _loads(payload, label)
    if not isinstance(value, dict):
        raise ValueError(f"inspection {label} must be an object")
    return value


def _jsonl(payload: bytes, label: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    rows: list[dict[str, Any]] = []
    for line in payload.splitlines():
        value = _loads(line, label)
        if not isinstance(value, dict):
            raise ValueError(f"inspection {label} row must be an object")
        rows.append(value)
    return rows


def _loads(payload: bytes, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"inspection {label} is invalid JSON") from exc


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _optional(value: Any) -> str | None:
    clean = str(value or "").strip()
    return clean or None


def _require_allowed(value: str | None, allowed: frozenset[str], field: str) -> None:
    if value is not None and value not in allowed:
        raise InspectionRequestError(f"{field} is not allowed")


__all__ = [
    "DEFAULT_LIMIT",
    "INSPECTION_VERSION",
    "InspectionFilters",
    "InspectionLimitError",
    "InspectionRequestError",
    "MAX_LIMIT",
    "build_oled_scientific_agent_trajectory_inspection",
    "parse_inspection_filters",
]
