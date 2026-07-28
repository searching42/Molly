"""Deterministic observer-only metrics over PR-BE verified trajectory bytes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ai4s_agent.oled_bounded_discovery_session_view import (
    validated_oled_bounded_project_id,
)
from ai4s_agent.oled_categorical_dataset_execution import (
    _publish_payload_directory,
)
from ai4s_agent.oled_real_phase1_execution import _stable_hash
from ai4s_agent.oled_scientific_agent_trajectory_projection import (
    _ReadOnlyProjectStorage,
    _canonical_json_bytes,
    _canonical_jsonl_bytes,
    _lexical_absolute,
    _reject_output_source_overlap,
    _require_existing_directory,
    _sha256,
    _unique_object,
)
from ai4s_agent.oled_scientific_agent_trajectory_verifier import (
    _BoundTrajectoryProjection,
    _verified_oled_scientific_agent_trajectory_projection,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.storage import ProjectStorage


_AUDIT_METRICS_VERSION = "scientific_agent_trajectory_audit_metrics.v1"
_AUDIT_FINDING_VERSION = "scientific_agent_trajectory_audit_finding.v1"
_AUDIT_SOURCE_BINDING_VERSION = (
    "scientific_agent_trajectory_audit_source_binding.v1"
)
_AUDIT_PUBLICATION_VERSION = (
    "scientific_agent_trajectory_audit_publication.v1"
)
_SOURCE_PUBLICATION_NAMES = {
    "events.jsonl",
    "source_bindings.json",
    "telemetry_findings.jsonl",
    "trajectory.json",
}
_AUDIT_PUBLICATION_NAMES = {
    "audit_findings.jsonl",
    "audit_manifest.json",
    "audit_metrics.json",
    "report.md",
    "source_binding.json",
}
_COVERAGE_EVENT_KINDS = {
    "action": {
        "action_requested",
        "task_dispatched",
        "stage_completed",
        "stage_failed",
    },
    "evidence": {"publication_verified"},
    "authorization": {"action_authorized"},
}
_SUCCESS_TERMINAL_STATUSES = {
    "COMPLETED_TOP_N",
    "STOPPED_BOUNDED_NO_SOLUTION",
}
_BOUNDED_STOP_REASONS = {
    "non_supply_policy_prevented_complete_top_n",
    "max_iterations_reached",
    "max_generation_rounds_reached",
    "max_generated_candidates_would_be_exceeded",
}
_BUDGET_FIELDS = (
    "iterations",
    "generation_rounds",
    "generated_candidates",
)


@dataclass(frozen=True)
class OledScientificAgentTrajectoryAuditMetricsPublication:
    audit_id: str
    publication_id: str
    output_dir: Path
    audit_metrics_json: Path
    audit_findings_jsonl: Path
    source_binding_json: Path
    audit_manifest_json: Path
    report_md: Path


@dataclass(frozen=True)
class _PreparedAuditPublication:
    audit_id: str
    publication_id: str
    payloads: Mapping[str, bytes]


def publish_oled_scientific_agent_trajectory_audit_metrics(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    trajectory_publication_dir: Path,
    output_root: Path | None = None,
) -> OledScientificAgentTrajectoryAuditMetricsPublication:
    """Publish v1 metrics from the bytes held by PR-BE's pinned seam.

    No audit output path is created until the bound verifier context has
    completed its post-consumer stability check.
    """

    clean_project = validated_oled_bounded_project_id(project_id)
    with _verified_oled_scientific_agent_trajectory_projection(
        storage=storage,
        project_id=clean_project,
        session_id=session_id,
        actions_root=actions_root,
        publication_dir=trajectory_publication_dir,
    ) as bound:
        prepared = _prepare_audit_publication(bound)

    read_only_storage = _ReadOnlyProjectStorage(storage)
    project_dir = read_only_storage.project_dir(clean_project)
    session_dir = _require_existing_directory(
        _lexical_absolute(
            project_dir / "bounded-discovery-sessions" / str(session_id or "")
        ),
        "PR-BF Session",
    )
    runs_root = _require_existing_directory(
        _lexical_absolute(project_dir / "runs"),
        "PR-BF runs root",
    )
    root = (
        _lexical_absolute(output_root)
        if output_root is not None
        else _lexical_absolute(project_dir / "trajectory-audits")
    )
    _reject_output_source_overlap(
        root=root,
        session_dir=session_dir,
        actions_project_root=_lexical_absolute(actions_root / clean_project),
        child_run_dirs=[
            runs_root,
            _lexical_absolute(trajectory_publication_dir),
        ],
    )
    output_dir = root / prepared.publication_id
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads=dict(prepared.payloads),
            artifact_label="scientific trajectory audit metrics",
        )
    return OledScientificAgentTrajectoryAuditMetricsPublication(
        audit_id=prepared.audit_id,
        publication_id=prepared.publication_id,
        output_dir=output_dir,
        audit_metrics_json=output_dir / "audit_metrics.json",
        audit_findings_jsonl=output_dir / "audit_findings.jsonl",
        source_binding_json=output_dir / "source_binding.json",
        audit_manifest_json=output_dir / "audit_manifest.json",
        report_md=output_dir / "report.md",
    )


def _prepare_audit_publication(
    bound: _BoundTrajectoryProjection,
) -> _PreparedAuditPublication:
    return _prepare_audit_publication_from_verified_bytes(
        payloads=bound.payloads,
        verified_trajectory_id=bound.result.trajectory_id,
        verified_publication_id=bound.result.publication_id,
    )


def _prepare_audit_publication_from_verified_bytes(
    *,
    payloads: Mapping[str, bytes],
    verified_trajectory_id: str,
    verified_publication_id: str,
) -> _PreparedAuditPublication:
    """Compute an audit from already verified bytes; perform no path reads."""

    if set(payloads) != _SOURCE_PUBLICATION_NAMES:
        raise ValueError("PR-BF verified trajectory byte roster is invalid")
    source_artifacts = {
        name: _sha256(payloads[name]) for name in sorted(_SOURCE_PUBLICATION_NAMES)
    }
    audit_identity = {
        "audit_metrics_version": _AUDIT_METRICS_VERSION,
        "source_trajectory_id": verified_trajectory_id,
        "source_publication_id": verified_publication_id,
        "source_artifacts": source_artifacts,
    }
    audit_id = "scientific-agent-trajectory-audit:" + _stable_hash(audit_identity)

    receipt = _json_object(payloads["trajectory.json"], "trajectory receipt")
    source_payload = _json_object(
        payloads["source_bindings.json"], "trajectory source bindings"
    )
    events = _jsonl_objects(payloads["events.jsonl"], "trajectory events")
    telemetry_findings = _jsonl_objects(
        payloads["telemetry_findings.jsonl"], "trajectory telemetry findings"
    )
    sources = source_payload.get("sources")
    if not isinstance(sources, list) or not all(
        isinstance(item, dict) for item in sources
    ):
        raise ValueError("PR-BF trajectory source binding roster is invalid")

    findings = _integrity_findings(
        audit_id=audit_id,
        receipt=receipt,
        source_payload=source_payload,
        events=events,
        sources=sources,
        telemetry_findings=telemetry_findings,
        verified_trajectory_id=verified_trajectory_id,
        verified_publication_id=verified_publication_id,
        source_artifacts=source_artifacts,
    )
    metrics = _metrics_payload(
        audit_id=audit_id,
        receipt=receipt,
        events=events,
        sources=sources,
        verified_trajectory_id=verified_trajectory_id,
        verified_publication_id=verified_publication_id,
        finding_count=len(findings),
    )
    source_binding = {
        "source_binding_version": _AUDIT_SOURCE_BINDING_VERSION,
        "audit_id": audit_id,
        "source_trajectory_id": verified_trajectory_id,
        "source_publication_id": verified_publication_id,
        "verification": {
            "context_bound_verified_bytes": True,
            "exact_external_replay": True,
            "exact_file_roster": True,
            "exact_bytes": True,
        },
        "source_artifacts": source_artifacts,
    }
    metrics_bytes = _canonical_json_bytes(metrics)
    findings_bytes = _canonical_jsonl_bytes(findings)
    source_binding_bytes = _canonical_json_bytes(source_binding)
    report_bytes = _report_bytes(
        audit_id=audit_id,
        metrics=metrics,
        findings=findings,
    )
    artifact_digests = {
        "audit_findings.jsonl": _sha256(findings_bytes),
        "audit_metrics.json": _sha256(metrics_bytes),
        "report.md": _sha256(report_bytes),
        "source_binding.json": _sha256(source_binding_bytes),
    }
    publication_identity = {
        "publication_version": _AUDIT_PUBLICATION_VERSION,
        "audit_id": audit_id,
        "artifacts": artifact_digests,
    }
    publication_id = (
        "scientific-agent-trajectory-audit-publication:"
        + _stable_hash(publication_identity)
    )
    manifest = {
        "publication_version": _AUDIT_PUBLICATION_VERSION,
        "audit_metrics_version": _AUDIT_METRICS_VERSION,
        "publication_id": publication_id,
        "audit_id": audit_id,
        "source_trajectory_id": verified_trajectory_id,
        "source_publication_id": verified_publication_id,
        "counts": {
            "finding_count": len(findings),
            "metric_group_count": len(metrics["metrics"]),
        },
        "artifacts": artifact_digests,
        "claims": {
            "observer_only": True,
            "context_bound_verified_bytes_consumed": True,
            "scientific_execution_modified": False,
            "trajectory_projection_modified": False,
            "session_or_control_plane_modified": False,
            "scientific_trust_anchor_created": False,
            "root_cause_inferred": False,
            "counterfactual_alternatives_invented": False,
        },
    }
    result_payloads = {
        "audit_findings.jsonl": findings_bytes,
        "audit_manifest.json": _canonical_json_bytes(manifest),
        "audit_metrics.json": metrics_bytes,
        "report.md": report_bytes,
        "source_binding.json": source_binding_bytes,
    }
    if set(result_payloads) != _AUDIT_PUBLICATION_NAMES:
        raise AssertionError("PR-BF audit publication roster is incomplete")
    return _PreparedAuditPublication(
        audit_id=audit_id,
        publication_id=publication_id,
        payloads=result_payloads,
    )


def _integrity_findings(
    *,
    audit_id: str,
    receipt: dict[str, Any],
    source_payload: dict[str, Any],
    events: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    telemetry_findings: list[dict[str, Any]],
    verified_trajectory_id: str,
    verified_publication_id: str,
    source_artifacts: dict[str, str],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    def add(
        code: str,
        *,
        artifacts: tuple[str, ...],
        details: dict[str, Any],
        record_id: str | None = None,
    ) -> None:
        refs = [
            {
                "artifact_name": name,
                "sha256": source_artifacts[name],
                **({"record_id": record_id} if record_id else {}),
            }
            for name in sorted(set(artifacts))
        ]
        body = {
            "finding_version": _AUDIT_FINDING_VERSION,
            "audit_id": audit_id,
            "reason_code": code,
            "source_refs": refs,
            "details": details,
            "authority_effect": "audit_metrics_only",
            "root_cause_claimed": False,
        }
        findings.append(
            {
                **body,
                "finding_id": "scientific-agent-trajectory-audit-finding:"
                + _stable_hash(body),
            }
        )

    if receipt.get("trajectory_id") != verified_trajectory_id:
        add(
            "verified_trajectory_identity_mismatch",
            artifacts=("trajectory.json",),
            details={
                "expected": verified_trajectory_id,
                "observed": _string_or_none(receipt.get("trajectory_id")),
            },
        )
    if receipt.get("publication_id") != verified_publication_id:
        add(
            "verified_publication_identity_mismatch",
            artifacts=("trajectory.json",),
            details={
                "expected": verified_publication_id,
                "observed": _string_or_none(receipt.get("publication_id")),
            },
        )
    if source_payload.get("trajectory_id") != verified_trajectory_id:
        add(
            "source_binding_trajectory_identity_mismatch",
            artifacts=("source_bindings.json",),
            details={
                "expected": verified_trajectory_id,
                "observed": _string_or_none(source_payload.get("trajectory_id")),
            },
        )

    declared_artifacts = receipt.get("artifacts")
    if not isinstance(declared_artifacts, dict):
        add(
            "source_artifact_manifest_missing",
            artifacts=("trajectory.json",),
            details={"expected_artifact_count": 3},
        )
    else:
        for name in (
            "events.jsonl",
            "source_bindings.json",
            "telemetry_findings.jsonl",
        ):
            if declared_artifacts.get(name) != source_artifacts[name]:
                add(
                    "source_artifact_digest_mismatch",
                    artifacts=("trajectory.json", name),
                    details={"artifact_name": name},
                )

    counts = receipt.get("counts")
    expected_counts = {
        "event_count": len(events),
        "source_count": len(sources),
        "telemetry_finding_count": len(telemetry_findings),
    }
    for key, observed in expected_counts.items():
        declared = counts.get(key) if isinstance(counts, dict) else None
        if declared != observed:
            add(
                key.replace("_count", "") + "_count_mismatch",
                artifacts=(
                    "trajectory.json",
                    {
                        "event_count": "events.jsonl",
                        "source_count": "source_bindings.json",
                        "telemetry_finding_count": "telemetry_findings.jsonl",
                    }[key],
                ),
                details={"declared_count": declared, "observed_count": observed},
            )

    declared_manifest = source_payload.get("source_manifest_digest")
    computed_manifest = "sha256:" + _stable_hash(sources)
    if declared_manifest != computed_manifest or receipt.get(
        "source_manifest_digest"
    ) != computed_manifest:
        add(
            "source_manifest_digest_mismatch",
            artifacts=("trajectory.json", "source_bindings.json"),
            details={"computed_source_manifest_digest": computed_manifest},
        )

    binding_keys = [_binding_key(item) for item in sources]
    duplicate_bindings = sorted(
        key for key, count in Counter(binding_keys).items() if count > 1
    )
    for key in duplicate_bindings:
        add(
            "duplicate_source_binding",
            artifacts=("source_bindings.json",),
            details={"binding_identity": key},
        )
    binding_set = set(binding_keys)

    event_ids = [
        str(event.get("event_id"))
        for event in events
        if isinstance(event.get("event_id"), str)
    ]
    for event_id, count in sorted(Counter(event_ids).items()):
        if count > 1:
            add(
                "duplicate_event_id",
                artifacts=("events.jsonl",),
                details={"occurrence_count": count},
                record_id=event_id,
            )
    for index, event in enumerate(events):
        event_id = _string_or_none(event.get("event_id"))
        if event.get("sequence_index") != index:
            add(
                "event_sequence_mismatch",
                artifacts=("events.jsonl",),
                details={
                    "expected_sequence_index": index,
                    "observed_sequence_index": event.get("sequence_index"),
                },
                record_id=event_id,
            )
        if event.get("trajectory_id") != verified_trajectory_id:
            add(
                "event_trajectory_identity_mismatch",
                artifacts=("events.jsonl",),
                details={"sequence_index": index},
                record_id=event_id,
            )
        source = event.get("source")
        if not isinstance(source, dict) or _binding_key(source) not in binding_set:
            add(
                "event_source_binding_missing",
                artifacts=("events.jsonl", "source_bindings.json"),
                details={
                    "event_kind": _string_or_none(event.get("event_kind")),
                    "sequence_index": index,
                },
                record_id=event_id,
            )

    terminal_status = _string_or_none(receipt.get("terminal_status"))
    terminal_events = _events_of_kind(events, "terminal_result_committed")
    terminal_anchors = _terminal_anchors(events, terminal_status=terminal_status)
    if len(terminal_anchors) != 1:
        add(
            "terminal_anchor_count_mismatch",
            artifacts=("trajectory.json", "events.jsonl"),
            details={"observed_count": len(terminal_anchors)},
        )
    for event in terminal_events:
        outcome = event.get("outcome")
        observed_status = outcome.get("status") if isinstance(outcome, dict) else None
        if observed_status != terminal_status:
            add(
                "terminal_status_mismatch",
                artifacts=("trajectory.json", "events.jsonl"),
                details={
                    "receipt_status": terminal_status,
                    "event_status": _string_or_none(observed_status),
                },
                record_id=_string_or_none(event.get("event_id")),
            )

    findings.sort(
        key=lambda item: (
            str(item["reason_code"]),
            str(item["finding_id"]),
        )
    )
    return findings


def _metrics_payload(
    *,
    audit_id: str,
    receipt: dict[str, Any],
    events: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    verified_trajectory_id: str,
    verified_publication_id: str,
    finding_count: int,
) -> dict[str, Any]:
    binding_set = {_binding_key(item) for item in sources}
    terminal_status = _string_or_none(receipt.get("terminal_status"))
    state_events = _events_of_kind(events, "state_committed")
    recovery_events = [
        event
        for event in state_events
        if _outcome_value(event, "status") == "RECOVERY_REQUIRED"
    ]
    coverage_events: dict[str, list[dict[str, Any]]] = {
        name: [
            event
            for event in events
            if event.get("event_kind") in eligible_kinds
        ]
        for name, eligible_kinds in _COVERAGE_EVENT_KINDS.items()
    }
    coverage_events["observation_to_decision"] = [
        event
        for event in state_events
        if _nonnegative_int(event.get("session_revision")) not in {None, 0}
    ]
    coverage_events["recovery"] = recovery_events
    coverage_events["terminal"] = _terminal_anchors(
        events, terminal_status=terminal_status
    )
    coverage = {
        name: _coverage(events_for_category, binding_set=binding_set)
        for name, events_for_category in coverage_events.items()
    }

    dispatched = _events_of_kind(events, "task_dispatched")
    completed = _events_of_kind(events, "stage_completed")
    failed = _events_of_kind(events, "stage_failed")
    dispatched_ids = [
        value
        for event in dispatched
        if (value := _string_or_none(event.get("child_run_id"))) is not None
    ]
    terminal_child_status: dict[str, str] = {}
    for event in [*completed, *failed]:
        child_id = _string_or_none(event.get("child_run_id"))
        child_status = _string_or_none(_outcome_value(event, "child_status"))
        if child_id is not None and child_status is not None:
            terminal_child_status[child_id] = child_status
    dispatch_counts = Counter(dispatched_ids)
    failed_ids = sorted(
        child_id
        for child_id, status in terminal_child_status.items()
        if status in {"failed", "integrity_failed"}
    )
    reason_codes = sorted(
        {
            str(code)
            for event in failed
            for code in event.get("reason_codes", [])
            if isinstance(code, str)
        }
    )

    terminal_events = _events_of_kind(events, "terminal_result_committed")
    terminal_outcome = (
        terminal_events[0].get("outcome")
        if len(terminal_events) == 1
        and isinstance(terminal_events[0].get("outcome"), dict)
        else None
    )
    budget = _budget_metric(terminal_outcome)
    top_n = _top_n_metric(
        terminal_status=terminal_status,
        terminal_outcome=terminal_outcome,
    )
    correct_stop = _correct_stop_metric(
        terminal_status=terminal_status,
        terminal_outcome=terminal_outcome,
    )
    revisions = sorted(
        {
            revision
            for event in events
            if (revision := _nonnegative_int(event.get("session_revision")))
            is not None
        }
    )
    metrics = {
        "provenance_coverage": coverage,
        "trajectory_length": {
            "event_count": len(events),
            "session_revision_count": len(revisions),
            "first_session_revision": revisions[0] if revisions else None,
            "last_session_revision": revisions[-1] if revisions else None,
        },
        "action_outcome": {
            "action_request_count": len(_events_of_kind(events, "action_requested")),
            "task_dispatch_count": len(dispatched),
            "succeeded_child_count": sum(
                status == "succeeded" for status in terminal_child_status.values()
            ),
            "failed_child_count": sum(
                status == "failed" for status in terminal_child_status.values()
            ),
            "integrity_failed_child_count": sum(
                status == "integrity_failed"
                for status in terminal_child_status.values()
            ),
            "unresolved_child_count": len(
                set(dispatched_ids) - set(terminal_child_status)
            ),
        },
        "tool_failure": {
            "stage_failed_event_count": len(failed),
            "child_run_ids": failed_ids,
            "reason_codes": reason_codes,
        },
        "retry": {
            "dispatch_attempt_count": len(dispatched_ids),
            "duplicate_dispatch_count": sum(
                count - 1 for count in dispatch_counts.values() if count > 1
            ),
            "duplicate_child_run_ids": sorted(
                child_id for child_id, count in dispatch_counts.items() if count > 1
            ),
        },
        "reconciliation": {
            "explicit_recovery_required_state_count": len(recovery_events),
            "session_revisions": sorted(
                revision
                for event in recovery_events
                if (
                    revision := _nonnegative_int(event.get("session_revision"))
                )
                is not None
            ),
            "inferred_reconciliation_count": 0,
        },
        "gate": _gate_metric(events),
        "latency": {
            "status": "unavailable",
            "value_milliseconds": None,
            "reason_code": "projection_v1_has_no_wall_clock_event_fields",
        },
        "budget_consumption": budget,
        "wasted_computation": {
            "status": "not_derivable",
            "value": None,
            "failed_terminal_child_count": len(failed_ids),
            "reason_code": "projection_v1_has_no_cost_or_reuse_evidence",
        },
        "top_n_completion": top_n,
        "bounded_search_correct_stop": correct_stop,
    }
    return {
        "audit_metrics_version": _AUDIT_METRICS_VERSION,
        "audit_id": audit_id,
        "source_trajectory_id": verified_trajectory_id,
        "source_publication_id": verified_publication_id,
        "finding_count": finding_count,
        "metrics": metrics,
    }


def _coverage(
    events: list[dict[str, Any]], *, binding_set: set[str]
) -> dict[str, Any]:
    eligible = len(events)
    covered = sum(
        isinstance(event.get("source"), dict)
        and _binding_key(event["source"]) in binding_set
        for event in events
    )
    if eligible == 0:
        status = "not_applicable"
        basis_points = None
    elif covered == eligible:
        status = "complete"
        basis_points = 10_000
    elif covered == 0:
        status = "none"
        basis_points = 0
    else:
        status = "partial"
        basis_points = covered * 10_000 // eligible
    return {
        "eligible_event_count": eligible,
        "source_bound_event_count": covered,
        "coverage_basis_points": basis_points,
        "status": status,
    }


def _gate_metric(events: list[dict[str, Any]]) -> dict[str, int]:
    authorizations = _events_of_kind(events, "action_authorized")
    approved = sum(_outcome_value(event, "approved") is True for event in authorizations)
    rejected = sum(_outcome_value(event, "approved") is False for event in authorizations)
    return {
        "authorization_event_count": len(authorizations),
        "approved_count": approved,
        "rejected_count": rejected,
        "unknown_status_count": len(authorizations) - approved - rejected,
    }


def _budget_metric(terminal_outcome: Any) -> dict[str, Any]:
    usage = terminal_outcome.get("usage") if isinstance(terminal_outcome, dict) else None
    if not isinstance(usage, dict):
        return {
            "status": "unavailable",
            "reason_code": "terminal_usage_not_available",
            **{field: None for field in _BUDGET_FIELDS},
        }
    values = {field: _nonnegative_int(usage.get(field)) for field in _BUDGET_FIELDS}
    if any(value is None for value in values.values()):
        return {
            "status": "unavailable",
            "reason_code": "terminal_usage_incomplete",
            **values,
        }
    return {"status": "available", "reason_code": None, **values}


def _top_n_metric(
    *, terminal_status: str | None, terminal_outcome: Any
) -> dict[str, Any]:
    if terminal_status not in _SUCCESS_TERMINAL_STATUSES:
        return {"status": "not_applicable", "has_complete_top_n": None}
    value = (
        terminal_outcome.get("has_complete_top_n")
        if isinstance(terminal_outcome, dict)
        else None
    )
    return {
        "status": "available" if isinstance(value, bool) else "unavailable",
        "has_complete_top_n": value if isinstance(value, bool) else None,
    }


def _correct_stop_metric(
    *, terminal_status: str | None, terminal_outcome: Any
) -> dict[str, Any]:
    if terminal_status not in _SUCCESS_TERMINAL_STATUSES:
        return {
            "status": "not_applicable",
            "correct_stop": None,
            "terminal_status": terminal_status,
            "stop_reason": None,
        }
    if not isinstance(terminal_outcome, dict):
        return {
            "status": "unavailable",
            "correct_stop": None,
            "terminal_status": terminal_status,
            "stop_reason": None,
        }
    stop_reason = _string_or_none(terminal_outcome.get("stop_reason"))
    complete = terminal_outcome.get("has_complete_top_n")
    if not isinstance(complete, bool) or stop_reason is None:
        return {
            "status": "unavailable",
            "correct_stop": None,
            "terminal_status": terminal_status,
            "stop_reason": stop_reason,
        }
    correct = (
        terminal_status == "COMPLETED_TOP_N"
        and complete
        and stop_reason == "target_top_n_complete"
    ) or (
        terminal_status == "STOPPED_BOUNDED_NO_SOLUTION"
        and not complete
        and stop_reason in _BOUNDED_STOP_REASONS
    )
    return {
        "status": "available",
        "correct_stop": correct,
        "terminal_status": terminal_status,
        "stop_reason": stop_reason,
    }


def _terminal_anchors(
    events: list[dict[str, Any]], *, terminal_status: str | None
) -> list[dict[str, Any]]:
    terminal_events = _events_of_kind(events, "terminal_result_committed")
    if terminal_events:
        return terminal_events
    if terminal_status in _SUCCESS_TERMINAL_STATUSES:
        return []
    matching_states = [
        event
        for event in _events_of_kind(events, "state_committed")
        if _outcome_value(event, "status") == terminal_status
    ]
    return matching_states[-1:] if matching_states else []


def _events_of_kind(
    events: list[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_kind") == kind]


def _outcome_value(event: dict[str, Any], key: str) -> Any:
    outcome = event.get("outcome")
    return outcome.get(key) if isinstance(outcome, dict) else None


def _binding_key(binding: dict[str, Any]) -> str:
    return _canonical_json_bytes(binding).decode("utf-8").rstrip("\n")


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"PR-BF {label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"PR-BF {label} must be an object")
    return value


def _jsonl_objects(payload: bytes, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        text = payload.decode("utf-8")
        for line in text.splitlines():
            if not line:
                raise ValueError("blank row")
            value = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(value, dict):
                raise ValueError("non-object row")
            result.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"PR-BF {label} is invalid JSONL") from exc
    return result


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _report_bytes(
    *,
    audit_id: str,
    metrics: dict[str, Any],
    findings: list[dict[str, Any]],
) -> bytes:
    source_trajectory_id = str(metrics["source_trajectory_id"])
    source_publication_id = str(metrics["source_publication_id"])
    coverage = metrics["metrics"]["provenance_coverage"]
    lines = [
        "# Scientific agent trajectory audit metrics",
        "",
        f"- Audit ID: `{audit_id}`",
        f"- Source trajectory: `{source_trajectory_id}`",
        f"- Source publication: `{source_publication_id}`",
        "- Input authority: PR-BE context-bound exact verified bytes",
        "- Authority effect: observer-only; not a scientific trust anchor",
        "",
        "## Provenance coverage",
        "",
        "| Category | Covered | Eligible | Basis points | Status |",
        "|---|---:|---:|---:|---|",
    ]
    for name in (
        "action",
        "evidence",
        "authorization",
        "observation_to_decision",
        "recovery",
        "terminal",
    ):
        item = coverage[name]
        basis_points = (
            "n/a"
            if item["coverage_basis_points"] is None
            else str(item["coverage_basis_points"])
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    str(item["source_bound_event_count"]),
                    str(item["eligible_event_count"]),
                    basis_points,
                    str(item["status"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Deterministic summary",
            "",
            f"- Trajectory events: `{metrics['metrics']['trajectory_length']['event_count']}`",
            f"- Gate authorizations: `{metrics['metrics']['gate']['authorization_event_count']}`",
            "- Stage failures: `"
            + str(metrics["metrics"]["tool_failure"]["stage_failed_event_count"])
            + "`",
            f"- Top-N status: `{metrics['metrics']['top_n_completion']['status']}`",
            "- Correct-stop status: `"
            + str(metrics["metrics"]["bounded_search_correct_stop"]["status"])
            + "`",
            f"- Audit findings: `{len(findings)}`",
            "",
            "Latency and wasted-computation values remain unavailable when the verified",
            "projection does not carry the facts needed to compute them. No source path is",
            "re-read and no failure cause or counterfactual alternative is inferred.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


__all__ = [
    "OledScientificAgentTrajectoryAuditMetricsPublication",
    "publish_oled_scientific_agent_trajectory_audit_metrics",
]
