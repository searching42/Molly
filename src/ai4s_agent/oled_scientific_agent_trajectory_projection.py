"""Post-hoc, observer-only projection of a terminal OLED discovery Session."""

from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai4s_agent.oled_bounded_discovery_session import (
    _SESSION_VERSION,
    _TERMINAL,
    _validate_external_state,
    _validate_state_child_structure,
    _validate_state_transition,
    _validated_spec,
    _validated_state_payload,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    _read_json_bytes as _read_action_json_bytes,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    _validated_request as _validated_action_request,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    _validated_state as _validated_action_state,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    validated_oled_bounded_project_id,
)
from ai4s_agent.oled_categorical_dataset_execution import (
    _publish_payload_directory,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _read_regular_file_bound,
)
from ai4s_agent.storage import ProjectStorage


_PROJECTION_VERSION = "scientific_agent_trajectory_projection.v1"
_PUBLICATION_VERSION = "scientific_agent_trajectory_projection_publication.v1"
_SOURCE_BINDING_VERSION = "scientific_agent_trajectory_source_bindings.v1"
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_STATE_REVISIONS = 4096
_MAX_ACTION_RECORDS = 4096
_MAX_PROJECTED_EVENTS = 32768
_MAX_SOURCE_BINDINGS = 16384
_EVENT_ORDER = {
    "state_committed": 10,
    "action_requested": 20,
    "action_authorized": 30,
    "task_dispatched": 40,
    "stage_completed": 50,
    "stage_failed": 60,
    "publication_verified": 70,
    "terminal_result_committed": 90,
}


@dataclass(frozen=True)
class OledScientificAgentTrajectoryProjection:
    trajectory_id: str
    publication_id: str
    output_dir: Path
    receipt_json: Path
    events_jsonl: Path
    source_bindings_json: Path
    telemetry_findings_jsonl: Path


@dataclass(frozen=True)
class _CapturedFile:
    payload: bytes
    sha256: str


def publish_oled_scientific_agent_trajectory_projection(
    *,
    storage: ProjectStorage,
    project_id: str,
    session_id: str,
    actions_root: Path,
    output_root: Path | None = None,
) -> OledScientificAgentTrajectoryProjection:
    """Publish a deterministic read-only projection of one terminal Session."""

    clean_project = validated_oled_bounded_project_id(project_id)
    project_dir = storage.project_dir(clean_project)
    session_dir = (
        project_dir / "bounded-discovery-sessions" / str(session_id or "")
    ).absolute()
    if (
        not session_dir.is_relative_to(project_dir)
        or not session_dir.is_dir()
        or session_dir.is_symlink()
    ):
        raise ValueError("PR-BD Session is unavailable")

    captures: dict[Path, _CapturedFile] = {}
    spec_capture = _capture_canonical_json(
        session_dir / "session_spec.json", captures=captures
    )
    spec = _validated_captured_spec(
        spec_capture.payload,
        session_dir=session_dir,
    )
    states = _read_immutable_states(session_dir, captures=captures)
    terminal_state = states[-1]
    if terminal_state["status"] not in _TERMINAL:
        raise ValueError("PR-BD only projects terminal Sessions")
    _validate_external_state(
        storage, clean_project, session_dir, spec, terminal_state
    )

    source_bindings: list[dict[str, Any]] = [
        {
            "logical_role": "session_spec",
            "source_artifact_id": "session_spec.json",
            "source_publication_id": None,
            "sha256": spec_capture.sha256,
            "manifest_sha256": None,
        }
    ]
    raw_events: list[dict[str, Any]] = []
    for state in states:
        revision = int(state["revision"])
        state_capture = captures[session_dir / f"state_{revision:06d}.json"]
        binding = {
            "logical_role": "session_revision",
            "source_artifact_id": f"state_{revision:06d}.json",
            "source_publication_id": None,
            "sha256": state_capture.sha256,
            "manifest_sha256": str(state["state_digest"]),
        }
        source_bindings.append(binding)
        raw_events.append(
            _event(
                kind="state_committed",
                revision=revision,
                child=None,
                source=binding,
                outcome={"status": state["status"], "current_step": state["current_step"]},
                reason_codes=[],
            )
        )

    action_events, action_bindings, telemetry_snapshot, telemetry_findings = (
        _project_actions(
            actions_root=actions_root,
            project_id=clean_project,
            session_id=str(session_id),
            terminal_revision=int(terminal_state["revision"]),
            captures=captures,
        )
    )
    raw_events.extend(action_events)
    source_bindings.extend(action_bindings)

    child_events, child_bindings = _project_children(
        storage=storage,
        project_id=clean_project,
        states=states,
        captures=captures,
    )
    raw_events.extend(child_events)
    source_bindings.extend(child_bindings)

    result_capture = _capture_canonical_json(
        session_dir / "session_result.json", captures=captures
    )
    result = _read_action_json_bytes(result_capture.payload)
    result_binding = {
        "logical_role": "terminal_result",
        "source_artifact_id": "session_result.json",
        "source_publication_id": str(result.get("result_id") or "") or None,
        "sha256": result_capture.sha256,
        "manifest_sha256": None,
    }
    source_bindings.append(result_binding)
    raw_events.append(
        _event(
            kind="terminal_result_committed",
            revision=int(terminal_state["revision"]),
            child=None,
            source=result_binding,
            outcome={
                "status": result.get("status"),
                "stop_reason": result.get("stop_reason"),
                "result_source": result.get("result_source"),
                "has_complete_top_n": result.get("has_complete_top_n"),
                "usage": result.get("usage"),
            },
            reason_codes=[str(result.get("stop_reason") or "")]
            if result.get("stop_reason")
            else [],
        )
    )
    if len(raw_events) > _MAX_PROJECTED_EVENTS:
        raise ValueError("PR-BD projected event roster exceeds the v1 limit")
    if len(source_bindings) > _MAX_SOURCE_BINDINGS:
        raise ValueError("PR-BD source roster exceeds the v1 limit")

    source_bindings = sorted(
        source_bindings,
        key=lambda item: (
            str(item["logical_role"]),
            str(item["source_artifact_id"]),
            str(item.get("source_publication_id") or ""),
        ),
    )
    _require_unique_source_bindings(source_bindings)
    source_manifest_digest = "sha256:" + _stable_hash(source_bindings)
    trajectory_identity = {
        "projection_version": _PROJECTION_VERSION,
        "session_id": str(session_id),
        "session_spec_sha256": spec_capture.sha256,
        "terminal_state_digest": terminal_state["state_digest"],
        "source_manifest_digest": source_manifest_digest,
    }
    trajectory_id = "scientific-agent-trajectory:" + _stable_hash(
        trajectory_identity
    )

    ordered_events = _ordered_events(raw_events, trajectory_id=trajectory_id)
    telemetry_snapshot_digest = "sha256:" + _stable_hash(telemetry_snapshot)
    publication_identity = {
        "publication_version": _PUBLICATION_VERSION,
        "trajectory_id": trajectory_id,
        "telemetry_snapshot_digest": telemetry_snapshot_digest,
    }
    publication_id = "scientific-agent-trajectory-publication:" + _stable_hash(
        publication_identity
    )
    source_payload = {
        "source_binding_version": _SOURCE_BINDING_VERSION,
        "trajectory_id": trajectory_id,
        "source_manifest_digest": source_manifest_digest,
        "sources": source_bindings,
    }
    events_bytes = _canonical_jsonl_bytes(ordered_events)
    source_bytes = _canonical_json_bytes(source_payload)
    findings_bytes = _canonical_jsonl_bytes(telemetry_findings)
    receipt = {
        "publication_version": _PUBLICATION_VERSION,
        "projection_version": _PROJECTION_VERSION,
        "publication_id": publication_id,
        "trajectory_id": trajectory_id,
        "session_id": str(session_id),
        "terminal_revision": int(terminal_state["revision"]),
        "terminal_status": terminal_state["status"],
        "source_manifest_digest": source_manifest_digest,
        "telemetry_snapshot_digest": telemetry_snapshot_digest,
        "counts": {
            "event_count": len(ordered_events),
            "source_count": len(source_bindings),
            "telemetry_finding_count": len(telemetry_findings),
        },
        "artifacts": {
            "events.jsonl": _sha256(events_bytes),
            "source_bindings.json": _sha256(source_bytes),
            "telemetry_findings.jsonl": _sha256(findings_bytes),
        },
        "claims": {
            "observer_only": True,
            "post_hoc_projection": True,
            "scientific_trust_anchor_created": False,
            "scientific_execution_modified": False,
            "private_chain_of_thought_recorded": False,
            "counterfactual_alternatives_invented": False,
            "mutable_telemetry_authoritative": False,
        },
    }
    receipt_bytes = _canonical_json_bytes(receipt)
    # Re-run the complete PR-AV external-state validation after every source
    # binding has been captured, then prove all captured named files still have
    # the same exact bytes before publication.
    _validate_external_state(
        storage, clean_project, session_dir, spec, terminal_state
    )
    _recheck_captures(captures)

    root = (
        output_root.absolute()
        if output_root is not None
        else (project_dir / "trajectory-projections").absolute()
    )
    root.mkdir(parents=True, exist_ok=True)
    output_dir = root / publication_id
    with _pinned_output_parents_without_symlink_components(root) as pinned:
        _publish_payload_directory(
            output_dir=output_dir,
            parent_descriptor=pinned[root],
            payloads={
                "events.jsonl": events_bytes,
                "source_bindings.json": source_bytes,
                "telemetry_findings.jsonl": findings_bytes,
                "trajectory.json": receipt_bytes,
            },
            artifact_label="scientific trajectory projection",
        )
    return OledScientificAgentTrajectoryProjection(
        trajectory_id=trajectory_id,
        publication_id=publication_id,
        output_dir=output_dir,
        receipt_json=output_dir / "trajectory.json",
        events_jsonl=output_dir / "events.jsonl",
        source_bindings_json=output_dir / "source_bindings.json",
        telemetry_findings_jsonl=output_dir / "telemetry_findings.jsonl",
    )


def _validated_captured_spec(
    payload: bytes, *, session_dir: Path
) -> dict[str, Any]:
    spec = _read_action_json_bytes(payload)
    if spec.get("session_version") != _SESSION_VERSION:
        raise ValueError("PR-BD Session spec version is invalid")
    request = {
        key: value
        for key, value in spec.items()
        if key not in {"session_version", "session_id", "input_bindings"}
    }
    normalized = _validated_spec(request)
    bindings = spec.get("input_bindings")
    if not isinstance(bindings, dict):
        raise ValueError("PR-BD Session input bindings are invalid")
    expected_id = "oled-bounded-session-" + _stable_hash(
        {**normalized, "input_bindings": bindings}
    )
    if spec.get("session_id") != expected_id or session_dir.name != expected_id:
        raise ValueError("PR-BD Session spec identity mismatch")
    return spec


def _read_immutable_states(
    session_dir: Path, *, captures: dict[Path, _CapturedFile]
) -> list[dict[str, Any]]:
    names = sorted(
        path.name
        for path in session_dir.glob("state_*.json")
        if path.name[6:-5].isdigit()
    )
    if not names or names != [f"state_{index:06d}.json" for index in range(len(names))]:
        raise ValueError("PR-BD immutable Session history is incomplete")
    if len(names) > _MAX_STATE_REVISIONS:
        raise ValueError("PR-BD Session history exceeds the v1 limit")
    states: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        captured = _capture_canonical_json(session_dir / name, captures=captures)
        state = _validated_state_payload(
            _read_action_json_bytes(captured.payload),
            session_dir=session_dir,
            expected_revision=index,
        )
        _validate_state_child_structure(state)
        if index == 0:
            if state.get("previous_state_digest") is not None:
                raise ValueError("PR-BD initial Session predecessor is invalid")
        else:
            previous = states[-1]
            if state.get("previous_state_digest") != previous["state_digest"]:
                raise ValueError("PR-BD immutable Session chain is invalid")
            _validate_state_transition(previous, state)
        states.append(state)
    return states


def _project_actions(
    *,
    actions_root: Path,
    project_id: str,
    session_id: str,
    terminal_revision: int,
    captures: dict[Path, _CapturedFile],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    project_root = (actions_root.absolute() / project_id).absolute()
    if not project_root.exists():
        return [], [], [], []
    if not project_root.is_dir() or project_root.is_symlink():
        raise ValueError("PR-BD action root is invalid")
    events: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    telemetry_snapshot: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    action_dirs = sorted(project_root.glob("oled-session-action-*"))
    if len(action_dirs) > _MAX_ACTION_RECORDS:
        raise ValueError("PR-BD action roster exceeds the v1 limit")
    for action_dir in action_dirs:
        if not action_dir.is_dir() or action_dir.is_symlink():
            raise ValueError("PR-BD action directory is invalid")
        request_path = action_dir / "request.json"
        request_capture = _capture_file(request_path, captures=captures)
        request = _validated_action_request(
            _read_action_json_bytes(request_capture.payload),
            project_id=project_id,
            action_id=action_dir.name,
        )
        if _json_bytes(request) != request_capture.payload:
            raise ValueError("PR-BD immutable action request is not canonical")
        if request["session_id"] != session_id:
            continue
        revision = int(request["expected_revision"])
        if revision > terminal_revision:
            raise ValueError("PR-BD action request exceeds terminal revision")
        binding = {
            "logical_role": "action_request",
            "source_artifact_id": str(request["action_id"]),
            "source_publication_id": None,
            "sha256": request_capture.sha256,
            "manifest_sha256": str(request["request_digest"]),
        }
        bindings.append(binding)
        events.append(
            _event(
                kind="action_requested",
                revision=revision,
                child=None,
                source=binding,
                outcome={"action": request["action"]},
                reason_codes=[],
            )
        )
        state_path = action_dir / "action.json"
        snapshot = {"action_id": request["action_id"], "sha256": None, "status": "missing"}
        try:
            state_capture = _capture_file(state_path, captures=captures)
            snapshot["sha256"] = state_capture.sha256
            state = _validated_action_state(
                _read_action_json_bytes(state_capture.payload),
                project_id=project_id,
                action_id=action_dir.name,
            )
            if _json_bytes(state) != state_capture.payload:
                raise ValueError("mutable action telemetry is not canonical")
            snapshot["status"] = state["status"]
            completed = state.get("completed_revision")
            inconsistent = (
                (completed is not None and int(completed) > terminal_revision)
                or (
                    state["status"] in {"QUEUED", "RUNNING"}
                    and revision < terminal_revision
                )
                or (
                    state["status"] in {"SUCCEEDED", "RECOVERED"}
                    and (completed is None or int(completed) <= revision)
                )
            )
            if inconsistent:
                findings.append(
                    _telemetry_finding(
                        action_id=str(request["action_id"]),
                        reason="telemetry_conflicts_with_session_history",
                        telemetry_sha256=state_capture.sha256,
                    )
                )
        except (OSError, ValueError):
            findings.append(
                _telemetry_finding(
                    action_id=str(request["action_id"]),
                    reason="telemetry_missing_or_invalid",
                    telemetry_sha256=snapshot["sha256"],
                )
            )
        telemetry_snapshot.append(snapshot)
    telemetry_snapshot.sort(key=lambda item: str(item["action_id"]))
    findings.sort(key=lambda item: (str(item["action_id"]), str(item["reason_code"])))
    return events, bindings, telemetry_snapshot, findings


def _project_children(
    *,
    storage: ProjectStorage,
    project_id: str,
    states: list[dict[str, Any]],
    captures: dict[Path, _CapturedFile],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    previous_children: dict[str, dict[str, Any]] = {}
    for state in states:
        revision = int(state["revision"])
        current_children = {
            str(child["label"]): child for child in state["children"]
        }
        for label, child in sorted(current_children.items()):
            previous = previous_children.get(label)
            if previous is None:
                stage_path = storage.run_dir(project_id, str(child["run_id"])) / "stage.json"
                stage_capture = _capture_file(stage_path, captures=captures)
                stage_binding = {
                    "logical_role": "child_stage",
                    "source_artifact_id": str(child["run_id"]),
                    "source_publication_id": None,
                    "sha256": stage_capture.sha256,
                    "manifest_sha256": None,
                }
                bindings.append(stage_binding)
                events.append(
                    _event(
                        kind="task_dispatched",
                        revision=revision,
                        child=child,
                        source=stage_binding,
                        outcome={"child_status": child["status"]},
                        reason_codes=[],
                    )
                )
            if child["status"] in {"succeeded", "failed", "integrity_failed"} and (
                previous is None or previous["status"] != child["status"]
            ):
                stage_path = storage.run_dir(project_id, str(child["run_id"])) / "stage.json"
                stage_capture = _capture_file(stage_path, captures=captures)
                stage_binding = {
                    "logical_role": "child_stage",
                    "source_artifact_id": str(child["run_id"]),
                    "source_publication_id": None,
                    "sha256": stage_capture.sha256,
                    "manifest_sha256": None,
                }
                if stage_binding not in bindings:
                    bindings.append(stage_binding)
                gate_snapshot = child.get("gate_snapshot")
                if gate_snapshot is not None:
                    decision_path = (
                        storage.run_dir(project_id, str(child["run_id"]))
                        / "gate_decisions.json"
                    )
                    decision_capture = _capture_file(
                        decision_path, captures=captures
                    )
                    decision_payload = json.loads(
                        decision_capture.payload.decode("utf-8"),
                        object_pairs_hook=_unique_object,
                    )
                    decision = _matching_gate_decision(
                        decision_payload,
                        gate_snapshot=gate_snapshot,
                    )
                    decision_binding = {
                        "logical_role": "gate_decision",
                        "source_artifact_id": str(child["run_id"]),
                        "source_publication_id": str(
                            gate_snapshot["snapshot_id"]
                        ),
                        "sha256": decision_capture.sha256,
                        "manifest_sha256": str(
                            gate_snapshot["snapshot_hash"]
                        ),
                    }
                    bindings.append(decision_binding)
                    events.append(
                        _event(
                            kind="action_authorized",
                            revision=revision,
                            child=child,
                            source=decision_binding,
                            outcome={
                                "gate": decision.get("gate"),
                                "approved": True,
                                "snapshot_id": gate_snapshot["snapshot_id"],
                            },
                            reason_codes=["exact_gate_snapshot_approved"],
                        )
                    )
                events.append(
                    _event(
                        kind="stage_completed"
                        if child["status"] == "succeeded"
                        else "stage_failed",
                        revision=revision,
                        child=child,
                        source=stage_binding,
                        outcome={"child_status": child["status"]},
                        reason_codes=[str(child["status"])],
                    )
                )
                if child["status"] == "succeeded":
                    registry_path = (
                        storage.run_dir(project_id, str(child["run_id"]))
                        / "artifact_registry.json"
                    )
                    registry_capture = _capture_file(
                        registry_path, captures=captures
                    )
                    registry = json.loads(
                        registry_capture.payload.decode("utf-8"),
                        object_pairs_hook=_unique_object,
                    )
                    publication_binding = {
                        "logical_role": "child_publication",
                        "source_artifact_id": str(child["run_id"]),
                        "source_publication_id": _publication_id_from_registry(registry),
                        "sha256": registry_capture.sha256,
                        "manifest_sha256": child["artifact_manifest_sha256"],
                    }
                    bindings.append(publication_binding)
                    events.append(
                        _event(
                            kind="publication_verified",
                            revision=revision,
                            child=child,
                            source=publication_binding,
                            outcome={"verification": "exact_replay_passed"},
                            reason_codes=["external_anchor_exact_replay"],
                        )
                    )
        previous_children = current_children
    return events, bindings


def _matching_gate_decision(
    payload: Any, *, gate_snapshot: Any
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(gate_snapshot, dict):
        raise ValueError("PR-BD gate authorization source is invalid")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("PR-BD gate decision roster is invalid")
    matches = [
        item
        for item in decisions
        if isinstance(item, dict)
        and item.get("approved") is True
        and item.get("approved_snapshot_id") == gate_snapshot.get("snapshot_id")
        and item.get("approved_snapshot_hash") == gate_snapshot.get("snapshot_hash")
    ]
    if len(matches) != 1:
        raise ValueError("PR-BD exact gate authorization is invalid")
    return matches[0]


def _event(
    *,
    kind: str,
    revision: int,
    child: dict[str, Any] | None,
    source: dict[str, Any],
    outcome: dict[str, Any],
    reason_codes: list[str],
) -> dict[str, Any]:
    if kind not in _EVENT_ORDER:
        raise ValueError("PR-BD event kind is unsupported")
    return {
        "event_version": _PROJECTION_VERSION,
        "event_kind": kind,
        "session_revision": revision,
        "child_run_id": str(child["run_id"]) if child else None,
        "task_id": str(child["task_id"]) if child else None,
        "source": dict(source),
        "outcome": outcome,
        "reason_codes": sorted(reason_codes),
    }


def _ordered_events(
    events: list[dict[str, Any]], *, trajectory_id: str
) -> list[dict[str, Any]]:
    ordered = sorted(
        events,
        key=lambda item: (
            int(item["session_revision"]),
            _EVENT_ORDER[str(item["event_kind"])],
            str(item.get("child_run_id") or ""),
            str(item["source"]["source_artifact_id"]),
        ),
    )
    result: list[dict[str, Any]] = []
    for index, event in enumerate(ordered):
        identity = {**event, "trajectory_id": trajectory_id}
        result.append(
            {
                **event,
                "trajectory_id": trajectory_id,
                "sequence_index": index,
                "event_id": "scientific-agent-trajectory-event:"
                + _stable_hash(identity),
            }
        )
    return result


def _telemetry_finding(
    *, action_id: str, reason: str, telemetry_sha256: str | None
) -> dict[str, Any]:
    payload = {
        "finding_version": _PROJECTION_VERSION,
        "action_id": action_id,
        "reason_code": reason,
        "telemetry_sha256": telemetry_sha256,
        "authority_effect": "ignored_for_scientific_facts",
    }
    return {**payload, "finding_id": "trajectory-telemetry-finding:" + _stable_hash(payload)}


def _publication_id_from_registry(registry: Any) -> str | None:
    if not isinstance(registry, dict):
        raise ValueError("PR-BD Artifact Registry is invalid")
    artifact_paths = registry.get("artifacts", registry)
    if not isinstance(artifact_paths, dict):
        raise ValueError("PR-BD Artifact Registry roster is invalid")
    identities: set[str] = set()
    for relative in artifact_paths.values():
        if not isinstance(relative, str):
            raise ValueError("PR-BD Artifact Registry path is invalid")
        for component in Path(relative).parts:
            if component.startswith("oled-") and ":" in component:
                identities.add(component)
    return sorted(identities)[0] if len(identities) == 1 else None


def _require_unique_source_bindings(bindings: list[dict[str, Any]]) -> None:
    identities = [
        (item["logical_role"], item["source_artifact_id"]) for item in bindings
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("PR-BD source bindings are not unique")


def _capture_canonical_json(
    path: Path, *, captures: dict[Path, _CapturedFile]
) -> _CapturedFile:
    captured = _capture_file(path, captures=captures)
    value = json.loads(captured.payload.decode("utf-8"), object_pairs_hook=_unique_object)
    # Persisted scientific artifacts use the repository canonical JSON
    # contract, which permits finite floats.  PR-BD's stricter serializer is
    # reserved for the new projection artifacts themselves.
    if not isinstance(value, dict) or _json_bytes(value) != captured.payload:
        raise ValueError("PR-BD source JSON is not canonical")
    return captured


def _capture_file(
    path: Path, *, captures: dict[Path, _CapturedFile]
) -> _CapturedFile:
    absolute = path.absolute()
    payload, sha256 = _read_regular_file_bound(
        absolute,
        max_bytes=_MAX_JSON_BYTES,
        reject_symlink_components=True,
        allow_empty=True,
    )
    captured = _CapturedFile(payload=payload, sha256=sha256)
    existing = captures.get(absolute)
    if existing is not None and existing != captured:
        raise ValueError("PR-BD source changed during projection")
    captures[absolute] = captured
    return captured


def _recheck_captures(captures: dict[Path, _CapturedFile]) -> None:
    for path, expected in captures.items():
        payload, sha256 = _read_regular_file_bound(
            path,
            max_bytes=max(_MAX_JSON_BYTES, len(expected.payload)),
            reject_symlink_components=True,
            allow_empty=True,
        )
        if payload != expected.payload or sha256 != expected.sha256:
            raise ValueError("PR-BD source changed before publication")


def _canonical_json_bytes(value: Any) -> bytes:
    normalized = _canonical_value(value)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_jsonl_bytes(values: list[dict[str, Any]]) -> bytes:
    rows = [
        json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for value in values
    ]
    return (("\n".join(rows) + "\n") if rows else "").encode("utf-8")


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("PR-BD canonical JSON rejects non-finite numbers")
        raise ValueError("PR-BD canonical JSON does not permit floats")
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("PR-BD canonical JSON keys must be strings")
            clean = unicodedata.normalize("NFC", key)
            if clean in normalized:
                raise ValueError("PR-BD canonical JSON key normalization collided")
            normalized[clean] = _canonical_value(item)
        return normalized
    raise ValueError("PR-BD canonical JSON contains an unsupported value")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("PR-BD source JSON contains duplicate keys")
        result[key] = value
    return result


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "OledScientificAgentTrajectoryProjection",
    "publish_oled_scientific_agent_trajectory_projection",
]
