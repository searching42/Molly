"""Post-hoc, observer-only projection of a terminal OLED discovery Session."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
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
from ai4s_agent.oled_scientific_agent_source_evidence import (
    validate_dispatch_authority,
    validate_dispatch_receipt,
    validate_failure_evidence,
    validate_recovery_receipt,
)
from ai4s_agent.schemas import StageState
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
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _CapturedDirectoryRoster:
    path: Path
    kind: str
    names: tuple[str, ...]
    existed: bool
    device: int | None
    inode: int | None


class _ReadOnlyProjectStorage(ProjectStorage):
    """Resolve existing project/run paths without creating filesystem state."""

    def __init__(self, source: ProjectStorage) -> None:
        # Deliberately do not call ProjectStorage.__init__: it creates
        # ``projects/``.  A projection must fail without changing the source
        # workspace when any authoritative path is absent.
        self.workspace_dir = source.workspace_dir
        self.projects_root = source.projects_root

    def project_dir(self, project_id: str) -> Path:
        path = _lexical_absolute(self.projects_root / project_id)
        if not path.is_relative_to(self.projects_root):
            raise ValueError("PR-BD project path escapes the workspace")
        return _require_existing_directory(path, "PR-BD project")

    def run_dir(self, project_id: str, run_id: str) -> Path:
        project = self.project_dir(project_id)
        runs_root = _require_existing_directory(
            _lexical_absolute(project / "runs"), "PR-BD runs root"
        )
        path = _lexical_absolute(runs_root / run_id)
        if not path.is_relative_to(runs_root):
            raise ValueError("PR-BD child run path escapes the runs root")
        return _require_existing_directory(path, "PR-BD child run")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_existing_directory(path: Path, label: str) -> Path:
    """Open every existing component with O_NOFOLLOW and create nothing."""

    absolute = _lexical_absolute(path)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("PR-BD read-only path resolution requires safe dirfd support")
    descriptor = -1
    try:
        descriptor = os.open(
            absolute.anchor,
            os.O_RDONLY | directory_flag | no_follow,
        )
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | directory_flag | no_follow,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        opened = os.fstat(descriptor)
        named = os.stat(absolute, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or opened.st_dev != named.st_dev
            or opened.st_ino != named.st_ino
        ):
            raise ValueError(f"{label} is not a stable existing directory")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"{label} is unavailable or contains a symlink") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)
    return absolute


def _reject_output_source_overlap(
    *,
    root: Path,
    session_dir: Path,
    actions_project_root: Path,
    child_run_dirs: list[Path],
) -> None:
    output = _lexical_absolute(root)
    sources = [
        _lexical_absolute(session_dir),
        _lexical_absolute(actions_project_root),
        *[_lexical_absolute(path) for path in child_run_dirs],
    ]
    if any(
        output == source
        or output.is_relative_to(source)
        or source.is_relative_to(output)
        for source in sources
    ):
        raise ValueError("PR-BD output root overlaps an authoritative source tree")


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
    read_only_storage = _ReadOnlyProjectStorage(storage)
    project_dir = read_only_storage.project_dir(clean_project)
    session_dir = _lexical_absolute(
        project_dir / "bounded-discovery-sessions" / str(session_id or "")
    )
    if (
        not session_dir.is_relative_to(project_dir)
        or not session_dir.is_dir()
        or session_dir.is_symlink()
    ):
        raise ValueError("PR-BD Session is unavailable")

    captures: dict[Path, _CapturedFile] = {}
    directory_rosters: list[_CapturedDirectoryRoster] = []
    spec_capture = _capture_canonical_json(
        session_dir / "session_spec.json", captures=captures
    )
    spec = _validated_captured_spec(
        spec_capture.payload,
        session_dir=session_dir,
    )
    states = _read_immutable_states(
        session_dir,
        captures=captures,
        directory_rosters=directory_rosters,
    )
    terminal_state = states[-1]
    if terminal_state["status"] not in _TERMINAL:
        raise ValueError("PR-BD only projects terminal Sessions")
    _validate_external_state(
        read_only_storage, clean_project, session_dir, spec, terminal_state
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

    (
        action_events,
        action_bindings,
        telemetry_snapshot,
        telemetry_findings,
        recovery_receipts,
    ) = (
        _project_actions(
            storage=read_only_storage,
            actions_root=actions_root,
            project_id=clean_project,
            session_id=str(session_id),
            terminal_revision=int(terminal_state["revision"]),
            states=states,
            captures=captures,
            directory_rosters=directory_rosters,
        )
    )
    raw_events.extend(action_events)
    source_bindings.extend(action_bindings)

    child_events, child_bindings = _project_children(
        storage=read_only_storage,
        project_id=clean_project,
        states=states,
        captures=captures,
        directory_rosters=directory_rosters,
        recovery_receipts=recovery_receipts,
    )
    raw_events.extend(child_events)
    source_bindings.extend(child_bindings)

    terminal_roster = _capture_directory_roster(
        session_dir,
        kind="session_terminal",
    )
    directory_rosters.append(terminal_roster)
    if terminal_state["status"] in {"COMPLETED_TOP_N", "STOPPED_BOUNDED_NO_SOLUTION"} and not terminal_roster.names:
        raise ValueError("PR-BD terminal result is unavailable")
    if terminal_roster.names:
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
        read_only_storage, clean_project, session_dir, spec, terminal_state
    )
    _recheck_captures(captures)
    _recheck_directory_rosters(directory_rosters)

    root = (
        _lexical_absolute(output_root)
        if output_root is not None
        else _lexical_absolute(project_dir / "trajectory-projections")
    )
    _reject_output_source_overlap(
        root=root,
        session_dir=session_dir,
        actions_project_root=_lexical_absolute(actions_root / clean_project),
        child_run_dirs=[
            read_only_storage.run_dir(clean_project, str(child["run_id"]))
            for child in terminal_state["children"]
        ],
    )
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
    session_dir: Path,
    *,
    captures: dict[Path, _CapturedFile],
    directory_rosters: list[_CapturedDirectoryRoster],
) -> list[dict[str, Any]]:
    roster = _capture_directory_roster(session_dir, kind="session_states")
    directory_rosters.append(roster)
    names = list(roster.names)
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
    storage: ProjectStorage,
    actions_root: Path,
    project_id: str,
    session_id: str,
    terminal_revision: int,
    states: list[dict[str, Any]],
    captures: dict[Path, _CapturedFile],
    directory_rosters: list[_CapturedDirectoryRoster],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    project_root = _lexical_absolute(actions_root / project_id)
    roster = _capture_directory_roster(
        project_root,
        kind="actions",
        allow_missing=True,
    )
    directory_rosters.append(roster)
    if not roster.existed:
        return [], [], [], [], {}
    if not project_root.is_dir() or project_root.is_symlink():
        raise ValueError("PR-BD action root is invalid")
    events: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    telemetry_snapshot: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    recovery_receipts: dict[str, dict[str, Any]] = {}
    action_dirs = [project_root / name for name in roster.names]
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
        receipt_root = action_dir / "recovery-receipts"
        receipt_roster = _capture_directory_roster(
            receipt_root,
            kind="recovery_receipts",
            allow_missing=True,
        )
        directory_rosters.append(receipt_roster)
        for receipt_name in receipt_roster.names:
            receipt_dir = receipt_root / receipt_name
            payload_roster = _capture_directory_roster(
                receipt_dir,
                kind="receipt_payloads",
            )
            directory_rosters.append(payload_roster)
            if payload_roster.names != ("receipt.json",):
                raise ValueError("PR-BD recovery receipt file roster is invalid")
            receipt_capture = _capture_canonical_json(
                receipt_dir / "receipt.json",
                captures=captures,
            )
            receipt_payload = validate_recovery_receipt(
                _read_action_json_bytes(receipt_capture.payload)
            )
            _validate_projected_recovery_receipt(
                storage=storage,
                project_id=project_id,
                request=request,
                receipt=receipt_payload,
                states=states,
                captures=captures,
                directory_rosters=directory_rosters,
            )
            receipt_id = str(receipt_payload["receipt_id"])
            if receipt_id in recovery_receipts:
                raise ValueError("PR-BD recovery receipt identity is duplicated")
            recovery_receipts[receipt_id] = {
                "payload": receipt_payload,
                "sha256": receipt_capture.sha256,
            }
            bindings.append(
                {
                    "logical_role": "action_recovery_receipt",
                    "source_artifact_id": receipt_id,
                    "source_publication_id": str(receipt_payload["receipt_version"]),
                    "sha256": receipt_capture.sha256,
                    "manifest_sha256": receipt_capture.sha256,
                }
            )
    telemetry_snapshot.sort(key=lambda item: str(item["action_id"]))
    findings.sort(key=lambda item: (str(item["action_id"]), str(item["reason_code"])))
    return events, bindings, telemetry_snapshot, findings, recovery_receipts


def _project_children(
    *,
    storage: ProjectStorage,
    project_id: str,
    states: list[dict[str, Any]],
    captures: dict[Path, _CapturedFile],
    directory_rosters: list[_CapturedDirectoryRoster],
    recovery_receipts: dict[str, dict[str, Any]],
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
                stage_payload = StageState.model_validate(
                    _read_action_json_bytes(stage_capture.payload)
                )
                dispatch_events, dispatch_bindings = _project_dispatch_receipts(
                    run_dir=storage.run_dir(project_id, str(child["run_id"])),
                    child=child,
                    revision=revision,
                    captures=captures,
                    directory_rosters=directory_rosters,
                    expected_authority_roster=stage_payload.details.get(
                        "dispatch_authority_roster"
                    ),
                )
                bindings.extend(dispatch_bindings)
                if dispatch_events:
                    events.extend(dispatch_events)
                else:
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
                stage_payload = StageState.model_validate(
                    _read_action_json_bytes(stage_capture.payload)
                )
                failure_evidence = None
                if "failure_evidence" in stage_payload.details:
                    failure_evidence = validate_failure_evidence(
                        stage_payload.details["failure_evidence"]
                    )
                    _validate_projected_failure_evidence(
                        evidence=failure_evidence,
                        child=child,
                        states=states,
                        recovery_receipts=recovery_receipts,
                    )
                outcome = {"child_status": child["status"]}
                reason_codes = [str(child["status"])]
                if failure_evidence is not None and child["status"] != "succeeded":
                    reason_codes.extend(failure_evidence["reason_codes"])
                    outcome["recovery_disposition"] = failure_evidence[
                        "recovery_disposition"
                    ]
                    if failure_evidence["recovery_receipt_id"] is not None:
                        outcome["recovery_receipt_id"] = failure_evidence[
                            "recovery_receipt_id"
                        ]
                    if failure_evidence["causal_link"] is not None:
                        outcome["causal_link"] = failure_evidence["causal_link"]
                events.append(
                    _event(
                        kind="stage_completed"
                        if child["status"] == "succeeded"
                        else "stage_failed",
                        revision=revision,
                        child=child,
                        source=stage_binding,
                        outcome=outcome,
                        reason_codes=sorted(set(reason_codes)),
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


def _captured_dispatch_receipts(
    *,
    run_dir: Path,
    captures: dict[Path, _CapturedFile],
    directory_rosters: list[_CapturedDirectoryRoster],
) -> list[tuple[dict[str, Any], _CapturedFile, _CapturedFile]]:
    root = run_dir / "dispatch-receipts"
    roster = _capture_directory_roster(
        root,
        kind="dispatch_receipts",
        allow_missing=True,
    )
    directory_rosters.append(roster)
    receipts: list[tuple[dict[str, Any], _CapturedFile, _CapturedFile]] = []
    for receipt_name in roster.names:
        receipt_dir = root / receipt_name
        payload_roster = _capture_directory_roster(
            receipt_dir,
            kind="receipt_payloads",
        )
        directory_rosters.append(payload_roster)
        if payload_roster.names != ("authority.json", "receipt.json"):
            raise ValueError("PR-BD dispatch receipt file roster is invalid")
        authority_capture = _capture_canonical_json(
            receipt_dir / "authority.json",
            captures=captures,
        )
        authority = validate_dispatch_authority(
            _read_action_json_bytes(authority_capture.payload)
        )
        receipt_capture = _capture_canonical_json(
            receipt_dir / "receipt.json",
            captures=captures,
        )
        receipt = validate_dispatch_receipt(
            _read_action_json_bytes(receipt_capture.payload)
        )
        if receipt["receipt_id"] != receipt_name:
            raise ValueError("PR-BD dispatch receipt path identity is invalid")
        if (
            receipt["dispatch_authority_id"] != authority["authority_id"]
            or receipt["request_or_stage_digest"] != authority_capture.sha256
            or receipt["child_run_id"] != authority["child_run_id"]
            or receipt["task_id"] != authority["task_id"]
            or receipt["attempt_id"] != authority["attempt_id"]
            or receipt["dispatch_kind"] != authority["dispatch_kind"]
            or receipt["execution_started"] is not authority["execution_started"]
        ):
            raise ValueError("PR-BD dispatch authority binding is invalid")
        receipts.append((receipt, receipt_capture, authority_capture))
    receipts.sort(
        key=lambda item: (
            int(item[0]["dispatch_ordinal"]),
            str(item[0]["receipt_id"]),
        )
    )
    if [int(item[0]["dispatch_ordinal"]) for item in receipts] != list(
        range(1, len(receipts) + 1)
    ):
        raise ValueError("PR-BD dispatch receipt ordinal roster is invalid")
    attempt_ids = [str(item[0]["attempt_id"]) for item in receipts]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("PR-BD dispatch receipt attempt roster is invalid")
    predecessor = None
    for receipt, _, _ in receipts:
        if receipt["predecessor_receipt_id"] != predecessor:
            raise ValueError("PR-BD dispatch receipt predecessor chain is invalid")
        predecessor = receipt["receipt_id"]
    return receipts


def _project_dispatch_receipts(
    *,
    run_dir: Path,
    child: dict[str, Any],
    revision: int,
    captures: dict[Path, _CapturedFile],
    directory_rosters: list[_CapturedDirectoryRoster],
    expected_authority_roster: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    receipts = _captured_dispatch_receipts(
        run_dir=run_dir,
        captures=captures,
        directory_rosters=directory_rosters,
    )
    observed_authority_roster = [
        {
            "receipt_id": str(receipt["receipt_id"]),
            "dispatch_authority_id": str(receipt["dispatch_authority_id"]),
            "authority_sha256": authority_capture.sha256,
        }
        for receipt, _, authority_capture in receipts
        if receipt["dispatch_kind"] in {"initial", "retry", "duplicate_rejected"}
    ]
    if observed_authority_roster:
        if expected_authority_roster != observed_authority_roster:
            raise ValueError("PR-BD dispatch StageState authority roster is invalid")
    elif expected_authority_roster is not None:
        raise ValueError("PR-BD dispatch StageState authority roster is invalid")
    for receipt, receipt_capture, authority_capture in receipts:
        if (
            receipt["child_run_id"] != child["run_id"]
            or receipt["task_id"] != child["task_id"]
        ):
            raise ValueError("PR-BD dispatch receipt child binding is invalid")
        binding = {
            "logical_role": "child_dispatch_receipt",
            "source_artifact_id": str(receipt["receipt_id"]),
            "source_publication_id": str(receipt["receipt_version"]),
            "sha256": receipt_capture.sha256,
            "manifest_sha256": authority_capture.sha256,
        }
        bindings.append(binding)
        if receipt["dispatch_kind"] not in {
            "initial",
            "retry",
            "duplicate_rejected",
        }:
            continue
        events.append(
            _event(
                kind="task_dispatched",
                revision=revision,
                child=child,
                source=binding,
                outcome={
                    "child_status": child["status"],
                    "dispatch_kind": receipt["dispatch_kind"],
                    "dispatch_ordinal": receipt["dispatch_ordinal"],
                    "execution_started": receipt["execution_started"],
                },
                reason_codes=list(receipt["reason_codes"]),
            )
        )
    return events, bindings


def _validate_projected_recovery_receipt(
    *,
    storage: ProjectStorage,
    project_id: str,
    request: dict[str, Any],
    receipt: dict[str, Any],
    states: list[dict[str, Any]],
    captures: dict[Path, _CapturedFile],
    directory_rosters: list[_CapturedDirectoryRoster],
) -> None:
    if (
        receipt["action_id"] != request["action_id"]
        or receipt["request_digest"] != request["request_digest"]
        or receipt["expected_revision"] != request["expected_revision"]
    ):
        raise ValueError("PR-BD recovery receipt request binding is invalid")
    completed_revision = int(receipt["completed_revision"])
    if completed_revision >= len(states):
        raise ValueError("PR-BD recovery receipt revision is unavailable")
    completed_state = states[completed_revision]
    child_matches = [
        child
        for child in completed_state["children"]
        if child["run_id"] == receipt["recovered_child_run_id"]
        and child["status"] == "succeeded"
    ]
    if len(child_matches) != 1:
        raise ValueError("PR-BD recovery receipt child binding is invalid")
    run_dir = storage.run_dir(project_id, str(receipt["recovered_child_run_id"]))
    stage_capture = _capture_file(
        run_dir / "stage.json",
        captures=captures,
    )
    StageState.model_validate(_read_action_json_bytes(stage_capture.payload))
    if stage_capture.sha256 != receipt["recovered_stage_sha256"]:
        raise ValueError("PR-BD recovery receipt StageState binding is invalid")
    dispatch_ids = sorted(
        str(payload["receipt_id"])
        for payload, _, _ in _captured_dispatch_receipts(
            run_dir=run_dir,
            captures=captures,
            directory_rosters=directory_rosters,
        )
    )
    if dispatch_ids != receipt["source_dispatch_receipt_ids"]:
        raise ValueError("PR-BD recovery receipt dispatch binding is invalid")


def _validate_projected_failure_evidence(
    *,
    evidence: dict[str, Any],
    child: dict[str, Any],
    states: list[dict[str, Any]],
    recovery_receipts: dict[str, dict[str, Any]],
) -> None:
    terminal_child_ids = {
        str(item["run_id"]) for item in states[-1]["children"]
    }
    link = evidence["causal_link"]
    if link is not None and link["cause_child_run_id"] not in terminal_child_ids:
        raise ValueError("PR-BD failure causal link child is unavailable")
    receipt_id = evidence["recovery_receipt_id"]
    if receipt_id is None:
        return
    bound = recovery_receipts.get(str(receipt_id))
    if bound is None:
        raise ValueError("PR-BD failure recovery receipt is unavailable")
    receipt = bound["payload"]
    if receipt["recovered_child_run_id"] != child["run_id"]:
        raise ValueError("PR-BD failure recovery receipt child mismatch")
    source_digests = set(evidence["source_record_digests"])
    if source_digests and bound["sha256"] not in source_digests:
        raise ValueError("PR-BD failure recovery source digest mismatch")


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
            (
                int(item["outcome"].get("dispatch_ordinal", 0))
                if item["event_kind"] == "task_dispatched"
                and isinstance(item.get("outcome"), dict)
                else 0
            ),
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
    absolute = _lexical_absolute(path)
    named_before = os.stat(absolute, follow_symlinks=False)
    payload, sha256 = _read_regular_file_bound(
        absolute,
        max_bytes=_MAX_JSON_BYTES,
        reject_symlink_components=True,
        allow_empty=True,
    )
    named_after = os.stat(absolute, follow_symlinks=False)
    if (
        not stat.S_ISREG(named_before.st_mode)
        or not stat.S_ISREG(named_after.st_mode)
        or named_before.st_dev != named_after.st_dev
        or named_before.st_ino != named_after.st_ino
        or named_before.st_size != named_after.st_size
        or named_before.st_mtime_ns != named_after.st_mtime_ns
        or named_before.st_ctime_ns != named_after.st_ctime_ns
        or named_after.st_size != len(payload)
    ):
        raise ValueError("PR-BD source named file changed during capture")
    captured = _CapturedFile(
        payload=payload,
        sha256=sha256,
        device=named_after.st_dev,
        inode=named_after.st_ino,
        size=named_after.st_size,
        mtime_ns=named_after.st_mtime_ns,
        ctime_ns=named_after.st_ctime_ns,
    )
    existing = captures.get(absolute)
    if existing is not None and existing != captured:
        raise ValueError("PR-BD source changed during projection")
    captures[absolute] = captured
    return captured


def _capture_directory_roster(
    path: Path,
    *,
    kind: str,
    allow_missing: bool = False,
) -> _CapturedDirectoryRoster:
    absolute = _lexical_absolute(path)
    if not os.path.lexists(absolute):
        if allow_missing:
            return _CapturedDirectoryRoster(
                path=absolute,
                kind=kind,
                names=(),
                existed=False,
                device=None,
                inode=None,
            )
        raise ValueError("PR-BD authoritative source directory is unavailable")
    _require_existing_directory(absolute, "PR-BD authoritative source directory")
    before = os.stat(absolute, follow_symlinks=False)
    names = _selected_directory_names(absolute, kind=kind)
    after = os.stat(absolute, follow_symlinks=False)
    if (
        not stat.S_ISDIR(before.st_mode)
        or not stat.S_ISDIR(after.st_mode)
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
    ):
        raise ValueError("PR-BD authoritative source directory changed")
    return _CapturedDirectoryRoster(
        path=absolute,
        kind=kind,
        names=names,
        existed=True,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _selected_directory_names(path: Path, *, kind: str) -> tuple[str, ...]:
    if kind == "session_states":
        return tuple(
            sorted(
                name
                for name in os.listdir(path)
                if name.startswith("state_")
                and name.endswith(".json")
                and name[6:-5].isdigit()
            )
        )
    if kind == "session_terminal":
        return tuple(
            name for name in ("session_result.json",) if os.path.lexists(path / name)
        )
    if kind == "actions":
        return tuple(
            sorted(
                name
                for name in os.listdir(path)
                if name.startswith("oled-session-action-")
            )
        )
    if kind in {"dispatch_receipts", "recovery_receipts", "receipt_payloads"}:
        return tuple(sorted(os.listdir(path)))
    raise ValueError("PR-BD source roster kind is unsupported")


def _recheck_captures(captures: dict[Path, _CapturedFile]) -> None:
    for path, expected in captures.items():
        named_before = os.stat(path, follow_symlinks=False)
        payload, sha256 = _read_regular_file_bound(
            path,
            max_bytes=max(_MAX_JSON_BYTES, len(expected.payload)),
            reject_symlink_components=True,
            allow_empty=True,
        )
        named_after = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(named_before.st_mode)
            or not stat.S_ISREG(named_after.st_mode)
            or named_before.st_dev != expected.device
            or named_before.st_ino != expected.inode
            or named_after.st_dev != expected.device
            or named_after.st_ino != expected.inode
            or named_after.st_size != expected.size
            or named_after.st_mtime_ns != expected.mtime_ns
            or named_after.st_ctime_ns != expected.ctime_ns
            or payload != expected.payload
            or sha256 != expected.sha256
        ):
            raise ValueError("PR-BD source changed before publication")


def _recheck_directory_rosters(
    rosters: list[_CapturedDirectoryRoster],
) -> None:
    for expected in rosters:
        exists = os.path.lexists(expected.path)
        if exists != expected.existed:
            raise ValueError("PR-BD authoritative source roster changed")
        if not exists:
            continue
        _require_existing_directory(
            expected.path,
            "PR-BD authoritative source directory",
        )
        current = os.stat(expected.path, follow_symlinks=False)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != expected.device
            or current.st_ino != expected.inode
        ):
            raise ValueError("PR-BD authoritative source directory changed")
        if _selected_directory_names(expected.path, kind=expected.kind) != expected.names:
            raise ValueError("PR-BD authoritative source roster changed")


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
