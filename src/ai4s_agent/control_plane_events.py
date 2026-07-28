"""Non-authoritative durable projection of committed control-plane facts.

The projector observes exact-replayed Session revisions, verified child
artifacts, and validated action telemetry.  Its JSONL journal is a UI replay
channel only: it never selects, advances, approves, recovers, or completes a
scientific task, and no executor may use it as a source of truth.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:  # pragma: no cover - POSIX CI exercises the primary branch.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from ai4s_agent._utils import now_iso
from ai4s_agent.oled_bounded_discovery_session import (
    COMPLETED_TOP_N,
    FAILED,
    RECOVERY_REQUIRED,
    STOPPED_BOUNDED_NO_SOLUTION,
)
from ai4s_agent.oled_bounded_discovery_session_actions import (
    OledBoundedDiscoverySessionActionService,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    replay_oled_bounded_discovery_projection_source,
    validated_oled_bounded_project_id,
)
from ai4s_agent.storage import ProjectStorage


_EVENT_SCHEMA = "control_plane_durable_event.v1"
_PROJECTION_SCHEMA = "control_plane_event_projection.v1"
_EVENT_KEYS = {
    "schema_version",
    "event_id",
    "project_id",
    "session_id",
    "event_type",
    "occurred_at",
    "observed_at",
    "source_key",
    "data",
    "durable",
}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()
_ACTION_EVENT_TYPES = {
    "QUEUED": "action.queued",
    "RUNNING": "action.running",
    "SUCCEEDED": "action.succeeded",
    "FAILED": "action.failed",
    "RECOVERED": "action.recovered",
    "RECOVERY_REQUIRED": "action.recovery_required",
}
_DURABLE_EVENT_TYPES = {
    "session.created",
    "session.stage_changed",
    "action.queued",
    "action.running",
    "action.succeeded",
    "action.failed",
    "action.recovered",
    "action.recovery_required",
    "gate.waiting",
    "gate.approved",
    "artifact.registered",
    "session.completed",
    "session.failed",
    "session.recovery_required",
    "session.reconciliation_available",
}
_TERMINAL_EVENT_TYPES = {
    COMPLETED_TOP_N: "session.completed",
    STOPPED_BOUNDED_NO_SOLUTION: "session.completed",
    RECOVERY_REQUIRED: "session.recovery_required",
    FAILED: "session.failed",
}


class ControlPlaneEventProjector:
    """Append monotonic observations while preserving authoritative replay."""

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        actions: OledBoundedDiscoverySessionActionService,
        events_root: Path,
    ) -> None:
        self.storage = storage
        self.actions = actions
        root_candidate = Path(events_root)
        if root_candidate.is_symlink():
            raise ValueError("event projection root is a symbolic link")
        self.events_root = root_candidate.resolve()
        self.events_root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.events_root, 0o700)

    def project(
        self,
        *,
        project_id: str,
        session_id: str,
        after_event_id: int = 0,
    ) -> dict[str, Any]:
        """Return an exact snapshot plus durable observations after one cursor."""

        if isinstance(after_event_id, bool) or after_event_id < 0:
            raise ValueError("Last-Event-ID must be a non-negative integer")
        clean_project = validated_oled_bounded_project_id(project_id)
        clean_session = self._validated_session_id(session_id)
        snapshot, observations = self._authoritative_observations(
            project_id=clean_project,
            session_id=clean_session,
        )
        projection_dir = self._projection_dir(clean_project, clean_session)
        with self._projection_lock(projection_dir):
            journal = projection_dir / "events.jsonl"
            events = self._read_journal(
                journal,
                expected_project_id=clean_project,
                expected_session_id=clean_session,
            )
            known = {str(item["source_key"]) for item in events}
            for observation in observations:
                source_key = str(observation["source_key"])
                if source_key in known:
                    continue
                event = {
                    "schema_version": _EVENT_SCHEMA,
                    "event_id": len(events) + 1,
                    "project_id": clean_project,
                    "session_id": clean_session,
                    "event_type": observation["event_type"],
                    "occurred_at": observation["occurred_at"],
                    "observed_at": now_iso(),
                    "source_key": source_key,
                    "data": observation["data"],
                    "durable": True,
                }
                self._append_event(journal, event)
                events.append(event)
                known.add(source_key)
            latest = len(events)
            if after_event_id > latest:
                raise ValueError("durable event cursor is unavailable; reload the snapshot")
            replay = [item for item in events if int(item["event_id"]) > after_event_id]
        return {
            "schema_version": _PROJECTION_SCHEMA,
            "snapshot": snapshot,
            "durable_events": replay,
            "cursor": {
                "requested_after": after_event_id,
                "latest_event_id": latest,
            },
            "authority": {
                "sources": [
                    "session_state",
                    "action_telemetry",
                    "immutable_execution_and_publication_records",
                    "gate_decisions",
                    "artifact_registry",
                    "recovery_results",
                ],
                "projector_is_authoritative": False,
                "events_may_drive_execution": False,
            },
        }

    def _authoritative_observations(
        self, *, project_id: str, session_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        snapshot, states, reconciliation = replay_oled_bounded_discovery_projection_source(
            storage=self.storage,
            project_id=project_id,
            session_id=session_id,
        )
        observations = self._session_observations(states)
        if reconciliation is not None:
            observations.append(
                {
                    "source_key": (
                        f"session:revision:{reconciliation['session_revision']}:"
                        f"reconciliation:{reconciliation['run_id']}:"
                        f"{reconciliation['observed_stage_status']}"
                    ),
                    "event_type": "session.reconciliation_available",
                    "occurred_at": str(states[-1].get("updated_at") or ""),
                    "data": dict(reconciliation),
                }
            )
        for action in self.actions.list_session_actions(
            project_id=project_id,
            session_id=session_id,
        ):
            status = str(action.get("status") or "")
            persisted_status = str(action.get("persisted_status") or status)
            event_type = _ACTION_EVENT_TYPES.get(status)
            if event_type is None:
                continue
            action_id = str(action.get("action_id") or "")
            observations.append(
                {
                    "source_key": f"action:{action_id}:QUEUED",
                    "event_type": "action.queued",
                    "occurred_at": str(action.get("created_at") or ""),
                    "data": {
                        "action_id": action_id,
                        "action": str(action.get("action") or ""),
                        "status": "QUEUED",
                        "expected_revision": action.get("expected_revision"),
                    },
                }
            )
            if persisted_status in {"RUNNING", "SUCCEEDED", "FAILED", "RECOVERED"}:
                observations.append(
                    {
                        "source_key": f"action:{action_id}:RUNNING",
                        "event_type": "action.running",
                        "occurred_at": str(action.get("updated_at") or ""),
                        "data": {
                            "action_id": action_id,
                            "action": str(action.get("action") or ""),
                            "status": "RUNNING",
                            "expected_revision": action.get("expected_revision"),
                        },
                    }
                )
            if status == "QUEUED":
                continue
            observations.append(
                {
                    "source_key": f"action:{action_id}:{status}",
                    "event_type": event_type,
                    "occurred_at": str(action.get("updated_at") or action.get("created_at") or ""),
                    "data": {
                        "action_id": action_id,
                        "action": str(action.get("action") or ""),
                        "status": status,
                        "persisted_status": persisted_status,
                        "expected_revision": action.get("expected_revision"),
                        "completed_revision": action.get("completed_revision"),
                    },
                }
            )
        observations.sort(
            key=lambda item: (
                str(item.get("occurred_at") or ""),
                str(item.get("source_key") or ""),
            )
        )
        return snapshot, observations

    @staticmethod
    def _session_observations(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for index, state in enumerate(states):
            revision = int(state["revision"])
            occurred_at = str(state.get("updated_at") or state.get("created_at") or "")
            previous = states[index - 1] if index else None
            if revision == 0:
                observations.append(
                    {
                        "source_key": "session:revision:0:created",
                        "event_type": "session.created",
                        "occurred_at": occurred_at,
                        "data": {
                            "revision": 0,
                            "status": state["status"],
                            "current_step": state["current_step"],
                        },
                    }
                )
            if previous is not None and (
                previous["status"] != state["status"]
                or previous["current_step"] != state["current_step"]
            ):
                observations.append(
                    {
                        "source_key": f"session:revision:{revision}:stage",
                        "event_type": "session.stage_changed",
                        "occurred_at": occurred_at,
                        "data": {
                            "revision": revision,
                            "status": state["status"],
                            "current_step": state["current_step"],
                        },
                    }
                )
            before_children = {
                str(item["label"]): item for item in (previous or {}).get("children", [])
            }
            for child in state["children"]:
                label = str(child["label"])
                before = before_children.get(label)
                if before == child:
                    continue
                child_status = str(child["status"])
                if child_status == "waiting_user":
                    observations.append(
                        {
                            "source_key": f"session:revision:{revision}:gate:{label}:waiting",
                            "event_type": "gate.waiting",
                            "occurred_at": occurred_at,
                            "data": {
                                "revision": revision,
                                "gate": "gate_5_final_threshold",
                                "run_id": child["run_id"],
                                "task_id": child["task_id"],
                            },
                        }
                    )
                if (
                    before is not None
                    and before.get("status") == "waiting_user"
                    and child_status == "succeeded"
                ):
                    observations.append(
                        {
                            "source_key": f"session:revision:{revision}:gate:{label}:approved",
                            "event_type": "gate.approved",
                            "occurred_at": occurred_at,
                            "data": {
                                "revision": revision,
                                "gate": "gate_5_final_threshold",
                                "run_id": child["run_id"],
                                "task_id": child["task_id"],
                            },
                        }
                    )
                observations.extend(
                    ControlPlaneEventProjector._artifact_observations(
                        revision=revision,
                        occurred_at=occurred_at,
                        child=child,
                        previous_child=before,
                    )
                )
                if child_status in {"recovery_required", "integrity_failed", "failed"}:
                    observations.append(
                        {
                            "source_key": f"session:revision:{revision}:child:{label}:{child_status}",
                            "event_type": (
                                "action.recovery_required"
                                if child_status == "recovery_required"
                                else "action.failed"
                            ),
                            "occurred_at": occurred_at,
                            "data": {
                                "revision": revision,
                                "run_id": child["run_id"],
                                "task_id": child["task_id"],
                                "status": child_status,
                            },
                        }
                    )
            terminal_type = _TERMINAL_EVENT_TYPES.get(str(state["status"]))
            if terminal_type and (previous is None or previous["status"] != state["status"]):
                observations.append(
                    {
                        "source_key": f"session:revision:{revision}:terminal:{state['status']}",
                        "event_type": terminal_type,
                        "occurred_at": occurred_at,
                        "data": {
                            "revision": revision,
                            "status": state["status"],
                            "current_step": state["current_step"],
                        },
                    }
                )
        return observations

    @staticmethod
    def _artifact_observations(
        *,
        revision: int,
        occurred_at: str,
        child: dict[str, Any],
        previous_child: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        current = child.get("artifacts") if isinstance(child.get("artifacts"), dict) else {}
        previous = (
            previous_child.get("artifacts")
            if isinstance((previous_child or {}).get("artifacts"), dict)
            else {}
        )
        result: list[dict[str, Any]] = []
        for artifact_id in sorted(set(current) - set(previous)):
            result.append(
                {
                    "source_key": f"session:revision:{revision}:artifact:{child['run_id']}:{artifact_id}",
                    "event_type": "artifact.registered",
                    "occurred_at": occurred_at,
                    "data": {
                        "revision": revision,
                        "run_id": child["run_id"],
                        "task_id": child["task_id"],
                        "artifact_id": artifact_id,
                    },
                }
            )
        return result

    def _projection_dir(self, project_id: str, session_id: str) -> Path:
        clean_project = validated_oled_bounded_project_id(project_id)
        project_candidate = self.events_root / clean_project
        if project_candidate.is_symlink():
            raise ValueError("event projection project is a symbolic link")
        project_root = project_candidate.resolve()
        if not project_root.is_relative_to(self.events_root):
            raise ValueError("event projection project escapes storage")
        project_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(project_root, 0o700)
        clean_session = self._validated_session_id(session_id)
        session_candidate = project_root / clean_session
        if session_candidate.is_symlink():
            raise ValueError("event projection session is a symbolic link")
        path = session_candidate.resolve()
        if not path.is_relative_to(project_root):
            raise ValueError("event projection session escapes storage")
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path, 0o700)
        return path

    @staticmethod
    def _validated_session_id(value: str) -> str:
        clean = str(value or "")
        if (
            clean != clean.strip()
            or not clean.startswith("oled-bounded-session-")
            or Path(clean).name != clean
            or "/" in clean
            or "\\" in clean
        ):
            raise ValueError("event projection session ID is invalid")
        return clean

    @contextmanager
    def _projection_lock(self, directory: Path) -> Iterator[None]:
        lock_path = directory / ".projection.lock"
        if lock_path.is_symlink():
            raise ValueError("event projection lock is a symbolic link")
        key = str(lock_path)
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key, threading.RLock())
        with lock:
            with lock_path.open("a+", encoding="utf-8") as descriptor:
                os.chmod(lock_path, 0o600)
                if fcntl is not None:
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(descriptor.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _read_journal(
        path: Path,
        *,
        expected_project_id: str,
        expected_session_id: str,
    ) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        if path.is_symlink() or not path.is_file():
            raise ValueError("durable event journal is not a regular file")
        os.chmod(path, 0o600)
        data = path.read_bytes()
        if not data:
            return []
        lines = data.splitlines(keepends=True)
        events: list[dict[str, Any]] = []
        source_keys: set[str] = set()
        offset = 0
        for index, raw in enumerate(lines):
            if index == len(lines) - 1 and not raw.endswith((b"\n", b"\r")):
                with path.open("r+b") as output:
                    output.truncate(offset)
                    output.flush()
                    os.fsync(output.fileno())
                break
            try:
                loaded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError("durable event journal contains a corrupt record") from exc
            if not isinstance(loaded, dict):
                raise ValueError("durable event journal record must be an object")
            expected_id = len(events) + 1
            event_id = loaded.get("event_id")
            source_key = loaded.get("source_key")
            if (
                set(loaded) != _EVENT_KEYS
                or loaded.get("schema_version") != _EVENT_SCHEMA
                or isinstance(event_id, bool)
                or event_id != expected_id
                or loaded.get("project_id") != expected_project_id
                or loaded.get("session_id") != expected_session_id
                or loaded.get("durable") is not True
                or not isinstance(source_key, str)
                or not source_key
                or source_key in source_keys
                or loaded.get("event_type") not in _DURABLE_EVENT_TYPES
                or not isinstance(loaded.get("occurred_at"), str)
                or not isinstance(loaded.get("observed_at"), str)
                or not isinstance(loaded.get("data"), dict)
            ):
                raise ValueError("durable event journal identity or schema mismatch")
            events.append(loaded)
            source_keys.add(source_key)
            offset += len(raw)
        return events

    @staticmethod
    def _append_event(path: Path, event: dict[str, Any]) -> None:
        encoded = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        fd = os.open(
            path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("could not append durable control-plane event")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)


__all__ = ["ControlPlaneEventProjector"]
