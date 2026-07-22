"""Non-blocking, fail-closed controls for PR-AV session transitions."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.oled_bounded_discovery_session import (
    WAITING_USER,
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
)
from ai4s_agent.oled_bounded_discovery_session_view import (
    build_oled_bounded_discovery_session_view,
)
from ai4s_agent.storage import ProjectStorage


_ACTIVE = {"QUEUED", "RUNNING"}
_TERMINAL = {"SUCCEEDED", "FAILED"}


class OledBoundedDiscoverySessionActionService:
    """Run one coordinator transition outside the HTTP request thread.

    The service is deliberately same-process and single-worker. A record left
    QUEUED or RUNNING by another process instance is exposed as
    RECOVERY_REQUIRED and is never replayed automatically.
    """

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        actions_root: Path,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.storage = storage
        self.actions_root = actions_root.resolve()
        self.actions_root.mkdir(parents=True, exist_ok=True)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="oled-bounded-session",
        )
        self._instance_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._futures: dict[str, Future[Any]] = {}

    def enqueue_advance(
        self, *, project_id: str, session_id: str, expected_revision: int
    ) -> dict[str, Any]:
        return self._enqueue(
            project_id=project_id,
            session_id=session_id,
            expected_revision=expected_revision,
            action="advance",
            actor="",
            note="",
        )

    def enqueue_approval(
        self,
        *,
        project_id: str,
        session_id: str,
        expected_revision: int,
        actor: str,
        note: str = "",
    ) -> dict[str, Any]:
        clean_actor = str(actor or "").strip()
        if not clean_actor:
            raise ValueError("PR-AW gate approval actor is required")
        return self._enqueue(
            project_id=project_id,
            session_id=session_id,
            expected_revision=expected_revision,
            action="approve",
            actor=clean_actor,
            note=str(note or "").strip(),
        )

    def get_action(self, *, project_id: str, action_id: str) -> dict[str, Any]:
        record = self._read(project_id, action_id)
        status = str(record.get("status") or "")
        if status in _ACTIVE and not self._owned_active_action(record):
            return {
                **_public_record(record),
                "persisted_status": status,
                "status": "RECOVERY_REQUIRED",
                "error": {
                    "code": "action_process_interrupted",
                    "message": (
                        "The process owning this action is unavailable; inspect the "
                        "session before issuing another transition."
                    ),
                },
            }
        public = _public_record(record)
        if status == "SUCCEEDED":
            # Action JSON is mutable control-plane metadata, not a scientific
            # trust anchor.  Never serve its persisted result blindly: rebuild
            # the current presentation through PR-AV's exact external-fact
            # replay on every successful-action read.
            public["result"] = build_oled_bounded_discovery_session_view(
                storage=self.storage,
                project_id=project_id,
                session_id=str(record["session_id"]),
            )
        return public

    def _enqueue(
        self,
        *,
        project_id: str,
        session_id: str,
        expected_revision: int,
        action: str,
        actor: str,
        note: str,
    ) -> dict[str, Any]:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("PR-AW expected_revision must be an integer")
        view = build_oled_bounded_discovery_session_view(
            storage=self.storage,
            project_id=project_id,
            session_id=session_id,
        )
        if view["revision"] != expected_revision:
            raise ValueError("PR-AW session revision conflict")
        if action == "advance" and view["status"] == WAITING_USER:
            raise ValueError("PR-AW session requires gate approval")
        if action == "approve" and view["status"] != WAITING_USER:
            raise ValueError("PR-AW session is not waiting for approval")
        if view["status"] not in {"ACTIVE", WAITING_USER}:
            raise ValueError("PR-AW terminal session cannot be advanced")
        clean_project = _safe_segment(project_id, "project_id")
        with self._lock:
            active = self._active_action(
                clean_project,
                session_id,
                current_revision=int(view["revision"]),
            )
            if active is not None:
                raise ValueError(
                    "PR-AW session already has an active or unreconciled action"
                )
            action_id = "oled-session-action-" + uuid.uuid4().hex
            timestamp = now_iso()
            record = {
                "action_version": "oled_bounded_discovery_session_action.v1",
                "action_id": action_id,
                "project_id": clean_project,
                "session_id": session_id,
                "action": action,
                "expected_revision": expected_revision,
                "actor": actor,
                "note": note,
                "status": "QUEUED",
                "created_at": timestamp,
                "updated_at": timestamp,
                "instance_id": self._instance_id,
                "result": None,
                "error": None,
            }
            self._write(record)
            try:
                self._futures[action_id] = self._executor.submit(
                    self._execute, clean_project, action_id
                )
            except Exception as exc:
                failed = {
                    **record,
                    "status": "FAILED",
                    "updated_at": now_iso(),
                    "error": _error("action_dispatch_failed", exc),
                }
                self._write(failed)
                raise ValueError("PR-AW action dispatch failed") from exc
            return _public_record(record)

    def _execute(self, project_id: str, action_id: str) -> None:
        with self._lock:
            record = self._read(project_id, action_id)
            if (
                record.get("instance_id") != self._instance_id
                or record.get("status") != "QUEUED"
            ):
                return
            running = {**record, "status": "RUNNING", "updated_at": now_iso()}
            self._write(running)
        try:
            if running["action"] == "advance":
                result = advance_oled_bounded_discovery_session(
                    storage=self.storage,
                    project_id=project_id,
                    session_id=str(running["session_id"]),
                    expected_revision=int(running["expected_revision"]),
                )
            else:
                result = approve_oled_bounded_discovery_session_gate(
                    storage=self.storage,
                    project_id=project_id,
                    session_id=str(running["session_id"]),
                    expected_revision=int(running["expected_revision"]),
                    actor=str(running["actor"]),
                    note=str(running["note"]),
                )
            view = build_oled_bounded_discovery_session_view(
                storage=self.storage,
                project_id=project_id,
                session_id=result.session_id,
            )
            completed = {
                **running,
                "status": "SUCCEEDED",
                "updated_at": now_iso(),
                "completed_revision": view["revision"],
                "result": None,
                "error": None,
            }
        except Exception as exc:
            completed = {
                **running,
                "status": "FAILED",
                "updated_at": now_iso(),
                "result": None,
                "error": _error("session_action_failed", exc),
            }
        with self._lock:
            current = self._read(project_id, action_id)
            if current.get("instance_id") == self._instance_id:
                self._write(completed)

    def _active_action(
        self,
        project_id: str,
        session_id: str,
        *,
        current_revision: int,
    ) -> dict[str, Any] | None:
        root = self._project_root(project_id)
        for path in sorted(root.glob("oled-session-action-*/action.json")):
            try:
                record = _read_json(path)
            except (OSError, ValueError):
                continue
            if not (
                record.get("session_id") == session_id
                and record.get("status") in _ACTIVE
            ):
                continue
            expected = record.get("expected_revision")
            if not isinstance(expected, int) or isinstance(expected, bool):
                return record
            # A previous process action remains a hard recovery boundary while
            # the session is still at the revision it intended to mutate. Once
            # exact session facts have advanced beyond it, that old control
            # record must not permanently lock the next legitimate transition.
            if expected < current_revision:
                continue
            return record
        return None

    def _owned_active_action(self, record: dict[str, Any]) -> bool:
        action_id = str(record.get("action_id") or "")
        future = self._futures.get(action_id)
        return (
            record.get("instance_id") == self._instance_id
            and future is not None
            and not future.done()
        )

    def _project_root(self, project_id: str) -> Path:
        clean = _safe_segment(project_id, "project_id")
        path = (self.actions_root / clean).resolve()
        if not path.is_relative_to(self.actions_root):
            raise ValueError("PR-AW project action root escapes storage")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _path(self, project_id: str, action_id: str) -> Path:
        clean_action = _safe_action_id(action_id)
        root = self._project_root(project_id)
        path = (root / clean_action / "action.json").resolve()
        if not path.is_relative_to(root):
            raise ValueError("PR-AW action path escapes storage")
        return path

    def _read(self, project_id: str, action_id: str) -> dict[str, Any]:
        path = self._path(project_id, action_id)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("PR-AW session action is unavailable")
        record = _read_json(path)
        if (
            record.get("action_id") != action_id
            or record.get("project_id") != _safe_segment(project_id, "project_id")
        ):
            raise ValueError("PR-AW session action identity is invalid")
        return record

    def _write(self, record: dict[str, Any]) -> None:
        path = self._path(str(record["project_id"]), str(record["action_id"]))
        write_json(path, record)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("PR-AW session action is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("PR-AW session action must be an object")
    return {str(key): item for key, item in value.items()}


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"instance_id", "actor", "note", "result"}
    }


def _safe_segment(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if (
        not clean
        or clean in {".", ".."}
        or "/" in clean
        or "\\" in clean
        or Path(clean).name != clean
    ):
        raise ValueError(f"PR-AW {label} is invalid")
    return clean


def _safe_action_id(value: str) -> str:
    clean = _safe_segment(value, "action_id")
    if not clean.startswith("oled-session-action-"):
        raise ValueError("PR-AW action_id is invalid")
    return clean


def _error(code: str, exc: Exception) -> dict[str, str]:
    return {"code": code, "message": str(exc) or exc.__class__.__name__}


__all__ = ["OledBoundedDiscoverySessionActionService"]
