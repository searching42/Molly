"""Non-blocking, fail-closed controls for PR-AV session transitions."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso
from ai4s_agent.oled_bounded_discovery_session import (
    WAITING_USER,
    _write_mutable_json,
    advance_oled_bounded_discovery_session,
    approve_oled_bounded_discovery_session_gate,
)
from ai4s_agent.oled_categorical_dataset_execution import _publish_payload_directory
from ai4s_agent.oled_bounded_discovery_session_view import (
    build_oled_bounded_discovery_session_view,
    validated_oled_bounded_project_id,
)
from ai4s_agent.oled_real_phase1_execution import _json_bytes, _stable_hash
from ai4s_agent.oled_supplementary_scoped_candidate_response import (
    _read_regular_file_bound,
)
from ai4s_agent.oled_supplementary_material_identity_review import (
    _pinned_output_parents_without_symlink_components,
)
from ai4s_agent.storage import ProjectStorage


_REQUEST_VERSION = "oled_bounded_discovery_session_action_request.v1"
_STATE_VERSION = "oled_bounded_discovery_session_action_state.v1"
_ACTIVE = {"QUEUED", "RUNNING"}
_STATUSES = _ACTIVE | {"SUCCEEDED", "FAILED"}
_REQUEST_KEYS = {
    "request_version",
    "action_id",
    "project_id",
    "session_id",
    "action",
    "expected_revision",
    "actor",
    "note",
    "created_at",
    "request_nonce",
    "request_digest",
}
_ACTION_ID_FIELDS = (
    "request_version",
    "project_id",
    "session_id",
    "action",
    "expected_revision",
    "actor",
    "note",
    "created_at",
    "request_nonce",
)
_STATE_KEYS = {
    "state_version",
    "action_id",
    "project_id",
    "status",
    "updated_at",
    "instance_id",
    "request_digest",
    "completed_revision",
    "error",
}


class OledBoundedDiscoverySessionActionService:
    """Run one coordinator transition outside the HTTP request thread.

    The immutable request envelope is the only action input. ``action.json``
    contains mutable scheduling state and is never allowed to select a session,
    revision, operation, or approval identity.
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
        clean_project = validated_oled_bounded_project_id(project_id)
        request, state, _ = self._read(clean_project, action_id)
        status = str(state["status"])
        if status in _ACTIVE and not self._owned_active_action(state):
            return {
                **_public_record(request, state),
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
        public = _public_record(request, state)
        if status == "SUCCEEDED":
            public["result"] = build_oled_bounded_discovery_session_view(
                storage=self.storage,
                project_id=clean_project,
                session_id=str(request["session_id"]),
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
        clean_project = validated_oled_bounded_project_id(project_id)
        clean_session = _safe_segment(session_id, "session_id")
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int):
            raise ValueError("PR-AW expected_revision must be an integer")
        view = build_oled_bounded_discovery_session_view(
            storage=self.storage,
            project_id=clean_project,
            session_id=clean_session,
        )
        if view["revision"] != expected_revision:
            raise ValueError("PR-AW session revision conflict")
        if action == "advance" and view["status"] == WAITING_USER:
            raise ValueError("PR-AW session requires gate approval")
        if action == "approve" and view["status"] != WAITING_USER:
            raise ValueError("PR-AW session is not waiting for approval")
        if view["status"] not in {"ACTIVE", WAITING_USER}:
            raise ValueError("PR-AW terminal session cannot be advanced")

        with self._lock:
            active = self._active_action(
                clean_project,
                clean_session,
                current_revision=int(view["revision"]),
            )
            if active is not None:
                raise ValueError(
                    "PR-AW session already has an active or unreconciled action"
                )
            timestamp = now_iso()
            request_identity = {
                "request_version": _REQUEST_VERSION,
                "project_id": clean_project,
                "session_id": clean_session,
                "action": action,
                "expected_revision": expected_revision,
                "actor": actor,
                "note": note,
                "created_at": timestamp,
                "request_nonce": uuid.uuid4().hex,
            }
            action_id = "oled-session-action-" + _stable_hash(request_identity)
            request_base = {**request_identity, "action_id": action_id}
            frozen_request = {
                **request_base,
                "request_digest": "sha256:" + _stable_hash(request_base),
            }
            frozen_bytes = _json_bytes(frozen_request)
            queued_state = {
                "state_version": _STATE_VERSION,
                "action_id": action_id,
                "project_id": clean_project,
                "status": "QUEUED",
                "updated_at": timestamp,
                "instance_id": self._instance_id,
                "request_digest": frozen_request["request_digest"],
                "completed_revision": None,
                "error": None,
            }
            self._publish_initial_action(
                project_id=clean_project,
                action_id=action_id,
                request_bytes=frozen_bytes,
                queued_state=queued_state,
            )
            try:
                self._futures[action_id] = self._executor.submit(
                    self._execute,
                    clean_project,
                    action_id,
                    frozen_request,
                    frozen_bytes,
                    queued_state,
                )
            except Exception as exc:
                failed = _failed_state(
                    queued_state, "action_dispatch_failed", exc
                )
                self._write_state(failed)
                raise ValueError("PR-AW action dispatch failed") from exc
            return _public_record(frozen_request, queued_state)

    def _execute(
        self,
        project_id: str,
        action_id: str,
        frozen_request: dict[str, Any],
        frozen_bytes: bytes,
        queued_state: dict[str, Any],
    ) -> None:
        with self._lock:
            try:
                request, state, request_bytes = self._read(project_id, action_id)
                if (
                    request != frozen_request
                    or request_bytes != frozen_bytes
                    or state != queued_state
                ):
                    raise ValueError("PR-AW frozen action request changed before execution")
            except Exception as exc:
                self._write_state(
                    _failed_state(queued_state, "action_request_integrity_failed", exc)
                )
                return
            if (
                state["instance_id"] != self._instance_id
                or state["status"] != "QUEUED"
            ):
                return
            running = {**state, "status": "RUNNING", "updated_at": now_iso()}
            self._write_state(running)

        try:
            if frozen_request["action"] == "advance":
                result = advance_oled_bounded_discovery_session(
                    storage=self.storage,
                    project_id=project_id,
                    session_id=str(frozen_request["session_id"]),
                    expected_revision=int(frozen_request["expected_revision"]),
                )
            else:
                result = approve_oled_bounded_discovery_session_gate(
                    storage=self.storage,
                    project_id=project_id,
                    session_id=str(frozen_request["session_id"]),
                    expected_revision=int(frozen_request["expected_revision"]),
                    actor=str(frozen_request["actor"]),
                    note=str(frozen_request["note"]),
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
                "error": None,
            }
        except Exception as exc:
            completed = _failed_state(running, "session_action_failed", exc)

        with self._lock:
            try:
                current_request, current_state, current_bytes = self._read(
                    project_id, action_id
                )
                if (
                    current_request != frozen_request
                    or current_bytes != frozen_bytes
                    or current_state != running
                ):
                    raise ValueError("PR-AW action records changed during execution")
            except Exception as exc:
                completed = _failed_state(
                    running, "action_record_integrity_failed", exc
                )
            self._write_state(completed)

    def _active_action(
        self,
        project_id: str,
        session_id: str,
        *,
        current_revision: int,
    ) -> dict[str, Any] | None:
        root = self._project_root(project_id)
        for action_dir in sorted(root.glob("oled-session-action-*")):
            if not action_dir.is_dir() or action_dir.is_symlink():
                raise ValueError("PR-AW action directory is invalid")
            request, state, _ = self._read(project_id, action_dir.name)
            if not (
                request["session_id"] == session_id and state["status"] in _ACTIVE
            ):
                continue
            expected = int(request["expected_revision"])
            if expected < current_revision:
                continue
            return {**request, **state}
        return None

    def _owned_active_action(self, state: dict[str, Any]) -> bool:
        action_id = str(state["action_id"])
        future = self._futures.get(action_id)
        return (
            state["instance_id"] == self._instance_id
            and future is not None
            and not future.done()
        )

    def _project_root(self, project_id: str) -> Path:
        clean = validated_oled_bounded_project_id(project_id)
        path = self.actions_root / clean
        if path.exists() and path.is_symlink():
            raise ValueError("PR-AW project action root is a symbolic link")
        path.mkdir(parents=True, exist_ok=True)
        if path.resolve() != path:
            raise ValueError("PR-AW project action root escapes storage")
        return path

    def _publish_initial_action(
        self,
        *,
        project_id: str,
        action_id: str,
        request_bytes: bytes,
        queued_state: dict[str, Any],
    ) -> None:
        root = self._project_root(project_id)
        output_dir = self._action_dir(project_id, action_id)
        with _pinned_output_parents_without_symlink_components(root) as pinned:
            _publish_payload_directory(
                output_dir=output_dir,
                parent_descriptor=pinned[root],
                payloads={
                    "request.json": request_bytes,
                    "action.json": _json_bytes(queued_state),
                },
                artifact_label="bounded session action",
            )

    def _action_dir(self, project_id: str, action_id: str) -> Path:
        clean_action = _safe_action_id(action_id)
        return self._project_root(project_id) / clean_action

    def _request_path(self, project_id: str, action_id: str) -> Path:
        return self._action_dir(project_id, action_id) / "request.json"

    def _state_path(self, project_id: str, action_id: str) -> Path:
        return self._action_dir(project_id, action_id) / "action.json"

    def _read(
        self, project_id: str, action_id: str
    ) -> tuple[dict[str, Any], dict[str, Any], bytes]:
        clean_project = validated_oled_bounded_project_id(project_id)
        clean_action = _safe_action_id(action_id)
        request_path = self._request_path(clean_project, clean_action)
        state_path = self._state_path(clean_project, clean_action)
        if not request_path.exists() or not state_path.exists():
            raise FileNotFoundError("PR-AW session action is unavailable")
        request_bytes, _ = _read_regular_file_bound(
            request_path,
            max_bytes=64 * 1024,
            reject_symlink_components=True,
        )
        state_bytes, _ = _read_regular_file_bound(
            state_path,
            max_bytes=64 * 1024,
            reject_symlink_components=True,
        )
        request = _validated_request(
            _read_json_bytes(request_bytes),
            project_id=clean_project,
            action_id=clean_action,
        )
        state = _validated_state(
            _read_json_bytes(state_bytes),
            project_id=clean_project,
            action_id=clean_action,
        )
        if state["request_digest"] != request["request_digest"]:
            raise ValueError("PR-AW action state request binding mismatch")
        if _json_bytes(request) != request_bytes or _json_bytes(state) != state_bytes:
            raise ValueError("PR-AW session action JSON is not canonical")
        return request, state, request_bytes

    def _write_state(self, state: dict[str, Any]) -> None:
        validated = _validated_state(
            state,
            project_id=str(state["project_id"]),
            action_id=str(state["action_id"]),
        )
        _write_mutable_json(
            self._state_path(validated["project_id"], validated["action_id"]),
            validated,
        )


def _validated_request(
    request: dict[str, Any], *, project_id: str, action_id: str
) -> dict[str, Any]:
    if set(request) != _REQUEST_KEYS:
        raise ValueError("PR-AW action request fields are invalid")
    payload = dict(request)
    digest = payload.pop("request_digest")
    if digest != "sha256:" + _stable_hash(payload):
        raise ValueError("PR-AW action request digest mismatch")
    if (
        payload["request_version"] != _REQUEST_VERSION
        or payload["project_id"] != project_id
        or payload["action_id"] != action_id
        or payload["action"] not in {"advance", "approve"}
    ):
        raise ValueError("PR-AW action request identity is invalid")
    expected_action_id = "oled-session-action-" + _stable_hash(
        {key: payload[key] for key in _ACTION_ID_FIELDS}
    )
    if action_id != expected_action_id:
        raise ValueError("PR-AW action request does not match its path identity")
    if not isinstance(payload["session_id"], str):
        raise ValueError("PR-AW action request session is invalid")
    _safe_segment(payload["session_id"], "session_id")
    revision = payload["expected_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("PR-AW action request revision is invalid")
    if not all(
        isinstance(payload[key], str)
        for key in ("actor", "note", "created_at", "request_nonce")
    ):
        raise ValueError("PR-AW action request text is invalid")
    if len(payload["request_nonce"]) != 32 or any(
        character not in "0123456789abcdef"
        for character in payload["request_nonce"]
    ):
        raise ValueError("PR-AW action request nonce is invalid")
    if payload["action"] == "approve" and not payload["actor"]:
        raise ValueError("PR-AW approval request actor is invalid")
    if payload["action"] == "advance" and (payload["actor"] or payload["note"]):
        raise ValueError("PR-AW advance request approval fields are invalid")
    return {**payload, "request_digest": digest}


def _validated_state(
    state: dict[str, Any], *, project_id: str, action_id: str
) -> dict[str, Any]:
    if set(state) != _STATE_KEYS:
        raise ValueError("PR-AW action state fields are invalid")
    if (
        state["state_version"] != _STATE_VERSION
        or state["project_id"] != project_id
        or state["action_id"] != action_id
        or state["status"] not in _STATUSES
    ):
        raise ValueError("PR-AW action state identity is invalid")
    if not all(isinstance(state[key], str) for key in ("updated_at", "instance_id")):
        raise ValueError("PR-AW action state text is invalid")
    if not (
        isinstance(state["request_digest"], str)
        and state["request_digest"].startswith("sha256:")
    ):
        raise ValueError("PR-AW action state request digest is invalid")
    revision = state["completed_revision"]
    if revision is not None and (
        isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
    ):
        raise ValueError("PR-AW completed revision is invalid")
    if state["error"] is not None and not isinstance(state["error"], dict):
        raise ValueError("PR-AW action error is invalid")
    if state["status"] in _ACTIVE and (revision is not None or state["error"] is not None):
        raise ValueError("PR-AW active action state is invalid")
    if state["status"] == "SUCCEEDED" and (
        revision is None or state["error"] is not None
    ):
        raise ValueError("PR-AW successful action state is invalid")
    if state["status"] == "FAILED" and (
        revision is not None or not isinstance(state["error"], dict)
    ):
        raise ValueError("PR-AW failed action state is invalid")
    return dict(state)


def _read_json_bytes(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("PR-AW session action is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("PR-AW session action must be an object")
    return {str(key): item for key, item in value.items()}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("PR-AW session action contains duplicate JSON keys")
        value[key] = item
    return value


def _public_record(
    request: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    return {
        "action_version": request["request_version"],
        "action_id": request["action_id"],
        "project_id": request["project_id"],
        "session_id": request["session_id"],
        "action": request["action"],
        "expected_revision": request["expected_revision"],
        "request_digest": request["request_digest"],
        "status": state["status"],
        "created_at": request["created_at"],
        "updated_at": state["updated_at"],
        "completed_revision": state["completed_revision"],
        "error": state["error"],
    }


def _safe_segment(value: str, label: str) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise ValueError(f"PR-AW {label} must be canonical")
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or Path(value).name != value
    ):
        raise ValueError(f"PR-AW {label} is invalid")
    return value


def _safe_action_id(value: str) -> str:
    clean = _safe_segment(value, "action_id")
    prefix = "oled-session-action-"
    suffix = clean[len(prefix) :] if clean.startswith(prefix) else ""
    if len(suffix) != 64 or any(
        character not in "0123456789abcdef" for character in suffix
    ):
        raise ValueError("PR-AW action_id is invalid")
    return clean


def _failed_state(
    state: dict[str, Any], code: str, exc: Exception
) -> dict[str, Any]:
    return {
        **state,
        "status": "FAILED",
        "updated_at": now_iso(),
        "completed_revision": None,
        "error": _error(code, exc),
    }


def _error(code: str, exc: Exception) -> dict[str, str]:
    return {"code": code, "message": str(exc) or exc.__class__.__name__}


__all__ = ["OledBoundedDiscoverySessionActionService"]
