from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

try:  # pragma: no cover - exercised on POSIX CI, fallback keeps imports portable.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


_LOCKS: dict[str, threading.RLock] = defaultdict(threading.RLock)


def install_json_rmw_locks() -> None:
    """Install locked read-modify-write helpers for mutable ProjectStorage JSON files."""

    from ai4s_agent.storage import ProjectStorage

    if getattr(ProjectStorage.register_artifact_path, "_json_rmw_locked", False):
        return

    def register_artifact_path_locked(
        self: Any,
        project_id: str,
        run_id: str,
        artifact_id: str,
        relative_path: str,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)

        def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            artifacts = payload.get("artifacts", {})
            if not isinstance(artifacts, dict):
                artifacts = {}
            artifacts[str(artifact_id)] = str(relative_path)
            return {"artifacts": artifacts}

        return locked_storage_json_update(self, run_path, "artifact_registry.json", mutate)

    def append_asset_promotion_record_locked(
        self: Any,
        project_id: str,
        run_id: str,
        record: Any,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)

        def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            records = payload.get("records", [])
            if not isinstance(records, list):
                records = []
            records.append(record.model_dump(mode="json"))
            return {"records": records}

        return locked_storage_json_update(self, run_path, "asset_promotion_records.json", mutate)

    def append_gate_decision_locked(
        self: Any,
        project_id: str,
        run_id: str,
        decision: Any,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)

        def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            decisions = payload.get("decisions", [])
            if not isinstance(decisions, list):
                decisions = []
            decisions.append(decision.model_dump(mode="json"))
            return {"run_id": str(run_id), "decisions": decisions}

        return locked_storage_json_update(self, run_path, "gate_decisions.json", mutate)

    register_artifact_path_locked._json_rmw_locked = True  # type: ignore[attr-defined]
    ProjectStorage.register_artifact_path = register_artifact_path_locked  # type: ignore[method-assign]
    ProjectStorage.append_asset_promotion_record = append_asset_promotion_record_locked  # type: ignore[method-assign]
    ProjectStorage.append_gate_decision = append_gate_decision_locked  # type: ignore[method-assign]


def locked_storage_json_update(
    storage: Any,
    base_path: Path,
    filename: str,
    mutator: Callable[[dict[str, Any]], dict[str, Any]],
) -> Path:
    """Serialize JSON RMW while preserving ProjectStorage containment checks."""

    base = base_path.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    json_path = (base / filename).resolve()
    if not json_path.is_relative_to(base):
        raise ValueError("json_path escapes base directory")
    lock_path = (base / f".{filename}.lock").resolve()
    if not lock_path.is_relative_to(base):
        raise ValueError("json lock path escapes base directory")
    lock_key = str(json_path)
    with _LOCKS[lock_key]:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                payload = storage._read_json(base, filename)
                updated = mutator(payload)
                if not isinstance(updated, dict):
                    raise TypeError("JSON RMW mutator must return an object")
                return storage._write_json(base, filename, updated)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
