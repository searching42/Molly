from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from ai4s_agent._utils import write_json

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
        json_path = (run_path / "artifact_registry.json").resolve()

        def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            artifacts = payload.get("artifacts", {})
            if not isinstance(artifacts, dict):
                artifacts = {}
            artifacts[str(artifact_id)] = str(relative_path)
            return {"artifacts": artifacts}

        return locked_json_update(json_path, mutate)

    def append_asset_promotion_record_locked(
        self: Any,
        project_id: str,
        run_id: str,
        record: Any,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)
        json_path = (run_path / "asset_promotion_records.json").resolve()

        def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            records = payload.get("records", [])
            if not isinstance(records, list):
                records = []
            records.append(record.model_dump(mode="json"))
            return {"records": records}

        return locked_json_update(json_path, mutate)

    def append_gate_decision_locked(
        self: Any,
        project_id: str,
        run_id: str,
        decision: Any,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)
        json_path = (run_path / "gate_decisions.json").resolve()

        def mutate(payload: dict[str, Any]) -> dict[str, Any]:
            decisions = payload.get("decisions", [])
            if not isinstance(decisions, list):
                decisions = []
            decisions.append(decision.model_dump(mode="json"))
            return {"run_id": str(run_id), "decisions": decisions}

        return locked_json_update(json_path, mutate)

    register_artifact_path_locked._json_rmw_locked = True  # type: ignore[attr-defined]
    ProjectStorage.register_artifact_path = register_artifact_path_locked  # type: ignore[method-assign]
    ProjectStorage.append_asset_promotion_record = append_asset_promotion_record_locked  # type: ignore[method-assign]
    ProjectStorage.append_gate_decision = append_gate_decision_locked  # type: ignore[method-assign]


def locked_json_update(path: Path, mutator: Callable[[dict[str, Any]], dict[str, Any]]) -> Path:
    """Serialize JSON read-modify-write updates for one file path."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    lock_key = str(resolved)
    with _LOCKS[lock_key]:
        lock_path = resolved.with_name(f".{resolved.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                payload = _read_json_object(resolved)
                updated = mutator(payload)
                if not isinstance(updated, dict):
                    raise TypeError("JSON RMW mutator must return an object")
                write_json(resolved, updated)
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return resolved


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}
