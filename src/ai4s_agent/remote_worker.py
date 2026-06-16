from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import RemoteWorkerAssignment, RemoteWorkerConfig, RemoteWorkerRequest


class RemoteWorkerRegistry:
    """Persist remote worker metadata and produce non-executable assignment plans."""

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.registry_dir = (self.workspace_dir / "workers").resolve()
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = (self.registry_dir / "remote_workers.json").resolve()
        if not self.registry_path.is_relative_to(self.registry_dir):
            raise ValueError("remote worker registry path escapes worker directory")

    def list_workers(self, *, include_disabled: bool = False) -> list[RemoteWorkerConfig]:
        workers = self._read_workers()
        if include_disabled:
            return workers
        return [worker for worker in workers if worker.enabled]

    def save_worker(self, worker: RemoteWorkerConfig) -> RemoteWorkerConfig:
        validated = RemoteWorkerConfig.model_validate(worker.model_dump(mode="json"))
        self._safe_worker_id(validated.worker_id)
        workers = self._read_workers(include_invalid=False)
        kept = [item for item in workers if item.worker_id != validated.worker_id]
        kept.append(validated)
        self._write_workers(kept)
        return validated

    def plan_assignment(self, request: RemoteWorkerRequest) -> RemoteWorkerAssignment:
        validated = RemoteWorkerRequest.model_validate(request.model_dump(mode="json"))
        required = sorted(validated.required_capabilities)
        worker = self._select_worker(validated)
        assignment_id = f"assign-{validated.run_id}-{validated.task_id}"
        if worker is None:
            return RemoteWorkerAssignment(
                assignment_id=assignment_id,
                project_id=validated.project_id,
                run_id=validated.run_id,
                task_id=validated.task_id,
                status="no_worker",
                missing_capabilities=required,
                requires_confirmation=True,
                required_permissions=["remote_worker:select"],
                budget_limit_sec=validated.budget_limit_sec,
                executable=False,
                notes=["No enabled worker matches the requested capabilities."],
            )
        if not worker.enabled:
            return RemoteWorkerAssignment(
                assignment_id=assignment_id,
                project_id=validated.project_id,
                run_id=validated.run_id,
                task_id=validated.task_id,
                worker_id=worker.worker_id,
                transport=worker.transport,
                host=worker.host,
                status="disabled",
                missing_capabilities=required,
                requires_confirmation=True,
                required_permissions=[f"remote_worker:{worker.worker_id}"],
                budget_limit_sec=validated.budget_limit_sec,
                executable=False,
                notes=["The requested worker is disabled."],
            )
        matched = sorted(set(required).intersection(worker.capabilities))
        missing = sorted(set(required).difference(worker.capabilities))
        if missing:
            return RemoteWorkerAssignment(
                assignment_id=assignment_id,
                project_id=validated.project_id,
                run_id=validated.run_id,
                task_id=validated.task_id,
                worker_id=worker.worker_id,
                transport=worker.transport,
                host=worker.host,
                matched_capabilities=matched,
                missing_capabilities=missing,
                status="no_worker",
                requires_confirmation=True,
                required_permissions=[f"remote_worker:{worker.worker_id}"],
                budget_limit_sec=validated.budget_limit_sec,
                executable=False,
                notes=["Preferred worker does not satisfy all requested capabilities."],
            )
        permissions = [f"remote_worker:{worker.worker_id}"]
        if worker.transport == "ssh":
            permissions.append("external_network:ssh")
        return RemoteWorkerAssignment(
            assignment_id=assignment_id,
            project_id=validated.project_id,
            run_id=validated.run_id,
            task_id=validated.task_id,
            worker_id=worker.worker_id,
            transport=worker.transport,
            host=worker.host,
            matched_capabilities=matched,
            status="needs_confirmation",
            requires_confirmation=True,
            required_permissions=permissions,
            budget_limit_sec=validated.budget_limit_sec,
            executable=False,
            notes=["Remote worker assignment is a plan only; execution requires an explicit gate approval."],
        )

    def _select_worker(self, request: RemoteWorkerRequest) -> RemoteWorkerConfig | None:
        workers = self._read_workers(include_invalid=False)
        if request.preferred_worker_id:
            return next((worker for worker in workers if worker.worker_id == request.preferred_worker_id), None)
        required = set(request.required_capabilities)
        return next(
            (
                worker
                for worker in workers
                if worker.enabled and required.issubset(set(worker.capabilities))
            ),
            None,
        )

    def _read_workers(self, *, include_invalid: bool = False) -> list[RemoteWorkerConfig]:
        if not self.registry_path.exists():
            return []
        try:
            loaded = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        raw_workers = loaded.get("workers", []) if isinstance(loaded, dict) else []
        workers: list[RemoteWorkerConfig] = []
        for item in raw_workers:
            if not isinstance(item, dict):
                continue
            try:
                workers.append(RemoteWorkerConfig.model_validate(item))
            except (ValidationError, ValueError):
                if include_invalid:
                    raise
        return workers

    def _write_workers(self, workers: list[RemoteWorkerConfig]) -> Path:
        payload: dict[str, Any] = {
            "updated_at": now_iso(),
            "workers": [worker.model_dump(mode="json") for worker in sorted(workers, key=lambda item: item.worker_id)],
        }
        return write_json(self.registry_path, payload)

    def _safe_worker_id(self, worker_id: str) -> str:
        clean = str(worker_id or "").strip()
        worker_dir = (self.registry_dir / clean).resolve()
        if not worker_dir.is_relative_to(self.registry_dir):
            raise ValueError("worker_id escapes worker directory")
        return clean
