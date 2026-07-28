from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import ProjectMemoryRecord
from ai4s_agent.storage import ProjectStorage


class RunMemoryEntry(BaseModel):
    run_id: str
    stage: str
    artifact_id: str
    artifact_path: str
    collected_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectMemoryManifest(BaseModel):
    project_id: str
    updated_at: str
    runs: list[str] = Field(default_factory=list)
    artifacts: list[RunMemoryEntry] = Field(default_factory=list)
    total_runs: int = 0


class ProjectMemory:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.storage = ProjectStorage(workspace_dir=self.workspace_dir)
        self.projects_dir = self.storage.projects_root
        self.memory_dir = (self.workspace_dir / "memory").resolve()
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def collect_run_artifacts(
        self,
        project_id: str,
        run_id: str,
        *,
        artifact_ids: list[str] | None = None,
        require_confirmation: bool = False,
    ) -> list[RunMemoryEntry]:
        run_dir = self._run_dir(project_id, run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"run not found: {run_id}")

        entries: list[RunMemoryEntry] = []
        collected_at = now_iso()
        registry_path = run_dir / "artifact_registry.json"
        registry: dict[str, str] = {}
        if registry_path.exists():
            try:
                loaded = json.loads(registry_path.read_text(encoding="utf-8"))
                registry = loaded.get("artifacts", {}) if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                pass

        target_ids = set(artifact_ids) if artifact_ids else set(registry.keys())
        for artifact_id in sorted(target_ids):
            relative_path = registry.get(artifact_id, "")
            artifact_path = ""
            if relative_path:
                candidate = (run_dir / relative_path).resolve()
                if candidate.is_relative_to(run_dir) and candidate.exists():
                    artifact_path = str(candidate)
            entry = RunMemoryEntry(
                run_id=run_id,
                stage=artifact_id,
                artifact_id=artifact_id,
                artifact_path=artifact_path,
                collected_at=collected_at,
                metadata={"confirmed": not require_confirmation},
            )
            entries.append(entry)

        if entries:
            memory_path = self._memory_path(project_id, f"{run_id}_memory.json")
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(memory_path, {
                "run_id": run_id,
                "collected_at": collected_at,
                "entries": [entry.model_dump(mode="json") for entry in entries],
            })

        self._update_manifest(project_id, entries)
        return entries

    def get_project_memory(self, project_id: str) -> list[RunMemoryEntry]:
        manifest_path = self._memory_path(project_id, "memory_manifest.json")
        if not manifest_path.exists():
            return []
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = loaded.get("artifacts", []) if isinstance(loaded, dict) else []
        except json.JSONDecodeError:
            return []
        return [
            RunMemoryEntry(
                run_id=str(item.get("run_id", "")),
                stage=str(item.get("stage", "")),
                artifact_id=str(item.get("artifact_id", "")),
                artifact_path=str(item.get("artifact_path", "")),
                collected_at=str(item.get("collected_at", "")),
                metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
            )
            for item in artifacts
            if isinstance(item, dict)
        ]

    def confirm_memory_entry(self, project_id: str, artifact_id: str, *, run_id: str) -> bool:
        entries = self.get_project_memory(project_id)
        updated = False
        for entry in entries:
            if entry.run_id == run_id and entry.artifact_id == artifact_id:
                entry.metadata["confirmed"] = True
                updated = True
        if updated:
            self._write_entries(project_id, entries)
        return updated

    def save_project_record(self, project_id: str, record: ProjectMemoryRecord) -> ProjectMemoryRecord:
        records = self.list_project_records(project_id, include_disabled_project=True, include_disabled_records=True)
        updated = ProjectMemoryRecord.model_validate(record.model_dump(mode="json") | {"updated_at": now_iso()})
        kept = [item for item in records if item.record_id != updated.record_id]
        kept.append(updated)
        self._write_project_records(project_id, kept)
        return updated

    def list_project_records(
        self,
        project_id: str,
        *,
        include_disabled_project: bool = False,
        include_disabled_records: bool = False,
    ) -> list[ProjectMemoryRecord]:
        if not include_disabled_project and not self.project_memory_enabled(project_id):
            return []
        records = self._read_project_records(project_id)
        if include_disabled_records:
            return records
        return [record for record in records if not record.disabled]

    def update_project_record(
        self,
        project_id: str,
        record_id: str,
        updates: dict[str, Any],
    ) -> ProjectMemoryRecord | None:
        clean_id = str(record_id or "").strip()
        records = self.list_project_records(project_id, include_disabled_project=True, include_disabled_records=True)
        changed: ProjectMemoryRecord | None = None
        updated_records: list[ProjectMemoryRecord] = []
        blocked = {"record_id", "created_at"}
        safe_updates = {str(key): value for key, value in updates.items() if str(key) not in blocked}
        safe_updates["updated_at"] = now_iso()
        for record in records:
            if record.record_id == clean_id:
                changed = ProjectMemoryRecord.model_validate(record.model_dump(mode="json") | safe_updates)
                updated_records.append(changed)
            else:
                updated_records.append(record)
        if changed is None:
            return None
        self._write_project_records(project_id, updated_records)
        return changed

    def delete_project_record(self, project_id: str, record_id: str) -> bool:
        clean_id = str(record_id or "").strip()
        records = self.list_project_records(project_id, include_disabled_project=True, include_disabled_records=True)
        kept = [record for record in records if record.record_id != clean_id]
        if len(kept) == len(records):
            return False
        self._write_project_records(project_id, kept)
        return True

    def export_project_records(self, project_id: str) -> dict[str, Any]:
        records = self.list_project_records(project_id, include_disabled_project=True, include_disabled_records=True)
        return {
            "project_id": project_id,
            "enabled": self.project_memory_enabled(project_id),
            "exported_at": now_iso(),
            "records": [record.model_dump(mode="json") for record in records],
        }

    def set_project_memory_enabled(self, project_id: str, enabled: bool) -> Path:
        return write_json(
            self._memory_path(project_id, "project_memory_policy.json"),
            {"project_id": project_id, "enabled": bool(enabled), "updated_at": now_iso()},
        )

    def project_memory_enabled(self, project_id: str) -> bool:
        policy_path = self._memory_path(project_id, "project_memory_policy.json")
        if not policy_path.exists():
            return True
        try:
            loaded = json.loads(policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return True
        if not isinstance(loaded, dict):
            return True
        return bool(loaded.get("enabled", True))

    def _update_manifest(self, project_id: str, entries: list[RunMemoryEntry]) -> None:
        existing = self.get_project_memory(project_id)
        existing_runs = {entry.run_id for entry in existing}
        existing_keys = {(entry.run_id, entry.artifact_id) for entry in existing}
        for entry in entries:
            if (entry.run_id, entry.artifact_id) not in existing_keys:
                existing.append(entry)
                existing_keys.add((entry.run_id, entry.artifact_id))
            existing_runs.add(entry.run_id)
        self._write_entries(project_id, existing)

    def _write_entries(self, project_id: str, entries: list[RunMemoryEntry]) -> None:
        runs = sorted({entry.run_id for entry in entries})
        manifest = ProjectMemoryManifest(
            project_id=project_id,
            updated_at=now_iso(),
            runs=runs,
            artifacts=entries,
            total_runs=len(runs),
        )
        manifest_path = self._memory_path(project_id, "memory_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest.model_dump(mode="json"))

    def _read_project_records(self, project_id: str) -> list[ProjectMemoryRecord]:
        path = self._memory_path(project_id, "project_memory_records.json")
        if not path.exists():
            return []
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        records = loaded.get("records", []) if isinstance(loaded, dict) else []
        result: list[ProjectMemoryRecord] = []
        for item in records:
            if not isinstance(item, dict):
                continue
            try:
                result.append(ProjectMemoryRecord.model_validate(item))
            except ValueError:
                continue
        return result

    def _write_project_records(self, project_id: str, records: list[ProjectMemoryRecord]) -> Path:
        return write_json(
            self._memory_path(project_id, "project_memory_records.json"),
            {
                "project_id": project_id,
                "updated_at": now_iso(),
                "records": [record.model_dump(mode="json") for record in records],
            },
        )

    def _run_dir(self, project_id: str, run_id: str) -> Path:
        project_dir = self.storage.project_dir(project_id)
        runs_dir = (project_dir / "runs").resolve()
        run_dir = (runs_dir / run_id).resolve()
        if not run_dir.is_relative_to(runs_dir):
            raise ValueError("run_id escapes project runs directory")
        return run_dir

    def _memory_path(self, project_id: str, filename: str) -> Path:
        project_memory_dir = (self.memory_dir / project_id).resolve()
        if not project_memory_dir.is_relative_to(self.memory_dir):
            raise ValueError("project_id escapes memory directory")
        path = (project_memory_dir / filename).resolve()
        if not path.is_relative_to(project_memory_dir):
            raise ValueError("memory filename escapes project memory directory")
        return path


class PermissionLevel(str, Enum):
    AUTO = "auto"
    PROJECT_APPROVED = "project-approved"
    CONFIRM_EACH_TIME = "confirm-each-time"


class PermissionDecision(BaseModel):
    action: str
    level: PermissionLevel
    allowed: bool
    reason: str
    project_id: str = ""
    run_id: str = ""
    actor: str = ""


class PermissionPolicy:
    def __init__(self, *, require_actor_for_project_approved: bool = False) -> None:
        self._defaults: dict[str, PermissionLevel] = {
            "train_model": PermissionLevel.CONFIRM_EACH_TIME,
            "predict_candidates": PermissionLevel.PROJECT_APPROVED,
            "filter_rank": PermissionLevel.AUTO,
            "render_report": PermissionLevel.AUTO,
            "generate_candidates_expensive": PermissionLevel.CONFIRM_EACH_TIME,
            "upload_dataset": PermissionLevel.PROJECT_APPROVED,
            "promote_asset": PermissionLevel.CONFIRM_EACH_TIME,
            "register_model": PermissionLevel.CONFIRM_EACH_TIME,
            "inspect_dataset": PermissionLevel.AUTO,
            "clean_dataset": PermissionLevel.AUTO,
        }
        self._dataset_public: dict[str, bool] = {}
        self.require_actor_for_project_approved = require_actor_for_project_approved

    def resolve(self, action: str) -> PermissionLevel:
        return self._defaults.get(action, PermissionLevel.CONFIRM_EACH_TIME)

    def decide(
        self,
        action: str,
        *,
        project_id: str = "",
        run_id: str = "",
        project_approved: bool = False,
        confirmed: bool = False,
        actor: str = "",
    ) -> PermissionDecision:
        level = self.resolve(action)
        clean_actor = str(actor or "").strip()
        if level == PermissionLevel.AUTO:
            return PermissionDecision(
                action=action,
                level=level,
                allowed=True,
                reason="AUTO_ALLOWED",
                project_id=project_id,
                run_id=run_id,
                actor=clean_actor,
            )
        if level == PermissionLevel.PROJECT_APPROVED:
            if project_approved and self.require_actor_for_project_approved and not clean_actor:
                return PermissionDecision(
                    action=action,
                    level=level,
                    allowed=False,
                    reason="PROJECT_APPROVAL_ACTOR_REQUIRED",
                    project_id=project_id,
                    run_id=run_id,
                    actor=clean_actor,
                )
            return PermissionDecision(
                action=action,
                level=level,
                allowed=bool(project_approved),
                reason="PROJECT_APPROVED" if project_approved else "PROJECT_APPROVAL_REQUIRED",
                project_id=project_id,
                run_id=run_id,
                actor=clean_actor,
            )
        if not confirmed:
            return PermissionDecision(
                action=action,
                level=level,
                allowed=False,
                reason="CONFIRMATION_REQUIRED",
                project_id=project_id,
                run_id=run_id,
                actor=clean_actor,
            )
        if not clean_actor:
            return PermissionDecision(
                action=action,
                level=level,
                allowed=False,
                reason="CONFIRMATION_ACTOR_REQUIRED",
                project_id=project_id,
                run_id=run_id,
                actor=clean_actor,
            )
        return PermissionDecision(
            action=action,
            level=level,
            allowed=True,
            reason="CONFIRMED",
            project_id=project_id,
            run_id=run_id,
            actor=clean_actor,
        )

    def set_policy(self, action: str, level: PermissionLevel) -> None:
        self._defaults[action] = level

    def set_dataset_public(self, dataset_path: str, public: bool = True) -> None:
        self._dataset_public[dataset_path] = public

    def is_dataset_public(self, dataset_path: str) -> bool:
        return self._dataset_public.get(dataset_path, False)

    def external_llm_context_allowed(self, dataset_path: str) -> bool:
        return self.is_dataset_public(dataset_path)
