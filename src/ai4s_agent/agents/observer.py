from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso
from ai4s_agent.job_manager import JobManager
from ai4s_agent.schemas import AssetManifest, ObservedArtifact, RunObservation
from ai4s_agent.storage import ProjectStorage


class ObserverAgent:
    """Collects audited run state without executing adapters."""

    def __init__(self, *, storage: ProjectStorage, jobs: JobManager | None = None) -> None:
        self.storage = storage
        self.jobs = jobs

    def observe_run(self, project_id: str, run_id: str) -> RunObservation:
        run_dir = self.storage.run_dir(project_id, run_id)
        stage_state = self.storage.read_stage_state(project_id, run_id)
        artifacts = self._observe_artifacts(project_id, run_id, run_dir)
        notes: list[str] = []
        if stage_state is None:
            notes.append("No stage.json found for this run.")
        return RunObservation(
            project_id=project_id,
            run_id=run_id,
            generated_at=now_iso(),
            stage_state=stage_state,
            artifacts=artifacts,
            logs=self._read_logs(run_id, run_dir),
            reports=self._collect_reports(run_dir),
            asset_manifests=self._collect_asset_manifests(project_id, run_id),
            approval_records=self._collect_approval_records(run_dir),
            notes=notes,
        )

    def _observe_artifacts(self, project_id: str, run_id: str, run_dir: Path) -> list[ObservedArtifact]:
        by_id: dict[str, ObservedArtifact] = {}
        stage_state = self.storage.read_stage_state(project_id, run_id)
        if stage_state is not None:
            for artifact in stage_state.artifacts:
                by_id[artifact.artifact_id] = self._artifact_ref(
                    run_dir,
                    artifact.artifact_id,
                    artifact.relative_path,
                    artifact.producer_task_id,
                )
        for artifact_id, relative_path in self.storage.read_artifact_registry(project_id, run_id).items():
            by_id.setdefault(artifact_id, self._artifact_ref(run_dir, artifact_id, relative_path, None))
        return list(by_id.values())

    def _artifact_ref(
        self,
        run_dir: Path,
        artifact_id: str,
        relative_path: str,
        producer_task_id: str | None,
    ) -> ObservedArtifact:
        path = (run_dir / relative_path).resolve()
        inside_run_dir = path.is_relative_to(run_dir)
        size_bytes = 0
        try:
            exists = inside_run_dir and path.exists() and path.is_file()
            size_bytes = path.stat().st_size if exists else 0
        except OSError:
            exists = False
        return ObservedArtifact(
            artifact_id=artifact_id,
            relative_path=relative_path,
            exists=exists,
            size_bytes=size_bytes,
            producer_task_id=producer_task_id,
        )

    def _read_logs(self, run_id: str, run_dir: Path) -> list[dict[str, str]]:
        if self.jobs is not None:
            return self.jobs.get_logs(run_id, limit=200)
        log_path = run_dir / "job_log.jsonl"
        if not log_path.exists():
            return []
        entries: list[dict[str, str]] = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append({key: str(payload.get(key, "")) for key in ("ts", "level", "source", "message")})
        return entries

    def _collect_reports(self, run_dir: Path) -> dict[str, dict[str, Any]]:
        reports: dict[str, dict[str, Any]] = {}
        report_sources: dict[str, str] = {}
        skipped = {
            "asset_promotion_records",
            "artifact_registry",
            "gate_decisions",
            "stage",
            "verification_report",
        }
        for path in sorted(run_dir.rglob("*.json")):
            if not path.is_file():
                continue
            key = self._report_key(path.stem)
            if key in skipped:
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                self._store_report(reports, report_sources, key, path.stem, loaded)
        return reports

    @staticmethod
    def _report_key(stem: str) -> str:
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0].startswith("run-"):
            return parts[1]
        return stem

    @staticmethod
    def _store_report(
        reports: dict[str, dict[str, Any]],
        report_sources: dict[str, str],
        key: str,
        stem: str,
        loaded: dict[str, Any],
    ) -> None:
        if key not in reports:
            reports[key] = loaded
            report_sources[key] = stem
            return
        existing_stem = report_sources.get(key, key)
        if stem == key and existing_stem != key:
            reports[ObserverAgent._unique_report_key(reports, existing_stem)] = reports[key]
            reports[key] = loaded
            report_sources[key] = stem
            return
        reports[ObserverAgent._unique_report_key(reports, stem)] = loaded

    @staticmethod
    def _unique_report_key(reports: dict[str, dict[str, Any]], preferred: str) -> str:
        if preferred not in reports:
            return preferred
        index = 2
        while f"{preferred}__{index}" in reports:
            index += 1
        return f"{preferred}__{index}"

    def _collect_asset_manifests(self, project_id: str, run_id: str) -> list[AssetManifest]:
        assets_root = self.storage.assets_dir(project_id)
        manifests: list[AssetManifest] = []
        for path in sorted(assets_root.rglob("asset_manifest.json")):
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                manifest = AssetManifest.model_validate(loaded)
            except Exception:
                continue
            if manifest.created_from_run_id == run_id:
                manifests.append(manifest)
        return manifests

    def _collect_approval_records(self, run_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        gate_payload = self._read_json_object(run_dir / "gate_decisions.json")
        decisions = gate_payload.get("decisions", [])
        if isinstance(decisions, list):
            for decision in decisions:
                if isinstance(decision, dict) and bool(decision.get("approved")):
                    records.append({**decision, "approval_type": "gate", "source_file": "gate_decisions.json"})

        promotion_payload = self._read_json_object(run_dir / "asset_promotion_records.json")
        promotions = promotion_payload.get("records", [])
        if isinstance(promotions, list):
            for record in promotions:
                if isinstance(record, dict):
                    records.append(
                        {
                            **record,
                            "approval_type": "asset_promotion",
                            "source_file": "asset_promotion_records.json",
                        }
                    )
        return records

    @staticmethod
    def _read_json_object(path: Path) -> dict[str, Any]:
        if not path.exists() or not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
