from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso
from ai4s_agent.schemas import AssetManifest, AssetPromotionRecord, AssetStatus, GateDecision, StageState


def _ensure_relative(parent: Path, child: Path, label: str) -> None:
    if not child.is_relative_to(parent):
        raise ValueError(f"{label} escapes base directory")


def _parse_version_name(name: str) -> int | None:
    if len(name) >= 4 and name.startswith("v") and name[1:].isdigit():
        return int(name[1:])
    return None


class ArtifactStore:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir.resolve()

    def _safe_path(self, run_id: str, filename: str) -> Path:
        run_path = (self.base_dir / run_id).resolve()
        artifact_path = (run_path / filename).resolve()
        if not run_path.is_relative_to(self.base_dir):
            raise ValueError("run_id escapes base_dir")
        if not artifact_path.is_relative_to(run_path):
            raise ValueError("artifact path escapes base_dir")
        return artifact_path

    def run_dir(self, run_id: str) -> Path:
        path = (self.base_dir / run_id).resolve()
        if not path.is_relative_to(self.base_dir):
            raise ValueError("run_id escapes base_dir")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_id: str, filename: str, payload: dict[str, Any]) -> Path:
        path = self._safe_path(run_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def read_json(self, run_id: str, filename: str) -> dict[str, Any]:
        path = self._safe_path(run_id, filename)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}


class ProjectStorage:
    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.projects_root = (self.workspace_dir / "projects").resolve()
        self.projects_root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        path = (self.projects_root / project_id).resolve()
        _ensure_relative(self.projects_root, path, "project_id")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(self, project_id: str, run_id: str) -> Path:
        project = self.project_dir(project_id)
        runs_base = (project / "runs").resolve()
        run_path = (runs_base / run_id).resolve()
        _ensure_relative(runs_base, run_path, "run_id")
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path

    def assets_dir(self, project_id: str) -> Path:
        project = self.project_dir(project_id)
        path = (project / "assets").resolve()
        _ensure_relative(project, path, "assets")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _asset_scope_dir(self, project_id: str, scope: list[str]) -> Path:
        base = self.assets_dir(project_id)
        path = base
        for part in scope:
            path = (path / part).resolve()
            _ensure_relative(base, path, "asset_scope")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_asset_version(self, project_id: str, scope: list[str]) -> str:
        scope_path = self._asset_scope_dir(project_id, scope)
        latest = 0
        for child in scope_path.iterdir():
            if not child.is_dir():
                continue
            parsed = _parse_version_name(child.name)
            if parsed is not None:
                latest = max(latest, parsed)
        return f"v{latest + 1:03d}"

    def create_asset_version_dir(self, project_id: str, scope: list[str]) -> Path:
        scope_path = self._asset_scope_dir(project_id, scope)
        for _ in range(1000):
            version = self.allocate_asset_version(project_id, scope)
            version_path = (scope_path / version).resolve()
            _ensure_relative(scope_path, version_path, "asset_version")
            try:
                version_path.mkdir(parents=False, exist_ok=False)
            except FileExistsError:
                continue
            return version_path
        raise RuntimeError("could not allocate unique asset version")

    def write_stage_state(self, project_id: str, run_id: str, state: StageState) -> Path:
        return self._write_json(
            self.run_dir(project_id, run_id),
            "stage.json",
            state.model_dump(mode="json"),
        )

    def read_stage_state(self, project_id: str, run_id: str) -> StageState | None:
        payload = self._read_json(self.run_dir(project_id, run_id), "stage.json")
        if not payload:
            return None
        return StageState.model_validate(payload)

    def register_artifact_path(
        self,
        project_id: str,
        run_id: str,
        artifact_id: str,
        relative_path: str,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)
        payload = self._read_json(run_path, "artifact_registry.json")
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        artifacts[artifact_id] = relative_path
        return self._write_json(run_path, "artifact_registry.json", {"artifacts": artifacts})

    def read_artifact_registry(self, project_id: str, run_id: str) -> dict[str, str]:
        payload = self._read_json(self.run_dir(project_id, run_id), "artifact_registry.json")
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            return {}
        return {str(k): str(v) for k, v in artifacts.items()}

    def write_asset_manifest(
        self,
        project_id: str,
        scope: list[str],
        version: str,
        manifest: AssetManifest,
    ) -> Path:
        scope_path = self._asset_scope_dir(project_id, scope)
        version_path = (scope_path / version).resolve()
        _ensure_relative(scope_path, version_path, "asset_manifest_version")
        version_path.mkdir(parents=True, exist_ok=True)
        return self._write_json(
            version_path,
            "asset_manifest.json",
            manifest.model_dump(mode="json"),
        )

    def append_asset_promotion_record(
        self,
        project_id: str,
        run_id: str,
        record: AssetPromotionRecord,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)
        payload = self._read_json(run_path, "asset_promotion_records.json")
        records = payload.get("records", [])
        if not isinstance(records, list):
            records = []
        records.append(record.model_dump(mode="json"))
        return self._write_json(run_path, "asset_promotion_records.json", {"records": records})

    def append_gate_decision(
        self,
        project_id: str,
        run_id: str,
        decision: GateDecision,
    ) -> Path:
        run_path = self.run_dir(project_id, run_id)
        payload = self._read_json(run_path, "gate_decisions.json")
        decisions = payload.get("decisions", [])
        if not isinstance(decisions, list):
            decisions = []
        decisions.append(decision.model_dump(mode="json"))
        return self._write_json(run_path, "gate_decisions.json", {"run_id": run_id, "decisions": decisions})

    def read_gate_decisions(self, project_id: str, run_id: str) -> list[dict[str, Any]]:
        payload = self._read_json(self.run_dir(project_id, run_id), "gate_decisions.json")
        decisions = payload.get("decisions", [])
        return [item for item in decisions if isinstance(item, dict)] if isinstance(decisions, list) else []

    def register_model_asset(
        self,
        project_id: str,
        run_id: str,
        model_dir: Path,
        *,
        property_id: str,
        backend: str,
        content_hash: str,
        approved_by: str = "",
        approval_note: str = "",
    ) -> tuple[AssetManifest, Path]:
        import shutil

        actor = str(approved_by or "").strip()
        if not actor:
            raise ValueError("model registration requires user confirmation")
        model_dir = model_dir.expanduser().resolve()
        if not model_dir.exists() or not model_dir.is_dir():
            raise FileNotFoundError(f"model_dir not found: {model_dir}")
        run_path = self.run_dir(project_id, run_id)
        _ensure_relative(run_path, model_dir, "model_dir")

        scope = ["models", backend, property_id]
        version_dir = self.create_asset_version_dir(project_id, scope)
        version = version_dir.name

        dest_dir = version_dir / "model"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for src in model_dir.iterdir():
            if src.is_symlink():
                raise ValueError(f"model_dir contains symlink: {src}")
            if src.is_file():
                shutil.copy2(src, dest_dir / src.name, follow_symlinks=False)
            elif src.is_dir():
                for nested in src.rglob("*"):
                    if nested.is_symlink():
                        raise ValueError(f"model_dir contains symlink: {nested}")
                shutil.copytree(src, dest_dir / src.name, symlinks=False)

        manifest = AssetManifest(
            asset_id=f"model/{backend}/{property_id}",
            asset_type="trained_model",
            version=version,
            status=AssetStatus.CANDIDATE,
            created_from_run_id=run_id,
            source_artifacts=[str(model_dir)],
            content_hash=content_hash,
        )
        self.write_asset_manifest(project_id, scope, version, manifest)
        self._write_json(
            version_dir,
            "model_registration_record.json",
            {
                "run_id": run_id,
                "asset_id": manifest.asset_id,
                "asset_type": manifest.asset_type,
                "version": manifest.version,
                "approved_by": actor,
                "approved_at": now_iso(),
                "note": str(approval_note or ""),
            },
        )
        return manifest, version_dir

    def _write_json(self, base_path: Path, filename: str, payload: dict[str, Any]) -> Path:
        path = (base_path / filename).resolve()
        _ensure_relative(base_path, path, "json_path")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _read_json(self, base_path: Path, filename: str) -> dict[str, Any]:
        path = (base_path / filename).resolve()
        _ensure_relative(base_path, path, "json_path")
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}
