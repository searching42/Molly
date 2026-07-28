from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json as atomic_write_json
from ai4s_agent.schemas import (
    AssetManifest,
    AssetPromotionRecord,
    AssetStatus,
    GateDecision,
    PromotedModelAsset,
    StageState,
)


def _ensure_relative(parent: Path, child: Path, label: str) -> None:
    if not child.is_relative_to(parent):
        raise ValueError(f"{label} escapes base directory")


def _parse_version_name(name: str) -> int | None:
    if len(name) >= 4 and name.startswith("v") and name[1:].isdigit():
        return int(name[1:])
    return None


def _normalize_token(value: str | None) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        clean = str(item or "").strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def _clean_string_dict(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()
        clean_value = str(raw or "").strip()
        if clean_key and clean_value:
            result[clean_key] = clean_value
    return result


def _clean_numeric_dict(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            continue
        result[str(key)] = number
    return result


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
        return atomic_write_json(path, payload)

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

    def register_new_artifact_registry_paths(
        self,
        project_id: str,
        run_id: str,
        artifacts: dict[str, str],
    ) -> Path:
        """Atomically add a new immutable group of artifact-path mappings.

        The absent-key check and update occur under the registry lock.  This
        preserves unrelated entries written concurrently and prevents another
        executor from replacing a published immutable artifact group.
        """

        from ai4s_agent.json_rmw_lock import locked_storage_json_update

        clean = {str(key): str(value) for key, value in artifacts.items()}
        run_path = self.run_dir(project_id, run_id)

        def add_new(payload: dict[str, Any]) -> dict[str, Any]:
            existing = payload.get("artifacts", {})
            if not isinstance(existing, dict):
                existing = {}
            conflicts = sorted(key for key in clean if key in existing)
            if conflicts:
                raise ValueError(
                    "artifact registry paths are already immutable: " + ", ".join(conflicts)
                )
            updated = {str(key): str(value) for key, value in existing.items()}
            updated.update(clean)
            return {"artifacts": updated}

        return locked_storage_json_update(
            self,
            run_path,
            "artifact_registry.json",
            add_new,
        )

    def remove_artifact_registry_paths_if_all_equal(
        self,
        project_id: str,
        run_id: str,
        artifacts: dict[str, str],
    ) -> Path:
        """Remove an immutable group only if every mapping still matches.

        This is the compensating action if an immutable publication changes
        after its registry insertion.  It intentionally leaves both unrelated
        concurrent updates and a concurrently changed group intact, avoiding
        a partially removed publication group.
        """

        from ai4s_agent.json_rmw_lock import locked_storage_json_update

        clean = {str(key): str(value) for key, value in artifacts.items()}
        run_path = self.run_dir(project_id, run_id)

        def remove_if_all_equal(payload: dict[str, Any]) -> dict[str, Any]:
            existing = payload.get("artifacts", {})
            if not isinstance(existing, dict):
                existing = {}
            updated = {str(key): str(value) for key, value in existing.items()}
            if not all(updated.get(key) == value for key, value in clean.items()):
                return {"artifacts": updated}
            for key in clean:
                del updated[key]
            return {"artifacts": updated}

        return locked_storage_json_update(
            self,
            run_path,
            "artifact_registry.json",
            remove_if_all_equal,
        )

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

    def build_promoted_model_asset_draft(
        self,
        project_id: str,
        version_dir: Path,
        *,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version_path, model_path, manifest = self._registered_model_asset_context(project_id, version_dir)
        backend, property_id = self._registered_model_terms(manifest)
        model_payloads = self._read_model_asset_payloads(model_path)
        merged: dict[str, Any] = {}
        merged_metrics: dict[str, float] = {}
        for payload in model_payloads:
            merged_metrics.update(_clean_numeric_dict(payload.get("metrics")))
            merged.update(payload)

        domain = str(merged.get("domain") or "oled").strip() or "oled"
        use_case = str(merged.get("use_case") or merged.get("intended_use") or "").strip() or "scalar_prediction"
        model_id = str(merged.get("model_id") or "").strip() or f"{property_id}_{manifest.version}"
        metrics = merged_metrics
        applicability = merged.get("applicability") if isinstance(merged.get("applicability"), dict) else {}
        applicability = {str(key): value for key, value in applicability.items()}
        for key in ("train_size", "split", "dataset", "dataset_id", "target_transform", "objective_type"):
            if key in merged and key not in applicability:
                applicability[key] = merged[key]
        feature_requirements = (
            _clean_string_list(merged.get("feature_requirements"))
            or _clean_string_list(merged.get("required_inputs"))
            or list(_clean_string_dict(merged.get("input_columns")).keys())
        )
        input_columns = _clean_string_dict(merged.get("input_columns"))
        limitations = _clean_string_list(merged.get("limitations"))
        warnings: list[str] = []
        if not model_payloads:
            warnings.append("no_model_metadata_found")
        if not input_columns and feature_requirements:
            warnings.append("input_columns_not_found")

        draft: dict[str, Any] = {
            "version_dir": str(version_path),
            "asset_id": f"{manifest.asset_id}/{manifest.version}",
            "model_id": model_id,
            "domain": domain,
            "property_id": property_id,
            "use_case": use_case,
            "backend": backend,
            "model_dir": str(model_path),
            "metrics": metrics,
            "applicability": applicability,
            "feature_requirements": feature_requirements,
            "input_columns": input_columns,
            "limitations": limitations,
            "rollback_asset_id": str((overrides or {}).get("rollback_asset_id") or ""),
            "warnings": warnings,
        }
        for key, value in (overrides or {}).items():
            if key == "metrics":
                if isinstance(value, dict):
                    draft[key] = _clean_numeric_dict(value)
                continue
            if key == "input_columns":
                if isinstance(value, dict):
                    draft[key] = _clean_string_dict(value)
                continue
            if key == "applicability":
                if isinstance(value, dict):
                    draft[key] = {str(item_key): item_value for item_key, item_value in value.items()}
                continue
            if key in {"feature_requirements", "limitations"}:
                draft[key] = _clean_string_list(value)
                continue
            if key in {"backend", "property_id"}:
                continue
            if key in draft and value not in (None, ""):
                draft[key] = value
        return draft

    def promote_registered_model_asset(
        self,
        project_id: str,
        run_id: str,
        version_dir: Path,
        *,
        model_id: str,
        domain: str,
        property_id: str,
        use_case: str,
        backend: str,
        approved_by: str,
        metrics: dict[str, float] | None = None,
        applicability: dict[str, Any] | None = None,
        feature_requirements: list[str] | None = None,
        input_columns: dict[str, str] | None = None,
        limitations: list[str] | None = None,
        rollback_asset_id: str = "",
        note: str = "",
    ) -> tuple[PromotedModelAsset, Path]:
        actor = str(approved_by or "").strip()
        if not actor:
            raise ValueError("model promotion requires user confirmation")
        version_path, model_path, manifest = self._registered_model_asset_context(project_id, version_dir)
        registered_backend, registered_property = self._registered_model_terms(manifest)
        if registered_backend != backend or registered_property != property_id:
            raise ValueError(
                "model promotion metadata must match registered model asset "
                f"{manifest.asset_id}"
            )
        expected_asset_id = f"model/{backend}/{property_id}"
        if manifest.asset_id != expected_asset_id:
            raise ValueError(
                "model promotion metadata must match registered model asset "
                f"{manifest.asset_id}"
            )
        promoted_at = now_iso()
        asset = PromotedModelAsset(
            asset_id=f"{manifest.asset_id}/{manifest.version}",
            model_id=model_id,
            domain=domain,
            property_id=property_id,
            use_case=use_case,
            backend=backend,
            model_dir=str(model_path),
            status=AssetStatus.CONFIRMED,
            created_from_run_id=manifest.created_from_run_id or run_id,
            source_artifacts=manifest.source_artifacts,
            approved_by=actor,
            approved_at=promoted_at,
            metrics=metrics or {},
            applicability=applicability or {},
            feature_requirements=feature_requirements or [],
            input_columns=input_columns or {},
            limitations=limitations or [],
            rollback_asset_id=rollback_asset_id,
        )
        promoted_path = self._write_json(
            version_path,
            "promoted_model_asset.json",
            asset.model_dump(mode="json"),
        )
        confirmed_manifest = manifest.model_copy(update={"status": AssetStatus.CONFIRMED})
        self._write_json(version_path, "asset_manifest.json", confirmed_manifest.model_dump(mode="json"))
        self.append_asset_promotion_record(
            project_id,
            run_id,
            AssetPromotionRecord(
                run_id=run_id,
                asset_id=asset.asset_id,
                asset_type="promoted_model_asset",
                version=manifest.version,
                source_artifacts=asset.source_artifacts,
                approved_by=actor,
                approved_at=promoted_at,
                note=str(note or ""),
            ),
        )
        return asset, promoted_path

    def list_promoted_model_assets(
        self,
        project_id: str,
        *,
        domain: str | None = None,
        property_id: str | None = None,
        use_case: str | None = None,
    ) -> list[PromotedModelAsset]:
        requested_domain = _normalize_token(domain)
        requested_property = _normalize_token(property_id)
        requested_use_case = _normalize_token(use_case)
        assets: list[PromotedModelAsset] = []
        for path in sorted(self.assets_dir(project_id).rglob("promoted_model_asset.json")):
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                continue
            asset = PromotedModelAsset.model_validate(loaded)
            if requested_domain and _normalize_token(asset.domain) != requested_domain:
                continue
            if requested_property and _normalize_token(asset.property_id) != requested_property:
                continue
            if requested_use_case and _normalize_token(asset.use_case) != requested_use_case:
                continue
            assets.append(asset)
        return sorted(assets, key=lambda item: (item.approved_at, item.asset_id), reverse=True)

    def _registered_model_asset_context(
        self,
        project_id: str,
        version_dir: Path,
    ) -> tuple[Path, Path, AssetManifest]:
        asset_root = self.assets_dir(project_id)
        version_path = version_dir.expanduser().resolve()
        _ensure_relative(asset_root, version_path, "model_asset_version")
        if not version_path.exists() or not version_path.is_dir():
            raise FileNotFoundError(f"model asset version not found: {version_path}")
        model_path = (version_path / "model").resolve()
        _ensure_relative(version_path, model_path, "promoted_model_dir")
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(f"model payload directory not found: {model_path}")
        manifest_payload = self._read_json(version_path, "asset_manifest.json")
        if not manifest_payload:
            raise FileNotFoundError(f"asset_manifest.json not found: {version_path}")
        manifest = AssetManifest.model_validate(manifest_payload)
        if manifest.asset_type != "trained_model":
            raise ValueError("model promotion requires an asset_manifest with asset_type trained_model")
        return version_path, model_path, manifest

    @staticmethod
    def _registered_model_terms(manifest: AssetManifest) -> tuple[str, str]:
        parts = manifest.asset_id.split("/")
        if len(parts) != 3 or parts[0] != "model":
            raise ValueError(f"invalid registered model asset_id: {manifest.asset_id}")
        return parts[1], parts[2]

    @staticmethod
    def _read_model_asset_payloads(model_path: Path) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for name in ("model_metadata.json", "model_manifest.json", "domain_model_manifest.json"):
            path = model_path / name
            if not path.exists():
                continue
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if isinstance(loaded, dict):
                payloads.append(loaded)
        return payloads

    def _write_json(self, base_path: Path, filename: str, payload: dict[str, Any]) -> Path:
        path = (base_path / filename).resolve()
        _ensure_relative(base_path, path, "json_path")
        return atomic_write_json(path, payload)

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
