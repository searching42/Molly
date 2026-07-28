from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from pydantic import ValidationError

from ai4s_agent._utils import now_iso
from ai4s_agent.memory import PermissionPolicy
from ai4s_agent.schemas import AssetPromotionRecord
from ai4s_agent.storage import ProjectStorage


def register_project_asset_routes(app: Flask, *, projects: ProjectStorage, permissions: PermissionPolicy) -> None:
    @app.post("/api/projects/<project_id>/runs/<run_id>/models/register")
    def register_model(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        actor = str(payload.get("approved_by") or payload.get("actor") or "").strip()
        decision = permissions.decide(
            "register_model",
            project_id=clean_project_id,
            run_id=clean_run_id,
            confirmed=_as_bool(payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "model registration requires per-action confirmation",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403

        model_dir_raw = str(payload.get("model_dir") or "").strip()
        property_id = str(payload.get("property_id") or "").strip()
        backend = str(payload.get("backend") or "").strip()
        content_hash = str(payload.get("content_hash") or "").strip()
        if not model_dir_raw or not property_id or not backend or not content_hash:
            return jsonify(
                {
                    "ok": False,
                    "error": "model_dir, property_id, backend, and content_hash required",
                }
            ), 400
        try:
            manifest, version_dir = projects.register_model_asset(
                clean_project_id,
                clean_run_id,
                Path(model_dir_raw),
                property_id=property_id,
                backend=backend,
                content_hash=content_hash,
                approved_by=actor,
                approval_note=str(payload.get("approval_note") or ""),
            )
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "manifest": manifest.model_dump(mode="json"),
                "version_dir": str(version_dir),
                "permission": decision.model_dump(mode="json"),
            }
        )

    @app.post("/api/projects/<project_id>/runs/<run_id>/models/promote/draft")
    def draft_promoted_model_asset(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        version_dir_raw = str(payload.get("version_dir") or "").strip()
        if not version_dir_raw:
            return jsonify({"ok": False, "error": "version_dir required"}), 400
        try:
            draft = projects.build_promoted_model_asset_draft(
                clean_project_id,
                Path(version_dir_raw),
                overrides=_promotion_draft_overrides(payload),
            )
        except (ValueError, FileNotFoundError, ValidationError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "draft": draft})

    @app.post("/api/projects/<project_id>/runs/<run_id>/models/promote")
    def promote_model_asset(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        actor = str(payload.get("approved_by") or payload.get("actor") or "").strip()
        decision = permissions.decide(
            "promote_asset",
            project_id=clean_project_id,
            run_id=clean_run_id,
            confirmed=_as_bool(payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "model asset promotion requires per-action confirmation",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403

        version_dir_raw = str(payload.get("version_dir") or "").strip()
        model_id = str(payload.get("model_id") or "").strip()
        domain = str(payload.get("domain") or "").strip()
        property_id = str(payload.get("property_id") or "").strip()
        use_case = str(payload.get("use_case") or "").strip()
        backend = str(payload.get("backend") or "").strip()
        if not version_dir_raw or not model_id or not domain or not property_id or not use_case or not backend:
            return jsonify(
                {
                    "ok": False,
                    "error": "version_dir, model_id, domain, property_id, use_case, and backend required",
                }
            ), 400
        try:
            metrics = _object_field(payload.get("metrics"), "metrics")
            applicability = _object_field(payload.get("applicability"), "applicability")
            input_columns = _string_dict_field(payload.get("input_columns"), "input_columns")
            promoted, promoted_path = projects.promote_registered_model_asset(
                clean_project_id,
                clean_run_id,
                Path(version_dir_raw),
                model_id=model_id,
                domain=domain,
                property_id=property_id,
                use_case=use_case,
                backend=backend,
                approved_by=actor,
                metrics=metrics,
                applicability=applicability,
                feature_requirements=_string_list(payload.get("feature_requirements")),
                input_columns=input_columns,
                limitations=_string_list(payload.get("limitations")),
                rollback_asset_id=str(payload.get("rollback_asset_id") or "").strip(),
                note=str(payload.get("note") or ""),
            )
        except (ValueError, FileNotFoundError, ValidationError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "promoted_model_asset": promoted.model_dump(mode="json"),
                "promoted_model_asset_path": str(promoted_path),
                "permission": decision.model_dump(mode="json"),
            }
        )

    @app.post("/api/projects/<project_id>/runs/<run_id>/assets/promote")
    def promote_asset(project_id: str, run_id: str):
        clean_project_id = str(project_id or "").strip()
        clean_run_id = str(run_id or "").strip()
        if not clean_project_id or not clean_run_id:
            return jsonify({"ok": False, "error": "project_id and run_id required"}), 400
        payload = request.get_json(silent=True) or {}
        actor = str(payload.get("approved_by") or payload.get("actor") or "").strip()
        decision = permissions.decide(
            "promote_asset",
            project_id=clean_project_id,
            run_id=clean_run_id,
            confirmed=_as_bool(payload.get("confirmed")),
            actor=actor,
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "asset promotion requires per-action confirmation",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403

        source_artifacts = _string_list(payload.get("source_artifacts"))
        asset_id = str(payload.get("asset_id") or "").strip()
        asset_type = str(payload.get("asset_type") or "").strip()
        version = str(payload.get("version") or "").strip()
        if not asset_id or not asset_type or not version:
            return jsonify({"ok": False, "error": "asset_id, asset_type, and version required"}), 400
        record = AssetPromotionRecord(
            run_id=clean_run_id,
            asset_id=asset_id,
            asset_type=asset_type,
            version=version,
            source_artifacts=source_artifacts,
            approved_by=actor,
            approved_at=now_iso(),
            note=str(payload.get("note") or ""),
        )
        try:
            path = projects.append_asset_promotion_record(clean_project_id, clean_run_id, record)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "record": record.model_dump(mode="json"),
                "record_path": str(path),
                "permission": decision.model_dump(mode="json"),
            }
        )


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _object_field(value: object, field_name: str) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_dict_field(value: object, field_name: str) -> dict[str, str]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, raw in value.items():
        clean_key = str(key or "").strip()
        clean_value = str(raw or "").strip()
        if clean_key and clean_value:
            result[clean_key] = clean_value
    return result


def _promotion_draft_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in ("model_id", "domain", "use_case", "rollback_asset_id"):
        value = str(payload.get(key) or "").strip()
        if value:
            overrides[key] = value
    for key in ("metrics", "applicability", "input_columns"):
        value = payload.get(key)
        if isinstance(value, dict):
            overrides[key] = value
    for key in ("feature_requirements", "limitations"):
        value = _string_list(payload.get(key))
        if value:
            overrides[key] = value
    return overrides
