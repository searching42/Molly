from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request

from ai4s_agent.memory import PermissionPolicy
from ai4s_agent.ui_cards import build_data_confirmation_card, build_run_confirmation_card


def register_review_routes(app: Flask, *, workspace: Path, permissions: PermissionPolicy) -> None:
    @app.post("/api/data-confirmation-card")
    def data_confirmation_card():
        payload = request.get_json(silent=True) or {}
        dataset_path_raw = str(payload.get("dataset_path") or "").strip()
        if not dataset_path_raw:
            return jsonify({"ok": False, "error": "dataset_path required"}), 400
        try:
            card = build_data_confirmation_card(payload, base=workspace)
        except (FileNotFoundError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "card": card})

    @app.post("/api/run-confirmation-card")
    def run_confirmation_card():
        payload = request.get_json(silent=True) or {}
        try:
            card = build_run_confirmation_card(payload)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "card": card})

    @app.post("/api/permissions/resolve")
    def resolve_permission():
        payload = request.get_json(silent=True) or {}
        action = str(payload.get("action") or "").strip()
        if not action:
            return jsonify({"ok": False, "error": "action required"}), 400
        decision = permissions.decide(
            action,
            project_id=str(payload.get("project_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            project_approved=_as_bool(payload.get("project_approved")),
            confirmed=_as_bool(payload.get("confirmed")),
            actor=str(payload.get("actor") or payload.get("approved_by") or ""),
        )
        return jsonify({"ok": True, **decision.model_dump(mode="json")})


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}
