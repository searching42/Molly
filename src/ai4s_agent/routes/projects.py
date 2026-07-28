from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any
import uuid

from flask import Flask, jsonify, request
from pydantic import ValidationError
from werkzeug.utils import secure_filename

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.memory import PermissionPolicy, ProjectMemory
from ai4s_agent.schemas import ProjectMemoryRecord
from ai4s_agent.storage import ProjectStorage


def register_project_routes(
    app: Flask,
    *,
    projects: ProjectStorage,
    project_memory: ProjectMemory,
    permissions: PermissionPolicy,
    allowed_file: Callable[[str], bool],
    copy_upload_stream: Callable[..., None],
    max_upload_bytes_default: int,
) -> None:
    @app.post("/api/projects")
    def create_project():
        payload = request.get_json(silent=True) or {}
        project_id = str(payload.get("project_id") or uuid.uuid4().hex[:8]).strip()
        name = str(payload.get("name") or project_id).strip()
        if not project_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        try:
            project_dir = projects.project_dir(project_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        write_json(project_dir / "project.json", {
            "project_id": project_id,
            "name": name,
            "created_at": now_iso(),
        })
        return jsonify({"ok": True, "project_id": project_id, "name": name})

    @app.get("/api/projects")
    def list_projects():
        projects_root = projects.projects_root
        result = []
        if projects_root.exists():
            for child in sorted(projects_root.iterdir()):
                if not child.is_dir():
                    continue
                meta = _read_json(child / "project.json")
                result.append({
                    "project_id": child.name,
                    "name": meta.get("name", child.name),
                    "created_at": meta.get("created_at", ""),
                })
        return jsonify({"ok": True, "projects": result})

    @app.get("/api/projects/<project_id>/memory")
    def list_project_memory(project_id: str):
        try:
            records = project_memory.list_project_records(str(project_id or "").strip())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify(
            {
                "ok": True,
                "project_id": project_id,
                "enabled": project_memory.project_memory_enabled(str(project_id or "").strip()),
                "records": [record.model_dump(mode="json") for record in records],
            }
        )

    @app.post("/api/projects/<project_id>/memory/records")
    def create_project_memory_record(project_id: str):
        payload = request.get_json(silent=True) or {}
        try:
            record = ProjectMemoryRecord.model_validate(payload)
            saved = project_memory.save_project_record(str(project_id or "").strip(), record)
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "record": saved.model_dump(mode="json")})

    @app.patch("/api/projects/<project_id>/memory/records/<record_id>")
    def update_project_memory_record(project_id: str, record_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        try:
            updated = project_memory.update_project_record(str(project_id or "").strip(), record_id, payload)
        except (ValidationError, ValueError) as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if updated is None:
            return jsonify({"ok": False, "error": "memory record not found"}), 404
        return jsonify({"ok": True, "record": updated.model_dump(mode="json")})

    @app.delete("/api/projects/<project_id>/memory/records/<record_id>")
    def delete_project_memory_record(project_id: str, record_id: str):
        try:
            deleted = project_memory.delete_project_record(str(project_id or "").strip(), record_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "deleted": deleted})

    @app.get("/api/projects/<project_id>/memory/export")
    def export_project_memory(project_id: str):
        try:
            exported = project_memory.export_project_records(str(project_id or "").strip())
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "export": exported})

    @app.post("/api/projects/<project_id>/memory/enabled")
    def set_project_memory_enabled(project_id: str):
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "payload must be an object"}), 400
        if not isinstance(payload.get("enabled"), bool):
            return jsonify({"ok": False, "error": "enabled boolean required"}), 400
        enabled = payload["enabled"]
        try:
            project_memory.set_project_memory_enabled(str(project_id or "").strip(), enabled)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        return jsonify({"ok": True, "project_id": project_id, "enabled": enabled})

    @app.post("/api/projects/<project_id>/upload")
    def upload_file(project_id: str):
        clean_id = str(project_id or "").strip()
        if not clean_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        decision = permissions.decide(
            "upload_dataset",
            project_id=clean_id,
            project_approved=_as_bool(request.form.get("project_approved"))
            or _as_bool(request.headers.get("X-Project-Approved")),
            actor=str(request.form.get("actor") or request.headers.get("X-Actor") or ""),
        )
        if not decision.allowed:
            return jsonify(
                {
                    "ok": False,
                    "error": "project approval required for dataset upload",
                    "permission": decision.model_dump(mode="json"),
                }
            ), 403
        max_upload_bytes = int(app.config.get("AI4S_MAX_UPLOAD_BYTES", max_upload_bytes_default) or max_upload_bytes_default)
        content_length = request.content_length
        if max_upload_bytes > 0 and content_length is not None and content_length > max_upload_bytes:
            return jsonify(
                {
                    "ok": False,
                    "error": f"upload exceeds size limit: {max_upload_bytes} bytes",
                    "max_upload_bytes": max_upload_bytes,
                }
            ), 413
        if "file" not in request.files:
            return jsonify({"ok": False, "error": "no file part"}), 400
        f = request.files["file"]
        if not f.filename or not allowed_file(f.filename):
            return jsonify({"ok": False, "error": "unsupported file type"}), 400
        filename = secure_filename(f.filename)
        if not filename or not allowed_file(filename):
            return jsonify({"ok": False, "error": "invalid filename after sanitization"}), 400
        try:
            project_dir = projects.project_dir(clean_id)
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        upload_dir = project_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest = upload_dir / filename
        if dest.exists() and dest.is_dir():
            return jsonify({"ok": False, "error": "invalid filename target"}), 400
        if dest.exists():
            return jsonify({"ok": False, "error": f"upload filename already exists: {filename}"}), 409
        try:
            with dest.open("xb") as out:
                copy_upload_stream(f.stream, out, max_bytes=max_upload_bytes)
        except FileExistsError:
            return jsonify({"ok": False, "error": f"upload filename already exists: {filename}"}), 409
        except ValueError as exc:
            dest.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": str(exc), "max_upload_bytes": max_upload_bytes}), 413
        return jsonify({"ok": True, "path": str(dest), "filename": filename})


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "project-approved"}
