from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TYPE_CHECKING

from flask import jsonify, request
from werkzeug.utils import secure_filename

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.memory import PermissionPolicy
from ai4s_agent.schemas import AssetManifest, AssetStatus
from ai4s_agent.server_permissions import ServerPermissionStore, decide_server_permission
from ai4s_agent.storage import ProjectStorage

if TYPE_CHECKING:
    from ai4s_agent.api_route_extensions import RouteExtensionContext


def install_immutable_upload_assets() -> None:
    """Upload route migration is handled by the explicit route hook."""


def apply_immutable_upload_assets_route_override(context: "RouteExtensionContext") -> None:
    """Route project uploads through immutable, versioned asset storage."""

    import ai4s_agent.api as api_module

    workspace = api_module._workspace_from_config(
        base_runs_dir=context.base_runs_dir,
        workspace_dir=context.workspace_dir,
    )
    runs = Path(context.base_runs_dir or api_module.DEFAULT_RUNS_DIR).resolve()
    projects = ProjectStorage(workspace_dir=workspace)
    permissions = PermissionPolicy()
    server_permissions = ServerPermissionStore(workspace_dir=workspace)
    context.route_overrides.apply_route_override(
        context.app,
        extension_id="immutable_upload_assets",
        endpoint="upload_file",
        view_func=_upload_file_view(
            app=context.app,
            projects=projects,
            permissions=permissions,
            server_permissions=server_permissions,
            allowed_file=api_module._allowed_file,
            max_upload_bytes_default=api_module.MAX_UPLOAD_BYTES,
            chunk_bytes=api_module.UPLOAD_COPY_CHUNK_BYTES,
            reject_legacy_duplicate_filename=(workspace == runs),
        ),
    )


def _upload_file_view(
    *,
    app: Any,
    projects: ProjectStorage,
    permissions: PermissionPolicy,
    server_permissions: ServerPermissionStore,
    allowed_file: Any,
    max_upload_bytes_default: int,
    chunk_bytes: int,
    reject_legacy_duplicate_filename: bool,
):
    def upload_file(project_id: str):
        clean_id = str(project_id or "").strip()
        if not clean_id:
            return jsonify({"ok": False, "error": "project_id required"}), 400
        actor = str(request.form.get("actor") or request.headers.get("X-Actor") or "").strip()
        legacy_project_approved = _as_bool(request.form.get("project_approved")) or _as_bool(request.headers.get("X-Project-Approved"))
        allow_legacy_flags = _config_bool(app.config.get("AI4S_ALLOW_CLIENT_PERMISSION_FLAGS", True), default=True)
        try:
            decision = decide_server_permission(
                server_permissions,
                permissions,
                "upload_dataset",
                project_id=clean_id,
                actor=actor,
                legacy_project_approved=legacy_project_approved,
                allow_legacy_client_flags=allow_legacy_flags,
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        if not decision["allowed"]:
            return jsonify(
                {
                    "ok": False,
                    "error": "server permission grant required for dataset upload",
                    "permission": decision,
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
        uploaded = request.files["file"]
        if not uploaded.filename or not allowed_file(uploaded.filename):
            return jsonify({"ok": False, "error": "unsupported file type"}), 400
        filename = secure_filename(uploaded.filename)
        if not filename or not allowed_file(filename):
            return jsonify({"ok": False, "error": "invalid filename after sanitization"}), 400
        try:
            if reject_legacy_duplicate_filename:
                legacy_path = _legacy_upload_path(projects.project_dir(clean_id), filename)
                if legacy_path.exists():
                    return jsonify({"ok": False, "error": f"upload filename already exists: {filename}"}), 409
            asset = _write_uploaded_asset(
                projects=projects,
                project_id=clean_id,
                filename=filename,
                original_filename=str(uploaded.filename or ""),
                stream=uploaded.stream,
                max_bytes=max_upload_bytes,
                chunk_bytes=chunk_bytes,
            )
        except ValueError as exc:
            return jsonify({"ok": False, "error": str(exc), "max_upload_bytes": max_upload_bytes}), 413 if "size limit" in str(exc) else 400
        except FileExistsError:
            return jsonify({"ok": False, "error": "asset version already exists; retry upload"}), 409
        return jsonify(
            {
                "ok": True,
                "path": asset["legacy_path"],
                "filename": filename,
                "asset": {**asset, "permission": decision},
                "permission": decision,
            }
        )

    return upload_file


def _write_uploaded_asset(
    *,
    projects: ProjectStorage,
    project_id: str,
    filename: str,
    original_filename: str,
    stream: Any,
    max_bytes: int,
    chunk_bytes: int,
) -> dict[str, Any]:
    project_dir = projects.project_dir(project_id)
    stem = _asset_stem(filename)
    scope = ["uploads", stem]
    version_dir = projects.create_asset_version_dir(project_id, scope)
    version = version_dir.name
    data_path = (version_dir / filename).resolve()
    if not data_path.is_relative_to(version_dir.resolve()):
        raise ValueError("upload asset path escapes version directory")
    digest = hashlib.sha256()
    total = 0
    try:
        with data_path.open("xb") as out:
            while True:
                chunk = stream.read(chunk_bytes)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes > 0 and total > max_bytes:
                    raise ValueError(f"upload exceeds size limit: {max_bytes} bytes")
                digest.update(chunk)
                out.write(chunk)
    except Exception:
        data_path.unlink(missing_ok=True)
        try:
            version_dir.rmdir()
        except OSError:
            pass
        raise
    content_hash = f"sha256:{digest.hexdigest()}"
    asset_id = f"upload/{stem}"
    manifest = AssetManifest(
        asset_id=asset_id,
        asset_type="uploaded_dataset",
        version=version,
        status=AssetStatus.CANDIDATE,
        created_from_run_id="",
        source_artifacts=[filename],
        content_hash=content_hash,
    )
    manifest_path = projects.write_asset_manifest(project_id, scope, version, manifest)
    record = {
        "asset_id": asset_id,
        "asset_type": manifest.asset_type,
        "version": version,
        "project_id": project_id,
        "filename": filename,
        "original_filename": original_filename,
        "content_hash": content_hash,
        "sha256": digest.hexdigest(),
        "size_bytes": total,
        "path": str(data_path),
        "relative_path": str(data_path.relative_to(project_dir)),
        "manifest_path": str(manifest_path),
        "created_at": now_iso(),
        "immutable": True,
    }
    write_json(version_dir / "upload_record.json", record)
    legacy_path = _write_legacy_upload_compat(project_dir, filename, data_path)
    return {**record, "legacy_path": str(legacy_path)}


def _legacy_upload_path(project_dir: Path, filename: str) -> Path:
    upload_dir = (project_dir / "uploads").resolve()
    if not upload_dir.is_relative_to(project_dir.resolve()):
        raise ValueError("legacy upload path escapes project directory")
    return (upload_dir / filename).resolve()


def _write_legacy_upload_compat(project_dir: Path, filename: str, source_path: Path) -> Path:
    upload_dir = (project_dir / "uploads").resolve()
    if not upload_dir.is_relative_to(project_dir.resolve()):
        raise ValueError("legacy upload path escapes project directory")
    upload_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = (upload_dir / filename).resolve()
    if not legacy_path.is_relative_to(upload_dir):
        raise ValueError("legacy upload path escapes upload directory")
    if legacy_path.exists():
        if legacy_path.is_dir():
            raise ValueError("invalid filename target")
        return legacy_path
    with source_path.open("rb") as src, legacy_path.open("xb") as dest:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dest.write(chunk)
    return legacy_path


def _asset_stem(filename: str) -> str:
    stem = Path(filename).stem.strip().lower().replace(" ", "_").replace("-", "_")
    clean = "".join(ch if ch.isalnum() or ch in {"_", "."} else "_" for ch in stem).strip("._")
    return clean or "upload"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _config_bool(value: Any, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value != 0
    clean = str(value).strip().lower()
    if clean in {"false", "0", "no", "n", "off"}:
        return False
    if clean in {"true", "1", "yes", "y", "on"}:
        return True
    return default
