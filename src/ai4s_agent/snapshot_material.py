from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_AUDIT_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "actor",
        "approved_by",
        "approved_at",
        "confirmed",
        "confirmation_note",
        "note",
        "project_approved",
        "user_confirmed",
    }
)

_OUTPUT_PATH_KEYS: frozenset[str] = frozenset(
    {
        "artifact_dir",
        "artifacts_dir",
        "log_dir",
        "model_root",
        "output_csv",
        "output_dir",
        "output_json",
        "output_path",
        "output_root",
        "report_dir",
        "save_dir",
        "temp_dir",
        "tmp_dir",
        "work_dir",
        "working_dir",
    }
)


def build_execution_snapshot(
    *,
    run_id: str,
    task_id: str,
    adapter: str,
    run_plan: Mapping[str, Any],
    task_options: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    run_dir: Path,
    approved_gates: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    """Build canonical snapshot material for a RunPlan task.

    The snapshot hash is intentionally based on computational inputs only:
    run plan, task options, execution payload, and content manifests.  Human
    audit fields such as actor/confirmed/note are separated into
    ``audit_metadata`` and are not included in the canonical hash material.
    """

    clean_payload = _json_safe(dict(execution_payload))
    execution_payload_only = _strip_audit_metadata(clean_payload)
    audit_metadata = _audit_metadata(clean_payload)
    resource_manifest = resource_manifest_for_payload(
        payload=execution_payload_only,
        artifact_paths=artifact_paths,
        run_dir=run_dir,
    )
    input_artifacts = {
        artifact_id: entry
        for artifact_id, entry in resource_manifest.items()
        if str(entry.get("artifact_id") or "")
    }
    hash_material = {
        "schema_version": 2,
        "run_id": str(run_id),
        "task_id": str(task_id),
        "adapter": str(adapter or ""),
        "run_plan": _json_safe(dict(run_plan)),
        "task_options": _json_safe(dict(task_options)),
        "execution_payload": execution_payload_only,
        "input_artifacts": input_artifacts,
        "resource_manifest": resource_manifest,
        "approved_gates": sorted(str(gate) for gate in approved_gates),
    }
    snapshot_hash = hashlib.sha256(_canonical_json(hash_material).encode("utf-8")).hexdigest()
    return {
        "snapshot_id": f"{run_id}:{task_id}:{snapshot_hash[:16]}",
        "snapshot_hash": snapshot_hash,
        **hash_material,
        # Backward-compatible alias for existing UI/tests while consumers move
        # to execution_payload.
        "payload": execution_payload_only,
        "audit_metadata": audit_metadata,
    }


def resource_manifest_for_payload(
    *,
    payload: Mapping[str, Any],
    artifact_paths: Mapping[str, str],
    run_dir: Path,
) -> dict[str, dict[str, Any]]:
    references = _payload_path_references(payload)
    artifact_lookup = _resolved_artifact_lookup(artifact_paths)
    manifest: dict[str, dict[str, Any]] = {}
    for reference in references:
        if reference["key"] in _OUTPUT_PATH_KEYS:
            continue
        raw_path = str(reference["path"])
        path = _resolve_payload_path(raw_path, run_dir=run_dir)
        artifact_id = artifact_lookup.get(str(_safe_resolve(path))) or ""
        manifest_key = artifact_id or reference["field"]
        base_key = manifest_key
        suffix = 2
        while manifest_key in manifest:
            manifest_key = f"{base_key}#{suffix}"
            suffix += 1
        manifest[manifest_key] = {
            "field": reference["field"],
            "key": reference["key"],
            "path": str(path),
            "raw_path": raw_path,
            "artifact_id": artifact_id,
            **_path_manifest(path),
        }
    return manifest


def install_run_plan_executor_snapshot_builder() -> None:
    """Install the v2 snapshot builder onto RunPlanExecutor.

    This keeps the executor state machine unchanged while moving the canonical
    snapshot material construction into this module.  A later cleanup can turn
    this into a direct import from executor.py.
    """

    from ai4s_agent.executor import RunPlanExecutor

    if getattr(RunPlanExecutor._execution_snapshot, "_snapshot_material_v2", False):
        return

    def _execution_snapshot_v2(
        self: Any,
        *,
        task_id: str,
        spec_default_adapter: str | None,
        run_plan: Any,
        run_dir: Path,
        artifact_paths: dict[str, str],
        approved_gates: set[str],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        spec = self.registry.get(task_id)
        clean_options = self._json_safe({str(key): value for key, value in options.items()})
        adapter_name = self._adapter_name_for(task_id, spec_default_adapter, clean_options)
        required_gates = set(spec.gates)
        payload = self._payload_for(
            task_id,
            run_id=run_plan.run_id,
            run_dir=run_dir,
            artifact_paths=artifact_paths,
            actor="",
            approved_gates=set(approved_gates) | required_gates,
            options=clean_options,
        )
        return build_execution_snapshot(
            run_id=run_plan.run_id,
            task_id=task_id,
            adapter=adapter_name or "",
            run_plan=run_plan.model_dump(mode="json"),
            task_options=clean_options,
            execution_payload=payload,
            artifact_paths=artifact_paths,
            run_dir=run_dir,
            approved_gates=required_gates,
        )

    _execution_snapshot_v2._snapshot_material_v2 = True  # type: ignore[attr-defined]
    RunPlanExecutor._execution_snapshot = _execution_snapshot_v2  # type: ignore[method-assign]


def _strip_audit_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_audit_metadata(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _AUDIT_PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_strip_audit_metadata(item) for item in value]
    return value


def _audit_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in sorted(_AUDIT_PAYLOAD_KEYS):
        if key in payload:
            result[key] = _json_safe(payload[key])
    return result


def _payload_path_references(value: Any, *, field: str = "payload", key: str = "") -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            clean_key = str(item_key)
            child_field = f"{field}.{clean_key}"
            references.extend(_payload_path_references(item_value, field=child_field, key=clean_key))
        return references
    if isinstance(value, list | tuple):
        for idx, item in enumerate(value):
            references.extend(_payload_path_references(item, field=f"{field}[{idx}]", key=key))
        return references
    if isinstance(value, str) and _looks_like_path(value):
        references.append({"field": field, "key": key, "path": value})
    return references


def _looks_like_path(value: str) -> bool:
    clean = value.strip()
    if not clean:
        return False
    return "/" in clean or "\\" in clean or clean.endswith((".json", ".csv", ".pkl", ".pt", ".py"))


def _resolved_artifact_lookup(artifact_paths: Mapping[str, str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for artifact_id, path_raw in artifact_paths.items():
        path = Path(str(path_raw)).expanduser()
        lookup[str(_safe_resolve(path))] = str(artifact_id)
    return lookup


def _resolve_payload_path(raw_path: str, *, run_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return _safe_resolve(path)
    return _safe_resolve(run_dir / path)


def _path_manifest(path: Path) -> dict[str, Any]:
    entry: dict[str, Any] = {}
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    entry["exists"] = True
    if path.is_symlink():
        return {**entry, "kind": "symlink", "target": str(path.readlink())}
    if path.is_file():
        return {**entry, "kind": "file", "size_bytes": stat.st_size, "sha256": _file_sha256(path)}
    if path.is_dir():
        return {**entry, "kind": "directory", **_directory_manifest(path)}
    return {**entry, "kind": "other", "size_bytes": stat.st_size}


def _directory_manifest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for child in sorted(path.rglob("*"), key=lambda item: str(item.relative_to(path))):
        rel = str(child.relative_to(path))
        try:
            stat = child.lstat()
        except FileNotFoundError:
            continue
        digest.update(rel.encode("utf-8"))
        if child.is_symlink():
            digest.update(b"symlink")
            digest.update(str(child.readlink()).encode("utf-8"))
            continue
        if child.is_file():
            file_count += 1
            total_size += stat.st_size
            digest.update(b"file")
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(_file_sha256(child).encode("utf-8"))
        elif child.is_dir():
            digest.update(b"dir")
    return {"file_count": file_count, "size_bytes": total_size, "manifest_sha256": digest.hexdigest()}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except FileNotFoundError:
        return path.absolute()


def _canonical_json(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
