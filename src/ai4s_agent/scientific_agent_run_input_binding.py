"""Server-owned BR1 logical input binding for conversation-launched runs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.br1_acceptance_readiness import FREEZE_SCHEMA, _validate_schema
from ai4s_agent.generation_publication import read_regular_file_bound
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_confirmation import digest_json


INPUT_BINDING_SCHEMA = "scientific_agent_br1_input_binding.v1"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_SAFE_RELATIVE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_MAX_DATASET_BYTES = 32 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_TEMPLATE_BYTES = 16 * 1024 * 1024


class ScientificAgentRunInputBindingError(ValueError):
    """A logical input bundle is not eligible for an immutable run binding."""


def _safe_id(value: Any, *, field: str) -> str:
    clean = str(value or "").strip()
    if _SAFE_ID.fullmatch(clean) is None:
        raise ScientificAgentRunInputBindingError(
            f"{field} must be a canonical logical identifier"
        )
    return clean


def _read_bytes(path: Path, *, max_bytes: int, label: str) -> tuple[bytes, str]:
    try:
        payload, digest = read_regular_file_bound(path, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        raise ScientificAgentRunInputBindingError(
            f"{label} is unavailable or changed"
        ) from exc
    return payload, "sha256:" + digest


def _read_json(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    payload, digest = _read_bytes(path, max_bytes=_MAX_JSON_BYTES, label=label)
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScientificAgentRunInputBindingError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ScientificAgentRunInputBindingError(f"{label} must be a JSON object")
    return parsed, digest


def _safe_source_file(root: Path, relative_path: str, *, label: str) -> Path:
    if not _SAFE_RELATIVE.fullmatch(relative_path):
        raise ScientificAgentRunInputBindingError(f"{label} path is not allowlisted")
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ScientificAgentRunInputBindingError(f"{label} path escapes the bundle")
    current = path
    while current != root.resolve():
        if current.is_symlink():
            raise ScientificAgentRunInputBindingError(f"{label} path is unsafe")
        current = current.parent
    return path


class ScientificAgentRunInputBindingService:
    """Resolve a bundle ID under a server-owned project root and bind exact bytes."""

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        bundles_root: Path | None = None,
        require_reinvent4_template: bool = False,
    ) -> None:
        self.storage = storage
        self.bundles_root = None if bundles_root is None else Path(bundles_root).resolve()
        self.require_reinvent4_template = bool(require_reinvent4_template)

    def _bundle_dir(self, project_id: str, bundle_id: str) -> Path:
        project = _safe_id(project_id, field="project_id")
        bundle = _safe_id(bundle_id, field="input_bundle_id")
        if self.bundles_root is None:
            project_root = self.storage.project_dir(project)
            roots = [
                project_root / "br1-input-bundles",
                project_root / "assets" / "br1-input-bundles",
            ]
        else:
            roots = [self.bundles_root / project]
        candidates: list[tuple[Path, Path]] = []
        for candidate_root in roots:
            root = candidate_root.resolve()
            path = (root / bundle).resolve()
            if not path.is_relative_to(root):
                raise ScientificAgentRunInputBindingError(
                    "input bundle escapes the server root"
                )
            if path.exists() or path.is_symlink():
                candidates.append((root, path))
        if len(candidates) > 1:
            raise ScientificAgentRunInputBindingError("input bundle is ambiguous")
        if not candidates:
            raise ScientificAgentRunInputBindingError("input bundle is unavailable")
        _root, path = candidates[0]
        if path.is_symlink() or not path.is_dir():
            raise ScientificAgentRunInputBindingError("input bundle is unavailable")
        return path

    @staticmethod
    def _verify_freeze_package(
        *, bundle_dir: Path, project_id: str, bundle_id: str
    ) -> tuple[dict[str, Any], dict[str, bytes], dict[str, str]]:
        package_path = _safe_source_file(
            bundle_dir, "freeze_package.json", label="freeze package"
        )
        package, _package_file_digest = _read_json(
            package_path, label="freeze package"
        )
        try:
            _validate_schema(
                package,
                "br1_acceptance_candidate_freeze.schema.json",
                FREEZE_SCHEMA,
            )
        except (OSError, ValueError) as exc:
            raise ScientificAgentRunInputBindingError(
                "freeze package is not an eligible BR1 candidate"
            ) from exc
        material = dict(package)
        claimed = material.pop("freeze_package_digest", None)
        if claimed != digest_json(material):
            raise ScientificAgentRunInputBindingError("freeze package digest mismatch")

        roster = package.get("artifact_roster")
        if not isinstance(roster, dict):
            raise ScientificAgentRunInputBindingError("freeze artifact roster is unavailable")
        expected = {
            "raw_dataset": ("raw_dataset.csv", "raw_dataset_digest", _MAX_DATASET_BYTES),
            "source_manifest": (
                "source_dataset_manifest.json",
                "source_dataset_manifest_digest",
                _MAX_JSON_BYTES,
            ),
            "mapping_policy": (
                "mapping_policy.json",
                "mapping_policy_digest",
                _MAX_JSON_BYTES,
            ),
        }
        payloads: dict[str, bytes] = {}
        digests: dict[str, str] = {}
        for artifact_id, (filename, package_digest_key, max_bytes) in expected.items():
            entry = roster.get(artifact_id)
            if not isinstance(entry, dict) or entry.get("relative_path") != filename:
                raise ScientificAgentRunInputBindingError(
                    f"freeze artifact binding is invalid: {artifact_id}"
                )
            source = _safe_source_file(bundle_dir, filename, label=artifact_id)
            payload, digest = _read_bytes(source, max_bytes=max_bytes, label=artifact_id)
            if len(payload) != entry.get("size_bytes") or digest != entry.get("sha256"):
                raise ScientificAgentRunInputBindingError(
                    f"freeze artifact bytes changed: {artifact_id}"
                )
            if digest != package.get(package_digest_key):
                raise ScientificAgentRunInputBindingError(
                    f"freeze artifact digest binding is invalid: {artifact_id}"
                )
            payloads[artifact_id] = payload
            digests[artifact_id] = digest

        manifest, manifest_digest = _read_json(
            _safe_source_file(bundle_dir, "source_dataset_manifest.json", label="source manifest"),
            label="source manifest",
        )
        if manifest.get("project_id") not in {None, project_id}:
            raise ScientificAgentRunInputBindingError("source manifest crosses project scope")
        mapping, mapping_digest = _read_json(
            _safe_source_file(bundle_dir, "mapping_policy.json", label="mapping policy"),
            label="mapping policy",
        )
        if manifest_digest != package.get("source_dataset_manifest_digest"):
            raise ScientificAgentRunInputBindingError("source manifest digest is not current")
        if mapping_digest != package.get("mapping_policy_digest"):
            raise ScientificAgentRunInputBindingError("mapping policy digest is not current")
        if manifest.get("derived_raw_dataset_sha256") not in {
            package.get("raw_dataset_digest"),
            digests["raw_dataset"],
        }:
            raise ScientificAgentRunInputBindingError("source manifest Raw binding is invalid")
        if mapping.get("schema_version") != "br1_raw_dataset_mapping_policy.v1":
            raise ScientificAgentRunInputBindingError("mapping policy schema is not BR1")
        return package, payloads, digests

    def _template(self, bundle_dir: Path) -> tuple[str, bytes, str] | None:
        candidates = [
            name
            for name in ("reinvent4_config_template.toml", "reinvent4_config_template.json")
            if (bundle_dir / name).exists()
        ]
        if len(candidates) > 1:
            raise ScientificAgentRunInputBindingError(
                "REINVENT4 configuration template is ambiguous"
            )
        if not candidates:
            if self.require_reinvent4_template:
                raise ScientificAgentRunInputBindingError(
                    "REINVENT4 configuration template is missing"
                )
            return None
        filename = candidates[0]
        source = _safe_source_file(bundle_dir, filename, label="REINVENT4 template")
        payload, digest = _read_bytes(
            source, max_bytes=_MAX_TEMPLATE_BYTES, label="REINVENT4 template"
        )
        if b"{{molly_output_csv}}" not in payload or b"{{molly_seed}}" not in payload:
            raise ScientificAgentRunInputBindingError(
                "REINVENT4 configuration template lacks required Molly bindings"
            )
        return filename, payload, digest

    @staticmethod
    def _copy_immutable(destination: Path, payload: bytes, *, label: str) -> None:
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if destination.is_symlink():
            raise ScientificAgentRunInputBindingError(f"bound {label} is unsafe")
        if destination.exists():
            existing, _ = _read_bytes(destination, max_bytes=max(len(payload), 1), label=label)
            if existing != payload:
                raise ScientificAgentRunInputBindingError(f"bound {label} is immutable")
            return
        temporary = destination.with_name(f".{destination.name}.binding")
        if temporary.exists() or temporary.is_symlink():
            raise ScientificAgentRunInputBindingError("input binding temporary file is unsafe")
        try:
            temporary.write_bytes(payload)
            temporary.chmod(0o600)
            temporary.replace(destination)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()

    def bind(
        self, *, project_id: str, run_id: str, input_bundle_id: str
    ) -> dict[str, Any]:
        project = _safe_id(project_id, field="project_id")
        run = _safe_id(run_id, field="run_id")
        bundle = _safe_id(input_bundle_id, field="input_bundle_id")
        bundle_dir = self._bundle_dir(project, bundle)
        package, payloads, digests = self._verify_freeze_package(
            bundle_dir=bundle_dir,
            project_id=project,
            bundle_id=bundle,
        )
        template = self._template(bundle_dir)

        run_dir = self.storage.run_dir(project, run)
        input_dir = run_dir / "inputs" / "br1"
        names = {
            "uploaded_dataset": ("uploaded_dataset.csv", payloads["raw_dataset"]),
            "source_dataset_manifest": (
                "source_dataset_manifest.json",
                payloads["source_manifest"],
            ),
            "br1_mapping_policy": ("mapping_policy.json", payloads["mapping_policy"]),
        }
        if template is not None:
            names["reinvent4_config_template"] = (template[0], template[1])
        relative_artifacts = {
            artifact_id: f"inputs/br1/{filename}"
            for artifact_id, (filename, _payload) in names.items()
        }
        binding_relative = "inputs/br1/binding.json"
        relative_artifacts["br1_input_binding"] = binding_relative
        existing_registry = self.storage.read_artifact_registry(project, run)
        for artifact_id, relative in relative_artifacts.items():
            existing = existing_registry.get(artifact_id)
            if existing is not None and existing != relative:
                raise ScientificAgentRunInputBindingError(
                    f"logical artifact is already bound: {artifact_id}"
                )

        binding_payload = {
            "schema_version": INPUT_BINDING_SCHEMA,
            "project_id": project,
            "run_id": run,
            "input_bundle_id": bundle,
            "freeze_package_id": str(package["package_id"]),
            "freeze_package_digest": str(package["freeze_package_digest"]),
            "artifact_digests": {
                "uploaded_dataset": digests["raw_dataset"],
                "source_dataset_manifest": digests["source_manifest"],
                "br1_mapping_policy": digests["mapping_policy"],
                **({"reinvent4_config_template": template[2]} if template else {}),
            },
        }
        binding_digest = digest_json(binding_payload)
        binding_payload["binding_digest"] = binding_digest
        for artifact_id, (filename, payload) in names.items():
            self._copy_immutable(
                run_dir / "inputs" / "br1" / filename,
                payload,
                label=artifact_id,
            )
        binding_path = run_dir / "inputs" / "br1" / "binding.json"
        if binding_path.exists() or binding_path.is_symlink():
            existing, _ = _read_json(binding_path, label="input binding")
            if existing != binding_payload:
                raise ScientificAgentRunInputBindingError(
                    "logical input binding is immutable"
                )
        else:
            write_json(binding_path, binding_payload)

        missing = {
            key: relative
            for key, relative in relative_artifacts.items()
            if key not in existing_registry
        }
        if missing:
            try:
                self.storage.register_new_artifact_registry_paths(
                    project,
                    run,
                    missing,
                )
            except ValueError as exc:
                raise ScientificAgentRunInputBindingError(
                    "logical input Registry changed during binding"
                ) from exc
        return {
            "schema_version": INPUT_BINDING_SCHEMA,
            "project_id": project,
            "run_id": run,
            "input_bundle_id": bundle,
            "freeze_package_id": str(package["package_id"]),
            "freeze_package_digest": str(package["freeze_package_digest"]),
            "binding_digest": binding_digest,
            "artifact_ids": sorted(relative_artifacts),
            "reinvent4_template_bound": template is not None,
            "idempotent": not bool(missing),
        }


__all__ = [
    "INPUT_BINDING_SCHEMA",
    "ScientificAgentRunInputBindingError",
    "ScientificAgentRunInputBindingService",
]
