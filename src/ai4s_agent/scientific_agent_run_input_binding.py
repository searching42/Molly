"""Server-owned BR1 logical input binding for conversation-launched runs."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from collections.abc import Callable
from typing import Any

from ai4s_agent._utils import write_json
from ai4s_agent.br1_acceptance_readiness import (
    BR1AcceptanceReadinessError,
    FREEZE_SCHEMA,
    _validate_schema,
    require_br1_acceptance_owner_approval,
)
from ai4s_agent.generation_publication import read_regular_file_bound
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.structured_dataset_confirmation import digest_json


INPUT_BINDING_SCHEMA = "scientific_agent_br1_input_binding.v1"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_SAFE_RELATIVE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_MAX_DATASET_BYTES = 32 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_TEMPLATE_BYTES = 16 * 1024 * 1024
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

_PROPOSAL_PACKAGE_BINDING_KEYS = (
    "freeze_package_id",
    "freeze_package_digest",
    "repository_commit",
    "worker_implementation_digest",
    "provider_name",
    "expected_provider_version",
    "execution_profile_id",
    "execution_profile_digest",
    "raw_dataset_digest",
    "source_dataset_manifest_digest",
    "mapping_policy_digest",
    "canonical_source_dataset_digest",
    "canonical_provider_input_digest",
    "source_publication_digest",
    "source_publication_registry_digest",
    "source_authority_digest",
    "source_authority_file_digest",
    "report_digest",
    "summary_digest",
    "input_row_count",
    "claim_boundary",
)


class ScientificAgentRunInputBindingError(ValueError):
    """A logical input bundle is not eligible for an immutable run binding."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "BR1_INPUT_BUNDLE_UNAVAILABLE",
        bundle_ids: tuple[str, ...] = (),
    ) -> None:
        self.reason_code = reason_code
        self.bundle_ids = tuple(bundle_ids)
        super().__init__(message)


def resolve_server_br1_deployment_identity() -> tuple[str, str]:
    """Resolve the deployment identity used to reject stale frozen bundles."""

    commit = str(os.environ.get("MOLLY_REPOSITORY_COMMIT") or "").strip().lower()
    if not commit:
        repository_root = Path(__file__).resolve().parents[2]
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repository_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip().lower()
        except (OSError, subprocess.SubprocessError):
            commit = ""
    worker_digest = str(
        os.environ.get("MOLLY_WORKER_IMPLEMENTATION_DIGEST") or ""
    ).strip().lower()
    if not worker_digest:
        worker_path = Path(__file__).with_name("molly_worker.py")
        try:
            _worker_bytes, worker_hex = read_regular_file_bound(
                worker_path, max_bytes=32 * 1024 * 1024
            )
        except (OSError, ValueError) as exc:
            raise ScientificAgentRunInputBindingError(
                "current BR1 worker implementation identity is unavailable",
                reason_code="BR1_DEPLOYMENT_IDENTITY_UNAVAILABLE",
            ) from exc
        worker_digest = "sha256:" + worker_hex
    if _COMMIT.fullmatch(commit) is None or _DIGEST.fullmatch(worker_digest) is None:
        raise ScientificAgentRunInputBindingError(
            "current BR1 deployment identity is invalid",
            reason_code="BR1_DEPLOYMENT_IDENTITY_UNAVAILABLE",
        )
    return commit, worker_digest


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
    root_path = root.resolve()
    raw_path = root / relative_path
    current = raw_path
    while current != root_path:
        if current.is_symlink():
            raise ScientificAgentRunInputBindingError(f"{label} path is unsafe")
        current = current.parent
    path = raw_path.resolve()
    if not path.is_relative_to(root_path):
        raise ScientificAgentRunInputBindingError(f"{label} path escapes the bundle")
    return path


class ScientificAgentRunInputBindingService:
    """Resolve a bundle ID under a server-owned project root and bind exact bytes."""

    def __init__(
        self,
        *,
        storage: ProjectStorage,
        bundles_root: Path | None = None,
        require_reinvent4_template: bool = False,
        trusted_owner_ids: set[str] | Callable[[], set[str]] | None = None,
        deployment_identity: Callable[[], tuple[str, str]] | None = None,
    ) -> None:
        self.storage = storage
        self.bundles_root = None if bundles_root is None else Path(bundles_root).resolve()
        self.require_reinvent4_template = bool(require_reinvent4_template)
        self._trusted_owner_ids_source = trusted_owner_ids
        self._deployment_identity_source = deployment_identity

    def _trusted_owner_ids(self) -> set[str]:
        source = self._trusted_owner_ids_source
        if callable(source):
            values = source()
        elif source is not None:
            values = source
        else:
            values = {
                item.strip()
                for item in str(os.environ.get("MOLLY_BR1_TRUSTED_OWNER_IDS") or "").split(",")
                if item.strip()
            }
        return {
            str(item).strip()
            for item in values
            if _SAFE_ID.fullmatch(str(item).strip()) is not None
        }

    def _current_deployment_identity(self) -> tuple[str, str]:
        source = self._deployment_identity_source
        try:
            identity = source() if source is not None else resolve_server_br1_deployment_identity()
            commit, worker_digest = identity
        except ScientificAgentRunInputBindingError:
            raise
        except Exception as exc:
            raise ScientificAgentRunInputBindingError(
                "current BR1 deployment identity is unavailable",
                reason_code="BR1_DEPLOYMENT_IDENTITY_UNAVAILABLE",
            ) from exc
        commit = str(commit or "").strip().lower()
        worker_digest = str(worker_digest or "").strip().lower()
        if _COMMIT.fullmatch(commit) is None or _DIGEST.fullmatch(worker_digest) is None:
            raise ScientificAgentRunInputBindingError(
                "current BR1 deployment identity is invalid",
                reason_code="BR1_DEPLOYMENT_IDENTITY_UNAVAILABLE",
            )
        return commit, worker_digest

    def _bundle_roots(self, project_id: str) -> list[Path]:
        project = _safe_id(project_id, field="project_id")
        if self.bundles_root is None:
            project_root = self.storage.project_dir(project)
            return [
                project_root / "br1-input-bundles",
                project_root / "assets" / "br1-input-bundles",
            ]
        return [self.bundles_root / project]

    def _bundle_dir(self, project_id: str, bundle_id: str) -> Path:
        _safe_id(project_id, field="project_id")
        bundle = _safe_id(bundle_id, field="input_bundle_id")
        roots = self._bundle_roots(project_id)
        candidates: list[tuple[Path, Path]] = []
        for candidate_root in roots:
            if candidate_root.is_symlink():
                raise ScientificAgentRunInputBindingError("input bundle root is unsafe")
            root = candidate_root.resolve()
            raw_path = root / bundle
            if raw_path.is_symlink():
                raise ScientificAgentRunInputBindingError("input bundle is unavailable")
            path = raw_path.resolve()
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

    def _verify_owner_approval(
        self, *, bundle_dir: Path, package: dict[str, Any]
    ) -> None:
        proposal_path = _safe_source_file(
            bundle_dir,
            "owner_acceptance_proposal.json",
            label="owner acceptance proposal",
        )
        proposal, _proposal_file_digest = _read_json(
            proposal_path, label="owner acceptance proposal"
        )
        approval_path = _safe_source_file(
            bundle_dir,
            "owner_acceptance_approval.json",
            label="owner acceptance approval",
        )
        approval: dict[str, Any] | None
        if approval_path.exists():
            approval, _approval_file_digest = _read_json(
                approval_path, label="owner acceptance approval"
            )
        else:
            approval = None
        try:
            require_br1_acceptance_owner_approval(
                approval,
                proposal=proposal,
                trusted_owner_ids=self._trusted_owner_ids(),
            )
        except BR1AcceptanceReadinessError as exc:
            message = str(exc)
            reason_code = (
                "BR1_OWNER_APPROVAL_REQUIRED"
                if message.startswith("WAITING_OWNER:")
                else "BR1_OWNER_APPROVAL_INVALID"
            )
            raise ScientificAgentRunInputBindingError(
                "exact BR1 owner approval is not valid",
                reason_code=reason_code,
            ) from exc
        for key in _PROPOSAL_PACKAGE_BINDING_KEYS:
            package_key = "package_id" if key == "freeze_package_id" else key
            if proposal.get(key) != package.get(package_key):
                raise ScientificAgentRunInputBindingError(
                    "owner acceptance proposal does not bind this freeze package",
                    reason_code="BR1_OWNER_APPROVAL_INVALID",
                )
        current_commit, current_worker_digest = self._current_deployment_identity()
        if (
            package.get("repository_commit") != current_commit
            or package.get("worker_implementation_digest") != current_worker_digest
        ):
            raise ScientificAgentRunInputBindingError(
                "frozen BR1 package is stale for the current deployment",
                reason_code="BR1_FREEZE_STALE",
            )

    def _verify_bundle(
        self, *, bundle_dir: Path, project_id: str, bundle_id: str
    ) -> tuple[dict[str, Any], dict[str, bytes], dict[str, str]]:
        package, payloads, digests = self._verify_freeze_package(
            bundle_dir=bundle_dir,
            project_id=project_id,
            bundle_id=bundle_id,
        )
        self._verify_owner_approval(bundle_dir=bundle_dir, package=package)
        return package, payloads, digests

    def list_eligible_bundle_ids(self, *, project_id: str) -> list[str]:
        """Return only safe logical IDs for currently owner-approved bundles."""

        project = _safe_id(project_id, field="project_id")
        eligible: set[str] = set()
        for candidate_root in self._bundle_roots(project):
            if not candidate_root.exists():
                continue
            if candidate_root.is_symlink() or not candidate_root.is_dir():
                raise ScientificAgentRunInputBindingError("input bundle root is unsafe")
            for path in candidate_root.iterdir():
                bundle_id = path.name
                if _SAFE_ID.fullmatch(bundle_id) is None or path.is_symlink() or not path.is_dir():
                    continue
                try:
                    self._verify_bundle(
                        bundle_dir=path,
                        project_id=project,
                        bundle_id=bundle_id,
                    )
                except ScientificAgentRunInputBindingError:
                    continue
                eligible.add(bundle_id)
        return sorted(eligible)

    def _read_existing_binding(self, *, project_id: str, run_id: str) -> dict[str, Any] | None:
        registry = self.storage.read_artifact_registry(project_id, run_id)
        relative = registry.get("br1_input_binding")
        if relative is None:
            return None
        if relative != "inputs/br1/binding.json":
            raise ScientificAgentRunInputBindingError(
                "logical BR1 input binding path is invalid",
                reason_code="INPUT_BINDING_IMMUTABLE",
            )
        run_dir = self.storage.run_dir(project_id, run_id)
        raw_path = run_dir / relative
        if raw_path.is_symlink():
            raise ScientificAgentRunInputBindingError(
                "logical BR1 input binding is unavailable",
                reason_code="INPUT_BINDING_IMMUTABLE",
            )
        path = raw_path.resolve()
        if not path.is_relative_to(run_dir.resolve()) or not path.is_file():
            raise ScientificAgentRunInputBindingError(
                "logical BR1 input binding is unavailable",
                reason_code="INPUT_BINDING_IMMUTABLE",
            )
        binding, _binding_file_digest = _read_json(path, label="input binding")
        claimed = binding.get("binding_digest")
        material = dict(binding)
        material.pop("binding_digest", None)
        if binding.get("schema_version") != INPUT_BINDING_SCHEMA or claimed != digest_json(material):
            raise ScientificAgentRunInputBindingError(
                "logical BR1 input binding digest is invalid",
                reason_code="INPUT_BINDING_IMMUTABLE",
            )
        if binding.get("project_id") != project_id or binding.get("run_id") != run_id:
            raise ScientificAgentRunInputBindingError(
                "logical BR1 input binding scope is invalid",
                reason_code="INPUT_BINDING_IMMUTABLE",
            )
        return binding

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
        existing_binding = self._read_existing_binding(project_id=project, run_id=run)
        if existing_binding is not None and existing_binding.get("input_bundle_id") != bundle:
            raise ScientificAgentRunInputBindingError(
                "an immutable BR1 input binding already exists for this run",
                reason_code="INPUT_BINDING_IMMUTABLE",
            )
        bundle_dir = self._bundle_dir(project, bundle)
        package, payloads, digests = self._verify_bundle(
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

    def bind_eligible(
        self,
        *,
        project_id: str,
        run_id: str,
        input_bundle_id: str = "",
    ) -> dict[str, Any]:
        """Bind an explicit logical ID or the sole eligible server-owned bundle."""

        project = _safe_id(project_id, field="project_id")
        run = _safe_id(run_id, field="run_id")
        requested = str(input_bundle_id or "").strip()
        if requested:
            return self.bind(
                project_id=project,
                run_id=run,
                input_bundle_id=requested,
            )
        existing = self._read_existing_binding(project_id=project, run_id=run)
        if existing is not None:
            return self.bind(
                project_id=project,
                run_id=run,
                input_bundle_id=str(existing["input_bundle_id"]),
            )
        eligible = self.list_eligible_bundle_ids(project_id=project)
        if not eligible:
            raise ScientificAgentRunInputBindingError(
                "no owner-approved BR1 input bundle is available",
                reason_code="BR1_INPUT_BUNDLE_REQUIRED",
            )
        if len(eligible) > 1:
            raise ScientificAgentRunInputBindingError(
                "more than one owner-approved BR1 input bundle is available",
                reason_code="BR1_INPUT_BUNDLE_SELECTION_REQUIRED",
                bundle_ids=tuple(eligible),
            )
        return self.bind(
            project_id=project,
            run_id=run,
            input_bundle_id=eligible[0],
        )


__all__ = [
    "INPUT_BINDING_SCHEMA",
    "ScientificAgentRunInputBindingError",
    "ScientificAgentRunInputBindingService",
    "resolve_server_br1_deployment_identity",
]
