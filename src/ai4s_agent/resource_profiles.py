from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Sequence

from platformdirs import user_config_path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.runtime_environments import (
    _absolute_config_path,
    _open_private_directory,
    _private_process_lock,
    _read_private_json,
    _write_private_json,
)

CONNECTION_PROFILE_SCHEMA_VERSION = "molly_connection_profiles.v1"
CAPABILITY_PROBE_SCHEMA_VERSION = "molly_capability_probe.v1"
TRANSFER_MANIFEST_SCHEMA_VERSION = "molly_transfer_manifest.v1"
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_SAFE_SSH_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
# ``hostname -s`` is an operating-system identity, not necessarily a DNS name.
# Some managed clusters legitimately use underscores in their short hostname.
# The value is still argv-separated and exact-compared; shell metacharacters,
# whitespace, paths, usernames, and endpoint syntax remain forbidden.
_SAFE_HOSTNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}[A-Za-z0-9]$|^[A-Za-z0-9]$")
_SAFE_CAPABILITY = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,95}$")
_MAX_PROBE_BYTES = 1024 * 1024
_MAX_TRANSFER_FILE_BYTES = 100 * 1024 * 1024 * 1024
_MAX_TRANSFER_ARTIFACTS = 100_000
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*$")
_TRANSFER_AFTER_FILE_READ_HOOK: Callable[[str, int], None] | None = None
_SENSITIVE_MARKERS = (
    "password",
    "passphrase",
    "private_key",
    "api_key",
    "authorization",
    "token",
    "secret",
    "credential",
    "bearer",
)


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _safe_identifier(value: Any, *, field: str) -> str:
    clean = str(value or "").strip().lower()
    if not _SAFE_ID.fullmatch(clean):
        raise ValueError(f"{field} must be a lowercase safe identifier")
    return clean


def _canonical_identifier(value: Any, *, field: str) -> str:
    raw = str(value or "")
    clean = _safe_identifier(raw, field=field)
    if raw != clean:
        raise ValueError(f"{field} must use its canonical lowercase representation")
    return clean


def _validated_digest(value: Any, *, field: str) -> str:
    clean = str(value or "")
    if not _SHA256_PATTERN.fullmatch(clean):
        raise ValueError(f"{field} must be a lowercase sha256 digest")
    return clean


def _normalized_security_label(value: Any) -> tuple[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized, normalized.replace("_", "")


def _contains_sensitive_marker(value: Any) -> bool:
    normalized, compact = _normalized_security_label(value)
    return any(
        marker in normalized or marker.replace("_", "") in compact
        for marker in _SENSITIVE_MARKERS
    )


def _safe_capabilities(value: Sequence[Any]) -> list[str]:
    result: list[str] = []
    for item in value:
        clean = str(item or "").strip().lower()
        if not _SAFE_CAPABILITY.fullmatch(clean):
            raise ValueError("capabilities must contain only safe lowercase labels")
        if clean not in result:
            result.append(clean)
    return sorted(result)


def _safe_remote_root(value: Any) -> str:
    clean = str(value or "").strip()
    path = PurePosixPath(clean)
    if (
        not clean
        or not path.is_absolute()
        or ".." in path.parts
        or any(ord(char) < 32 for char in clean)
    ):
        raise ValueError("remote_root must be an absolute normalized POSIX path")
    return str(path)


class ConnectionProfile(BaseModel):
    """Private local connection metadata; never an executable command template."""

    model_config = ConfigDict(extra="forbid")

    connection_id: str
    transport: Literal["ssh"] = "ssh"
    display_name: str = ""
    ssh_host_alias: str
    expected_hostname: str
    remote_root: str
    scheduler: Literal["direct"] = "direct"
    known_hosts_path: str = ""
    declared_capabilities: list[str] = Field(default_factory=list)
    max_concurrent_jobs: int = 1
    default_timeout_sec: int = 3600
    enabled: bool = True

    @field_validator("connection_id", mode="before")
    @classmethod
    def validate_connection_id(cls, value: Any) -> str:
        return _safe_identifier(value, field="connection_id")

    @field_validator("ssh_host_alias", mode="before")
    @classmethod
    def validate_ssh_alias(cls, value: Any) -> str:
        clean = str(value or "").strip()
        if not _SAFE_SSH_ALIAS.fullmatch(clean):
            raise ValueError("ssh_host_alias must be a safe SSH config alias")
        return clean

    @field_validator("expected_hostname", mode="before")
    @classmethod
    def validate_expected_hostname(cls, value: Any) -> str:
        clean = str(value or "").strip().lower()
        if not _SAFE_HOSTNAME.fullmatch(clean):
            raise ValueError("expected_hostname must be a safe short hostname")
        return clean

    @field_validator("remote_root", mode="before")
    @classmethod
    def validate_remote_root(cls, value: Any) -> str:
        return _safe_remote_root(value)

    @field_validator("known_hosts_path", mode="before")
    @classmethod
    def validate_known_hosts_path(cls, value: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        path = Path(clean).expanduser()
        if (
            not path.is_absolute()
            or ".." in path.parts
            or any(ord(char) < 32 for char in clean)
        ):
            raise ValueError("known_hosts_path must be an absolute local path")
        return str(path)

    @field_validator("declared_capabilities", mode="before")
    @classmethod
    def validate_declared_capabilities(cls, value: Any) -> list[str]:
        return _safe_capabilities(list(value or []))

    @field_validator("display_name", mode="before")
    @classmethod
    def validate_display_name(cls, value: Any) -> str:
        clean = str(value or "").strip()
        if len(clean) > 200 or any(ord(char) < 32 for char in clean):
            raise ValueError("display_name contains unsafe text")
        return clean

    @field_validator("max_concurrent_jobs", "default_timeout_sec", mode="before")
    @classmethod
    def validate_positive_int(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("connection numeric limits must be positive integers")
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("connection numeric limits must be positive integers")
        return parsed

    def digest(self) -> str:
        return _sha256(_canonical_bytes(self.model_dump(mode="json")))


class ResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gpu_count_max: int = 0
    cpu_threads_max: int = 1
    walltime_sec_max: int = 3600

    @field_validator("gpu_count_max", mode="before")
    @classmethod
    def validate_gpu_count(cls, value: Any) -> int:
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError("gpu_count_max must be a non-negative integer")
        return int(value)

    @field_validator("cpu_threads_max", "walltime_sec_max", mode="before")
    @classmethod
    def validate_positive_limits(cls, value: Any) -> int:
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError("resource limits must be positive integers")
        return int(value)


class ExecutionProfile(BaseModel):
    """Repository-owned task contract. It cannot contain arbitrary shell text."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    task_type: Literal["molecular_generation", "document_parsing", "model_training"]
    worker_entrypoint: Literal["molly-worker"] = "molly-worker"
    worker_action: Literal["execute"] = "execute"
    allowed_environment: str
    required_capabilities: list[str]
    resource_limits: ResourceLimits
    input_contract: str
    output_contract: str
    allowed_input_purposes: list[str]
    allowed_media_types: list[str]
    device_policy: Literal["cpu_only", "gpu_allowed", "gpu_required"]
    process_policy: Literal["nice_19_single_thread", "bounded_resources"]

    @field_validator("profile_id", "input_contract", "output_contract", mode="before")
    @classmethod
    def validate_ids(cls, value: Any, info: Any) -> str:
        return _safe_identifier(value, field=info.field_name)

    @field_validator("allowed_environment", mode="before")
    @classmethod
    def validate_environment(cls, value: Any) -> str:
        return _safe_identifier(value, field="allowed_environment")

    @field_validator("required_capabilities", mode="before")
    @classmethod
    def validate_capabilities(cls, value: Any) -> list[str]:
        result = _safe_capabilities(list(value or []))
        if not result:
            raise ValueError("required_capabilities must not be empty")
        return result

    @field_validator("allowed_input_purposes", mode="before")
    @classmethod
    def validate_allowed_purposes(cls, value: Any) -> list[str]:
        result = [_safe_identifier(item, field="input purpose") for item in list(value or [])]
        if not result or len(result) != len(set(result)):
            raise ValueError("allowed_input_purposes must be non-empty and unique")
        return sorted(result)

    @field_validator("allowed_media_types", mode="before")
    @classmethod
    def validate_allowed_media_types(cls, value: Any) -> list[str]:
        result: list[str] = []
        for item in list(value or []):
            clean = str(item or "").strip().lower()
            if not re.fullmatch(r"[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*", clean):
                raise ValueError("allowed_media_types contains an invalid media type")
            result.append(clean)
        if not result or len(result) != len(set(result)):
            raise ValueError("allowed_media_types must be non-empty and unique")
        return sorted(result)

    def digest(self) -> str:
        return _sha256(_canonical_bytes(self.model_dump(mode="json")))


EXECUTION_PROFILES: dict[str, ExecutionProfile] = {
    profile.profile_id: profile
    for profile in (
        ExecutionProfile(
            profile_id="reinvent4-cpu-v1",
            task_type="molecular_generation",
            allowed_environment="reinvent4",
            required_capabilities=["reinvent4", "cpu"],
            resource_limits=ResourceLimits(
                gpu_count_max=0,
                cpu_threads_max=1,
                walltime_sec_max=6 * 3600,
            ),
            input_contract="reinvent4-generation-input-v1",
            output_contract="reinvent4-generation-output-v1",
            allowed_input_purposes=["execution-request", "generator-config"],
            allowed_media_types=["application/json", "application/toml"],
            device_policy="cpu_only",
            process_policy="nice_19_single_thread",
        ),
        ExecutionProfile(
            profile_id="mineru-v1",
            task_type="document_parsing",
            allowed_environment="mineru",
            required_capabilities=["mineru", "gpu"],
            resource_limits=ResourceLimits(
                gpu_count_max=1,
                cpu_threads_max=16,
                walltime_sec_max=4 * 3600,
            ),
            input_contract="pdf-corpus-input-v1",
            output_contract="parsed-corpus-output-v1",
            allowed_input_purposes=["corpus-manifest", "source-pdf"],
            allowed_media_types=["application/json", "application/pdf"],
            device_policy="gpu_allowed",
            process_policy="bounded_resources",
        ),
        ExecutionProfile(
            profile_id="unimol-train-v1",
            task_type="model_training",
            allowed_environment="unimol",
            required_capabilities=["unimol", "gpu"],
            resource_limits=ResourceLimits(
                gpu_count_max=1,
                cpu_threads_max=16,
                walltime_sec_max=48 * 3600,
            ),
            input_contract="unimol-training-input-v1",
            output_contract="unimol-training-output-v1",
            allowed_input_purposes=["execution-request", "training-data", "training-config"],
            allowed_media_types=["application/csv", "application/json", "application/parquet"],
            device_policy="gpu_required",
            process_policy="bounded_resources",
        ),
    )
}


LEGACY_PINNED_PROFILE_BINDINGS: dict[str, tuple[str, str]] = {
    "molly-gpu-main-gpu_worker_main-reinvent4-v2": ("gpu-worker-main", "reinvent4-cpu-v1"),
    "molly-compute-main-compute_worker_main-reinvent4-v1": ("compute-worker-main", "reinvent4-cpu-v1"),
}


class CudaCapabilityDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "unavailable", "unknown"] = "unknown"
    device_name: str = ""
    compute_capability: str = ""
    driver_version: str = ""
    runtime_version: str = ""
    toolkit_version: str = ""
    pytorch_cuda_version: str = ""
    cudnn_version: str = ""

    @field_validator(
        "device_name",
        "compute_capability",
        "driver_version",
        "runtime_version",
        "toolkit_version",
        "pytorch_cuda_version",
        "cudnn_version",
        mode="before",
    )
    @classmethod
    def validate_safe_probe_text(cls, value: Any) -> str:
        clean = str(value or "").strip()
        if (
            len(clean) > 160
            or any(ord(char) < 32 for char in clean)
            or _contains_sensitive_marker(clean)
        ):
            raise ValueError("capability probe text is invalid")
        return clean


class CapabilityDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_threads: int | None = None
    memory_bytes: int | None = None
    cuda: CudaCapabilityDetails | None = None
    software_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("cpu_threads", "memory_bytes", mode="before")
    @classmethod
    def validate_non_negative_numbers(cls, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or int(value) < 0:
            raise ValueError("capability numeric detail must be non-negative")
        return int(value)

    @field_validator("software_versions")
    @classmethod
    def validate_software_versions(cls, value: dict[str, str]) -> dict[str, str]:
        _reject_sensitive_keys(value)
        result: dict[str, str] = {}
        for key, version in value.items():
            clean_key = _safe_identifier(key, field="software name")
            clean_version = str(version or "").strip()
            if (
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}", clean_version)
                or _contains_sensitive_marker(clean_version)
            ):
                raise ValueError("software versions must contain version labels only")
            result[clean_key] = clean_version
        return dict(sorted(result.items()))


class CapabilityProbeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[CAPABILITY_PROBE_SCHEMA_VERSION] = CAPABILITY_PROBE_SCHEMA_VERSION
    connection_id: str
    connection_profile_digest: str
    status: Literal["available", "unavailable", "mismatch"]
    checked_at: str
    hostname: str = ""
    verified_capabilities: list[str] = Field(default_factory=list)
    details: CapabilityDetails = Field(default_factory=CapabilityDetails)
    error_code: str = ""

    @field_validator("connection_id", mode="before")
    @classmethod
    def validate_connection_id(cls, value: Any) -> str:
        return _safe_identifier(value, field="connection_id")

    @field_validator("verified_capabilities", mode="before")
    @classmethod
    def validate_verified_capabilities(cls, value: Any) -> list[str]:
        result = _safe_capabilities(list(value or []))
        if any(_contains_sensitive_marker(item) for item in result):
            raise ValueError("probe capability labels must not contain credential markers")
        return result

    @field_validator("hostname", mode="before")
    @classmethod
    def validate_hostname(cls, value: Any) -> str:
        clean = str(value or "").strip().lower()
        if clean and (
            not _SAFE_HOSTNAME.fullmatch(clean) or _contains_sensitive_marker(clean)
        ):
            raise ValueError("probe hostname is invalid")
        return clean


@dataclass(frozen=True)
class ResourceProfileAuthoritySnapshot:
    """One non-mixed private snapshot used by resource authorization."""

    connection: ConnectionProfile
    probe: CapabilityProbeResult | None
    source_digest: str
    profile_capability_digest: str = ""


class TransferArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str
    purpose: str
    media_type: str
    size_bytes: int
    sha256: str

    @field_validator("relative_path", mode="before")
    @classmethod
    def validate_relative_path(cls, value: Any) -> str:
        raw = str(value or "")
        pure = PurePosixPath(raw)
        if (
            not raw
            or raw != pure.as_posix()
            or pure.is_absolute()
            or raw.startswith("/")
            or ".." in pure.parts
            or "." in pure.parts
            or "\\" in raw
            or any(ord(char) < 32 for char in raw)
        ):
            raise ValueError("transfer artifact path must be a canonical safe relative path")
        return raw

    @field_validator("purpose", mode="before")
    @classmethod
    def validate_purpose(cls, value: Any) -> str:
        return _canonical_identifier(value, field="artifact purpose")

    @field_validator("media_type", mode="before")
    @classmethod
    def validate_media_type(cls, value: Any) -> str:
        raw = str(value or "")
        if not _MEDIA_TYPE_PATTERN.fullmatch(raw):
            raise ValueError("transfer artifact media_type must be canonical and valid")
        return raw

    @field_validator("size_bytes", mode="before")
    @classmethod
    def validate_size_bytes(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("transfer artifact size must be a non-negative integer")
        parsed = value
        if parsed < 0 or parsed > _MAX_TRANSFER_FILE_BYTES:
            raise ValueError("transfer artifact size is outside the allowed range")
        return parsed

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: Any) -> str:
        return _validated_digest(value, field="artifact sha256")


class TransferManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[TRANSFER_MANIFEST_SCHEMA_VERSION] = TRANSFER_MANIFEST_SCHEMA_VERSION
    request_id: str
    connection_id: str
    connection_profile_digest: str
    execution_profile_id: str
    execution_profile_digest: str
    target_purpose: str
    artifacts: tuple[TransferArtifact, ...] = Field(
        min_length=1,
        max_length=_MAX_TRANSFER_ARTIFACTS,
    )
    total_size_bytes: int
    roster_sha256: str
    manifest_sha256: str

    @field_validator(
        "request_id",
        "connection_id",
        "execution_profile_id",
        "target_purpose",
        mode="before",
    )
    @classmethod
    def validate_identifiers(cls, value: Any, info: Any) -> str:
        return _canonical_identifier(value, field=info.field_name)

    @field_validator(
        "connection_profile_digest",
        "execution_profile_digest",
        "roster_sha256",
        "manifest_sha256",
        mode="before",
    )
    @classmethod
    def validate_digests(cls, value: Any, info: Any) -> str:
        return _validated_digest(value, field=info.field_name)

    @field_validator("total_size_bytes", mode="before")
    @classmethod
    def validate_total_size(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("transfer manifest total size must be a non-negative integer")
        parsed = value
        if parsed < 0:
            raise ValueError("transfer manifest total size must be a non-negative integer")
        return parsed

    @model_validator(mode="after")
    def validate_manifest_digest(self) -> "TransferManifest":
        paths = [item.relative_path for item in self.artifacts]
        if paths != sorted(paths):
            raise ValueError("transfer manifest artifacts must use deterministic path order")
        if len(paths) != len(set(paths)):
            raise ValueError("transfer manifest artifact paths must be unique")
        execution_profile = EXECUTION_PROFILES.get(self.execution_profile_id)
        if execution_profile is None:
            raise ValueError("transfer manifest execution profile is not allowlisted")
        if self.execution_profile_digest != execution_profile.digest():
            raise ValueError("transfer manifest execution profile digest mismatch")
        if self.target_purpose != execution_profile.task_type.replace("_", "-"):
            raise ValueError("transfer manifest target purpose does not match execution profile")
        for artifact in self.artifacts:
            if artifact.purpose not in execution_profile.allowed_input_purposes:
                raise ValueError("transfer manifest artifact purpose is not allowlisted")
            if artifact.media_type not in execution_profile.allowed_media_types:
                raise ValueError("transfer manifest artifact media type is not allowlisted")
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        if self.manifest_sha256 != _sha256(_canonical_bytes(payload)):
            raise ValueError("transfer manifest digest mismatch")
        if self.total_size_bytes != sum(item.size_bytes for item in self.artifacts):
            raise ValueError("transfer manifest total size mismatch")
        roster = [item.model_dump(mode="json") for item in self.artifacts]
        if self.roster_sha256 != _sha256(_canonical_bytes({"artifacts": roster})):
            raise ValueError("transfer manifest roster digest mismatch")
        return self


def verify_transfer_manifest_binding(
    manifest: TransferManifest | Mapping[str, Any],
    *,
    connection: ConnectionProfile,
    execution_profile: ExecutionProfile,
) -> TransferManifest:
    """Verify an imported manifest against the exact currently selected profiles."""

    raw = manifest.model_dump(mode="json") if isinstance(manifest, TransferManifest) else manifest
    validated = TransferManifest.model_validate(raw)
    expected_target = execution_profile.task_type.replace("_", "-")
    if not connection.enabled:
        raise ValueError("transfer manifest connection profile is disabled")
    if (
        validated.connection_id != connection.connection_id
        or validated.connection_profile_digest != connection.digest()
    ):
        raise ValueError("transfer manifest connection profile binding mismatch")
    if (
        validated.execution_profile_id != execution_profile.profile_id
        or validated.execution_profile_digest != execution_profile.digest()
        or validated.target_purpose != expected_target
    ):
        raise ValueError("transfer manifest execution profile binding mismatch")
    missing_capabilities = sorted(
        set(execution_profile.required_capabilities).difference(connection.declared_capabilities)
    )
    if missing_capabilities:
        raise ValueError("transfer manifest connection capability binding mismatch")
    for artifact in validated.artifacts:
        if artifact.purpose not in execution_profile.allowed_input_purposes:
            raise ValueError("transfer manifest artifact purpose binding mismatch")
        if artifact.media_type not in execution_profile.allowed_media_types:
            raise ValueError("transfer manifest artifact media type binding mismatch")
    return validated


class ResourceProfileStore:
    """Single user-level source for private connection profiles and probe telemetry."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.workspace_dir = _absolute_config_path(workspace_dir)
        env = environ if environ is not None else os.environ
        root = config_dir or env.get("MOLLY_CONFIG_DIR") or user_config_path("Molly", appauthor=False)
        self.config_dir = _absolute_config_path(root)
        self.path = self.config_dir / "connections.json"
        self.previous_path = self.config_dir / "connection_profiles.json"
        self.probes_path = self.config_dir / "capability_probes.json"
        self.lock_path = self.config_dir / ".resource_profiles.lock"
        self.legacy_path = self.workspace_dir / "workers" / "remote_workers.json"
        self._lock = threading.RLock()

    def list_connections(self, *, include_disabled: bool = False) -> list[ConnectionProfile]:
        with self._lock, self._process_lock() as config_fd:
            profiles = self._read_profiles_locked(config_fd)
        if include_disabled:
            return profiles
        return [profile for profile in profiles if profile.enabled]

    def get_connection(self, connection_id: str) -> ConnectionProfile:
        clean = _safe_identifier(connection_id, field="connection_id")
        for profile in self.list_connections(include_disabled=True):
            if profile.connection_id == clean:
                return profile
        raise ValueError(f"connection profile not found: {clean}")

    def save_connection(self, profile: ConnectionProfile) -> ConnectionProfile:
        validated = ConnectionProfile.model_validate(profile.model_dump(mode="json"))
        with self._lock, self._process_lock() as config_fd:
            profiles = self._read_profiles_locked(config_fd)
            profiles = [item for item in profiles if item.connection_id != validated.connection_id]
            profiles.append(validated)
            self._write_profiles_locked(config_fd, profiles)
        return validated

    def delete_connection(self, connection_id: str) -> bool:
        clean = _safe_identifier(connection_id, field="connection_id")
        with self._lock, self._process_lock() as config_fd:
            profiles = self._read_profiles_locked(config_fd)
            kept = [item for item in profiles if item.connection_id != clean]
            if len(kept) == len(profiles):
                return False
            self._write_profiles_locked(config_fd, kept)
            probes = self._read_probe_payload_locked(config_fd)
            probes.pop(clean, None)
            self._write_probe_payload_locked(config_fd, probes)
        return True

    def get_last_probe(self, connection_id: str) -> CapabilityProbeResult | None:
        clean = _safe_identifier(connection_id, field="connection_id")
        with self._lock, self._process_lock() as config_fd:
            payload = self._read_probe_payload_locked(config_fd).get(clean)
        if not isinstance(payload, dict):
            return None
        try:
            return CapabilityProbeResult.model_validate(payload)
        except ValueError:
            return None

    def authority_snapshot(
        self,
        connection_id: str,
        *,
        execution_profile_id: str | None = None,
    ) -> ResourceProfileAuthoritySnapshot:
        """Read one connection and its last probe under the same process lock.

        Unlike the public convenience getters, corrupt probe bytes fail closed
        instead of being projected as a missing probe.
        """

        clean = _safe_identifier(connection_id, field="connection_id")
        with self._lock, self._process_lock() as config_fd:
            profiles = self._read_profiles_locked(config_fd)
            raw_probes = self._read_probe_payload_locked(config_fd)
            connection = next(
                (
                    profile
                    for profile in profiles
                    if profile.connection_id == clean
                ),
                None,
            )
            if connection is None:
                raise ValueError(f"connection profile not found: {clean}")
            raw_probe = raw_probes.get(clean)
            probe = (
                None
                if raw_probe is None
                else CapabilityProbeResult.model_validate(raw_probe)
            )
            source_material = {
                "schema_version": "resource_profile_authority_snapshot.v1",
                "connection_id": clean,
                "connection_profile_digest": connection.digest(),
                "capability_probe_digest": (
                    "" if probe is None else _sha256(_canonical_bytes(probe.model_dump(mode="json")))
                ),
            }
            profile_capability_digest = ""
            if execution_profile_id is not None:
                execution_profile = self.resolve_execution_profile(execution_profile_id)
                required = set(execution_profile.required_capabilities)
                matching_connections: list[dict[str, Any]] = []
                for candidate in profiles:
                    candidate_raw_probe = raw_probes.get(candidate.connection_id)
                    candidate_probe = (
                        None
                        if candidate_raw_probe is None
                        else CapabilityProbeResult.model_validate(candidate_raw_probe)
                    )
                    probe_matches = bool(
                        candidate_probe is not None
                        and candidate_probe.connection_profile_digest == candidate.digest()
                    )
                    declared_ready = bool(
                        candidate.enabled
                        and required.issubset(candidate.declared_capabilities)
                    )
                    verified_ready = bool(
                        declared_ready
                        and probe_matches
                        and candidate_probe is not None
                        and candidate_probe.status == "available"
                        and required.issubset(candidate_probe.verified_capabilities)
                    )
                    matching_connections.append(
                        {
                            "connection_digest": candidate.digest(),
                            "enabled": bool(candidate.enabled),
                            "declared_capabilities": sorted(candidate.declared_capabilities),
                            "probe_digest": (
                                _sha256(
                                    _canonical_bytes(
                                        candidate_probe.model_dump(mode="json")
                                    )
                                )
                                if probe_matches and candidate_probe is not None
                                else ""
                            ),
                            "probe_status": (
                                candidate_probe.status
                                if probe_matches and candidate_probe is not None
                                else "unknown"
                            ),
                            "probe_matches_connection_digest": probe_matches,
                            "declared_ready": declared_ready,
                            "verified_ready": verified_ready,
                        }
                    )
                profile_capability_digest = _sha256(
                    _canonical_bytes(
                        {
                            "profile_id": execution_profile.profile_id,
                            "profile_digest": execution_profile.digest(),
                            "connections": sorted(
                                matching_connections,
                                key=lambda item: item["connection_digest"],
                            ),
                        }
                    )
                )
                source_material["profile_capability_digest"] = profile_capability_digest
        return ResourceProfileAuthoritySnapshot(
            connection=connection,
            probe=probe,
            source_digest=_sha256(_canonical_bytes(source_material)),
            profile_capability_digest=profile_capability_digest,
        )

    def save_probe(self, result: CapabilityProbeResult) -> CapabilityProbeResult:
        validated = CapabilityProbeResult.model_validate(result.model_dump(mode="json"))
        with self._lock, self._process_lock() as config_fd:
            current = next(
                (
                    profile
                    for profile in self._read_profiles_locked(config_fd)
                    if profile.connection_id == validated.connection_id
                ),
                None,
            )
            if current is None or current.digest() != validated.connection_profile_digest:
                raise ValueError("connection profile changed during capability probe")
            payload = self._read_probe_payload_locked(config_fd)
            payload[validated.connection_id] = validated.model_dump(mode="json")
            self._write_probe_payload_locked(config_fd, payload)
        return validated

    def resolve_execution_profile(self, profile_id: str) -> ExecutionProfile:
        clean = _safe_identifier(profile_id, field="execution_profile_id")
        profile = EXECUTION_PROFILES.get(clean)
        if profile is None:
            raise ValueError(f"execution profile is not allowed: {clean}")
        return profile

    def resolve_legacy_pinned_profile(
        self,
        legacy_profile_id: str,
    ) -> tuple[ConnectionProfile, ExecutionProfile]:
        binding = LEGACY_PINNED_PROFILE_BINDINGS.get(str(legacy_profile_id or "").strip())
        if binding is None:
            raise ValueError("legacy pinned transport profile is not allowed")
        connection_id, execution_profile_id = binding
        return self.get_connection(connection_id), self.resolve_execution_profile(execution_profile_id)

    def public_state(self) -> dict[str, Any]:
        connections = self.list_connections(include_disabled=True)
        probes = {
            connection.connection_id: self.get_last_probe(connection.connection_id)
            for connection in connections
        }
        return {
            "schema_version": CONNECTION_PROFILE_SCHEMA_VERSION,
            "connections": [
                {
                    **profile.model_dump(mode="json"),
                    "connection_profile_digest": profile.digest(),
                    "last_probe": (
                        probe.model_dump(mode="json")
                        if (
                            (probe := probes.get(profile.connection_id)) is not None
                            and probe.connection_profile_digest == profile.digest()
                        )
                        else None
                    ),
                }
                for profile in connections
            ],
            "execution_profiles": [
                {
                    **profile.model_dump(mode="json"),
                    "execution_profile_digest": profile.digest(),
                }
                for profile in sorted(EXECUTION_PROFILES.values(), key=lambda item: item.profile_id)
            ],
        }

    def _read_profiles_locked(self, config_fd: int) -> list[ConnectionProfile]:
        payload = self._read_json_file(config_fd, self.path.name)
        if payload is None:
            previous = self._read_json_file(config_fd, self.previous_path.name)
        else:
            previous = None
        if previous is not None:
            if previous.get("schema_version") != CONNECTION_PROFILE_SCHEMA_VERSION:
                raise ValueError("unsupported previous connection profile schema")
            raw = previous.get("connections")
            if not isinstance(raw, list):
                raise ValueError("previous connection profile roster must be a list")
            self._write_profiles_locked(
                config_fd,
                [ConnectionProfile.model_validate(item) for item in raw]
            )
            self._replace_private_config_with_tombstone(
                config_fd, self.previous_path.name
            )
            payload = self._read_json_file(config_fd, self.path.name)
        if payload is None:
            self._migrate_legacy_locked(config_fd)
            payload = self._read_json_file(config_fd, self.path.name)
        if payload is None:
            return []
        if payload.get("schema_version") != CONNECTION_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported connection profile schema")
        raw = payload.get("connections")
        if not isinstance(raw, list):
            raise ValueError("connection profile roster must be a list")
        profiles = [ConnectionProfile.model_validate(item) for item in raw]
        ids = [profile.connection_id for profile in profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate connection profile ID")
        return sorted(profiles, key=lambda item: item.connection_id)

    def _write_profiles_locked(
        self, config_fd: int, profiles: Sequence[ConnectionProfile]
    ) -> None:
        payload = {
            "schema_version": CONNECTION_PROFILE_SCHEMA_VERSION,
            "updated_at": now_iso(),
            "connections": [
                item.model_dump(mode="json")
                for item in sorted(profiles, key=lambda profile: profile.connection_id)
            ],
        }
        self._secure_write_json(config_fd, self.path.name, payload)

    def _read_probe_payload_locked(self, config_fd: int) -> dict[str, Any]:
        payload = self._read_json_file(config_fd, self.probes_path.name)
        if payload is None:
            return {}
        if payload.get("schema_version") != CAPABILITY_PROBE_SCHEMA_VERSION:
            raise ValueError("unsupported capability probe schema")
        probes = payload.get("probes")
        if not isinstance(probes, dict):
            raise ValueError("capability probe roster must be an object")
        return dict(probes)

    def _write_probe_payload_locked(
        self, config_fd: int, probes: Mapping[str, Any]
    ) -> None:
        self._secure_write_json(
            config_fd,
            self.probes_path.name,
            {
                "schema_version": CAPABILITY_PROBE_SCHEMA_VERSION,
                "updated_at": now_iso(),
                "probes": dict(sorted(probes.items())),
            },
        )

    def _migrate_legacy_locked(self, config_fd: int) -> None:
        try:
            legacy_parent_fd = _open_private_directory(
                self.legacy_path.parent,
                create=False,
                enforce_private_mode=False,
            )
        except ValueError:
            return
        try:
            legacy = self._read_json_file(
                legacy_parent_fd, self.legacy_path.name
            )
            if legacy is None:
                return
            converted: list[ConnectionProfile] = []
            workers = legacy.get("workers")
            for item in workers if isinstance(workers, list) else []:
                if (
                    not isinstance(item, dict)
                    or str(item.get("transport") or "ssh") != "ssh"
                ):
                    continue
                host = str(item.get("host") or "").strip()
                work_dir = str(item.get("work_dir") or "/tmp/molly-runs").strip()
                if not host:
                    continue
                try:
                    converted.append(
                        ConnectionProfile(
                            connection_id=str(item.get("worker_id") or ""),
                            ssh_host_alias=host,
                            expected_hostname=host.split(".", 1)[0],
                            remote_root=work_dir,
                            display_name=str(item.get("display_name") or ""),
                            declared_capabilities=item.get("capabilities") or [],
                            max_concurrent_jobs=item.get("max_concurrent_jobs", 1),
                            default_timeout_sec=item.get("default_timeout_sec", 3600),
                            enabled=item.get("enabled", True),
                        )
                    )
                except ValueError:
                    continue
            if converted:
                self._write_profiles_locked(config_fd, converted)
                self._replace_private_config_with_tombstone(
                    legacy_parent_fd, self.legacy_path.name
                )
        finally:
            os.close(legacy_parent_fd)

    @staticmethod
    def _replace_private_config_with_tombstone(
        directory_fd: int, name: str
    ) -> None:
        _write_private_json(
            directory_fd,
            name,
            {
                "schema_version": "remote_workers.migrated.v1",
                "migrated_at": now_iso(),
                "connection_metadata_removed": True,
            },
        )

    @staticmethod
    def _read_json_file(
        directory_fd: int, name: str
    ) -> dict[str, Any] | None:
        payload = _read_private_json(directory_fd, name)
        if payload is None:
            return None
        _reject_sensitive_keys(payload)
        return payload

    @staticmethod
    def _secure_write_json(
        directory_fd: int, name: str, payload: Mapping[str, Any]
    ) -> None:
        _reject_sensitive_keys(payload)
        _write_private_json(directory_fd, name, payload)

    @contextmanager
    def _process_lock(self):
        with _private_process_lock(
            self.config_dir, self.lock_path.name
        ) as config_fd:
            yield config_fd


class CapabilityProbeService:
    """Runs exactly one fixed, read-only worker probe entrypoint over SSH."""

    def __init__(
        self,
        *,
        store: ResourceProfileStore,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
    ) -> None:
        self.store = store
        self.runner = runner

    def probe(self, connection_id: str) -> CapabilityProbeResult:
        profile = self.store.get_connection(connection_id)
        if not profile.enabled:
            raise ValueError("connection profile is disabled")
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        if profile.known_hosts_path:
            command.extend(
                [
                    "-o",
                    f"UserKnownHostsFile={profile.known_hosts_path}",
                ]
            )
        command.extend([profile.ssh_host_alias, "--", "molly-worker", "probe", "--json"])
        checked_at = now_iso()
        try:
            completed = self.runner(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(profile.default_timeout_sec, 60),
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            result = CapabilityProbeResult(
                connection_id=profile.connection_id,
                connection_profile_digest=profile.digest(),
                status="unavailable",
                checked_at=checked_at,
                error_code="probe_transport_failed",
            )
            return self.store.save_probe(result)
        stdout = bytes(completed.stdout or b"")
        if completed.returncode == 255:
            result = CapabilityProbeResult(
                connection_id=profile.connection_id,
                connection_profile_digest=profile.digest(),
                status="unavailable",
                checked_at=checked_at,
                error_code="probe_transport_failed",
            )
            return self.store.save_probe(result)
        if completed.returncode != 0 or not stdout or len(stdout) > _MAX_PROBE_BYTES:
            result = CapabilityProbeResult(
                connection_id=profile.connection_id,
                connection_profile_digest=profile.digest(),
                status="unavailable",
                checked_at=checked_at,
                error_code="probe_response_unavailable",
            )
            return self.store.save_probe(result)
        try:
            payload = json.loads(stdout.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("probe response must be an object")
            _reject_sensitive_keys(payload)
            hostname = str(payload.get("hostname") or "").strip().lower()
            if not _SAFE_HOSTNAME.fullmatch(hostname) or _contains_sensitive_marker(hostname):
                raise ValueError("probe hostname is invalid")
            capabilities = _safe_capabilities(list(payload.get("capabilities") or []))
            if any(_contains_sensitive_marker(item) for item in capabilities):
                raise ValueError("probe capability label is invalid")
            details = CapabilityDetails.model_validate(payload.get("details") or {})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            result = CapabilityProbeResult(
                connection_id=profile.connection_id,
                connection_profile_digest=profile.digest(),
                status="unavailable",
                checked_at=checked_at,
                error_code="probe_response_invalid",
            )
            return self.store.save_probe(result)
        status_value: Literal["available", "mismatch"] = (
            "available" if hostname == profile.expected_hostname else "mismatch"
        )
        result = CapabilityProbeResult(
            connection_id=profile.connection_id,
            connection_profile_digest=profile.digest(),
            status=status_value,
            checked_at=checked_at,
            hostname=hostname,
            verified_capabilities=capabilities,
            details=details,
            error_code="" if status_value == "available" else "hostname_mismatch",
        )
        return self.store.save_probe(result)


@dataclass(frozen=True)
class _PinnedTransferDirectory:
    relative_path: str
    descriptor: int
    parent_relative_path: str | None
    name: str
    initial_stat: os.stat_result


@dataclass(frozen=True)
class _PinnedTransferFile:
    relative_path: str
    descriptor: int
    parent_relative_path: str
    name: str
    initial_stat: os.stat_result


def build_transfer_manifest(
    *,
    request_id: str,
    input_root: Path,
    artifacts: Sequence[Mapping[str, str]],
    connection: ConnectionProfile,
    execution_profile: ExecutionProfile,
    target_purpose: str,
) -> TransferManifest:
    clean_request_id = _canonical_identifier(request_id, field="request_id")
    clean_purpose = _canonical_identifier(target_purpose, field="target_purpose")
    if not connection.enabled:
        raise ValueError("transfer connection profile is disabled")
    expected_target = execution_profile.task_type.replace("_", "-")
    if clean_purpose != expected_target:
        raise ValueError("target_purpose does not match execution profile task type")
    missing_capabilities = sorted(
        set(execution_profile.required_capabilities).difference(connection.declared_capabilities)
    )
    if missing_capabilities:
        raise ValueError(
            "connection does not declare required execution capabilities: "
            + ", ".join(missing_capabilities)
        )
    descriptor_by_path: dict[str, TransferArtifact] = {}
    for raw in artifacts:
        candidate = TransferArtifact(
            relative_path=raw.get("relative_path"),
            purpose=raw.get("purpose"),
            media_type=raw.get("media_type"),
            size_bytes=0,
            sha256="sha256:" + ("0" * 64),
        )
        if candidate.relative_path in descriptor_by_path:
            raise ValueError("transfer artifact descriptor paths must be unique")
        if candidate.purpose not in execution_profile.allowed_input_purposes:
            raise ValueError("transfer artifact purpose is not allowed by execution profile")
        if candidate.media_type not in execution_profile.allowed_media_types:
            raise ValueError("transfer artifact media_type is not allowed by execution profile")
        descriptor_by_path[candidate.relative_path] = candidate
    if not descriptor_by_path:
        raise ValueError("transfer manifest requires at least one artifact")
    if len(descriptor_by_path) > _MAX_TRANSFER_ARTIFACTS:
        raise ValueError("transfer manifest exceeds artifact count limit")

    root_parent_fd, root_fd, root_name = _open_transfer_root(Path(input_root))
    directories: dict[str, _PinnedTransferDirectory] = {}
    files: dict[str, _PinnedTransferFile] = {}
    try:
        directories[""] = _PinnedTransferDirectory(
            relative_path="",
            descriptor=root_fd,
            parent_relative_path=None,
            name=root_name,
            initial_stat=os.fstat(root_fd),
        )
        _pin_transfer_tree(
            directory_relative_path="",
            directories=directories,
            files=files,
        )
        if set(descriptor_by_path) != set(files):
            raise ValueError("transfer artifact descriptors must cover the complete input roster")

        records: list[TransferArtifact] = []
        for index, relative_path in enumerate(sorted(files)):
            pinned = files[relative_path]
            digest, size = _hash_pinned_transfer_file(pinned)
            descriptor = descriptor_by_path[relative_path]
            records.append(
                TransferArtifact(
                    relative_path=relative_path,
                    purpose=descriptor.purpose,
                    media_type=descriptor.media_type,
                    size_bytes=size,
                    sha256=digest,
                )
            )
            if _TRANSFER_AFTER_FILE_READ_HOOK is not None:
                _TRANSFER_AFTER_FILE_READ_HOOK(relative_path, index)

        roster = [item.model_dump(mode="json") for item in records]
        payload: dict[str, Any] = {
            "schema_version": TRANSFER_MANIFEST_SCHEMA_VERSION,
            "request_id": clean_request_id,
            "connection_id": connection.connection_id,
            "connection_profile_digest": connection.digest(),
            "execution_profile_id": execution_profile.profile_id,
            "execution_profile_digest": execution_profile.digest(),
            "target_purpose": clean_purpose,
            "artifacts": roster,
            "total_size_bytes": sum(item.size_bytes for item in records),
            "roster_sha256": _sha256(_canonical_bytes({"artifacts": roster})),
        }
        payload["manifest_sha256"] = _sha256(_canonical_bytes(payload))
        manifest = TransferManifest.model_validate(payload)
        manifest = verify_transfer_manifest_binding(
            manifest,
            connection=connection,
            execution_profile=execution_profile,
        )
        if set(files) != _scan_transfer_roster(root_fd):
            raise ValueError("transfer input roster changed while being read")
        _verify_pinned_transfer_tree(
            root_parent_fd=root_parent_fd,
            directories=directories,
            files=files,
            records=records,
        )
        return manifest
    finally:
        for pinned_file in files.values():
            os.close(pinned_file.descriptor)
        for pinned_directory in reversed(tuple(directories.values())):
            os.close(pinned_directory.descriptor)
        os.close(root_parent_fd)


def _reject_sensitive_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if _contains_sensitive_marker(key):
                raise ValueError("resource profiles and probe telemetry must not contain credentials")
            _reject_sensitive_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive_keys(item)


def _open_transfer_root(path: Path) -> tuple[int, int, str]:
    absolute = Path(os.path.abspath(path.expanduser()))
    if absolute == Path("/"):
        raise ValueError("transfer input root must not be the filesystem root")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open("/", directory_flags)
    try:
        parts = absolute.parts[1:]
        for component in parts[:-1]:
            next_fd = os.open(
                component,
                directory_flags,
                dir_fd=parent_fd,
            )
            os.close(parent_fd)
            parent_fd = next_fd
        name = parts[-1]
        root_fd = os.open(name, directory_flags, dir_fd=parent_fd)
        return parent_fd, root_fd, name
    except OSError as exc:
        os.close(parent_fd)
        raise ValueError(
            "transfer input root must contain only non-symlink directory components"
        ) from exc


def _pin_transfer_tree(
    *,
    directory_relative_path: str,
    directories: dict[str, _PinnedTransferDirectory],
    files: dict[str, _PinnedTransferFile],
) -> None:
    directory = directories[directory_relative_path]
    with os.scandir(directory.descriptor) as iterator:
        entries = sorted(iterator, key=lambda item: item.name)
    for entry in entries:
        if entry.is_symlink():
            raise ValueError("transfer input roster must not contain symlinks")
        relative_path = (
            f"{directory_relative_path}/{entry.name}"
            if directory_relative_path
            else entry.name
        )
        if entry.is_dir(follow_symlinks=False):
            child_fd = os.open(
                entry.name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory.descriptor,
            )
            directories[relative_path] = _PinnedTransferDirectory(
                relative_path=relative_path,
                descriptor=child_fd,
                parent_relative_path=directory_relative_path,
                name=entry.name,
                initial_stat=os.fstat(child_fd),
            )
            _pin_transfer_tree(
                directory_relative_path=relative_path,
                directories=directories,
                files=files,
            )
            continue
        if not entry.is_file(follow_symlinks=False):
            raise ValueError("transfer input roster contains an unsupported entry")
        file_fd = os.open(
            entry.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory.descriptor,
        )
        initial = os.fstat(file_fd)
        if not stat.S_ISREG(initial.st_mode) or initial.st_size > _MAX_TRANSFER_FILE_BYTES:
            os.close(file_fd)
            raise ValueError("transfer artifact is not an allowed regular file")
        files[relative_path] = _PinnedTransferFile(
            relative_path=relative_path,
            descriptor=file_fd,
            parent_relative_path=directory_relative_path,
            name=entry.name,
            initial_stat=initial,
        )


def _stat_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _hash_file_descriptor(descriptor: int) -> tuple[str, int]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > _MAX_TRANSFER_FILE_BYTES:
            raise ValueError("transfer artifact exceeds size limit")
        digest.update(chunk)
    return f"sha256:{digest.hexdigest()}", size


def _hash_pinned_transfer_file(pinned: _PinnedTransferFile) -> tuple[str, int]:
    before = os.fstat(pinned.descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(pinned.initial_stat):
        raise ValueError("transfer artifact changed before being read")
    digest, size = _hash_file_descriptor(pinned.descriptor)
    after = os.fstat(pinned.descriptor)
    if _stat_fingerprint(before) != _stat_fingerprint(after):
        raise ValueError("transfer artifact changed while being read")
    return digest, size


def _verify_pinned_transfer_tree(
    *,
    root_parent_fd: int,
    directories: Mapping[str, _PinnedTransferDirectory],
    files: Mapping[str, _PinnedTransferFile],
    records: Sequence[TransferArtifact],
) -> None:
    record_by_path = {record.relative_path: record for record in records}
    for relative_path, pinned in directories.items():
        current = os.fstat(pinned.descriptor)
        if relative_path:
            assert pinned.parent_relative_path is not None
            parent_fd = directories[pinned.parent_relative_path].descriptor
        else:
            parent_fd = root_parent_fd
        named = os.stat(pinned.name, dir_fd=parent_fd, follow_symlinks=False)
        expected = _stat_fingerprint(pinned.initial_stat)
        if _stat_fingerprint(current) != expected or _stat_fingerprint(named) != expected:
            raise ValueError("transfer input directory tree changed while being read")
    for relative_path, pinned in files.items():
        current = os.fstat(pinned.descriptor)
        parent_fd = directories[pinned.parent_relative_path].descriptor
        named = os.stat(pinned.name, dir_fd=parent_fd, follow_symlinks=False)
        expected = _stat_fingerprint(pinned.initial_stat)
        if _stat_fingerprint(current) != expected or _stat_fingerprint(named) != expected:
            raise ValueError("transfer artifact changed after being read")
        digest, size = _hash_file_descriptor(pinned.descriptor)
        after_rehash = os.fstat(pinned.descriptor)
        record = record_by_path[relative_path]
        if (
            _stat_fingerprint(after_rehash) != expected
            or digest != record.sha256
            or size != record.size_bytes
        ):
            raise ValueError("transfer artifact content changed after being read")


def _scan_transfer_roster(
    directory_fd: int,
    *,
    prefix: PurePosixPath | None = None,
) -> set[str]:
    roster: set[str] = set()
    with os.scandir(directory_fd) as iterator:
        entries = list(iterator)
    for entry in entries:
        relative = (prefix / entry.name) if prefix is not None else PurePosixPath(entry.name)
        if entry.is_symlink():
            raise ValueError("transfer input roster must not contain symlinks")
        if entry.is_file(follow_symlinks=False):
            roster.add(relative.as_posix())
            continue
        if not entry.is_dir(follow_symlinks=False):
            raise ValueError("transfer input roster contains an unsupported entry")
        child_fd = os.open(
            entry.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            roster.update(_scan_transfer_roster(child_fd, prefix=relative))
        finally:
            os.close(child_fd)
    return roster
