"""Restricted, server-owned runtime installation.

The discovery layer in :mod:`molly.web.environments` is deliberately read
only.  This module is the next, separate boundary: it accepts only a plan
created from a server-owned compatibility manifest and performs installation
inside a newly allocated runtime directory.  Neither a browser nor an LLM
can provide a command, package version, URL, or destination path.

The implementation uses ordinary files and Python's standard library so the
same state machine can be exercised with a local fixture and with a simulated
SSH transport.  The default scientific manifest intentionally contains no
invented hashes; entries without a hash are displayed as blocked and cannot
be approved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile
import threading
import time
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit
import urllib.request
import zipfile

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

from molly.core.ids import (
    canonical_json_bytes,
    new_server_id,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_identifier,
    validate_sha256,
)

from .environments import (
    COMPATIBILITY_CATALOG,
    COMPATIBILITY_CATALOG_VERSION,
    CompatibilityEntry,
    EnvironmentManager,
    EnvironmentProfile,
    EnvironmentReport,
    match_environment,
    _default_runner,
)


INSTALLATION_STATE_VERSION = 1
RUNTIME_CONFIG_VERSION = 1
INSTALLATION_TIMEOUT_SECONDS = 3_600.0
INSTALLATION_RECOVERY_STALE_SECONDS = 300.0
MAX_INSTALL_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_INSTALL_DISK_BYTES = 4 * 1024 * 1024 * 1024
MAX_INSTALL_OUTPUT_BYTES = 512 * 1024
MAX_INSTALL_ENTRIES = 32
MAX_EXTRACTED_FILE_BYTES = 4 * 1024 * 1024 * 1024
MAX_PERSISTED_PLANS = 128
MAX_PERSISTED_RUNTIME_CONFIGS = 128
# Allow one bounded compaction window when reading a state file produced by an
# older version that did not compact runtime-config history.
MAX_STATE_RECORDS_ON_READ = max(MAX_PERSISTED_PLANS, MAX_PERSISTED_RUNTIME_CONFIGS) * 2
RUNTIME_MANIFEST_ENV = "MOLLY_RUNTIME_MANIFEST_PATH"
_STORE_THREAD_LOCK = threading.RLock()
_ACTIVE_INSTALLATION_STATES = frozenset(
    {"APPROVED", "INSTALLING", "VERIFYING", "ENABLING", "RECOVERING", "ROLLING_BACK"}
)
_RECOVERY_THREAD_LEASES: set[str] = set()


class InstallationError(RuntimeError):
    """Base class for expected installation failures."""


class InstallationConfigError(InstallationError, ValueError):
    """A manifest, plan, binding, or persisted installation is invalid."""


class InstallationIntegrityError(InstallationError):
    """A fixed source, hash, archive, or staged runtime failed validation."""


class InstallationConflictError(InstallationError):
    """A concurrent writer or already-used runtime prevented a safe change."""


class InstallationExecutionError(InstallationError):
    """A bounded local or SSH installation operation failed."""


def _bounded_text(value: Any, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise InstallationConfigError(f"{field} is invalid")
    if any(char in value for char in "\x00\r\n"):
        raise InstallationConfigError(f"{field} contains a control character")
    return value


def _bounded_int(value: Any, *, field: str, minimum: int = 0, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise InstallationConfigError(f"{field} is invalid")
    return value


def _timestamp(value: Any, *, field: str) -> str:
    try:
        return normalize_timestamp(value, field=field)
    except Exception as exc:
        raise InstallationConfigError(f"{field} is invalid") from exc


def _relative_path(value: Any, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > 255:
        raise InstallationConfigError(f"{field} is invalid")
    if not value and allow_empty:
        return ""
    if not value or "\\" in value or "\x00" in value or value.startswith(("/", "~")):
        raise InstallationConfigError(f"{field} is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise InstallationConfigError(f"{field} is invalid")
    return str(path)


def _source_url(value: Any) -> str:
    result = _bounded_text(value, field="manifest source_url", maximum=2_048)
    parsed = urlsplit(result)
    # ``file://`` is useful only for server-owned test fixtures.  A browser
    # payload never reaches this constructor.  Production entries are HTTPS.
    if parsed.scheme not in {"https", "file"} or parsed.username or parsed.password:
        raise InstallationConfigError("manifest source_url must be HTTPS or a server-owned file URL")
    if parsed.scheme == "https" and not parsed.hostname:
        raise InstallationConfigError("manifest source_url must have a host")
    if parsed.scheme == "file" and not parsed.path:
        raise InstallationConfigError("manifest file URL is invalid")
    return result


def _safe_id(value: Any, *, field: str) -> str:
    try:
        return validate_identifier(value, field=field)
    except Exception as exc:
        raise InstallationConfigError(f"{field} is invalid") from exc


def _public_hash_safe(value: Any) -> Any:
    """Hide artifact SHA fields from ordinary API projections."""

    if isinstance(value, Mapping):
        return {
            key: _public_hash_safe(item)
            for key, item in value.items()
            if key != "sha256"
        }
    if isinstance(value, (list, tuple)):
        return [_public_hash_safe(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class InstallManifestEntry:
    """One immutable, allow-listed installable component."""

    component_id: str
    name: str
    version: str
    source: str
    source_url: str
    estimated_download_bytes: int
    estimated_disk_bytes: int
    estimated_duration_seconds: int
    install_subdirectory: str
    requires_license: bool = False
    license_name: str = ""
    sha256: str | None = None
    install_kind: str = "archive"
    install_filename: str = ""
    max_download_bytes: int = MAX_INSTALL_DOWNLOAD_BYTES
    required_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _safe_id(self.component_id, field="component_id"))
        object.__setattr__(self, "name", _bounded_text(self.name, field="component name", maximum=160))
        object.__setattr__(self, "version", _bounded_text(self.version, field="component version", maximum=160))
        object.__setattr__(self, "source", _bounded_text(self.source, field="component source", maximum=512))
        object.__setattr__(self, "source_url", _source_url(self.source_url))
        object.__setattr__(
            self,
            "estimated_download_bytes",
            _bounded_int(
                self.estimated_download_bytes,
                field="estimated_download_bytes",
                maximum=MAX_INSTALL_DOWNLOAD_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "estimated_disk_bytes",
            _bounded_int(
                self.estimated_disk_bytes,
                field="estimated_disk_bytes",
                maximum=MAX_INSTALL_DISK_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "estimated_duration_seconds",
            _bounded_int(
                self.estimated_duration_seconds,
                field="estimated_duration_seconds",
                maximum=7 * 24 * 60 * 60,
            ),
        )
        object.__setattr__(
            self,
            "install_subdirectory",
            _relative_path(self.install_subdirectory, field="install_subdirectory"),
        )
        if not isinstance(self.requires_license, bool):
            raise InstallationConfigError("requires_license is invalid")
        object.__setattr__(
            self,
            "license_name",
            _bounded_text(self.license_name, field="license_name", maximum=160)
            if self.license_name
            else "",
        )
        if self.sha256 is not None:
            try:
                object.__setattr__(self, "sha256", validate_sha256(self.sha256, field="manifest sha256"))
            except Exception as exc:
                raise InstallationConfigError("manifest sha256 is invalid") from exc
            if self.estimated_download_bytes <= 0 or self.estimated_disk_bytes <= 0:
                raise InstallationConfigError(
                    "installable manifest entries require positive download and disk estimates"
                )
        if self.install_kind not in {"archive", "zip", "tar", "file"}:
            raise InstallationConfigError("install_kind is not allow-listed")
        filename = self.install_filename
        if filename:
            filename = _relative_path(filename, field="install_filename")
        elif self.install_kind == "file":
            filename = "payload.bin"
        object.__setattr__(self, "install_filename", filename)
        object.__setattr__(
            self,
            "max_download_bytes",
            _bounded_int(
                self.max_download_bytes,
                field="max_download_bytes",
                minimum=1,
                maximum=MAX_INSTALL_DOWNLOAD_BYTES,
            ),
        )
        if self.estimated_download_bytes > self.max_download_bytes:
            raise InstallationConfigError("manifest download estimate exceeds its fixed limit")
        if isinstance(self.required_paths, (str, bytes, bytearray)):
            raise InstallationConfigError("required_paths must be a sequence")
        normalized_paths: list[str] = []
        for item in self.required_paths:
            normalized = _relative_path(item, field="required_paths")
            if normalized not in normalized_paths:
                normalized_paths.append(normalized)
        object.__setattr__(self, "required_paths", tuple(normalized_paths))

    @property
    def installable(self) -> bool:
        return bool(self.sha256)

    def to_dict(self, *, include_internal: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "component_id": self.component_id,
            "name": self.name,
            "version": self.version,
            "source": self.source,
            "source_url": self.source_url,
            "estimated_download_bytes": self.estimated_download_bytes,
            "estimated_disk_bytes": self.estimated_disk_bytes,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "install_subdirectory": self.install_subdirectory,
            "requires_license": self.requires_license,
            "license_name": self.license_name,
            "install_kind": self.install_kind,
            "install_filename": self.install_filename,
            "max_download_bytes": self.max_download_bytes,
            "required_paths": list(self.required_paths),
        }
        if include_internal:
            value["sha256"] = self.sha256
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InstallManifestEntry":
        if not isinstance(value, Mapping):
            raise InstallationConfigError("manifest entry must be an object")
        allowed = {
            "component_id",
            "name",
            "version",
            "source",
            "source_url",
            "estimated_download_bytes",
            "estimated_disk_bytes",
            "estimated_duration_seconds",
            "install_subdirectory",
            "requires_license",
            "license_name",
            "sha256",
            "install_kind",
            "install_filename",
            "max_download_bytes",
            "required_paths",
        }
        if set(value) - allowed:
            raise InstallationConfigError("manifest entry contains unsupported fields")
        paths = value.get("required_paths", ())
        if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes, bytearray)):
            raise InstallationConfigError("manifest required_paths must be a list")
        return cls(
            component_id=value.get("component_id"),
            name=value.get("name"),
            version=value.get("version"),
            source=value.get("source"),
            source_url=value.get("source_url"),
            estimated_download_bytes=value.get("estimated_download_bytes"),
            estimated_disk_bytes=value.get("estimated_disk_bytes"),
            estimated_duration_seconds=value.get("estimated_duration_seconds"),
            install_subdirectory=value.get("install_subdirectory"),
            requires_license=value.get("requires_license", False),
            license_name=value.get("license_name", ""),
            sha256=value.get("sha256"),
            install_kind=value.get("install_kind", "archive"),
            install_filename=value.get("install_filename", ""),
            max_download_bytes=value.get("max_download_bytes", MAX_INSTALL_DOWNLOAD_BYTES),
            required_paths=tuple(paths),
        )

    @classmethod
    def from_compatibility_entry(cls, entry: CompatibilityEntry) -> "InstallManifestEntry":
        return cls(
            component_id=entry.component_id,
            name=entry.name,
            version=entry.version,
            source=entry.source,
            source_url=entry.source_url,
            estimated_download_bytes=entry.estimated_download_bytes,
            estimated_disk_bytes=entry.estimated_disk_bytes,
            estimated_duration_seconds=entry.estimated_duration_seconds,
            install_subdirectory=entry.install_subdirectory,
            requires_license=entry.requires_license,
            license_name="REINVENT4 license / credentials" if entry.requires_license else "",
            sha256=None,
        )


@dataclass(frozen=True, slots=True)
class InstallManifest:
    """A server-owned compatibility catalog and its resource limits."""

    catalog_version: str
    entries: tuple[InstallManifestEntry, ...]
    max_total_download_bytes: int = MAX_INSTALL_DOWNLOAD_BYTES
    max_total_disk_bytes: int = MAX_INSTALL_DISK_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(self, "catalog_version", _safe_id(self.catalog_version, field="catalog_version"))
        object.__setattr__(self, "entries", tuple(self.entries))
        if not self.entries or len(self.entries) > MAX_INSTALL_ENTRIES:
            raise InstallationConfigError("install manifest has an invalid entry count")
        seen: set[str] = set()
        for entry in self.entries:
            if not isinstance(entry, InstallManifestEntry):
                raise InstallationConfigError("install manifest accepts fixed entries only")
            if entry.component_id in seen:
                raise InstallationConfigError("install manifest contains a duplicate component")
            seen.add(entry.component_id)
        object.__setattr__(
            self,
            "max_total_download_bytes",
            _bounded_int(
                self.max_total_download_bytes,
                field="max_total_download_bytes",
                minimum=1,
                maximum=MAX_INSTALL_DOWNLOAD_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "max_total_disk_bytes",
            _bounded_int(
                self.max_total_disk_bytes,
                field="max_total_disk_bytes",
                minimum=1,
                maximum=MAX_INSTALL_DISK_BYTES,
            ),
        )

    @classmethod
    def from_compatibility_catalog(
        cls,
        catalog: Sequence[CompatibilityEntry] = COMPATIBILITY_CATALOG,
        *,
        catalog_version: str = COMPATIBILITY_CATALOG_VERSION,
    ) -> "InstallManifest":
        return cls(
            catalog_version=catalog_version,
            entries=tuple(InstallManifestEntry.from_compatibility_entry(item) for item in catalog),
        )

    @classmethod
    def from_json_file(cls, path: Path | str) -> "InstallManifest":
        """Load a server-owned fixed manifest; never accept it from HTTP input."""

        configured = Path(path)
        if configured.is_symlink() or not configured.is_file():
            raise InstallationConfigError("runtime install manifest is not a regular file")
        try:
            metadata = configured.stat()
            if metadata.st_mode & 0o022:
                raise InstallationConfigError("runtime install manifest must not be group/world writable")
            if metadata.st_size > 4 * 1024 * 1024:
                raise InstallationConfigError("runtime install manifest is too large")
            value = json.loads(configured.read_text(encoding="utf-8"))
        except InstallationError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InstallationConfigError("runtime install manifest could not be read") from exc
        if not isinstance(value, Mapping):
            raise InstallationConfigError("runtime install manifest must be an object")
        if set(value) - {
            "catalog_version",
            "max_total_download_bytes",
            "max_total_disk_bytes",
            "entries",
        }:
            raise InstallationConfigError("runtime install manifest contains unsupported fields")
        entries = value.get("entries")
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
            raise InstallationConfigError("runtime install manifest entries must be a list")
        return cls(
            catalog_version=value.get("catalog_version"),
            entries=tuple(InstallManifestEntry.from_dict(item) for item in entries),
            max_total_download_bytes=value.get("max_total_download_bytes", MAX_INSTALL_DOWNLOAD_BYTES),
            max_total_disk_bytes=value.get("max_total_disk_bytes", MAX_INSTALL_DISK_BYTES),
        )

    @property
    def entry_map(self) -> Mapping[str, InstallManifestEntry]:
        return {entry.component_id: entry for entry in self.entries}

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def get(self, component_id: str) -> InstallManifestEntry | None:
        return self.entry_map.get(component_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_version": self.catalog_version,
            "max_total_download_bytes": self.max_total_download_bytes,
            "max_total_disk_bytes": self.max_total_disk_bytes,
            "entries": [entry.to_dict(include_internal=True) for entry in self.entries],
        }


def default_install_manifest() -> InstallManifest:
    """Return the production manifest without fabricating artifact hashes."""

    return InstallManifest.from_compatibility_catalog()


def _configured_manifest_path(root: Path, manifest_path: Path | str | None) -> Path | None:
    if manifest_path is not None:
        return Path(manifest_path)
    configured = os.environ.get(RUNTIME_MANIFEST_ENV, "").strip()
    if configured:
        return Path(configured)
    candidate = root / "runtime_install_manifest.json"
    return candidate if candidate.exists() else None


def _runtime_target(profile: EnvironmentProfile, runtime_id: str) -> str:
    if profile.mode == "local":
        return f".molly/runtimes/{runtime_id}"
    return f"~/.local/share/molly/runtimes/{runtime_id}"


def _runtime_stage(profile: EnvironmentProfile, runtime_id: str, installation_id: str) -> str:
    if profile.mode == "local":
        return f".molly/.runtime-staging/{runtime_id}-{installation_id}"
    return f"~/.local/share/molly/runtimes/.staging/{runtime_id}-{installation_id}"


def _existing_target(profile: EnvironmentProfile, report: EnvironmentReport) -> str:
    disk = report.data.get("disk", {}) if isinstance(report.data, Mapping) else {}
    path = disk.get("path") if isinstance(disk, Mapping) else None
    if isinstance(path, str) and path:
        return path[:1_024]
    return f"{profile.target_label}（现有环境）"


@dataclass(frozen=True, slots=True)
class InstallPlan:
    """A digest-bound, non-executable installation proposal."""

    plan_id: str
    runtime_id: str
    environment_ref: str
    connection_digest: str
    report_digest: str
    catalog_version: str
    catalog_digest: str
    selected_device: str
    target_directory: str
    entries: tuple[InstallManifestEntry, ...]
    directory_required: bool
    blockers: tuple[str, ...]
    reasons: Mapping[str, str]
    created_at: str
    status: str
    plan_digest: str
    reused_components: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _safe_id(self.plan_id, field="plan_id"))
        object.__setattr__(self, "runtime_id", _safe_id(self.runtime_id, field="runtime_id"))
        object.__setattr__(self, "environment_ref", _safe_id(self.environment_ref, field="environment_ref"))
        for value, field in (
            (self.connection_digest, "connection_digest"),
            (self.report_digest, "report_digest"),
            (self.catalog_digest, "catalog_digest"),
            (self.plan_digest, "plan_digest"),
        ):
            try:
                validate_sha256(value, field=field)
            except Exception as exc:
                raise InstallationConfigError(f"{field} is invalid") from exc
        object.__setattr__(self, "catalog_version", _safe_id(self.catalog_version, field="catalog_version"))
        if self.selected_device not in {"CPU", "GPU"}:
            raise InstallationConfigError("selected_device is invalid")
        object.__setattr__(self, "target_directory", _bounded_text(self.target_directory, field="target_directory", maximum=512))
        if self.status not in {
            "NO_INSTALL_REQUIRED",
            "READY_TO_CONFIRM",
            "ALREADY_CONFIRMED",
            "READY_TO_INSTALL",
            "BLOCKED",
        }:
            raise InstallationConfigError("install plan status is invalid")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, field="plan created_at"))
        object.__setattr__(self, "blockers", tuple(_bounded_text(item, field="plan blocker", maximum=512) for item in self.blockers))
        object.__setattr__(self, "reasons", {str(key): str(value) for key, value in self.reasons.items()})
        object.__setattr__(
            self,
            "reused_components",
            tuple(thaw_json(item) for item in self.reused_components),
        )
        if len(self.entries) > MAX_INSTALL_ENTRIES:
            raise InstallationConfigError("install plan has too many entries")
        if self.plan_digest != sha256_bytes(
            canonical_json_bytes(self._payload(include_internal=True))
        ):
            raise InstallationIntegrityError("install plan digest does not match its content")

    @property
    def requires_confirmation(self) -> bool:
        return self.status in {"READY_TO_CONFIRM", "READY_TO_INSTALL", "BLOCKED"}

    @property
    def will_execute(self) -> bool:
        return self.status == "READY_TO_INSTALL"

    @property
    def estimated_download_bytes(self) -> int:
        return sum(item.estimated_download_bytes for item in self.entries)

    @property
    def estimated_disk_bytes(self) -> int:
        return sum(item.estimated_disk_bytes for item in self.entries)

    @property
    def estimated_duration_seconds(self) -> int:
        return sum(item.estimated_duration_seconds for item in self.entries)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(entry.component_id for entry in self.entries)

    def _payload(self, *, include_internal: bool) -> dict[str, Any]:
        items = []
        for entry in self.entries:
            item = entry.to_dict(include_internal=include_internal)
            item.update(
                {
                    "action": "install_in_isolated_directory",
                    "install_location": f"{self.target_directory}/{entry.install_subdirectory}",
                    "reason": self.reasons.get(entry.component_id, "缺少兼容组件"),
                }
            )
            items.append(item)
        if self.directory_required:
            items.append(
                {
                    "component_id": "runtime-directory",
                    "name": "可写运行目录",
                    "version": "server-owned",
                    "source": "服务端隔离目录策略",
                    "source_url": "server-owned",
                    "estimated_download_bytes": 0,
                    "estimated_disk_bytes": 0,
                    "estimated_duration_seconds": 0,
                    "install_subdirectory": "",
                    "action": "prepare_isolated_directory",
                    "install_location": self.target_directory,
                    "reason": self.reasons.get("runtime-directory", "隔离运行目录尚未确认可写"),
                    "requires_license": False,
                    "license_name": "",
                }
            )
        return {
            "version": INSTALLATION_STATE_VERSION,
            "plan_id": self.plan_id,
            "runtime_id": self.runtime_id,
            "environment_ref": self.environment_ref,
            "connection_digest": self.connection_digest,
            "report_digest": self.report_digest,
            "catalog_version": self.catalog_version,
            "catalog_digest": self.catalog_digest,
            "selected_device": self.selected_device,
            "target_directory": self.target_directory,
            "status": self.status,
            "blockers": list(self.blockers),
            "directory_required": self.directory_required,
            "created_at": self.created_at,
            "reused_components": thaw_json(list(self.reused_components)),
            "items": items,
        }

    def to_dict(self, *, public: bool = True) -> dict[str, Any]:
        value = self._payload(include_internal=not public)
        value["plan_digest"] = self.plan_digest
        value["reused_components"] = _public_hash_safe(value["reused_components"])
        value["estimated_download_bytes"] = self.estimated_download_bytes
        value["estimated_disk_bytes"] = self.estimated_disk_bytes
        value["estimated_duration_seconds"] = self.estimated_duration_seconds
        value["requires_confirmation"] = self.requires_confirmation
        value["will_execute"] = self.will_execute
        value["integrity_policy"] = "固定清单 SHA-256、大小上限、隔离目录和原子启用"
        value["risk"] = list(self.blockers) or [
            "确认只登记重新探测到的现有兼容环境，不下载或修改环境"
            if self.status == "READY_TO_CONFIRM"
            else "已确认配置会继续绑定当前连接和固定清单"
            if self.status == "ALREADY_CONFIRMED"
            else "只写入新的隔离目录，不覆盖已有环境"
        ]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InstallPlan":
        if not isinstance(value, Mapping):
            raise InstallationConfigError("install plan must be an object")
        if value.get("version") != INSTALLATION_STATE_VERSION:
            raise InstallationConfigError("install plan version is unsupported")
        items = value.get("items", ())
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise InstallationConfigError("install plan items must be a list")
        entries: list[InstallManifestEntry] = []
        reasons: dict[str, str] = {}
        directory_required = bool(value.get("directory_required", False))
        for item in items:
            if not isinstance(item, Mapping):
                raise InstallationConfigError("install plan item is invalid")
            component_id = item.get("component_id")
            if component_id == "runtime-directory":
                directory_required = True
                reasons["runtime-directory"] = str(item.get("reason", ""))
                continue
            entries.append(
                InstallManifestEntry.from_dict(
                    {
                        key: item[key]
                        for key in {
                            "component_id",
                            "name",
                            "version",
                            "source",
                            "source_url",
                            "estimated_download_bytes",
                            "estimated_disk_bytes",
                            "estimated_duration_seconds",
                            "install_subdirectory",
                            "requires_license",
                            "license_name",
                            "sha256",
                            "install_kind",
                            "install_filename",
                            "max_download_bytes",
                            "required_paths",
                        }
                        if key in item
                    }
                )
            )
            reasons[str(component_id)] = str(item.get("reason", ""))
        raw_blockers = value.get("blockers", ())
        if not isinstance(raw_blockers, Sequence) or isinstance(raw_blockers, (str, bytes, bytearray)):
            raise InstallationConfigError("install plan blockers are invalid")
        raw_reused = value.get("reused_components", ())
        if not isinstance(raw_reused, Sequence) or isinstance(raw_reused, (str, bytes, bytearray)):
            raise InstallationConfigError("install plan reused_components are invalid")
        return cls(
            plan_id=value.get("plan_id"),
            runtime_id=value.get("runtime_id"),
            environment_ref=value.get("environment_ref"),
            connection_digest=value.get("connection_digest"),
            report_digest=value.get("report_digest"),
            catalog_version=value.get("catalog_version"),
            catalog_digest=value.get("catalog_digest"),
            selected_device=value.get("selected_device"),
            target_directory=value.get("target_directory"),
            entries=tuple(entries),
            directory_required=directory_required,
            blockers=tuple(str(item) for item in raw_blockers),
            reasons=reasons,
            created_at=value.get("created_at"),
            status=value.get("status"),
            plan_digest=value.get("plan_digest"),
            reused_components=tuple(raw_reused),
        )

    @classmethod
    def build(
        cls,
        profile: EnvironmentProfile,
        report: EnvironmentReport,
        match: Mapping[str, Any],
        manifest: InstallManifest,
        *,
        selected_component_ids: Sequence[str] | None = None,
        existing_runtime: Any | None = None,
        force_reinstall: bool = False,
    ) -> "InstallPlan":
        if not isinstance(match, Mapping):
            raise InstallationConfigError("environment match is invalid")
        missing = match.get("missing", ())
        if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes, bytearray)):
            raise InstallationConfigError("environment missing-component list is invalid")
        missing_by_id: dict[str, Mapping[str, Any]] = {}
        for item in missing:
            if not isinstance(item, Mapping):
                raise InstallationConfigError("environment missing-component entry is invalid")
            component_id = _safe_id(item.get("component_id"), field="missing component_id")
            missing_by_id[component_id] = item

        if force_reinstall:
            missing_by_id.setdefault(
                "unimol-weights",
                {"component_id": "unimol-weights", "reason": "固定清单已变化，需要重新验证并安装模型权重"},
            )

        if selected_component_ids is None:
            selected = tuple(component_id for component_id in missing_by_id if component_id != "runtime-directory")
        else:
            if not isinstance(selected_component_ids, Sequence) or isinstance(
                selected_component_ids, (str, bytes, bytearray)
            ):
                raise InstallationConfigError("selected_component_ids must be a list")
            selected = tuple(_safe_id(item, field="selected component_id") for item in selected_component_ids)
            if len(selected) != len(set(selected)):
                raise InstallationConfigError("selected_component_ids must be unique")
        allowed = set(missing_by_id) - {"runtime-directory"}
        unknown = set(selected) - allowed
        if unknown:
            raise InstallationConfigError("LLM or client selected a component outside the current plan")
        omitted = allowed - set(selected)
        entries: list[InstallManifestEntry] = []
        blockers: list[str] = [str(item) for item in match.get("blockers", ()) if isinstance(item, str)]
        reasons = {
            component_id: str(item.get("reason", "缺少兼容组件"))
            for component_id, item in missing_by_id.items()
        }
        for component_id in sorted(selected):
            entry = manifest.get(component_id)
            if entry is None:
                blockers.append(f"固定清单没有组件 {component_id}")
                continue
            entries.append(entry)
            if not entry.installable:
                blockers.append(f"{entry.name} 尚未登记可验证的 SHA-256，安装已暂停")
            if entry.requires_license and not bool(
                ((report.data.get("reinvent4") if isinstance(report.data, Mapping) else {}) or {}).get(
                    "license_present"
                )
            ):
                blockers.append(f"{entry.name} 需要许可证或凭据，安装已暂停")
        for component_id in sorted(omitted):
            blockers.append(f"用户未确认缺失组件 {component_id}")
        if sum(entry.estimated_download_bytes for entry in entries) > manifest.max_total_download_bytes:
            blockers.append("安装计划超过固定下载大小上限")
        if sum(entry.estimated_disk_bytes for entry in entries) > manifest.max_total_disk_bytes:
            blockers.append("安装计划超过固定磁盘占用上限")
        disk_data = report.data.get("disk", {}) if isinstance(report.data, Mapping) else {}
        available_bytes = disk_data.get("available_bytes") if isinstance(disk_data, Mapping) else None
        if isinstance(available_bytes, int) and sum(entry.estimated_disk_bytes for entry in entries) > available_bytes:
            blockers.append("当前可用磁盘空间不足以容纳固定安装计划")
        if "runtime-directory" in missing_by_id:
            disk_writable = disk_data.get("parent_writable") if isinstance(disk_data, Mapping) else False
            if not bool(disk_writable):
                blockers.append("隔离运行目录及其父目录不可写")
        if not bool(match.get("selected_device") in {"CPU", "GPU"}):
            blockers.append("没有可用的 CPU/GPU 执行设备")

        ready_existing = match.get("status") == "READY" and not missing_by_id
        already_confirmed = bool(
            existing_runtime is not None and getattr(existing_runtime, "state", "") == "CONFIRMED"
        )
        runtime_id = (
            getattr(existing_runtime, "runtime_id", "")
            if already_confirmed
            else new_server_id("runtime")
        )
        plan_id = new_server_id("install-plan")
        reused_components = tuple(
            item
            for item in match.get("reusable", ())
            if isinstance(item, Mapping)
            and item.get("component_id") not in missing_by_id
        )
        if already_confirmed and getattr(existing_runtime, "components", ()):
            reused_components = tuple(getattr(existing_runtime, "components"))
        target_directory = (
            getattr(existing_runtime, "target_directory", "")
            if already_confirmed
            else _existing_target(profile, report)
            if ready_existing
            else _runtime_target(profile, runtime_id)
        )
        status = (
            "ALREADY_CONFIRMED"
            if already_confirmed
            else "READY_TO_CONFIRM"
            if ready_existing
            else "BLOCKED"
            if blockers
            else "READY_TO_INSTALL"
        )
        created_at = utc_timestamp()
        payload = {
            "version": INSTALLATION_STATE_VERSION,
            "plan_id": plan_id,
            "runtime_id": runtime_id,
            "environment_ref": profile.environment_ref,
            "connection_digest": profile.connection_digest,
            "report_digest": report.report_digest,
            "catalog_version": manifest.catalog_version,
            "catalog_digest": manifest.digest,
            "selected_device": str(match.get("selected_device") or "CPU"),
            "target_directory": target_directory,
            "status": status,
            "blockers": blockers,
            "directory_required": "runtime-directory" in missing_by_id,
            "created_at": created_at,
            "reused_components": thaw_json(list(reused_components)),
            "items": [
                {
                    **entry.to_dict(include_internal=True),
                    "action": "install_in_isolated_directory",
                    "install_location": f"{target_directory}/{entry.install_subdirectory}",
                    "reason": reasons.get(entry.component_id, "缺少兼容组件"),
                }
                for entry in entries
            ],
        }
        digest = sha256_bytes(canonical_json_bytes(payload))
        return cls(
            plan_id=plan_id,
            runtime_id=runtime_id,
            environment_ref=profile.environment_ref,
            connection_digest=profile.connection_digest,
            report_digest=report.report_digest,
            catalog_version=manifest.catalog_version,
            catalog_digest=manifest.digest,
            selected_device=str(match.get("selected_device") or "CPU"),
            target_directory=target_directory,
            entries=tuple(entries),
            directory_required="runtime-directory" in missing_by_id,
            blockers=tuple(blockers),
            reasons=reasons,
            created_at=created_at,
            status=status,
            plan_digest=digest,
            reused_components=reused_components,
        )


@dataclass(frozen=True, slots=True)
class InstallationRecord:
    """Persisted installation transaction and its resumable progress."""

    installation_id: str
    plan_id: str
    runtime_id: str
    environment_ref: str
    connection_digest: str
    report_digest: str
    plan_digest: str
    catalog_version: str
    catalog_digest: str
    selected_device: str
    state: str
    revision: int
    created_at: str
    updated_at: str
    stage_directory: str
    target_directory: str
    completed_component_ids: tuple[str, ...] = ()
    verification: Mapping[str, Any] = None  # type: ignore[assignment]
    error: str = ""
    rollback_completed: bool = False
    side_effects_started: bool = False
    consent_at: str = ""
    worker_pid: int = 0

    def __post_init__(self) -> None:
        for value, field in (
            (self.installation_id, "installation_id"),
            (self.plan_id, "plan_id"),
            (self.runtime_id, "runtime_id"),
            (self.environment_ref, "environment_ref"),
            (self.catalog_version, "catalog_version"),
        ):
            _safe_id(value, field=field)
        for value, field in (
            (self.connection_digest, "connection_digest"),
            (self.report_digest, "report_digest"),
            (self.plan_digest, "plan_digest"),
            (self.catalog_digest, "catalog_digest"),
        ):
            try:
                validate_sha256(value, field=field)
            except Exception as exc:
                raise InstallationConfigError(f"{field} is invalid") from exc
        if self.selected_device not in {"CPU", "GPU"}:
            raise InstallationConfigError("installation selected_device is invalid")
        if self.state not in {
            "APPROVED",
            "INSTALLING",
            "VERIFYING",
            "ENABLING",
            "RECOVERING",
            "ROLLING_BACK",
            "CONFIRMED",
            "FAILED",
            "ROLLED_BACK",
        }:
            raise InstallationConfigError("installation state is invalid")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 1:
            raise InstallationConfigError("installation revision is invalid")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, field="installation created_at"))
        object.__setattr__(self, "updated_at", _timestamp(self.updated_at, field="installation updated_at"))
        object.__setattr__(
            self,
            "stage_directory",
            _bounded_text(self.stage_directory, field="stage_directory", maximum=1_024)
            if self.stage_directory
            else "",
        )
        object.__setattr__(self, "target_directory", _bounded_text(self.target_directory, field="target_directory", maximum=1_024))
        ids = tuple(_safe_id(item, field="completed component_id") for item in self.completed_component_ids)
        if len(ids) != len(set(ids)):
            raise InstallationConfigError("completed component IDs must be unique")
        object.__setattr__(self, "completed_component_ids", ids)
        object.__setattr__(self, "verification", thaw_json(self.verification or {}))
        if self.error:
            object.__setattr__(self, "error", _bounded_text(self.error, field="installation error", maximum=2_000))
        if not isinstance(self.rollback_completed, bool) or not isinstance(self.side_effects_started, bool):
            raise InstallationConfigError("installation flags are invalid")
        if self.consent_at:
            object.__setattr__(self, "consent_at", _timestamp(self.consent_at, field="consent_at"))
        if isinstance(self.worker_pid, bool) or not isinstance(self.worker_pid, int) or self.worker_pid < 0:
            raise InstallationConfigError("worker_pid is invalid")

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        value = {
            "version": INSTALLATION_STATE_VERSION,
            "installation_id": self.installation_id,
            "plan_id": self.plan_id,
            "runtime_id": self.runtime_id,
            "environment_ref": self.environment_ref,
            "connection_digest": self.connection_digest,
            "report_digest": self.report_digest,
            "plan_digest": self.plan_digest,
            "catalog_version": self.catalog_version,
            "catalog_digest": self.catalog_digest,
            "selected_device": self.selected_device,
            "state": self.state,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "stage_directory": self.stage_directory,
            "target_directory": self.target_directory,
            "completed_component_ids": list(self.completed_component_ids),
            "verification": thaw_json(self.verification),
            "error": self.error,
            "rollback_completed": self.rollback_completed,
            "side_effects_started": self.side_effects_started,
            "consent_at": self.consent_at,
        }
        if not public:
            value["worker_pid"] = self.worker_pid
        else:
            value.pop("stage_directory")
            value.pop("target_directory")
            value.pop("worker_pid", None)
            value["verification"] = _public_hash_safe(value["verification"])
            value["binding"] = {
                "connection_digest": self.connection_digest,
                "report_digest": self.report_digest,
                "plan_digest": self.plan_digest,
                "catalog_version": self.catalog_version,
            }
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InstallationRecord":
        if not isinstance(value, Mapping):
            raise InstallationConfigError("installation record must be an object")
        if value.get("version") != INSTALLATION_STATE_VERSION:
            raise InstallationConfigError("installation record version is unsupported")
        completed = value.get("completed_component_ids", ())
        if not isinstance(completed, Sequence) or isinstance(completed, (str, bytes, bytearray)):
            raise InstallationConfigError("completed_component_ids is invalid")
        verification = value.get("verification", {})
        if not isinstance(verification, Mapping):
            raise InstallationConfigError("installation verification is invalid")
        return cls(
            installation_id=value.get("installation_id"),
            plan_id=value.get("plan_id"),
            runtime_id=value.get("runtime_id"),
            environment_ref=value.get("environment_ref"),
            connection_digest=value.get("connection_digest"),
            report_digest=value.get("report_digest"),
            plan_digest=value.get("plan_digest"),
            catalog_version=value.get("catalog_version"),
            catalog_digest=value.get("catalog_digest"),
            selected_device=value.get("selected_device"),
            state=value.get("state"),
            revision=value.get("revision"),
            created_at=value.get("created_at"),
            updated_at=value.get("updated_at"),
            stage_directory=value.get("stage_directory"),
            target_directory=value.get("target_directory"),
            completed_component_ids=tuple(completed),
            verification=verification,
            error=value.get("error", ""),
            rollback_completed=value.get("rollback_completed", False),
            side_effects_started=value.get("side_effects_started", False),
            consent_at=value.get("consent_at", ""),
            worker_pid=value.get("worker_pid", 0),
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """A confirmed runtime pointer, written only after atomic enable."""

    runtime_id: str
    installation_id: str
    environment_ref: str
    connection_digest: str
    report_digest: str
    plan_digest: str
    catalog_version: str
    catalog_digest: str
    selected_device: str
    target_directory: str
    components: tuple[Mapping[str, Any], ...]
    created_at: str
    verified_at: str
    state: str = "CONFIRMED"
    config_digest: str = ""

    def __post_init__(self) -> None:
        for value, field in (
            (self.runtime_id, "runtime_id"),
            (self.installation_id, "installation_id"),
            (self.environment_ref, "environment_ref"),
            (self.catalog_version, "catalog_version"),
        ):
            _safe_id(value, field=field)
        for value, field in (
            (self.connection_digest, "connection_digest"),
            (self.report_digest, "report_digest"),
            (self.plan_digest, "plan_digest"),
            (self.catalog_digest, "catalog_digest"),
        ):
            try:
                validate_sha256(value, field=field)
            except Exception as exc:
                raise InstallationConfigError(f"{field} is invalid") from exc
        if self.selected_device not in {"CPU", "GPU"}:
            raise InstallationConfigError("runtime selected_device is invalid")
        if self.state not in {"CONFIRMED", "INVALIDATED"}:
            raise InstallationConfigError("runtime config state is invalid")
        object.__setattr__(self, "created_at", _timestamp(self.created_at, field="runtime created_at"))
        object.__setattr__(self, "verified_at", _timestamp(self.verified_at, field="runtime verified_at"))
        object.__setattr__(self, "target_directory", _bounded_text(self.target_directory, field="runtime target_directory", maximum=1_024))
        object.__setattr__(self, "components", tuple(thaw_json(item) for item in self.components))
        if self.config_digest:
            try:
                validate_sha256(self.config_digest, field="runtime config_digest")
            except Exception as exc:
                raise InstallationConfigError("runtime config_digest is invalid") from exc
            if self.config_digest != sha256_bytes(canonical_json_bytes(self._payload())):
                raise InstallationIntegrityError("runtime config digest does not match its content")

    def _payload(self) -> dict[str, Any]:
        return {
            "version": RUNTIME_CONFIG_VERSION,
            "runtime_id": self.runtime_id,
            "installation_id": self.installation_id,
            "environment_ref": self.environment_ref,
            "connection_digest": self.connection_digest,
            "report_digest": self.report_digest,
            "plan_digest": self.plan_digest,
            "catalog_version": self.catalog_version,
            "catalog_digest": self.catalog_digest,
            "selected_device": self.selected_device,
            "target_directory": self.target_directory,
            "components": thaw_json(list(self.components)),
            "created_at": self.created_at,
            "verified_at": self.verified_at,
            "state": self.state,
        }

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        value = self._payload()
        value["config_digest"] = self.config_digest
        if public:
            value["components"] = _public_hash_safe(value["components"])
            value.pop("target_directory", None)
            value.pop("config_digest", None)
            value["status_label"] = "已确认" if self.state == "CONFIRMED" else "已失效"
        return value

    @classmethod
    def confirmed(
        cls,
        *,
        record: InstallationRecord,
        components: Sequence[Mapping[str, Any]],
        verified_at: str,
        report_digest: str | None = None,
    ) -> "RuntimeConfig":
        candidate = cls(
            runtime_id=record.runtime_id,
            installation_id=record.installation_id,
            environment_ref=record.environment_ref,
            connection_digest=record.connection_digest,
            report_digest=report_digest or record.report_digest,
            plan_digest=record.plan_digest,
            catalog_version=record.catalog_version,
            catalog_digest=record.catalog_digest,
            selected_device=record.selected_device,
            target_directory=record.target_directory,
            components=tuple(components),
            created_at=record.created_at,
            verified_at=verified_at,
        )
        return replace(candidate, config_digest=sha256_bytes(canonical_json_bytes(candidate._payload())))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        if not isinstance(value, Mapping) or value.get("version") != RUNTIME_CONFIG_VERSION:
            raise InstallationConfigError("runtime config version is unsupported")
        components = value.get("components", ())
        if not isinstance(components, Sequence) or isinstance(components, (str, bytes, bytearray)):
            raise InstallationConfigError("runtime components are invalid")
        return cls(
            runtime_id=value.get("runtime_id"),
            installation_id=value.get("installation_id"),
            environment_ref=value.get("environment_ref"),
            connection_digest=value.get("connection_digest"),
            report_digest=value.get("report_digest"),
            plan_digest=value.get("plan_digest"),
            catalog_version=value.get("catalog_version"),
            catalog_digest=value.get("catalog_digest"),
            selected_device=value.get("selected_device"),
            target_directory=value.get("target_directory"),
            components=tuple(components),
            created_at=value.get("created_at"),
            verified_at=value.get("verified_at"),
            state=value.get("state", "CONFIRMED"),
            config_digest=value.get("config_digest", ""),
        )


class RuntimeInstallationStore:
    """Atomic, owner-only persistence with transaction-level CAS."""

    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.is_symlink():
            raise InstallationConfigError("installation settings root cannot be a symlink")
        self.root = configured.absolute()
        self.state_path = self.root / "runtime_installations.json"
        self.lock_path = self.root / ".runtime-installations.lock"

    @contextmanager
    def _write_lock(self):
        if self.root.is_symlink():
            raise InstallationConfigError("installation settings root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        with _STORE_THREAD_LOCK:
            descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.fchmod(descriptor, 0o600)
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)

    def _check_file(self) -> None:
        if self.state_path.is_symlink():
            raise InstallationConfigError("installation state file cannot be a symlink")
        if self.state_path.exists() and not self.state_path.is_file():
            raise InstallationConfigError("installation state file is not regular")

    @staticmethod
    def _compact_state(
        value: dict[str, Any],
        *,
        maximum_plans: int = MAX_PERSISTED_PLANS,
        maximum_runtime_configs: int = MAX_PERSISTED_RUNTIME_CONFIGS,
    ) -> dict[str, Any]:
        """Drop only unreferenced/terminal records before the bounded store grows.

        Plans are retained while an installation may still need them for
        recovery or audit.  A caller that needs to insert a new plan passes
        ``maximum_plans - 1`` so the following insert remains within the
        persisted capacity.  This also makes a state file produced by an
        older version self-healing on its next write instead of becoming
        permanently unreadable at plan 129.
        """

        plans = value["plans"]
        installations = value["installations"]
        runtime_configs = value["runtime_configs"]
        maximum_plans = max(0, int(maximum_plans))
        maximum_runtime_configs = max(0, int(maximum_runtime_configs))

        def created_key(raw: Mapping[str, Any]) -> str:
            return str(raw.get("created_at", ""))

        while len(plans) > maximum_plans:
            referenced = {
                str(raw.get("plan_id"))
                for raw in installations.values()
                if isinstance(raw, Mapping) and raw.get("plan_id")
            }
            unreferenced = [
                (created_key(raw), plan_id)
                for plan_id, raw in plans.items()
                if plan_id not in referenced and isinstance(raw, Mapping)
            ]
            if unreferenced:
                _, plan_id = min(unreferenced)
                del plans[plan_id]
                continue

            confirmed_runtime_installation_ids = {
                str(raw.get("installation_id"))
                for raw in runtime_configs.values()
                if isinstance(raw, Mapping)
                and raw.get("state") == "CONFIRMED"
                and raw.get("installation_id")
            }
            terminal = [
                (created_key(raw), installation_id, str(raw.get("plan_id")))
                for installation_id, raw in installations.items()
                if isinstance(raw, Mapping)
                and (
                    raw.get("state") in {"FAILED", "ROLLED_BACK"}
                    or (
                        raw.get("state") == "CONFIRMED"
                        and installation_id not in confirmed_runtime_installation_ids
                    )
                )
            ]
            if not terminal:
                break
            _, installation_id, plan_id = min(terminal)
            del installations[installation_id]
            plans.pop(plan_id, None)

        while len(runtime_configs) > maximum_runtime_configs:
            invalidated = [
                (str(raw.get("verified_at", raw.get("created_at", ""))), runtime_id)
                for runtime_id, raw in runtime_configs.items()
                if isinstance(raw, Mapping) and raw.get("state") == "INVALIDATED"
            ]
            if not invalidated:
                # Confirmed configurations are live pointers and cannot be
                # evicted silently.  The write path will reject a new config
                # when no invalidated history is available for compaction.
                break
            _, runtime_id = min(invalidated)
            del runtime_configs[runtime_id]
        return value

    def _read_state(self) -> dict[str, Any]:
        self._check_file()
        if not self.state_path.exists():
            return {"version": INSTALLATION_STATE_VERSION, "plans": {}, "installations": {}, "runtime_configs": {}}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InstallationConfigError("installation state could not be read") from exc
        if not isinstance(value, dict) or value.get("version") != INSTALLATION_STATE_VERSION:
            raise InstallationConfigError("installation state version is unsupported")
        for key in ("plans", "installations", "runtime_configs"):
            if not isinstance(value.get(key), dict):
                raise InstallationConfigError("installation state has an invalid shape")
        if (
            len(value["plans"]) > MAX_STATE_RECORDS_ON_READ
            or len(value["installations"]) > MAX_STATE_RECORDS_ON_READ
            or len(value["runtime_configs"]) > MAX_STATE_RECORDS_ON_READ
        ):
            raise InstallationConfigError("installation state contains too many records")
        for key, raw in value["plans"].items():
            if not isinstance(key, str) or not isinstance(raw, Mapping) or raw.get("plan_id") != key:
                raise InstallationConfigError("installation plan identity is inconsistent")
            InstallPlan.from_dict(raw)
        for key, raw in value["installations"].items():
            if not isinstance(key, str) or not isinstance(raw, Mapping) or raw.get("installation_id") != key:
                raise InstallationConfigError("installation record identity is inconsistent")
            InstallationRecord.from_dict(raw)
        for key, raw in value["runtime_configs"].items():
            if not isinstance(key, str) or not isinstance(raw, Mapping) or raw.get("runtime_id") != key:
                raise InstallationConfigError("runtime config identity is inconsistent")
            RuntimeConfig.from_dict(raw)
        return self._compact_state(value)

    def _write_state(self, value: Mapping[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._check_file()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".runtime-installations.", suffix=".tmp", dir=str(self.root)
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.state_path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()

    def save_plan(self, plan: InstallPlan) -> None:
        if not isinstance(plan, InstallPlan):
            raise TypeError("save_plan accepts InstallPlan")
        with self._write_lock():
            value = self._read_state()
            existing = value["plans"].get(plan.plan_id)
            if existing is not None and existing != plan.to_dict(public=False):
                raise InstallationConflictError("install plan ID is already bound to different content")
            if existing is not None:
                return
            self._compact_state(
                value,
                maximum_plans=MAX_PERSISTED_PLANS - 1,
                maximum_runtime_configs=MAX_PERSISTED_RUNTIME_CONFIGS - 1,
            )
            if len(value["plans"]) >= MAX_PERSISTED_PLANS:
                raise InstallationConflictError("installation plan store is at capacity")
            value["plans"][plan.plan_id] = plan.to_dict(public=False)
            self._write_state(value)

    def get_plan(self, plan_id: str) -> InstallPlan:
        _safe_id(plan_id, field="plan_id")
        value = self._read_state()["plans"].get(plan_id)
        if value is None:
            raise InstallationConfigError("install plan was not found")
        return InstallPlan.from_dict(value)

    def get_installation(self, installation_id: str) -> InstallationRecord:
        _safe_id(installation_id, field="installation_id")
        value = self._read_state()["installations"].get(installation_id)
        if value is None:
            raise InstallationConfigError("installation was not found")
        return InstallationRecord.from_dict(value)

    def get_installation_for_plan(self, plan_id: str) -> InstallationRecord | None:
        _safe_id(plan_id, field="plan_id")
        for raw in self._read_state()["installations"].values():
            if isinstance(raw, Mapping) and raw.get("plan_id") == plan_id:
                return InstallationRecord.from_dict(raw)
        return None

    def get_installation_for_environment(
        self,
        environment_ref: str,
    ) -> InstallationRecord | None:
        _safe_id(environment_ref, field="environment_ref")
        candidates = [
            InstallationRecord.from_dict(raw)
            for raw in self._read_state()["installations"].values()
            if isinstance(raw, Mapping)
            and raw.get("environment_ref") == environment_ref
            and raw.get("state") in _ACTIVE_INSTALLATION_STATES
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.updated_at)

    def claim_approval(self, plan: InstallPlan) -> tuple[InstallationRecord, bool]:
        """Atomically create the one approval record, or return its owner."""

        with self._write_lock():
            value = self._read_state()
            for raw in value["installations"].values():
                if isinstance(raw, Mapping) and raw.get("plan_id") == plan.plan_id:
                    return InstallationRecord.from_dict(raw), False
            active_states = _ACTIVE_INSTALLATION_STATES
            confirmed_runtime_keys = {
                (raw.get("runtime_id"), raw.get("installation_id"))
                for raw in value["runtime_configs"].values()
                if isinstance(raw, Mapping) and raw.get("state") == "CONFIRMED"
            }
            for raw in value["installations"].values():
                if not isinstance(raw, Mapping):
                    continue
                if raw.get("environment_ref") != plan.environment_ref:
                    continue
                if raw.get("state") in active_states or (
                    raw.get("state") == "CONFIRMED"
                    and (raw.get("runtime_id"), raw.get("installation_id"))
                    in confirmed_runtime_keys
                ):
                    raise InstallationConflictError(
                        "当前连接已有另一个安装或确认事务正在进行"
                    )
            for raw in value["runtime_configs"].values():
                if not isinstance(raw, Mapping):
                    continue
                if (
                    raw.get("environment_ref") == plan.environment_ref
                    and raw.get("state") == "CONFIRMED"
                    and raw.get("runtime_id") != plan.runtime_id
                ):
                    raise InstallationConflictError("当前连接已有已确认运行配置")
            now = utc_timestamp()
            record = InstallationRecord(
                installation_id=new_server_id("installation"),
                plan_id=plan.plan_id,
                runtime_id=plan.runtime_id,
                environment_ref=plan.environment_ref,
                connection_digest=plan.connection_digest,
                report_digest=plan.report_digest,
                plan_digest=plan.plan_digest,
                catalog_version=plan.catalog_version,
                catalog_digest=plan.catalog_digest,
                selected_device=plan.selected_device,
                state="APPROVED",
                revision=1,
                created_at=now,
                updated_at=now,
                stage_directory="",
                target_directory=plan.target_directory,
                consent_at=now,
            )
            value["installations"][record.installation_id] = record.to_dict()
            self._write_state(value)
            return record, True

    def claim_execution(self, installation_id: str) -> tuple[InstallationRecord, bool]:
        with self._write_lock():
            value = self._read_state()
            raw = value["installations"].get(installation_id)
            if raw is None:
                raise InstallationConfigError("installation was not found")
            current = InstallationRecord.from_dict(raw)
            if current.state != "APPROVED":
                return current, False
            updated = replace(
                current,
                state="INSTALLING",
                revision=current.revision + 1,
                updated_at=utc_timestamp(),
                worker_pid=os.getpid(),
                side_effects_started=False,
            )
            value["installations"][installation_id] = updated.to_dict()
            self._write_state(value)
            return updated, True

    def claim_recovery(
        self,
        installation_id: str,
        *,
        force: bool = False,
    ) -> tuple[InstallationRecord, bool]:
        """Atomically acquire the lease used by ENABLING recovery.

        The durable RECOVERING state protects across processes.  The small
        in-process set also prevents two ``force=True`` callers in one server
        process from stealing one another's lease while a recovery is running.
        """

        _safe_id(installation_id, field="installation_id")
        with self._write_lock():
            value = self._read_state()
            raw = value["installations"].get(installation_id)
            if raw is None:
                raise InstallationConfigError("installation was not found")
            current = InstallationRecord.from_dict(raw)
            if current.state not in {"ENABLING", "RECOVERING"}:
                return current, False
            if installation_id in _RECOVERY_THREAD_LEASES:
                return current, False
            if current.worker_pid and _process_alive(current.worker_pid):
                # ``force`` is for reclaiming a durable lease after a crash,
                # not for stealing a lease held by another live process.
                # A same-process force call remains useful for deterministic
                # recovery tests and for an interrupted worker whose durable
                # record predates the in-process lease set.
                if not force or current.worker_pid != os.getpid():
                    return current, False
            updated = replace(
                current,
                state="RECOVERING",
                revision=current.revision + 1,
                updated_at=utc_timestamp(),
                error="",
                worker_pid=os.getpid(),
            )
            value["installations"][installation_id] = updated.to_dict()
            self._write_state(value)
            _RECOVERY_THREAD_LEASES.add(installation_id)
            return updated, True

    @staticmethod
    def release_recovery(installation_id: str) -> None:
        with _STORE_THREAD_LOCK:
            _RECOVERY_THREAD_LEASES.discard(installation_id)

    def update_installation(
        self,
        record: InstallationRecord,
        *,
        expected_revision: int,
    ) -> InstallationRecord:
        with self._write_lock():
            value = self._read_state()
            raw = value["installations"].get(record.installation_id)
            if raw is None:
                raise InstallationConflictError("installation disappeared during update")
            current = InstallationRecord.from_dict(raw)
            if current.revision != expected_revision:
                raise InstallationConflictError("installation changed concurrently")
            if record.revision != expected_revision + 1:
                raise InstallationConflictError("installation revision is not monotonic")
            value["installations"][record.installation_id] = record.to_dict()
            self._write_state(value)
            return record

    def recover_stale(
        self,
        installation_id: str,
        *,
        stale_after_seconds: float = INSTALLATION_RECOVERY_STALE_SECONDS,
        force: bool = False,
    ) -> InstallationRecord:
        if stale_after_seconds < 0:
            raise InstallationConfigError("stale_after_seconds is invalid")
        with self._write_lock():
            value = self._read_state()
            raw = value["installations"].get(installation_id)
            if raw is None:
                raise InstallationConfigError("installation was not found")
            current = InstallationRecord.from_dict(raw)
            if current.state not in {"INSTALLING", "VERIFYING", "ENABLING"}:
                return current
            try:
                age = max(0.0, time.time() - _parse_timestamp(current.updated_at))
            except Exception:
                age = stale_after_seconds + 1
            if not force and (age < stale_after_seconds or _process_alive(current.worker_pid)):
                raise InstallationConflictError("installation worker is still within its recovery lease")
            updated = replace(
                current,
                state="APPROVED",
                revision=current.revision + 1,
                updated_at=utc_timestamp(),
                error="",
                worker_pid=0,
            )
            value["installations"][installation_id] = updated.to_dict()
            self._write_state(value)
            return updated

    def save_runtime_config(self, config: RuntimeConfig) -> None:
        with self._write_lock():
            value = self._read_state()
            existing = value["runtime_configs"].get(config.runtime_id)
            if existing is not None:
                current = RuntimeConfig.from_dict(existing)
                if current.state == "INVALIDATED":
                    value["runtime_configs"][config.runtime_id] = config.to_dict()
                    self._write_state(value)
                    return
                if current.config_digest != config.config_digest:
                    raise InstallationConflictError("runtime ID is already bound to different content")
                return
            for runtime_id, raw in tuple(value["runtime_configs"].items()):
                if not isinstance(raw, Mapping) or raw.get("environment_ref") != config.environment_ref:
                    continue
                current = RuntimeConfig.from_dict(raw)
                if current.state != "CONFIRMED":
                    continue
                raise InstallationConflictError(
                    "当前连接已有另一个已确认运行配置"
                )
            self._compact_state(
                value,
                maximum_runtime_configs=MAX_PERSISTED_RUNTIME_CONFIGS - 1,
            )
            if len(value["runtime_configs"]) >= MAX_PERSISTED_RUNTIME_CONFIGS:
                raise InstallationConflictError("runtime config store is at capacity")
            value["runtime_configs"][config.runtime_id] = config.to_dict()
            self._write_state(value)

    def get_runtime_config(self, environment_ref: str) -> RuntimeConfig | None:
        _safe_id(environment_ref, field="environment_ref")
        candidates = [
            RuntimeConfig.from_dict(raw)
            for raw in self._read_state()["runtime_configs"].values()
            if isinstance(raw, Mapping) and raw.get("environment_ref") == environment_ref
        ]
        if not candidates:
            return None
        active = [item for item in candidates if item.state == "CONFIRMED"]
        return max(active or candidates, key=lambda item: item.verified_at)

    def mark_runtime_invalidated(self, runtime_id: str) -> RuntimeConfig:
        with self._write_lock():
            value = self._read_state()
            raw = value["runtime_configs"].get(runtime_id)
            if raw is None:
                raise InstallationConfigError("runtime config was not found")
            current = RuntimeConfig.from_dict(raw)
            if current.state == "INVALIDATED":
                return current
            updated = replace(current, state="INVALIDATED", config_digest="")
            updated = replace(
                updated,
                config_digest=sha256_bytes(canonical_json_bytes(updated._payload())),
            )
            value["runtime_configs"][runtime_id] = updated.to_dict()
            self._write_state(value)
            return updated


def _parse_timestamp(value: str) -> float:
    from datetime import datetime

    raw = value[:-1] + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(raw).timestamp()


def _process_alive(process_id: int) -> bool:
    if not process_id or process_id == os.getpid():
        return process_id == os.getpid() and process_id != 0
    try:
        os.kill(process_id, 0)
    except (OSError, ValueError):
        return False
    return True


class InstallExecutor(Protocol):
    """Fixed transport/executor contract used by the manager and tests."""

    def install(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        *,
        completed_component_ids: Sequence[str] = (),
        progress: Callable[[str], None] | None = None,
        timeout_seconds: float,
        transaction_id: str | None = None,
    ) -> Mapping[str, Any]: ...

    def verify(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        result: Mapping[str, Any],
        transaction_id: str | None = None,
    ) -> Mapping[str, Any]: ...

    def finalize(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        *,
        transaction_id: str | None = None,
    ) -> None: ...

    def rollback(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        *,
        finalized: bool,
        transaction_id: str | None = None,
    ) -> bool: ...


def _fixed_ssh_argv(profile: EnvironmentProfile) -> tuple[str, ...]:
    if profile.mode != "ssh" or not profile.ssh_target or not profile.ssh_user or not profile.ssh_port:
        raise InstallationConfigError("SSH installation requires a complete connection profile")
    # The option terminator belongs before the destination.  Everything after
    # the destination is the fixed remote command, never client-provided text.
    return (
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-l",
        profile.ssh_user,
        "-p",
        str(profile.ssh_port),
        "--",
        profile.ssh_target,
        "python3",
        "-",
    )


_REMOTE_INSTALL_SCRIPT = r'''
import hashlib, json, os, pathlib, re, shutil, stat, sys, tarfile, tempfile, time, urllib.request, zipfile

MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_DISK_BYTES = 4 * 1024 * 1024 * 1024
MAX_TIMEOUT = 3600.0
RUNTIME_BASE = (pathlib.Path.home() / ".local/share/molly/runtimes").resolve()
TRANSACTION_BASE = RUNTIME_BASE / ".transactions"
OWNERSHIP_MARKER = ".molly-ownership.json"
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

def owned_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("missing owned path")
    path = pathlib.Path(os.path.expanduser(value)).resolve()
    if not path.is_relative_to(RUNTIME_BASE):
        raise ValueError("path is outside the owned runtime area")
    return path

def transaction_path(value):
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValueError("invalid transaction ID")
    return TRANSACTION_BASE / (value + ".json")

def read_state(transaction_id):
    path = transaction_path(transaction_id)
    if path.is_symlink() or not path.is_file():
        return {}
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("invalid remote transaction state")
    return value

def write_state(transaction_id, value):
    TRANSACTION_BASE.mkdir(parents=True, exist_ok=True)
    path = transaction_path(transaction_id)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".transaction-", dir=TRANSACTION_BASE)
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

def ownership_payload(transaction_id, plan_digest, runtime_id):
    return {
        "transaction_id": transaction_id,
        "plan_digest": plan_digest,
        "runtime_id": runtime_id,
    }

def ownership_matches(directory, transaction_id, plan_digest, runtime_id):
    marker = pathlib.Path(directory) / OWNERSHIP_MARKER
    if marker.is_symlink() or not marker.is_file():
        return False
    try:
        with marker.open(encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return value == ownership_payload(transaction_id, plan_digest, runtime_id)

def ensure_ownership_marker(stage, transaction_id, plan_digest, runtime_id):
    marker = pathlib.Path(stage) / OWNERSHIP_MARKER
    if marker.is_symlink():
        raise ValueError("runtime ownership marker cannot be a symlink")
    if marker.exists():
        if not ownership_matches(stage, transaction_id, plan_digest, runtime_id):
            raise ValueError("runtime ownership marker does not match the transaction")
        return
    descriptor = -1
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            descriptor = -1
            json.dump(ownership_payload(transaction_id, plan_digest, runtime_id), output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not ownership_matches(stage, transaction_id, plan_digest, runtime_id):
        raise ValueError("runtime ownership marker could not be established")

def safe_rel(value):
    path = pathlib.PurePosixPath(value)
    if not isinstance(value, str) or not value or path.is_absolute() or "\\" in value or any(p in {"", ".", ".."} for p in path.parts):
        raise ValueError("unsafe relative path")
    return path

def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("fixed installation timeout exceeded")

def fetch(item, destination, deadline):
    url = item["source_url"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("remote sources must be HTTPS")
    expected = item["sha256"]
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("missing fixed SHA-256")
    maximum = min(int(item["max_download_bytes"]), MAX_DOWNLOAD_BYTES)
    digest = hashlib.sha256()
    size = 0
    check_deadline(deadline)
    request = urllib.request.Request(url, method="GET")
    with OPENER.open(request, timeout=max(1.0, min(30.0, deadline - time.monotonic()))) as response, open(destination, "wb") as output:
        while True:
            check_deadline(deadline)
            block = response.read(64 * 1024)
            if not block:
                break
            size += len(block)
            if size > maximum:
                raise ValueError("download safety limit exceeded")
            digest.update(block)
            output.write(block)
    if digest.hexdigest() != expected:
        raise ValueError("download SHA-256 mismatch")
    if size != int(item["estimated_download_bytes"]):
        raise ValueError("download size mismatch")
    return size

def safe_extract(archive, destination, kind, disk_limit, deadline):
    destination = pathlib.Path(destination).resolve()
    total = 0
    if kind in ("archive", "zip"):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                check_deadline(deadline)
                relative = safe_rel(member.filename)
                target = (destination / relative).resolve()
                if not target.is_relative_to(destination):
                    raise ValueError("archive path traversal")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise ValueError("archive symlinks are not allowed")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                total += member.file_size
                if total > int(disk_limit) or total > MAX_DISK_BYTES:
                    raise ValueError("extracted size safety limit exceeded")
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as input_file, open(target, "wb") as output:
                    while block := input_file.read(64 * 1024):
                        check_deadline(deadline)
                        output.write(block)
                member_mode = (member.external_attr >> 16) & 0o777
                os.chmod(target, 0o700 if member_mode & 0o111 else 0o600)
    else:
        with tarfile.open(archive, "r:*") as source:
            for member in source.getmembers():
                check_deadline(deadline)
                relative = safe_rel(member.name)
                target = (destination / relative).resolve()
                if not target.is_relative_to(destination) or member.issym() or member.islnk() or member.isdev():
                    raise ValueError("unsafe tar member")
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ValueError("unsupported tar member")
                total += member.size
                if total > int(disk_limit) or total > MAX_DISK_BYTES:
                    raise ValueError("extracted size safety limit exceeded")
                target.parent.mkdir(parents=True, exist_ok=True)
                input_file = source.extractfile(member)
                if input_file is None:
                    raise ValueError("tar member could not be read")
                with input_file, open(target, "wb") as output:
                    while block := input_file.read(64 * 1024):
                        check_deadline(deadline)
                        output.write(block)
                os.chmod(target, 0o700 if member.mode & 0o111 else 0o600)

def component_complete(stage, item):
    destination = (stage / safe_rel(item["install_subdirectory"])).resolve()
    if not destination.is_dir() or destination.is_symlink() or not destination.is_relative_to(stage):
        return False
    required = item.get("required_paths", [])
    if required:
        return all(
            (destination / safe_rel(path)).is_file()
            and not (destination / safe_rel(path)).is_symlink()
            and (destination / safe_rel(path)).resolve().is_relative_to(destination)
            for path in required
        )
    return any(path.is_file() and not path.is_symlink() for path in destination.rglob("*"))

def python_component_is_executable(stage, item):
    if item.get("component_id") != "python":
        return True
    destination = (pathlib.Path(stage) / safe_rel(item["install_subdirectory"])).resolve()
    if not destination.is_dir() or destination.is_symlink():
        return False
    required = item.get("required_paths", [])
    candidates = []
    for raw_path in required:
        candidate = (destination / safe_rel(raw_path)).resolve()
        name = candidate.name.casefold()
        if name == "python" or name.startswith("python3") or name == "python.exe":
            candidates.append(candidate)
    if not candidates:
        candidates = [
            path for path in destination.rglob("*")
            if path.is_file()
            and (
                path.name.casefold() == "python"
                or path.name.casefold().startswith("python3")
                or path.name.casefold() == "python.exe"
            )
        ]
    return any(
        path.is_file()
        and not path.is_symlink()
        and path.is_relative_to(destination)
        and bool(path.stat().st_mode & 0o111)
        and os.access(path, os.X_OK)
        for path in candidates
    )

def digest_file(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while True:
            block = source.read(64 * 1024)
            if not block:
                break
            size += len(block)
            if size > MAX_DISK_BYTES:
                raise ValueError("verified file exceeds fixed disk limit")
            digest.update(block)
    return size, digest.hexdigest()

def verified_files(stage, entries):
    result = []
    stage = pathlib.Path(stage).resolve()
    for item in entries:
        raw_paths = item.get("required_paths", [])
        if not raw_paths and item.get("install_kind") == "file":
            raw_paths = [item.get("install_filename", "")]
        for raw_path in raw_paths:
            relative = safe_rel(raw_path)
            candidate = (stage / safe_rel(item["install_subdirectory"]) / relative).resolve()
            if not candidate.is_file() or candidate.is_symlink() or not candidate.is_relative_to(stage):
                continue
            size, digest = digest_file(candidate)
            result.append({
                "component_id": item.get("component_id"),
                "path": str(candidate),
                "size_bytes": size,
                "sha256": digest,
            })
    return result

def main(request):
    operation = request.get("operation")
    transaction_id = request.get("transaction_id")
    transaction_path(transaction_id)
    plan_digest = request.get("plan_digest")
    if not isinstance(plan_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", plan_digest):
        raise ValueError("missing plan digest")
    state = read_state(transaction_id)
    if state and state.get("plan_digest") != plan_digest:
        raise ValueError("remote transaction is bound to another plan")
    deadline = time.monotonic() + min(float(request.get("timeout_seconds", MAX_TIMEOUT)), MAX_TIMEOUT)
    stage = owned_path(request.get("stage_directory"))
    target = owned_path(request.get("target_directory"))
    if operation == "install":
        if state.get("state") == "ENABLED" and target.is_dir() and not target.is_symlink():
            if not ownership_matches(target, transaction_id, plan_digest, request.get("runtime_id")):
                raise ValueError("enabled runtime ownership marker does not match the transaction")
            print(json.dumps({"ok": True, "verified": True, "state": "ENABLED", "target_exists": True, "verified_files": verified_files(target, request.get("entries", []))}, separators=(",", ":")))
            return
        if stage.exists() and stage.is_symlink():
            raise ValueError("staging directory cannot be a symlink")
        stage.mkdir(parents=True, exist_ok=True)
        ensure_ownership_marker(stage, transaction_id, plan_digest, request.get("runtime_id"))
        completed = set(state.get("completed_component_ids", []))
        write_state(transaction_id, {"state": "INSTALLING", "plan_digest": plan_digest, "runtime_id": request.get("runtime_id"), "stage_directory": request.get("stage_directory"), "target_directory": request.get("target_directory"), "completed_component_ids": sorted(completed)})
        for item in request.get("entries", []):
            check_deadline(deadline)
            if item.get("component_id") in completed and component_complete(stage, item):
                continue
            destination = (stage / safe_rel(item["install_subdirectory"])).resolve()
            if not destination.is_relative_to(stage):
                raise ValueError("unsafe install destination")
            destination.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(prefix=".download-", dir=stage, delete=False) as temporary:
                downloaded = temporary.name
            try:
                size = fetch(item, downloaded, deadline)
                if item["install_kind"] == "file":
                    output = (destination / safe_rel(item["install_filename"])).resolve()
                    if (
                        not output.is_relative_to(destination)
                        or int(item["estimated_disk_bytes"]) <= 0
                        or int(item["estimated_disk_bytes"]) < size
                    ):
                        raise ValueError("file exceeds fixed disk estimate")
                    shutil.copyfile(downloaded, output)
                    os.chmod(output, 0o700 if item.get("component_id") == "python" else 0o600)
                else:
                    safe_extract(downloaded, destination, item["install_kind"], item["estimated_disk_bytes"], deadline)
            finally:
                try:
                    os.unlink(downloaded)
                except FileNotFoundError:
                    pass
            if not component_complete(stage, item) or not python_component_is_executable(stage, item):
                raise ValueError("required runtime file is missing")
            completed.add(item["component_id"])
            write_state(transaction_id, {"state": "INSTALLING", "plan_digest": plan_digest, "runtime_id": request.get("runtime_id"), "stage_directory": request.get("stage_directory"), "target_directory": request.get("target_directory"), "completed_component_ids": sorted(completed)})
        write_state(transaction_id, {"state": "VERIFIED", "plan_digest": plan_digest, "runtime_id": request.get("runtime_id"), "stage_directory": request.get("stage_directory"), "target_directory": request.get("target_directory"), "completed_component_ids": sorted(completed)})
        print(json.dumps({"ok": True, "verified": True, "state": "VERIFIED", "target_exists": False, "verified_files": verified_files(stage, request.get("entries", []))}, separators=(",", ":")))
        return
    if operation == "status":
        target_present = target.is_dir() and not target.is_symlink()
        stage_present = stage.is_dir() and not stage.is_symlink()
        target_exists = target_present and ownership_matches(target, transaction_id, plan_digest, request.get("runtime_id"))
        stage_exists = stage_present and ownership_matches(stage, transaction_id, plan_digest, request.get("runtime_id"))
        remote_state = state.get("state")
        verified = remote_state in {"VERIFIED", "ENABLED"} and (stage_exists or target_exists)
        evidence_root = target if target_exists else stage
        print(json.dumps({"ok": True, "state": remote_state, "verified": verified, "target_exists": target_exists, "stage_exists": stage_exists, "unowned_target_exists": target_present and not target_exists, "verified_files": verified_files(evidence_root, request.get("entries", [])) if verified else []}, separators=(",", ":")))
        return
    if operation == "finalize":
        target_present = target.is_dir() and not target.is_symlink()
        stage_present = stage.is_dir() and not stage.is_symlink()
        target_exists = target_present and ownership_matches(target, transaction_id, plan_digest, request.get("runtime_id"))
        if state.get("state") == "ENABLED" and target_present:
            if not target_exists:
                raise ValueError("enabled runtime target ownership marker does not match")
            print(json.dumps({"ok": True, "state": "ENABLED", "target_exists": True, "verified_files": verified_files(target, request.get("entries", []))}, separators=(",", ":")))
            return
        if state.get("state") != "VERIFIED":
            raise ValueError("remote transaction is not ready to finalize")
        if target_present and not stage_present:
            if not target_exists:
                raise ValueError("runtime target ownership marker does not match")
            write_state(transaction_id, {**state, "state": "ENABLED"})
            print(json.dumps({"ok": True, "state": "ENABLED", "target_exists": True, "verified_files": verified_files(target, request.get("entries", []))}, separators=(",", ":")))
            return
        if target.exists() or target.is_symlink() or not stage_present:
            raise ValueError("remote runtime target or staging directory is unsafe")
        if not ownership_matches(stage, transaction_id, plan_digest, request.get("runtime_id")):
            raise ValueError("staging ownership marker does not match the transaction")
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(stage, target)
        write_state(transaction_id, {**state, "state": "ENABLED"})
        print(json.dumps({"ok": True, "state": "ENABLED", "target_exists": True, "verified_files": verified_files(target, request.get("entries", []))}, separators=(",", ":")))
        return
    if operation == "rollback":
        # ``finalize`` can finish the atomic rename and state write before
        # the SSH response reaches the caller.  The durable remote state is
        # authoritative; do not let the caller's stale finalized flag leave
        # an enabled target behind.  The VERIFIED/stage-missing case is the
        # same rename window before the state write.
        remote_state = state.get("state")
        target_present = target.is_dir() and not target.is_symlink()
        stage_present = stage.is_dir() and not stage.is_symlink()
        target_owned = target_present and ownership_matches(target, transaction_id, plan_digest, request.get("runtime_id"))
        stage_owned = stage_present and ownership_matches(stage, transaction_id, plan_digest, request.get("runtime_id"))
        remove_target = (
            remote_state == "ENABLED"
            or (remote_state == "VERIFIED" and not stage_present)
        ) and target_present
        if remove_target and not target_owned:
            raise ValueError("runtime target ownership marker does not match")
        if stage_present and not stage_owned:
            raise ValueError("staging ownership marker does not match")
        if remove_target:
            shutil.rmtree(target)
        if stage_present:
            shutil.rmtree(stage)
        write_state(transaction_id, {**state, "state": "ROLLED_BACK"})
        print(json.dumps({"ok": True}, separators=(",", ":")))
        return
    raise ValueError("unsupported fixed installation operation")

'''


class RestrictedInstallExecutor:
    """Install fixed archives/files locally or through a fixed SSH script."""

    def __init__(
        self,
        *,
        runner: Callable[[Sequence[str], bytes | None, float], tuple[int, bytes]] | None = None,
        opener: Any | None = None,
    ) -> None:
        self.runner = runner or _default_runner
        self.opener = opener or urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler()
        )

    def install(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        *,
        completed_component_ids: Sequence[str] = (),
        progress: Callable[[str], None] | None = None,
        timeout_seconds: float,
        transaction_id: str | None = None,
    ) -> Mapping[str, Any]:
        if profile.mode == "ssh":
            return self._remote_call(
                profile,
                plan,
                stage_directory,
                "install",
                timeout_seconds,
                transaction_id=transaction_id,
                completed_component_ids=completed_component_ids,
            )
        stage = _owned_directory(Path(stage_directory), create=True)
        deadline = time.monotonic() + min(float(timeout_seconds), INSTALLATION_TIMEOUT_SECONDS)
        completed = set(completed_component_ids)
        results: list[dict[str, Any]] = []
        for entry in plan.entries:
            if time.monotonic() > deadline:
                raise InstallationExecutionError("安装超过固定超时时间")
            if entry.component_id in completed and _component_is_complete(stage, entry):
                results.append({"component_id": entry.component_id, "reused_stage": True})
                continue
            try:
                result = self._install_entry(entry, stage, deadline)
            except InstallationError:
                raise
            except Exception as exc:
                raise InstallationExecutionError(f"安装 {entry.name} 失败") from exc
            results.append(result)
            if progress is not None:
                progress(entry.component_id)
        return {"transport": "local", "verified": True, "components": results}

    def verify(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        result: Mapping[str, Any],
        transaction_id: str | None = None,
    ) -> Mapping[str, Any]:
        if profile.mode == "ssh":
            status = self._remote_call(
                profile,
                plan,
                stage_directory,
                "status",
                INSTALLATION_TIMEOUT_SECONDS,
                transaction_id=transaction_id,
            )
            remote_state = status.get("state")
            if remote_state not in {"VERIFIED", "ENABLED"} or not bool(status.get("verified")):
                raise InstallationIntegrityError("远程事务状态未通过固定验证")
            return {
                "staged_components": list(plan.component_ids),
                "transport_verified": True,
                "remote_state": remote_state,
                "target_exists": bool(status.get("target_exists")),
                "verified_files": thaw_json(status.get("verified_files", [])),
            }
        stage = Path(stage_directory)
        for entry in plan.entries:
            if not _component_is_complete(stage, entry):
                raise InstallationIntegrityError(f"{entry.name} 安装后缺少固定验证文件")
        downloads = stage / ".downloads"
        if downloads.exists() or downloads.is_symlink():
            if downloads.is_symlink():
                raise InstallationIntegrityError("临时下载目录不能是符号链接")
            shutil.rmtree(downloads)
        return {"staged_components": list(plan.component_ids), "transport_verified": True}

    def finalize(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        *,
        transaction_id: str | None = None,
    ) -> None:
        if profile.mode == "ssh":
            self._remote_call(
                profile,
                plan,
                stage_directory,
                "finalize",
                INSTALLATION_TIMEOUT_SECONDS,
                transaction_id=transaction_id,
            )
            return
        stage = _owned_directory(Path(stage_directory), create=False)
        target = _local_target_from_stage(stage, plan.runtime_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise InstallationConflictError("runtime target already exists")
        os.replace(stage, target)

    def rollback(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        *,
        finalized: bool,
        transaction_id: str | None = None,
    ) -> bool:
        if profile.mode == "ssh":
            try:
                self._remote_call(
                    profile,
                    plan,
                    stage_directory,
                    "rollback",
                    INSTALLATION_TIMEOUT_SECONDS,
                    finalized=finalized,
                    transaction_id=transaction_id,
                )
            except Exception:
                # A remote cleanup failure is itself recoverable.  The manager
                # keeps the durable transaction in ROLLING_BACK so a later
                # recovery can retry the idempotent helper operation.
                return False
        else:
            stage = Path(stage_directory)
            if _is_safe_runtime_path(stage) and (stage.exists() or stage.is_symlink()):
                if stage.is_symlink():
                    stage.unlink()
                else:
                    shutil.rmtree(stage)
            if finalized:
                target = _local_target_from_stage(stage, plan.runtime_id)
                if _is_safe_runtime_path(target) and (target.exists() or target.is_symlink()):
                    if target.is_symlink():
                        target.unlink()
                    else:
                        shutil.rmtree(target)
        return True

    def _install_entry(self, entry: InstallManifestEntry, stage: Path, deadline: float) -> dict[str, Any]:
        if not entry.installable or not entry.sha256:
            raise InstallationIntegrityError(f"{entry.name} 没有固定 SHA-256")
        if entry.estimated_download_bytes > entry.max_download_bytes:
            raise InstallationIntegrityError(f"{entry.name} 超过固定下载上限")
        downloads = stage / ".downloads"
        _owned_directory(downloads, create=True)
        download_path = downloads / f"{entry.component_id}.payload"
        if not _verified_file(download_path, entry.sha256, entry.estimated_download_bytes):
            remaining = max(0.01, deadline - time.monotonic())
            _download_fixed(self.opener, entry, download_path, remaining)
        destination = _owned_directory(stage / entry.install_subdirectory, create=True)
        if entry.install_kind == "file":
            output = destination / entry.install_filename
            if entry.estimated_disk_bytes <= 0 or entry.estimated_download_bytes > entry.estimated_disk_bytes:
                raise InstallationIntegrityError(f"{entry.name} exceeds its fixed disk estimate")
            shutil.copyfile(download_path, output)
            os.chmod(output, 0o700 if entry.component_id == "python" else 0o600)
        else:
            _safe_extract_archive(
                download_path,
                destination,
                entry.install_kind,
                entry.estimated_disk_bytes,
                deadline=deadline,
            )
        if not _component_is_complete(stage, entry):
            raise InstallationIntegrityError(f"{entry.name} 安装后未通过固定文件验证")
        if entry.component_id == "python" and not _python_component_is_executable(stage, entry):
            raise InstallationIntegrityError("Python runtime 安装后不可执行")
        return {
            "component_id": entry.component_id,
            "version": entry.version,
            "sha256": entry.sha256,
            "size_bytes": entry.estimated_download_bytes,
        }

    def _remote_call(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        stage_directory: str,
        operation: str,
        timeout_seconds: float,
        *,
        finalized: bool = False,
        transaction_id: str | None = None,
        completed_component_ids: Sequence[str] = (),
    ) -> Mapping[str, Any]:
        if operation not in {"install", "status", "finalize", "rollback"}:
            raise InstallationConfigError("unsupported remote installation operation")
        request = {
            "operation": operation,
            "stage_directory": stage_directory,
            "target_directory": plan.target_directory,
            "finalized": finalized,
            "transaction_id": transaction_id or "unknown",
            "runtime_id": plan.runtime_id,
            "plan_digest": plan.plan_digest,
            "completed_component_ids": list(completed_component_ids),
            "timeout_seconds": min(float(timeout_seconds), INSTALLATION_TIMEOUT_SECONDS),
            "entries": [entry.to_dict(include_internal=True) for entry in plan.entries],
        }
        try:
            request_json = json.dumps(request, ensure_ascii=True, separators=(",", ":"))
            remote_source = (
                _REMOTE_INSTALL_SCRIPT
                + "\ntry:\n"
                + "    main(json.loads("
                + repr(request_json)
                + "))\n"
                + "except Exception as exc:\n"
                + "    print(json.dumps({'ok': False, 'error': str(exc)[:512]}, separators=(',', ':')))\n"
                + "    raise SystemExit(2)\n"
            )
            returncode, output = self.runner(
                _fixed_ssh_argv(profile),
                remote_source.encode("utf-8"),
                min(float(timeout_seconds), INSTALLATION_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            raise InstallationExecutionError("SSH 安装传输失败") from exc
        if returncode != 0:
            raise InstallationExecutionError("SSH 固定安装工具执行失败")
        try:
            value = json.loads(output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InstallationExecutionError("SSH 安装工具返回了无效结果") from exc
        if not isinstance(value, Mapping) or not bool(value.get("ok")):
            raise InstallationExecutionError("SSH 安装工具未确认成功")
        return dict(value)


def _is_safe_runtime_path(path: Path) -> bool:
    candidate = path.absolute()
    parts = candidate.parts
    return len(parts) >= 2 and (".runtime-staging" in parts or "runtimes" in parts)


def _owned_directory(path: Path, *, create: bool) -> Path:
    if path.exists() and path.is_symlink():
        raise InstallationIntegrityError("runtime directory cannot be a symlink")
    if path.exists() and not path.is_dir():
        raise InstallationIntegrityError("runtime directory is not a directory")
    if create:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or path.is_symlink():
        raise InstallationIntegrityError("runtime directory could not be prepared safely")
    os.chmod(path, 0o700)
    return path


def _local_target_from_stage(stage: Path, runtime_id: str) -> Path:
    root = stage.absolute()
    marker = ".runtime-staging"
    if marker not in root.parts:
        raise InstallationIntegrityError("staging path is outside the owned runtime area")
    marker_index = len(root.parts) - 1 - tuple(reversed(root.parts)).index(marker)
    if marker_index == 0 or len(root.parts) != marker_index + 2:
        raise InstallationIntegrityError("staging path has an invalid runtime layout")
    stage_name = root.parts[-1]
    if not stage_name.startswith(f"{runtime_id}-"):
        raise InstallationIntegrityError("staging path is bound to another runtime")
    base = Path(*root.parts[:marker_index])
    return base / "runtimes" / runtime_id


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Keep a fixed manifest source bound to its declared authority."""

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Any:
        return None


def _verified_file(path: Path, expected_sha256: str, expected_size: int) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if expected_size and path.stat().st_size != expected_size:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(64 * 1024):
                digest.update(block)
        return digest.hexdigest() == expected_sha256
    except OSError:
        return False


def _file_digest(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise InstallationIntegrityError("固定验证文件不可读")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while block := handle.read(64 * 1024):
                size += len(block)
                if size > MAX_EXTRACTED_FILE_BYTES:
                    raise InstallationIntegrityError("固定验证文件超过磁盘上限")
                digest.update(block)
    except OSError as exc:
        raise InstallationIntegrityError("固定验证文件不可读") from exc
    if size <= 0:
        raise InstallationIntegrityError("固定验证文件为空")
    return size, digest.hexdigest()


def _download_fixed(opener: Any, entry: InstallManifestEntry, destination: Path, timeout_seconds: float) -> None:
    deadline = time.monotonic() + min(float(timeout_seconds), INSTALLATION_TIMEOUT_SECONDS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{entry.component_id}.", suffix=".part", dir=str(destination.parent))
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        parsed = urlsplit(entry.source_url)
        if parsed.scheme == "file":
            source = Path(unquote(parsed.path)).absolute()
            if source.is_symlink() or not source.is_file():
                raise InstallationExecutionError("固定安装源文件不可读")
            handle: Any = source.open("rb")
        else:
            request = urllib.request.Request(entry.source_url, method="GET")
            handle = opener.open(request, timeout=min(30.0, max(1.0, deadline - time.monotonic())))
        with handle, os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            while True:
                if time.monotonic() > deadline:
                    raise InstallationExecutionError("固定安装源下载超时")
                block = handle.read(64 * 1024)
                if not block:
                    break
                size += len(block)
                if size > entry.max_download_bytes or size > MAX_INSTALL_DOWNLOAD_BYTES:
                    raise InstallationIntegrityError("下载超过固定大小上限")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if size != entry.estimated_download_bytes:
            raise InstallationIntegrityError("下载大小与固定清单不一致")
        if digest.hexdigest() != entry.sha256:
            raise InstallationIntegrityError("下载内容 SHA-256 校验失败")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination)
    except (InstallationError, OSError, urllib.error.URLError) as exc:
        if isinstance(exc, InstallationError):
            raise
        raise InstallationExecutionError("固定安装源下载失败") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _safe_extract_archive(
    archive: Path,
    destination: Path,
    kind: str,
    disk_limit: int,
    *,
    deadline: float | None = None,
) -> None:
    destination = _owned_directory(destination, create=True).resolve()
    total = 0
    try:
        if kind in {"archive", "zip"}:
            source = zipfile.ZipFile(archive)
            members = source.infolist()
            with source:
                for member in members:
                    if deadline is not None and time.monotonic() > deadline:
                        raise InstallationExecutionError("安装解压超过固定超时时间")
                    relative = _relative_path(member.filename, field="archive member")
                    target = (destination / relative).resolve()
                    if not target.is_relative_to(destination):
                        raise InstallationIntegrityError("archive path traversal detected")
                    mode = (member.external_attr >> 16) & stat.S_IFMT(0o170000)
                    if mode == stat.S_IFLNK:
                        raise InstallationIntegrityError("archive symlinks are not allowed")
                    if member.is_dir():
                        _owned_directory(target, create=True)
                        continue
                    total += member.file_size
                    if total > disk_limit or total > MAX_EXTRACTED_FILE_BYTES:
                        raise InstallationIntegrityError("extracted files exceed fixed disk limit")
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    with source.open(member) as input_file, target.open("wb") as output:
                        while block := input_file.read(64 * 1024):
                            if deadline is not None and time.monotonic() > deadline:
                                raise InstallationExecutionError("安装解压超过固定超时时间")
                            output.write(block)
                    member_mode = (member.external_attr >> 16) & 0o777
                    os.chmod(target, 0o700 if member_mode & 0o111 else 0o600)
        else:
            with tarfile.open(archive, "r:*") as source:
                for member in source.getmembers():
                    if deadline is not None and time.monotonic() > deadline:
                        raise InstallationExecutionError("安装解压超过固定超时时间")
                    relative = _relative_path(member.name, field="archive member")
                    target = (destination / relative).resolve()
                    if not target.is_relative_to(destination) or member.issym() or member.islnk() or member.isdev():
                        raise InstallationIntegrityError("tar contains an unsafe member")
                    if member.isdir():
                        _owned_directory(target, create=True)
                        continue
                    if not member.isfile():
                        raise InstallationIntegrityError("tar contains an unsupported member")
                    total += member.size
                    if total > disk_limit or total > MAX_EXTRACTED_FILE_BYTES:
                        raise InstallationIntegrityError("extracted files exceed fixed disk limit")
                    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    input_file = source.extractfile(member)
                    if input_file is None:
                        raise InstallationIntegrityError("tar member could not be read")
                    with input_file, target.open("wb") as output:
                        while block := input_file.read(64 * 1024):
                            if deadline is not None and time.monotonic() > deadline:
                                raise InstallationExecutionError("安装解压超过固定超时时间")
                            output.write(block)
                    os.chmod(target, 0o700 if member.mode & 0o111 else 0o600)
    except (InstallationError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        if isinstance(exc, InstallationError):
            raise
        raise InstallationIntegrityError("固定安装包无法安全解压") from exc


def _python_component_is_executable(
    stage: Path,
    entry: InstallManifestEntry,
) -> bool:
    destination = (stage / entry.install_subdirectory).resolve()
    if not destination.is_dir() or destination.is_symlink():
        return False
    relative_paths = entry.required_paths
    if not relative_paths and entry.install_kind == "file":
        relative_paths = (entry.install_filename,)
    candidates = [
        (destination / relative).resolve()
        for relative in relative_paths
        if (
            PurePosixPath(relative).name.casefold() == "python"
            or PurePosixPath(relative).name.casefold().startswith("python3")
            or PurePosixPath(relative).name.casefold() == "python.exe"
        )
    ]
    if not candidates:
        candidates = [
            item
            for item in destination.rglob("*")
            if item.is_file()
            and (
                item.name.casefold() == "python"
                or item.name.casefold().startswith("python3")
                or item.name.casefold() == "python.exe"
            )
        ]
    return any(
        candidate.is_file()
        and not candidate.is_symlink()
        and candidate.is_relative_to(destination)
        and bool(candidate.stat().st_mode & stat.S_IXUSR)
        and os.access(candidate, os.X_OK)
        for candidate in candidates
    )


def _component_is_complete(stage: Path, entry: InstallManifestEntry) -> bool:
    destination = (stage / entry.install_subdirectory).resolve()
    if not destination.is_dir() or destination.is_symlink() or not destination.is_relative_to(stage.resolve()):
        return False
    if entry.required_paths:
        return all(
            (destination / relative).is_file()
            and not (destination / relative).is_symlink()
            and (destination / relative).resolve().is_relative_to(destination)
            for relative in entry.required_paths
        )
    return any(item.is_file() and not item.is_symlink() for item in destination.rglob("*"))


_REQUIRED_RUNTIME_COMPONENT_IDS = frozenset(
    {"python", "unimol", "reinvent4", "unimol-weights"}
)


def _runtime_components_from_match(
    match: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return the complete, freshly matched runtime binding.

    Installation entries are not evidence that a component became usable.
    Only the post-confirmation discovery match can establish that fact.  Keep
    the complete reusable list so a runtime config records both installed and
    reused interpreters/components instead of only the downloaded entries.
    """

    if not isinstance(match, Mapping) or match.get("status") != "READY":
        raise InstallationIntegrityError(
            "安装后重新探测未匹配到完整可用运行环境"
        )
    raw_reusable = match.get("reusable", ())
    if not isinstance(raw_reusable, Sequence) or isinstance(
        raw_reusable, (str, bytes, bytearray)
    ):
        raise InstallationIntegrityError("安装后重新探测的复用组件列表无效")
    components: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_reusable:
        if not isinstance(raw, Mapping):
            raise InstallationIntegrityError("安装后重新探测的组件绑定无效")
        component_id = raw.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise InstallationIntegrityError("安装后重新探测的组件缺少标识")
        if component_id in seen:
            raise InstallationIntegrityError("安装后重新探测包含重复组件")
        seen.add(component_id)
        value = thaw_json(raw)
        value["verified"] = True
        components.append(value)
    missing = _REQUIRED_RUNTIME_COMPONENT_IDS - seen
    if missing:
        raise InstallationIntegrityError(
            "安装后重新探测仍缺少固定组件：" + ", ".join(sorted(missing))
        )
    return tuple(components)


class InstallationManager:
    """Create, approve, execute, verify, and persist fixed runtime plans."""

    def __init__(
        self,
        root: Path | str,
        *,
        environment_manager: EnvironmentManager,
        manifest: InstallManifest | None = None,
        manifest_path: Path | str | None = None,
        store: RuntimeInstallationStore | None = None,
        executor: InstallExecutor | None = None,
        timeout_seconds: float = INSTALLATION_TIMEOUT_SECONDS,
    ) -> None:
        if not isinstance(environment_manager, EnvironmentManager):
            raise TypeError("environment_manager must be an EnvironmentManager")
        if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or not 0 < timeout_seconds <= INSTALLATION_TIMEOUT_SECONDS:
            raise InstallationConfigError("installation timeout is invalid")
        configured = Path(root)
        if configured.is_symlink():
            raise InstallationConfigError("installation root cannot be a symlink")
        self.root = configured.absolute()
        self.environment_manager = environment_manager
        if manifest is not None and manifest_path is not None:
            raise InstallationConfigError("provide either manifest or manifest_path, not both")
        configured_manifest = _configured_manifest_path(self.root, manifest_path)
        self.manifest = (
            manifest
            if manifest is not None
            else InstallManifest.from_json_file(configured_manifest)
            if configured_manifest is not None
            else default_install_manifest()
        )
        if not isinstance(self.manifest, InstallManifest):
            raise TypeError("manifest must be an InstallManifest")
        self.store = store or RuntimeInstallationStore(self.root)
        self.executor = executor or RestrictedInstallExecutor()
        self.timeout_seconds = float(timeout_seconds)

    def _current_binding(self, environment_ref: str) -> tuple[EnvironmentProfile, EnvironmentReport, Mapping[str, Any]]:
        profile = self.environment_manager.store.get_profile(environment_ref)
        public = self.environment_manager.get_public(environment_ref)
        detection = public.get("detection")
        if not isinstance(detection, Mapping):
            raise InstallationConfigError("请先完成当前连接的只读环境检测")
        report_value = detection.get("report")
        match = detection.get("match")
        if not isinstance(report_value, Mapping) or not isinstance(match, Mapping):
            raise InstallationConfigError("当前环境检测报告不完整")
        report_payload = {
            key: value for key, value in report_value.items() if key != "report_digest"
        }
        expected_report_digest = sha256_bytes(canonical_json_bytes(report_payload))
        if report_value.get("report_digest") != expected_report_digest:
            raise InstallationIntegrityError("当前环境检测报告摘要无效")
        report = EnvironmentReport(
            environment_ref=profile.environment_ref,
            connection_digest=profile.connection_digest,
            mode=profile.mode,
            target_label=profile.target_label,
            detected_at=report_value.get("detected_at"),
            probes=tuple(report_value.get("probes", ())),
            data={
                key: value
                for key, value in report_value.items()
                if key in {"system", "disk", "gpu", "python", "unimol", "reinvent4", "weights"}
            },
            report_digest=report_value.get("report_digest"),
        )
        if report.connection_digest != profile.connection_digest:
            raise InstallationConfigError("当前环境检测报告已绑定到其他连接")
        return profile, report, match

    def build_plan(
        self,
        environment_ref: str,
        *,
        selected_component_ids: Sequence[str] | None = None,
    ) -> InstallPlan:
        profile, report, match = self._current_binding(environment_ref)
        existing_runtime = self.runtime_for_environment(environment_ref)
        force_reinstall = bool(
            existing_runtime is not None
            and existing_runtime.state == "INVALIDATED"
            and existing_runtime.catalog_digest != self.manifest.digest
        )
        plan = InstallPlan.build(
            profile,
            report,
            match,
            self.manifest,
            selected_component_ids=selected_component_ids,
            existing_runtime=existing_runtime,
            force_reinstall=force_reinstall,
        )
        self.store.save_plan(plan)
        return plan

    def _validate_approval(self, plan: InstallPlan, payload: Mapping[str, Any]) -> None:
        allowed = {"confirm", "plan_id", "plan_digest", "connection_digest", "report_digest"}
        if set(payload) - allowed:
            raise InstallationConfigError("安装确认包含不支持的字段")
        if payload.get("confirm") is not True:
            raise InstallationConfigError("必须明确确认一次安装计划")
        for field in ("plan_id", "plan_digest", "connection_digest", "report_digest"):
            if payload.get(field) != getattr(plan, field):
                raise InstallationConflictError(f"安装确认的 {field} 与服务器计划不一致")
        _, report, _ = self._current_binding(plan.environment_ref)
        allowed_report_digests = {plan.report_digest}
        existing = self.store.get_installation_for_plan(plan.plan_id)
        if existing is not None and isinstance(existing.verification, Mapping):
            for key in ("reprobe", "final_reprobe"):
                reprobe = existing.verification.get(key)
                if isinstance(reprobe, Mapping) and isinstance(
                    reprobe.get("report_digest"), str
                ):
                    allowed_report_digests.add(reprobe["report_digest"])
        if report.report_digest not in allowed_report_digests:
            raise InstallationConflictError("环境检测报告已变化，请重新检测并生成安装计划")
        if self.manifest.catalog_version != plan.catalog_version or self.manifest.digest != plan.catalog_digest:
            raise InstallationConflictError("固定兼容性清单已变化，请重新生成安装计划")

    def confirm(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, Mapping):
            raise InstallationConfigError("安装确认必须是 JSON 对象")
        plan_id = payload.get("plan_id")
        _safe_id(plan_id, field="plan_id")
        plan = self.store.get_plan(plan_id)
        self._validate_approval(plan, payload)
        if plan.status == "ALREADY_CONFIRMED":
            config = self.runtime_for_environment(plan.environment_ref)
            return {
                "installation": None,
                "runtime_config": config.to_dict(public=True) if config else None,
                "idempotent_replay": True,
            }
        if plan.status == "NO_INSTALL_REQUIRED":
            config = self.runtime_for_environment(plan.environment_ref)
            return {
                "installation": None,
                "runtime_config": config.to_dict(public=True) if config else None,
            }
        if plan.status == "READY_TO_CONFIRM":
            return self._confirm_existing_environment(plan)
        if plan.status != "READY_TO_INSTALL":
            raise InstallationConfigError("当前安装计划未通过固定清单和许可证检查")
        existing_runtime = self.runtime_for_environment(plan.environment_ref)
        if existing_runtime is not None and existing_runtime.state == "CONFIRMED" and existing_runtime.runtime_id != plan.runtime_id:
            raise InstallationConflictError("当前连接已有已确认运行配置，请使用最新环境方案")
        existing, created = self.store.claim_approval(plan)
        if existing.state == "CONFIRMED":
            config = self.store.get_runtime_config(plan.environment_ref)
            return {"installation": existing.to_dict(public=True), "runtime_config": config.to_dict(public=True) if config else None}
        if existing.state in {"FAILED", "ROLLED_BACK"}:
            return {
                "installation": existing.to_dict(public=True),
                "runtime_config": None,
                "idempotent_replay": True,
            }
        if existing.state != "APPROVED":
            return {"installation": existing.to_dict(public=True), "runtime_config": None, "idempotent_replay": True}
        claimed, should_execute = self.store.claim_execution(existing.installation_id)
        if not should_execute:
            return {"installation": claimed.to_dict(public=True), "runtime_config": None, "idempotent_replay": True}
        completed = self._execute(claimed, plan)
        config = self.store.get_runtime_config(plan.environment_ref)
        return {
            "installation": completed.to_dict(public=True),
            "runtime_config": config.to_dict(public=True) if config else None,
            "idempotent_replay": not created,
        }

    def _reprobe(
        self,
        profile: EnvironmentProfile,
        *,
        runtime_directory: str | Path | None = None,
        verified_weight_records: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[EnvironmentReport, Mapping[str, Any]]:
        detector = self.environment_manager.detector
        detector_applied_weight_records = False
        if runtime_directory is not None:
            detect_for_runtime = getattr(detector, "detect_for_runtime", None)
            if callable(detect_for_runtime):
                try:
                    parameters = inspect.signature(detect_for_runtime).parameters
                except (TypeError, ValueError):
                    parameters = {}
                accepts_records = "verified_weight_records" in parameters or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters.values()
                )
                detector_applied_weight_records = accepts_records
                report = (
                    detect_for_runtime(
                        profile,
                        runtime_directory,
                        verified_weight_records=verified_weight_records,
                    )
                    if accepts_records
                    else detect_for_runtime(profile, runtime_directory)
                )
            else:
                report = detector.detect(profile)
        else:
            report = detector.detect(profile)
        if not isinstance(report, EnvironmentReport):
            raise InstallationIntegrityError("安装后重新探测没有返回有效报告")
        if report.connection_digest != profile.connection_digest:
            raise InstallationIntegrityError("安装后重新探测未绑定当前连接")
        if verified_weight_records and not detector_applied_weight_records:
            report = EnvironmentReport.from_probe(
                profile,
                report.to_dict(include_digest=False),
                verified_weight_records=verified_weight_records,
            )
        match = match_environment(profile, report)
        return report, match

    def _verified_weight_records(
        self,
        profile: EnvironmentProfile,
        plan: InstallPlan,
        runtime_directory: str | Path,
        result: Mapping[str, Any],
        *,
        expected_records: Mapping[str, Mapping[str, Any]] | None = None,
        entries: Sequence[InstallManifestEntry] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Build trusted weight evidence for the target-aware re-probe.

        Local files are hashed from the staged/enabled directory itself.  SSH
        returns the same evidence from the fixed remote helper after it has
        verified and copied each required file.  A missing or malformed
        record is intentionally left absent so the subsequent match remains
        ``PLAN_REQUIRED`` instead of being promoted by a plan entry alone.
        """

        records: dict[str, dict[str, Any]] = {}
        weight_entries = [
            entry
            for entry in (entries if entries is not None else plan.entries)
            if entry.component_id == "unimol-weights"
        ]
        if profile.mode == "local":
            base = Path(runtime_directory).absolute().resolve()
            for entry in weight_entries:
                relative_paths = entry.required_paths or (
                    (entry.install_filename,) if entry.install_kind == "file" else ()
                )
                for relative in relative_paths:
                    candidate = (
                        base / entry.install_subdirectory / relative
                    ).resolve()
                    if not candidate.is_relative_to(base):
                        raise InstallationIntegrityError(
                            "权重验证路径不属于目标运行目录"
                        )
                    if not candidate.is_file() or candidate.is_symlink():
                        continue
                    size, digest = _file_digest(candidate)
                    records[str(candidate)] = {
                        "size_bytes": size,
                        "sha256": digest,
                    }
        else:
            raw_files = result.get("verified_files", ()) if isinstance(result, Mapping) else ()
            if isinstance(raw_files, Sequence) and not isinstance(
                raw_files, (str, bytes, bytearray)
            ):
                for raw in raw_files:
                    if not isinstance(raw, Mapping) or raw.get("component_id") != "unimol-weights":
                        continue
                    path = raw.get("path")
                    if not isinstance(path, str) or not path:
                        continue
                    try:
                        digest = validate_sha256(raw.get("sha256"), field="verified weight digest")
                        size = _bounded_int(
                            raw.get("size_bytes"),
                            field="verified weight size",
                            minimum=1,
                            maximum=MAX_EXTRACTED_FILE_BYTES,
                        )
                    except InstallationError:
                        continue
                    records[path] = {"size_bytes": size, "sha256": digest}

        if expected_records:
            expected_fingerprint = sorted(
                (
                    validate_sha256(item.get("sha256"), field="expected weight digest"),
                    _bounded_int(
                        item.get("size_bytes"),
                        field="expected weight size",
                        minimum=1,
                        maximum=MAX_EXTRACTED_FILE_BYTES,
                    ),
                )
                for item in expected_records.values()
                if isinstance(item, Mapping)
            )
            actual_fingerprint = sorted(
                (item["sha256"], int(item["size_bytes"]))
                for item in records.values()
            )
            if not expected_fingerprint or actual_fingerprint != expected_fingerprint:
                raise InstallationIntegrityError(
                    "最终权重证据与 staging 摘要不一致"
                )

        for entry in weight_entries:
            if entry.install_kind != "file":
                continue
            expected = (entry.sha256, entry.estimated_download_bytes)
            actual = sorted(
                (item["sha256"], int(item["size_bytes"]))
                for item in records.values()
            )
            if actual and actual != [expected]:
                raise InstallationIntegrityError(
                    "权重证据与固定 manifest 摘要或大小不一致"
                )
        return records

    def _persist_reprobe(
        self,
        profile: EnvironmentProfile,
        report: EnvironmentReport,
        match: Mapping[str, Any],
        *,
        expected_report_digest: str | None = None,
    ) -> None:
        self.environment_manager.store.save_detection(
            profile.environment_ref,
            {
                "environment": profile.to_public_dict(),
                "report": report.to_dict(),
                "match": thaw_json(match),
                "read_only": True,
                "installation_enabled": False,
                "installation_available": True,
                "probe_names": list(report.probes),
            },
            expected_connection_digest=profile.connection_digest,
            expected_report_digest=expected_report_digest,
        )

    def _confirm_existing_environment(
        self,
        plan: InstallPlan,
        *,
        resume: bool = False,
    ) -> dict[str, Any]:
        profile, original_report, _ = self._current_binding(plan.environment_ref)
        pending_record = self.store.get_installation_for_plan(plan.plan_id)
        persisted_config = self.store.get_runtime_config(plan.environment_ref)
        if (
            pending_record is not None
            and persisted_config is not None
            and persisted_config.state == "CONFIRMED"
            and persisted_config.runtime_id == plan.runtime_id
            and persisted_config.installation_id == pending_record.installation_id
            and persisted_config.plan_digest == plan.plan_digest
            and persisted_config.connection_digest == plan.connection_digest
            and persisted_config.catalog_digest == plan.catalog_digest
            and persisted_config.report_digest == original_report.report_digest
        ):
            # A crash after the config fsync but before the installation state
            # update must replay the same durable config, not create a new
            # verified_at/config_digest pair.
            current = pending_record
            if current.state != "CONFIRMED":
                current = self._update(
                    current,
                    state="CONFIRMED",
                    side_effects_started=False,
                    worker_pid=0,
                )
            return {
                "installation": current.to_dict(public=True),
                "runtime_config": persisted_config.to_dict(public=True),
                "idempotent_replay": True,
            }
        existing_runtime = self.runtime_for_environment(plan.environment_ref)
        if existing_runtime is not None and existing_runtime.state == "CONFIRMED" and existing_runtime.runtime_id != plan.runtime_id:
            return {
                "installation": None,
                "runtime_config": existing_runtime.to_dict(public=True),
                "idempotent_replay": True,
            }
        # An installed runtime can be invalidated by a later report/catalog
        # check and then legitimately re-confirmed.  In that case the old
        # target is the candidate being confirmed; probing the ordinary
        # connection would inspect the host's default environment instead and
        # could lose the isolated runtime's component evidence.  Keep the
        # target and its verified weight evidence only when the connection and
        # fixed catalog still match.  Missing or stale evidence deliberately
        # remains empty so the target-aware probe can only produce
        # ``PLAN_REQUIRED``, never a false confirmation.
        previous_runtime_directory: str | Path | None = None
        previous_weight_records: dict[str, dict[str, Any]] = {}
        persisted_config = self.store.get_runtime_config(plan.environment_ref)
        if (
            persisted_config is not None
            and persisted_config.state == "INVALIDATED"
            and persisted_config.connection_digest == profile.connection_digest
            and persisted_config.target_directory
        ):
            previous_runtime_directory = persisted_config.target_directory
            try:
                previous_record = self.store.get_installation(
                    persisted_config.installation_id
                )
                previous_plan = self.store.get_plan(previous_record.plan_id)
                previous_expected = (
                    previous_record.verification.get("verified_weight_records")
                    if isinstance(previous_record.verification, Mapping)
                    else None
                )
                expected_records = (
                    thaw_json(previous_expected)
                    if isinstance(previous_expected, Mapping)
                    else None
                )
                previous_result = (
                    previous_record.verification.get("install_result", {})
                    if isinstance(previous_record.verification, Mapping)
                    else {}
                )
                if profile.mode == "ssh":
                    previous_result = self.executor.verify(
                        profile,
                        previous_plan,
                        previous_record.stage_directory,
                        previous_result,
                        transaction_id=previous_record.installation_id,
                    )
                previous_weight_records = self._verified_weight_records(
                    profile,
                    previous_plan,
                    previous_runtime_directory,
                    previous_result,
                    expected_records=expected_records,
                    entries=self.manifest.entries,
                )
            except InstallationError:
                previous_weight_records = {}
        record, created = self.store.claim_approval(plan)
        if record.state == "CONFIRMED":
            config = self.store.get_runtime_config(plan.environment_ref)
            return {
                "installation": record.to_dict(public=True),
                "runtime_config": config.to_dict(public=True) if config else None,
                "idempotent_replay": True,
            }
        if record.state not in {"APPROVED", "VERIFYING"}:
            return {
                "installation": record.to_dict(public=True),
                "runtime_config": None,
                "idempotent_replay": True,
            }
        if not created and not resume:
            return {
                "installation": record.to_dict(public=True),
                "runtime_config": None,
                "idempotent_replay": True,
            }
        try:
            if (
                previous_runtime_directory is not None
                and record.target_directory != str(previous_runtime_directory)
            ):
                record = self._update(
                    record,
                    target_directory=str(previous_runtime_directory),
                )
            # The original report is still the approval binding.  The new
            # report is the runtime's post-confirmation binding.
            current_profile, current_report, current_match = self._current_binding(plan.environment_ref)
            allowed_report_digests = {original_report.report_digest}
            persisted_reprobe = record.verification.get("reprobe") if isinstance(record.verification, Mapping) else None
            if isinstance(persisted_reprobe, Mapping) and isinstance(persisted_reprobe.get("report_digest"), str):
                allowed_report_digests.add(persisted_reprobe["report_digest"])
            if current_profile.connection_digest != profile.connection_digest or current_report.report_digest not in allowed_report_digests:
                raise InstallationConflictError("连接或探测报告在确认期间发生变化")
            persisted_reprobe = record.verification.get("reprobe") if isinstance(record.verification, Mapping) else None
            if (
                isinstance(persisted_reprobe, Mapping)
                and persisted_reprobe.get("report_digest") == current_report.report_digest
            ):
                reprobe, reprobe_match = current_report, current_match
            elif previous_runtime_directory is not None:
                reprobe, reprobe_match = self._reprobe(
                    profile,
                    runtime_directory=previous_runtime_directory,
                    verified_weight_records=previous_weight_records,
                )
            else:
                reprobe, reprobe_match = self._reprobe(profile)
            runtime_components = _runtime_components_from_match(reprobe_match)
            verification = {
                "reprobe": {
                    "report_digest": reprobe.report_digest,
                    "detected_at": reprobe.detected_at,
                    "match_status": reprobe_match.get("status"),
                    "components": {
                        str(item["component_id"]): True
                        for item in runtime_components
                        if item.get("component_id") in _REQUIRED_RUNTIME_COMPONENT_IDS
                    },
                }
            }
            current = self._update(
                record,
                state="VERIFYING",
                verification=verification,
                side_effects_started=False,
                worker_pid=os.getpid(),
            )
            self._persist_reprobe(
                profile,
                reprobe,
                reprobe_match,
                expected_report_digest=current_report.report_digest,
            )
            config_record = replace(current, report_digest=reprobe.report_digest)
            config = RuntimeConfig.confirmed(
                record=config_record,
                components=runtime_components,
                verified_at=utc_timestamp(),
            )
            self.store.save_runtime_config(config)
            current = self.store.get_installation(record.installation_id)
            current = self._update(
                current,
                state="CONFIRMED",
                verification=verification,
                side_effects_started=False,
                worker_pid=0,
            )
            return {
                "installation": current.to_dict(public=True),
                "runtime_config": config.to_dict(public=True),
                "idempotent_replay": not created,
            }
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            committed = self._reconcile_committed_existing_environment(
                record,
                plan,
                profile,
            )
            if committed is not None:
                config = self.store.get_runtime_config(plan.environment_ref)
                if config is not None:
                    return {
                        "installation": committed.to_dict(public=True),
                        "runtime_config": config.to_dict(public=True),
                        "idempotent_replay": True,
                    }
            failed = self._fail_record(
                record,
                plan,
                exc,
                profile=profile,
                stage_directory="",
                finalized=False,
            )
            return {
                "installation": failed.to_dict(public=True),
                "runtime_config": None,
                "idempotent_replay": False,
            }

    def recover(self, installation_id: str, *, force: bool = False) -> dict[str, Any]:
        record = self.store.get_installation(installation_id)
        plan = self.store.get_plan(record.plan_id)
        if record.state == "ROLLING_BACK":
            return self._recover_rollback(record, plan)
        if plan.status == "READY_TO_CONFIRM":
            if not force and record.state == "VERIFYING" and _process_alive(record.worker_pid):
                raise InstallationConflictError("installation worker is still active")
            return self._confirm_existing_environment(plan, resume=True)
        if record.state in {"ENABLING", "RECOVERING"}:
            claimed, should_recover = self.store.claim_recovery(
                record.installation_id,
                force=force,
            )
            if not should_recover:
                config = self.store.get_runtime_config(plan.environment_ref)
                return {
                    "installation": claimed.to_dict(public=True),
                    "runtime_config": config.to_dict(public=True) if config else None,
                }
            try:
                return self._recover_enabling(claimed, plan)
            finally:
                self.store.release_recovery(record.installation_id)
        record = self.store.recover_stale(
            installation_id,
            stale_after_seconds=0,
            force=force,
        )
        if record.state != "APPROVED":
            return {"installation": record.to_dict(public=True), "runtime_config": None}
        profile, report, _ = self._current_binding(plan.environment_ref)
        if report.report_digest != record.report_digest or profile.connection_digest != record.connection_digest:
            raise InstallationConflictError("恢复前连接或探测报告已变化")
        claimed, should_execute = self.store.claim_execution(record.installation_id)
        if not should_execute:
            return {"installation": claimed.to_dict(public=True), "runtime_config": None}
        completed = self._execute(claimed, plan)
        config = self.store.get_runtime_config(plan.environment_ref)
        return {
            "installation": completed.to_dict(public=True),
            "runtime_config": config.to_dict(public=True) if config else None,
        }

    def _transaction_reprobe_digests(self, record: InstallationRecord) -> set[str]:
        if not isinstance(record.verification, Mapping):
            return set()
        digests: set[str] = set()
        for key in ("reprobe", "final_reprobe"):
            value = record.verification.get(key)
            digest = value.get("report_digest") if isinstance(value, Mapping) else None
            if isinstance(digest, str):
                digests.add(digest)
        return digests

    def _failed_report_is_cleared(
        self,
        plan: InstallPlan,
        record: InstallationRecord,
    ) -> bool:
        expected_digests = self._transaction_reprobe_digests(record)
        if not expected_digests:
            return True
        try:
            detection = self.environment_manager.store.get_detection(
                plan.environment_ref
            )
        except Exception:
            return False
        if detection is None:
            return True
        report = detection.get("report") if isinstance(detection, Mapping) else None
        if not isinstance(report, Mapping):
            return False
        current_digest = report.get("report_digest")
        if not isinstance(current_digest, str) or current_digest not in expected_digests:
            # A newer detection belongs to the connection's current state and
            # must not be removed as part of this old rollback.
            return True
        try:
            cleared = self.environment_manager.store.clear_detection(
                plan.environment_ref,
                expected_report_digest=current_digest,
            )
        except Exception:
            return False
        if cleared:
            return True
        try:
            current = self.environment_manager.store.get_detection(
                plan.environment_ref
            )
        except Exception:
            return False
        if current is None:
            return True
        current_report = current.get("report") if isinstance(current, Mapping) else None
        return not isinstance(current_report, Mapping) or current_report.get("report_digest") not in expected_digests

    def _enter_rollback(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
        exc: BaseException,
        *,
        finalized: bool,
    ) -> InstallationRecord:
        current = self.store.get_installation(record.installation_id)
        if current.state in {"CONFIRMED", "ROLLING_BACK"}:
            return current
        verification = thaw_json(current.verification)
        verification["rollback_finalized"] = bool(finalized)
        return self._update(
            current,
            state="ROLLING_BACK",
            verification=verification,
            error=_safe_install_error(exc),
            rollback_completed=False,
            worker_pid=0,
            side_effects_started=current.side_effects_started,
        )

    def _complete_rollback(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
        *,
        profile: EnvironmentProfile | None,
    ) -> InstallationRecord:
        current = self.store.get_installation(record.installation_id)
        if current.state == "CONFIRMED":
            return current
        config = self.store.get_runtime_config(plan.environment_ref)
        if config is not None and config.state == "CONFIRMED":
            try:
                self.store.mark_runtime_invalidated(config.runtime_id)
            except InstallationError:
                return current
        if not self._failed_report_is_cleared(plan, current):
            return current
        finalized = bool(
            isinstance(current.verification, Mapping)
            and current.verification.get("rollback_finalized")
        )
        if current.stage_directory and profile is None:
            return current
        if profile is not None and current.stage_directory:
            rollback_result = self.executor.rollback(
                profile,
                plan,
                current.stage_directory,
                finalized=finalized,
                transaction_id=current.installation_id,
            )
            if rollback_result is False:
                return current
        latest = self.store.get_installation(current.installation_id)
        return self._update(
            latest,
            state="FAILED",
            rollback_completed=True,
            worker_pid=0,
            side_effects_started=latest.side_effects_started,
        )

    def _rollback_result(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
        *,
        profile: EnvironmentProfile | None,
    ) -> dict[str, Any]:
        try:
            current = self._complete_rollback(record, plan, profile=profile)
        except Exception:
            current = self.store.get_installation(record.installation_id)
        config = self.store.get_runtime_config(plan.environment_ref)
        return {
            "installation": current.to_dict(public=True),
            "runtime_config": config.to_dict(public=True) if config else None,
        }

    def _recover_rollback(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
    ) -> dict[str, Any]:
        profile = self.environment_manager.store.get_profile(plan.environment_ref)
        return self._rollback_result(record, plan, profile=profile)

    def _recover_enabling(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
    ) -> dict[str, Any]:
        try:
            return self._recover_enabling_checked(record, plan)
        except Exception as exc:
            try:
                profile: EnvironmentProfile | None = self.environment_manager.store.get_profile(
                    plan.environment_ref
                )
            except InstallationError:
                profile = None
            rolling_back = self._enter_rollback(
                record,
                plan,
                exc,
                finalized=True,
            )
            return self._rollback_result(rolling_back, plan, profile=profile)

    def _recover_enabling_checked(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
    ) -> dict[str, Any]:
        """Finish the tiny window between atomic enable and config persistence."""

        profile, report, _ = self._current_binding(plan.environment_ref)
        allowed_report_digests = {record.report_digest}
        if isinstance(record.verification, Mapping):
            for key in ("reprobe", "final_reprobe"):
                persisted_reprobe = record.verification.get(key)
                if isinstance(persisted_reprobe, Mapping) and isinstance(
                    persisted_reprobe.get("report_digest"), str
                ):
                    allowed_report_digests.add(persisted_reprobe["report_digest"])
        if report.report_digest not in allowed_report_digests or profile.connection_digest != record.connection_digest:
            raise InstallationConflictError("恢复前连接或探测报告已变化")
        existing = self.store.get_runtime_config(plan.environment_ref)
        if existing is not None and (
            existing.runtime_id != plan.runtime_id
            or existing.plan_digest != plan.plan_digest
            or existing.catalog_digest != plan.catalog_digest
        ):
            raise InstallationConflictError("运行配置已绑定到其他安装计划")
        existing_confirmed = existing is not None and existing.state == "CONFIRMED" and existing.report_digest == report.report_digest
        expected_weight_records: Mapping[str, Mapping[str, Any]] | None = None
        if isinstance(record.verification, Mapping):
            persisted_weight_records = record.verification.get("verified_weight_records")
            if isinstance(persisted_weight_records, Mapping):
                expected_weight_records = thaw_json(persisted_weight_records)
        if profile.mode == "local":
            target = (self.root / "runtimes" / plan.runtime_id).absolute()
            runtime_directory: str | Path = str(target)
            if not target.is_dir() or target.is_symlink():
                # The worker may have crashed after recording ENABLING but
                # before os.replace.  Put the transaction back into the
                # normal resumable path; never treat a missing target as
                # successfully enabled.
                if existing_confirmed and existing is not None:
                    self.store.mark_runtime_invalidated(existing.runtime_id)
                reset = self._update(record, state="APPROVED", error="")
                claimed, should_execute = self.store.claim_execution(reset.installation_id)
                if not should_execute:
                    return {"installation": claimed.to_dict(public=True), "runtime_config": None}
                completed = self._execute(claimed, plan)
                config = self.store.get_runtime_config(plan.environment_ref)
                return {
                    "installation": completed.to_dict(public=True),
                    "runtime_config": config.to_dict(public=True) if config else None,
                }
            self.executor.verify(
                profile,
                plan,
                str(target),
                (record.verification.get("install_result", {}) if isinstance(record.verification, Mapping) else {}),
                transaction_id=record.installation_id,
            )
            final_result: Mapping[str, Any] = (
                record.verification.get("install_result", {})
                if isinstance(record.verification, Mapping)
                else {}
            )
        else:
            runtime_directory = plan.target_directory
            remote_verification = self.executor.verify(
                profile,
                plan,
                record.stage_directory,
                (record.verification.get("install_result", {}) if isinstance(record.verification, Mapping) else {}),
                transaction_id=record.installation_id,
            )
            final_result = remote_verification
            if remote_verification.get("remote_state") != "ENABLED":
                self.executor.finalize(
                    profile,
                    plan,
                    record.stage_directory,
                    transaction_id=record.installation_id,
                )
                remote_verification = self.executor.verify(
                    profile,
                    plan,
                    record.stage_directory,
                    (record.verification.get("install_result", {}) if isinstance(record.verification, Mapping) else {}),
                    transaction_id=record.installation_id,
                )
                final_result = remote_verification
            if remote_verification.get("remote_state") != "ENABLED" or not remote_verification.get("target_exists"):
                raise InstallationIntegrityError("恢复后远端 runtime 未确认启用")
        try:
            weight_records = self._verified_weight_records(
                profile,
                plan,
                runtime_directory,
                final_result,
                expected_records=expected_weight_records,
            )
        except InstallationError as exc:
            if not existing_confirmed or existing is None:
                raise
            self.store.mark_runtime_invalidated(existing.runtime_id)
            rolling_back = self._enter_rollback(
                record,
                plan,
                exc,
                finalized=True,
            )
            return self._rollback_result(rolling_back, plan, profile=profile)
        if existing_confirmed:
            current = self.store.get_installation(record.installation_id)
            if current.state != "CONFIRMED":
                current = self._update(
                    current,
                    state="CONFIRMED",
                    verification=current.verification,
                    side_effects_started=True,
                    worker_pid=0,
                )
            return {
                "installation": current.to_dict(public=True),
                "runtime_config": existing.to_dict(public=True),
            }
        runtime_directory: str | Path = (
            str(target) if profile.mode == "local" else plan.target_directory
        )
        reprobe, reprobe_match = self._reprobe(
            profile,
            runtime_directory=runtime_directory,
            verified_weight_records=weight_records,
        )
        runtime_components = _runtime_components_from_match(reprobe_match)
        verification = {
            **thaw_json(record.verification),
            "final_reprobe": {
                "report_digest": reprobe.report_digest,
                "detected_at": reprobe.detected_at,
                "match_status": reprobe_match.get("status"),
                "selected_device": reprobe_match.get("selected_device"),
            },
            "runtime_components": thaw_json(list(runtime_components)),
            "verified_weight_records": thaw_json(weight_records),
        }
        current = self._update(record, verification=verification)
        self._persist_reprobe(
            profile,
            reprobe,
            reprobe_match,
            expected_report_digest=report.report_digest,
        )
        if existing is None:
            existing = RuntimeConfig.confirmed(
                record=current,
                components=runtime_components,
                verified_at=utc_timestamp(),
                report_digest=reprobe.report_digest,
            )
            self.store.save_runtime_config(existing)
        current = self.store.get_installation(record.installation_id)
        if current.state != "CONFIRMED":
            current = self._update(
                current,
                state="CONFIRMED",
                verification=current.verification,
                side_effects_started=True,
                worker_pid=0,
            )
        return {
            "installation": current.to_dict(public=True),
            "runtime_config": existing.to_dict(public=True),
        }

    def _execute(self, record: InstallationRecord, plan: InstallPlan) -> InstallationRecord:
        profile: EnvironmentProfile | None = None
        stage_directory = ""
        target_was_absent = False
        finalized = False
        finalize_attempted = False
        try:
            profile, _report, _match = self._current_binding(plan.environment_ref)
            stage_directory = record.stage_directory or _runtime_stage(profile, plan.runtime_id, record.installation_id)
            target_directory = plan.target_directory
            if profile.mode == "local":
                expected_stage = (self.root / ".runtime-staging" / f"{plan.runtime_id}-{record.installation_id}").absolute()
                if stage_directory.startswith(".molly/"):
                    stage_path = expected_stage
                else:
                    stage_path = Path(stage_directory).absolute()
                if stage_path != expected_stage:
                    raise InstallationIntegrityError("持久化安装阶段目录不属于当前运行配置")
                # The public plan uses the stable .molly path.  The persisted local
                # path is still derived server-side and never accepted from JSON.
                stage_directory = str(stage_path)
                _owned_directory(stage_path.parent, create=True)
                target = (self.root / "runtimes" / plan.runtime_id).absolute()
                target_directory = str(target)
                target_was_absent = not target.exists() and not target.is_symlink()
            current = self._update(
                record,
                stage_directory=stage_directory,
                target_directory=target_directory,
                side_effects_started=True,
            )
            result = self.executor.install(
                profile,
                plan,
                stage_directory,
                completed_component_ids=current.completed_component_ids,
                progress=lambda component_id: self._mark_component(current, component_id),
                timeout_seconds=self.timeout_seconds,
                transaction_id=record.installation_id,
            )
            # The callback above updates the durable record.  Reload it so the
            # following CAS uses the current revision after every component.
            current = self.store.get_installation(record.installation_id)
            current = self._update(
                current,
                state="VERIFYING",
                verification={"install_result": thaw_json(result)},
            )
            verified = self.executor.verify(
                profile,
                plan,
                stage_directory,
                result,
                transaction_id=record.installation_id,
            )
            _, report_before_reprobe, _ = self._current_binding(plan.environment_ref)
            allowed_report_digests = {plan.report_digest}
            persisted_reprobe = current.verification.get("reprobe") if isinstance(current.verification, Mapping) else None
            if isinstance(persisted_reprobe, Mapping) and isinstance(persisted_reprobe.get("report_digest"), str):
                allowed_report_digests.add(persisted_reprobe["report_digest"])
            if report_before_reprobe.report_digest not in allowed_report_digests:
                raise InstallationConflictError("探测报告在安装期间发生变化，已取消启用")
            stage_weight_records = self._verified_weight_records(
                profile,
                plan,
                stage_directory,
                result,
            )
            reprobe, reprobe_match = self._reprobe(
                profile,
                runtime_directory=stage_directory,
                verified_weight_records=stage_weight_records,
            )
            runtime_components = _runtime_components_from_match(reprobe_match)
            verified_components = {
                str(item["component_id"]): True
                for item in runtime_components
                if item.get("component_id") in _REQUIRED_RUNTIME_COMPONENT_IDS
            }
            verification = {
                **thaw_json(current.verification),
                "staged": thaw_json(verified),
                "reprobe": {
                    "report_digest": reprobe.report_digest,
                    "detected_at": reprobe.detected_at,
                    "match_status": reprobe_match.get("status"),
                    "selected_device": reprobe_match.get("selected_device"),
                    "components": verified_components,
                },
                "runtime_components": thaw_json(list(runtime_components)),
                "verified_weight_records": thaw_json(stage_weight_records),
            }
            current = self._update(current, verification=verification)
            # Re-check all approval bindings immediately before the atomic
            # enable operation.  A changed report must never enable a stale
            # runtime, even if the installation itself succeeded.
            _, latest_report, _ = self._current_binding(plan.environment_ref)
            if latest_report.report_digest not in allowed_report_digests:
                raise InstallationConflictError("探测报告在安装期间变化，已取消启用")
            current = self._update(current, state="ENABLING", verification=verification)
            finalize_attempted = True
            self.executor.finalize(
                profile,
                plan,
                stage_directory,
                transaction_id=record.installation_id,
            )
            finalized = True
            final_runtime_directory: str | Path = (
                str(target) if profile.mode == "local" else plan.target_directory
            )
            final_result: Mapping[str, Any] = result
            if profile.mode == "ssh":
                final_result = self.executor.verify(
                    profile,
                    plan,
                    stage_directory,
                    result,
                    transaction_id=record.installation_id,
                )
                if final_result.get("remote_state") != "ENABLED" or not final_result.get("target_exists"):
                    raise InstallationIntegrityError("远端 runtime 原子启用后未通过状态验证")
            _, final_binding_report, _ = self._current_binding(plan.environment_ref)
            if final_binding_report.report_digest != latest_report.report_digest:
                raise InstallationConflictError("连接或探测报告在启用期间发生变化，已取消启用")
            final_weight_records = self._verified_weight_records(
                profile,
                plan,
                final_runtime_directory,
                final_result,
                expected_records=stage_weight_records,
            )
            final_reprobe, final_match = self._reprobe(
                profile,
                runtime_directory=final_runtime_directory,
                verified_weight_records=final_weight_records,
            )
            final_runtime_components = _runtime_components_from_match(final_match)
            verification = {
                **verification,
                "final_reprobe": {
                    "report_digest": final_reprobe.report_digest,
                    "detected_at": final_reprobe.detected_at,
                    "match_status": final_match.get("status"),
                    "selected_device": final_match.get("selected_device"),
                },
                "runtime_components": thaw_json(list(final_runtime_components)),
                "verified_weight_records": thaw_json(final_weight_records),
            }
            current = self._update(current, verification=verification)
            self._persist_reprobe(
                profile,
                final_reprobe,
                final_match,
                expected_report_digest=final_binding_report.report_digest,
            )
            config = RuntimeConfig.confirmed(
                record=current,
                components=final_runtime_components,
                verified_at=utc_timestamp(),
                report_digest=final_reprobe.report_digest,
            )
            self.store.save_runtime_config(config)
            current = self.store.get_installation(record.installation_id)
            current = self._update(
                current,
                state="CONFIRMED",
                verification=verification,
                side_effects_started=True,
                worker_pid=0,
            )
            return current
        except BaseException as exc:
            # BaseException intentionally includes a process-style test crash
            # only after the last durable update.  We do not try to mutate the
            # state on KeyboardInterrupt/SystemExit; the recovery API can
            # resume the APPROVED/INSTALLING/ENABLING transaction from its stage.
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if profile is not None:
                committed = self._reconcile_committed_runtime(
                    record,
                    plan,
                    profile,
                    stage_directory,
                )
                if committed is not None:
                    return committed
            return self._fail_record(
                record,
                plan,
                exc,
                profile=profile,
                stage_directory=stage_directory,
                finalized=finalized or (
                    profile is not None
                    and profile.mode == "ssh"
                    and finalize_attempted
                ) or (
                    profile is not None
                    and profile.mode == "local"
                    and target_was_absent
                    and (self.root / "runtimes" / plan.runtime_id).is_dir()
                    and not (self.root / "runtimes" / plan.runtime_id).is_symlink()
                    and not Path(stage_directory).exists()
                ),
            )

    def _mark_component(self, record: InstallationRecord, component_id: str) -> None:
        current = self.store.get_installation(record.installation_id)
        if component_id in current.completed_component_ids:
            return
        self._update(
            current,
            completed_component_ids=tuple((*current.completed_component_ids, component_id)),
            side_effects_started=True,
        )

    def _update(self, record: InstallationRecord, **changes: Any) -> InstallationRecord:
        updated = replace(record, **changes, revision=record.revision + 1, updated_at=utc_timestamp())
        return self.store.update_installation(updated, expected_revision=record.revision)

    def _reconcile_committed_runtime(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
        profile: EnvironmentProfile,
        stage_directory: str,
    ) -> InstallationRecord | None:
        """Reconcile an ambiguous exception after RuntimeConfig was committed.

        ``save_runtime_config`` is an atomic write, but a filesystem error can
        be raised after the replacement reached disk.  Once a matching config
        exists, deleting the enabled target would create a confirmed pointer
        to nothing.  Keep the target and either complete the installation or
        leave it in ``ENABLING`` for the normal recovery endpoint.
        """

        try:
            config = self.store.get_runtime_config(plan.environment_ref)
        except InstallationError:
            return None
        if config is None or config.state != "CONFIRMED":
            return None
        if (
            config.runtime_id != plan.runtime_id
            or config.installation_id != record.installation_id
            or config.plan_digest != plan.plan_digest
            or config.connection_digest != plan.connection_digest
            or config.catalog_digest != plan.catalog_digest
        ):
            return None

        current = self.store.get_installation(record.installation_id)
        final_reprobe = (
            current.verification.get("final_reprobe")
            if isinstance(current.verification, Mapping)
            else None
        )
        if (
            not isinstance(final_reprobe, Mapping)
            or config.report_digest != final_reprobe.get("report_digest")
        ):
            return None
        expected_target = (
            str((self.root / "runtimes" / plan.runtime_id).absolute())
            if profile.mode == "local"
            else plan.target_directory
        )
        if config.target_directory != expected_target:
            return None
        expected_records: Mapping[str, Mapping[str, Any]] | None = None
        if isinstance(current.verification, Mapping):
            persisted_records = current.verification.get("verified_weight_records")
            if isinstance(persisted_records, Mapping):
                expected_records = thaw_json(persisted_records)

        def invalidate_config() -> None:
            try:
                self.store.mark_runtime_invalidated(config.runtime_id)
            except InstallationError:
                pass

        target_exists = False
        if profile.mode == "local":
            target = self.root / "runtimes" / plan.runtime_id
            target_exists = target.is_dir() and not target.is_symlink()
            if not target_exists:
                invalidate_config()
                return None
            try:
                self._verified_weight_records(
                    profile,
                    plan,
                    target,
                    (
                        current.verification.get("install_result", {})
                        if isinstance(current.verification, Mapping)
                        else {}
                    ),
                    expected_records=expected_records,
                )
            except InstallationError:
                invalidate_config()
                return None
        else:
            try:
                status = self.executor.verify(
                    profile,
                    plan,
                    stage_directory,
                    (
                        current.verification.get("install_result", {})
                        if isinstance(current.verification, Mapping)
                        else {}
                    ),
                    transaction_id=record.installation_id,
                )
                target_exists = (
                    status.get("remote_state") == "ENABLED"
                    and bool(status.get("target_exists"))
                )
                if target_exists:
                    self._verified_weight_records(
                        profile,
                        plan,
                        plan.target_directory,
                        status,
                        expected_records=expected_records,
                    )
            except Exception:
                target_exists = False

        if profile.mode == "ssh" and not target_exists:
            # A remote status failure is ambiguous: do not delete a target we
            # could not inspect, but never leave a public confirmed pointer
            # while its existence/evidence is unknown.  ENABLING remains
            # recoverable through the normal status/finalize path.
            invalidate_config()

        if target_exists:
            if current.state != "CONFIRMED":
                current = self._update(
                    current,
                    state="CONFIRMED",
                    side_effects_started=True,
                    worker_pid=0,
                )
            return current

        if current.state != "ENABLING":
            current = self._update(
                current,
                state="ENABLING",
                side_effects_started=True,
                worker_pid=0,
            )
        return current

    def _reconcile_committed_existing_environment(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
        profile: EnvironmentProfile,
    ) -> InstallationRecord | None:
        """Complete a compatible-environment confirmation after an ambiguous write.

        The compatible path has no isolated target to roll back.  If the
        RuntimeConfig replacement reached durable storage but the following
        filesystem operation reported an error, only a matching config and
        matching persisted re-probe may promote the installation to CONFIRMED.
        """

        try:
            config = self.store.get_runtime_config(plan.environment_ref)
            if config is None or config.state != "CONFIRMED":
                return None
            if (
                config.runtime_id != plan.runtime_id
                or config.installation_id != record.installation_id
                or config.plan_digest != plan.plan_digest
                or config.connection_digest != plan.connection_digest
                or config.catalog_digest != plan.catalog_digest
            ):
                return None
            current = self.store.get_installation(record.installation_id)
            reprobe = (
                current.verification.get("reprobe")
                if isinstance(current.verification, Mapping)
                else None
            )
            if (
                not isinstance(reprobe, Mapping)
                or config.report_digest != reprobe.get("report_digest")
                or config.target_directory != current.target_directory
            ):
                return None
            current_profile, current_report, _ = self._current_binding(
                plan.environment_ref
            )
            if (
                current_profile.connection_digest != profile.connection_digest
                or current_report.report_digest != config.report_digest
            ):
                return None
            if current.state != "CONFIRMED":
                current = self._update(
                    current,
                    state="CONFIRMED",
                    side_effects_started=False,
                    worker_pid=0,
                )
            return current
        except (InstallationConfigError, InstallationIntegrityError, InstallationConflictError):
            return None

    def _fail_record(
        self,
        record: InstallationRecord,
        plan: InstallPlan,
        exc: BaseException,
        *,
        profile: EnvironmentProfile | None,
        stage_directory: str,
        finalized: bool,
    ) -> InstallationRecord:
        try:
            rolling_back = self._enter_rollback(
                record,
                plan,
                exc,
                finalized=finalized,
            )
        except Exception as update_exc:
            raise InstallationExecutionError("安装失败且无法持久化失败状态") from update_exc
        try:
            return self._complete_rollback(
                rolling_back,
                plan,
                profile=profile,
            )
        except SystemExit:
            raise
        except Exception:
            try:
                return self.store.get_installation(record.installation_id)
            except Exception as update_exc:
                raise InstallationExecutionError("安装失败且无法持久化失败状态") from update_exc

    def runtime_for_environment(self, environment_ref: str) -> RuntimeConfig | None:
        config = self.store.get_runtime_config(environment_ref)
        if config is None:
            return None
        try:
            profile, report, _ = self._current_binding(environment_ref)
        except InstallationError:
            return self.store.mark_runtime_invalidated(config.runtime_id)
        local_target_missing = (
            profile.mode == "local"
            and config.target_directory
            == str((self.root / "runtimes" / config.runtime_id).absolute())
            and (
                not Path(config.target_directory).is_dir()
                or Path(config.target_directory).is_symlink()
            )
        )
        if (
            config.state != "CONFIRMED"
            or config.connection_digest != profile.connection_digest
            or config.report_digest != report.report_digest
            or config.catalog_version != self.manifest.catalog_version
            or config.catalog_digest != self.manifest.digest
            or local_target_missing
        ):
            return self.store.mark_runtime_invalidated(config.runtime_id)
        return config

    def runtime_public(self, environment_ref: str) -> dict[str, Any] | None:
        config = self.runtime_for_environment(environment_ref)
        return config.to_dict(public=True) if config else None

    def installation_public(self, environment_ref: str) -> dict[str, Any] | None:
        record = self.store.get_installation_for_environment(environment_ref)
        return record.to_dict(public=True) if record is not None else None


def _safe_install_error(exc: BaseException) -> str:
    if isinstance(exc, InstallationIntegrityError):
        return "固定来源或安装内容校验失败，已回滚隔离目录"
    if isinstance(exc, InstallationConflictError):
        return "安装期间连接、清单或运行目录发生变化，已取消并回滚"
    if isinstance(exc, InstallationExecutionError):
        return "安装工具执行失败，已回滚隔离目录"
    return "安装未完成，已回滚隔离目录"


__all__ = [
    "InstallExecutor",
    "InstallManifest",
    "InstallManifestEntry",
    "InstallPlan",
    "InstallationConfigError",
    "InstallationConflictError",
    "InstallationError",
    "InstallationExecutionError",
    "InstallationIntegrityError",
    "InstallationManager",
    "InstallationRecord",
    "MAX_PERSISTED_PLANS",
    "MAX_PERSISTED_RUNTIME_CONFIGS",
    "INSTALLATION_RECOVERY_STALE_SECONDS",
    "INSTALLATION_STATE_VERSION",
    "INSTALLATION_TIMEOUT_SECONDS",
    "MAX_INSTALL_DISK_BYTES",
    "MAX_INSTALL_DOWNLOAD_BYTES",
    "RUNTIME_MANIFEST_ENV",
    "RestrictedInstallExecutor",
    "RuntimeConfig",
    "RuntimeInstallationStore",
    "default_install_manifest",
]
