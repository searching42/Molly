from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import threading
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Literal, Mapping

from platformdirs import user_config_path
from pydantic import BaseModel, ConfigDict, field_validator

try:  # pragma: no cover - POSIX CI exercises the primary path.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


ENVIRONMENT_PROFILE_SCHEMA_VERSION = "molly_environment_profiles.v1"
LEGACY_TRANSPORT_PROFILE_SCHEMA_VERSION = "molly_legacy_transport_profiles.v1"
_MAX_PRIVATE_CONFIG_BYTES = 16 * 1024 * 1024
_SENSITIVE_KEYS = (
    "authorization",
    "bearer",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)


def _identifier(value: Any, *, field: str) -> str:
    clean = str(value or "").strip().lower()
    if (
        not clean
        or len(clean) > 96
        or not clean[0].isalnum()
        or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for char in clean)
    ):
        raise ValueError(f"{field} must be a lowercase safe identifier")
    return clean


def _remote_path(value: Any, *, field: str) -> str:
    clean = str(value or "").strip()
    path = PurePosixPath(clean)
    if (
        not clean
        or not path.is_absolute()
        or ".." in path.parts
        or any(ord(char) < 32 for char in clean)
    ):
        raise ValueError(f"{field} must be an absolute normalized POSIX path")
    return str(path)


def _reject_sensitive_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(marker in normalized for marker in _SENSITIVE_KEYS):
                raise ValueError("environment profiles must not contain secrets")
            _reject_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_keys(child)


class RuntimeEnvironmentProfile(BaseModel):
    """Private runtime paths bound to a logical connection profile."""

    model_config = ConfigDict(extra="forbid")

    environment_id: str
    connection_id: str
    repository_root: str
    python_path: str
    conda_environment: str = ""

    @field_validator("environment_id", "connection_id", mode="before")
    @classmethod
    def validate_identifier(cls, value: Any, info: Any) -> str:
        return _identifier(value, field=info.field_name)

    @field_validator("repository_root", "python_path", mode="before")
    @classmethod
    def validate_remote_path(cls, value: Any, info: Any) -> str:
        return _remote_path(value, field=info.field_name)

    @field_validator("conda_environment", mode="before")
    @classmethod
    def validate_conda_environment(cls, value: Any) -> str:
        clean = str(value or "").strip()
        if len(clean) > 128 or any(ord(char) < 32 for char in clean):
            raise ValueError("conda_environment contains unsafe text")
        return clean

    def digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class LegacyTransportCompatibilityProfile(BaseModel):
    """Private exact-replay metadata for a retired transport profile."""

    model_config = ConfigDict(extra="forbid")

    legacy_profile_id: str
    ssh_target: str
    expected_hostname: str
    repository_root: str
    python_path: str
    host_key_policy: Literal["strict_pinned_known_hosts"] = (
        "strict_pinned_known_hosts"
    )
    device_policy: str = ""
    process_policy: str = ""
    config_renderer: Literal["reinvent4_v1", "reinvent4_v2"]

    @field_validator("legacy_profile_id", mode="before")
    @classmethod
    def validate_legacy_profile_id(cls, value: Any) -> str:
        return _identifier(value, field="legacy_profile_id")

    @field_validator("ssh_target", "expected_hostname", mode="before")
    @classmethod
    def validate_endpoint_label(cls, value: Any, info: Any) -> str:
        clean = str(value or "").strip()
        if (
            not clean
            or len(clean) > 255
            or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                for char in clean
            )
        ):
            raise ValueError(f"{info.field_name} must be a safe endpoint label")
        return clean

    @field_validator("repository_root", "python_path", mode="before")
    @classmethod
    def validate_legacy_remote_path(cls, value: Any, info: Any) -> str:
        return _remote_path(value, field=info.field_name)

    @field_validator("device_policy", "process_policy", mode="before")
    @classmethod
    def validate_optional_policy(cls, value: Any, info: Any) -> str:
        clean = str(value or "").strip()
        if len(clean) > 96 or any(ord(char) < 32 for char in clean):
            raise ValueError(f"{info.field_name} contains unsafe text")
        return clean

    def historical_contract(self) -> dict[str, str]:
        contract = {
            "profile_id": self.legacy_profile_id,
            "ssh_target": self.ssh_target,
            "expected_hostname": self.expected_hostname,
            "repo": self.repository_root,
            "python": self.python_path,
            "host_key_policy": self.host_key_policy,
        }
        if self.device_policy:
            contract["device_policy"] = self.device_policy
        if self.process_policy:
            contract["process_policy"] = self.process_policy
        return contract


def _absolute_config_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    return Path(os.path.abspath(path))


def _open_private_directory(
    path: Path, *, create: bool, enforce_private_mode: bool = True
) -> int:
    if path == Path("/"):
        raise ValueError("Molly private config directory must not be filesystem root")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_flag is None:
        raise ValueError("Molly private config requires O_NOFOLLOW directory support")
    flags = os.O_RDONLY | directory_flag | no_follow
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        if enforce_private_mode:
            os.fchmod(descriptor, 0o700)
        return descriptor
    except (OSError, ValueError) as exc:
        os.close(descriptor)
        raise ValueError(
            "Molly private config path must contain only real directories"
        ) from exc


def _read_private_json(directory_fd: int, name: str) -> dict[str, Any] | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError("private config must be a regular non-symlink file") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > _MAX_PRIVATE_CONFIG_BYTES
        ):
            raise ValueError("private config must be a bounded regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_PRIVATE_CONFIG_BYTES:
                raise ValueError("private config exceeds the size limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            != (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            or named.st_dev != metadata.st_dev
            or named.st_ino != metadata.st_ino
            or not stat.S_ISREG(named.st_mode)
        ):
            raise ValueError("private config changed while being read")
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private config file is invalid") from exc
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise ValueError("private config file must contain an object")
    _reject_sensitive_keys(payload)
    return payload


def _write_private_json(directory_fd: int, name: str, payload: Mapping[str, Any]) -> None:
    _reject_sensitive_keys(payload)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    temporary = f".{name}.{secrets.token_hex(12)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600, dir_fd=directory_fd)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = -1
        os.replace(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        target_fd = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        try:
            os.fchmod(target_fd, 0o600)
        finally:
            os.close(target_fd)
        os.fsync(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_fd)
        except FileNotFoundError:
            pass


@contextmanager
def _private_process_lock(config_dir: Path, lock_name: str) -> Iterator[int]:
    directory_fd = _open_private_directory(config_dir, create=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        try:
            descriptor = os.open(lock_name, flags, 0o600, dir_fd=directory_fd)
        except OSError as exc:
            raise ValueError(
                "Molly private config lock must be a regular non-symlink file"
            ) from exc
        os.fchmod(descriptor, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield directory_fd
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
    finally:
        os.close(directory_fd)


class RuntimeEnvironmentStore:
    """User-level, permission-restricted environment profile storage."""

    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = environ if environ is not None else os.environ
        root = config_dir or env.get("MOLLY_CONFIG_DIR") or user_config_path(
            "Molly", appauthor=False
        )
        self.config_dir = _absolute_config_path(root)
        self.path = self.config_dir / "environments.json"
        self.lock_path = self.config_dir / ".environment_profiles.lock"
        self._lock = threading.RLock()

    def list_environments(self) -> list[RuntimeEnvironmentProfile]:
        with self._lock, self._process_lock() as directory_fd:
            return self._read_locked(directory_fd)

    def get_environment(self, environment_id: str) -> RuntimeEnvironmentProfile:
        clean = _identifier(environment_id, field="environment_id")
        for profile in self.list_environments():
            if profile.environment_id == clean:
                return profile
        raise ValueError(f"runtime environment profile not found: {clean}")

    def save_environment(
        self, profile: RuntimeEnvironmentProfile
    ) -> RuntimeEnvironmentProfile:
        validated = RuntimeEnvironmentProfile.model_validate(
            profile.model_dump(mode="json")
        )
        with self._lock, self._process_lock() as directory_fd:
            profiles = [
                item
                for item in self._read_locked(directory_fd)
                if item.environment_id != validated.environment_id
            ]
            profiles.append(validated)
            self._write_locked(directory_fd, profiles)
        return validated

    def delete_environment(self, environment_id: str) -> bool:
        clean = _identifier(environment_id, field="environment_id")
        with self._lock, self._process_lock() as directory_fd:
            profiles = self._read_locked(directory_fd)
            kept = [item for item in profiles if item.environment_id != clean]
            if len(kept) == len(profiles):
                return False
            self._write_locked(directory_fd, kept)
            return True

    def _read_locked(self, directory_fd: int) -> list[RuntimeEnvironmentProfile]:
        payload = _read_private_json(directory_fd, self.path.name)
        if payload is None:
            return []
        if payload.get("schema_version") != ENVIRONMENT_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported environment profile schema")
        raw = payload.get("environments")
        if not isinstance(raw, list):
            raise ValueError("environment profile roster must be a list")
        profiles = [RuntimeEnvironmentProfile.model_validate(item) for item in raw]
        identifiers = [item.environment_id for item in profiles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate environment profile ID")
        return sorted(profiles, key=lambda item: item.environment_id)

    def _write_locked(
        self, directory_fd: int, profiles: list[RuntimeEnvironmentProfile]
    ) -> None:
        payload = {
            "schema_version": ENVIRONMENT_PROFILE_SCHEMA_VERSION,
            "environments": [
                item.model_dump(mode="json")
                for item in sorted(profiles, key=lambda item: item.environment_id)
            ],
        }
        _write_private_json(directory_fd, self.path.name, payload)

    @contextmanager
    def _process_lock(self) -> Iterator[int]:
        with _private_process_lock(self.config_dir, self.lock_path.name) as directory_fd:
            yield directory_fd


class LegacyTransportCompatibilityStore:
    """Read-only private mapping used only to replay retired publications."""

    def __init__(
        self,
        *,
        config_dir: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = environ if environ is not None else os.environ
        root = config_dir or env.get("MOLLY_CONFIG_DIR") or user_config_path(
            "Molly", appauthor=False
        )
        self.config_dir = _absolute_config_path(root)
        self.path = self.config_dir / "legacy_transport_profiles.json"
        self.lock_path = self.config_dir / ".legacy_transport_profiles.lock"
        self._lock = threading.RLock()

    def get_profile(self, legacy_profile_id: str) -> LegacyTransportCompatibilityProfile:
        clean = _identifier(legacy_profile_id, field="legacy_profile_id")
        with self._lock, _private_process_lock(
            self.config_dir, self.lock_path.name
        ) as directory_fd:
            payload = _read_private_json(directory_fd, self.path.name)
        if payload is None:
            raise ValueError("legacy transport compatibility profile is unavailable")
        if payload.get("schema_version") != LEGACY_TRANSPORT_PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported legacy transport compatibility schema")
        raw = payload.get("profiles")
        if not isinstance(raw, list):
            raise ValueError("legacy transport compatibility roster must be a list")
        profiles = [
            LegacyTransportCompatibilityProfile.model_validate(item) for item in raw
        ]
        identifiers = [item.legacy_profile_id for item in profiles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate legacy transport compatibility profile ID")
        for profile in profiles:
            if profile.legacy_profile_id == clean:
                return profile
        raise ValueError("legacy transport compatibility profile is unavailable")


__all__ = [
    "ENVIRONMENT_PROFILE_SCHEMA_VERSION",
    "LEGACY_TRANSPORT_PROFILE_SCHEMA_VERSION",
    "LegacyTransportCompatibilityProfile",
    "LegacyTransportCompatibilityStore",
    "RuntimeEnvironmentProfile",
    "RuntimeEnvironmentStore",
]
