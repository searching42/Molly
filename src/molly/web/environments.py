"""Server-owned connection profiles and read-only runtime discovery.

This module deliberately stops at discovery.  It does not expose an arbitrary
command endpoint and never installs packages, downloads model weights, or
changes a local or remote environment.  The separate ``installations``
module consumes a digest-bound, server-owned plan only after explicit user
confirmation.  SSH here is only used as a transport for one fixed Python
probe script; the connection profile cannot provide a command, shell
fragment, private key, or download URL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import signal
import shutil
import sys
import tempfile
import threading
from textwrap import dedent
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None

from molly.core.ids import (
    canonical_json_bytes,
    normalize_timestamp,
    sha256_bytes,
    utc_timestamp,
    validate_identifier,
    validate_sha256,
)


ENVIRONMENT_CONFIG_VERSION = 1
ENVIRONMENT_REPORT_VERSION = 1
COMPATIBILITY_CATALOG_VERSION = "2026.09"
COMPATIBLE_WEIGHT_NAMES = frozenset({"unimolv1.pt"})
DEFAULT_SSH_PORT = 22
MAX_ENVIRONMENT_NAME_LENGTH = 80
MAX_SSH_TARGET_LENGTH = 255
MAX_SSH_USER_LENGTH = 64
MAX_ENVIRONMENT_PROFILES = 32
MAX_DETECTION_OUTPUT_BYTES = 512 * 1024
DETECTION_TIMEOUT_SECONDS = 30.0
READ_ONLY_PROBE_NAMES = (
    "detect_system",
    "detect_gpu",
    "list_python_environments",
    "probe_unimol",
    "probe_reinvent4",
    "check_disk",
    "probe_weights",
    "build_install_plan",
)


class EnvironmentConfigError(ValueError):
    """A connection profile or persisted discovery record is invalid."""


class EnvironmentDetectionError(RuntimeError):
    """A fixed read-only environment probe could not complete safely."""


def _text(value: Any, *, field: str, maximum: int = 256, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str) or len(value) > maximum:
        raise EnvironmentConfigError(f"{field} is invalid")
    if any(char in value for char in "\x00\r\n"):
        raise EnvironmentConfigError(f"{field} contains a control character")
    return value


def _display_name(value: Any, *, fallback: str) -> str:
    candidate = fallback if value is None else value
    result = _text(candidate, field="environment display name", maximum=MAX_ENVIRONMENT_NAME_LENGTH)
    if not result.strip():
        raise EnvironmentConfigError("environment display name is required")
    return result.strip()


def _port(value: Any) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65_535:
        raise EnvironmentConfigError("SSH port must be an integer between 1 and 65535")
    return value


def _ssh_target(value: Any) -> str:
    result = _text(value, field="SSH alias or host", maximum=MAX_SSH_TARGET_LENGTH).strip()
    if (
        not result
        or result.startswith("-")
        or any(char.isspace() for char in result)
        or "/" in result
        or "\\" in result
        or "@" in result
    ):
        raise EnvironmentConfigError("SSH alias or host is invalid")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:%+\[\]-]*", result):
        raise EnvironmentConfigError("SSH alias or host is invalid")
    return result


def _ssh_user(value: Any) -> str:
    result = _text(value, field="SSH user", maximum=MAX_SSH_USER_LENGTH).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}", result):
        raise EnvironmentConfigError("SSH user is invalid")
    return result


def _profile_ref(mode: str, target: str | None, user: str | None, port: int | None) -> str:
    digest = _connection_digest(mode, target, user, port)
    return validate_identifier(f"environment:{mode}-{digest[:32]}", field="environment_ref")


def _connection_digest(
    mode: str,
    target: str | None,
    user: str | None,
    port: int | None,
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "mode": mode,
                "ssh_target": target if mode == "ssh" else None,
                "ssh_user": user if mode == "ssh" else None,
                "ssh_port": port if mode == "ssh" else None,
            }
        )
    )


_STORE_THREAD_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class EnvironmentProfile:
    """A non-secret local or SSH connection profile."""

    environment_ref: str
    display_name: str
    mode: str
    ssh_target: str | None = None
    ssh_user: str | None = None
    ssh_port: int | None = None
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        try:
            validate_identifier(self.environment_ref, field="environment_ref")
        except Exception as exc:
            raise EnvironmentConfigError("environment_ref is invalid") from exc
        if self.mode not in {"local", "ssh"}:
            raise EnvironmentConfigError("environment mode must be local or ssh")
        object.__setattr__(
            self,
            "display_name",
            _display_name(self.display_name, fallback=self.environment_ref),
        )
        if self.mode == "local":
            if any(value is not None for value in (self.ssh_target, self.ssh_user, self.ssh_port)):
                raise EnvironmentConfigError("local profiles cannot contain SSH fields")
        else:
            object.__setattr__(self, "ssh_target", _ssh_target(self.ssh_target))
            object.__setattr__(self, "ssh_user", _ssh_user(self.ssh_user))
            object.__setattr__(self, "ssh_port", _port(self.ssh_port))
        if self.created_at:
            object.__setattr__(
                self,
                "created_at",
                _timestamp(self.created_at, field="environment created_at"),
            )
        if self.updated_at:
            object.__setattr__(
                self,
                "updated_at",
                _timestamp(self.updated_at, field="environment updated_at"),
            )

    @property
    def target_label(self) -> str:
        if self.mode == "local":
            return "本地"
        return f"{self.ssh_user}@{self.ssh_target}:{self.ssh_port}"

    @property
    def connection_digest(self) -> str:
        """Stable identity of the endpoint authority, excluding display metadata."""

        return _connection_digest(self.mode, self.ssh_target, self.ssh_user, self.ssh_port)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_ref": self.environment_ref,
            "display_name": self.display_name,
            "mode": self.mode,
            "ssh_target": self.ssh_target,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "connection_digest": self.connection_digest,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentProfile":
        if not isinstance(value, Mapping):
            raise EnvironmentConfigError("environment profile must be an object")
        try:
            profile = cls(
                environment_ref=value["environment_ref"],
                display_name=value["display_name"],
                mode=value["mode"],
                ssh_target=value.get("ssh_target"),
                ssh_user=value.get("ssh_user"),
                ssh_port=value.get("ssh_port"),
                created_at=value.get("created_at", ""),
                updated_at=value.get("updated_at", ""),
            )
            recorded_digest = value.get("connection_digest")
            if recorded_digest is not None and recorded_digest != profile.connection_digest:
                raise EnvironmentConfigError("environment profile connection digest is inconsistent")
            return profile
        except (KeyError, TypeError, ValueError) as exc:
            raise EnvironmentConfigError("environment profile is malformed") from exc

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        environment_ref: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> "EnvironmentProfile":
        if not isinstance(payload, Mapping):
            raise EnvironmentConfigError("environment profile must be an object")
        allowed = {
            "environment_ref",
            "display_name",
            "mode",
            "ssh_target",
            "ssh_alias",
            "host",
            "ssh_user",
            "user",
            "ssh_port",
            "port",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise EnvironmentConfigError("environment profile contains an unsupported field")
        mode = payload.get("mode", "local")
        if not isinstance(mode, str) or mode.casefold() not in {"local", "ssh"}:
            raise EnvironmentConfigError("environment mode must be local or ssh")
        mode = mode.casefold()
        target_value = payload.get("ssh_target", payload.get("ssh_alias", payload.get("host")))
        user_value = payload.get("ssh_user", payload.get("user"))
        port_value = payload.get("ssh_port", payload.get("port", DEFAULT_SSH_PORT))
        if mode == "local":
            target = user = None
            port = None
        else:
            target = _ssh_target(target_value)
            user = _ssh_user(user_value)
            port = _port(port_value)
        ref = environment_ref or payload.get("environment_ref")
        if ref is None or ref == "":
            ref = _profile_ref(mode, target, user, port)
        if not isinstance(ref, str):
            raise EnvironmentConfigError("environment_ref is invalid")
        now = updated_at or utc_timestamp()
        return cls(
            environment_ref=ref,
            display_name=_display_name(
                payload.get("display_name"),
                fallback="本地运行" if mode == "local" else str(target),
            ),
            mode=mode,
            ssh_target=target,
            ssh_user=user,
            ssh_port=port,
            created_at=created_at or now,
            updated_at=now,
        )

    def to_public_dict(self, *, detection: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(detection, Mapping):
            report = detection.get("report")
            if not isinstance(report, Mapping) or (
                report.get("environment_ref") != self.environment_ref
                or report.get("connection_digest") != self.connection_digest
            ):
                detection = None
        match_value = detection.get("match", {}) if isinstance(detection, Mapping) else {}
        report_value = detection.get("report", {}) if isinstance(detection, Mapping) else {}
        match = match_value if isinstance(match_value, Mapping) else {}
        report = report_value if isinstance(report_value, Mapping) else {}
        return {
            "environment_ref": self.environment_ref,
            "name": self.display_name,
            "mode": self.mode,
            "mode_label": "本地运行" if self.mode == "local" else "SSH 运行",
            "target_label": self.target_label,
            "ssh_target": self.ssh_target,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "connection_digest": self.connection_digest,
            "last_detected_at": report.get("detected_at"),
            "status": match.get("status", "UNDETECTED"),
            "selected_device": match.get("selected_device"),
            "detection_available": bool(detection),
        }


def _timestamp(value: str, *, field: str) -> str:
    try:
        return normalize_timestamp(value, field=field)
    except Exception as exc:
        raise EnvironmentConfigError(f"{field} is invalid") from exc


class EnvironmentConfigStore:
    """Persist connection profiles and last discovery reports atomically."""

    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.is_symlink():
            raise EnvironmentConfigError("environment settings root cannot be a symlink")
        self.root = configured.absolute()
        self.profiles_path = self.root / "environment_profiles.json"
        self.reports_path = self.root / "environment_reports.json"
        self.lock_path = self.root / ".environment.lock"

    @contextmanager
    def _write_lock(self):
        """Serialize profile/report read-modify-write operations across workers."""

        if self.root.is_symlink():
            raise EnvironmentConfigError("environment settings root cannot be a symlink")
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

    @staticmethod
    def _check_file(path: Path) -> None:
        if path.is_symlink():
            raise EnvironmentConfigError("environment settings file cannot be a symlink")
        if path.exists() and not path.is_file():
            raise EnvironmentConfigError("environment settings file is not a regular file")

    @classmethod
    def _read_json(cls, path: Path, *, default: Mapping[str, Any]) -> dict[str, Any]:
        cls._check_file(path)
        if not path.exists():
            return dict(default)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvironmentConfigError("environment settings could not be read") from exc
        if not isinstance(value, dict):
            raise EnvironmentConfigError("environment settings have an invalid shape")
        return value

    def _write_json(self, path: Path, value: Mapping[str, Any]) -> None:
        if self.root.is_symlink():
            raise EnvironmentConfigError("environment settings root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self._check_file(path)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(self.root)
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _read_profiles(self) -> dict[str, EnvironmentProfile]:
        value = self._read_json(
            self.profiles_path,
            default={"version": ENVIRONMENT_CONFIG_VERSION, "profiles": {}},
        )
        if value.get("version") != ENVIRONMENT_CONFIG_VERSION:
            raise EnvironmentConfigError("environment settings version is unsupported")
        raw_profiles = value.get("profiles", {})
        if not isinstance(raw_profiles, Mapping) or len(raw_profiles) > MAX_ENVIRONMENT_PROFILES:
            raise EnvironmentConfigError("environment profile list is invalid")
        profiles: dict[str, EnvironmentProfile] = {}
        for key, raw in raw_profiles.items():
            if not isinstance(key, str) or not isinstance(raw, Mapping):
                raise EnvironmentConfigError("environment profile entry is invalid")
            if raw.get("environment_ref") != key:
                raise EnvironmentConfigError("environment profile identity is inconsistent")
            profile = EnvironmentProfile.from_dict(raw)
            profiles[key] = profile
        return profiles

    def _read_reports(self) -> dict[str, dict[str, Any]]:
        value = self._read_json(
            self.reports_path,
            default={"version": ENVIRONMENT_REPORT_VERSION, "reports": {}},
        )
        if value.get("version") != ENVIRONMENT_REPORT_VERSION:
            raise EnvironmentConfigError("environment report version is unsupported")
        raw_reports = value.get("reports", {})
        if not isinstance(raw_reports, Mapping) or len(raw_reports) > MAX_ENVIRONMENT_PROFILES:
            raise EnvironmentConfigError("environment report list is invalid")
        reports: dict[str, dict[str, Any]] = {}
        for key, raw in raw_reports.items():
            if not isinstance(key, str) or not isinstance(raw, Mapping):
                raise EnvironmentConfigError("environment report entry is invalid")
            try:
                validate_identifier(key, field="environment_ref")
            except Exception as exc:
                raise EnvironmentConfigError("environment report key is invalid") from exc
            reports[key] = dict(raw)
        return reports

    def list_profiles(self) -> tuple[EnvironmentProfile, ...]:
        profiles = self._read_profiles()
        return tuple(profiles[key] for key in sorted(profiles))

    def get_profile(self, environment_ref: str) -> EnvironmentProfile:
        try:
            validate_identifier(environment_ref, field="environment_ref")
        except Exception as exc:
            raise EnvironmentConfigError("environment profile was not found") from exc
        try:
            return self._read_profiles()[environment_ref]
        except KeyError as exc:
            raise EnvironmentConfigError("environment profile was not found") from exc

    def upsert_profile(self, payload: Mapping[str, Any]) -> EnvironmentProfile:
        with self._write_lock():
            profiles = self._read_profiles()
            if not isinstance(payload, Mapping):
                raise EnvironmentConfigError("environment profile must be an object")
            raw_requested_ref = payload.get("environment_ref")
            requested_ref = raw_requested_ref if raw_requested_ref not in (None, "") else None
            if requested_ref is not None and not isinstance(requested_ref, str):
                raise EnvironmentConfigError("unknown environment profile ID")
            existing = profiles.get(requested_ref) if isinstance(requested_ref, str) else None
            if requested_ref is not None and existing is None:
                raise EnvironmentConfigError("unknown environment profile ID")
            profile = EnvironmentProfile.from_payload(
                payload,
                environment_ref=existing.environment_ref if existing else requested_ref,
                created_at=existing.created_at if existing else None,
            )
            migrated_from: str | None = None
            duplicate = next(
                (
                    candidate
                    for candidate in profiles.values()
                    if candidate.environment_ref != (existing.environment_ref if existing else "")
                    and candidate.connection_digest == profile.connection_digest
                ),
                None,
            )
            if duplicate is not None:
                raise EnvironmentConfigError(
                    "an environment profile for this connection already exists"
                )
            if existing is not None and existing.connection_digest != profile.connection_digest:
                migrated_ref = _profile_ref(
                    profile.mode,
                    profile.ssh_target,
                    profile.ssh_user,
                    profile.ssh_port,
                )
                if migrated_ref in profiles and migrated_ref != existing.environment_ref:
                    raise EnvironmentConfigError(
                        "an environment profile for this connection already exists"
                    )
                profile = EnvironmentProfile.from_payload(
                    payload,
                    environment_ref=migrated_ref,
                    created_at=None,
                )
                migrated_from = existing.environment_ref
            elif existing is None and profile.environment_ref in profiles:
                raise EnvironmentConfigError("an environment profile for this connection already exists")
            if profile.environment_ref not in profiles and existing is None and len(profiles) >= MAX_ENVIRONMENT_PROFILES:
                raise EnvironmentConfigError("too many environment profiles")
            if migrated_from is not None:
                del profiles[migrated_from]
            profiles[profile.environment_ref] = profile
            self._write_json(
                self.profiles_path,
                {
                    "version": ENVIRONMENT_CONFIG_VERSION,
                    "profiles": {key: value.to_dict() for key, value in profiles.items()},
                },
            )
            if migrated_from is not None:
                self._clear_detection_unlocked(migrated_from)
            return profile

    def get_detection(self, environment_ref: str) -> dict[str, Any] | None:
        return self._read_reports().get(environment_ref)

    def save_detection(
        self,
        environment_ref: str,
        detection: Mapping[str, Any],
        *,
        expected_connection_digest: str | None = None,
        expected_report_digest: str | None = None,
    ) -> None:
        with self._write_lock():
            try:
                profile = self._read_profiles()[environment_ref]
            except KeyError as exc:
                raise EnvironmentConfigError("environment profile was not found") from exc
            if expected_connection_digest is not None and profile.connection_digest != expected_connection_digest:
                raise EnvironmentConfigError("environment connection changed during detection")
            if not isinstance(detection, Mapping):
                raise EnvironmentConfigError("environment detection must be an object")
            report = detection.get("report")
            if not isinstance(report, Mapping) or report.get("connection_digest") != profile.connection_digest:
                raise EnvironmentConfigError("environment report is bound to a different connection")
            try:
                canonical_json_bytes(detection)
            except (TypeError, ValueError) as exc:
                raise EnvironmentConfigError("environment detection is not JSON serializable") from exc
            reports = self._read_reports()
            if expected_report_digest is not None:
                current = reports.get(environment_ref)
                current_report = current.get("report") if isinstance(current, Mapping) else None
                if not isinstance(current_report, Mapping) or current_report.get("report_digest") != expected_report_digest:
                    raise EnvironmentConfigError("environment report changed during verification")
            reports[environment_ref] = dict(detection)
            self._write_json(
                self.reports_path,
                {"version": ENVIRONMENT_REPORT_VERSION, "reports": reports},
            )

    def clear_detection(self, environment_ref: str) -> None:
        with self._write_lock():
            self._clear_detection_unlocked(environment_ref)

    def _clear_detection_unlocked(self, environment_ref: str) -> None:
        reports = self._read_reports()
        if environment_ref not in reports:
            return
        del reports[environment_ref]
        self._write_json(
            self.reports_path,
            {"version": ENVIRONMENT_REPORT_VERSION, "reports": reports},
        )


CommandRunner = Callable[[Sequence[str], bytes | None, float], tuple[int, bytes]]


class _ProbeOutputLimitExceeded(Exception):
    """Internal signal used to stop a probe as soon as either pipe is too large."""


async def _read_limited_stream(stream: asyncio.StreamReader, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await stream.read(min(64 * 1024, limit - total + 1))
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > limit:
            raise _ProbeOutputLimitExceeded
        chunks.append(chunk)


def _default_runner(
    argv: Sequence[str], input_bytes: bytes | None, timeout_seconds: float
) -> tuple[int, bytes]:
    async def run() -> tuple[int, bytes]:
        try:
            process = await asyncio.create_subprocess_exec(
                *(str(item) for item in argv),
                stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            raise EnvironmentDetectionError(
                "fixed environment probe could not start"
            ) from exc

        if input_bytes is not None and process.stdin is not None:
            try:
                process.stdin.write(input_bytes)
                process.stdin.close()
            except (BrokenPipeError, ConnectionResetError):
                pass

        stdout_task = asyncio.create_task(
            _read_limited_stream(process.stdout, MAX_DETECTION_OUTPUT_BYTES)
        )
        stderr_task = asyncio.create_task(
            _read_limited_stream(process.stderr, MAX_DETECTION_OUTPUT_BYTES)
        )

        async def stop_process() -> None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except OSError:
                    try:
                        process.kill()
                    except ProcessLookupError:
                        pass
            else:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=0.75)
            except asyncio.TimeoutError:
                pass
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError:
                    try:
                        process.kill()
                    except (OSError, ProcessLookupError):
                        pass
            else:
                try:
                    process.kill()
                except (OSError, ProcessLookupError):
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

        stream_tasks = (stdout_task, stderr_task)
        terminated = False
        completed = False

        async def terminate_once() -> None:
            nonlocal terminated
            if not terminated:
                await stop_process()
                terminated = True

        async def drain_streams(timeout_seconds: float = 1.0) -> None:
            _, pending = await asyncio.wait(stream_tasks, timeout=timeout_seconds)
            if pending:
                for task in pending:
                    task.cancel()
            await asyncio.gather(*stream_tasks, return_exceptions=True)

        try:
            deadline = asyncio.get_running_loop().time() + timeout_seconds
            done, pending = await asyncio.wait(
                stream_tasks,
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_EXCEPTION,
            )
            errors = [
                task.exception()
                for task in done
                if not task.cancelled() and task.exception() is not None
            ]
            if errors:
                await terminate_once()
                await drain_streams()
                if isinstance(errors[0], _ProbeOutputLimitExceeded):
                    raise EnvironmentDetectionError(
                        "environment probe output exceeded the safety limit"
                    ) from errors[0]
                raise errors[0]
            if pending:
                await terminate_once()
                await drain_streams()
                raise EnvironmentDetectionError("fixed environment probe timed out")
            remaining = deadline - asyncio.get_running_loop().time()
            if process.returncode is None and remaining <= 0:
                await terminate_once()
                await drain_streams()
                raise EnvironmentDetectionError("fixed environment probe timed out")
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=max(0.01, remaining))
                except asyncio.TimeoutError as exc:
                    await terminate_once()
                    await drain_streams()
                    raise EnvironmentDetectionError(
                        "fixed environment probe timed out"
                    ) from exc
            stdout, _stderr = await asyncio.gather(*stream_tasks)
            completed = True
            return process.returncode or 0, stdout
        finally:
            if not completed:
                await terminate_once()
            await drain_streams()
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

    try:
        return asyncio.run(run())
    except EnvironmentDetectionError:
        raise
    except (OSError, RuntimeError) as exc:
        raise EnvironmentDetectionError("fixed environment probe could not start") from exc


_PROBE_SCRIPT = dedent(
    r'''
    import importlib.metadata
    import importlib.util
    import json
    import os
    from pathlib import Path
    import platform
    import selectors
    import signal
    import shutil
    import subprocess
    import sys
    import threading
    import time

    def text(value, limit=256):
        value = str(value or "")
        return value.replace("\x00", "")[:limit]

    MAX_SUBPROCESS_OUTPUT_BYTES = 64 * 1024
    _ACTIVE_PROCESS_GROUPS = set()

    def terminate_process_group(group_id):
        kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM)
        if os.name == "posix":
            try:
                os.killpg(group_id, kill_signal)
                return
            except ProcessLookupError:
                return
            except OSError:
                pass
        try:
            os.kill(group_id, kill_signal)
        except (OSError, ProcessLookupError):
            pass

    def cleanup_active_process_groups(signum, _frame):
        for group_id in tuple(_ACTIVE_PROCESS_GROUPS):
            terminate_process_group(group_id)
        raise SystemExit(128 + signum)

    for signal_name in ("SIGTERM", "SIGINT", "SIGHUP"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            signal.signal(signal_value, cleanup_active_process_groups)

    def enforce_probe_deadline():
        try:
            budget = float(os.environ.get("MOLLY_PROBE_TIMEOUT_SECONDS", "0"))
        except (TypeError, ValueError):
            budget = 0
        if budget <= 0:
            return
        time.sleep(budget)
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except OSError:
            pass

    threading.Thread(target=enforce_probe_deadline, daemon=True).start()

    def bounded_command(path, args=("--version",), timeout=5):
        try:
            process = subprocess.Popen(
                [path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=False, close_fds=True, start_new_session=(os.name == "posix"),
            )
        except OSError:
            return 1, ""
        _ACTIVE_PROCESS_GROUPS.add(process.pid)
        selector = selectors.DefaultSelector()
        registered = {}
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
            if stream is not None:
                selector.register(stream, selectors.EVENT_READ, name)
                registered[stream] = name
        deadline = time.monotonic() + max(0.1, float(timeout))
        aborted = False

        def terminate_process():
            terminate_process_group(process.pid)
            if process.poll() is None and os.name != "posix":
                try:
                    process.kill()
                except OSError:
                    pass
            try:
                process.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired, TimeoutError):
                pass

        try:
            while registered:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    aborted = True
                    return 1, ""
                events = selector.select(remaining)
                if not events:
                    aborted = True
                    return 1, ""
                for key, _ in events:
                    stream = key.fileobj
                    name = key.data
                    capacity = MAX_SUBPROCESS_OUTPUT_BYTES - len(buffers[name]) + 1
                    reader = getattr(stream, "read1", stream.read)
                    chunk = reader(min(64 * 1024, max(1, capacity)))
                    if not chunk:
                        try:
                            selector.unregister(stream)
                        except (KeyError, ValueError):
                            pass
                        registered.pop(stream, None)
                        stream.close()
                        continue
                    buffers[name].extend(chunk)
                    if len(buffers[name]) > MAX_SUBPROCESS_OUTPUT_BYTES:
                        aborted = True
                        return 1, ""
            remaining = max(0.1, deadline - time.monotonic())
            process.wait(timeout=remaining)
        except (OSError, ValueError, subprocess.TimeoutExpired, TimeoutError):
            aborted = True
            return 1, ""
        finally:
            _ACTIVE_PROCESS_GROUPS.discard(process.pid)
            if aborted or process.poll() is None:
                terminate_process()
            for stream in tuple(registered):
                try:
                    selector.unregister(stream)
                except (KeyError, ValueError):
                    pass
                stream.close()
            selector.close()
        output = bytes(buffers["stdout"] or buffers["stderr"])
        return process.returncode if process.returncode is not None else 1, output.decode("utf-8", "replace")

    def command(name, args=("--version",), timeout=5):
        path = shutil.which(name)
        if not path:
            return {"available": False, "version": "", "path": ""}
        returncode, output = bounded_command(path, args, timeout=timeout)
        return {
            "available": returncode == 0,
            "version": text(output.strip().splitlines()[0] if output.strip() else "", 160),
            "path": text(path, 512),
        }

    def distribution(names):
        for name in names:
            try:
                return {"installed": True, "package": name, "version": text(importlib.metadata.version(name), 80)}
            except importlib.metadata.PackageNotFoundError:
                pass
        return {"installed": False, "package": "", "version": ""}

    def importable(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    PYTHON_ENVIRONMENT_PROBE = """
    import importlib.metadata
    import importlib.util
    import json
    import platform
    import sys

    def package(names):
        for name in names:
            try:
                return {"installed": True, "importable": True, "package": name, "version": importlib.metadata.version(name)}
            except importlib.metadata.PackageNotFoundError:
                pass
        return {"installed": False, "importable": False, "package": "", "version": ""}

    def module(name):
        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False

    unimol = package(("unimol-tools", "unimol_tools"))
    unimol["importable"] = module("unimol_tools")
    reinvent = package(("reinvent4", "reinvent"))
    reinvent["importable"] = module("reinvent")
    print(json.dumps({
        "executable": sys.executable,
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "unimol": unimol,
        "reinvent4": reinvent,
    }, ensure_ascii=True, separators=(",", ":")))
    """

    def add_python_candidate(candidates, seen, path, source):
        if not path:
            return
        try:
            candidate = Path(str(path)).expanduser()
            if not candidate.is_file():
                return
            key = str(candidate)
        except (OSError, TypeError, ValueError):
            return
        if key in seen:
            return
        seen.add(key)
        candidates.append((key, text(source, 80)))

    def add_environment_python(candidates, seen, environment, source):
        root = Path(str(environment)).expanduser()
        add_python_candidate(candidates, seen, root / "bin" / "python", source)
        add_python_candidate(candidates, seen, root / "Scripts" / "python.exe", source)

    python_candidates = []
    python_seen = set()
    add_python_candidate(python_candidates, python_seen, sys.executable, "probe interpreter")
    for name in ("python3", "python"):
        add_python_candidate(python_candidates, python_seen, shutil.which(name), "PATH")
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT"):
        value = os.environ.get(variable)
        if value:
            add_environment_python(python_candidates, python_seen, value, variable)
    add_environment_python(python_candidates, python_seen, Path.cwd() / ".venv", "project .venv")

    for manager_name in ("conda", "mamba", "micromamba"):
        manager_path = shutil.which(manager_name)
        if not manager_path:
            continue
        returncode, output = bounded_command(manager_path, ("env", "list", "--json"), timeout=8)
        if returncode != 0:
            continue
        try:
            manager_data = json.loads(output)
        except json.JSONDecodeError:
            continue
        environments = manager_data.get("envs", ()) if isinstance(manager_data, dict) else ()
        if not isinstance(environments, list):
            continue
        for environment in environments[:32]:
            if isinstance(environment, str):
                add_environment_python(python_candidates, python_seen, environment, manager_name)

    uv_path = shutil.which("uv")
    if uv_path:
        returncode, output = bounded_command(uv_path, ("python", "list", "--only-installed"), timeout=8)
        if returncode == 0:
            for line in output.splitlines()[:64]:
                for token in reversed(line.split()):
                    if token.startswith(("/", "~")):
                        add_python_candidate(python_candidates, python_seen, token, "uv")
                        break

    python_environments = []
    for path, source in python_candidates[:32]:
        returncode, output = bounded_command(path, ("-c", PYTHON_ENVIRONMENT_PROBE), timeout=8)
        if returncode != 0:
            continue
        try:
            environment = json.loads(output)
        except json.JSONDecodeError:
            continue
        if not isinstance(environment, dict):
            continue
        environment["source"] = source
        environment["executable"] = text(path, 512)
        environment["name"] = text(Path(path).parent.parent.name, 128)
        python_environments.append(environment)

    def repository_candidates():
        home = Path.home()
        values = [
            os.environ.get("REINVENT4_REPOSITORY"),
            os.environ.get("REINVENT4_HOME"),
            str(home / "REINVENT4"),
            str(home / "reinvent4"),
            "/opt/REINVENT4",
            "/opt/reinvent4",
        ]
        result = []
        seen = set()
        for raw in values:
            if not raw:
                continue
            path = Path(raw).expanduser()
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            result.append({
                "path": text(path, 512),
                "exists": path.is_dir(),
                "git": (path / ".git").exists(),
                "config": (path / "pyproject.toml").is_file() or (path / "setup.py").is_file(),
            })
        return result[:8]

    def weight_entries(roots):
        suffixes = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
        entries = []
        total = 0
        for root in roots:
            if not root.is_dir():
                continue
            try:
                children = sorted(root.iterdir(), key=lambda item: item.name)[:64]
            except OSError:
                continue
            for child in children:
                if not child.is_file() or child.suffix.casefold() not in suffixes:
                    continue
                try:
                    size = max(0, int(child.stat().st_size))
                except OSError:
                    continue
                entries.append({"name": text(child.name, 180), "path": text(child, 512), "size_bytes": size})
                total += size
                if len(entries) >= 128:
                    return entries, total
        return entries, total

    run_directory = Path(os.environ.get("MOLLY_PROBE_RUN_DIRECTORY", "~/.local/share/molly/runtimes")).expanduser()
    parent = run_directory
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    try:
        usage = shutil.disk_usage(parent)
        disk = {
            "path": text(run_directory, 512),
            "exists": run_directory.is_dir(),
            "writable": run_directory.is_dir() and os.access(run_directory, os.W_OK),
            "parent_writable": parent.is_dir() and os.access(parent, os.W_OK),
            "total_bytes": int(usage.total),
            "available_bytes": int(usage.free),
        }
    except OSError:
        disk = {
            "path": text(run_directory, 512), "exists": run_directory.is_dir(),
            "writable": False, "parent_writable": False,
            "total_bytes": 0, "available_bytes": 0,
        }

    devices = []
    nvidia_path = shutil.which("nvidia-smi")
    if nvidia_path:
        returncode, output = bounded_command(
            nvidia_path,
            ("--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"),
            timeout=5,
        )
        if returncode == 0:
            for line in output.splitlines()[:8]:
                fields = [item.strip() for item in line.split(",")]
                if len(fields) < 3:
                    continue
                try:
                    memory = int(float(fields[1]))
                except ValueError:
                    memory = 0
                devices.append({"name": text(fields[0], 120), "memory_mib": max(0, memory), "driver_version": text(fields[2], 80)})
    nvcc = command("nvcc", ("--version",))
    gpu = {
        "available": bool(devices),
        "devices": devices,
        "cuda": {"available": nvcc["available"], "version": nvcc["version"]},
        "nvidia_smi": command("nvidia-smi", ("--version",)),
    }

    python_tools = {name: command(name) for name in ("python", "python3", "conda", "mamba", "micromamba", "uv")}
    unimol = distribution(("unimol-tools", "unimol_tools"))
    unimol["importable"] = importable("unimol_tools")
    reinvent = distribution(("reinvent4", "reinvent"))
    reinvent["importable"] = importable("reinvent")
    license_path = os.environ.get("REINVENT4_LICENSE_PATH", "")
    reinvent["repositories"] = repository_candidates()
    reinvent["license_present"] = bool((license_path and Path(license_path).expanduser().is_file()) or os.environ.get("REINVENT4_LICENSE"))
    roots = [
        Path.home() / ".cache" / "unimol",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        Path.home() / ".cache" / "molly" / "weights",
        run_directory / "weights",
    ]
    entries, total = weight_entries(roots)
    report = {
        "system": {"os": text(platform.system(), 64), "release": text(platform.release(), 128), "architecture": text(platform.machine(), 64)},
        "disk": disk,
        "gpu": gpu,
        "python": {"executable": text(sys.executable, 512), "version": text(platform.python_version(), 80), "implementation": text(platform.python_implementation(), 80), "managers": python_tools, "environments": python_environments},
        "unimol": unimol,
        "reinvent4": reinvent,
        "weights": {"entries": entries, "total_bytes": total},
    }
    print(json.dumps(report, ensure_ascii=True, separators=(",", ":")))
    '''
).strip()


def _clean_string(value: Any, *, maximum: int = 512) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\x00", "").replace("\r", "").replace("\n", "")[:maximum]


def _clean_int(value: Any, *, maximum: int = 2**63 - 1) -> int:
    if isinstance(value, bool):
        return 0
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(result, maximum))


def _clean_command(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    return {
        "available": bool(raw.get("available")),
        "version": _clean_string(raw.get("version"), maximum=160),
        "path": _clean_string(raw.get("path")),
    }


def _clean_package(value: Any) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    return {
        "installed": bool(raw.get("installed")),
        "importable": bool(raw.get("importable")),
        "package": _clean_string(raw.get("package"), maximum=80),
        "version": _clean_string(raw.get("version"), maximum=80),
    }


def _normalized_weight_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    try:
        return str(Path(value).expanduser().resolve(strict=False))
    except (OSError, RuntimeError, TypeError, ValueError):
        return ""


def _weight_candidate_id(path: str, size_bytes: int) -> str:
    return sha256_bytes(
        canonical_json_bytes({"path": path, "size_bytes": size_bytes})
    )


def _normalize_probe(
    raw: Mapping[str, Any],
    *,
    verified_weight_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    trusted_weight_records: dict[str, dict[str, Any]] = {}
    for path, record in (verified_weight_records or {}).items():
        normalized_path = _normalized_weight_path(path)
        if not normalized_path or not isinstance(record, Mapping):
            raise EnvironmentConfigError("verified weight record is invalid")
        try:
            digest = validate_sha256(
                record.get("sha256", ""), field="verified weight digest"
            )
            size_bytes = _clean_int(record.get("size_bytes"), maximum=2**50)
        except Exception as exc:
            raise EnvironmentConfigError("verified weight digest is invalid") from exc
        trusted_weight_records[normalized_path] = {
            "sha256": digest,
            "size_bytes": size_bytes,
        }
    system = raw.get("system") if isinstance(raw.get("system"), Mapping) else {}
    disk = raw.get("disk") if isinstance(raw.get("disk"), Mapping) else {}
    gpu_raw = raw.get("gpu") if isinstance(raw.get("gpu"), Mapping) else {}
    python_raw = raw.get("python") if isinstance(raw.get("python"), Mapping) else {}
    unimol_raw = raw.get("unimol") if isinstance(raw.get("unimol"), Mapping) else {}
    reinvent_raw = raw.get("reinvent4") if isinstance(raw.get("reinvent4"), Mapping) else {}
    weights_raw = raw.get("weights") if isinstance(raw.get("weights"), Mapping) else {}
    devices = []
    for item in gpu_raw.get("devices", ()) if isinstance(gpu_raw.get("devices", ()), Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        devices.append(
            {
                "name": _clean_string(item.get("name"), maximum=120),
                "memory_mib": _clean_int(item.get("memory_mib"), maximum=4 * 1024 * 1024),
                "driver_version": _clean_string(item.get("driver_version"), maximum=80),
            }
        )
    repositories = []
    for item in reinvent_raw.get("repositories", ()) if isinstance(reinvent_raw.get("repositories", ()), Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        repositories.append(
            {
                "path": _clean_string(item.get("path")),
                "exists": bool(item.get("exists")),
                "git": bool(item.get("git")),
                "config": bool(item.get("config")),
            }
        )
    weights = []
    for item in weights_raw.get("entries", ()) if isinstance(weights_raw.get("entries", ()), Sequence) else ():
        if not isinstance(item, Mapping):
            continue
        name = _clean_string(item.get("name"), maximum=180)
        path = _clean_string(item.get("path"))
        size_bytes = _clean_int(item.get("size_bytes"), maximum=2**50)
        normalized_path = _normalized_weight_path(path)
        weight = {
            "name": name,
            "path": path,
            "size_bytes": size_bytes,
            "candidate_id": _weight_candidate_id(normalized_path, size_bytes),
            "verification_status": "pending",
        }
        verified = trusted_weight_records.get(normalized_path)
        if (
            verified is not None
            and size_bytes > 0
            and verified["size_bytes"] == size_bytes
        ):
            weight["sha256"] = verified["sha256"]
            weight["verification_status"] = "verified"
        weights.append(
            weight
        )
    managers = python_raw.get("managers", {}) if isinstance(python_raw.get("managers"), Mapping) else {}
    cuda_raw = gpu_raw.get("cuda") if isinstance(gpu_raw.get("cuda"), Mapping) else {}
    python_environments = []
    raw_python_environments = python_raw.get("environments", ())
    if isinstance(raw_python_environments, Sequence) and not isinstance(
        raw_python_environments, (str, bytes, bytearray)
    ):
        for item in raw_python_environments[:32]:
            if not isinstance(item, Mapping):
                continue
            source = _clean_string(item.get("source"), maximum=80)
            executable = _clean_string(item.get("executable"))
            python_environments.append(
                {
                    "name": _clean_string(item.get("name"), maximum=128) or source or executable,
                    "source": source,
                    "executable": executable,
                    "version": _clean_string(item.get("version"), maximum=80),
                    "implementation": _clean_string(item.get("implementation"), maximum=80),
                    "unimol": _clean_package(item.get("unimol")),
                    "reinvent4": _clean_package(item.get("reinvent4")),
                }
            )
    return {
        "system": {
            "os": _clean_string(system.get("os"), maximum=64) or "未知",
            "release": _clean_string(system.get("release"), maximum=128),
            "architecture": _clean_string(system.get("architecture"), maximum=64) or "未知",
        },
        "disk": {
            "path": _clean_string(disk.get("path")),
            "exists": bool(disk.get("exists")),
            "writable": bool(disk.get("writable")),
            "parent_writable": bool(disk.get("parent_writable")),
            "total_bytes": _clean_int(disk.get("total_bytes"), maximum=2**60),
            "available_bytes": _clean_int(disk.get("available_bytes"), maximum=2**60),
        },
        "gpu": {
            "available": bool(gpu_raw.get("available")) and bool(devices),
            "devices": devices[:8],
            "cuda": {
                "available": bool(cuda_raw.get("available")),
                "version": _clean_string(cuda_raw.get("version"), maximum=160),
            },
            "nvidia_smi": _clean_command(gpu_raw.get("nvidia_smi")),
        },
        "python": {
            "executable": _clean_string(python_raw.get("executable")),
            "version": _clean_string(python_raw.get("version"), maximum=80),
            "implementation": _clean_string(python_raw.get("implementation"), maximum=80),
            "managers": {str(key): _clean_command(value) for key, value in list(managers.items())[:8]},
            "environments": python_environments,
        },
        "unimol": _clean_package(unimol_raw),
        "reinvent4": {
            **_clean_package(reinvent_raw),
            "repositories": repositories[:8],
            "license_present": bool(reinvent_raw.get("license_present")),
        },
        "weights": {
            "entries": weights[:128],
            "total_bytes": _clean_int(weights_raw.get("total_bytes"), maximum=2**52),
            "verification_status": (
                "verified"
                if any(item.get("verification_status") == "verified" for item in weights)
                else "pending"
            ),
        },
    }


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Bounded, sanitized output of the fixed discovery probe."""

    environment_ref: str
    connection_digest: str
    mode: str
    target_label: str
    detected_at: str
    probes: tuple[str, ...]
    data: Mapping[str, Any]
    report_digest: str

    @classmethod
    def from_probe(
        cls,
        profile: EnvironmentProfile,
        raw: Mapping[str, Any],
        *,
        detected_at: str | None = None,
        verified_weight_records: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> "EnvironmentReport":
        normalized = _normalize_probe(
            raw, verified_weight_records=verified_weight_records
        )
        timestamp = (
            _timestamp(detected_at, field="environment detected_at")
            if detected_at is not None
            else utc_timestamp()
        )
        payload = {
            "version": ENVIRONMENT_REPORT_VERSION,
            "environment_ref": profile.environment_ref,
            "connection_digest": profile.connection_digest,
            "mode": profile.mode,
            "target_label": profile.target_label,
            "detected_at": timestamp,
            "probes": list(READ_ONLY_PROBE_NAMES),
            **normalized,
        }
        digest = sha256_bytes(canonical_json_bytes(payload))
        return cls(
            environment_ref=profile.environment_ref,
            connection_digest=profile.connection_digest,
            mode=profile.mode,
            target_label=profile.target_label,
            detected_at=timestamp,
            probes=READ_ONLY_PROBE_NAMES,
            data=normalized,
            report_digest=digest,
        )

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "version": ENVIRONMENT_REPORT_VERSION,
            "environment_ref": self.environment_ref,
            "connection_digest": self.connection_digest,
            "mode": self.mode,
            "target_label": self.target_label,
            "detected_at": self.detected_at,
            "probes": list(self.probes),
            **{key: dict(item) if isinstance(item, Mapping) else item for key, item in self.data.items()},
        }
        if include_digest:
            value["report_digest"] = self.report_digest
        return value


@dataclass(frozen=True, slots=True)
class CompatibilityEntry:
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }


COMPATIBILITY_CATALOG = (
    CompatibilityEntry(
        component_id="python",
        name="Python 3.10+",
        version="python>=3.10",
        source="Conda / Mamba / uv 固定环境清单",
        source_url="https://www.python.org/downloads/",
        estimated_download_bytes=0,
        estimated_disk_bytes=0,
        estimated_duration_seconds=0,
        install_subdirectory="python",
    ),
    CompatibilityEntry(
        component_id="unimol",
        name="Uni-Mol",
        version="unimol-tools==0.1.5",
        source="PyPI · unimol-tools",
        source_url="https://pypi.org/project/unimol-tools/0.1.5/",
        estimated_download_bytes=256 * 1024 * 1024,
        estimated_disk_bytes=768 * 1024 * 1024,
        estimated_duration_seconds=900,
        install_subdirectory="unimol",
    ),
    CompatibilityEntry(
        component_id="reinvent4",
        name="REINVENT4",
        version="reinvent4==4.7.15",
        source="REINVENT4 官方仓库（需许可证/凭据）",
        source_url="https://github.com/MolecularAI/REINVENT4",
        estimated_download_bytes=256 * 1024 * 1024,
        estimated_disk_bytes=768 * 1024 * 1024,
        estimated_duration_seconds=900,
        install_subdirectory="reinvent4",
        requires_license=True,
    ),
    CompatibilityEntry(
        component_id="unimol-weights",
        name="Uni-Mol 预训练权重",
        version="unimolv1",
        source="Uni-Mol 固定权重清单",
        source_url="https://github.com/dptech-corp/Uni-Mol",
        estimated_download_bytes=512 * 1024 * 1024,
        estimated_disk_bytes=1024 * 1024 * 1024,
        estimated_duration_seconds=1_200,
        install_subdirectory="weights",
    ),
)


def _version_matches(actual: str, requirement: str) -> bool:
    expected = requirement.split("==", 1)[-1]
    return bool(actual) and actual == expected


def _python_matches(version: str) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", version)
    return match is not None and (int(match.group(1)), int(match.group(2))) >= (3, 10)


def _repository_ready(repositories: Sequence[Mapping[str, Any]]) -> bool:
    return any(bool(item.get("exists")) and bool(item.get("config")) for item in repositories)


def _weights_ready(weights: Mapping[str, Any], *, unimol_ready: bool) -> bool:
    if not unimol_ready:
        return False
    entries = weights.get("entries", ())
    if not isinstance(entries, Sequence):
        return False
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", "")).casefold()
        if name not in COMPATIBLE_WEIGHT_NAMES or item.get("verification_status") != "verified":
            continue
        size_bytes = _clean_int(item.get("size_bytes"))
        path = _normalized_weight_path(item.get("path"))
        if size_bytes <= 0 or item.get("candidate_id") != _weight_candidate_id(path, size_bytes):
            continue
        try:
            validate_sha256(item.get("sha256", ""), field="weight sha256")
        except Exception:
            continue
        return True
    return False


def _python_candidates(
    python_data: Mapping[str, Any],
    top_level_unimol: Mapping[str, Any],
    top_level_reinvent: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    raw_environments = python_data.get("environments", ())
    if isinstance(raw_environments, Sequence) and not isinstance(
        raw_environments, (str, bytes, bytearray)
    ) and raw_environments:
        return tuple(item for item in raw_environments if isinstance(item, Mapping))
    return (
        {
            "name": "probe interpreter",
            "source": "probe interpreter",
            "executable": python_data.get("executable", ""),
            "version": python_data.get("version", ""),
            "implementation": python_data.get("implementation", ""),
            "unimol": top_level_unimol,
            "reinvent4": top_level_reinvent,
        },
    )


def _environment_package_ready(
    environment: Mapping[str, Any],
    component: str,
    entry: CompatibilityEntry,
) -> bool:
    package = environment.get(component, {})
    return (
        bool(environment.get("executable"))
        and _python_matches(str(environment.get("version", "")))
        and isinstance(package, Mapping)
        and bool(package.get("installed"))
        and bool(package.get("importable"))
        and _version_matches(str(package.get("version", "")), entry.version)
    )


def _environment_summary(environment: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not environment:
        return None
    return {
        "name": _clean_string(environment.get("name"), maximum=128),
        "source": _clean_string(environment.get("source"), maximum=80),
        "executable": _clean_string(environment.get("executable")),
        "version": _clean_string(environment.get("version"), maximum=80),
    }


def _install_location(profile: EnvironmentProfile, entry: CompatibilityEntry) -> str:
    base = ".molly/runtimes/<runtime-id>" if profile.mode == "local" else "~/.local/share/molly/runtimes/<runtime-id>"
    return f"{base}/{entry.install_subdirectory}"


def match_environment(
    profile: EnvironmentProfile,
    report: EnvironmentReport,
    *,
    catalog: Sequence[CompatibilityEntry] = COMPATIBILITY_CATALOG,
) -> dict[str, Any]:
    """Prefer a compatible existing environment and otherwise build a plan.

    The returned plan is descriptive only.  It carries no executable command
    and is never applied by this PR.
    """

    data = report.data
    python_data = data.get("python", {})
    disk = data.get("disk", {})
    gpu = data.get("gpu", {})
    unimol = data.get("unimol", {})
    reinvent = data.get("reinvent4", {})
    weights = data.get("weights", {})
    top_level_unimol = unimol if isinstance(unimol, Mapping) else {}
    top_level_reinvent = reinvent if isinstance(reinvent, Mapping) else {}
    entries = {entry.component_id: entry for entry in catalog}
    python_environments = _python_candidates(
        python_data, top_level_unimol, top_level_reinvent
    )
    selected_python_environment = next(
        (
            environment
            for environment in python_environments
            if bool(environment.get("executable"))
            and _python_matches(str(environment.get("version", "")))
        ),
        None,
    )
    unimol_environment = next(
        (
            environment
            for environment in python_environments
            if _environment_package_ready(environment, "unimol", entries["unimol"])
        ),
        None,
    )
    reinvent_environment = next(
        (
            environment
            for environment in python_environments
            if _environment_package_ready(environment, "reinvent4", entries["reinvent4"])
        ),
        None,
    )
    python_ready = selected_python_environment is not None
    unimol_ready = unimol_environment is not None
    unimol = (
        unimol_environment.get("unimol", {})
        if unimol_environment is not None
        else {}
    )
    reinvent = dict(
        reinvent_environment.get("reinvent4", {})
        if reinvent_environment is not None
        else {}
    )
    reinvent["repositories"] = top_level_reinvent.get("repositories", ())
    reinvent["license_present"] = bool(top_level_reinvent.get("license_present"))
    repositories = reinvent.get("repositories", ())
    reinvent_ready = (
        reinvent_environment is not None
        and _repository_ready(repositories if isinstance(repositories, Sequence) else ())
    )
    weights_ready = _weights_ready(weights, unimol_ready=unimol_ready)
    directory_ready = bool(disk.get("exists")) and bool(disk.get("writable"))
    license_attention = reinvent_environment is not None and not bool(
        reinvent.get("license_present")
    )

    reusable: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    if python_ready:
        reusable.append(
            {
                "component_id": "python",
                "name": "Python",
                "version": selected_python_environment.get("version", ""),
                "environment": _environment_summary(selected_python_environment),
            }
        )
    else:
        missing.append({"component_id": "python", "reason": "未找到兼容的 Python 3.10+ 解释器"})
    if isinstance(python_data.get("managers"), Mapping):
        for manager in ("conda", "mamba", "uv"):
            value = python_data["managers"].get(manager)
            if isinstance(value, Mapping) and value.get("available"):
                reusable.append({"component_id": manager, "name": manager.title(), "version": value.get("version", "")})
    if unimol_ready:
        reusable.append(
            {
                "component_id": "unimol",
                "name": "Uni-Mol",
                "version": unimol.get("version", ""),
                "environment": _environment_summary(unimol_environment),
            }
        )
    else:
        missing.append({"component_id": "unimol", "reason": "未找到兼容的 Uni-Mol 版本"})
    if reinvent_ready:
        reusable.append(
            {
                "component_id": "reinvent4",
                "name": "REINVENT4",
                "version": reinvent.get("version", ""),
                "environment": _environment_summary(reinvent_environment),
            }
        )
    else:
        missing.append({"component_id": "reinvent4", "reason": "未找到兼容的 REINVENT4 环境或仓库"})
    if weights_ready:
        reusable.append({"component_id": "unimol-weights", "name": "Uni-Mol 预训练权重", "version": entries["unimol-weights"].version})
    else:
        missing.append({"component_id": "unimol-weights", "reason": "未找到可确认兼容的模型权重"})
    if directory_ready:
        reusable.append({"component_id": "runtime-directory", "name": "可写运行目录", "version": "现有目录"})
    else:
        missing.append({"component_id": "runtime-directory", "reason": "隔离运行目录尚未确认可写"})

    blockers: list[str] = []
    if license_attention:
        blockers.append("REINVENT4 许可证或凭据未确认，安装前必须暂停并由用户补充")
    if not python_ready:
        blockers.append("没有可用的 Python 3.10+ 解释器")

    plan_items: list[dict[str, Any]] = []
    for item in missing:
        entry = entries.get(item["component_id"])
        if entry is None:
            plan_items.append(
                {
                    "component_id": item["component_id"],
                    "name": item["component_id"],
                    "action": "prepare_isolated_directory",
                    "source": "服务器端固定运行目录策略",
                    "version": "server-owned",
                    "estimated_download_bytes": 0,
                    "estimated_disk_bytes": 0,
                    "estimated_duration_seconds": 0,
                    "install_location": ".molly/runtimes/<runtime-id>" if profile.mode == "local" else "~/.local/share/molly/runtimes/<runtime-id>",
                    "reason": item["reason"],
                }
            )
            continue
        value = entry.to_dict()
        value.update(
            {
                "action": "install_in_isolated_directory",
                "install_location": _install_location(profile, entry),
                "reason": item["reason"],
            }
        )
        plan_items.append(value)

    device_candidates = [
        {
            "device": "GPU",
            "available": bool(gpu.get("available")) and bool(gpu.get("cuda", {}).get("available")),
            "reason": "检测到 GPU 和 CUDA" if gpu.get("available") and gpu.get("cuda", {}).get("available") else "未确认可用的 GPU/CUDA",
        },
        {"device": "CPU", "available": True, "reason": "CPU 是始终可用的后备执行方式"},
    ]
    selected_device = "GPU" if device_candidates[0]["available"] else "CPU"
    ready = not missing and not blockers
    status = "READY" if ready else "BLOCKED" if blockers else "PLAN_REQUIRED"
    selected_candidate = "existing" if ready else "isolated"
    plan = {
        "status": "NO_INSTALL_REQUIRED" if ready else "BLOCKED" if blockers else "INSTALL_PREVIEW",
        "will_execute": False,
        "requires_confirmation": True,
        "catalog_version": COMPATIBILITY_CATALOG_VERSION,
        "selected_candidate": selected_candidate,
        "target_directory": ".molly/runtimes/<runtime-id>" if profile.mode == "local" else "~/.local/share/molly/runtimes/<runtime-id>",
        "integrity_policy": "安装阶段必须按兼容性清单校验固定 SHA-256；只读检测阶段不下载文件",
        "items": plan_items,
        "estimated_download_bytes": sum(int(item.get("estimated_download_bytes", 0)) for item in plan_items),
        "estimated_disk_bytes": sum(int(item.get("estimated_disk_bytes", 0)) for item in plan_items),
        "estimated_duration_seconds": sum(int(item.get("estimated_duration_seconds", 0)) for item in plan_items),
        "notes": [
            "本次仅生成只读安装预览，不会下载、安装、覆盖或修改现有环境。",
            "真正执行前仍需再次确认固定来源、版本、大小和隔离目录。",
        ],
    }
    return {
        "status": status,
        "selected_device": selected_device,
        "selected_candidate": selected_candidate,
        "candidate_environments": [
            {
                "candidate_id": "existing",
                "label": "复用已检测环境",
                "ready": ready,
                "reason": "所有必需组件与运行目录均已匹配" if ready else "存在缺失或不兼容组件",
            },
            {
                "candidate_id": "isolated",
                "label": "隔离目录环境",
                "ready": False,
                "reason": "仅生成待确认安装预览",
            },
        ],
        "device_candidates": device_candidates,
        "python_environments": [
            {
                **(_environment_summary(environment) or {}),
                "unimol_compatible": _environment_package_ready(
                    environment, "unimol", entries["unimol"]
                ),
                "reinvent4_compatible": _environment_package_ready(
                    environment, "reinvent4", entries["reinvent4"]
                ),
            }
            for environment in python_environments
        ],
        "selected_python_environment": _environment_summary(selected_python_environment),
        "selected_unimol_environment": _environment_summary(unimol_environment),
        "selected_reinvent4_environment": _environment_summary(reinvent_environment),
        "compatibility": {
            "python": {"compatible": python_ready, "detected_version": selected_python_environment.get("version", "") if selected_python_environment else python_data.get("version", ""), "required": "Python >=3.10"},
            "unimol": {"compatible": unimol_ready, "detected_version": unimol.get("version", ""), "required": entries["unimol"].version},
            "reinvent4": {"compatible": reinvent_ready and not license_attention, "detected_version": reinvent.get("version", ""), "required": entries["reinvent4"].version},
            "unimol-weights": {"compatible": weights_ready, "required": entries["unimol-weights"].version},
            "runtime-directory": {"compatible": directory_ready, "required": "可写隔离目录"},
        },
        "reusable": reusable,
        "missing": missing,
        "blockers": blockers,
        "plan": plan,
    }


class EnvironmentDetector:
    """Run the fixed local or SSH read-only probe."""

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        timeout_seconds: float = DETECTION_TIMEOUT_SECONDS,
        local_run_directory: Path | str | None = None,
    ) -> None:
        if runner is not None and not callable(runner):
            raise TypeError("environment detector runner must be callable")
        if isinstance(timeout_seconds, bool) or not 1 <= float(timeout_seconds) <= 300:
            raise ValueError("environment detector timeout is out of range")
        self.runner = runner or _default_runner
        self.timeout_seconds = float(timeout_seconds)
        self.local_run_directory = (
            Path(local_run_directory).absolute()
            if local_run_directory is not None
            else Path.cwd().absolute() / ".molly" / "runtimes"
        )

    @staticmethod
    def _ssh_argv(profile: EnvironmentProfile, interpreter: str) -> tuple[str, ...]:
        if profile.mode != "ssh" or not profile.ssh_target or not profile.ssh_user or not profile.ssh_port:
            raise EnvironmentDetectionError("SSH environment profile is incomplete")
        return (
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "RequestTTY=no",
            "-o",
            "ConnectionAttempts=1",
            "-o",
            "ConnectTimeout=15",
            "-p",
            str(profile.ssh_port),
            "-l",
            profile.ssh_user,
            "--",
            profile.ssh_target,
            interpreter,
            "-",
        )

    def _run(self, argv: Sequence[str], *, input_bytes: bytes | None = None) -> bytes:
        try:
            returncode, stdout = self.runner(argv, input_bytes, self.timeout_seconds)
        except (TypeError, ValueError) as exc:
            raise EnvironmentDetectionError("environment probe runner returned an invalid result") from exc
        if not isinstance(returncode, int) or not isinstance(stdout, bytes):
            raise EnvironmentDetectionError("environment probe runner returned an invalid result")
        if returncode != 0:
            raise EnvironmentDetectionError("environment probe failed; no remote output was retained")
        if len(stdout) > MAX_DETECTION_OUTPUT_BYTES:
            raise EnvironmentDetectionError("environment probe output exceeded the safety limit")
        return stdout

    def detect(
        self,
        profile: EnvironmentProfile,
        *,
        run_directory: Path | str | None = None,
    ) -> EnvironmentReport:
        if not isinstance(profile, EnvironmentProfile):
            raise TypeError("environment detector requires an EnvironmentProfile")
        if profile.mode == "local":
            probe_run_directory = (
                Path(run_directory).absolute()
                if run_directory is not None
                else self.local_run_directory
            )
        else:
            probe_run_directory = str(run_directory) if run_directory is not None else ""
        probe_prefix = (
            "import os; os.environ['MOLLY_PROBE_TIMEOUT_SECONDS'] = "
            + repr(str(self.timeout_seconds + 0.5))
            + "\n"
        )
        if probe_run_directory:
            probe_prefix += (
                "os.environ['MOLLY_PROBE_RUN_DIRECTORY'] = "
                + repr(str(probe_run_directory))
                + "\n"
            )
        if profile.mode == "local":
            output = self._run((sys.executable, "-c", probe_prefix + _PROBE_SCRIPT))
        else:
            output = None
            last_error: Exception | None = None
            for interpreter in ("python3", "python"):
                try:
                    output = self._run(
                        self._ssh_argv(profile, interpreter),
                        input_bytes=(probe_prefix + _PROBE_SCRIPT + "\n").encode("utf-8"),
                    )
                    break
                except EnvironmentDetectionError as exc:
                    last_error = exc
            if output is None:
                raise EnvironmentDetectionError("SSH environment probe failed") from last_error
        try:
            raw = json.loads(output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EnvironmentDetectionError("environment probe returned invalid JSON") from exc
        if not isinstance(raw, Mapping):
            raise EnvironmentDetectionError("environment probe returned an invalid object")
        return EnvironmentReport.from_probe(profile, raw)

    def detect_for_runtime(
        self,
        profile: EnvironmentProfile,
        runtime_directory: Path | str,
    ) -> EnvironmentReport:
        """Probe a server-owned staged/enabled runtime directory."""

        return self.detect(profile, run_directory=runtime_directory)


class EnvironmentManager:
    """Coordinate persisted profiles, read-only detection, and matching."""

    def __init__(
        self,
        root: Path | str,
        *,
        store: EnvironmentConfigStore | None = None,
        detector: EnvironmentDetector | None = None,
    ) -> None:
        self.store = store or EnvironmentConfigStore(root)
        self.detector = detector or EnvironmentDetector(
            local_run_directory=self.store.root / "runtimes"
        )

    def _current_detection(self, profile: EnvironmentProfile) -> dict[str, Any] | None:
        detection = self.store.get_detection(profile.environment_ref)
        if not isinstance(detection, Mapping):
            return None
        report = detection.get("report")
        if not isinstance(report, Mapping) or (
            report.get("environment_ref") != profile.environment_ref
            or report.get("connection_digest") != profile.connection_digest
        ):
            return None
        return dict(detection)

    def list_public(self) -> list[dict[str, Any]]:
        profiles = self.store.list_profiles()
        return [
            profile.to_public_dict(detection=self._current_detection(profile))
            for profile in profiles
        ]

    def upsert_profile(self, payload: Mapping[str, Any]) -> EnvironmentProfile:
        return self.store.upsert_profile(payload)

    def get_public(self, environment_ref: str) -> dict[str, Any]:
        profile = self.store.get_profile(environment_ref)
        detection = self._current_detection(profile)
        return {
            "environment": profile.to_public_dict(detection=detection),
            "detection": detection,
            "read_only": True,
            "installation_enabled": False,
            "installation_available": True,
        }

    def detect(self, environment_ref: str) -> dict[str, Any]:
        profile = self.store.get_profile(environment_ref)
        report = self.detector.detect(profile)
        if (
            not isinstance(report, EnvironmentReport)
            or report.environment_ref != profile.environment_ref
            or report.connection_digest != profile.connection_digest
        ):
            raise EnvironmentDetectionError("environment detector returned an invalid report")
        match = match_environment(profile, report)
        detection = {
            "environment": profile.to_public_dict(),
            "report": report.to_dict(),
            "match": match,
            "read_only": True,
            "installation_enabled": False,
            "installation_available": True,
            "probe_names": list(READ_ONLY_PROBE_NAMES),
        }
        self.store.save_detection(
            environment_ref,
            detection,
            expected_connection_digest=profile.connection_digest,
        )
        detection["environment"] = profile.to_public_dict(detection=detection)
        return detection


__all__ = [
    "COMPATIBILITY_CATALOG",
    "COMPATIBILITY_CATALOG_VERSION",
    "CompatibilityEntry",
    "DEFAULT_SSH_PORT",
    "DETECTION_TIMEOUT_SECONDS",
    "EnvironmentConfigError",
    "EnvironmentConfigStore",
    "EnvironmentDetectionError",
    "EnvironmentDetector",
    "EnvironmentManager",
    "EnvironmentProfile",
    "EnvironmentReport",
    "READ_ONLY_PROBE_NAMES",
    "match_environment",
]
