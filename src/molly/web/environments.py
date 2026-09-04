"""Server-owned connection profiles and read-only runtime discovery.

This module deliberately stops at discovery and planning.  It does not expose
an arbitrary command endpoint and it never installs packages, downloads model
weights, or changes a local or remote environment.  SSH is only used as a
transport for one fixed Python probe script; the connection profile cannot
provide a command, shell fragment, private key, or download URL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from textwrap import dedent
from typing import Any

from molly.core.ids import (
    canonical_json_bytes,
    normalize_timestamp,
    sha256_bytes,
    utc_timestamp,
    validate_identifier,
)


ENVIRONMENT_CONFIG_VERSION = 1
ENVIRONMENT_REPORT_VERSION = 1
COMPATIBILITY_CATALOG_VERSION = "2026.09"
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
    raw = "local" if mode == "local" else f"ssh-{user}-{target}-{port}"
    slug = re.sub(r"[^A-Za-z0-9._:-]+", "-", raw).strip("-")[:110]
    return validate_identifier(f"environment:{slug or 'profile'}", field="environment_ref")


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_ref": self.environment_ref,
            "display_name": self.display_name,
            "mode": self.mode,
            "ssh_target": self.ssh_target,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentProfile":
        if not isinstance(value, Mapping):
            raise EnvironmentConfigError("environment profile must be an object")
        try:
            return cls(
                environment_ref=value["environment_ref"],
                display_name=value["display_name"],
                mode=value["mode"],
                ssh_target=value.get("ssh_target"),
                ssh_user=value.get("ssh_user"),
                ssh_port=value.get("ssh_port"),
                created_at=value.get("created_at", ""),
                updated_at=value.get("updated_at", ""),
            )
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
        profiles = self._read_profiles()
        requested_ref = payload.get("environment_ref") if isinstance(payload, Mapping) else None
        existing = profiles.get(requested_ref) if isinstance(requested_ref, str) else None
        profile = EnvironmentProfile.from_payload(
            payload,
            environment_ref=existing.environment_ref if existing else requested_ref,
            created_at=existing.created_at if existing else None,
        )
        if profile.environment_ref not in profiles and len(profiles) >= MAX_ENVIRONMENT_PROFILES:
            raise EnvironmentConfigError("too many environment profiles")
        profiles[profile.environment_ref] = profile
        self._write_json(
            self.profiles_path,
            {
                "version": ENVIRONMENT_CONFIG_VERSION,
                "profiles": {key: value.to_dict() for key, value in profiles.items()},
            },
        )
        if existing is not None and (
            existing.mode,
            existing.ssh_target,
            existing.ssh_user,
            existing.ssh_port,
        ) != (
            profile.mode,
            profile.ssh_target,
            profile.ssh_user,
            profile.ssh_port,
        ):
            self.clear_detection(profile.environment_ref)
        return profile

    def get_detection(self, environment_ref: str) -> dict[str, Any] | None:
        return self._read_reports().get(environment_ref)

    def save_detection(self, environment_ref: str, detection: Mapping[str, Any]) -> None:
        self.get_profile(environment_ref)
        if not isinstance(detection, Mapping):
            raise EnvironmentConfigError("environment detection must be an object")
        try:
            canonical_json_bytes(detection)
        except (TypeError, ValueError) as exc:
            raise EnvironmentConfigError("environment detection is not JSON serializable") from exc
        reports = self._read_reports()
        reports[environment_ref] = dict(detection)
        self._write_json(
            self.reports_path,
            {"version": ENVIRONMENT_REPORT_VERSION, "reports": reports},
        )

    def clear_detection(self, environment_ref: str) -> None:
        reports = self._read_reports()
        if environment_ref not in reports:
            return
        del reports[environment_ref]
        self._write_json(
            self.reports_path,
            {"version": ENVIRONMENT_REPORT_VERSION, "reports": reports},
        )


CommandRunner = Callable[[Sequence[str], bytes | None, float], tuple[int, bytes]]


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
            )
        except OSError as exc:
            raise EnvironmentDetectionError(
                "fixed environment probe could not start"
            ) from exc
        try:
            stdout, _ = await asyncio.wait_for(
                process.communicate(input_bytes), timeout=timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.communicate()
            raise EnvironmentDetectionError(
                "fixed environment probe timed out"
            ) from exc
        stdout = bytes(stdout or b"")
        if len(stdout) > MAX_DETECTION_OUTPUT_BYTES:
            raise EnvironmentDetectionError("environment probe output exceeded the safety limit")
        return process.returncode or 0, stdout

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
    import shutil
    import subprocess
    import sys

    def text(value, limit=256):
        value = str(value or "")
        return value.replace("\x00", "")[:limit]

    def command(name, args=("--version",), timeout=5):
        path = shutil.which(name)
        if not path:
            return {"available": False, "version": "", "path": ""}
        try:
            result = subprocess.run(
                [path, *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout, check=False, shell=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {"available": False, "version": "", "path": ""}
        output = (result.stdout or result.stderr or b"").decode("utf-8", "replace")
        return {
            "available": result.returncode == 0,
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
        try:
            result = subprocess.run(
                [nvidia_path, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, check=False, shell=False,
            )
            if result.returncode == 0:
                for line in result.stdout.decode("utf-8", "replace").splitlines()[:8]:
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) < 3:
                        continue
                    try:
                        memory = int(float(fields[1]))
                    except ValueError:
                        memory = 0
                    devices.append({"name": text(fields[0], 120), "memory_mib": max(0, memory), "driver_version": text(fields[2], 80)})
        except (OSError, subprocess.TimeoutExpired):
            pass
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
        "python": {"executable": text(sys.executable, 512), "version": text(platform.python_version(), 80), "implementation": text(platform.python_implementation(), 80), "managers": python_tools},
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


def _normalize_probe(raw: Mapping[str, Any]) -> dict[str, Any]:
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
        weights.append(
            {
                "name": _clean_string(item.get("name"), maximum=180),
                "path": _clean_string(item.get("path")),
                "size_bytes": _clean_int(item.get("size_bytes"), maximum=2**50),
            }
        )
    managers = python_raw.get("managers", {}) if isinstance(python_raw.get("managers"), Mapping) else {}
    cuda_raw = gpu_raw.get("cuda") if isinstance(gpu_raw.get("cuda"), Mapping) else {}
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
        },
        "unimol": {
            "installed": bool(unimol_raw.get("installed")),
            "importable": bool(unimol_raw.get("importable")),
            "package": _clean_string(unimol_raw.get("package"), maximum=80),
            "version": _clean_string(unimol_raw.get("version"), maximum=80),
        },
        "reinvent4": {
            "installed": bool(reinvent_raw.get("installed")),
            "importable": bool(reinvent_raw.get("importable")),
            "package": _clean_string(reinvent_raw.get("package"), maximum=80),
            "version": _clean_string(reinvent_raw.get("version"), maximum=80),
            "repositories": repositories[:8],
            "license_present": bool(reinvent_raw.get("license_present")),
        },
        "weights": {
            "entries": weights[:128],
            "total_bytes": _clean_int(weights_raw.get("total_bytes"), maximum=2**52),
        },
    }


@dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """Bounded, sanitized output of the fixed discovery probe."""

    environment_ref: str
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
    ) -> "EnvironmentReport":
        normalized = _normalize_probe(raw)
        timestamp = (
            _timestamp(detected_at, field="environment detected_at")
            if detected_at is not None
            else utc_timestamp()
        )
        payload = {
            "version": ENVIRONMENT_REPORT_VERSION,
            "environment_ref": profile.environment_ref,
            "mode": profile.mode,
            "target_label": profile.target_label,
            "detected_at": timestamp,
            "probes": list(READ_ONLY_PROBE_NAMES),
            **normalized,
        }
        digest = sha256_bytes(canonical_json_bytes(payload))
        return cls(
            environment_ref=profile.environment_ref,
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
    return any("unimol" in str(item.get("name", "")).casefold() for item in entries if isinstance(item, Mapping))


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
    entries = {entry.component_id: entry for entry in catalog}
    python_ready = bool(python_data.get("executable")) and _python_matches(
        str(python_data.get("version", ""))
    )
    unimol_ready = bool(unimol.get("installed")) and bool(unimol.get("importable")) and _version_matches(
        str(unimol.get("version", "")), entries["unimol"].version
    )
    repositories = reinvent.get("repositories", ())
    reinvent_ready = (
        bool(reinvent.get("installed"))
        and bool(reinvent.get("importable"))
        and _version_matches(str(reinvent.get("version", "")), entries["reinvent4"].version)
        and _repository_ready(repositories if isinstance(repositories, Sequence) else ())
    )
    weights_ready = _weights_ready(weights, unimol_ready=unimol_ready)
    directory_ready = bool(disk.get("exists")) and bool(disk.get("writable"))
    license_attention = bool(reinvent.get("installed")) and not bool(reinvent.get("license_present"))

    reusable: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    if python_ready:
        reusable.append({"component_id": "python", "name": "Python", "version": python_data.get("version", "")})
    else:
        missing.append({"component_id": "python", "reason": "未找到兼容的 Python 3.10+ 解释器"})
    if isinstance(python_data.get("managers"), Mapping):
        for manager in ("conda", "mamba", "uv"):
            value = python_data["managers"].get(manager)
            if isinstance(value, Mapping) and value.get("available"):
                reusable.append({"component_id": manager, "name": manager.title(), "version": value.get("version", "")})
    if unimol_ready:
        reusable.append({"component_id": "unimol", "name": "Uni-Mol", "version": unimol.get("version", "")})
    else:
        missing.append({"component_id": "unimol", "reason": "未找到兼容的 Uni-Mol 版本"})
    if reinvent_ready:
        reusable.append({"component_id": "reinvent4", "name": "REINVENT4", "version": reinvent.get("version", "")})
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
        "integrity_policy": "安装阶段必须按兼容性清单校验固定 SHA-256；本 PR 不下载文件",
        "items": plan_items,
        "estimated_download_bytes": sum(int(item.get("estimated_download_bytes", 0)) for item in plan_items),
        "estimated_disk_bytes": sum(int(item.get("estimated_disk_bytes", 0)) for item in plan_items),
        "estimated_duration_seconds": sum(int(item.get("estimated_duration_seconds", 0)) for item in plan_items),
        "notes": [
            "本次仅生成安装预览，不会下载、安装、覆盖或修改现有环境。",
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
        "compatibility": {
            "python": {"compatible": python_ready, "detected_version": python_data.get("version", ""), "required": "Python >=3.10"},
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
            profile.ssh_target,
            "--",
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

    def detect(self, profile: EnvironmentProfile) -> EnvironmentReport:
        if not isinstance(profile, EnvironmentProfile):
            raise TypeError("environment detector requires an EnvironmentProfile")
        if profile.mode == "local":
            prefix = "import os; os.environ['MOLLY_PROBE_RUN_DIRECTORY'] = " + repr(
                str(self.local_run_directory)
            ) + "\n"
            output = self._run((sys.executable, "-c", prefix + _PROBE_SCRIPT))
        else:
            output = None
            last_error: Exception | None = None
            for interpreter in ("python3", "python"):
                try:
                    output = self._run(
                        self._ssh_argv(profile, interpreter),
                        input_bytes=(_PROBE_SCRIPT + "\n").encode("utf-8"),
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

    def list_public(self) -> list[dict[str, Any]]:
        profiles = self.store.list_profiles()
        return [
            profile.to_public_dict(detection=self.store.get_detection(profile.environment_ref))
            for profile in profiles
        ]

    def upsert_profile(self, payload: Mapping[str, Any]) -> EnvironmentProfile:
        return self.store.upsert_profile(payload)

    def get_public(self, environment_ref: str) -> dict[str, Any]:
        profile = self.store.get_profile(environment_ref)
        detection = self.store.get_detection(environment_ref)
        return {
            "environment": profile.to_public_dict(detection=detection),
            "detection": detection,
            "read_only": True,
            "installation_enabled": False,
        }

    def detect(self, environment_ref: str) -> dict[str, Any]:
        profile = self.store.get_profile(environment_ref)
        report = self.detector.detect(profile)
        if not isinstance(report, EnvironmentReport) or report.environment_ref != profile.environment_ref:
            raise EnvironmentDetectionError("environment detector returned an invalid report")
        match = match_environment(profile, report)
        detection = {
            "environment": profile.to_public_dict(),
            "report": report.to_dict(),
            "match": match,
            "read_only": True,
            "installation_enabled": False,
            "probe_names": list(READ_ONLY_PROBE_NAMES),
        }
        self.store.save_detection(environment_ref, detection)
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
