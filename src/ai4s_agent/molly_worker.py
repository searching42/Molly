from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterator, Mapping, Sequence, TextIO

from platformdirs import user_config_path, user_data_path

from ai4s_agent._utils import now_iso
from ai4s_agent.remote_execution_lifecycle import (
    RemoteExecutionApproval,
    RemoteExecutionRequest,
    RemoteObservation,
    RemoteOutputArtifact,
    RemotePublication,
)
from ai4s_agent.remote_output_contracts import (
    verify_remote_output_contents,
    verify_remote_output_contract,
)
from ai4s_agent.resource_profiles import EXECUTION_PROFILES, ExecutionProfile

try:  # Python 3.11+ ships the TOML parser in the standard library.
    import tomllib
except ImportError:  # pragma: no cover - exercised by Python 3.10 deployments.
    import tomli as tomllib  # type: ignore[no-redef]

try:  # pragma: no cover - the supported deployment target is POSIX.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


WORKER_CONFIG_SCHEMA = "molly_worker_config.v1"
WORKER_STATE_SCHEMA = "molly_worker_state.v1"
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
_TERMINAL_STATUSES = {"SUCCEEDED", "FAILED", "CANCELLED"}
_ACTIVE_STATUSES = {"ACCEPTED", "RUNNING", "CANCEL_REQUESTED"}
_DEFAULT_TERMINATION_GRACE_SEC = 2.0
_REINVENT_PLACEHOLDERS = {
    "{{molly_output_csv}}",
    "{{molly_design_request_id}}",
    "{{molly_seed}}",
    "{{molly_design_request_sha256}}",
}


class WorkerProtocolError(RuntimeError):
    """A bounded failure that can cross the worker CLI boundary safely."""

    def __init__(self, code: str) -> None:
        clean = str(code or "").strip().lower()
        if not _SAFE_ID.fullmatch(clean):
            clean = "worker_protocol_error"
        super().__init__(clean)
        self.code = clean


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "_FileIdentity":
        if not stat.S_ISREG(metadata.st_mode):
            raise WorkerProtocolError("unsafe_worker_file")
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )


@dataclass(frozen=True)
class _AttemptInputs:
    root: Path
    paths: Mapping[str, Path]
    identities: Mapping[str, _FileIdentity]
    digests: Mapping[str, str]


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _open_regular_no_follow(path: Path) -> int:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise WorkerProtocolError("unsafe_worker_file") from exc
    try:
        _FileIdentity.from_stat(os.fstat(descriptor))
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _path_regular_identity(path: Path) -> _FileIdentity:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise WorkerProtocolError("unsafe_worker_file") from exc
    return _FileIdentity.from_stat(metadata)


def _descriptor_digest(descriptor: int) -> tuple[_FileIdentity, str]:
    initial = _FileIdentity.from_stat(os.fstat(descriptor))
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        received = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            received += len(chunk)
            digest.update(chunk)
        final = _FileIdentity.from_stat(os.fstat(descriptor))
        if initial != final or received != initial.size:
            raise WorkerProtocolError("worker_file_changed")
        return initial, "sha256:" + digest.hexdigest()
    finally:
        with contextlib.suppress(OSError):
            os.lseek(descriptor, 0, os.SEEK_SET)


def _digest_file(path: Path) -> tuple[int, str]:
    descriptor = _open_regular_no_follow(path)
    try:
        identity, digest = _descriptor_digest(descriptor)
        return identity.size, digest
    finally:
        os.close(descriptor)


def _safe_identifier(value: Any, *, field: str) -> str:
    clean = str(value or "")
    if clean != clean.strip().lower() or not _SAFE_ID.fullmatch(clean):
        raise WorkerProtocolError(f"invalid_{field}")
    return clean


def _safe_digest(value: Any, *, field: str) -> str:
    clean = str(value or "")
    if not _SHA256.fullmatch(clean):
        raise WorkerProtocolError(f"invalid_{field}")
    return clean


def _safe_relative_path(value: Any) -> str:
    clean = str(value or "")
    path = PurePosixPath(clean)
    if (
        not clean
        or clean != path.as_posix()
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or "\\" in clean
        or any(ord(char) < 32 for char in clean)
    ):
        raise WorkerProtocolError("invalid_relative_path")
    return clean


def _absolute_path(value: Any, *, field: str, required: bool = False) -> Path | None:
    clean = str(value or "").strip()
    if not clean:
        if required:
            raise WorkerProtocolError(f"missing_{field}")
        return None
    candidate = Path(clean).expanduser()
    if (
        not candidate.is_absolute()
        or ".." in candidate.parts
        or any(ord(char) < 32 for char in clean)
    ):
        raise WorkerProtocolError(f"invalid_{field}")
    return candidate


def _read_json_stream(stream: BinaryIO, *, max_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    payload = stream.read(max_bytes + 1)
    if not payload or len(payload) > max_bytes:
        raise WorkerProtocolError("invalid_json_input")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("invalid_json_input") from exc
    if not isinstance(decoded, dict):
        raise WorkerProtocolError("invalid_json_input")
    return decoded


def _read_json_file(path: Path, *, max_bytes: int = _MAX_JSON_BYTES) -> dict[str, Any]:
    descriptor = _open_regular_no_follow(path)
    try:
        initial = _FileIdentity.from_stat(os.fstat(descriptor))
        if initial.size > max_bytes:
            raise WorkerProtocolError("invalid_worker_json")
        payload = b""
        while len(payload) <= max_bytes:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload += chunk
        final = _FileIdentity.from_stat(os.fstat(descriptor))
        if initial != final or len(payload) != initial.size:
            raise WorkerProtocolError("worker_file_changed")
    finally:
        os.close(descriptor)
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerProtocolError("invalid_worker_json") from exc
    if not isinstance(decoded, dict):
        raise WorkerProtocolError("invalid_worker_json")
    return decoded


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise WorkerProtocolError("unsafe_worker_path")


def _ensure_private_directory(path: Path) -> Path:
    if not path.is_absolute() or path == Path("/") or ".." in path.parts:
        raise WorkerProtocolError("invalid_worker_root")
    _reject_symlink_components(path)
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = os.lstat(path)
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise WorkerProtocolError("unsafe_worker_path")
    os.chmod(path, 0o700)
    return path


def _write_private_bytes(path: Path, payload: bytes) -> None:
    parent = _ensure_private_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary_path, path)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_private_bytes(path, _canonical_bytes(payload) + b"\n")


@dataclass(frozen=True)
class WorkerSettings:
    root: Path
    reinvent4_repository: Path | None = None
    reinvent4_python: Path | None = None
    unimol_repository: Path | None = None
    unimol_python: Path | None = None

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "WorkerSettings":
        source = dict(os.environ if env is None else env)
        configured_path = source.get("MOLLY_WORKER_CONFIG")
        config_path = (
            _absolute_path(configured_path, field="worker_config", required=True)
            if configured_path
            else user_config_path("Molly", appauthor=False) / "worker.json"
        )
        payload: dict[str, Any] = {}
        if config_path.is_file():
            _reject_symlink_components(config_path.parent)
            metadata = os.lstat(config_path)
            if stat.S_ISLNK(metadata.st_mode) or metadata.st_mode & 0o077:
                raise WorkerProtocolError("unsafe_worker_config")
            payload = _read_json_file(config_path)
            if payload.get("schema_version") != WORKER_CONFIG_SCHEMA:
                raise WorkerProtocolError("unsupported_worker_config")
            allowed = {
                "schema_version",
                "root",
                "reinvent4_repository",
                "reinvent4_python",
                "unimol_repository",
                "unimol_python",
            }
            if set(payload).difference(allowed):
                raise WorkerProtocolError("invalid_worker_config")
        root_value = source.get("MOLLY_WORKER_ROOT") or payload.get("root")
        root = (
            _absolute_path(root_value, field="worker_root", required=True)
            if root_value
            else user_data_path("Molly", appauthor=False) / "worker"
        )
        return cls(
            root=root,
            reinvent4_repository=_absolute_path(
                source.get("MOLLY_WORKER_REINVENT4_REPOSITORY")
                or payload.get("reinvent4_repository"),
                field="reinvent4_repository",
            ),
            reinvent4_python=_absolute_path(
                source.get("MOLLY_WORKER_REINVENT4_PYTHON")
                or payload.get("reinvent4_python"),
                field="reinvent4_python",
            ),
            unimol_repository=_absolute_path(
                source.get("MOLLY_WORKER_UNIMOL_REPOSITORY")
                or payload.get("unimol_repository"),
                field="unimol_repository",
            ),
            unimol_python=_absolute_path(
                source.get("MOLLY_WORKER_UNIMOL_PYTHON")
                or payload.get("unimol_python"),
                field="unimol_python",
            ),
        )


class WorkerStore:
    def __init__(self, root: Path) -> None:
        self.root = _ensure_private_directory(Path(root))
        self.jobs_root = _ensure_private_directory(self.root / "jobs")

    def job_dir(self, request_id: str, *, create: bool = False) -> Path:
        clean = _safe_identifier(request_id, field="request_id")
        path = self.jobs_root / clean
        if create:
            _ensure_private_directory(path)
            for name in ("inputs", "outputs", "work", "logs"):
                _ensure_private_directory(path / name)
        else:
            try:
                metadata = os.lstat(path)
            except FileNotFoundError as exc:
                raise WorkerProtocolError("request_not_staged") from exc
            if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                raise WorkerProtocolError("unsafe_worker_path")
        return path

    @contextlib.contextmanager
    def lock(self, request_id: str) -> Iterator[Path]:
        if fcntl is None:  # pragma: no cover
            raise WorkerProtocolError("worker_lock_unavailable")
        job_dir = self.job_dir(request_id, create=False)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(job_dir / ".lock", flags, 0o600)
        except OSError as exc:
            raise WorkerProtocolError("unsafe_worker_lock") from exc
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield job_dir
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def envelope_path(self, request_id: str) -> Path:
        return self.job_dir(request_id) / "envelope.json"

    def state_path(self, request_id: str) -> Path:
        return self.job_dir(request_id) / "state.json"

    def read_envelope(self, request_id: str) -> dict[str, Any]:
        return _read_json_file(self.envelope_path(request_id))

    def read_state(self, request_id: str) -> dict[str, Any]:
        payload = _read_json_file(self.state_path(request_id))
        if payload.get("schema_version") != WORKER_STATE_SCHEMA:
            raise WorkerProtocolError("invalid_worker_state")
        return payload

    def write_state(
        self,
        request: RemoteExecutionRequest,
        observation: RemoteObservation,
        *,
        pid: int | None = None,
        process_token: str = "",
        adapter_pid: int | None = None,
        adapter_process_token: str = "",
    ) -> None:
        _write_private_json(
            self.state_path(request.request_id),
            {
                "schema_version": WORKER_STATE_SCHEMA,
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "pid": pid,
                "process_token": process_token,
                "adapter_pid": adapter_pid,
                "adapter_process_token": adapter_process_token,
                "observation": observation.model_dump(mode="json"),
            },
        )

    def input_path(
        self,
        request_id: str,
        relative_path: str,
        *,
        create_parents: bool = False,
    ) -> Path:
        clean = _safe_relative_path(relative_path)
        base = self.job_dir(request_id) / "inputs"
        current = base
        parts = PurePosixPath(clean).parts
        for component in parts[:-1]:
            current = current / component
            if create_parents:
                _ensure_private_directory(current)
            else:
                try:
                    metadata = os.lstat(current)
                except FileNotFoundError as exc:
                    raise WorkerProtocolError("staged_input_missing") from exc
                if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                    raise WorkerProtocolError("unsafe_worker_path")
        return current / parts[-1]

    def output_path(
        self,
        request_id: str,
        relative_path: str,
        *,
        create_parents: bool = False,
    ) -> Path:
        clean = _safe_relative_path(relative_path)
        base = self.job_dir(request_id) / "outputs"
        current = base
        parts = PurePosixPath(clean).parts
        for component in parts[:-1]:
            current = current / component
            if create_parents:
                current = _ensure_private_directory(current)
            else:
                try:
                    metadata = os.lstat(current)
                except FileNotFoundError as exc:
                    raise WorkerProtocolError("published_output_missing") from exc
                if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                    raise WorkerProtocolError("unsafe_worker_path")
        return current / parts[-1]


class MollyWorker:
    def __init__(
        self,
        settings: WorkerSettings,
        *,
        popen_factory: Any = subprocess.Popen,
        adapter_popen_factory: Any = subprocess.Popen,
        run_command: Any = subprocess.run,
        termination_grace_sec: float = _DEFAULT_TERMINATION_GRACE_SEC,
        adapter_timeout_sec: float | None = None,
    ) -> None:
        self.settings = settings
        self._store: WorkerStore | None = None
        self.popen_factory = popen_factory
        self.adapter_popen_factory = adapter_popen_factory
        self.run_command = run_command
        self.termination_grace_sec = max(0.05, float(termination_grace_sec))
        self.adapter_timeout_sec = (
            None if adapter_timeout_sec is None else max(0.05, float(adapter_timeout_sec))
        )

    @property
    def store(self) -> WorkerStore:
        if self._store is None:
            self._store = WorkerStore(self.settings.root)
        return self._store

    def probe(self) -> dict[str, Any]:
        capabilities = {"cpu"}
        versions: dict[str, str] = {}
        cuda: dict[str, Any] = {
            "status": "unknown",
            "device_name": "",
            "compute_capability": "",
            "driver_version": "",
            "runtime_version": "",
            "toolkit_version": "",
            "pytorch_cuda_version": "",
            "cudnn_version": "",
        }
        gpu = self._probe_gpu()
        if gpu:
            capabilities.add("gpu")
            cuda.update(gpu)
            cuda["status"] = "available"
        else:
            cuda["status"] = "unavailable"

        reinvent_version = self._probe_python_distribution(
            self.settings.reinvent4_python,
            module="reinvent",
            distribution="reinvent",
        )
        if (
            reinvent_version
            and self.settings.reinvent4_repository is not None
            and self.settings.reinvent4_repository.is_dir()
        ):
            capabilities.add("reinvent4")
            versions["reinvent"] = reinvent_version

        unimol_version = self._probe_python_distribution(
            self.settings.unimol_python,
            module="unimol_tools",
            distribution="unimol-tools",
        )
        if (
            unimol_version
            and self.settings.unimol_repository is not None
            and self.settings.unimol_repository.is_dir()
        ):
            capabilities.add("unimol")
            versions["unimol-tools"] = unimol_version
            pytorch_cuda = self._probe_unimol_cuda()
            if pytorch_cuda:
                cuda["pytorch_cuda_version"] = pytorch_cuda

        return {
            "hostname": os.uname().nodename.split(".", 1)[0].lower(),
            "capabilities": sorted(capabilities),
            "details": {
                "cpu_threads": os.cpu_count() or 0,
                "memory_bytes": self._memory_bytes(),
                "cuda": cuda,
                "software_versions": dict(sorted(versions.items())),
            },
        }

    def stage(self, envelope_payload: Mapping[str, Any]) -> dict[str, Any]:
        request, approval = self._validate_envelope(envelope_payload)
        self._validate_execution_profile(request)
        job_dir = self.store.job_dir(request.request_id, create=True)
        canonical = _canonical_bytes(
            {
                "request": request.model_dump(mode="json"),
                "approval": approval.model_dump(mode="json"),
            }
        ) + b"\n"
        envelope_path = job_dir / "envelope.json"
        with self.store.lock(request.request_id):
            try:
                os.lstat(envelope_path)
                envelope_exists = True
            except FileNotFoundError:
                envelope_exists = False
            if envelope_exists:
                existing = _canonical_bytes(self.store.read_envelope(request.request_id)) + b"\n"
                if existing != canonical:
                    raise WorkerProtocolError("staging_binding_mismatch")
            else:
                _write_private_bytes(envelope_path, canonical)
            state_path = job_dir / "state.json"
            try:
                os.lstat(state_path)
                state_exists = True
            except FileNotFoundError:
                state_exists = False
            if not state_exists:
                observation = self._observation(request, status="ACCEPTED")
                self.store.write_state(request, observation)
        return {
            "ok": True,
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
        }

    def stage_input(
        self,
        *,
        request_id: str,
        relative_path: str,
        size_bytes: int,
        sha256: str,
        stream: BinaryIO,
    ) -> dict[str, Any]:
        clean_request_id = _safe_identifier(request_id, field="request_id")
        clean_path = _safe_relative_path(relative_path)
        expected_digest = _safe_digest(sha256, field="sha256")
        if isinstance(size_bytes, bool) or size_bytes < 0:
            raise WorkerProtocolError("invalid_input_size")
        request, _ = self._load_envelope(clean_request_id)
        descriptor = next(
            (
                artifact
                for artifact in request.input_manifest.artifacts
                if artifact.relative_path == clean_path
            ),
            None,
        )
        if (
            descriptor is None
            or descriptor.size_bytes != size_bytes
            or descriptor.sha256 != expected_digest
        ):
            raise WorkerProtocolError("input_manifest_binding_mismatch")
        target = self.store.input_path(
            clean_request_id,
            clean_path,
            create_parents=True,
        )
        with self.store.lock(clean_request_id):
            try:
                target_metadata = os.lstat(target)
                target_exists = True
            except FileNotFoundError:
                target_metadata = None
                target_exists = False
            if target_exists:
                if target_metadata is None or not stat.S_ISREG(target_metadata.st_mode):
                    raise WorkerProtocolError("unsafe_worker_file")
                existing_size, existing_digest = _digest_file(target)
                if existing_size != size_bytes or existing_digest != expected_digest:
                    raise WorkerProtocolError("staged_input_binding_mismatch")
                self._verify_replayed_input_stream(
                    stream,
                    size_bytes=size_bytes,
                    sha256=expected_digest,
                )
            else:
                self._receive_input(
                    target,
                    stream=stream,
                    size_bytes=size_bytes,
                    sha256=expected_digest,
                )
        return {
            "ok": True,
            "request_id": clean_request_id,
            "relative_path": clean_path,
            "size_bytes": size_bytes,
            "sha256": expected_digest,
        }

    def verify_inputs(
        self,
        *,
        request_id: str,
        request_sha256: str,
    ) -> dict[str, Any]:
        clean_request_id = _safe_identifier(request_id, field="request_id")
        clean_digest = _safe_digest(request_sha256, field="request_sha256")
        with self.store.lock(clean_request_id):
            request, _ = self._load_envelope(clean_request_id)
            if request.request_sha256 != clean_digest:
                raise WorkerProtocolError("request_binding_mismatch")
            self._verify_staged_inputs(request)
        return {
            "ok": True,
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "manifest_sha256": request.input_manifest.manifest_sha256,
        }

    def execute(self, envelope_payload: Mapping[str, Any]) -> RemoteObservation:
        request, approval = self._validate_envelope(envelope_payload)
        self._validate_execution_profile(request)
        with self.store.lock(request.request_id):
            stored_request, stored_approval = self._load_envelope(request.request_id)
            if (
                stored_request.request_sha256 != request.request_sha256
                or stored_approval.approval_sha256 != approval.approval_sha256
            ):
                raise WorkerProtocolError("execution_binding_mismatch")
            self._verify_staged_inputs(request)
            state = self.store.read_state(request.request_id)
            observation = RemoteObservation.model_validate(state["observation"])
            if observation.status in _ACTIVE_STATUSES | _TERMINAL_STATUSES and state.get("pid"):
                return self._refresh_locked(request, state)
            if observation.status in _TERMINAL_STATUSES:
                return observation
            self._require_adapter_available(request)
            accepted = self._observation(request, status="ACCEPTED")
            self.store.write_state(request, accepted)
            process = self._spawn_runner(request.request_id)
            token = self._process_token(int(process.pid))
            self.store.write_state(
                request,
                accepted,
                pid=int(process.pid),
                process_token=token,
            )
            return accepted

    def status(
        self,
        *,
        request_id: str,
        request_sha256: str,
    ) -> RemoteObservation:
        clean_request_id = _safe_identifier(request_id, field="request_id")
        clean_digest = _safe_digest(request_sha256, field="request_sha256")
        with self.store.lock(clean_request_id):
            request, _ = self._load_envelope(clean_request_id)
            if request.request_sha256 != clean_digest:
                raise WorkerProtocolError("request_binding_mismatch")
            state = self.store.read_state(clean_request_id)
            return self._refresh_locked(request, state)

    def cancel(
        self,
        *,
        request_id: str,
        request_sha256: str,
    ) -> RemoteObservation:
        clean_request_id = _safe_identifier(request_id, field="request_id")
        clean_digest = _safe_digest(request_sha256, field="request_sha256")
        with self.store.lock(clean_request_id):
            request, _ = self._load_envelope(clean_request_id)
            if request.request_sha256 != clean_digest:
                raise WorkerProtocolError("request_binding_mismatch")
            state = self.store.read_state(clean_request_id)
            current = RemoteObservation.model_validate(state["observation"])
            if current.status in {"SUCCEEDED", "CANCELLED"}:
                return current
            runner_pid = self._state_pid(state, field="pid")
            runner_token = str(state.get("process_token") or "")
            adapter_pid = self._state_pid(state, field="adapter_pid")
            adapter_token = str(state.get("adapter_process_token") or "")
            if current.status == "FAILED":
                self._terminate_bound_process_group(adapter_pid, adapter_token)
                self._terminate_bound_process_group(runner_pid, runner_token)
                return current
            cancelled = self._observation(request, status="CANCEL_REQUESTED")
            self.store.write_state(
                request,
                cancelled,
                pid=runner_pid,
                process_token=runner_token,
                adapter_pid=adapter_pid,
                adapter_process_token=adapter_token,
            )
            self._terminate_bound_process_group(adapter_pid, adapter_token)
            self._terminate_bound_process_group(runner_pid, runner_token)
            terminal = self._observation(request, status="CANCELLED")
            self.store.write_state(request, terminal)
            return terminal

    def fetch_output(
        self,
        *,
        request_id: str,
        relative_path: str,
        size_bytes: int,
        sha256: str,
        destination: BinaryIO,
    ) -> None:
        clean_request_id = _safe_identifier(request_id, field="request_id")
        clean_path = _safe_relative_path(relative_path)
        clean_digest = _safe_digest(sha256, field="sha256")
        if isinstance(size_bytes, bool) or size_bytes < 0:
            raise WorkerProtocolError("invalid_output_size")
        with self.store.lock(clean_request_id):
            state = self.store.read_state(clean_request_id)
            observation = RemoteObservation.model_validate(state["observation"])
            if observation.status != "SUCCEEDED" or observation.publication is None:
                raise WorkerProtocolError("output_not_published")
            artifact = next(
                (
                    item
                    for item in observation.publication.artifacts
                    if item.relative_path == clean_path
                ),
                None,
            )
            if (
                artifact is None
                or artifact.size_bytes != size_bytes
                or artifact.sha256 != clean_digest
            ):
                raise WorkerProtocolError("output_binding_mismatch")
            source = self.store.output_path(clean_request_id, clean_path)
            descriptor = _open_regular_no_follow(source)
            try:
                initial = _FileIdentity.from_stat(os.fstat(descriptor))
                if initial.size != size_bytes:
                    raise WorkerProtocolError("output_content_mismatch")
                digest = hashlib.sha256()
                remaining = size_bytes
                while remaining:
                    chunk = os.read(descriptor, min(64 * 1024, remaining))
                    if not chunk:
                        raise WorkerProtocolError("output_content_mismatch")
                    digest.update(chunk)
                    destination.write(chunk)
                    remaining -= len(chunk)
                if os.read(descriptor, 1):
                    raise WorkerProtocolError("output_content_mismatch")
                final = _FileIdentity.from_stat(os.fstat(descriptor))
                path_identity = _path_regular_identity(source)
                if (
                    final != initial
                    or path_identity != initial
                    or "sha256:" + digest.hexdigest() != clean_digest
                ):
                    raise WorkerProtocolError("output_content_mismatch")
            finally:
                os.close(descriptor)

    def run_job(self, request_id: str) -> int:
        clean_request_id = _safe_identifier(request_id, field="request_id")
        try:
            with self.store.lock(clean_request_id):
                request, approval = self._load_envelope(clean_request_id)
                state = self.store.read_state(clean_request_id)
                current = RemoteObservation.model_validate(state["observation"])
                if current.status in {"CANCEL_REQUESTED", "CANCELLED"}:
                    terminal = self._observation(request, status="CANCELLED")
                    self.store.write_state(request, terminal)
                    return 1
                if current.status in {"SUCCEEDED", "FAILED"}:
                    return 0 if current.status == "SUCCEEDED" else 1
                pid = os.getpid()
                token = self._process_token(pid)
                running = self._observation(request, status="RUNNING")
                self.store.write_state(
                    request,
                    running,
                    pid=pid,
                    process_token=token,
                )
            attempt_inputs = self._snapshot_verified_inputs(request)
            self._verify_attempt_inputs(attempt_inputs)
            self._execute_adapter(request, attempt_inputs)
            self._verify_attempt_inputs(attempt_inputs)
            publication = self._build_publication(request, approval)
            succeeded = self._observation(
                request,
                status="SUCCEEDED",
                publication=publication,
            )
            with self.store.lock(clean_request_id):
                current = RemoteObservation.model_validate(
                    self.store.read_state(clean_request_id)["observation"]
                )
                if current.status in {"CANCEL_REQUESTED", "CANCELLED"}:
                    cancelled = self._observation(request, status="CANCELLED")
                    self.store.write_state(request, cancelled)
                    return 1
                self.store.write_state(request, succeeded)
            return 0
        except BaseException as exc:
            code = exc.code if isinstance(exc, WorkerProtocolError) else "adapter_failed"
            traceback.print_exc(file=sys.stderr)
            with contextlib.suppress(Exception):
                with self.store.lock(clean_request_id):
                    request, _ = self._load_envelope(clean_request_id)
                    current = RemoteObservation.model_validate(
                        self.store.read_state(clean_request_id)["observation"]
                    )
                    if current.status in {"CANCEL_REQUESTED", "CANCELLED"}:
                        terminal = self._observation(request, status="CANCELLED")
                    else:
                        terminal = self._observation(
                            request,
                            status="FAILED",
                            error_code=code,
                        )
                    self.store.write_state(request, terminal)
            return 1

    def _validate_envelope(
        self,
        payload: Mapping[str, Any],
    ) -> tuple[RemoteExecutionRequest, RemoteExecutionApproval]:
        if set(payload) != {"request", "approval"}:
            raise WorkerProtocolError("invalid_execution_envelope")
        try:
            request = RemoteExecutionRequest.model_validate(payload["request"])
            approval = RemoteExecutionApproval.model_validate(payload["approval"])
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError("invalid_execution_envelope") from exc
        if (
            approval.request_id != request.request_id
            or approval.request_sha256 != request.request_sha256
        ):
            raise WorkerProtocolError("approval_binding_mismatch")
        return request, approval

    def _load_envelope(
        self,
        request_id: str,
    ) -> tuple[RemoteExecutionRequest, RemoteExecutionApproval]:
        return self._validate_envelope(self.store.read_envelope(request_id))

    def _validate_execution_profile(self, request: RemoteExecutionRequest) -> ExecutionProfile:
        profile = EXECUTION_PROFILES.get(request.execution_profile_id)
        if (
            profile is None
            or profile.digest() != request.execution_profile_digest
            or profile.output_contract != request.output_contract
            or profile.profile_id != request.input_manifest.execution_profile_id
        ):
            raise WorkerProtocolError("execution_profile_binding_mismatch")
        resources = request.requested_resources
        limits = profile.resource_limits
        if (
            resources.gpu_count > limits.gpu_count_max
            or resources.cpu_threads > limits.cpu_threads_max
            or resources.walltime_sec > limits.walltime_sec_max
            or (profile.device_policy == "cpu_only" and resources.gpu_count != 0)
            or (profile.device_policy == "gpu_required" and resources.gpu_count < 1)
        ):
            raise WorkerProtocolError("resource_limit_exceeded")
        return profile

    def _receive_input(
        self,
        target: Path,
        *,
        stream: BinaryIO,
        size_bytes: int,
        sha256: str,
    ) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags, 0o600)
        except OSError as exc:
            raise WorkerProtocolError("unsafe_worker_file") from exc
        digest = hashlib.sha256()
        remaining = size_bytes
        try:
            while remaining:
                chunk = stream.read(min(64 * 1024, remaining))
                if not chunk:
                    raise WorkerProtocolError("input_size_mismatch")
                if not isinstance(chunk, (bytes, bytearray)):
                    raise WorkerProtocolError("invalid_input_stream")
                digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    view = view[written:]
                remaining -= len(chunk)
            if stream.read(1):
                raise WorkerProtocolError("input_size_mismatch")
            if "sha256:" + digest.hexdigest() != sha256:
                raise WorkerProtocolError("input_digest_mismatch")
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            descriptor = -1
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _verify_replayed_input_stream(
        stream: BinaryIO,
        *,
        size_bytes: int,
        sha256: str,
    ) -> None:
        digest = hashlib.sha256()
        remaining = size_bytes
        while remaining:
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                raise WorkerProtocolError("input_size_mismatch")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1):
            raise WorkerProtocolError("input_size_mismatch")
        if "sha256:" + digest.hexdigest() != sha256:
            raise WorkerProtocolError("input_digest_mismatch")

    def _verify_staged_inputs(self, request: RemoteExecutionRequest) -> None:
        expected = {artifact.relative_path: artifact for artifact in request.input_manifest.artifacts}
        inputs_root = self.store.job_dir(request.request_id) / "inputs"
        observed: set[str] = set()
        for root, directories, files in os.walk(inputs_root, followlinks=False):
            root_path = Path(root)
            for name in directories:
                metadata = os.lstat(root_path / name)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise WorkerProtocolError("unsafe_worker_path")
            for name in files:
                path = root_path / name
                metadata = os.lstat(path)
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                    raise WorkerProtocolError("unsafe_worker_path")
                relative = path.relative_to(inputs_root).as_posix()
                descriptor = expected.get(relative)
                if descriptor is None:
                    raise WorkerProtocolError("unexpected_staged_input")
                size, digest = _digest_file(path)
                if size != descriptor.size_bytes or digest != descriptor.sha256:
                    raise WorkerProtocolError("staged_input_binding_mismatch")
                observed.add(relative)
        if observed != set(expected):
            raise WorkerProtocolError("staged_input_missing")

    def _snapshot_verified_inputs(
        self,
        request: RemoteExecutionRequest,
    ) -> _AttemptInputs:
        """Copy each manifest-bound fd into a private, read-only attempt tree."""

        self._verify_staged_inputs(request)
        job_dir = self.store.job_dir(request.request_id)
        attempts_root = _ensure_private_directory(job_dir / "work" / "attempts")
        attempt_root = Path(tempfile.mkdtemp(prefix="attempt-", dir=attempts_root))
        os.chmod(attempt_root, 0o700)
        snapshot_root = _ensure_private_directory(attempt_root / "inputs")
        paths: dict[str, Path] = {}
        identities: dict[str, _FileIdentity] = {}
        digests: dict[str, str] = {}
        try:
            for artifact in sorted(
                request.input_manifest.artifacts,
                key=lambda item: item.relative_path,
            ):
                source = self.store.input_path(
                    request.request_id,
                    artifact.relative_path,
                )
                target = snapshot_root.joinpath(*PurePosixPath(artifact.relative_path).parts)
                _ensure_private_directory(target.parent)
                source_descriptor = _open_regular_no_follow(source)
                try:
                    initial = _FileIdentity.from_stat(os.fstat(source_descriptor))
                    if initial.size != artifact.size_bytes:
                        raise WorkerProtocolError("staged_input_binding_mismatch")
                    flags = (
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | getattr(os, "O_NOFOLLOW", 0)
                    )
                    target_descriptor = os.open(target, flags, 0o600)
                    digest = hashlib.sha256()
                    received = 0
                    try:
                        while True:
                            chunk = os.read(source_descriptor, 1024 * 1024)
                            if not chunk:
                                break
                            received += len(chunk)
                            digest.update(chunk)
                            view = memoryview(chunk)
                            while view:
                                written = os.write(target_descriptor, view)
                                view = view[written:]
                        os.fsync(target_descriptor)
                    finally:
                        os.close(target_descriptor)
                    final = _FileIdentity.from_stat(os.fstat(source_descriptor))
                    path_identity = _path_regular_identity(source)
                    copied_digest = "sha256:" + digest.hexdigest()
                    if (
                        initial != final
                        or initial != path_identity
                        or received != artifact.size_bytes
                        or copied_digest != artifact.sha256
                    ):
                        raise WorkerProtocolError("staged_input_binding_mismatch")
                finally:
                    os.close(source_descriptor)
                os.chmod(target, 0o400)
                snapshot_descriptor = _open_regular_no_follow(target)
                try:
                    snapshot_identity, snapshot_digest = _descriptor_digest(
                        snapshot_descriptor
                    )
                finally:
                    os.close(snapshot_descriptor)
                if (
                    snapshot_identity.size != artifact.size_bytes
                    or snapshot_digest != artifact.sha256
                ):
                    raise WorkerProtocolError("attempt_input_binding_mismatch")
                paths[artifact.relative_path] = target
                identities[artifact.relative_path] = snapshot_identity
                digests[artifact.relative_path] = snapshot_digest
            for root, directories, _ in os.walk(snapshot_root, topdown=False):
                for directory in directories:
                    os.chmod(Path(root) / directory, 0o500)
                os.chmod(root, 0o500)
            os.chmod(attempt_root, 0o500)
            return _AttemptInputs(
                root=attempt_root,
                paths=paths,
                identities=identities,
                digests=digests,
            )
        except BaseException:
            shutil.rmtree(attempt_root, ignore_errors=True)
            raise

    @staticmethod
    def _verify_attempt_inputs(snapshot: _AttemptInputs) -> None:
        for relative_path, path in snapshot.paths.items():
            descriptor = _open_regular_no_follow(path)
            try:
                identity, digest = _descriptor_digest(descriptor)
            finally:
                os.close(descriptor)
            if (
                identity != snapshot.identities[relative_path]
                or digest != snapshot.digests[relative_path]
            ):
                raise WorkerProtocolError("attempt_input_binding_mismatch")

    @staticmethod
    def _read_attempt_bytes(
        snapshot: _AttemptInputs,
        relative_path: str,
        *,
        max_bytes: int,
    ) -> bytes:
        path = snapshot.paths[relative_path]
        descriptor = _open_regular_no_follow(path)
        try:
            initial = _FileIdentity.from_stat(os.fstat(descriptor))
            if initial.size > max_bytes:
                raise WorkerProtocolError("input_too_large")
            digest = hashlib.sha256()
            payload = b""
            while len(payload) < initial.size:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, initial.size - len(payload)),
                )
                if not chunk:
                    break
                payload += chunk
                digest.update(chunk)
            final = _FileIdentity.from_stat(os.fstat(descriptor))
            path_identity = _path_regular_identity(path)
            if (
                initial != snapshot.identities[relative_path]
                or final != initial
                or path_identity != initial
                or len(payload) != initial.size
                or "sha256:" + digest.hexdigest() != snapshot.digests[relative_path]
            ):
                raise WorkerProtocolError("attempt_input_binding_mismatch")
            return payload
        finally:
            os.close(descriptor)

    @classmethod
    def _read_attempt_json(
        cls,
        snapshot: _AttemptInputs,
        relative_path: str,
    ) -> dict[str, Any]:
        payload = cls._read_attempt_bytes(
            snapshot,
            relative_path,
            max_bytes=_MAX_JSON_BYTES,
        )
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerProtocolError("invalid_worker_json") from exc
        if not isinstance(decoded, dict):
            raise WorkerProtocolError("invalid_worker_json")
        return decoded

    def _observation(
        self,
        request: RemoteExecutionRequest,
        *,
        status: str,
        error_code: str = "",
        publication: RemotePublication | None = None,
    ) -> RemoteObservation:
        job_suffix = request.request_sha256.split(":", 1)[1][:16]
        prefix = request.request_id[:70].rstrip(".-_") or "request"
        return RemoteObservation.model_validate(
            {
                "schema_version": "molly_remote_execution_observation.v1",
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "status": status,
                "remote_job_id": f"job-{prefix}-{job_suffix}",
                "observed_at": now_iso(),
                "error_code": error_code,
                "publication": (
                    publication.model_dump(mode="json") if publication is not None else None
                ),
            }
        )

    def _refresh_locked(
        self,
        request: RemoteExecutionRequest,
        state: Mapping[str, Any],
    ) -> RemoteObservation:
        try:
            observation = RemoteObservation.model_validate(state["observation"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WorkerProtocolError("invalid_worker_state") from exc
        if observation.status not in _ACTIVE_STATUSES:
            return observation
        pid = self._state_pid(state, field="pid")
        token = str(state.get("process_token") or "")
        if pid is None:
            return observation
        if self._process_matches(pid, token):
            return observation
        adapter_pid = self._state_pid(state, field="adapter_pid")
        adapter_token = str(state.get("adapter_process_token") or "")
        self._terminate_bound_process_group(adapter_pid, adapter_token)
        self._terminate_bound_process_group(pid, token)
        if observation.status == "CANCEL_REQUESTED":
            terminal = self._observation(request, status="CANCELLED")
        else:
            terminal = self._observation(
                request,
                status="FAILED",
                error_code="worker_process_lost",
            )
        self.store.write_state(request, terminal)
        return terminal

    @staticmethod
    def _state_pid(state: Mapping[str, Any], *, field: str) -> int | None:
        value = state.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise WorkerProtocolError("invalid_worker_state")
        return value

    @staticmethod
    def _process_token(pid: int) -> str:
        try:
            stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeDecodeError):
            try:
                os.kill(pid, 0)
            except OSError:
                return ""
            return f"pid:{pid}"
        closing = stat_line.rfind(")")
        fields = stat_line[closing + 2 :].split()
        if closing < 0 or len(fields) < 20:
            return ""
        return f"linux-start:{fields[19]}"

    @classmethod
    def _process_matches(cls, pid: int, token: str) -> bool:
        if not token:
            return False
        return cls._process_token(pid) == token

    @staticmethod
    def _process_group_exists(process_group: int) -> bool:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _wait_process_group_exit(self, process_group: int) -> bool:
        deadline = time.monotonic() + self.termination_grace_sec
        while self._process_group_exists(process_group):
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.05, self.termination_grace_sec / 4))
        return True

    def _terminate_known_process_group(self, process_group: int) -> None:
        if not self._process_group_exists(process_group):
            return
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise WorkerProtocolError("worker_cancel_failed") from exc
        if self._wait_process_group_exit(process_group):
            return
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError as exc:
            raise WorkerProtocolError("worker_cancel_failed") from exc
        if not self._wait_process_group_exit(process_group):
            raise WorkerProtocolError("worker_cancel_failed")

    def _terminate_spawned_process_group(self, process: Any) -> None:
        """Terminate and reap a process-group leader plus all surviving descendants."""

        process_group = int(process.pid)
        if self._process_group_exists(process_group):
            try:
                os.killpg(process_group, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise WorkerProtocolError("worker_cancel_failed") from exc
        try:
            process.wait(timeout=self.termination_grace_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                raise WorkerProtocolError("worker_cancel_failed") from exc
            try:
                process.wait(timeout=self.termination_grace_sec)
            except subprocess.TimeoutExpired as exc:
                raise WorkerProtocolError("worker_cancel_failed") from exc
        if self._process_group_exists(process_group):
            self._terminate_known_process_group(process_group)

    def _terminate_bound_process_group(
        self,
        process_group: int | None,
        process_token: str,
    ) -> None:
        if process_group is None or not self._process_group_exists(process_group):
            return
        current_token = self._process_token(process_group)
        if current_token and current_token != process_token:
            return
        if current_token == process_token or not current_token:
            self._terminate_known_process_group(process_group)

    def _spawn_runner(self, request_id: str) -> Any:
        job_dir = self.store.job_dir(request_id)
        stdout_path = job_dir / "logs" / "worker.stdout.log"
        stderr_path = job_dir / "logs" / "worker.stderr.log"
        stdout = open(stdout_path, "ab", buffering=0)
        stderr = open(stderr_path, "ab", buffering=0)
        try:
            os.chmod(stdout_path, 0o600)
            os.chmod(stderr_path, 0o600)
            command = [
                sys.executable,
                "-m",
                "ai4s_agent.molly_worker",
                "_run-job",
                "--request-id",
                request_id,
            ]
            environment = self._worker_environment()
            environment["MOLLY_WORKER_ROOT"] = str(self.settings.root)
            return self.popen_factory(
                command,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=environment,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            stdout.close()
            stderr.close()

    def _require_adapter_available(self, request: RemoteExecutionRequest) -> None:
        profile = self._validate_execution_profile(request)
        if profile.profile_id not in {
            "reinvent4-cpu-v1",
            "reinvent4-br1-v2",
            "unimol-predict-br1-v1",
            "unimol-train-br1-v2",
            "unimol-train-v1",
        }:
            raise WorkerProtocolError("adapter_unavailable")
        capabilities = set(self.probe()["capabilities"])
        if not set(profile.required_capabilities).issubset(capabilities):
            raise WorkerProtocolError("required_capability_unavailable")

    def _execute_adapter(
        self,
        request: RemoteExecutionRequest,
        inputs: _AttemptInputs,
    ) -> None:
        if request.execution_profile_id in {"reinvent4-cpu-v1", "reinvent4-br1-v2"}:
            self._execute_reinvent4(request, inputs)
            return
        if request.execution_profile_id in {"unimol-train-v1", "unimol-train-br1-v2"}:
            self._execute_unimol(request, inputs)
            return
        if request.execution_profile_id == "unimol-predict-br1-v1":
            self._execute_unimol_prediction(request, inputs)
            return
        raise WorkerProtocolError("adapter_unavailable")

    def _execute_reinvent4(
        self,
        request: RemoteExecutionRequest,
        inputs: _AttemptInputs | None = None,
    ) -> None:
        if inputs is None:
            inputs = self._snapshot_verified_inputs(request)
        repository = self.settings.reinvent4_repository
        python = self.settings.reinvent4_python
        if repository is None or python is None:
            raise WorkerProtocolError("reinvent4_environment_unavailable")
        self._require_runtime_path(repository, directory=True)
        self._require_runtime_path(python, executable=True)
        config_artifact = self._single_input_for_purpose(request, "generator-config")
        try:
            template = self._read_attempt_bytes(
                inputs,
                config_artifact.relative_path,
                max_bytes=16 * 1024 * 1024,
            ).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkerProtocolError("input_not_utf8") from exc
        if "{{molly_output_csv}}" not in template:
            raise WorkerProtocolError("reinvent4_output_binding_missing")
        task_payload = self._optional_task_payload(request, inputs)
        seed_value = task_payload.get("seed", 0)
        if isinstance(seed_value, bool):
            raise WorkerProtocolError("invalid_reinvent4_seed")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError("invalid_reinvent4_seed") from exc
        if seed < 0 or seed > 2**31 - 1:
            raise WorkerProtocolError("invalid_reinvent4_seed")
        output_path = self.store.output_path(request.request_id, "candidates.csv")
        if output_path.exists():
            raise WorkerProtocolError("output_already_exists")
        replacements = {
            "{{molly_output_csv}}": str(output_path),
            "{{molly_design_request_id}}": request.request_id,
            "{{molly_seed}}": str(seed),
            "{{molly_design_request_sha256}}": request.request_sha256,
        }
        rendered = template
        for token, value in replacements.items():
            rendered = rendered.replace(token, value)
        if any(token in rendered for token in _REINVENT_PLACEHOLDERS):
            raise WorkerProtocolError("reinvent4_template_binding_incomplete")
        try:
            effective = tomllib.loads(rendered)
        except (tomllib.TOMLDecodeError, UnicodeError) as exc:
            raise WorkerProtocolError("invalid_reinvent4_config") from exc
        parameters = effective.get("parameters")
        if (
            not isinstance(parameters, dict)
            or parameters.get("output_file") != str(output_path)
        ):
            raise WorkerProtocolError("reinvent4_output_binding_invalid")
        job_dir = self.store.job_dir(request.request_id)
        effective_config = job_dir / "work" / "effective_config.toml"
        _write_private_bytes(effective_config, rendered.encode("utf-8"))
        environment = self._adapter_environment()
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        nice = "/usr/bin/nice" if Path("/usr/bin/nice").is_file() else "nice"
        command = [
            nice,
            "-n",
            "19",
            str(python),
            "-m",
            "reinvent.Reinvent",
        ]
        config_descriptor = _open_regular_no_follow(effective_config)
        try:
            config_identity, config_digest = _descriptor_digest(config_descriptor)
            command.append(self._descriptor_path(config_descriptor))
            self._run_adapter_command(
                request,
                command,
                cwd=repository,
                env=environment,
                pass_fds=(config_descriptor,),
            )
            final_identity, final_digest = _descriptor_digest(config_descriptor)
            if final_identity != config_identity or final_digest != config_digest:
                raise WorkerProtocolError("effective_config_changed")
        finally:
            os.close(config_descriptor)
        if not output_path.is_file():
            raise WorkerProtocolError("reinvent4_output_missing")
        prefix = self._read_prefix(output_path, 4096)
        if not prefix.startswith(b"SMILES,") or b"\x00" in prefix:
            raise WorkerProtocolError("reinvent4_output_invalid")
        if request.output_contract == "reinvent4-generation-output-v2":
            audit_output = self.store.output_path(
                request.request_id, "generation_audit.json"
            )
            _write_private_json(
                audit_output,
                {
                    "schema_version": "reinvent4_generation_audit.v1",
                    "remote_request": request.model_dump(mode="json"),
                    "request_id": request.request_id,
                    "request_sha256": request.request_sha256,
                    "input_manifest_sha256": request.input_manifest.manifest_sha256,
                    "effective_config_digest": config_digest,
                    "provider_version": self.probe()["details"][
                        "software_versions"
                    ].get("reinvent", "unknown"),
                    "seed": seed,
                },
            )

    def _execute_unimol(
        self,
        request: RemoteExecutionRequest,
        inputs: _AttemptInputs | None = None,
    ) -> None:
        if inputs is None:
            inputs = self._snapshot_verified_inputs(request)
        repository = self.settings.unimol_repository
        python = self.settings.unimol_python
        if repository is None or python is None:
            raise WorkerProtocolError("unimol_environment_unavailable")
        self._require_runtime_path(repository, directory=True)
        self._require_runtime_path(python, executable=True)
        data_artifacts = [
            artifact
            for artifact in request.input_manifest.artifacts
            if artifact.purpose == "training-data"
        ]
        if len(data_artifacts) != 1:
            raise WorkerProtocolError("unimol_training_data_invalid")
        data_artifact = data_artifacts[0]
        if data_artifact.media_type not in {
            "application/csv",
            "application/parquet",
        }:
            raise WorkerProtocolError("unimol_training_data_invalid")
        config_artifact = self._single_input_for_purpose(request, "training-config")
        config = self._validate_unimol_config(
            self._read_attempt_json(inputs, config_artifact.relative_path)
        )
        data_path = inputs.paths[data_artifact.relative_path]
        job_dir = self.store.job_dir(request.request_id)
        scratch = _ensure_private_directory(job_dir / "work" / "unimol-model")
        model_output = self.store.output_path(
            request.request_id,
            "model/model.pt",
            create_parents=True,
        )
        model_config_output = self.store.output_path(
            request.request_id,
            "model/config.yaml",
            create_parents=True,
        )
        model_weights_output = self.store.output_path(
            request.request_id,
            "model/model_0.pth",
            create_parents=True,
        )
        target_scaler_output = self.store.output_path(
            request.request_id,
            "model/target_scaler.ss",
            create_parents=True,
        )
        metrics_output = self.store.output_path(
            request.request_id,
            "model/training_metrics.json",
            create_parents=True,
        )
        audit_output = self.store.output_path(
            request.request_id,
            "model/training_audit.json",
            create_parents=True,
        )
        data_descriptor = _open_regular_no_follow(data_path)
        try:
            data_identity, data_digest = _descriptor_digest(data_descriptor)
            if (
                data_identity != inputs.identities[data_artifact.relative_path]
                or data_digest != inputs.digests[data_artifact.relative_path]
            ):
                raise WorkerProtocolError("attempt_input_binding_mismatch")
            adapter_request = job_dir / "work" / "unimol_adapter_request.json"
            _write_private_json(
                adapter_request,
                {
                    "data_path": self._descriptor_path(data_descriptor),
                    "data_suffix": (
                        ".csv"
                        if data_artifact.media_type == "application/csv"
                        else ".parquet"
                    ),
                    "scratch_path": str(scratch),
                    "model_output": str(model_output),
                    "model_config_output": str(model_config_output),
                    "model_weights_output": str(model_weights_output),
                    "publish_model_directory": (
                        request.output_contract == "unimol-training-output-v2"
                    ),
                    "target_scaler_output": str(target_scaler_output),
                    "metrics_output": str(metrics_output),
                    "config": config,
                },
            )
            runner = job_dir / "work" / "run_unimol.py"
            _write_private_bytes(runner, _UNIMOL_RUNNER.encode("utf-8"))
            environment = self._adapter_environment()
            environment.update(
                {
                    "OMP_NUM_THREADS": str(request.requested_resources.cpu_threads),
                    "MKL_NUM_THREADS": str(request.requested_resources.cpu_threads),
                    "OPENBLAS_NUM_THREADS": str(request.requested_resources.cpu_threads),
                    "CUDA_VISIBLE_DEVICES": str(config["gpu_device"]),
                }
            )
            self._run_adapter_command(
                request,
                [str(python), str(runner), str(adapter_request)],
                cwd=repository,
                env=environment,
                pass_fds=(data_descriptor,),
            )
            final_identity, final_digest = _descriptor_digest(data_descriptor)
            if final_identity != data_identity or final_digest != data_digest:
                raise WorkerProtocolError("attempt_input_binding_mismatch")
        finally:
            os.close(data_descriptor)
        expected_model_outputs = (
            [model_config_output, model_weights_output, target_scaler_output]
            if request.output_contract == "unimol-training-output-v2"
            else [model_output]
        )
        if not all(path.is_file() for path in expected_model_outputs) or not metrics_output.is_file():
            raise WorkerProtocolError("unimol_output_missing")
        _write_private_json(
            audit_output,
            {
                "schema_version": "unimol_training_audit.v1",
                "remote_request": request.model_dump(mode="json"),
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "input_manifest_sha256": request.input_manifest.manifest_sha256,
                "environment": "unimol",
                "provider_version": self.probe()["details"]["software_versions"].get(
                    "unimol-tools", "unknown"
                ),
                "config": config,
            },
        )

    def _execute_unimol_prediction(
        self,
        request: RemoteExecutionRequest,
        inputs: _AttemptInputs | None = None,
    ) -> None:
        if inputs is None:
            inputs = self._snapshot_verified_inputs(request)
        repository = self.settings.unimol_repository
        python = self.settings.unimol_python
        if repository is None or python is None:
            raise WorkerProtocolError("unimol_environment_unavailable")
        self._require_runtime_path(repository, directory=True)
        self._require_runtime_path(python, executable=True)
        data_artifact = self._single_input_for_purpose(request, "prediction-data")
        model_config_artifact = self._single_input_for_purpose(request, "model-config")
        model_weights_artifact = self._single_input_for_purpose(request, "model-weights")
        target_scaler_artifact = self._single_input_for_purpose(request, "target-scaler")
        prediction_config_artifact = self._single_input_for_purpose(
            request, "prediction-config"
        )
        if data_artifact.media_type != "application/csv":
            raise WorkerProtocolError("unimol_prediction_data_invalid")
        config = self._validate_unimol_prediction_config(
            self._read_attempt_json(
                inputs, prediction_config_artifact.relative_path
            )
        )
        bound_artifacts = {
            "data_path": data_artifact,
            "model_config_path": model_config_artifact,
            "model_weights_path": model_weights_artifact,
            "target_scaler_path": target_scaler_artifact,
        }
        descriptors: dict[str, int] = {}
        initial: dict[str, tuple[_FileIdentity, str]] = {}
        try:
            for name, artifact in bound_artifacts.items():
                descriptor = _open_regular_no_follow(
                    inputs.paths[artifact.relative_path]
                )
                identity, digest = _descriptor_digest(descriptor)
                if (
                    identity != inputs.identities[artifact.relative_path]
                    or digest != inputs.digests[artifact.relative_path]
                ):
                    raise WorkerProtocolError("attempt_input_binding_mismatch")
                descriptors[name] = descriptor
                initial[name] = (identity, digest)
            job_dir = self.store.job_dir(request.request_id)
            scratch = _ensure_private_directory(
                job_dir / "work" / "unimol-prediction"
            )
            predictions_output = self.store.output_path(
                request.request_id,
                "predictions.csv",
            )
            audit_output = self.store.output_path(
                request.request_id,
                "prediction_audit.json",
            )
            adapter_request = job_dir / "work" / "unimol_prediction_request.json"
            _write_private_json(
                adapter_request,
                {
                    name: self._descriptor_path(descriptor)
                    for name, descriptor in descriptors.items()
                }
                | {
                    "config": config,
                    "predictions_output": str(predictions_output),
                    "scratch_path": str(scratch),
                },
            )
            runner = job_dir / "work" / "run_unimol_prediction.py"
            _write_private_bytes(runner, _UNIMOL_PREDICTION_RUNNER.encode("utf-8"))
            environment = self._adapter_environment()
            environment.update(
                {
                    "OMP_NUM_THREADS": str(request.requested_resources.cpu_threads),
                    "MKL_NUM_THREADS": str(request.requested_resources.cpu_threads),
                    "OPENBLAS_NUM_THREADS": str(
                        request.requested_resources.cpu_threads
                    ),
                    "CUDA_VISIBLE_DEVICES": str(config["gpu_device"]),
                }
            )
            self._run_adapter_command(
                request,
                [str(python), str(runner), str(adapter_request)],
                cwd=repository,
                env=environment,
                pass_fds=tuple(descriptors.values()),
            )
            for name, descriptor in descriptors.items():
                identity, digest = _descriptor_digest(descriptor)
                if (identity, digest) != initial[name]:
                    raise WorkerProtocolError("attempt_input_binding_mismatch")
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)
        if not predictions_output.is_file():
            raise WorkerProtocolError("unimol_prediction_output_missing")
        prefix = self._read_prefix(predictions_output, 4096)
        if not prefix.startswith(b"candidate_id,predicted_value\n"):
            raise WorkerProtocolError("unimol_prediction_output_invalid")
        _write_private_json(
            audit_output,
            {
                "schema_version": "unimol_prediction_audit.v1",
                "remote_request": request.model_dump(mode="json"),
                "request_id": request.request_id,
                "request_sha256": request.request_sha256,
                "input_manifest_sha256": request.input_manifest.manifest_sha256,
                "environment": "unimol",
                "provider_version": self.probe()["details"]["software_versions"].get(
                    "unimol-tools", "unknown"
                ),
                "config": config,
            },
        )

    def _build_publication(
        self,
        request: RemoteExecutionRequest,
        approval: RemoteExecutionApproval,
    ) -> RemotePublication:
        roster_by_contract = {
            "reinvent4-generation-output-v1": (
                ("reinvent4_candidates", "candidates.csv", "text/csv"),
            ),
            "reinvent4-generation-output-v2": (
                ("reinvent4_candidates", "candidates.csv", "text/csv"),
                (
                    "reinvent4_generation_audit",
                    "generation_audit.json",
                    "application/json",
                ),
            ),
            "unimol-training-output-v1": (
                ("unimol_model", "model/model.pt", "application/octet-stream"),
                (
                    "unimol_training_audit",
                    "model/training_audit.json",
                    "application/json",
                ),
                (
                    "unimol_training_metrics",
                    "model/training_metrics.json",
                    "application/json",
                ),
            ),
            "unimol-training-output-v2": (
                (
                    "unimol_model_config",
                    "model/config.yaml",
                    "application/yaml",
                ),
                (
                    "unimol_model_weights",
                    "model/model_0.pth",
                    "application/octet-stream",
                ),
                (
                    "unimol_target_scaler",
                    "model/target_scaler.ss",
                    "application/octet-stream",
                ),
                (
                    "unimol_training_audit",
                    "model/training_audit.json",
                    "application/json",
                ),
                (
                    "unimol_training_metrics",
                    "model/training_metrics.json",
                    "application/json",
                ),
            ),
            "unimol-prediction-output-v1": (
                (
                    "unimol_predictions",
                    "predictions.csv",
                    "text/csv",
                ),
                (
                    "unimol_prediction_audit",
                    "prediction_audit.json",
                    "application/json",
                ),
            ),
        }
        roster = roster_by_contract.get(request.output_contract)
        if roster is None:
            raise WorkerProtocolError("output_contract_unavailable")
        artifacts: list[RemoteOutputArtifact] = []
        verification_payloads: dict[str, bytes] = {}
        for artifact_id, relative_path, media_type in roster:
            path = self.store.output_path(request.request_id, relative_path)
            descriptor = _open_regular_no_follow(path)
            try:
                identity, sha256 = _descriptor_digest(descriptor)
                if request.output_contract in {
                    "reinvent4-generation-output-v1",
                    "unimol-prediction-output-v1",
                } and media_type == "text/csv":
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    verification_payloads[relative_path] = os.read(descriptor, 4096)
                elif media_type in {"application/json", "application/yaml"}:
                    if identity.size > 16 * 1024 * 1024:
                        raise WorkerProtocolError("output_content_invalid")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    payload = b""
                    while len(payload) < identity.size:
                        chunk = os.read(
                            descriptor,
                            min(64 * 1024, identity.size - len(payload)),
                        )
                        if not chunk:
                            break
                        payload += chunk
                    verification_payloads[relative_path] = payload
                else:
                    verification_payloads[relative_path] = b""
                final = _FileIdentity.from_stat(os.fstat(descriptor))
                path_identity = _path_regular_identity(path)
                if final != identity or path_identity != identity:
                    raise WorkerProtocolError("output_content_changed")
            finally:
                os.close(descriptor)
            artifacts.append(
                RemoteOutputArtifact(
                    artifact_id=artifact_id,
                    relative_path=relative_path,
                    media_type=media_type,
                    size_bytes=identity.size,
                    sha256=sha256,
                )
            )
        artifacts.sort(key=lambda item: (item.artifact_id, item.relative_path))
        verify_remote_output_contract(request.output_contract, artifacts)
        verify_remote_output_contents(
            request.output_contract,
            artifacts,
            lambda relative_path: verification_payloads[relative_path],
        )
        payload: dict[str, Any] = {
            "schema_version": "molly_remote_execution_publication.v1",
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "approval_sha256": approval.approval_sha256,
            "input_manifest_sha256": request.input_manifest.manifest_sha256,
            "output_contract": request.output_contract,
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
            "published_at": now_iso(),
        }
        payload["publication_sha256"] = _digest_bytes(_canonical_bytes(payload))
        return RemotePublication.model_validate(payload)

    def _single_input_for_purpose(
        self,
        request: RemoteExecutionRequest,
        purpose: str,
    ) -> Any:
        matches = [
            artifact
            for artifact in request.input_manifest.artifacts
            if artifact.purpose == purpose
        ]
        if len(matches) != 1:
            raise WorkerProtocolError(f"invalid_{purpose.replace('-', '_')}_input")
        return matches[0]

    def _optional_task_payload(
        self,
        request: RemoteExecutionRequest,
        inputs: _AttemptInputs,
    ) -> dict[str, Any]:
        matches = [
            artifact
            for artifact in request.input_manifest.artifacts
            if artifact.purpose == "execution-request"
        ]
        if not matches:
            return {}
        if len(matches) != 1 or matches[0].media_type != "application/json":
            raise WorkerProtocolError("invalid_execution_request_input")
        return self._read_attempt_json(inputs, matches[0].relative_path)

    @staticmethod
    def _validate_unimol_config(payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "smiles_col",
            "target_col",
            "epochs",
            "learning_rate",
            "batch_size",
            "early_stopping",
            "kfold",
            "gpu_device",
            "seed",
        }
        if set(payload).difference(allowed):
            raise WorkerProtocolError("invalid_unimol_config")

        def column(name: str, default: str) -> str:
            value = str(payload.get(name, default)).strip()
            if (
                not value
                or len(value) > 200
                or any(ord(char) < 32 for char in value)
            ):
                raise WorkerProtocolError("invalid_unimol_config")
            return value

        def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
            value = payload.get(name, default)
            if isinstance(value, bool):
                raise WorkerProtocolError("invalid_unimol_config")
            try:
                parsed = int(value)
            except (TypeError, ValueError) as exc:
                raise WorkerProtocolError("invalid_unimol_config") from exc
            if parsed < minimum or parsed > maximum:
                raise WorkerProtocolError("invalid_unimol_config")
            return parsed

        learning_rate_raw = payload.get("learning_rate", 1e-4)
        if isinstance(learning_rate_raw, bool):
            raise WorkerProtocolError("invalid_unimol_config")
        try:
            learning_rate = float(learning_rate_raw)
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError("invalid_unimol_config") from exc
        if not 0 < learning_rate <= 1:
            raise WorkerProtocolError("invalid_unimol_config")
        kfold = bounded_int("kfold", 1, 1, 1)
        return {
            "smiles_col": column("smiles_col", "SMILES"),
            "target_col": column("target_col", "TARGET"),
            "epochs": bounded_int("epochs", 6, 1, 1000),
            "learning_rate": learning_rate,
            "batch_size": bounded_int("batch_size", 8, 1, 4096),
            "early_stopping": bounded_int("early_stopping", 3, 1, 1000),
            "kfold": kfold,
            "gpu_device": bounded_int("gpu_device", 0, 0, 64),
            "seed": bounded_int("seed", 1729, 0, 2**31 - 1),
        }

    @staticmethod
    def _validate_unimol_prediction_config(
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        allowed = {
            "candidate_id_col",
            "gpu_device",
            "smiles_col",
            "target_property",
        }
        if set(payload) != allowed:
            raise WorkerProtocolError("invalid_unimol_prediction_config")

        def column(name: str) -> str:
            value = str(payload.get(name) or "").strip()
            if (
                not value
                or len(value) > 200
                or any(ord(char) < 32 for char in value)
            ):
                raise WorkerProtocolError("invalid_unimol_prediction_config")
            return value

        gpu_value = payload.get("gpu_device")
        if isinstance(gpu_value, bool):
            raise WorkerProtocolError("invalid_unimol_prediction_config")
        try:
            gpu_device = int(gpu_value)
        except (TypeError, ValueError) as exc:
            raise WorkerProtocolError("invalid_unimol_prediction_config") from exc
        if gpu_device < 0 or gpu_device > 64:
            raise WorkerProtocolError("invalid_unimol_prediction_config")
        return {
            "candidate_id_col": column("candidate_id_col"),
            "gpu_device": gpu_device,
            "smiles_col": column("smiles_col"),
            "target_property": column("target_property"),
        }

    def _run_adapter_command(
        self,
        request: RemoteExecutionRequest,
        command: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        pass_fds: Sequence[int] = (),
    ) -> None:
        job_dir = self.store.job_dir(request.request_id)
        stdout_path = job_dir / "logs" / "adapter.stdout.log"
        stderr_path = job_dir / "logs" / "adapter.stderr.log"
        process: Any | None = None
        return_code: int | None = None
        try:
            with open(stdout_path, "ab", buffering=0) as stdout, open(
                stderr_path,
                "ab",
                buffering=0,
            ) as stderr:
                os.chmod(stdout_path, 0o600)
                os.chmod(stderr_path, 0o600)
                # cancel() reads the adapter binding under this same lock. Keep
                # process creation and durable registration indivisible so it
                # can never terminate the runner while overlooking its adapter.
                with self.store.lock(request.request_id):
                    state = self.store.read_state(request.request_id)
                    observation = RemoteObservation.model_validate(state["observation"])
                    if observation.status in {"CANCEL_REQUESTED", "CANCELLED"}:
                        raise WorkerProtocolError("worker_cancelled")
                    try:
                        process = self.adapter_popen_factory(
                            list(command),
                            stdin=subprocess.DEVNULL,
                            stdout=stdout,
                            stderr=stderr,
                            cwd=cwd,
                            env=dict(env),
                            close_fds=True,
                            pass_fds=tuple(pass_fds),
                            start_new_session=True,
                        )
                    except OSError as exc:
                        raise WorkerProtocolError("adapter_launch_failed") from exc
                    try:
                        process_token = self._process_token(int(process.pid))
                        self.store.write_state(
                            request,
                            observation,
                            pid=self._state_pid(state, field="pid"),
                            process_token=str(state.get("process_token") or ""),
                            adapter_pid=int(process.pid),
                            adapter_process_token=process_token,
                        )
                    except BaseException:
                        self._terminate_spawned_process_group(process)
                        raise
                timeout = (
                    self.adapter_timeout_sec
                    if self.adapter_timeout_sec is not None
                    else float(request.requested_resources.walltime_sec)
                )
                try:
                    return_code = int(process.wait(timeout=timeout))
                except subprocess.TimeoutExpired as exc:
                    self._terminate_spawned_process_group(process)
                    raise WorkerProtocolError("walltime_exceeded") from exc
                if self._process_group_exists(int(process.pid)):
                    self._terminate_known_process_group(int(process.pid))
        finally:
            if process is not None:
                with contextlib.suppress(Exception):
                    with self.store.lock(request.request_id):
                        state = self.store.read_state(request.request_id)
                        if state.get("adapter_pid") == int(process.pid):
                            observation = RemoteObservation.model_validate(
                                state["observation"]
                            )
                            self.store.write_state(
                                request,
                                observation,
                                pid=self._state_pid(state, field="pid"),
                                process_token=str(state.get("process_token") or ""),
                            )
        if return_code is None:
            raise WorkerProtocolError("adapter_launch_failed")
        if return_code != 0:
            raise WorkerProtocolError("adapter_nonzero_exit")

    @staticmethod
    def _require_runtime_path(
        path: Path,
        *,
        directory: bool = False,
        executable: bool = False,
    ) -> None:
        try:
            resolved = path.resolve(strict=True)
            metadata = os.stat(resolved)
        except (FileNotFoundError, OSError) as exc:
            raise WorkerProtocolError("runtime_path_unavailable") from exc
        if directory and not stat.S_ISDIR(metadata.st_mode):
            raise WorkerProtocolError("runtime_path_unavailable")
        if executable and (
            not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK)
        ):
            raise WorkerProtocolError("runtime_path_unavailable")

    @staticmethod
    def _descriptor_path(descriptor: int) -> str:
        proc_path = Path(f"/proc/self/fd/{descriptor}")
        if Path("/proc/self/fd").is_dir():
            return str(proc_path)
        return f"/dev/fd/{descriptor}"

    @staticmethod
    def _read_prefix(path: Path, size: int) -> bytes:
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as exc:
            raise WorkerProtocolError("unsafe_worker_file") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise WorkerProtocolError("unsafe_worker_file")
            return os.read(descriptor, size)
        finally:
            os.close(descriptor)

    @staticmethod
    def _base_environment() -> dict[str, str]:
        allowed = {
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "PATH",
            "SSL_CERT_DIR",
            "SSL_CERT_FILE",
            "TMPDIR",
            "USER",
        }
        return {
            key: value
            for key, value in os.environ.items()
            if key in allowed and value
        }

    @classmethod
    def _adapter_environment(cls) -> dict[str, str]:
        environment = cls._base_environment()
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    @classmethod
    def _worker_environment(cls) -> dict[str, str]:
        environment = cls._base_environment()
        for key, value in os.environ.items():
            if key.startswith("MOLLY_WORKER_") and value:
                environment[key] = value
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def _probe_python_distribution(
        self,
        python: Path | None,
        *,
        module: str,
        distribution: str,
    ) -> str:
        if python is None:
            return ""
        try:
            self._require_runtime_path(python, executable=True)
        except WorkerProtocolError:
            return ""
        script = (
            "import importlib,importlib.metadata;"
            f"importlib.import_module({module!r});"
            f"print(importlib.metadata.version({distribution!r}))"
        )
        try:
            completed = self.run_command(
                [str(python), "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        output = bytes(completed.stdout or b"")
        if completed.returncode != 0 or not output or len(output) > _MAX_PROBE_OUTPUT_BYTES:
            return ""
        try:
            version = output.decode("utf-8").strip()
        except UnicodeDecodeError:
            return ""
        return version if _SAFE_VERSION.fullmatch(version) else ""

    def _probe_unimol_cuda(self) -> str:
        python = self.settings.unimol_python
        if python is None:
            return ""
        script = "import torch;print(torch.version.cuda or '')"
        try:
            completed = self.run_command(
                [str(python), "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        try:
            value = bytes(completed.stdout or b"").decode("utf-8").strip()
        except UnicodeDecodeError:
            return ""
        if completed.returncode != 0 or not _SAFE_VERSION.fullmatch(value):
            return ""
        return value

    def _probe_gpu(self) -> dict[str, str]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return {}
        try:
            completed = self.run_command(
                [
                    executable,
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        output = bytes(completed.stdout or b"")
        if completed.returncode != 0 or not output or len(output) > _MAX_PROBE_OUTPUT_BYTES:
            return {}
        try:
            first_line = output.decode("utf-8").splitlines()[0]
        except (UnicodeDecodeError, IndexError):
            return {}
        parts = [part.strip() for part in first_line.split(",", 1)]
        if len(parts) != 2 or any(len(part) > 160 for part in parts):
            return {}
        return {"device_name": parts[0], "driver_version": parts[1]}

    @staticmethod
    def _memory_bytes() -> int:
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            return 0
        return max(0, pages * page_size)


_UNIMOL_RUNNER = r'''from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.umask(0o077)

request_path = Path(sys.argv[1])
payload = json.loads(request_path.read_text(encoding="utf-8"))
config = payload["config"]
data_path = Path(payload["data_path"])
data_suffix = payload["data_suffix"]
scratch_path = Path(payload["scratch_path"])
model_output = Path(payload["model_output"])
model_config_output = Path(payload["model_config_output"])
model_weights_output = Path(payload["model_weights_output"])
publish_model_directory = bool(payload["publish_model_directory"])
target_scaler_output = Path(payload["target_scaler_output"])
metrics_output = Path(payload["metrics_output"])

from unimol_tools import MolTrain

bound_data_path = scratch_path / ("training-data" + data_suffix)
shutil.copyfile(data_path, bound_data_path)
os.chmod(bound_data_path, 0o400)

trainer = MolTrain(
    task="regression",
    data_type="molecule",
    epochs=config["epochs"],
    learning_rate=config["learning_rate"],
    batch_size=config["batch_size"],
    early_stopping=config["early_stopping"],
    metrics="mae,r2,mse",
    split="random",
    split_seed=config["seed"],
    seed=config["seed"],
    kfold=config["kfold"],
    save_path=str(scratch_path),
    remove_hs=False,
    smiles_col=config["smiles_col"],
    target_cols=config["target_col"],
    target_normalize="auto",
    use_cuda=True,
    use_amp=True,
    use_ddp=False,
    use_gpu=str(config["gpu_device"]),
    model_name="unimolv1",
    conf_cache_level=1,
)
trainer.fit(str(bound_data_path))

model_path = scratch_path / "model_0.pth"
config_path = scratch_path / "config.yaml"
target_scaler_path = scratch_path / "target_scaler.ss"
if not all(path.is_file() for path in (model_path, config_path, target_scaler_path)):
    raise RuntimeError("Uni-Mol prediction-capable model directory is incomplete")
if publish_model_directory:
    for source, destination in (
        (config_path, model_config_output),
        (model_path, model_weights_output),
        (target_scaler_path, target_scaler_output),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o600)
else:
    model_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(model_path, model_output)
    os.chmod(model_output, 0o600)

import numpy as np

predicted = np.asarray(trainer.cv_pred, dtype=float).reshape(-1)
observed = np.asarray(
    trainer.datahub.data["raw_data"][config["target_col"]], dtype=float
).reshape(-1)
if predicted.shape != observed.shape or predicted.size == 0:
    raise RuntimeError("Uni-Mol training metrics roster is invalid")
residual = predicted - observed
denominator = float(np.sum((observed - np.mean(observed)) ** 2))
metrics = {
    "mae": float(np.mean(np.abs(residual))),
    "mse": float(np.mean(residual ** 2)),
    "r2": (1.0 - float(np.sum(residual ** 2)) / denominator) if denominator else 0.0,
    "row_count": int(predicted.size),
}
metrics_output.write_text(
    json.dumps({"metrics": metrics}, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
os.chmod(metrics_output, 0o600)
'''


_UNIMOL_PREDICTION_RUNNER = r'''from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

os.umask(0o077)

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = payload["config"]
scratch_path = Path(payload["scratch_path"])
model_dir = scratch_path / "model"
model_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
for source_key, filename in (
    ("model_config_path", "config.yaml"),
    ("model_weights_path", "model_0.pth"),
    ("target_scaler_path", "target_scaler.ss"),
):
    destination = model_dir / filename
    shutil.copyfile(Path(payload[source_key]), destination)
    os.chmod(destination, 0o400)

data_path = Path(payload["data_path"])
with data_path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise RuntimeError("Uni-Mol prediction roster is empty")
candidate_id_col = config["candidate_id_col"]
smiles_col = config["smiles_col"]
candidate_ids = [str(row.get(candidate_id_col) or "") for row in rows]
if (
    any(not candidate_id for candidate_id in candidate_ids)
    or len(candidate_ids) != len(set(candidate_ids))
    or any(not str(row.get(smiles_col) or "") for row in rows)
):
    raise RuntimeError("Uni-Mol prediction roster identity is invalid")

from unimol_tools import MolPredict

predicted = MolPredict(load_model=str(model_dir)).predict(data=str(data_path))
values = list(predicted.reshape(-1))
if len(values) != len(candidate_ids):
    raise RuntimeError("Uni-Mol prediction result roster mismatch")
output = Path(payload["predictions_output"])
with output.open("x", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["candidate_id", "predicted_value"])
    for candidate_id, value in zip(candidate_ids, values, strict=True):
        numeric = float(value)
        if not __import__("math").isfinite(numeric):
            raise RuntimeError("Uni-Mol prediction is not finite")
        writer.writerow([candidate_id, format(numeric, ".17g")])
os.chmod(output, 0o600)
'''


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molly-worker")
    subcommands = parser.add_subparsers(dest="action", required=True)

    probe = subcommands.add_parser("probe")
    probe.add_argument("--json", action="store_true", required=True)

    stage = subcommands.add_parser("stage")
    stage.add_argument("--json", action="store_true", required=True)

    stage_input = subcommands.add_parser("stage-input")
    stage_input.add_argument("--request-id", required=True)
    stage_input.add_argument("--path", required=True)
    stage_input.add_argument("--size", required=True, type=int)
    stage_input.add_argument("--sha256", required=True)
    stage_input.add_argument("--json", action="store_true", required=True)

    verify = subcommands.add_parser("verify-inputs")
    verify.add_argument("--request-id", required=True)
    verify.add_argument("--json", action="store_true", required=True)

    execute = subcommands.add_parser("execute")
    execute.add_argument("--json", action="store_true", required=True)

    status = subcommands.add_parser("status")
    status.add_argument("--request-id", required=True)
    status.add_argument("--json", action="store_true", required=True)

    cancel = subcommands.add_parser("cancel")
    cancel.add_argument("--request-id", required=True)
    cancel.add_argument("--json", action="store_true", required=True)

    fetch = subcommands.add_parser("fetch-output")
    fetch.add_argument("--request-id", required=True)
    fetch.add_argument("--path", required=True)
    fetch.add_argument("--size", required=True, type=int)
    fetch.add_argument("--sha256", required=True)

    run_job = subcommands.add_parser("_run-job", help=argparse.SUPPRESS)
    run_job.add_argument("--request-id", required=True)
    return parser


def _write_json_response(stream: TextIO, payload: Mapping[str, Any]) -> None:
    stream.write(_canonical_bytes(payload).decode("utf-8"))
    stream.write("\n")
    stream.flush()


def _request_digest_payload(payload: Mapping[str, Any]) -> str:
    if set(payload) != {"request_sha256"}:
        raise WorkerProtocolError("invalid_request_binding")
    return _safe_digest(payload.get("request_sha256"), field="request_sha256")


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: BinaryIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    stdout_buffer: BinaryIO | None = None,
) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    input_stream = stdin if stdin is not None else sys.stdin.buffer
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr
    binary_output = stdout_buffer if stdout_buffer is not None else sys.stdout.buffer
    try:
        worker = MollyWorker(WorkerSettings.load())
        if args.action == "probe":
            _write_json_response(output_stream, worker.probe())
            return 0
        if args.action == "stage":
            _write_json_response(output_stream, worker.stage(_read_json_stream(input_stream)))
            return 0
        if args.action == "stage-input":
            response = worker.stage_input(
                request_id=args.request_id,
                relative_path=args.path,
                size_bytes=args.size,
                sha256=args.sha256,
                stream=input_stream,
            )
            _write_json_response(output_stream, response)
            return 0
        if args.action == "verify-inputs":
            request_sha256 = _request_digest_payload(_read_json_stream(input_stream))
            response = worker.verify_inputs(
                request_id=args.request_id,
                request_sha256=request_sha256,
            )
            _write_json_response(output_stream, response)
            return 0
        if args.action == "execute":
            observation = worker.execute(_read_json_stream(input_stream))
            _write_json_response(output_stream, observation.model_dump(mode="json"))
            return 0
        if args.action == "status":
            request_sha256 = _request_digest_payload(_read_json_stream(input_stream))
            observation = worker.status(
                request_id=args.request_id,
                request_sha256=request_sha256,
            )
            _write_json_response(output_stream, observation.model_dump(mode="json"))
            return 0
        if args.action == "cancel":
            request_sha256 = _request_digest_payload(_read_json_stream(input_stream))
            observation = worker.cancel(
                request_id=args.request_id,
                request_sha256=request_sha256,
            )
            _write_json_response(output_stream, observation.model_dump(mode="json"))
            return 0
        if args.action == "fetch-output":
            worker.fetch_output(
                request_id=args.request_id,
                relative_path=args.path,
                size_bytes=args.size,
                sha256=args.sha256,
                destination=binary_output,
            )
            binary_output.flush()
            return 0
        if args.action == "_run-job":
            return worker.run_job(args.request_id)
        raise WorkerProtocolError("unsupported_worker_action")
    except WorkerProtocolError as exc:
        _write_json_response(
            error_stream,
            {"ok": False, "error_code": exc.code},
        )
        return 2
    except Exception:
        _write_json_response(
            error_stream,
            {"ok": False, "error_code": "worker_internal_error"},
        )
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
