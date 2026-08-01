from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import subprocess
import time
from pathlib import PurePosixPath
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.resource_profiles import (
    ConnectionProfile,
    ExecutionProfile,
    TransferManifest,
    verify_transfer_manifest_binding,
)
from ai4s_agent.remote_output_contracts import verify_remote_output_contract
from ai4s_agent.remote_execution_storage import PinnedExecutionTree


EXECUTION_REQUEST_VERSION = "molly_remote_execution_request.v1"
APPROVAL_VERSION = "molly_remote_execution_approval.v1"
CANCELLATION_VERSION = "molly_remote_execution_cancellation.v1"
OBSERVATION_VERSION = "molly_remote_execution_observation.v1"
PUBLICATION_VERSION = "molly_remote_execution_publication.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,95}$")
_REMOTE_STATUSES = {
    "ACCEPTED",
    "RUNNING",
    "SUCCEEDED",
    "FAILED",
    "CANCEL_REQUESTED",
    "CANCELLED",
}
_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_COMMIT_BOUNDARY_HOOK: Callable[[str], None] | None = None
_LOCAL_IO_HOOK: Callable[[str], None] | None = None


def _commit_boundary(name: str) -> None:
    if _COMMIT_BOUNDARY_HOOK is not None:
        _COMMIT_BOUNDARY_HOOK(name)


def _local_io_boundary(name: str) -> None:
    if _LOCAL_IO_HOOK is not None:
        _LOCAL_IO_HOOK(name)


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _identifier(value: Any, field: str) -> str:
    raw = str(value or "")
    if raw != raw.strip().lower() or not _SAFE_ID.fullmatch(raw):
        raise ValueError(f"{field} must be a canonical lowercase identifier")
    return raw


def _sha256(value: Any, field: str) -> str:
    raw = str(value or "")
    if not _SHA256.fullmatch(raw):
        raise ValueError(f"{field} must be a lowercase sha256 digest")
    return raw


def _relative_path(value: Any, field: str) -> str:
    raw = str(value or "")
    path = PurePosixPath(raw)
    if (
        not raw
        or raw != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or "\\" in raw
        or any(ord(char) < 32 for char in raw)
    ):
        raise ValueError(f"{field} must be a canonical safe relative path")
    return raw


class RequestedResources(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_count: int
    cpu_threads: int
    walltime_sec: int

    @field_validator("gpu_count", mode="before")
    @classmethod
    def validate_gpu_count(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("gpu_count must be a non-negative integer")
        return value

    @field_validator("cpu_threads", "walltime_sec", mode="before")
    @classmethod
    def validate_positive(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("resource limits must be positive integers")
        return value


class RemoteExecutionRequest(BaseModel):
    """Immutable, non-executable-until-approved remote execution contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[EXECUTION_REQUEST_VERSION] = EXECUTION_REQUEST_VERSION
    request_id: str
    project_id: str
    run_id: str
    task_id: str
    connection_id: str
    connection_profile_digest: str
    execution_profile_id: str
    execution_profile_digest: str
    input_manifest: TransferManifest
    requested_resources: RequestedResources
    output_contract: str
    created_at: str
    request_sha256: str

    @field_validator(
        "request_id", "project_id", "run_id", "task_id", "connection_id",
        "execution_profile_id", "output_contract", mode="before"
    )
    @classmethod
    def validate_identifiers(cls, value: Any, info: Any) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "connection_profile_digest", "execution_profile_digest", "request_sha256",
        mode="before",
    )
    @classmethod
    def validate_digests(cls, value: Any, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_self_binding(self) -> "RemoteExecutionRequest":
        if self.input_manifest.request_id != self.request_id:
            raise ValueError("execution request and transfer manifest request IDs differ")
        if (
            self.input_manifest.connection_id != self.connection_id
            or self.input_manifest.connection_profile_digest
            != self.connection_profile_digest
            or self.input_manifest.execution_profile_id != self.execution_profile_id
            or self.input_manifest.execution_profile_digest
            != self.execution_profile_digest
        ):
            raise ValueError("execution request profile binding mismatch")
        payload = self.model_dump(mode="json", exclude={"request_sha256"})
        if self.request_sha256 != _digest(_canonical_bytes(payload)):
            raise ValueError("execution request digest mismatch")
        return self


class RemoteExecutionApproval(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[APPROVAL_VERSION] = APPROVAL_VERSION
    request_id: str
    request_sha256: str
    actor: str
    note: str = ""
    approved_at: str
    approval_sha256: str

    @field_validator("request_id", mode="before")
    @classmethod
    def validate_request_id(cls, value: Any) -> str:
        return _identifier(value, "request_id")

    @field_validator("request_sha256", "approval_sha256", mode="before")
    @classmethod
    def validate_digest(cls, value: Any, info: Any) -> str:
        return _sha256(value, info.field_name)

    @field_validator("actor", "note", mode="before")
    @classmethod
    def validate_text(cls, value: Any, info: Any) -> str:
        clean = str(value or "").strip()
        if (info.field_name == "actor" and not clean) or len(clean) > 500:
            raise ValueError(f"{info.field_name} is invalid")
        if any(ord(char) < 32 and char not in "\n\t" for char in clean):
            raise ValueError(f"{info.field_name} contains control characters")
        return clean

    @model_validator(mode="after")
    def validate_self_binding(self) -> "RemoteExecutionApproval":
        payload = self.model_dump(mode="json", exclude={"approval_sha256"})
        if self.approval_sha256 != _digest(_canonical_bytes(payload)):
            raise ValueError("approval digest mismatch")
        return self


class RemoteExecutionCancellation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[CANCELLATION_VERSION] = CANCELLATION_VERSION
    request_id: str
    request_sha256: str
    requested_at: str
    cancellation_sha256: str

    @field_validator("request_id", mode="before")
    @classmethod
    def validate_request_id(cls, value: Any) -> str:
        return _identifier(value, "request_id")

    @field_validator("request_sha256", "cancellation_sha256", mode="before")
    @classmethod
    def validate_digest(cls, value: Any, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_self_binding(self) -> "RemoteExecutionCancellation":
        payload = self.model_dump(mode="json", exclude={"cancellation_sha256"})
        if self.cancellation_sha256 != _digest(_canonical_bytes(payload)):
            raise ValueError("cancellation digest mismatch")
        return self


class RemoteOutputArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    relative_path: str
    media_type: str
    size_bytes: int
    sha256: str

    @field_validator("artifact_id", mode="before")
    @classmethod
    def validate_artifact_id(cls, value: Any) -> str:
        return _identifier(value, "artifact_id")

    @field_validator("relative_path", mode="before")
    @classmethod
    def validate_path(cls, value: Any) -> str:
        return _relative_path(value, "relative_path")

    @field_validator("media_type", mode="before")
    @classmethod
    def validate_media_type(cls, value: Any) -> str:
        raw = str(value or "")
        if not re.fullmatch(
            r"[a-z0-9][a-z0-9.+-]*/[a-z0-9][a-z0-9.+-]*", raw
        ):
            raise ValueError("output media_type must be canonical and valid")
        return raw

    @field_validator("size_bytes", mode="before")
    @classmethod
    def validate_size(cls, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("output size must be a non-negative integer")
        return value

    @field_validator("sha256", mode="before")
    @classmethod
    def validate_sha256(cls, value: Any) -> str:
        return _sha256(value, "sha256")


class RemotePublication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[PUBLICATION_VERSION] = PUBLICATION_VERSION
    request_id: str
    request_sha256: str
    approval_sha256: str
    input_manifest_sha256: str
    output_contract: str
    artifacts: tuple[RemoteOutputArtifact, ...] = Field(min_length=1)
    published_at: str
    publication_sha256: str

    @field_validator("request_id", "output_contract", mode="before")
    @classmethod
    def validate_ids(cls, value: Any, info: Any) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "request_sha256", "approval_sha256", "input_manifest_sha256",
        "publication_sha256", mode="before"
    )
    @classmethod
    def validate_digests(cls, value: Any, info: Any) -> str:
        return _sha256(value, info.field_name)

    @model_validator(mode="after")
    def validate_roster(self) -> "RemotePublication":
        keys = [(item.artifact_id, item.relative_path) for item in self.artifacts]
        artifact_ids = [item.artifact_id for item in self.artifacts]
        paths = [item.relative_path for item in self.artifacts]
        if (
            keys != sorted(keys)
            or len(artifact_ids) != len(set(artifact_ids))
            or len(paths) != len(set(paths))
        ):
            raise ValueError("publication artifacts must be unique and deterministically sorted")
        payload = self.model_dump(mode="json", exclude={"publication_sha256"})
        if self.publication_sha256 != _digest(_canonical_bytes(payload)):
            raise ValueError("publication digest mismatch")
        verify_remote_output_contract(self.output_contract, self.artifacts)
        return self


class RemoteObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[OBSERVATION_VERSION] = OBSERVATION_VERSION
    request_id: str
    request_sha256: str
    status: Literal[
        "ACCEPTED", "RUNNING", "SUCCEEDED", "FAILED", "CANCEL_REQUESTED", "CANCELLED"
    ]
    remote_job_id: str
    observed_at: str
    error_code: str = ""
    publication: RemotePublication | None = None

    @field_validator("request_id", "remote_job_id", mode="before")
    @classmethod
    def validate_ids(cls, value: Any, info: Any) -> str:
        return _identifier(value, info.field_name)

    @field_validator("request_sha256", mode="before")
    @classmethod
    def validate_request_digest(cls, value: Any) -> str:
        return _sha256(value, "request_sha256")

    @field_validator("error_code", mode="before")
    @classmethod
    def validate_error_code(cls, value: Any) -> str:
        clean = str(value or "").strip().lower()
        if clean and not _SAFE_ID.fullmatch(clean):
            raise ValueError("error_code is invalid")
        return clean

    @model_validator(mode="after")
    def validate_terminal_payload(self) -> "RemoteObservation":
        if self.status == "SUCCEEDED" and self.publication is None:
            raise ValueError("successful remote observation requires a publication")
        if self.status != "SUCCEEDED" and self.publication is not None:
            raise ValueError("non-success observation must not publish artifacts")
        return self


def build_remote_execution_request(
    *,
    project_id: str,
    run_id: str,
    task_id: str,
    transfer_manifest: TransferManifest | Mapping[str, Any],
    connection: ConnectionProfile,
    execution_profile: ExecutionProfile,
    requested_resources: RequestedResources | Mapping[str, Any],
    created_at: str | None = None,
) -> RemoteExecutionRequest:
    manifest = verify_transfer_manifest_binding(
        transfer_manifest,
        connection=connection,
        execution_profile=execution_profile,
    )
    resources = validate_requested_resources_against_execution_profile(
        requested_resources,
        execution_profile=execution_profile,
    )
    payload: dict[str, Any] = {
        "schema_version": EXECUTION_REQUEST_VERSION,
        "request_id": manifest.request_id,
        "project_id": _identifier(project_id, "project_id"),
        "run_id": _identifier(run_id, "run_id"),
        "task_id": _identifier(task_id, "task_id"),
        "connection_id": connection.connection_id,
        "connection_profile_digest": connection.digest(),
        "execution_profile_id": execution_profile.profile_id,
        "execution_profile_digest": execution_profile.digest(),
        "input_manifest": manifest.model_dump(mode="json"),
        "requested_resources": resources.model_dump(mode="json"),
        "output_contract": execution_profile.output_contract,
        "created_at": created_at or now_iso(),
    }
    payload["request_sha256"] = _digest(_canonical_bytes(payload))
    return RemoteExecutionRequest.model_validate(payload)


def validate_requested_resources_against_execution_profile(
    requested_resources: RequestedResources | Mapping[str, Any],
    *,
    execution_profile: ExecutionProfile,
) -> RequestedResources:
    """Validate complete resources against a fixed profile without executing.

    This is the single pure resource-contract check shared by remote request
    construction and the Scientific Agent resource-authority control plane. It
    deliberately creates no request, transport, worker action, or execution
    state.
    """

    resources = RequestedResources.model_validate(requested_resources)
    limits = execution_profile.resource_limits
    if execution_profile.device_policy == "cpu_only" and resources.gpu_count != 0:
        raise ValueError("CPU-only execution profile cannot request a GPU")
    if execution_profile.device_policy == "gpu_required" and resources.gpu_count < 1:
        raise ValueError("execution profile requires a GPU")
    if (
        resources.gpu_count > limits.gpu_count_max
        or resources.cpu_threads > limits.cpu_threads_max
        or resources.walltime_sec > limits.walltime_sec_max
    ):
        raise ValueError("requested resources exceed the execution profile contract")
    return resources


def build_remote_execution_approval(
    request: RemoteExecutionRequest,
    *,
    request_sha256: str,
    actor: str,
    note: str = "",
    approved_at: str | None = None,
) -> RemoteExecutionApproval:
    if request_sha256 != request.request_sha256:
        raise ValueError("approval does not bind the exact execution request")
    payload: dict[str, Any] = {
        "schema_version": APPROVAL_VERSION,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "actor": actor,
        "note": note,
        "approved_at": approved_at or now_iso(),
    }
    payload["approval_sha256"] = _digest(_canonical_bytes(payload))
    return RemoteExecutionApproval.model_validate(payload)


def build_remote_execution_cancellation(
    request: RemoteExecutionRequest,
    *,
    requested_at: str | None = None,
) -> RemoteExecutionCancellation:
    payload: dict[str, Any] = {
        "schema_version": CANCELLATION_VERSION,
        "request_id": request.request_id,
        "request_sha256": request.request_sha256,
        "requested_at": requested_at or now_iso(),
    }
    payload["cancellation_sha256"] = _digest(_canonical_bytes(payload))
    return RemoteExecutionCancellation.model_validate(payload)


class RemoteTransportError(RuntimeError):
    pass


class PinnedWorkerTransport:
    """Only invokes the fixed molly-worker protocol through an SSH config alias."""

    def __init__(
        self,
        *,
        runner: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.runner = runner
        self.popen_factory = popen_factory

    def dispatch(
        self,
        *,
        connection: ConnectionProfile,
        request: RemoteExecutionRequest,
        approval: RemoteExecutionApproval,
        tree: PinnedExecutionTree,
    ) -> RemoteObservation:
        envelope = {
            "request": request.model_dump(mode="json"),
            "approval": approval.model_dump(mode="json"),
        }
        self._stage(connection, request, envelope, tree)
        return self._invoke(connection, ["execute", "--json"], envelope)

    def _stage(
        self,
        connection: ConnectionProfile,
        request: RemoteExecutionRequest,
        envelope: Mapping[str, Any],
        tree: PinnedExecutionTree,
    ) -> None:
        command = self._ssh_command(connection, ["stage", "--json"])
        completed = self._run(command, connection, input=_canonical_bytes(envelope))
        try:
            payload = json.loads(bytes(completed.stdout or b"").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteTransportError("remote worker staging response is invalid") from exc
        if not isinstance(payload, dict) or payload != {
            "ok": True,
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
        }:
            raise RemoteTransportError("remote worker staging binding mismatch")
        for artifact in request.input_manifest.artifacts:
            descriptor = tree.open_file("inputs", artifact.relative_path)
            try:
                with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
                    copied = self._run(
                        self._ssh_command(
                            connection,
                            [
                                "stage-input",
                                "--request-id", request.request_id,
                                "--path", artifact.relative_path,
                                "--size", str(artifact.size_bytes),
                                "--sha256", artifact.sha256,
                                "--json",
                            ],
                        ),
                        connection,
                        stdin=stream,
                    )
            finally:
                os.close(descriptor)
            try:
                copied_payload = json.loads(bytes(copied.stdout or b"").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RemoteTransportError("remote input transfer response is invalid") from exc
            if copied_payload != {
                "ok": True,
                "request_id": request.request_id,
                "relative_path": artifact.relative_path,
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
            }:
                raise RemoteTransportError("remote input transfer binding mismatch")
        verified = self._run(
            self._ssh_command(connection, ["verify-inputs", "--request-id", request.request_id, "--json"]),
            connection,
            input=_canonical_bytes({"request_sha256": request.request_sha256}),
        )
        try:
            verified_payload = json.loads(bytes(verified.stdout or b"").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RemoteTransportError("remote input verification response is invalid") from exc
        if verified_payload != {
            "ok": True,
            "request_id": request.request_id,
            "request_sha256": request.request_sha256,
            "manifest_sha256": request.input_manifest.manifest_sha256,
        }:
            raise RemoteTransportError("remote input verification binding mismatch")

    def inspect(
        self, *, connection: ConnectionProfile, request: RemoteExecutionRequest
    ) -> RemoteObservation:
        return self._invoke(
            connection,
            ["status", "--request-id", request.request_id, "--json"],
            {"request_sha256": request.request_sha256},
        )

    def cancel(
        self, *, connection: ConnectionProfile, request: RemoteExecutionRequest
    ) -> RemoteObservation:
        return self._invoke(
            connection,
            ["cancel", "--request-id", request.request_id, "--json"],
            {"request_sha256": request.request_sha256},
        )

    def fetch_outputs(
        self,
        *,
        connection: ConnectionProfile,
        request: RemoteExecutionRequest,
        publication: RemotePublication,
        tree: PinnedExecutionTree,
    ) -> None:
        """Stream only the content-bound roster into descriptor-pinned staging."""

        def fetch_one(artifact: RemoteOutputArtifact, descriptor: int) -> None:
            self._stream_output_bounded(
                self._ssh_command(
                    connection,
                    [
                        "fetch-output",
                        "--request-id", request.request_id,
                        "--path", artifact.relative_path,
                        "--size", str(artifact.size_bytes),
                        "--sha256", artifact.sha256,
                    ],
                ),
                connection=connection,
                destination_fd=descriptor,
                max_bytes=artifact.size_bytes,
            )

        try:
            tree.publish_downloaded_outputs(
                artifacts=publication.artifacts,
                fetcher=fetch_one,
                digest=_digest,
                request_sha256=request.request_sha256,
                publication_sha256=publication.publication_sha256,
            )
        except ValueError as exc:
            raise RemoteTransportError(str(exc)) from exc

    def _invoke(
        self,
        connection: ConnectionProfile,
        worker_args: Sequence[str],
        payload: Mapping[str, Any],
    ) -> RemoteObservation:
        command = self._ssh_command(connection, worker_args)
        try:
            completed = self._run(command, connection, input=_canonical_bytes(payload))
        except RemoteTransportError:
            raise
        output = bytes(completed.stdout or b"")
        if completed.returncode != 0:
            raise RemoteTransportError("remote worker rejected the fixed request")
        if not output or len(output) > _MAX_RESPONSE_BYTES:
            raise RemoteTransportError("remote worker returned an invalid response")
        try:
            decoded = json.loads(output.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("response is not an object")
            return RemoteObservation.model_validate(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise RemoteTransportError("remote worker returned an invalid response") from exc

    def _run(
        self,
        command: Sequence[str],
        connection: ConnectionProfile,
        *,
        input: bytes | None = None,
        stdin: Any | None = None,
        stdout: Any = subprocess.PIPE,
    ) -> subprocess.CompletedProcess[bytes]:
        if input is not None and stdin is not None:
            raise RemoteTransportError("remote worker transport input is ambiguous")
        try:
            completed = self.runner(
                list(command), input=input, stdin=stdin, stdout=stdout, stderr=subprocess.PIPE,
                timeout=connection.default_timeout_sec, check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RemoteTransportError("remote worker transport unavailable") from exc
        if completed.returncode != 0:
            raise RemoteTransportError("remote worker rejected the fixed request")
        return completed

    def _stream_output_bounded(
        self,
        command: Sequence[str],
        *,
        connection: ConnectionProfile,
        destination_fd: int,
        max_bytes: int,
    ) -> None:
        try:
            process = self.popen_factory(
                list(command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        except OSError as exc:
            raise RemoteTransportError("remote output transport unavailable") from exc
        stdout = process.stdout
        if stdout is None:
            self._terminate_process(process)
            raise RemoteTransportError("remote output transport has no byte stream")
        selector = selectors.DefaultSelector()
        deadline = time.monotonic() + connection.default_timeout_sec
        received = 0
        try:
            selector.register(stdout, selectors.EVENT_READ)
            while True:
                remaining_time = deadline - time.monotonic()
                if remaining_time <= 0:
                    self._terminate_process(process)
                    raise RemoteTransportError("remote output transfer timed out")
                if not selector.select(remaining_time):
                    self._terminate_process(process)
                    raise RemoteTransportError("remote output transfer timed out")
                chunk = os.read(stdout.fileno(), min(64 * 1024, max_bytes - received + 1))
                if not chunk:
                    break
                received += len(chunk)
                if received > max_bytes:
                    self._terminate_process(process)
                    raise RemoteTransportError("remote output exceeded its declared size")
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            remaining_time = max(0.0, deadline - time.monotonic())
            try:
                return_code = process.wait(timeout=remaining_time)
            except subprocess.TimeoutExpired as exc:
                self._terminate_process(process)
                raise RemoteTransportError("remote output transfer timed out") from exc
            if return_code != 0:
                raise RemoteTransportError("remote output transfer failed")
            if received != max_bytes:
                raise RemoteTransportError("remote output transfer size mismatch")
        finally:
            selector.close()
            stdout.close()

    @staticmethod
    def _terminate_process(process: Any) -> None:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass

    @staticmethod
    def _connection_options(connection: ConnectionProfile) -> list[str]:
        options = [
            "-o", "BatchMode=yes", "-o", "ClearAllForwardings=yes",
            "-o", "StrictHostKeyChecking=yes",
        ]
        if connection.known_hosts_path:
            options.extend(["-o", f"UserKnownHostsFile={connection.known_hosts_path}"])
        return options

    @classmethod
    def _ssh_command(
        cls, connection: ConnectionProfile, worker_args: Sequence[str]
    ) -> list[str]:
        return [
            "ssh", *cls._connection_options(connection), connection.ssh_host_alias,
            "--", "molly-worker", *worker_args,
        ]



from ai4s_agent.remote_execution_service import (  # noqa: E402
    DescriptorRemoteExecutionLifecycleService as RemoteExecutionLifecycleService,
)


__all__ = [
    "PinnedWorkerTransport",
    "RemoteExecutionApproval",
    "RemoteExecutionCancellation",
    "RemoteExecutionLifecycleService",
    "RemoteExecutionRequest",
    "RemoteObservation",
    "RemoteOutputArtifact",
    "RemotePublication",
    "RemoteTransportError",
    "RequestedResources",
    "build_remote_execution_approval",
    "build_remote_execution_cancellation",
    "build_remote_execution_request",
]
