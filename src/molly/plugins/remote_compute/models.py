"""Secret-free durable compute identities and output references."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from molly.core.artifacts import ArtifactRecord
from molly.core.ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
)

from .errors import ComputeError


class JobState(str, Enum):
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


def _state(value: str | JobState) -> str:
    candidate = value.value if isinstance(value, JobState) else value
    if not isinstance(candidate, str):
        raise ComputeError("job state must be text")
    try:
        return JobState(candidate.strip().upper()).value
    except ValueError as exc:
        raise ComputeError(f"unknown job state: {candidate!r}") from exc


@dataclass(frozen=True, slots=True)
class ComputeProfile:
    """Server-owned backend/profile identity; credentials are references only."""

    profile_id: str
    profile_version: str = "1"
    backend_kind: str = "local"
    host_identity: str = "local"
    worker_ref: str = "worker:br1"
    environment_ref: str = "environment:br1"
    resource_constraints: Mapping[str, Any] = field(default_factory=dict)
    credential_ref: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.profile_id, field="profile_id")
        validate_identifier(self.profile_version, field="profile_version")
        backend = self.backend_kind.strip().lower() if isinstance(self.backend_kind, str) else ""
        if backend not in {"local", "remote"}:
            raise ComputeError("backend_kind must be local or remote")
        object.__setattr__(self, "backend_kind", backend)
        for name in ("host_identity", "worker_ref", "environment_ref"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or "\x00" in value:
                raise ComputeError(f"{name} must be bounded text")
            object.__setattr__(self, name, value.strip())
        object.__setattr__(self, "resource_constraints", freeze_json_mapping(self.resource_constraints, field="resource constraints"))
        if self.credential_ref is not None:
            object.__setattr__(self, "credential_ref", validate_reference(self.credential_ref, field="credential_ref"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "backend_kind": self.backend_kind,
            "host_identity": self.host_identity,
            "worker_ref": self.worker_ref,
            "environment_ref": self.environment_ref,
            "resource_constraints": thaw_json(self.resource_constraints),
            # Deliberately use a non-sensitive structural key here.  The
            # value is only a logical server-owned reference; actual secret
            # material never enters a profile digest or durable job record.
            "server_material_ref": self.credential_ref,
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def profile_digest(self) -> str:
        return self.digest


@dataclass(frozen=True, slots=True)
class JobHandle:
    """Exact durable identity for one compute submission."""

    job_id: str
    profile_id: str
    profile_digest: str
    task_digest: str
    idempotency_key: str
    input_artifact_ids: tuple[str, ...] = ()
    execution_config_digest: str | None = None
    submitted_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        for value, name in ((self.job_id, "job_id"), (self.profile_id, "profile_id")):
            validate_identifier(value, field=name)
        for value, name in ((self.profile_digest, "profile_digest"), (self.task_digest, "task_digest"), (self.idempotency_key, "idempotency_key")):
            object.__setattr__(self, name, validate_digest_reference(value, field=name))
        ids = tuple(validate_artifact_id(item, field="input_artifact_id") for item in self.input_artifact_ids)
        if len(ids) != len(set(ids)):
            raise ComputeError("input_artifact_ids must be unique")
        object.__setattr__(self, "input_artifact_ids", ids)
        if self.execution_config_digest is not None:
            object.__setattr__(self, "execution_config_digest", validate_digest_reference(self.execution_config_digest, field="execution_config_digest"))
        object.__setattr__(self, "submitted_at", normalize_timestamp(self.submitted_at, field="submitted_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "profile_id": self.profile_id,
            "profile_digest": self.profile_digest,
            "task_digest": self.task_digest,
            "idempotency_key": self.idempotency_key,
            "input_artifact_ids": list(self.input_artifact_ids),
            "execution_config_digest": self.execution_config_digest,
            "submitted_at": self.submitted_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "JobHandle":
        try:
            return cls(
                job_id=str(value["job_id"]),
                profile_id=str(value["profile_id"]),
                profile_digest=str(value["profile_digest"]),
                task_digest=str(value["task_digest"]),
                idempotency_key=str(value["idempotency_key"]),
                input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
                execution_config_digest=(None if value.get("execution_config_digest") is None else str(value["execution_config_digest"])),
                submitted_at=str(value["submitted_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ComputeError("durable JobHandle is malformed") from exc


@dataclass(frozen=True, slots=True)
class ComputeOutput:
    """A runner output before the backend publishes it into ArtifactStore."""

    name: str
    content: bytes
    media_type: str
    schema_name: str | None = None
    schema_version: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.name, field="output name")
        if not isinstance(self.content, (bytes, bytearray, memoryview)):
            raise ComputeError("compute output content must be bytes-like")
        object.__setattr__(self, "content", bytes(self.content))
        if not isinstance(self.media_type, str) or not self.media_type.strip() or any(char in self.media_type for char in "\r\n\x00"):
            raise ComputeError("compute output media_type is invalid")
        object.__setattr__(self, "media_type", self.media_type.strip())
        for value, name in ((self.schema_name, "schema_name"), (self.schema_version, "schema_version")):
            if value is not None:
                validate_identifier(value, field=name)


@dataclass(frozen=True, slots=True)
class ComputeOutputRef:
    name: str
    artifact_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.name, field="output name")
        validate_artifact_id(self.artifact_id)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "artifact_id": self.artifact_id}


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    job_id: str
    task_digest: str
    profile_digest: str
    outputs: tuple[ComputeOutputRef, ...]
    manifest_artifact_id: str

    def __post_init__(self) -> None:
        validate_identifier(self.job_id, field="job_id")
        object.__setattr__(self, "task_digest", validate_digest_reference(self.task_digest, field="task_digest"))
        object.__setattr__(self, "profile_digest", validate_digest_reference(self.profile_digest, field="profile_digest"))
        outputs = tuple(item if isinstance(item, ComputeOutputRef) else ComputeOutputRef(**item) for item in self.outputs)
        if not outputs or len({item.name for item in outputs}) != len(outputs):
            raise ComputeError("artifact bundle outputs must be non-empty and uniquely named")
        object.__setattr__(self, "outputs", outputs)
        validate_artifact_id(self.manifest_artifact_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_digest": self.task_digest,
            "profile_digest": self.profile_digest,
            "outputs": [item.to_dict() for item in self.outputs],
            "manifest_artifact_id": self.manifest_artifact_id,
        }


@dataclass(frozen=True, slots=True)
class JobStatus:
    handle: JobHandle
    state: str | JobState
    outputs: tuple[ComputeOutputRef, ...] = ()
    manifest_artifact_id: str | None = None
    error_type: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.handle, JobHandle):
            raise ComputeError("JobStatus requires a JobHandle")
        object.__setattr__(self, "state", _state(self.state))
        outputs = tuple(item if isinstance(item, ComputeOutputRef) else ComputeOutputRef(**item) for item in self.outputs)
        object.__setattr__(self, "outputs", outputs)
        if self.manifest_artifact_id is not None:
            validate_artifact_id(self.manifest_artifact_id)
        if self.error_type is not None:
            validate_identifier(self.error_type, field="error_type")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handle": self.handle.to_dict(),
            "state": self.state,
            "outputs": [item.to_dict() for item in self.outputs],
            "manifest_artifact_id": self.manifest_artifact_id,
            "error_type": self.error_type,
        }


__all__ = [
    "ArtifactBundle",
    "ComputeOutput",
    "ComputeOutputRef",
    "ComputeProfile",
    "JobHandle",
    "JobState",
    "JobStatus",
]
