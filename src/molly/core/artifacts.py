"""Immutable, content-addressed scientific artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from typing import Any

from .errors import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    CoreContractError,
    PathSecurityError,
)
from .ids import (
    artifact_id_for_sha256,
    artifact_sha256,
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
    validate_artifact_id,
    validate_artifact_ids,
    validate_identifier,
    validate_sha256,
)


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Immutable metadata for one exact byte sequence."""

    artifact_id: str
    sha256: str
    media_type: str
    schema_name: str | None = None
    schema_version: str | None = None
    producer_step_id: str | None = None
    input_artifact_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_timestamp)
    provenance: dict[str, Any] = field(default_factory=dict)
    size_bytes: int = 0

    def __post_init__(self) -> None:
        validate_artifact_id(self.artifact_id)
        validate_sha256(self.sha256)
        if self.artifact_id != artifact_id_for_sha256(self.sha256):
            raise CoreContractError("artifact_id must embed the record SHA-256")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise CoreContractError("media_type is required")
        if any(char in self.media_type for char in "\r\n\x00"):
            raise CoreContractError("media_type contains a control character")
        for value, field_name in (
            (self.schema_name, "schema_name"),
            (self.schema_version, "schema_version"),
            (self.producer_step_id, "producer_step_id"),
        ):
            if value is not None:
                validate_identifier(value, field=field_name)
        object.__setattr__(
            self,
            "input_artifact_ids",
            validate_artifact_ids(self.input_artifact_ids, field="input_artifact_ids"),
        )
        object.__setattr__(
            self,
            "created_at",
            normalize_timestamp(self.created_at, field="created_at"),
        )
        object.__setattr__(
            self,
            "provenance",
            freeze_json_mapping(self.provenance, field="provenance"),
        )
        if not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise CoreContractError("size_bytes must be a non-negative integer")

    @property
    def content_sha256(self) -> str:
        """Compatibility-free descriptive alias for the content digest."""

        return self.sha256

    @property
    def content_type(self) -> str:
        """Descriptive alias for the media type."""

        return self.media_type

    @property
    def size(self) -> int:
        """Descriptive alias used by a few filesystem callers."""

        return self.size_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "producer_step_id": self.producer_step_id,
            "input_artifact_ids": list(self.input_artifact_ids),
            "created_at": self.created_at,
            "provenance": thaw_json(self.provenance),
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        if not isinstance(value, Mapping):
            raise ArtifactIntegrityError("artifact metadata must be a JSON object")
        try:
            digest = value.get("sha256")
            if digest is None:
                digest = value["content_sha256"]
            size = value.get("size_bytes")
            if size is None:
                size = value["size"]
            return cls(
                artifact_id=str(value["artifact_id"]),
                sha256=str(digest),
                media_type=str(value["media_type"]),
                schema_name=(
                    None if value.get("schema_name") is None else str(value["schema_name"])
                ),
                schema_version=(
                    None
                    if value.get("schema_version") is None
                    else str(value["schema_version"])
                ),
                producer_step_id=(
                    None
                    if value.get("producer_step_id") is None
                    else str(value["producer_step_id"])
                ),
                input_artifact_ids=tuple(value.get("input_artifact_ids", ())),
                created_at=str(value["created_at"]),
                provenance=dict(value.get("provenance", {})),
                size_bytes=int(size),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("artifact metadata is malformed") from exc


class ArtifactStore:
    """Small local content-addressed store with no-replace publication."""

    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.is_symlink():
            raise PathSecurityError("artifact store root cannot be a symlink")
        self.root = configured.absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.objects_root = self.root / "objects"
        self.metadata_root = self.root / "metadata"
        self._ensure_directory(self.objects_root)
        self._ensure_directory(self.metadata_root)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise PathSecurityError(f"artifact store directory cannot be a symlink: {path}")
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir() or path.is_symlink():
            raise PathSecurityError(f"artifact store path is not a real directory: {path}")

    def _paths(self, artifact_id: str, *, create_prefix: bool = False) -> tuple[Path, Path]:
        digest = artifact_sha256(artifact_id)
        prefix = self.objects_root / digest[:2]
        if create_prefix:
            self._ensure_directory(prefix)
        elif prefix.is_symlink():
            raise PathSecurityError(f"artifact object directory cannot be a symlink: {prefix}")
        object_path = prefix / digest
        metadata_path = self.metadata_root / f"{digest}.json"
        for path in (object_path, metadata_path):
            if path.is_symlink():
                raise PathSecurityError(f"artifact path cannot be a symlink: {path}")
            if not path.absolute().is_relative_to(self.root):
                raise PathSecurityError("artifact key escapes the configured store root")
        return object_path, metadata_path

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _publish_no_replace(cls, path: Path, payload: bytes) -> bool:
        """Publish bytes through a fsynced temp file and atomic hard-link."""

        if path.is_symlink():
            raise PathSecurityError(f"cannot publish through symlink: {path}")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return False
            cls._fsync_directory(path.parent)
            return True
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _read_verified_bytes(path: Path, expected_sha256: str) -> bytes:
        if path.is_symlink():
            raise PathSecurityError("artifact object cannot be a symlink")
        if not path.exists():
            raise ArtifactNotFoundError(f"artifact object is missing: {path.name}")
        if not path.is_file():
            raise ArtifactIntegrityError("artifact object is not a regular file")
        payload = path.read_bytes()
        actual = sha256_bytes(payload)
        if actual != expected_sha256:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: expected {expected_sha256}, got {actual}"
            )
        return payload

    @staticmethod
    def _read_record(path: Path) -> ArtifactRecord:
        if path.is_symlink():
            raise PathSecurityError("artifact metadata cannot be a symlink")
        if not path.exists():
            raise ArtifactNotFoundError(f"artifact metadata is missing: {path.name}")
        if not path.is_file():
            raise ArtifactIntegrityError("artifact metadata is not a regular file")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactIntegrityError("artifact metadata is not valid UTF-8 JSON") from exc
        return ArtifactRecord.from_dict(value)

    def put(
        self,
        content: bytes | bytearray | memoryview,
        *,
        media_type: str | None = None,
        content_type: str | None = None,
        schema_name: str | None = None,
        schema_version: str | None = None,
        producer_step_id: str | None = None,
        input_artifact_ids: tuple[str, ...] | list[str] = (),
        created_at: str | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Publish bytes and immutable metadata, returning the first record."""

        if not isinstance(content, (bytes, bytearray, memoryview)):
            raise CoreContractError("artifact content must be bytes-like")
        if media_type is None:
            media_type = content_type
        elif content_type is not None and media_type != content_type:
            raise CoreContractError("media_type and content_type disagree")
        if media_type is None:
            raise CoreContractError("media_type is required")
        payload = bytes(content)
        digest = sha256_bytes(payload)
        record = ArtifactRecord(
            artifact_id=artifact_id_for_sha256(digest),
            sha256=digest,
            media_type=media_type,
            schema_name=schema_name,
            schema_version=schema_version,
            producer_step_id=producer_step_id,
            input_artifact_ids=tuple(input_artifact_ids),
            created_at=created_at or utc_timestamp(),
            provenance={} if provenance is None else provenance,
            size_bytes=len(payload),
        )
        object_path, metadata_path = self._paths(record.artifact_id, create_prefix=True)

        if object_path.exists():
            self._read_verified_bytes(object_path, digest)
        else:
            self._publish_no_replace(object_path, payload)
            self._read_verified_bytes(object_path, digest)

        if metadata_path.exists():
            existing = self._read_record(metadata_path)
            if existing.sha256 != digest or existing.artifact_id != record.artifact_id:
                raise ArtifactConflictError("existing metadata conflicts with artifact identity")
            existing_payload = self._read_verified_bytes(object_path, digest)
            if existing.size_bytes != len(existing_payload):
                raise ArtifactIntegrityError("existing metadata size does not match artifact bytes")
            return existing

        metadata_payload = canonical_json_bytes(record.to_dict()) + b"\n"
        self._publish_no_replace(metadata_path, metadata_payload)
        existing = self._read_record(metadata_path)
        if existing.sha256 != digest or existing.artifact_id != record.artifact_id:
            raise ArtifactConflictError("published metadata conflicts with artifact identity")
        return existing

    def put_bytes(self, content: bytes, **kwargs: Any) -> ArtifactRecord:
        """Explicit alias for callers that want to emphasize byte content."""

        return self.put(content, **kwargs)

    def put_json(self, value: Any, **kwargs: Any) -> ArtifactRecord:
        """Publish one canonical JSON artifact without adding a new backend."""

        kwargs.setdefault("media_type", "application/json")
        return self.put(canonical_json_bytes(value), **kwargs)

    def verify(self, artifact_id: str) -> ArtifactRecord:
        """Verify metadata and exact object bytes, returning the record."""

        validate_artifact_id(artifact_id)
        object_path, metadata_path = self._paths(artifact_id)
        record = self._read_record(metadata_path)
        if record.artifact_id != artifact_id:
            raise ArtifactIntegrityError("metadata artifact identity does not match lookup key")
        payload = self._read_verified_bytes(object_path, artifact_sha256(artifact_id))
        if len(payload) != record.size_bytes:
            raise ArtifactIntegrityError("artifact metadata size does not match object bytes")
        return record

    def read(self, artifact_id: str) -> bytes:
        """Read and verify one immutable artifact."""

        record = self.verify(artifact_id)
        object_path, _ = self._paths(record.artifact_id)
        return self._read_verified_bytes(object_path, record.sha256)

    def get(self, artifact_id: str) -> bytes:
        """Alias for :meth:`read`."""

        return self.read(artifact_id)

    def read_bytes(self, artifact_id: str) -> bytes:
        """Explicit byte-oriented alias for :meth:`read`."""

        return self.read(artifact_id)

    def get_bytes(self, artifact_id: str) -> bytes:
        """Explicit byte-oriented alias for :meth:`read`."""

        return self.read(artifact_id)

    def get_metadata(self, artifact_id: str) -> ArtifactRecord:
        """Look up metadata only after verifying the referenced object."""

        return self.verify(artifact_id)

    def metadata(self, artifact_id: str) -> ArtifactRecord:
        """Alias for :meth:`get_metadata`."""

        return self.get_metadata(artifact_id)

    def exists(self, artifact_id: str, *, verify: bool = False) -> bool:
        """Return whether a complete artifact exists; optionally verify bytes."""

        validate_artifact_id(artifact_id)
        object_path, metadata_path = self._paths(artifact_id)
        if (
            object_path.is_symlink()
            or metadata_path.is_symlink()
            or not object_path.is_file()
            or not metadata_path.is_file()
        ):
            if object_path.is_symlink() or metadata_path.is_symlink():
                raise PathSecurityError("artifact paths cannot be symlinks")
            return False
        if not object_path.exists() or not metadata_path.exists():
            return False
        if verify:
            self.verify(artifact_id)
        return True

    def object_path(self, artifact_id: str) -> Path:
        """Return the deterministic object path after validating the key."""

        return self._paths(artifact_id)[0]

    def metadata_path(self, artifact_id: str) -> Path:
        """Return the deterministic metadata path after validating the key."""

        return self._paths(artifact_id)[1]
