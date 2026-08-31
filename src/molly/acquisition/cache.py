"""Integrity-checked, cache-first storage for acquisition responses."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
from urllib.parse import urlsplit

from molly.core.ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    normalize_timestamp,
    sha256_bytes,
    utc_timestamp,
    validate_digest_reference,
    validate_identifier,
    validate_reference,
    validate_sha256,
)

from .errors import AcquisitionCacheError, AcquisitionIntegrityError
from .models import ACCEPTED_MEDIA_TYPES, MAX_RESPONSE_BYTES, AccessStatus, ArtifactClass, ContentFamily
from .policy import assert_no_secret_values, sanitize_url


def _validate_cache_url(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value or any(
        char.isspace() or ord(char) < 32 for char in value
    ):
        raise AcquisitionCacheError(f"cache {field} is malformed")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise AcquisitionCacheError(f"cache {field} is malformed") from exc
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise AcquisitionCacheError(f"cache {field} is not a sanitized HTTPS URL")
    if sanitize_url(value) != value:
        raise AcquisitionCacheError(f"cache {field} is not sanitized/canonical")


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A complete cache manifest with request and provenance bindings."""

    schema_version: str
    cache_identity: str
    provider: str
    provider_config_digest: str
    route_policy_version: str
    request_identity: str
    canonical_identifier: str | None
    route_id: str
    request_shape: Mapping[str, Any]
    source_url: str
    resolved_url: str
    redirect_chain: tuple[str, ...]
    response_status: int
    retrieved_at: str
    access_status: str
    license_status: str
    access_basis: str
    redistribution_basis: str
    content_type: str
    content_family: str
    body_sha256: str
    body_size: int
    artifact_class: str
    access_profile_ref: str | None
    cache_status: str
    stored_at: str = field(default_factory=utc_timestamp)

    def __post_init__(self) -> None:
        validate_identifier(self.schema_version, field="cache schema_version")
        validate_digest_reference(self.cache_identity, field="cache_identity")
        validate_identifier(self.provider, field="cache provider")
        validate_digest_reference(self.provider_config_digest, field="provider_config_digest")
        validate_digest_reference(self.request_identity, field="request_identity")
        validate_identifier(self.route_id, field="cache route_id")
        validate_identifier(self.route_policy_version, field="route_policy_version")
        if self.canonical_identifier is not None:
            validate_reference(self.canonical_identifier, field="canonical_identifier")
        if not isinstance(self.request_shape, Mapping):
            raise AcquisitionCacheError("cache request shape must be an object")
        try:
            object.__setattr__(
                self,
                "request_shape",
                freeze_json_mapping(self.request_shape, field="cache request shape"),
            )
        except Exception as exc:
            raise AcquisitionCacheError("cache request shape is not canonical JSON") from exc
        if not isinstance(self.response_status, int) or not 100 <= self.response_status <= 599:
            raise AcquisitionCacheError("cache response status is invalid")
        object.__setattr__(
            self,
            "retrieved_at",
            normalize_timestamp(self.retrieved_at, field="retrieved_at"),
        )
        for value, field_name in (
            (self.access_status, "access_status"),
            (self.license_status, "license_status"),
            (self.access_basis, "access_basis"),
            (self.redistribution_basis, "redistribution_basis"),
            (self.cache_status, "cache_status"),
        ):
            if (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > 512
                or any(char in value for char in "\x00\r\n")
            ):
                raise AcquisitionCacheError(f"cache {field_name} is malformed")
        try:
            AccessStatus(self.access_status)
        except ValueError as exc:
            raise AcquisitionCacheError("cache access status is outside the closed vocabulary") from exc
        if self.access_profile_ref is not None:
            validate_reference(self.access_profile_ref, field="access_profile_ref")
        validate_sha256(self.body_sha256, field="body_sha256")
        if (
            isinstance(self.body_size, bool)
            or not isinstance(self.body_size, int)
            or not 0 <= self.body_size <= MAX_RESPONSE_BYTES
        ):
            raise AcquisitionCacheError("cache body size is invalid")
        if not isinstance(self.artifact_class, str) or not self.artifact_class:
            raise AcquisitionCacheError("cache artifact class is required")
        try:
            artifact_class = ArtifactClass(self.artifact_class)
        except ValueError as exc:
            raise AcquisitionCacheError("cache artifact class is outside the closed vocabulary") from exc
        if artifact_class in {ArtifactClass.RUNTIME_SECRET, ArtifactClass.CREDENTIAL_REFERENCE}:
            raise AcquisitionCacheError("runtime secrets and credential references cannot be cached")
        content_type = self.content_type.casefold().split(";", 1)[0].strip()
        if content_type not in ACCEPTED_MEDIA_TYPES:
            raise AcquisitionCacheError("cache content type is outside the acquisition families")
        object.__setattr__(self, "content_type", content_type)
        try:
            content_family = ContentFamily(self.content_family)
        except ValueError as exc:
            raise AcquisitionCacheError("cache content family is invalid") from exc
        expected_family = {
            "application/json": ContentFamily.JSON,
            "application/xml": ContentFamily.XML,
            "text/xml": ContentFamily.XML,
            "text/html": ContentFamily.HTML,
            "application/pdf": ContentFamily.PDF,
        }[content_type]
        if content_family is not expected_family:
            raise AcquisitionCacheError("cache content family does not match content type")
        for value, field_name in (
            (self.source_url, "source_url"),
            (self.resolved_url, "resolved_url"),
        ):
            _validate_cache_url(value, field=field_name)
        object.__setattr__(self, "redirect_chain", tuple(self.redirect_chain))
        for value in self.redirect_chain:
            _validate_cache_url(value, field="redirect URL")
        object.__setattr__(self, "stored_at", normalize_timestamp(self.stored_at, field="stored_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cache_identity": validate_digest_reference(self.cache_identity, field="cache_identity"),
            "provider": self.provider,
            "provider_config_digest": validate_digest_reference(self.provider_config_digest, field="provider_config_digest"),
            "route_policy_version": self.route_policy_version,
            "request_identity": self.request_identity,
            "canonical_identifier": self.canonical_identifier,
            "route_id": self.route_id,
            "request_shape": dict(self.request_shape),
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "redirect_chain": list(self.redirect_chain),
            "response_status": self.response_status,
            "retrieved_at": self.retrieved_at,
            "access_status": self.access_status,
            "license_status": self.license_status,
            "access_basis": self.access_basis,
            "redistribution_basis": self.redistribution_basis,
            "content_type": self.content_type,
            "content_family": self.content_family,
            "body_sha256": self.body_sha256,
            "body_size": self.body_size,
            "artifact_class": self.artifact_class,
            "access_profile_ref": self.access_profile_ref,
            "cache_status": self.cache_status,
            "stored_at": self.stored_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheEntry":
        if not isinstance(value, Mapping):
            raise AcquisitionCacheError("cache manifest must be a JSON object")
        try:
            return cls(
                schema_version=str(value["schema_version"]),
                cache_identity=str(value["cache_identity"]),
                provider=str(value["provider"]),
                provider_config_digest=str(value["provider_config_digest"]),
                route_policy_version=str(value["route_policy_version"]),
                request_identity=str(value["request_identity"]),
                canonical_identifier=(
                    None if value.get("canonical_identifier") is None else str(value["canonical_identifier"])
                ),
                route_id=str(value["route_id"]),
                request_shape=dict(value["request_shape"]),
                source_url=str(value["source_url"]),
                resolved_url=str(value["resolved_url"]),
                redirect_chain=tuple(value.get("redirect_chain", ())),
                response_status=int(value["response_status"]),
                retrieved_at=str(value["retrieved_at"]),
                access_status=str(value["access_status"]),
                license_status=str(value["license_status"]),
                access_basis=str(value["access_basis"]),
                redistribution_basis=str(value["redistribution_basis"]),
                content_type=str(value["content_type"]),
                content_family=str(value["content_family"]),
                body_sha256=str(value["body_sha256"]),
                body_size=int(value["body_size"]),
                artifact_class=str(value["artifact_class"]),
                access_profile_ref=(
                    None
                    if value.get("access_profile_ref") is None
                    else str(value["access_profile_ref"])
                ),
                cache_status=str(value["cache_status"]),
                stored_at=str(value["stored_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionCacheError("cache manifest is malformed") from exc


@dataclass(frozen=True, slots=True)
class CachedResponse:
    entry: CacheEntry
    body: bytes


class AcquisitionCache:
    """A no-replace local cache; ArtifactStore remains content authority."""

    def __init__(self, root: Path | str) -> None:
        configured = Path(root)
        if configured.is_symlink():
            raise AcquisitionCacheError("cache root cannot be a symlink")
        self.root = configured.absolute()
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries_root = self.root / "entries"
        self.bodies_root = self.root / "bodies"
        self._ensure_directory(self.entries_root)
        self._ensure_directory(self.bodies_root)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise AcquisitionCacheError("cache directory cannot be a symlink")
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir() or path.is_symlink():
            raise AcquisitionCacheError("cache path is not a real directory")

    def _paths(self, cache_identity: str) -> tuple[Path, Path]:
        digest = validate_digest_reference(cache_identity, field="cache_identity")
        entry = self.entries_root / f"{digest}.json"
        body = self.bodies_root / digest[:2] / digest
        self._ensure_directory(body.parent)
        for path in (entry, body):
            if path.is_symlink() or not path.absolute().is_relative_to(self.root):
                raise AcquisitionCacheError("cache identity escapes the configured root")
        return entry, body

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _publish_no_replace(cls, path: Path, payload: bytes) -> bool:
        if path.is_symlink():
            raise AcquisitionCacheError("cannot publish cache data through a symlink")
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
    def _read_bytes(path: Path, expected_sha256: str) -> bytes:
        if path.is_symlink() or not path.exists() or not path.is_file():
            raise AcquisitionCacheError("cache body is missing or not a regular file")
        body = path.read_bytes()
        if sha256_bytes(body) != expected_sha256:
            raise AcquisitionIntegrityError("cache body digest mismatch")
        return body

    @staticmethod
    def _read_entry(path: Path) -> CacheEntry:
        if path.is_symlink() or not path.exists() or not path.is_file():
            raise AcquisitionCacheError("cache manifest is missing or not a regular file")
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcquisitionCacheError("cache manifest is not valid UTF-8 JSON") from exc
        try:
            canonical = canonical_json_bytes(value)
        except Exception as exc:
            raise AcquisitionCacheError("cache manifest is not canonical JSON") from exc
        if canonical + b"\n" != raw:
            raise AcquisitionCacheError("cache manifest is not canonical JSON")
        return CacheEntry.from_dict(value)

    def get(
        self,
        cache_identity: str,
        *,
        expected_binding: Mapping[str, Any],
        secret_values: tuple[str, ...] = (),
    ) -> CachedResponse | None:
        """Return a verified hit, or None only when neither cache file exists."""

        entry_path, body_path = self._paths(cache_identity)
        if not entry_path.exists() and not body_path.exists():
            return None
        if not entry_path.exists() or not body_path.exists():
            raise AcquisitionCacheError("cache entry is partial")
        entry = self._read_entry(entry_path)
        expected_identity = validate_digest_reference(cache_identity, field="cache_identity")
        if validate_digest_reference(entry.cache_identity, field="cache_identity") != expected_identity:
            raise AcquisitionCacheError("cache manifest identity mismatch")
        for key, expected in expected_binding.items():
            if entry.to_dict().get(key) != expected:
                raise AcquisitionCacheError(f"cache manifest binding mismatch: {key}")
        body = self._read_bytes(body_path, entry.body_sha256)
        if len(body) != entry.body_size:
            raise AcquisitionIntegrityError("cache body size mismatch")
        try:
            assert_no_secret_values(body, secret_values)
            assert_no_secret_values(canonical_json_bytes(entry.to_dict()), secret_values)
        except AcquisitionIntegrityError:
            raise
        return CachedResponse(entry=entry, body=body)

    def put(
        self,
        cache_identity: str,
        body: bytes,
        *,
        manifest: Mapping[str, Any],
        secret_values: tuple[str, ...] = (),
    ) -> CacheEntry:
        """Commit body first and manifest second, never replacing either."""

        if not isinstance(body, bytes):
            raise AcquisitionCacheError("cache body must be bytes")
        identity = validate_digest_reference(cache_identity, field="cache_identity")
        body_digest = sha256_bytes(body)
        try:
            assert_no_secret_values(body, secret_values)
        except AcquisitionIntegrityError:
            raise
        candidate = dict(manifest)
        candidate.setdefault("cache_identity", identity)
        candidate.setdefault("body_sha256", body_digest)
        candidate.setdefault("body_size", len(body))
        if validate_digest_reference(str(candidate["cache_identity"]), field="cache_identity") != identity:
            raise AcquisitionCacheError("cache manifest identity does not match cache key")
        if str(candidate["body_sha256"]) != body_digest or int(candidate["body_size"]) != len(body):
            raise AcquisitionIntegrityError("cache manifest body binding is incorrect")
        entry = CacheEntry.from_dict(candidate)
        manifest_payload = canonical_json_bytes(entry.to_dict()) + b"\n"
        assert_no_secret_values(manifest_payload, secret_values)
        entry_path, body_path = self._paths(identity)

        if entry_path.exists() != body_path.exists():
            raise AcquisitionCacheError("cache entry is partial")

        if body_path.exists():
            existing_body = self._read_bytes(body_path, body_digest)
            if existing_body != body:
                raise AcquisitionIntegrityError("cache identity already contains different bytes")
        else:
            self._publish_no_replace(body_path, body)
            existing_body = self._read_bytes(body_path, body_digest)
            if existing_body != body:
                raise AcquisitionIntegrityError("cache body changed during publication")

        if entry_path.exists():
            existing = self._read_entry(entry_path)
            existing_value = existing.to_dict()
            candidate_value = entry.to_dict()
            # ``stored_at`` is cache bookkeeping, not part of the request
            # identity.  A concurrent identical fill may therefore have a
            # different timestamp, but every request/body binding must still
            # match exactly and the existing manifest remains authoritative.
            existing_value.pop("stored_at", None)
            candidate_value.pop("stored_at", None)
            if existing_value != candidate_value:
                raise AcquisitionCacheError("cache identity already contains a different manifest")
            return existing
        self._publish_no_replace(entry_path, manifest_payload)
        return self._read_entry(entry_path)


__all__ = ["AcquisitionCache", "CacheEntry", "CachedResponse"]
