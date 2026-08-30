"""Canonical identifiers, timestamps, JSON, and digest utilities for Core.

The production namespace deliberately keeps these helpers small and standard
library only.  Artifact identity is the SHA-256 digest of the exact bytes
published by :class:`molly.core.artifacts.ArtifactStore`; metadata never
substitutes for that byte-level verification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any

from .errors import CoreContractError, PathSecurityError


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "access_key",
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "passwd",
        "private_key",
        "secret",
        "token",
    }
)

def utc_timestamp(value: datetime | None = None) -> str:
    """Return an RFC 3339 UTC timestamp with a deterministic shape."""

    moment = value if value is not None else datetime.now(timezone.utc)
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise CoreContractError("timestamps must include timezone information")
    normalized = moment.astimezone(timezone.utc)
    return normalized.isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalize_timestamp(value: str, *, field: str = "timestamp") -> str:
    """Validate and normalize an RFC 3339 timestamp to UTC."""

    if not isinstance(value, str) or not value.strip():
        raise CoreContractError(f"{field} must be a non-empty timestamp")
    raw = value.strip()
    parse_value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError as exc:
        raise CoreContractError(f"{field} is not a valid ISO-8601 timestamp") from exc
    return utc_timestamp(parsed)


def validate_identifier(value: str, *, field: str = "identifier") -> str:
    """Validate a non-path identifier used by Core records."""

    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise CoreContractError(
            f"{field} must match {_IDENTIFIER_RE.pattern} and contain no path separators"
        )
    return value


def validate_reference(value: str, *, field: str = "reference") -> str:
    """Validate an opaque, stable non-filesystem identity/reference."""

    if not isinstance(value, str) or not value or len(value) > 512:
        raise CoreContractError(f"{field} must be a non-empty reference of at most 512 characters")
    if any(char.isspace() for char in value) or "\x00" in value:
        raise CoreContractError(f"{field} must not contain whitespace or NUL")
    if value.startswith(("/", "\\", "~")) or "\\" in value:
        raise PathSecurityError(f"{field} cannot be an absolute or platform path")
    if any(part == ".." for part in value.split("/")):
        raise PathSecurityError(f"{field} cannot contain parent traversal")
    return value


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    """Validate a lowercase, unprefixed SHA-256 hexadecimal digest."""

    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise CoreContractError(f"{field} must be a lowercase 64-character SHA-256 digest")
    return value


def validate_digest_reference(value: str, *, field: str = "digest") -> str:
    """Accept either a bare SHA-256 or the explicit ``sha256:`` form."""

    if isinstance(value, str) and value.startswith("sha256:"):
        value = value.removeprefix("sha256:")
    return validate_sha256(value, field=field)


def artifact_id_for_sha256(sha256: str) -> str:
    """Return the only production artifact-id representation."""

    return f"sha256:{validate_sha256(sha256)}"


def validate_artifact_id(value: str, *, field: str = "artifact_id") -> str:
    """Validate an artifact ID and its embedded digest."""

    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise CoreContractError(f"{field} must use the sha256:<digest> form")
    digest = value.removeprefix("sha256:")
    validate_sha256(digest, field=f"{field} digest")
    return value


def artifact_sha256(value: str, *, field: str = "artifact_id") -> str:
    """Extract and validate the bare digest from an artifact ID."""

    validate_artifact_id(value, field=field)
    return value.removeprefix("sha256:")


def validate_artifact_ids(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    """Validate an ordered, duplicate-free tuple of artifact IDs."""

    result = tuple(validate_artifact_id(value, field=field) for value in values)
    if len(result) != len(set(result)):
        raise CoreContractError(f"{field} must not contain duplicate artifact IDs")
    return result


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes with SHA-256."""

    return hashlib.sha256(value).hexdigest()


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    parts = set(normalized.split("_"))
    return normalized in _SENSITIVE_KEY_PARTS or bool(parts & _SENSITIVE_KEY_PARTS)


def _freeze_json(value: Any, *, field: str) -> Any:
    """Validate JSON-compatible values and recursively make them immutable."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CoreContractError(f"{field} cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise CoreContractError(f"{field} object keys must be non-empty strings")
            if _sensitive_key(key):
                raise CoreContractError(f"{field} cannot store credential-like key {key!r}")
            frozen[key] = _freeze_json(item, field=f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, field=field) for item in value)
    raise CoreContractError(f"{field} must contain only canonical JSON values")


def freeze_json_mapping(value: Mapping[str, Any] | None, *, field: str) -> Mapping[str, Any]:
    """Validate and freeze a JSON object used as non-secret metadata."""

    if value is None:
        value = {}
    frozen = _freeze_json(value, field=field)
    if not isinstance(frozen, Mapping):
        raise CoreContractError(f"{field} must be a JSON object")
    return frozen


def thaw_json(value: Any) -> Any:
    """Convert internal immutable JSON containers into ordinary JSON values."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically as compact UTF-8 bytes.

    JSON object keys are sorted, separators contain no insignificant
    whitespace, non-ASCII characters are escaped, and non-finite numbers are
    rejected.  These bytes are the only representation used for Core record
    digests.
    """

    try:
        normalized = _freeze_json(value, field="canonical JSON")
        return json.dumps(
            thaw_json(normalized),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CoreContractError("value is not canonical JSON") from exc


def json_sha256(value: Any) -> str:
    """Hash a canonical JSON value."""

    return sha256_bytes(canonical_json_bytes(value))
