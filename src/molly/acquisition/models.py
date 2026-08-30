"""Typed, server-owned models for CORE-03 literature acquisition.

This module contains configuration and normalized protocol records only.  It
does not parse scientific documents; acquired bodies remain immutable bytes
until a later document milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import ipaddress
import re
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import urlsplit

from molly.core.ids import (
    artifact_id_for_sha256,
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
    validate_sha256,
)

from .errors import AcquisitionConfigurationError, AcquisitionIntegrityError


MAX_RESPONSE_BYTES = 25 * 1024 * 1024
MAX_QUERY_LENGTH = 512
MAX_RESULT_LIMIT = 20
MAX_REDIRECTS = 5
MAX_CONNECT_TIMEOUT_SECONDS = 10.0
MAX_READ_TIMEOUT_SECONDS = 30.0
MAX_TOTAL_TIMEOUT_SECONDS = 60.0

ACCEPTED_MEDIA_TYPES = frozenset(
    {"application/json", "application/xml", "text/xml", "text/html", "application/pdf"}
)


class ProviderClass(str, Enum):
    METADATA = "METADATA"
    OA_RESOLUTION = "OA_RESOLUTION"
    FULL_TEXT = "FULL_TEXT"


class ArtifactClass(str, Enum):
    PUBLIC_ARTIFACT = "PUBLIC_ARTIFACT"
    PRIVATE_ARTIFACT = "PRIVATE_ARTIFACT"
    RUNTIME_SECRET = "RUNTIME_SECRET"
    CREDENTIAL_REFERENCE = "CREDENTIAL_REFERENCE"


class AcquisitionStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    NO_ELIGIBLE_SOURCE = "NO_ELIGIBLE_SOURCE"
    CACHE_HIT = "CACHE_HIT"
    ACQUIRED = "ACQUIRED"


class ContentFamily(str, Enum):
    JSON = "json"
    XML = "xml"
    HTML = "html"
    PDF = "pdf"


def _text(value: str, *, field: str, maximum: int = 512, required: bool = True) -> str:
    if not isinstance(value, str):
        raise AcquisitionConfigurationError(f"{field} must be text")
    if required and not value.strip():
        raise AcquisitionConfigurationError(f"{field} is required")
    if len(value) > maximum or any(char in value for char in "\x00\r\n"):
        raise AcquisitionConfigurationError(f"{field} is outside the bounded text contract")
    return value


def _enum_value(value: str | Enum, enum_type: type[Enum], *, field: str) -> str:
    candidate = value.value if isinstance(value, enum_type) else value
    if not isinstance(candidate, str):
        raise AcquisitionConfigurationError(f"{field} must be a string")
    try:
        normalized = candidate.strip()
        try:
            return enum_type(normalized).value
        except ValueError:
            return enum_type(normalized.upper()).value
    except ValueError as exc:
        raise AcquisitionConfigurationError(f"unknown {field}: {candidate!r}") from exc


def _canonical_config_host(value: str, *, field: str = "host") -> str:
    value = _text(value, field=field, maximum=253)
    if value != value.strip() or value.endswith(".") or any(char.isspace() for char in value):
        raise AcquisitionConfigurationError(f"{field} is not a canonical host")
    if any(ord(char) > 127 for char in value):
        raise AcquisitionConfigurationError(f"{field} must use an unambiguous ASCII hostname")
    if "/" in value or ":" in value or "@" in value:
        raise AcquisitionConfigurationError(f"{field} must not contain a URL or userinfo")
    normalized = value.casefold()
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*", normalized):
        raise AcquisitionConfigurationError(f"{field} is not a canonical DNS hostname")
    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        pass
    else:
        raise AcquisitionConfigurationError(f"{field} must be a DNS hostname, not an IP literal")
    return normalized


def _unique_texts(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    normalized = tuple(_text(value, field=field, maximum=128) for value in values)
    if len(normalized) != len(set(normalized)):
        raise AcquisitionConfigurationError(f"{field} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounded, idempotent-GET retry semantics."""

    initial_delay_seconds: float = 1.0
    maximum_delay_seconds: float = 32.0
    maximum_retries: int = 3
    jitter_fraction: float = 0.25
    maximum_retry_after_seconds: float = 32.0

    def __post_init__(self) -> None:
        if not 0 < self.initial_delay_seconds <= 32:
            raise AcquisitionConfigurationError("initial retry delay is outside the bound")
        if not self.initial_delay_seconds <= self.maximum_delay_seconds <= 32:
            raise AcquisitionConfigurationError("maximum retry delay is outside the bound")
        if isinstance(self.maximum_retries, bool) or not 0 <= self.maximum_retries <= 3:
            raise AcquisitionConfigurationError("maximum retries must be between 0 and 3")
        if not 0 <= self.jitter_fraction <= 1:
            raise AcquisitionConfigurationError("retry jitter must be between 0 and 1")
        if not 0 <= self.maximum_retry_after_seconds <= 32:
            raise AcquisitionConfigurationError("maximum Retry-After delay is outside the bound")

    def to_dict(self) -> dict[str, float | int]:
        return {
            "initial_delay_seconds": self.initial_delay_seconds,
            "maximum_delay_seconds": self.maximum_delay_seconds,
            "maximum_retries": self.maximum_retries,
            "jitter_fraction": self.jitter_fraction,
            "maximum_retry_after_seconds": self.maximum_retry_after_seconds,
        }


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """One exact server-configured route; it is never model supplied."""

    route_id: str
    host: str
    path_template: str = "/"
    port: int = 443
    scheme: str = "https"
    path_prefix: str | None = None
    allowed_methods: tuple[str, ...] = ("GET",)
    allowed_query_fields: tuple[str, ...] = ()
    allowed_header_names: tuple[str, ...] = ("accept", "accept-encoding", "user-agent")
    accepted_content_types: tuple[str, ...] = ("application/json",)
    access_basis: str = "configured-provider-access"
    license_status: str = "not-redistribution-asserted"
    redistribution_basis: str = "not-asserted"
    access_profile_ref: str | None = None
    route_policy_version: str = "v1"
    provider_id: str = ""

    def __post_init__(self) -> None:
        validate_identifier(self.route_id, field="route_id")
        object.__setattr__(self, "host", _canonical_config_host(self.host))
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise AcquisitionConfigurationError("route port must be between 1 and 65535")
        if not isinstance(self.scheme, str) or self.scheme.casefold() != "https":
            raise AcquisitionConfigurationError("CORE-03 routes require HTTPS")
        object.__setattr__(self, "scheme", "https")
        for value, field_name in ((self.path_template, "path_template"), (self.path_prefix or "", "path_prefix")):
            if value and (not value.startswith("/") or any(char in value for char in "\x00?#\\")):
                raise AcquisitionConfigurationError(f"{field_name} must be a safe absolute path")
            if value and any(part == ".." for part in value.split("/")):
                raise AcquisitionConfigurationError(f"{field_name} cannot contain parent traversal")
        if self.path_prefix is None:
            static = self.path_template.split("{", 1)[0]
            object.__setattr__(self, "path_prefix", static or "/")
        if not self.path_prefix:
            raise AcquisitionConfigurationError("path_prefix is required")
        if any(not isinstance(method, str) for method in self.allowed_methods):
            raise AcquisitionConfigurationError("route methods must be strings")
        methods = tuple(method.upper() for method in self.allowed_methods)
        if methods != ("GET",):
            raise AcquisitionConfigurationError("the initial acquisition route only permits GET")
        object.__setattr__(self, "allowed_methods", methods)
        query_fields = _unique_texts(self.allowed_query_fields, field="allowed_query_fields")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", value) for value in query_fields):
            raise AcquisitionConfigurationError("route query field names are malformed")
        object.__setattr__(self, "allowed_query_fields", query_fields)
        header_names = tuple(value.casefold() for value in self.allowed_header_names)
        if any(not re.fullmatch(r"[a-z0-9-]{1,64}", value) for value in header_names):
            raise AcquisitionConfigurationError("route header names are malformed")
        object.__setattr__(self, "allowed_header_names", _unique_texts(header_names, field="allowed_header_names"))
        media = tuple(value.casefold().split(";", 1)[0].strip() for value in self.accepted_content_types)
        if not media or any(value not in ACCEPTED_MEDIA_TYPES for value in media):
            raise AcquisitionConfigurationError("route has an unsupported accepted content type")
        object.__setattr__(self, "accepted_content_types", _unique_texts(media, field="accepted_content_types"))
        for value, field_name in (
            (self.access_basis, "access_basis"),
            (self.license_status, "license_status"),
            (self.redistribution_basis, "redistribution_basis"),
            (self.route_policy_version, "route_policy_version"),
        ):
            _text(value, field=field_name, maximum=256)
        if self.access_profile_ref is not None:
            object.__setattr__(
                self,
                "access_profile_ref",
                validate_reference(self.access_profile_ref, field="access_profile_ref"),
            )
        if self.provider_id:
            validate_identifier(self.provider_id, field="provider_id")

    def with_provider(self, provider_id: str) -> "ProviderRoute":
        if self.provider_id and self.provider_id != provider_id:
            raise AcquisitionConfigurationError("route is bound to a different provider")
        return replace(self, provider_id=provider_id)

    def render_path(self, values: Mapping[str, str] | None = None) -> str:
        rendered = self.path_template
        for key, value in (values or {}).items():
            if not isinstance(value, str) or any(char in value for char in "\x00\r\n?#\\"):
                raise AcquisitionConfigurationError("route path value is malformed")
            rendered = rendered.replace("{" + key + "}", value)
        if "{" in rendered or "}" in rendered:
            raise AcquisitionConfigurationError("route path has an unresolved placeholder")
        if not rendered.startswith("/") or any(part == ".." for part in rendered.split("/")):
            raise AcquisitionConfigurationError("rendered route path is unsafe")
        return rendered

    def to_binding_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "route_id": self.route_id,
            "host": self.host,
            "port": self.port,
            "scheme": self.scheme,
            "path_template": self.path_template,
            "path_prefix": self.path_prefix,
            "allowed_methods": list(self.allowed_methods),
            "allowed_query_fields": list(self.allowed_query_fields),
            "allowed_header_names": list(self.allowed_header_names),
            "accepted_content_types": list(self.accepted_content_types),
            "access_basis": self.access_basis,
            "license_status": self.license_status,
            "redistribution_basis": self.redistribution_basis,
            "access_profile_ref": self.access_profile_ref,
            "route_policy_version": self.route_policy_version,
        }


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Closed network/provider semantics with no secret material."""

    provider_id: str
    provider_class: str | ProviderClass
    routes: tuple[ProviderRoute, ...]
    connect_timeout_seconds: float = MAX_CONNECT_TIMEOUT_SECONDS
    read_timeout_seconds: float = MAX_READ_TIMEOUT_SECONDS
    total_timeout_seconds: float = MAX_TOTAL_TIMEOUT_SECONDS
    max_response_bytes: int = MAX_RESPONSE_BYTES
    max_concurrent_requests: int = 2
    requests_per_second: float = 1.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    access_profile_ref: str | None = None
    route_policy_version: str = "v1"
    logical_access_policy: str = "configured-server-owned-access"

    def __post_init__(self) -> None:
        validate_identifier(self.provider_id, field="provider_id")
        object.__setattr__(self, "provider_class", _enum_value(self.provider_class, ProviderClass, field="provider_class"))
        if not isinstance(self.retry_policy, RetryPolicy):
            raise AcquisitionConfigurationError("retry_policy must be a RetryPolicy")
        for value, maximum, field_name in (
            (self.connect_timeout_seconds, MAX_CONNECT_TIMEOUT_SECONDS, "connect_timeout_seconds"),
            (self.read_timeout_seconds, MAX_READ_TIMEOUT_SECONDS, "read_timeout_seconds"),
            (self.total_timeout_seconds, MAX_TOTAL_TIMEOUT_SECONDS, "total_timeout_seconds"),
        ):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= maximum:
                raise AcquisitionConfigurationError(f"{field_name} is outside the frozen maximum")
        if isinstance(self.max_response_bytes, bool) or not 0 < self.max_response_bytes <= MAX_RESPONSE_BYTES:
            raise AcquisitionConfigurationError("max_response_bytes is outside the frozen maximum")
        if isinstance(self.max_concurrent_requests, bool) or not 0 < self.max_concurrent_requests <= 2:
            raise AcquisitionConfigurationError("max_concurrent_requests is outside the frozen maximum")
        if not isinstance(self.requests_per_second, (int, float)) or isinstance(self.requests_per_second, bool) or not 0 < self.requests_per_second <= 1:
            raise AcquisitionConfigurationError("requests_per_second is outside the frozen maximum")
        _text(self.route_policy_version, field="route_policy_version", maximum=128)
        _text(self.logical_access_policy, field="logical_access_policy", maximum=256)
        if self.access_profile_ref is not None:
            object.__setattr__(
                self,
                "access_profile_ref",
                validate_reference(self.access_profile_ref, field="access_profile_ref"),
            )
        normalized_routes: list[ProviderRoute] = []
        seen: set[str] = set()
        for route in self.routes:
            if not isinstance(route, ProviderRoute):
                raise AcquisitionConfigurationError("routes must contain ProviderRoute values")
            route = route.with_provider(self.provider_id)
            if route.route_id in seen:
                raise AcquisitionConfigurationError("provider route IDs must be unique")
            seen.add(route.route_id)
            normalized_routes.append(route)
        if not normalized_routes:
            raise AcquisitionConfigurationError("provider must expose at least one route")
        object.__setattr__(self, "routes", tuple(normalized_routes))

    def route(self, route_id: str) -> ProviderRoute:
        validate_identifier(route_id, field="route_id")
        for route in self.routes:
            if route.route_id == route_id:
                return route
        raise AcquisitionConfigurationError(f"unknown route {route_id!r} for provider {self.provider_id!r}")

    def to_binding_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_class": self.provider_class,
            "routes": [route.to_binding_dict() for route in self.routes],
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "max_response_bytes": self.max_response_bytes,
            "max_concurrent_requests": self.max_concurrent_requests,
            "requests_per_second": self.requests_per_second,
            "retry_policy": self.retry_policy.to_dict(),
            "access_profile_ref": self.access_profile_ref,
            "route_policy_version": self.route_policy_version,
            "logical_access_policy": self.logical_access_policy,
        }

    @property
    def config_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_binding_dict()))

    @property
    def digest(self) -> str:
        return self.config_digest


@dataclass(frozen=True, slots=True)
class AcquisitionConfig:
    """Immutable aggregate of all provider and full-text route semantics."""

    providers: tuple[ProviderConfig, ...] = ()
    full_text_routes: tuple[ProviderRoute, ...] = ()
    route_policy_version: str = "v1"
    max_query_length: int = MAX_QUERY_LENGTH
    max_result_limit: int = MAX_RESULT_LIMIT

    def __post_init__(self) -> None:
        _text(self.route_policy_version, field="route_policy_version", maximum=128)
        if isinstance(self.max_query_length, bool) or not 1 <= self.max_query_length <= MAX_QUERY_LENGTH:
            raise AcquisitionConfigurationError("max_query_length is outside the frozen bound")
        if isinstance(self.max_result_limit, bool) or not 1 <= self.max_result_limit <= MAX_RESULT_LIMIT:
            raise AcquisitionConfigurationError("max_result_limit is outside the frozen bound")
        normalized: list[ProviderConfig] = []
        seen: set[str] = set()
        for provider in self.providers:
            if not isinstance(provider, ProviderConfig):
                raise AcquisitionConfigurationError("providers must contain ProviderConfig values")
            if provider.provider_id in seen:
                raise AcquisitionConfigurationError("provider IDs must be unique")
            seen.add(provider.provider_id)
            normalized.append(provider)
        object.__setattr__(self, "providers", tuple(normalized))
        routes: list[ProviderRoute] = []
        seen_routes: set[tuple[str, str]] = set()
        for route in self.full_text_routes:
            if not isinstance(route, ProviderRoute) or not route.provider_id:
                raise AcquisitionConfigurationError("full-text routes need a server-owned provider binding")
            if route.provider_id not in seen:
                raise AcquisitionConfigurationError("full-text route references an unknown provider")
            if (route.provider_id, route.route_id) in seen_routes:
                raise AcquisitionConfigurationError("full-text route IDs must be unique per provider")
            seen_routes.add((route.provider_id, route.route_id))
            routes.append(route)
        for provider in normalized:
            if provider.provider_class == ProviderClass.FULL_TEXT.value:
                for route in provider.routes:
                    if (provider.provider_id, route.route_id) not in seen_routes:
                        routes.append(route)
                        seen_routes.add((provider.provider_id, route.route_id))
        object.__setattr__(self, "full_text_routes", tuple(routes))

    def providers_for(self, provider_class: ProviderClass | str) -> tuple[ProviderConfig, ...]:
        value = _enum_value(provider_class, ProviderClass, field="provider_class")
        return tuple(provider for provider in self.providers if provider.provider_class == value)

    def provider(self, provider_id: str) -> ProviderConfig:
        validate_identifier(provider_id, field="provider_id")
        for provider in self.providers:
            if provider.provider_id == provider_id:
                return provider
        raise AcquisitionConfigurationError(f"unknown provider {provider_id!r}")

    def to_binding_dict(self) -> dict[str, Any]:
        return {
            "route_policy_version": self.route_policy_version,
            "max_query_length": self.max_query_length,
            "max_result_limit": self.max_result_limit,
            "providers": [provider.to_binding_dict() for provider in self.providers],
            "full_text_routes": [route.to_binding_dict() for route in self.full_text_routes],
        }

    @property
    def config_digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_binding_dict()))

    @property
    def digest(self) -> str:
        return self.config_digest


@dataclass(frozen=True, slots=True, repr=False)
class EphemeralAccessMaterial:
    """Host-only request material; it is deliberately not JSON serializable."""

    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    query_params: Mapping[str, str] = field(default_factory=dict, repr=False)
    secret_values: tuple[str, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        for name, values in (("headers", self.headers), ("query_params", self.query_params)):
            if not isinstance(values, Mapping):
                raise AcquisitionConfigurationError(f"access {name} must be a mapping")
            normalized: dict[str, str] = {}
            for key, value in values.items():
                if not isinstance(key, str) or not key.strip() or not isinstance(value, str):
                    raise AcquisitionConfigurationError("access material names and values must be text")
                if any(char in key + value for char in "\x00\r\n"):
                    raise AcquisitionConfigurationError("access material contains a control character")
                normalized[key.casefold() if name == "headers" else key] = value
            object.__setattr__(self, name, MappingProxyType(normalized))
        if isinstance(self.secret_values, (str, bytes)):
            raise AcquisitionConfigurationError("secret_values must be a sequence of strings")
        try:
            supplied_secrets = tuple(self.secret_values)
        except TypeError as exc:
            raise AcquisitionConfigurationError(
                "secret_values must be a sequence of strings"
            ) from exc
        secrets = tuple(value for value in supplied_secrets if isinstance(value, str) and value)
        if len(secrets) != len(supplied_secrets):
            raise AcquisitionConfigurationError("secret_values must contain non-empty strings")
        object.__setattr__(self, "secret_values", secrets)

    def __repr__(self) -> str:
        return "EphemeralAccessMaterial(<redacted>)"

    def all_secret_values(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.secret_values, *self.headers.values(), *self.query_params.values())))


@dataclass(frozen=True, slots=True)
class MetadataRecord:
    """Small source-neutral bibliographic projection; not a document parser."""

    provider: str
    provider_record_id: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    publication_year: int | None = None
    publication_date: str | None = None
    venue: str | None = None
    work_type: str | None = None
    oa_status: str | None = None
    license_hint: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.provider, field="metadata provider")
        for value, field_name in (
            (self.provider_record_id, "provider_record_id"),
            (self.doi, "doi"),
        ):
            if value is not None:
                validate_reference(value, field=field_name)
        for value, field_name in (
            (self.title, "title"),
            (self.publication_date, "publication_date"),
            (self.venue, "venue"),
            (self.work_type, "work_type"),
            (self.oa_status, "oa_status"),
            (self.license_hint, "license_hint"),
        ):
            if value is not None:
                _text(value, field=field_name, maximum=2_000)
        authors = tuple(_text(value, field="author", maximum=512) for value in self.authors)
        if len(authors) > 100:
            raise AcquisitionIntegrityError("metadata author list is too large")
        object.__setattr__(self, "authors", authors)
        if self.publication_year is not None and (
            isinstance(self.publication_year, bool)
            or not isinstance(self.publication_year, int)
            or not 1000 <= self.publication_year <= 3000
        ):
            raise AcquisitionIntegrityError("metadata publication year is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "provider_record_id": self.provider_record_id,
            "doi": self.doi,
            "title": self.title,
            "authors": list(self.authors),
            "publication_year": self.publication_year,
            "publication_date": self.publication_date,
            "venue": self.venue,
            "work_type": self.work_type,
            "oa_status": self.oa_status,
            "license_hint": self.license_hint,
        }


@dataclass(frozen=True, slots=True)
class SourceCandidate:
    """Untrusted URL suggestion returned by an OA/provider response."""

    url: str
    content_type_hint: str | None = None
    license_status: str | None = None
    access_status: str | None = None
    source_kind: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        _text(self.url, field="source candidate URL", maximum=8_192)
        for value, field_name in (
            (self.content_type_hint, "content_type_hint"),
            (self.license_status, "candidate license_status"),
            (self.access_status, "candidate access_status"),
            (self.source_kind, "candidate source_kind"),
            (self.version, "candidate version"),
        ):
            if value is not None:
                _text(value, field=field_name, maximum=512)


@dataclass(frozen=True, slots=True)
class AcquisitionProvenance:
    """Immutable, sanitized occurrence provenance for one acquired body."""

    schema_version: str
    acquisition_id: str
    provider: str
    provider_config_digest: str
    route_policy_version: str
    request_identity: str
    canonical_identifier: str
    source_url: str
    resolved_url: str
    redirect_chain: tuple[str, ...]
    response_status: int
    retrieved_at: str
    access_status: str
    license_status: str
    access_basis: str
    redistribution_basis: str
    artifact_class: str | ArtifactClass
    content_type: str
    content_family: str | ContentFamily
    content_sha256: str
    content_artifact_id: str
    cache_identity: str
    cache_status: str
    access_profile_ref: str | None = None
    evaluated_candidates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.schema_version, field="provenance schema_version")
        validate_identifier(self.acquisition_id, field="acquisition_id")
        validate_identifier(self.provider, field="provenance provider")
        validate_digest_reference(self.provider_config_digest, field="provider_config_digest")
        _text(self.route_policy_version, field="route_policy_version", maximum=128)
        validate_reference(self.request_identity, field="request_identity")
        validate_reference(self.canonical_identifier, field="canonical_identifier")
        for value, field_name in (
            (self.source_url, "source_url"),
            (self.resolved_url, "resolved_url"),
        ):
            _text(value, field=field_name, maximum=8_192)
            _validate_recorded_url(value, field=field_name)
        object.__setattr__(self, "redirect_chain", tuple(_text(value, field="redirect URL", maximum=8_192) for value in self.redirect_chain))
        for value in self.redirect_chain:
            _validate_recorded_url(value, field="redirect URL")
        if isinstance(self.response_status, bool) or not isinstance(self.response_status, int) or not 100 <= self.response_status <= 599:
            raise AcquisitionIntegrityError("response status is invalid")
        object.__setattr__(self, "retrieved_at", normalize_timestamp(self.retrieved_at, field="retrieved_at"))
        for value, field_name in (
            (self.access_status, "access_status"),
            (self.license_status, "license_status"),
            (self.access_basis, "access_basis"),
            (self.redistribution_basis, "redistribution_basis"),
            (self.content_type, "content_type"),
            (self.cache_status, "cache_status"),
        ):
            _text(value, field=field_name, maximum=512)
        artifact_class = _enum_value(self.artifact_class, ArtifactClass, field="artifact_class")
        if artifact_class in {
            ArtifactClass.RUNTIME_SECRET.value,
            ArtifactClass.CREDENTIAL_REFERENCE.value,
        }:
            raise AcquisitionIntegrityError(
                "runtime secrets and credential references cannot enter durable provenance"
            )
        object.__setattr__(self, "artifact_class", artifact_class)
        content_family = _enum_value(self.content_family, ContentFamily, field="content_family")
        expected_family = {
            "application/json": ContentFamily.JSON.value,
            "application/xml": ContentFamily.XML.value,
            "text/xml": ContentFamily.XML.value,
            "text/html": ContentFamily.HTML.value,
            "application/pdf": ContentFamily.PDF.value,
        }.get(self.content_type.casefold().split(";", 1)[0].strip())
        if expected_family != content_family:
            raise AcquisitionIntegrityError("content family does not match content type")
        object.__setattr__(self, "content_family", content_family)
        validate_sha256(self.content_sha256, field="content_sha256")
        validate_artifact_id(self.content_artifact_id, field="content_artifact_id")
        if self.content_artifact_id != artifact_id_for_sha256(self.content_sha256):
            raise AcquisitionIntegrityError("content artifact identity does not match its digest")
        validate_digest_reference(self.cache_identity, field="cache_identity")
        object.__setattr__(self, "evaluated_candidates", tuple(_text(value, field="evaluated candidate", maximum=8_192) for value in self.evaluated_candidates))
        if self.access_profile_ref is not None:
            validate_reference(self.access_profile_ref, field="access_profile_ref")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "acquisition_id": self.acquisition_id,
            "provider": self.provider,
            "provider_config_digest": validate_digest_reference(self.provider_config_digest, field="provider_config_digest"),
            "route_policy_version": self.route_policy_version,
            "request_identity": self.request_identity,
            "canonical_identifier": self.canonical_identifier,
            "source_url": self.source_url,
            "resolved_url": self.resolved_url,
            "redirect_chain": list(self.redirect_chain),
            "response_status": self.response_status,
            "retrieved_at": self.retrieved_at,
            "access_status": self.access_status,
            "license_status": self.license_status,
            "access_basis": self.access_basis,
            "redistribution_basis": self.redistribution_basis,
            "artifact_class": self.artifact_class,
            "content_type": self.content_type,
            "content_family": self.content_family,
            "content_sha256": self.content_sha256,
            "content_artifact_id": self.content_artifact_id,
            "cache_identity": validate_digest_reference(self.cache_identity, field="cache_identity"),
            "cache_status": self.cache_status,
            "access_profile_ref": self.access_profile_ref,
            "evaluated_candidates": list(self.evaluated_candidates),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())


def _validate_recorded_url(value: str, *, field: str) -> None:
    """Reject credential-bearing or non-HTTPS URLs in durable provenance."""

    if any(char.isspace() or ord(char) < 32 for char in value):
        raise AcquisitionIntegrityError(f"{field} contains whitespace/control data")
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise AcquisitionIntegrityError(f"{field} is not a valid URL") from exc
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise AcquisitionIntegrityError(f"{field} must be an HTTPS URL")
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise AcquisitionIntegrityError(f"{field} contains forbidden authority data")


__all__ = [
    "ACCEPTED_MEDIA_TYPES",
    "AcquisitionConfig",
    "AcquisitionProvenance",
    "AcquisitionStatus",
    "ArtifactClass",
    "ContentFamily",
    "EphemeralAccessMaterial",
    "MAX_CONNECT_TIMEOUT_SECONDS",
    "MAX_QUERY_LENGTH",
    "MAX_REDIRECTS",
    "MAX_READ_TIMEOUT_SECONDS",
    "MAX_RESPONSE_BYTES",
    "MAX_RESULT_LIMIT",
    "MAX_TOTAL_TIMEOUT_SECONDS",
    "MetadataRecord",
    "ProviderClass",
    "ProviderConfig",
    "ProviderRoute",
    "RetryPolicy",
    "SourceCandidate",
]
