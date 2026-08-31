"""Pure URL, identity, response, and access-policy helpers for CORE-03."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import email.utils
import ipaddress
import re
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, quote, unquote, urlencode, urljoin, urlsplit, urlunsplit

from molly.core.ids import canonical_json_bytes, sha256_bytes

from .errors import AcquisitionIntegrityError, AcquisitionPolicyError, CredentialLeakError
from .models import (
    ACCEPTED_MEDIA_TYPES,
    ArtifactClass,
    ContentFamily,
    ProviderRoute,
)


SECRET_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "email",
        "key",
        "mailto",
        "password",
        "passwd",
        "secret",
        "token",
    }
)
SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "cookie", "proxy-authorization", "set-cookie", "x-api-key"}
)


@dataclass(frozen=True, slots=True)
class CanonicalURL:
    """A URL after exact route validation, suitable for one request."""

    value: str
    hostname: str
    port: int
    path: str
    query: tuple[tuple[str, str], ...]


def normalize_doi(value: str) -> str:
    """Normalize one DOI without treating arbitrary URLs as identifiers."""

    if not isinstance(value, str):
        raise AcquisitionPolicyError("DOI must be text")
    candidate = value.strip()
    if len(candidate) > 512 or not candidate or any(char.isspace() for char in candidate):
        raise AcquisitionPolicyError("DOI is empty, too long, or contains whitespace")
    if any(ord(char) < 32 or ord(char) == 127 for char in candidate):
        raise AcquisitionPolicyError("DOI contains a control character")
    if candidate.casefold().startswith("doi:"):
        candidate = candidate[4:]
    elif candidate.casefold().startswith("https://doi.org/"):
        parsed = urlsplit(candidate)
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise AcquisitionPolicyError("DOI URL contains forbidden authority or fragment data")
        if parsed.hostname is None or parsed.hostname.casefold() != "doi.org":
            raise AcquisitionPolicyError("DOI URL must use the exact doi.org host")
        if parsed.port is not None and parsed.port != 443:
            raise AcquisitionPolicyError("DOI URL uses an alternate port")
        if parsed.query:
            raise AcquisitionPolicyError("DOI URL must not contain a query")
        candidate = parsed.path.lstrip("/")
    elif "://" in candidate:
        raise AcquisitionPolicyError("only the canonical HTTPS DOI URL form is accepted")
    candidate = candidate.strip().casefold()
    if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", candidate):
        raise AcquisitionPolicyError("value is not a bounded DOI")
    if any(part == ".." for part in candidate.split("/")) or "#" in candidate or "?" in candidate:
        raise AcquisitionPolicyError("DOI contains a forbidden path/query component")
    return candidate


def normalize_search_query(value: str, *, maximum: int = 512) -> str:
    import unicodedata

    if not isinstance(value, str):
        raise AcquisitionPolicyError("search query must be text")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized or len(normalized) > maximum or any(ord(char) < 32 for char in normalized):
        raise AcquisitionPolicyError("search query is outside the bounded contract")
    return normalized


def normalize_result_limit(value: Any, *, maximum: int = 20) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise AcquisitionPolicyError(f"result limit must be an integer between 1 and {maximum}")
    return value


def _canonical_hostname(hostname: str) -> str:
    if not isinstance(hostname, str) or not hostname or hostname.endswith("."):
        raise AcquisitionPolicyError("hostname is missing or non-canonical")
    if any(ord(char) > 127 or char.isspace() for char in hostname):
        raise AcquisitionPolicyError("hostname has ambiguous Unicode or whitespace")
    return hostname.casefold()


def _safe_path(path: str) -> str:
    if not path:
        return "/"
    if not path.startswith("/") or any(char in path for char in "\x00\r\n\\"):
        raise AcquisitionPolicyError("URL path is not safe")
    decoded = unquote(path)
    if any(part in {".", ".."} for part in decoded.split("/")):
        raise AcquisitionPolicyError("URL path contains traversal")
    return path


def canonicalize_url(
    value: str,
    *,
    allowed_hosts: Iterable[str],
    allowed_port: int,
    path_prefix: str = "/",
    allowed_query_fields: Iterable[str] = (),
) -> CanonicalURL:
    """Validate a URL against one exact server-owned route."""

    if not isinstance(value, str) or not value or len(value) > 16_384:
        raise AcquisitionPolicyError("URL is missing or too long")
    if any(char.isspace() or ord(char) < 32 for char in value):
        raise AcquisitionPolicyError("URL contains whitespace or a control character")
    try:
        parsed = urlsplit(value)
        hostname = _canonical_hostname(parsed.hostname or "")
        explicit_port = parsed.port
    except (ValueError, UnicodeError) as exc:
        raise AcquisitionPolicyError("URL authority is malformed") from exc
    if parsed.scheme.casefold() != "https":
        raise AcquisitionPolicyError("only HTTPS URLs are allowed")
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise AcquisitionPolicyError("URL userinfo/embedded credentials are forbidden")
    if parsed.fragment:
        raise AcquisitionPolicyError("URL fragments are forbidden")
    if explicit_port is not None and explicit_port != allowed_port:
        raise AcquisitionPolicyError("URL uses an alternate port")
    if hostname not in {_canonical_hostname(host) for host in allowed_hosts}:
        raise AcquisitionPolicyError("URL host is not an exact configured route")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise AcquisitionPolicyError("configured DNS routes do not accept IP-literal URLs")
    path = _safe_path(parsed.path)
    prefix = _safe_path(path_prefix)
    if not (path == prefix or prefix == "/" or path.startswith(prefix.rstrip("/") + "/")):
        raise AcquisitionPolicyError("URL path is outside the exact configured route")
    allowed_fields = {str(item) for item in allowed_query_fields}
    try:
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
    except ValueError as exc:
        raise AcquisitionPolicyError("URL query is malformed") from exc
    if len({key for key, _ in pairs}) != len(pairs):
        raise AcquisitionPolicyError("URL query contains duplicate fields")
    if any(key not in allowed_fields for key, _ in pairs):
        raise AcquisitionPolicyError("URL query contains an unconfigured field")
    pairs = tuple(sorted(pairs))
    netloc = hostname if explicit_port is None or allowed_port == 443 else f"{hostname}:{allowed_port}"
    canonical = urlunsplit(("https", netloc, path, urlencode(pairs), ""))
    return CanonicalURL(canonical, hostname, allowed_port, path, pairs)


def join_route_url(route: ProviderRoute, path: str, query: Mapping[str, Any] | None = None) -> CanonicalURL:
    """Construct and validate a URL using only a configured route."""

    if not path.startswith("/"):
        raise AcquisitionPolicyError("provider route path must be absolute")
    pairs = []
    for key, value in (query or {}).items():
        if not isinstance(key, str) or not isinstance(value, (str, int)) or isinstance(value, bool):
            raise AcquisitionPolicyError("provider query values must be bounded text or integers")
        pairs.append((key, str(value)))
    netloc = route.host if route.port == 443 else f"{route.host}:{route.port}"
    raw = urlunsplit((route.scheme, netloc, path, urlencode(pairs), ""))
    return canonicalize_url(
        raw,
        allowed_hosts=(route.host,),
        allowed_port=route.port,
        path_prefix=route.path_prefix or "/",
        allowed_query_fields=route.allowed_query_fields,
    )


def resolve_redirect(base_url: str, location: str, route: ProviderRoute) -> CanonicalURL:
    if not isinstance(location, str) or not location or len(location) > 16_384:
        raise AcquisitionPolicyError("redirect Location is missing or too long")
    if any(char.isspace() or ord(char) < 32 for char in location):
        raise AcquisitionPolicyError("redirect Location contains whitespace/control data")
    return canonicalize_url(
        urljoin(base_url, location),
        allowed_hosts=(route.host,),
        allowed_port=route.port,
        path_prefix=route.path_prefix or "/",
        allowed_query_fields=route.allowed_query_fields,
    )


def validate_resolved_addresses(hostname: str, addresses: Iterable[str]) -> tuple[str, ...]:
    """Reject every non-public DNS answer before opening a connection."""

    _canonical_hostname(hostname)
    normalized: list[str] = []
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise AcquisitionPolicyError("DNS returned a malformed address") from exc
        if (
            not address.is_global
            or address.is_loopback
            or address.is_private
            or address.is_link_local
            or address.is_multicast
            or address.is_unspecified
            or address.is_reserved
        ):
            raise AcquisitionPolicyError("DNS resolved to a denied non-public address")
        text = str(address)
        if text not in normalized:
            normalized.append(text)
    if not normalized:
        raise AcquisitionPolicyError("DNS returned no validated addresses")
    return tuple(normalized)


def content_family_for_media_type(value: str) -> ContentFamily:
    media = value.casefold().split(";", 1)[0].strip()
    try:
        return {
            "application/json": ContentFamily.JSON,
            "application/xml": ContentFamily.XML,
            "text/xml": ContentFamily.XML,
            "text/html": ContentFamily.HTML,
            "application/pdf": ContentFamily.PDF,
        }[media]
    except KeyError as exc:
        raise AcquisitionPolicyError("content type is outside the acquisition families") from exc


def normalize_content_type(values: str | Iterable[str] | None, *, allowed: Iterable[str]) -> str:
    if values is None:
        raise AcquisitionPolicyError("response Content-Type is required")
    if isinstance(values, str):
        candidates = [values]
    else:
        candidates = list(values)
    normalized = tuple(item.casefold().split(";", 1)[0].strip() for item in candidates)
    if not normalized or any(not item or item not in ACCEPTED_MEDIA_TYPES for item in normalized):
        raise AcquisitionPolicyError("response Content-Type is unsupported")
    if len(set(normalized)) != 1:
        raise AcquisitionPolicyError("response has contradictory Content-Type values")
    result = normalized[0]
    if result not in set(allowed):
        raise AcquisitionPolicyError("response Content-Type is not allowed by the route")
    return result


def sanitize_url(value: str, *, secret_values: Iterable[str] = ()) -> str:
    """Remove credential query values and redact known ephemeral values."""

    secrets = tuple(secret for secret in secret_values if secret)
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "<redacted>")
    try:
        parsed = urlsplit(redacted)
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        safe_pairs = []
        for key, item in pairs:
            normalized_key = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized_key in {re.sub(r"[^a-z0-9]+", "_", item_name).strip("_") for item_name in SECRET_QUERY_NAMES}:
                continue
            safe_pairs.append((key, "<redacted>" if item in secrets else item))
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(safe_pairs), ""))
    except (ValueError, UnicodeError):
        return _redact_text(redacted, secrets)


def _redact_text(value: str, secrets: Iterable[str]) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result[:16_384]


def assert_no_secret_values(value: bytes | str, secret_values: Iterable[str]) -> None:
    payload = value if isinstance(value, bytes) else value.encode("utf-8", "replace")
    for secret in secret_values:
        if secret and secret.encode("utf-8") in payload:
            raise CredentialLeakError("durable acquisition value contains access material")


def classify_route(route: ProviderRoute) -> ArtifactClass:
    """Classify route context without making it an ArtifactRecord property."""

    access = route.access_basis.casefold().strip()
    license_status = route.license_status.casefold().strip()
    redistribution = route.redistribution_basis.casefold().strip()
    if any(token in {"unknown", "denied", "unverified", "not-verified"} for token in (access, license_status)):
        raise AcquisitionPolicyError("route access or license status is not verified")
    if redistribution.startswith("verified") or "public-safe" in redistribution or redistribution in {"cc-by", "cc0", "public-domain"}:
        return ArtifactClass.PUBLIC_ARTIFACT
    if access and license_status:
        return ArtifactClass.PRIVATE_ARTIFACT
    raise AcquisitionPolicyError("route lacks an explicit access basis")


def cache_digest(binding: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(binding))


def parse_retry_after(value: str | None, *, now: datetime | None = None) -> float | None:
    if not value:
        return None
    candidate = value.strip()
    try:
        delay = float(candidate)
        if delay >= 0:
            return delay
    except ValueError:
        pass
    try:
        target = email.utils.parsedate_to_datetime(candidate)
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        reference = now or datetime.now(timezone.utc)
        return max(0.0, (target - reference).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = [
    "CanonicalURL",
    "SECRET_QUERY_NAMES",
    "SENSITIVE_HEADER_NAMES",
    "assert_no_secret_values",
    "cache_digest",
    "canonicalize_url",
    "classify_route",
    "content_family_for_media_type",
    "join_route_url",
    "normalize_content_type",
    "normalize_doi",
    "normalize_result_limit",
    "normalize_search_query",
    "parse_retry_after",
    "resolve_redirect",
    "sanitize_url",
    "validate_resolved_addresses",
]
