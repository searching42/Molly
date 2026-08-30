"""Closed provider adapters for metadata and open-access resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import parse_qsl, quote, urlsplit

from molly.core.ids import canonical_json_bytes, freeze_json_mapping, sha256_bytes, validate_identifier, validate_reference

from .errors import AcquisitionConfigurationError, AcquisitionIntegrityError
from .models import MetadataRecord, ProviderClass, ProviderConfig, ProviderRoute, SourceCandidate
from .policy import canonicalize_url, normalize_doi, normalize_result_limit, normalize_search_query


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    """A server-constructed request shape with no credentials or raw URL input."""

    provider_id: str
    provider_config: ProviderConfig
    route: ProviderRoute
    path: str
    query: Mapping[str, str | int]
    request_shape: Mapping[str, Any]
    canonical_identifier: str | None = None

    def __post_init__(self) -> None:
        validate_identifier(self.provider_id, field="request provider_id")
        if self.provider_config.provider_id != self.provider_id:
            raise AcquisitionConfigurationError("provider request/config provider mismatch")
        if self.route.provider_id != self.provider_id:
            raise AcquisitionConfigurationError("provider request/route mismatch")
        if not self.path.startswith("/") or any(char in self.path for char in "\x00\r\n"):
            raise AcquisitionConfigurationError("provider request path is unsafe")
        object.__setattr__(self, "query", freeze_json_mapping(self.query, field="provider query"))
        object.__setattr__(self, "request_shape", freeze_json_mapping(self.request_shape, field="request shape"))
        if self.canonical_identifier is not None:
            validate_reference(self.canonical_identifier, field="canonical_identifier")

    @property
    def request_identity(self) -> str:
        value = {
            "provider": self.provider_id,
            "route": self.route.route_id,
            "path": self.path,
            "query": dict(self.query),
            "shape": dict(self.request_shape),
            "canonical_identifier": self.canonical_identifier,
        }
        return sha256_bytes(canonical_json_bytes(value))


class MetadataProvider(Protocol):
    provider_id: str

    def search(self, query: str, limit: int) -> ProviderRequest:
        ...

    def lookup(self, doi: str) -> ProviderRequest:
        ...

    def normalize_search(self, body: bytes) -> tuple[MetadataRecord, ...]:
        ...

    def normalize_lookup(self, body: bytes) -> tuple[MetadataRecord, ...]:
        ...


class OpenAccessResolver(Protocol):
    provider_id: str

    def resolve(self, doi: str) -> ProviderRequest:
        ...

    def normalize_resolution(self, body: bytes) -> tuple[SourceCandidate, ...]:
        ...


class FullTextFetcher(Protocol):
    """Interface for a configured route that turns one source into a request."""

    provider_id: str

    def request(self, *, doi: str, source_url: str) -> ProviderRequest:
        ...


def _json_object(body: bytes) -> Mapping[str, Any]:
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionIntegrityError("provider response is not bounded UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise AcquisitionIntegrityError("provider response JSON must be an object")
    return value


def _string(value: Any, *, maximum: int = 2_000) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip()
    if any(char in cleaned for char in "\x00\r\n"):
        return None
    return cleaned[:maximum]


def _doi(value: Any) -> str | None:
    candidate = _string(value, maximum=512)
    if candidate is None:
        return None
    try:
        return normalize_doi(candidate)
    except Exception:
        return None


def _authors_openalex(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    authors: list[str] = []
    for item in value[:100]:
        if not isinstance(item, Mapping):
            continue
        author = item.get("author")
        name = _string(author.get("display_name") if isinstance(author, Mapping) else None, maximum=512)
        if name:
            authors.append(name)
    return tuple(dict.fromkeys(authors))


def _authors_crossref(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    authors: list[str] = []
    for item in value[:100]:
        if not isinstance(item, Mapping):
            continue
        given = _string(item.get("given"), maximum=256) or ""
        family = _string(item.get("family"), maximum=256) or ""
        name = " ".join(part for part in (given, family) if part)
        if name:
            authors.append(name)
    return tuple(dict.fromkeys(authors))


def _openalex_record(item: Mapping[str, Any], *, provider: str = "openalex") -> MetadataRecord:
    location = item.get("primary_location")
    source = location.get("source") if isinstance(location, Mapping) else None
    open_access = item.get("open_access")
    return MetadataRecord(
        provider=provider,
        provider_record_id=_string(item.get("id"), maximum=512),
        doi=_doi(item.get("doi")),
        title=_string(item.get("title")),
        authors=_authors_openalex(item.get("authorships")),
        publication_year=item.get("publication_year") if isinstance(item.get("publication_year"), int) else None,
        publication_date=_string(item.get("publication_date"), maximum=64),
        venue=_string(source.get("display_name") if isinstance(source, Mapping) else None, maximum=512),
        work_type=_string(item.get("type"), maximum=128),
        oa_status=(
            "open"
            if isinstance(open_access, Mapping) and open_access.get("is_oa") is True
            else _string(open_access.get("oa_status") if isinstance(open_access, Mapping) else None, maximum=128)
        ),
        license_hint=_string(open_access.get("license") if isinstance(open_access, Mapping) else None, maximum=256),
    )


def _crossref_record(item: Mapping[str, Any], *, provider: str = "crossref") -> MetadataRecord:
    date = item.get("published-print") or item.get("published-online") or item.get("issued")
    date_parts = date.get("date-parts") if isinstance(date, Mapping) else None
    year = None
    if isinstance(date_parts, Sequence) and date_parts and isinstance(date_parts[0], Sequence) and date_parts[0]:
        candidate = date_parts[0][0]
        year = candidate if isinstance(candidate, int) else None
    title = item.get("title")
    title_value = title[0] if isinstance(title, Sequence) and title and isinstance(title[0], str) else title
    container = item.get("container-title")
    venue = container[0] if isinstance(container, Sequence) and container and isinstance(container[0], str) else container
    return MetadataRecord(
        provider=provider,
        provider_record_id=_string(item.get("DOI"), maximum=512),
        doi=_doi(item.get("DOI")),
        title=_string(title_value),
        authors=_authors_crossref(item.get("author")),
        publication_year=year,
        publication_date=None,
        venue=_string(venue, maximum=512),
        work_type=_string(item.get("type"), maximum=128),
        oa_status=None,
        license_hint=None,
    )


class _JsonMetadataProvider:
    provider_id = ""

    def __init__(self, config: ProviderConfig) -> None:
        if config.provider_class != ProviderClass.METADATA.value:
            raise AcquisitionConfigurationError("metadata adapter requires a METADATA provider")
        self.config = config
        self.provider_id = config.provider_id

    def _route(self, route_id: str) -> ProviderRoute:
        return self.config.route(route_id)

    def _request(
        self,
        route_id: str,
        *,
        path_values: Mapping[str, str] | None = None,
        query: Mapping[str, str | int],
        shape: Mapping[str, Any],
        doi: str | None = None,
    ) -> ProviderRequest:
        route = self._route(route_id)
        path = route.render_path(path_values)
        return ProviderRequest(
            provider_id=self.provider_id,
            provider_config=self.config,
            route=route,
            path=path,
            query=query,
            request_shape=shape,
            canonical_identifier=doi,
        )


class OpenAlexProvider(_JsonMetadataProvider):
    """OpenAlex adapter with only bounded search and DOI lookup shapes."""

    def search(self, query: str, limit: int) -> ProviderRequest:
        normalized = normalize_search_query(query)
        bounded = normalize_result_limit(limit)
        return self._request(
            "search",
            query={"search": normalized, "per-page": bounded},
            shape={"operation": "search", "query": normalized, "limit": bounded},
        )

    def lookup(self, doi: str) -> ProviderRequest:
        normalized = normalize_doi(doi)
        return self._request(
            "lookup",
            query={"filter": f"doi:{normalized}"},
            shape={"operation": "lookup", "doi": normalized},
            doi=normalized,
        )

    def normalize_search(self, body: bytes) -> tuple[MetadataRecord, ...]:
        value = _json_object(body)
        results = value.get("results", ())
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            raise AcquisitionIntegrityError("OpenAlex results is not an array")
        return tuple(
            _openalex_record(item, provider=self.provider_id)
            for item in results[:20]
            if isinstance(item, Mapping)
        )

    def normalize_lookup(self, body: bytes) -> tuple[MetadataRecord, ...]:
        value = _json_object(body)
        item = value.get("results", [value])
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            return tuple(
                _openalex_record(entry, provider=self.provider_id)
                for entry in item[:20]
                if isinstance(entry, Mapping)
            )
        return (_openalex_record(item, provider=self.provider_id),) if isinstance(item, Mapping) else ()


class CrossrefProvider(_JsonMetadataProvider):
    """Crossref adapter with only bounded bibliographic search and lookup."""

    def search(self, query: str, limit: int) -> ProviderRequest:
        normalized = normalize_search_query(query)
        bounded = normalize_result_limit(limit)
        return self._request(
            "search",
            query={"query.bibliographic": normalized, "rows": bounded},
            shape={"operation": "search", "query": normalized, "limit": bounded},
        )

    def lookup(self, doi: str) -> ProviderRequest:
        normalized = normalize_doi(doi)
        return self._request(
            "lookup",
            path_values={"doi": quote(normalized, safe="")},
            query={},
            shape={"operation": "lookup", "doi": normalized},
            doi=normalized,
        )

    def normalize_search(self, body: bytes) -> tuple[MetadataRecord, ...]:
        value = _json_object(body)
        message = value.get("message")
        items = message.get("items", ()) if isinstance(message, Mapping) else ()
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise AcquisitionIntegrityError("Crossref items is not an array")
        return tuple(
            _crossref_record(item, provider=self.provider_id)
            for item in items[:20]
            if isinstance(item, Mapping)
        )

    def normalize_lookup(self, body: bytes) -> tuple[MetadataRecord, ...]:
        value = _json_object(body)
        item = value.get("message", value)
        return (_crossref_record(item, provider=self.provider_id),) if isinstance(item, Mapping) else ()


class UnpaywallResolver:
    """DOI-only OA resolver; returned locations remain untrusted candidates."""

    provider_id = "unpaywall"

    def __init__(self, config: ProviderConfig) -> None:
        if config.provider_class != ProviderClass.OA_RESOLUTION.value:
            raise AcquisitionConfigurationError("Unpaywall adapter requires an OA_RESOLUTION provider")
        self.config = config
        self.provider_id = config.provider_id

    def resolve(self, doi: str) -> ProviderRequest:
        normalized = normalize_doi(doi)
        route = self.config.route("lookup")
        return ProviderRequest(
            provider_id=self.config.provider_id,
            provider_config=self.config,
            route=route,
            path=route.render_path({"doi": quote(normalized, safe="")}),
            query={},
            request_shape={"operation": "resolve", "doi": normalized},
            canonical_identifier=normalized,
        )

    def normalize_resolution(self, body: bytes) -> tuple[SourceCandidate, ...]:
        value = _json_object(body)
        locations: list[Mapping[str, Any]] = []
        best = value.get("best_oa_location")
        if isinstance(best, Mapping):
            locations.append(best)
        raw_locations = value.get("oa_locations", ())
        if isinstance(raw_locations, Sequence) and not isinstance(raw_locations, (str, bytes)):
            locations.extend(item for item in raw_locations if isinstance(item, Mapping))
        result: list[SourceCandidate] = []
        seen: set[str] = set()
        top_license = _string(value.get("license"), maximum=256)
        for item in locations:
            candidate_url = item.get("url_for_pdf") or item.get("url")
            url = _string(candidate_url, maximum=8_192)
            if not url or url in seen:
                continue
            seen.add(url)
            url_type = (_string(item.get("url_type"), maximum=32) or "").casefold()
            content_hint = {
                "pdf": "application/pdf",
                "xml": "application/xml",
                "jats": "application/xml",
                "html": "text/html",
            }.get(url_type)
            result.append(
                SourceCandidate(
                    url=url,
                    content_type_hint=content_hint,
                    license_status=_string(item.get("license"), maximum=256) or top_license,
                    access_status="open",
                    source_kind=_string(item.get("host_type"), maximum=128),
                    version=_string(item.get("version"), maximum=128),
                )
            )
            if len(result) >= 20:
                break
        return tuple(result)


@dataclass(frozen=True, slots=True)
class ConfiguredFullTextFetcher:
    """Build a request only for one server-owned, exact full-text route."""

    provider_config: ProviderConfig
    route: ProviderRoute

    def __post_init__(self) -> None:
        if self.provider_config.provider_class != ProviderClass.FULL_TEXT.value:
            raise AcquisitionConfigurationError(
                "full-text fetcher requires a FULL_TEXT provider configuration"
            )
        if self.route.provider_id != self.provider_config.provider_id:
            raise AcquisitionConfigurationError("full-text fetcher route/provider mismatch")
        self.provider_config.route(self.route.route_id)

    @property
    def provider_id(self) -> str:
        return self.provider_config.provider_id

    def request(self, *, doi: str, source_url: str) -> ProviderRequest:
        canonical_doi = normalize_doi(doi)
        canonical = canonicalize_url(
            source_url,
            allowed_hosts=(self.route.host,),
            allowed_port=self.route.port,
            path_prefix=self.route.path_prefix or "/",
            allowed_query_fields=self.route.allowed_query_fields,
        )
        return ProviderRequest(
            provider_id=self.provider_id,
            provider_config=self.provider_config,
            route=self.route,
            path=canonical.path,
            query=dict(canonical.query),
            request_shape={
                "operation": "full_text",
                "doi": canonical_doi,
                "source_route": self.route.route_id,
                "source_url": canonical.value,
            },
            canonical_identifier=canonical_doi,
        )


__all__ = [
    "CrossrefProvider",
    "ConfiguredFullTextFetcher",
    "FullTextFetcher",
    "MetadataProvider",
    "OpenAccessResolver",
    "OpenAlexProvider",
    "ProviderRequest",
    "UnpaywallResolver",
]
