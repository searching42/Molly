"""Cache-first acquisition orchestration behind the CORE-02 tool boundary."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping, Sequence
import re
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlsplit

from molly.core.artifacts import ArtifactStore
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    new_server_id,
    sha256_bytes,
    thaw_json,
    utc_timestamp,
)
from molly.core.tools import MAX_TOOL_RESULT_DATA_BYTES, ArtifactDraft, ToolResult

from .cache import AcquisitionCache, CacheEntry, CachedResponse
from .errors import (
    AcquisitionCacheError,
    AcquisitionConfigurationError,
    AcquisitionIntegrityError,
    AcquisitionPolicyError,
    AcquisitionTransportError,
)
from .models import (
    AcquisitionConfig,
    AcquisitionProvenance,
    AcquisitionStatus,
    ArtifactClass,
    ContentFamily,
    EphemeralAccessMaterial,
    MetadataRecord,
    ProviderClass,
    ProviderConfig,
    ProviderRoute,
    SourceCandidate,
)
from .policy import (
    SECRET_QUERY_NAMES,
    assert_no_secret_values,
    cache_digest,
    canonicalize_url,
    classify_route,
    content_family_for_media_type,
    join_route_url,
    normalize_content_type,
    normalize_doi,
    normalize_result_limit,
    normalize_search_query,
    sanitize_url,
)
from .providers import (
    CrossrefProvider,
    ConfiguredFullTextFetcher,
    MetadataProvider,
    OpenAccessResolver,
    OpenAlexProvider,
    ProviderRequest,
    UnpaywallResolver,
)
from .transport import NetworkResponse, NetworkTransport, SafeNetworkTransport


class AccessProfileResolver(Protocol):
    """Host-only resolver for ephemeral request material."""

    def resolve(self, access_profile_ref: str) -> EphemeralAccessMaterial:
        ...


class EmptyAccessProfileResolver:
    def resolve(self, access_profile_ref: str) -> EphemeralAccessMaterial:
        raise AcquisitionConfigurationError(
            f"no server-owned access profile is configured for {access_profile_ref!r}"
        )


class StaticAccessProfileResolver:
    """Small host-side test/application resolver; its values never serialize."""

    def __init__(self, profiles: Mapping[str, EphemeralAccessMaterial]) -> None:
        self._profiles = dict(profiles)

    def resolve(self, access_profile_ref: str) -> EphemeralAccessMaterial:
        try:
            material = self._profiles[access_profile_ref]
        except KeyError as exc:
            raise AcquisitionConfigurationError(
                f"unknown server-owned access profile {access_profile_ref!r}"
            ) from exc
        if not isinstance(material, EphemeralAccessMaterial):
            raise AcquisitionConfigurationError("access profile resolver returned invalid material")
        return material


@dataclass(frozen=True, slots=True, repr=False)
class _FetchedResponse:
    response: NetworkResponse
    content_type: str
    content_family: ContentFamily
    cache_identity: str
    cache_status: str
    artifact_class: ArtifactClass

    def __repr__(self) -> str:
        return (
            f"_FetchedResponse(status={self.response.status_code}, "
            f"body_bytes={len(self.response.body)}, cache_status={self.cache_status!r})"
        )


@dataclass(frozen=True, slots=True)
class _EligibleSource:
    candidate: SourceCandidate
    route: ProviderRoute
    provider_config: ProviderConfig
    canonical_url: str
    artifact_class: ArtifactClass
    priority: int


def _provider_from_config(config: ProviderConfig) -> MetadataProvider | OpenAccessResolver | None:
    normalized = config.provider_id.casefold()
    if config.provider_class == ProviderClass.METADATA.value:
        if normalized in {"openalex", "openalex-api"} or "openalex" in normalized:
            return OpenAlexProvider(config)
        if normalized in {"crossref", "crossref-api"} or "crossref" in normalized:
            return CrossrefProvider(config)
    if config.provider_class == ProviderClass.OA_RESOLUTION.value:
        if normalized in {"unpaywall", "unpaywall-api"} or "unpaywall" in normalized:
            return UnpaywallResolver(config)
    return None


class AcquisitionService:
    """Server-owned metadata, OA resolution, and full-text service."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        cache: AcquisitionCache,
        config: AcquisitionConfig,
        transport: NetworkTransport | None = None,
        access_profiles: AccessProfileResolver | None = None,
        metadata_providers: Sequence[MetadataProvider] = (),
        oa_resolvers: Sequence[OpenAccessResolver] = (),
        clock: Any | None = None,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise AcquisitionConfigurationError("AcquisitionService requires an ArtifactStore")
        if not isinstance(cache, AcquisitionCache):
            raise AcquisitionConfigurationError("AcquisitionService requires an AcquisitionCache")
        if not isinstance(config, AcquisitionConfig):
            raise AcquisitionConfigurationError("AcquisitionService requires an AcquisitionConfig")
        self.artifact_store = artifact_store
        self.cache = cache
        self.config = config
        self.transport = transport or SafeNetworkTransport()
        self.access_profiles = access_profiles or EmptyAccessProfileResolver()
        self._clock = clock

        configured_metadata: list[MetadataProvider] = list(metadata_providers)
        configured_resolvers: list[OpenAccessResolver] = list(oa_resolvers)
        for provider_config in config.providers:
            provider = _provider_from_config(provider_config)
            if provider is None:
                continue
            if provider_config.provider_class == ProviderClass.METADATA.value:
                configured_metadata.append(provider)  # type: ignore[arg-type]
            elif provider_config.provider_class == ProviderClass.OA_RESOLUTION.value:
                configured_resolvers.append(provider)  # type: ignore[arg-type]
        self.metadata_providers = self._unique_providers(configured_metadata)
        self.oa_resolvers = self._unique_providers(configured_resolvers)

    @staticmethod
    def _unique_providers(values: Sequence[Any]) -> tuple[Any, ...]:
        result: list[Any] = []
        seen: set[str] = set()
        for value in values:
            provider_id = getattr(value, "provider_id", None)
            if not isinstance(provider_id, str) or not provider_id:
                raise AcquisitionConfigurationError("provider adapter lacks a stable provider_id")
            if provider_id not in seen:
                result.append(value)
                seen.add(provider_id)
        return tuple(result)

    @property
    def config_digest(self) -> str:
        return self.config.config_digest

    def _provider_config(self, provider_id: str) -> ProviderConfig:
        return self.config.provider(provider_id)

    def _access_material(
        self, route: ProviderRoute, config: ProviderConfig
    ) -> EphemeralAccessMaterial:
        access_ref = route.access_profile_ref or config.access_profile_ref
        if access_ref is None:
            return EphemeralAccessMaterial()
        material = self.access_profiles.resolve(access_ref)
        if not isinstance(material, EphemeralAccessMaterial):
            raise AcquisitionConfigurationError("access profile resolver returned invalid material")
        return material

    @staticmethod
    def _request_headers(route: ProviderRoute, material: EphemeralAccessMaterial) -> dict[str, str]:
        headers: dict[str, str] = {}
        if "accept-encoding" in route.allowed_header_names:
            headers["accept-encoding"] = "identity"
        if "user-agent" in route.allowed_header_names:
            headers["user-agent"] = "molly-core-v2/1"
        if "accept" in route.allowed_header_names:
            headers["accept"] = ", ".join(route.accepted_content_types)
        allowed = set(route.allowed_header_names)
        for key, value in material.headers.items():
            if key.casefold() not in allowed:
                raise AcquisitionPolicyError("access profile header is not allowed by the route")
            headers[key.casefold()] = value
        return headers

    @staticmethod
    def _request_url(
        request: ProviderRequest, material: EphemeralAccessMaterial
    ) -> tuple[str, str | None]:
        query = dict(request.query)
        for key, value in material.query_params.items():
            if key in query or key not in request.route.allowed_query_fields:
                raise AcquisitionPolicyError("access profile query field is not allowed by the route")
            query[key] = value
        actual = join_route_url(request.route, request.path, query).value
        access_ref = request.route.access_profile_ref or request.provider_config.access_profile_ref
        return actual, access_ref

    @staticmethod
    def _safe_source_url(url: str, material: EphemeralAccessMaterial) -> str:
        return sanitize_url(url, secret_values=material.all_secret_values())

    def _candidate_secret_values(self) -> tuple[str, ...]:
        """Collect only host-resolved values needed to redact OA candidates."""

        values: list[str] = []
        for route in self.config.full_text_routes:
            try:
                provider_config = self._provider_config(route.provider_id)
                values.extend(self._access_material(route, provider_config).all_secret_values())
            except AcquisitionConfigurationError:
                # Candidate URLs are untrusted data.  An unavailable optional
                # profile must not make inspection itself a source of authority;
                # the selected route will still fail closed if its profile is
                # actually required for a request.
                continue
        return tuple(dict.fromkeys(values))

    @staticmethod
    def _safe_candidate_url(url: str, secret_values: Iterable[str]) -> str:
        return sanitize_url(url, secret_values=secret_values)

    def _validate_network_response_urls(
        self,
        response: NetworkResponse,
        *,
        request: ProviderRequest,
        requested_url: str,
    ) -> None:
        """Re-check transport output before any response enters cache/data."""

        route = request.route
        expected = canonicalize_url(
            requested_url,
            allowed_hosts=(route.host,),
            allowed_port=route.port,
            path_prefix=route.path_prefix or "/",
            allowed_query_fields=route.allowed_query_fields,
        )
        actual_requested = canonicalize_url(
            response.requested_url,
            allowed_hosts=(route.host,),
            allowed_port=route.port,
            path_prefix=route.path_prefix or "/",
            allowed_query_fields=route.allowed_query_fields,
        )
        if actual_requested.value != expected.value:
            raise AcquisitionPolicyError("transport requested URL does not match the server request")
        for value in response.redirect_chain:
            canonicalize_url(
                value,
                allowed_hosts=(route.host,),
                allowed_port=route.port,
                path_prefix=route.path_prefix or "/",
                allowed_query_fields=route.allowed_query_fields,
            )
        canonicalize_url(
            response.resolved_url,
            allowed_hosts=(route.host,),
            allowed_port=route.port,
            path_prefix=route.path_prefix or "/",
            allowed_query_fields=route.allowed_query_fields,
        )

    def _cache_binding(
        self,
        request: ProviderRequest,
        *,
        source_url: str,
        access_profile_ref: str | None,
    ) -> tuple[str, dict[str, Any]]:
        binding: dict[str, Any] = {
            "provider": request.provider_id,
            "provider_config_digest": request.provider_config.config_digest,
            "route_policy_version": request.route.route_policy_version,
            "request_identity": request.request_identity,
            "canonical_identifier": request.canonical_identifier,
            "route_id": request.route.route_id,
            "request_shape": {
                **thaw_json(request.request_shape),
                "access_profile_ref": access_profile_ref,
            },
            "source_url": source_url,
        }
        return cache_digest(binding), binding

    @staticmethod
    def _response_from_cache(cached: CachedResponse) -> NetworkResponse:
        return NetworkResponse(
            status_code=cached.entry.response_status,
            headers={"content-type": (cached.entry.content_type,)},
            body=cached.body,
            requested_url=cached.entry.source_url,
            resolved_url=cached.entry.resolved_url,
            redirect_chain=cached.entry.redirect_chain,
        )

    def _fetch_request(self, request: ProviderRequest) -> _FetchedResponse:
        material = self._access_material(request.route, request.provider_config)
        actual_url, access_ref = self._request_url(request, material)
        safe_source_url = self._safe_source_url(actual_url, material)
        cache_identity, binding = self._cache_binding(
            request, source_url=safe_source_url, access_profile_ref=access_ref
        )
        expected_binding = {
            key: value
            for key, value in binding.items()
            if key in {
                "provider",
                "provider_config_digest",
                "route_policy_version",
                "request_identity",
                "canonical_identifier",
                "route_id",
                "request_shape",
                "source_url",
            }
        }
        cached = self.cache.get(
            cache_identity,
            expected_binding=expected_binding,
            secret_values=material.all_secret_values(),
        )
        if cached is not None:
            content_type = normalize_content_type(
                cached.entry.content_type, allowed=request.route.accepted_content_types
            )
            return _FetchedResponse(
                response=self._response_from_cache(cached),
                content_type=content_type,
                content_family=content_family_for_media_type(content_type),
                cache_identity=f"sha256:{cache_identity}",
                cache_status=AcquisitionStatus.CACHE_HIT.value,
                artifact_class=ArtifactClass(cached.entry.artifact_class),
            )

        headers = self._request_headers(request.route, material)
        response = self.transport.fetch(
            actual_url,
            route=request.route,
            config=request.provider_config,
            headers=headers,
            secret_values=material.all_secret_values(),
        )
        if not isinstance(response, NetworkResponse):
            raise AcquisitionTransportError("network transport returned an invalid response")
        if len(response.body) > request.provider_config.max_response_bytes:
            raise AcquisitionPolicyError("network response exceeds the configured response limit")
        for header_name, allowed_values, error in (
            (
                "content-encoding",
                {"identity", ""},
                "compressed Content-Encoding is not enabled",
            ),
            (
                "transfer-encoding",
                {"identity", ""},
                "Transfer-Encoding is not enabled; identity framing is required",
            ),
        ):
            if any(value.casefold().strip() not in allowed_values for value in response.header_values(header_name)):
                raise AcquisitionPolicyError(error)
        lengths = response.header_values("content-length")
        if lengths:
            if len(set(lengths)) != 1 or not lengths[0].isdigit():
                raise AcquisitionPolicyError("Content-Length is contradictory or malformed")
            if int(lengths[0]) > request.provider_config.max_response_bytes:
                raise AcquisitionPolicyError("Content-Length exceeds the configured response limit")
            if int(lengths[0]) != len(response.body):
                raise AcquisitionTransportError("response body length does not match Content-Length")
        self._validate_network_response_urls(
            response,
            request=request,
            requested_url=actual_url,
        )
        if 200 <= response.status_code <= 299:
            content_type = normalize_content_type(
                response.header_values("content-type"),
                allowed=request.route.accepted_content_types,
            )
        else:
            # Error response bodies are never normalized into model data, but
            # size/encoding policy has already been enforced by the transport.
            content_type = request.route.accepted_content_types[0]
        secret_values = material.all_secret_values()
        assert_no_secret_values(response.body, secret_values)
        safe_resolved_url = sanitize_url(response.resolved_url, secret_values=secret_values)
        safe_redirects = tuple(
            sanitize_url(value, secret_values=secret_values) for value in response.redirect_chain
        )
        artifact_class = classify_route(request.route)
        if 200 <= response.status_code <= 299:
            manifest = {
                **binding,
                "schema_version": "molly.acquisition.cache.v1",
                "cache_identity": cache_identity,
                "source_url": safe_source_url,
                "resolved_url": safe_resolved_url,
                "redirect_chain": list(safe_redirects),
                "response_status": response.status_code,
                "content_type": content_type,
                "content_family": content_family_for_media_type(content_type).value,
                "body_sha256": sha256_bytes(response.body),
                "body_size": len(response.body),
                "artifact_class": artifact_class.value,
                "stored_at": utc_timestamp(),
            }
            self.cache.put(
                cache_identity,
                response.body,
                manifest=manifest,
                secret_values=secret_values,
            )
        return _FetchedResponse(
            response=NetworkResponse(
                status_code=response.status_code,
                headers=response.headers,
                body=response.body,
                requested_url=safe_source_url,
                resolved_url=safe_resolved_url,
                redirect_chain=safe_redirects,
            ),
            content_type=content_type,
            content_family=content_family_for_media_type(content_type),
            cache_identity=f"sha256:{cache_identity}",
            cache_status="MISS",
            artifact_class=artifact_class,
        )

    @staticmethod
    def _records_data(
        *, status: AcquisitionStatus, records: Sequence[MetadataRecord], query: str | None = None, doi: str | None = None
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "status": status.value,
            "results": [record.to_dict() for record in records],
        }
        if query is not None:
            value["query"] = query
        if doi is not None:
            value["canonical_identifier"] = doi
        if len(canonical_json_bytes(value)) <= MAX_TOOL_RESULT_DATA_BYTES:
            return value
        bounded: list[dict[str, Any]] = []
        for record in records:
            candidate = {**value, "results": [*bounded, record.to_dict()]}
            if len(canonical_json_bytes(candidate)) > MAX_TOOL_RESULT_DATA_BYTES:
                if not bounded:
                    raise AcquisitionIntegrityError(
                        "one normalized metadata record exceeds the tool result bound"
                    )
                break
            bounded.append(record.to_dict())
        return {**value, "results": bounded}

    def metadata_search(self, query: str, limit: int = 20) -> ToolResult:
        normalized_query = normalize_search_query(query, maximum=self.config.max_query_length)
        bounded_limit = normalize_result_limit(limit, maximum=self.config.max_result_limit)
        records: list[MetadataRecord] = []
        seen: set[tuple[str | None, str | None, str | None]] = set()
        secret_values: list[str] = []
        for provider in sorted(self.metadata_providers, key=lambda item: item.provider_id):
            request = provider.search(normalized_query, bounded_limit)
            secret_values.extend(
                self._access_material(request.route, request.provider_config).all_secret_values()
            )
            fetched = self._fetch_request(request)
            if fetched.response.status_code == 404:
                continue
            if not 200 <= fetched.response.status_code <= 299:
                raise AcquisitionTransportError("metadata provider returned a non-success status")
            normalized = provider.normalize_search(fetched.response.body)
            for record in normalized:
                key = (record.doi, record.provider_record_id, record.title)
                if key not in seen:
                    seen.add(key)
                    records.append(record)
        records = records[:bounded_limit]
        status = AcquisitionStatus.FOUND if records else AcquisitionStatus.NOT_FOUND
        data = self._records_data(status=status, records=records, query=normalized_query)
        assert_no_secret_values(canonical_json_bytes(data), tuple(dict.fromkeys(secret_values)))
        return ToolResult(data)

    def metadata_lookup(self, identifier: str) -> ToolResult:
        doi = normalize_doi(identifier)
        records: list[MetadataRecord] = []
        seen: set[tuple[str | None, str | None, str | None]] = set()
        secret_values: list[str] = []
        for provider in sorted(self.metadata_providers, key=lambda item: item.provider_id):
            request = provider.lookup(doi)
            secret_values.extend(
                self._access_material(request.route, request.provider_config).all_secret_values()
            )
            fetched = self._fetch_request(request)
            if fetched.response.status_code == 404:
                continue
            if not 200 <= fetched.response.status_code <= 299:
                raise AcquisitionTransportError("metadata provider returned a non-success status")
            for record in provider.normalize_lookup(fetched.response.body):
                key = (record.doi, record.provider_record_id, record.title)
                if key not in seen:
                    seen.add(key)
                    records.append(record)
        status = AcquisitionStatus.FOUND if records else AcquisitionStatus.NOT_FOUND
        data = self._records_data(status=status, records=records, doi=doi)
        assert_no_secret_values(canonical_json_bytes(data), tuple(dict.fromkeys(secret_values)))
        return ToolResult(data)

    @staticmethod
    def _candidate_content_priority(candidate: SourceCandidate, route: ProviderRoute) -> int | None:
        hint = candidate.content_type_hint
        if hint is not None:
            media = hint.casefold().split(";", 1)[0].strip()
            if media not in route.accepted_content_types:
                return None
            family = content_family_for_media_type(media)
        else:
            # Without a provider content hint, do not pretend the first
            # configured media type is the actual family.  It remains an
            # eligible fallback, but explicit XML/HTML/PDF hints win.
            return 4
        return {
            ContentFamily.XML: 0,
            ContentFamily.HTML: 2,
            ContentFamily.PDF: 3,
            ContentFamily.JSON: 1,
        }[family]

    @staticmethod
    def _candidate_query_is_safe(url: str) -> bool:
        try:
            pairs = parse_qsl(urlsplit(url).query, keep_blank_values=True)
        except ValueError:
            return False
        for key, _ in pairs:
            normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
            if normalized in {
                re.sub(r"[^a-z0-9]+", "_", item).strip("_") for item in SECRET_QUERY_NAMES
            }:
                return False
        return True

    def _eligible_sources(self, candidates: Sequence[SourceCandidate]) -> tuple[_EligibleSource, ...]:
        result: list[_EligibleSource] = []
        routes: list[tuple[ProviderRoute, ProviderConfig]] = []
        for route in self.config.full_text_routes:
            provider_config = self._provider_config(route.provider_id)
            if provider_config.provider_class != ProviderClass.FULL_TEXT.value:
                raise AcquisitionConfigurationError("full-text route is not bound to a FULL_TEXT provider")
            routes.append((route, provider_config))
        for candidate in candidates:
            if not self._candidate_query_is_safe(candidate.url):
                continue
            for route, provider_config in routes:
                try:
                    canonical = canonicalize_url(
                        candidate.url,
                        allowed_hosts=(route.host,),
                        allowed_port=route.port,
                        path_prefix=route.path_prefix or "/",
                        allowed_query_fields=route.allowed_query_fields,
                    )
                    artifact_class = classify_route(route)
                    priority = self._candidate_content_priority(candidate, route)
                except Exception:
                    continue
                if priority is None:
                    continue
                if candidate.access_status and candidate.access_status.casefold() in {
                    "unknown",
                    "denied",
                    "closed",
                }:
                    continue
                if candidate.license_status and candidate.license_status.casefold() in {
                    "unknown",
                    "denied",
                    "unverified",
                    "closed",
                }:
                    continue
                result.append(
                    _EligibleSource(
                        candidate=candidate,
                        route=route,
                        provider_config=provider_config,
                        canonical_url=canonical.value,
                        artifact_class=artifact_class,
                        priority=priority,
                    )
                )
        result.sort(
            key=lambda item: (
                item.priority,
                item.route.provider_id,
                item.route.route_id,
                item.canonical_url,
            )
        )
        return tuple(result)

    def _resolver_for_full_text(self) -> OpenAccessResolver | None:
        for resolver in sorted(self.oa_resolvers, key=lambda item: item.provider_id):
            return resolver
        return None

    def acquire_full_text(self, identifier: str) -> ToolResult:
        doi = normalize_doi(identifier)
        candidate_secret_values = self._candidate_secret_values()
        resolver = self._resolver_for_full_text()
        if resolver is None:
            data = {
                "status": AcquisitionStatus.NO_ELIGIBLE_SOURCE.value,
                "canonical_identifier": doi,
            }
            assert_no_secret_values(canonical_json_bytes(data), candidate_secret_values)
            return ToolResult(data)
        resolution_request = resolver.resolve(doi)
        resolution = self._fetch_request(resolution_request)
        if resolution.response.status_code == 404:
            data = {
                "status": AcquisitionStatus.NO_ELIGIBLE_SOURCE.value,
                "canonical_identifier": doi,
            }
            assert_no_secret_values(canonical_json_bytes(data), candidate_secret_values)
            return ToolResult(data)
        if not 200 <= resolution.response.status_code <= 299:
            raise AcquisitionTransportError("OA resolver returned a non-success status")
        candidates = resolver.normalize_resolution(resolution.response.body)
        eligible = self._eligible_sources(candidates)
        if not eligible:
            data = {
                "status": AcquisitionStatus.NO_ELIGIBLE_SOURCE.value,
                "canonical_identifier": doi,
                "evaluated_candidates": [
                    self._safe_candidate_url(item.url, candidate_secret_values)
                    for item in candidates[:20]
                ],
            }
            assert_no_secret_values(canonical_json_bytes(data), candidate_secret_values)
            return ToolResult(data)
        selected = eligible[0]
        # The selected candidate is provider-returned data, never model input.
        full_text_request = ConfiguredFullTextFetcher(
            selected.provider_config, selected.route
        ).request(doi=doi, source_url=selected.canonical_url)
        fetched = self._fetch_request(full_text_request)
        if not 200 <= fetched.response.status_code <= 299:
            raise AcquisitionTransportError("configured full-text route returned a non-success status")
        secret_values = tuple(
            dict.fromkeys(
                (*candidate_secret_values,
                 *self._access_material(selected.route, selected.provider_config).all_secret_values())
            )
        )
        assert_no_secret_values(fetched.response.body, secret_values)
        content_sha = sha256_bytes(fetched.response.body)
        content_artifact_id = artifact_id_for_sha256(content_sha)
        provenance = AcquisitionProvenance(
            schema_version="molly.acquisition.provenance.v1",
            acquisition_id=new_server_id("acq"),
            provider=selected.route.provider_id,
            provider_config_digest=selected.provider_config.config_digest,
            route_policy_version=selected.route.route_policy_version,
            request_identity=full_text_request.request_identity,
            canonical_identifier=doi,
            source_url=sanitize_url(selected.canonical_url, secret_values=secret_values),
            resolved_url=sanitize_url(fetched.response.resolved_url, secret_values=secret_values),
            redirect_chain=tuple(
                sanitize_url(value, secret_values=secret_values)
                for value in fetched.response.redirect_chain
            ),
            response_status=fetched.response.status_code,
            retrieved_at=(self._clock() if callable(self._clock) else None) or utc_timestamp(),
            access_status=selected.candidate.access_status or "configured-authorized",
            license_status=selected.candidate.license_status or selected.route.license_status,
            access_basis=selected.route.access_basis,
            redistribution_basis=selected.route.redistribution_basis,
            artifact_class=selected.artifact_class,
            content_type=fetched.content_type,
            content_family=fetched.content_family,
            content_sha256=content_sha,
            content_artifact_id=content_artifact_id,
            cache_identity=fetched.cache_identity,
            cache_status=fetched.cache_status,
            access_profile_ref=(selected.route.access_profile_ref or selected.provider_config.access_profile_ref),
            evaluated_candidates=tuple(
                self._safe_candidate_url(item.url, secret_values)
                for item in candidates[:20]
            ),
        )
        provenance_bytes = provenance.canonical_bytes()
        assert_no_secret_values(provenance_bytes, secret_values)
        provenance_sha = sha256_bytes(provenance_bytes)
        provenance_artifact_id = artifact_id_for_sha256(provenance_sha)
        data = {
            "status": (
                AcquisitionStatus.CACHE_HIT.value
                if fetched.cache_status == AcquisitionStatus.CACHE_HIT.value
                else AcquisitionStatus.ACQUIRED.value
            ),
            "canonical_identifier": doi,
            "content_artifact_id": content_artifact_id,
            "provenance_artifact_id": provenance_artifact_id,
            "artifact_class": selected.artifact_class.value,
            "content_artifact_class": selected.artifact_class.value,
            "provenance_artifact_class": selected.artifact_class.value,
            "content_family": fetched.content_family.value,
            "content_type": fetched.content_type,
            "provider": selected.route.provider_id,
            "cache_status": fetched.cache_status,
            "artifact_roles": {
                "content": {
                    "artifact_id": content_artifact_id,
                    "class": selected.artifact_class.value,
                    "role": "FULL_TEXT_CONTENT",
                },
                "provenance": {
                    "artifact_id": provenance_artifact_id,
                    "class": selected.artifact_class.value,
                    "role": "ACQUISITION_PROVENANCE",
                },
            },
        }
        assert_no_secret_values(canonical_json_bytes(data), secret_values)
        return ToolResult(
            data,
            artifacts=(
                ArtifactDraft(fetched.response.body, fetched.content_type),
                ArtifactDraft(
                    provenance_bytes,
                    "application/json",
                    schema_name="molly.acquisition.provenance",
                    schema_version="1",
                ),
            ),
        )


__all__ = [
    "AccessProfileResolver",
    "AcquisitionService",
    "EmptyAccessProfileResolver",
    "StaticAccessProfileResolver",
]
