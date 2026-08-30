"""Focused offline and adversarial acceptance for CORE-03 acquisition."""

from __future__ import annotations

from dataclasses import replace
import io
import json
from pathlib import Path
from typing import Any

import pytest

from molly.acquisition import (
    AcquisitionCache,
    AcquisitionConfig,
    AcquisitionConfigurationError,
    AcquisitionIntegrityError,
    AcquisitionPolicyError,
    AcquisitionService,
    AcquisitionStatus,
    ArtifactClass,
    ConfiguredFullTextFetcher,
    EphemeralAccessMaterial,
    FullTextFetcher,
    NetworkResponse,
    ProviderClass,
    ProviderConfig,
    ProviderRoute,
    RetryPolicy,
    SafeNetworkTransport,
    SourceCandidate,
    StaticAccessProfileResolver,
    acquisition_tool_specs,
    canonicalize_url,
    normalize_doi,
    register_acquisition_tools,
    validate_resolved_addresses,
)
from molly.acquisition.errors import AcquisitionTimeoutError
from molly.core import (
    AgentLoop,
    ArtifactLineage,
    ArtifactStore,
    RunBudget,
    RunLedger,
    RunRequest,
    RunStatus,
    SchemaValidationError,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
)
from molly.core.agent_loop import TOOL_EXECUTION_FAILED, TOOL_EXECUTION_SUCCEEDED
from molly.core.ids import artifact_id_for_sha256, sha256_bytes


pytestmark = pytest.mark.unit


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "v2" / "synthetic"


def _public_ip() -> str:
    return ".".join(("93", "184", "216", "34"))


class FixtureTransport:
    """Deterministic transport used instead of external Internet access."""

    def __init__(self, *, candidate_payload: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.candidate_payload = candidate_payload or {
            "best_oa_location": {
                "url": "https://repository.example.org/articles/minimal.xml",
                "url_type": "xml",
                "license": "cc-by",
                "host_type": "repository",
            }
        }
        self.echo_secret = False

    def fetch(self, url, *, route, config, headers={}, secret_values=()):
        self.calls.append(
            {
                "url": url,
                "route": route.route_id,
                "provider": config.provider_id,
                "headers": dict(headers),
                "secret_values": tuple(secret_values),
            }
        )
        if self.echo_secret and config.provider_id == "unpaywall":
            body = ("echo:" + (tuple(secret_values)[0] if secret_values else "no-secret")).encode()
            return NetworkResponse(200, {"content-type": "application/json"}, body, url, url)
        if config.provider_id == "unpaywall":
            body = json.dumps(self.candidate_payload, separators=(",", ":")).encode()
            content_type = "application/json"
        elif config.provider_id == "repository":
            body = (FIXTURE_ROOT / "minimal.jats.xml").read_bytes()
            content_type = "application/xml"
        elif config.provider_id == "openalex":
            body = json.dumps(
                {
                    "results": [
                        {
                            "id": "https://openalex.example/works/W1",
                            "doi": "https://doi.org/10.1234/fixture",
                            "title": "Synthetic fixture work",
                            "authorships": [{"author": {"display_name": "A. Researcher"}}],
                            "publication_year": 2026,
                            "publication_date": "2026-01-01",
                            "primary_location": {"source": {"display_name": "Fixture Journal"}},
                            "type": "article",
                            "open_access": {"is_oa": True, "license": "cc-by"},
                        }
                    ]
                },
                separators=(",", ":"),
            ).encode()
            content_type = "application/json"
        elif config.provider_id == "crossref":
            body = json.dumps(
                {
                    "message": {
                        "items": [
                            {
                                "DOI": "10.1234/fixture",
                                "title": ["Synthetic Crossref work"],
                                "author": [{"given": "B.", "family": "Researcher"}],
                                "container-title": ["Fixture Journal"],
                                "issued": {"date-parts": [[2026]]},
                                "type": "journal-article",
                            }
                        ]
                    }
                },
                separators=(",", ":"),
            ).encode()
            content_type = "application/json"
        else:
            body = b"{}"
            content_type = "application/json"
        return NetworkResponse(200, {"content-type": content_type}, body, url, url)


def _config(*, redistribution_basis: str = "verified-public") -> AcquisitionConfig:
    openalex = ProviderConfig(
        provider_id="openalex",
        provider_class=ProviderClass.METADATA,
        routes=(
            ProviderRoute(
                "search",
                "api.openalex.example",
                "/works",
                allowed_query_fields=("search", "per-page", "filter"),
            ),
            ProviderRoute(
                "lookup",
                "api.openalex.example",
                "/works",
                allowed_query_fields=("search", "per-page", "filter"),
            ),
        ),
    )
    crossref = ProviderConfig(
        provider_id="crossref",
        provider_class=ProviderClass.METADATA,
        routes=(
            ProviderRoute(
                "search",
                "api.crossref.example",
                "/works",
                allowed_query_fields=("query.bibliographic", "rows"),
            ),
            ProviderRoute(
                "lookup",
                "api.crossref.example",
                "/works/{doi}",
                path_prefix="/works/",
            ),
        ),
    )
    unpaywall = ProviderConfig(
        provider_id="unpaywall",
        provider_class=ProviderClass.OA_RESOLUTION,
        routes=(
            ProviderRoute(
                "lookup",
                "api.unpaywall.example",
                "/v2/{doi}",
                path_prefix="/v2/",
                allowed_query_fields=("email",),
                access_profile_ref="unpaywall-profile",
            ),
        ),
    )
    repository = ProviderConfig(
        provider_id="repository",
        provider_class=ProviderClass.FULL_TEXT,
        routes=(
            ProviderRoute(
                "article",
                "repository.example.org",
                "/articles/{path}",
                path_prefix="/articles/",
                accepted_content_types=("application/xml", "text/html", "application/pdf"),
                access_basis="verified-open-access-route",
                license_status="verified-open-access",
                redistribution_basis=redistribution_basis,
            ),
        ),
    )
    return AcquisitionConfig(providers=(openalex, crossref, unpaywall, repository))


def _service(
    tmp_path: Path,
    *,
    transport: FixtureTransport | None = None,
    config: AcquisitionConfig | None = None,
    access_secret: str = "synthetic-access-value",
) -> tuple[AcquisitionService, FixtureTransport]:
    transport = transport or FixtureTransport()
    config = config or _config()
    profiles = StaticAccessProfileResolver(
        {
            "unpaywall-profile": EphemeralAccessMaterial(
                query_params={"email": access_secret},
                secret_values=(access_secret,),
            )
        }
    )
    service = AcquisitionService(
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        cache=AcquisitionCache(tmp_path / "cache"),
        config=config,
        transport=transport,
        access_profiles=profiles,
    )
    return service, transport


def _loop(
    tmp_path: Path,
    service: AcquisitionService,
    transport: FixtureTransport,
    actions: tuple[object, ...],
):
    class Provider:
        def __init__(self) -> None:
            self.actions = list(actions)
            self.calls = 0
            self.contexts = []

        def next_action(self, context, model_visible_tools):
            self.calls += 1
            self.contexts.append(context)
            return self.actions.pop(0)

    provider = Provider()
    registry = ToolRegistry()
    specs = register_acquisition_tools(registry, service)
    policy = ToolPolicy(
        allowed_tools=tuple(spec.name for spec in specs),
        allowed_side_effect_classes=(SideEffectClass.NETWORK_READ,),
    )
    request = RunRequest.create(
        goal="offline CORE-03 acquisition fixture",
        tool_policy_digest=policy.digest,
        budget=RunBudget(max_decisions=8, max_tool_calls=4, max_steps=4),
    )
    loop = AgentLoop(
        store=service.artifact_store,
        ledger=RunLedger(tmp_path / "events.jsonl"),
        lineage=ArtifactLineage(tmp_path / "lineage.jsonl"),
        registry=registry,
        policy=policy,
        decision_provider=provider,
    )
    return loop, request, provider, specs


def test_doi_and_search_inputs_are_bounded_and_canonical() -> None:
    assert normalize_doi(" DOI:10.1234/ABC-1 ") == "10.1234/abc-1"
    assert normalize_doi("https://doi.org/10.1234/ABC-1") == "10.1234/abc-1"
    with pytest.raises(AcquisitionPolicyError):
        normalize_doi("https://evil.example/10.1234/abc")
    with pytest.raises(AcquisitionPolicyError):
        normalize_doi("10.1234/a#fragment")


def test_provider_config_digest_is_bound_to_tool_spec_and_hidden_from_model(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    specs = acquisition_tool_specs(service)
    assert {spec.execution_config_digest for spec in specs} == {service.config_digest}
    assert all("execution_config_digest" not in spec.model_view() for spec in specs)
    changed_route = replace(
        service.config.providers[-1].routes[0],
        route_policy_version="v2",
    )
    changed_provider = replace(service.config.providers[-1], routes=(changed_route,))
    changed_config = replace(
        service.config,
        providers=(*service.config.providers[:-1], changed_provider),
    )
    changed_service, _ = _service(tmp_path / "changed", config=changed_config)
    assert changed_service.config_digest != service.config_digest
    assert acquisition_tool_specs(changed_service)[0].spec_digest != specs[0].spec_digest


def test_provider_routes_are_server_owned_dns_routes_and_full_text_fetcher_is_exact() -> None:
    with pytest.raises(AcquisitionConfigurationError):
        ProviderRoute("ip", _public_ip(), "/data")

    config = ProviderConfig(
        "repository",
        ProviderClass.FULL_TEXT,
        (
            ProviderRoute(
                "article",
                "repository.example.org",
                "/articles/{path}",
                path_prefix="/articles/",
                accepted_content_types=("application/xml",),
                access_basis="verified-open-access-route",
                license_status="verified-open-access",
                redistribution_basis="verified-public",
            ),
        ),
    )
    fetcher: FullTextFetcher = ConfiguredFullTextFetcher(config, config.routes[0])
    request = fetcher.request(
        doi="10.1234/fixture",
        source_url="https://repository.example.org/articles/paper.xml",
    )
    assert request.provider_id == "repository"
    assert request.route.route_id == "article"
    assert request.canonical_identifier == "10.1234/fixture"
    assert request.path == "/articles/paper.xml"


def test_metadata_search_normalizes_openalex_and_crossref_without_raw_payload(tmp_path: Path) -> None:
    service, transport = _service(tmp_path)
    result = service.metadata_search("  OLED   fixture ", limit=2)
    assert result.data["status"] == AcquisitionStatus.FOUND.value
    assert len(result.data["results"]) == 2
    assert {item["doi"] for item in result.data["results"]} == {"10.1234/fixture"}
    assert all("message" not in result.data for _ in [0])
    assert len(transport.calls) == 2
    assert all(call["url"].startswith("https://") for call in transport.calls)


def test_full_text_agent_loop_is_offline_cache_first_and_restartable(tmp_path: Path) -> None:
    service, transport = _service(tmp_path)
    action = ToolCallProposal("literature_acquire_full_text", {"doi": "10.1234/fixture"})
    loop, request, provider, _ = _loop(tmp_path, service, transport, (action, StopAction("done")))
    first = loop.run(request)
    assert first.status == RunStatus.STOPPED.value
    assert len(first.visible_artifact_ids) == 2
    assert provider.contexts[1].previous_tool_outcome["data"]["content_artifact_id"].startswith("sha256:")
    assert provider.contexts[1].previous_tool_outcome["data"]["artifact_class"] == "PUBLIC_ARTIFACT"
    assert len(transport.calls) == 2  # OA resolution and full-text body.
    success = [event for event in loop.ledger.events if event.event_type == TOOL_EXECUTION_SUCCEEDED]
    assert len(success) == 1
    assert success[0].output_artifact_ids == first.visible_artifact_ids
    assert loop.lineage.producer_steps(success[0].output_artifact_ids[0])

    resumed_provider_loop, resumed_request, resumed_provider, _ = _loop(
        tmp_path / "restart",
        service,
        transport,
        (StopAction("must not be called"),),
    )
    # Reuse the original durable run files with a new provider/loop object.
    resumed_provider_loop.store = service.artifact_store
    resumed_provider_loop.ledger = RunLedger(loop.ledger.path)
    resumed_provider_loop.lineage = ArtifactLineage(loop.lineage.path)
    resumed = resumed_provider_loop.run(request)
    assert resumed.status == RunStatus.STOPPED.value
    assert resumed_provider.calls == 0


def test_full_text_second_run_hits_resolution_and_body_cache(tmp_path: Path) -> None:
    service, transport = _service(tmp_path)
    first = service.acquire_full_text("10.1234/fixture")
    second = service.acquire_full_text("doi:10.1234/fixture")
    assert first.data["status"] == AcquisitionStatus.ACQUIRED.value
    assert second.data["status"] == AcquisitionStatus.CACHE_HIT.value
    assert first.data["content_artifact_id"] == second.data["content_artifact_id"]
    assert first.data["provenance_artifact_id"] != second.data["provenance_artifact_id"]
    assert len(transport.calls) == 2


def test_model_cannot_supply_arbitrary_url_or_provider_authority(tmp_path: Path) -> None:
    service, transport = _service(tmp_path)
    action = ToolCallProposal(
        "literature_acquire_full_text",
        {"doi": "10.1234/fixture", "url": "https://evil.example/paper.xml"},
    )
    loop, request, _, _ = _loop(tmp_path, service, transport, (action,))
    with pytest.raises(SchemaValidationError):
        loop.run(request)
    assert transport.calls == []


def test_untrusted_oa_candidates_are_exact_route_filtered_and_ranked(tmp_path: Path) -> None:
    transport = FixtureTransport(
        candidate_payload={
            "oa_locations": [
                {"url": "https://repository.example.org/articles/paper.pdf", "url_type": "pdf", "license": "cc-by"},
                {"url": "https://evil.example.org/articles/paper.xml", "url_type": "xml", "license": "cc-by"},
                {"url": "https://repository.example.org/articles/paper.html", "url_type": "html", "license": "cc-by"},
                {"url": "https://repository.example.org/articles/paper.xml", "url_type": "xml", "license": "cc-by"},
            ]
        }
    )
    service, transport = _service(tmp_path, transport=transport)
    result = service.acquire_full_text("10.1234/fixture")
    assert result.data["content_family"] == "xml"
    assert transport.calls[-1]["provider"] == "repository"
    assert "/paper.xml" in transport.calls[-1]["url"]
    assert "evil.example" not in result.data.get("evaluated_candidates", [])


def test_acquisition_tool_requires_network_read_policy_but_no_intrinsic_approval(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    spec = acquisition_tool_specs(service)[0]
    assert spec.side_effect_class == SideEffectClass.NETWORK_READ.value
    assert spec.requires_approval is False
    policy = ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(SideEffectClass.NETWORK_READ,),
        approval_required_side_effect_classes=(SideEffectClass.NETWORK_READ,),
    )
    assert policy.requires_approval(spec) is True


def test_public_private_classification_is_occurrence_context_not_artifact_record(tmp_path: Path) -> None:
    private_service, _ = _service(tmp_path / "private", config=_config(redistribution_basis="not-asserted"))
    public_service, _ = _service(tmp_path / "public", config=_config(redistribution_basis="verified-public"))
    private = private_service.acquire_full_text("10.1234/fixture")
    public = public_service.acquire_full_text("10.1234/fixture")
    assert private.data["content_artifact_id"] == public.data["content_artifact_id"]
    assert private.data["artifact_class"] == ArtifactClass.PRIVATE_ARTIFACT.value
    assert public.data["artifact_class"] == ArtifactClass.PUBLIC_ARTIFACT.value
    private_record = private_service.artifact_store.put(
        private.artifacts[0].content,
        media_type=private.artifacts[0].media_type,
    )
    assert not hasattr(private_record, "artifact_class")


def test_secret_query_material_is_not_recorded_anywhere_durable(tmp_path: Path) -> None:
    secret = "synthetic-access-value"
    service, transport = _service(tmp_path, access_secret=secret)
    result = service.acquire_full_text("10.1234/fixture")
    assert any("synthetic-access-value" in call["url"] for call in transport.calls)
    assert secret.encode() not in json.dumps(result.data, default=str).encode()
    for path in (tmp_path / "cache").rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    assert secret.encode() not in result.artifacts[1].content


def test_secret_reflection_fails_before_cache_or_artifact_draft(tmp_path: Path) -> None:
    transport = FixtureTransport()
    transport.echo_secret = True
    service, _ = _service(tmp_path, transport=transport)
    with pytest.raises(AcquisitionIntegrityError):
        service.acquire_full_text("10.1234/fixture")
    assert not tuple((tmp_path / "cache").rglob("entries/*"))


def test_cache_corruption_fails_closed_without_refetch(tmp_path: Path) -> None:
    service, transport = _service(tmp_path)
    service.acquire_full_text("10.1234/fixture")
    body_paths = tuple((tmp_path / "cache" / "bodies").rglob("*") )
    assert body_paths
    body_paths[-1].write_bytes(b"tampered")
    with pytest.raises(AcquisitionIntegrityError):
        service.acquire_full_text("10.1234/fixture")
    assert len(transport.calls) == 2


def test_cache_partial_entry_and_symlink_escape_fail_closed(tmp_path: Path) -> None:
    cache = AcquisitionCache(tmp_path / "cache")
    identity = "a" * 64
    entry_path, body_path = cache._paths(identity)
    entry_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="partial"):
        cache.get(identity, expected_binding={})
    body_path.unlink(missing_ok=True)
    entry_path.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    body_path.parent.rmdir()
    body_path.parent.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Exception):
        cache._paths(identity)


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://repository.example.org/articles/x.xml",
        "https://user:password@repository.example/articles/x.xml",
        "https://repository.example.org:8443/articles/x.xml",
        "https://repository.example.org/articles/../x.xml",
        "https://repository.example.org/articles/x.xml#fragment",
        "https://evilrepository.example.org/articles/x.xml",
    ],
)
def test_url_policy_rejects_unsafe_authority_forms(bad_url: str) -> None:
    route = ProviderRoute("article", "repository.example.org", "/articles/{path}", path_prefix="/articles/", accepted_content_types=("application/xml",))
    with pytest.raises(AcquisitionPolicyError):
        canonicalize_url(
            bad_url,
            allowed_hosts=(route.host,),
            allowed_port=route.port,
            path_prefix=route.path_prefix or "/",
            allowed_query_fields=route.allowed_query_fields,
        )


@pytest.mark.parametrize(
    "address",
    [
        ".".join(("127", "0", "0", "1")),
        ".".join(("10", "1", "2", "3")),
        ".".join(("169", "254", "1", "2")),
        "::1",
        "fd00::1",
        "fe80::1",
        "ff02::1",
    ],
)
def test_dns_address_policy_rejects_non_global_ipv4_and_ipv6(address: str) -> None:
    with pytest.raises(AcquisitionPolicyError):
        validate_resolved_addresses("repository.example.org", (address,))


class _Resolver:
    def __init__(self, addresses: tuple[str, ...]) -> None:
        self.addresses = addresses
        self.calls: list[tuple[str, int]] = []

    def resolve(self, hostname: str, port: int):
        self.calls.append((hostname, port))
        return self.addresses


class _Connection:
    def __init__(self, payload: bytes, peer: str | None = None) -> None:
        self.payload = payload
        self.peer = peer or _public_ip()
        self.sent = b""
        self.timeouts: list[float | None] = []

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def makefile(self, mode: str):
        return io.BytesIO(self.payload)

    def settimeout(self, value: float | None) -> None:
        self.timeouts.append(value)

    def getpeername(self):
        return (self.peer, 443)

    def close(self) -> None:
        return None


def _http_payload(status: int, body: bytes, *, headers: tuple[tuple[str, str], ...] = ()) -> bytes:
    values = list(headers)
    if not any(name.casefold() == "content-length" for name, _ in values):
        values.append(("Content-Length", str(len(body))))
    return (
        f"HTTP/1.1 {status} TEST\r\n"
        + "".join(f"{name}: {value}\r\n" for name, value in values)
        + "\r\n"
    ).encode() + body


def _transport_route() -> tuple[ProviderRoute, ProviderConfig]:
    route = ProviderRoute(
        "data",
        "api.example.org",
        "/data",
        accepted_content_types=("application/json",),
    )
    return route, ProviderConfig("transport", ProviderClass.METADATA, (route,))


def test_safe_transport_connects_to_validated_ip_and_keeps_dns_hostname_for_route(tmp_path: Path) -> None:
    route, config = _transport_route()
    resolver = _Resolver((_public_ip(),))
    connections: list[tuple[str, str]] = []

    def factory(address, selected_route, timeout):
        connections.append((address, selected_route.host))
        return _Connection(_http_payload(200, b"{}", headers=(("Content-Type", "application/json"),)), address)

    transport = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=factory,
        sleeper=lambda _: None,
    )
    response = transport.fetch(
        "https://api.example.org/data",
        route=route,
        config=config,
    )
    assert response.body == b"{}"
    assert connections == [(_public_ip(), "api.example.org")]
    assert resolver.calls == [("api.example.org", 443)]


def test_safe_transport_rejects_peer_address_change_before_publishing() -> None:
    route, config = _transport_route()
    resolver = _Resolver((_public_ip(),))
    different = ".".join(("93", "184", "216", "35"))
    transport = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(200, b"{}", headers=(("Content-Type", "application/json"),)), different
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionPolicyError):
        transport.fetch("https://api.example.org/data", route=route, config=config)


def test_safe_transport_manually_revalidates_redirects() -> None:
    route = ProviderRoute("data", "api.example.org", "/data", path_prefix="/", accepted_content_types=("application/json",))
    config = ProviderConfig("transport", ProviderClass.METADATA, (route,))
    resolver = _Resolver((_public_ip(),))
    created = 0

    def factory(address, selected_route, timeout):
        nonlocal created
        created += 1
        return _Connection(
            _http_payload(302, b"", headers=(("Location", "https://evil.example.org/data"),)),
            address,
        )

    transport = SafeNetworkTransport(resolver=resolver, connection_factory=factory, sleeper=lambda _: None)
    with pytest.raises(AcquisitionPolicyError):
        transport.fetch("https://api.example.org/data", route=route, config=config)
    assert created == 1


def test_safe_transport_enforces_content_size_content_type_and_encoding() -> None:
    route = ProviderRoute("data", "api.example.org", "/data", accepted_content_types=("application/json",))
    config = ProviderConfig("transport", ProviderClass.METADATA, (route,), max_response_bytes=4)
    resolver = _Resolver((_public_ip(),))
    oversized = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(200, b"12345", headers=(("Content-Type", "application/json"),)), address
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionPolicyError):
        oversized.fetch("https://api.example.org/data", route=route, config=config)

    bad_type = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(200, b"{}", headers=(("Content-Type", "text/plain"),)), address
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionPolicyError):
        bad_type.fetch("https://api.example.org/data", route=route, config=replace(config, max_response_bytes=100))

    compressed = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(200, b"{}", headers=(("Content-Type", "application/json"), ("Content-Encoding", "gzip"))), address
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionPolicyError):
        compressed.fetch("https://api.example.org/data", route=route, config=replace(config, max_response_bytes=100))


def test_safe_transport_rejects_transfer_encoding_and_contradictory_content_type() -> None:
    route, config = _transport_route()
    resolver = _Resolver((_public_ip(),))

    chunked = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(
                200,
                b"{}",
                headers=(
                    ("Content-Type", "application/json"),
                    ("Transfer-Encoding", "chunked"),
                ),
            ),
            address,
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionPolicyError):
        chunked.fetch("https://api.example.org/data", route=route, config=config)

    contradictory = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(
                200,
                b"{}",
                headers=(
                    ("Content-Type", "application/json"),
                    ("content-type", "text/html"),
                ),
            ),
            address,
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionPolicyError):
        contradictory.fetch("https://api.example.org/data", route=route, config=config)


def test_safe_transport_total_deadline_covers_rate_slot_before_connect() -> None:
    route, config = _transport_route()
    resolver = _Resolver((_public_ip(),))
    clock_values = iter((0.0, 61.0))
    transport = SafeNetworkTransport(
        resolver=resolver,
        monotonic=lambda: next(clock_values),
        connection_factory=lambda address, selected_route, timeout: _Connection(
            _http_payload(200, b"{}", headers=(("Content-Type", "application/json"),)), address
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(AcquisitionTimeoutError):
        transport.fetch("https://api.example.org/data", route=route, config=config)


def test_safe_transport_retries_bounded_server_error_without_real_sleep() -> None:
    route, _ = _transport_route()
    config = ProviderConfig(
        "transport",
        ProviderClass.METADATA,
        (route,),
        retry_policy=RetryPolicy(maximum_retries=1, jitter_fraction=0, maximum_retry_after_seconds=2),
    )
    resolver = _Resolver((_public_ip(),))
    clock_value = [0.0]
    sleeps: list[float] = []
    payloads = [
        _http_payload(503, b"busy", headers=(("Retry-After", "1"),)),
        _http_payload(200, b"{}", headers=(("Content-Type", "application/json"),)),
    ]

    def sleep(value: float) -> None:
        sleeps.append(value)
        clock_value[0] += value

    def factory(address, selected_route, timeout):
        return _Connection(payloads.pop(0), address)

    transport = SafeNetworkTransport(
        resolver=resolver,
        connection_factory=factory,
        monotonic=lambda: clock_value[0],
        sleeper=sleep,
        random_source=lambda: 0,
    )
    response = transport.fetch("https://api.example.org/data", route=route, config=config)
    assert response.status_code == 200
    assert len(sleeps) >= 1
    assert max(sleeps) <= 2


def test_safe_transport_does_not_retry_authentication_failure() -> None:
    route, config = _transport_route()
    resolver = _Resolver((_public_ip(),))
    calls = [0]

    def factory(address, selected_route, timeout):
        calls[0] += 1
        return _Connection(_http_payload(401, b"denied"), address)

    transport = SafeNetworkTransport(resolver=resolver, connection_factory=factory, sleeper=lambda _: None)
    response = transport.fetch("https://api.example.org/data", route=route, config=config)
    assert response.status_code == 401
    assert calls == [1]


def test_acquired_content_and_provenance_are_exact_sha256_artifacts(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    result = service.acquire_full_text("10.1234/fixture")
    content = result.artifacts[0].content
    provenance = result.artifacts[1].content
    assert result.data["content_artifact_id"] == artifact_id_for_sha256(sha256_bytes(content))
    assert result.data["provenance_artifact_id"] == artifact_id_for_sha256(sha256_bytes(provenance))
    payload = json.loads(provenance)
    assert payload["content_artifact_id"] == result.data["content_artifact_id"]
    assert payload["content_sha256"] == sha256_bytes(content)


def test_no_eligible_source_is_a_bounded_observation_not_an_arbitrary_fetch(tmp_path: Path) -> None:
    transport = FixtureTransport(
        candidate_payload={
            "best_oa_location": {
                "url": "https://unconfigured.example.org/articles/paper.xml",
                "url_type": "xml",
                "license": "cc-by",
            }
        }
    )
    service, transport = _service(tmp_path, transport=transport)
    result = service.acquire_full_text("10.1234/fixture")
    assert result.data["status"] == AcquisitionStatus.NO_ELIGIBLE_SOURCE.value
    assert result.artifacts == ()
    assert len(transport.calls) == 1
