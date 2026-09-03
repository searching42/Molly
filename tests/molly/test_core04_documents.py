"""Focused offline acceptance for the CORE-04 document boundary."""

from __future__ import annotations

import ast
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from molly.acquisition import (
    AcquisitionCache,
    AcquisitionConfig,
    AcquisitionService,
    AccessStatus,
    NetworkResponse,
    ProviderClass,
    ProviderConfig,
    ProviderRequest,
    ProviderRoute,
    SourceCandidate,
)
from molly.core import (
    AgentLoop,
    ArtifactLineage,
    ArtifactStore,
    RunLedger,
    RunRequest,
    RunStatus,
    SideEffectClass,
    StopAction,
    ToolCallProposal,
    ToolPolicy,
    ToolRegistry,
)
from molly.core.agent_loop import (
    TOOL_EXECUTION_FAILED,
    TOOL_EXECUTION_SUCCEEDED,
)
from molly.core.errors import CoreContractError
from molly.core.ids import artifact_id_for_sha256, sha256_bytes
from molly.documents import (
    CANONICAL_SCHEMA_NAME,
    CANONICAL_SCHEMA_VERSION,
    CanonicalBlockKind,
    CanonicalCell,
    CanonicalDocument,
    DocumentParserConfig,
    DocumentParserRegistry,
    DocumentParserRouter,
    DocumentService,
    MinerUFallbackParser,
    ParserQualityStatus,
    SourceLocator,
    SourceLocatorKind,
    document_tool_specs,
    register_document_tools,
)
from molly.documents.errors import (
    DocumentIntegrityError,
    DocumentLimitError,
    MalformedDocumentError,
    ParserQualityError,
    ParserUnavailableError,
    UnsupportedDocumentError,
)
from molly.documents.parsers.pdf_text import PdfTextParser


pytestmark = pytest.mark.unit


FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "v2" / "synthetic"
PROJECT_ROOT = Path(__file__).parents[2]


def _source_id(payload: bytes) -> str:
    return artifact_id_for_sha256(sha256_bytes(payload))


def _parse_fixture(name: str, media_type: str):
    payload = (FIXTURE_ROOT / name).read_bytes()
    router = DocumentParserRouter()
    return payload, router.parse(
        source_artifact_id=_source_id(payload),
        source_media_type=media_type,
        source_bytes=payload,
    )


def _canonical_json(document: CanonicalDocument) -> dict:
    return json.loads(document.canonical_bytes().decode("utf-8"))


def test_jats_fixture_has_source_neutral_deterministic_structure() -> None:
    payload, first = _parse_fixture("minimal.jats.xml", "application/xml")
    _, second = _parse_fixture("minimal.jats.xml", "application/xml")

    assert first.parser_id == "jats"
    assert first.title == "Synthetic OLED fixture"
    assert any(
        block.kind == CanonicalBlockKind.ABSTRACT.value
        and block.text == "Offline parser contract text only."
        for block in first.blocks
    )
    assert any(section.title == "Results" for section in first.sections)
    assert any(
        block.kind == CanonicalBlockKind.PARAGRAPH.value
        and block.text == "A synthetic observation."
        for block in first.blocks
    )
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.artifact_id == second.artifact_id
    assert first.source_artifact_id == _source_id(payload)
    assert all(
        item.locator.source_artifact_id == first.source_artifact_id
        for item in (*first.sections, *first.blocks, *first.tables, *first.figures, *first.references)
    )
    assert "PUBLIC_ARTIFACT" not in first.canonical_bytes().decode("utf-8")
    assert "access_profile_ref" not in first.canonical_bytes().decode("utf-8")


def test_html_fixture_has_tables_cells_and_no_script_content() -> None:
    _, document = _parse_fixture("minimal.html", "text/html")

    assert document.parser_id == "html"
    assert document.title == "Synthetic OLED fixture"
    assert any(block.text == "Offline parser contract text only." for block in document.blocks)
    assert len(document.tables) == 1
    table = document.tables[0]
    assert table.caption == "Synthetic property row"
    assert [(cell.row_index, cell.column_index, cell.text, cell.is_header) for cell in table.cells] == [
        (0, 0, "molecule", True),
        (0, 1, "property", True),
        (0, 2, "value", True),
        (1, 0, "fixture-001", False),
        (1, 1, "plqy", False),
        (1, 2, "0.65", False),
    ]
    assert len({cell.locator.path for cell in table.cells}) == len(table.cells)
    source = b"<html><body><p>kept</p><script>secret()</script><style>hidden{}</style></body></html>"
    document = DocumentParserRouter().parse(
        source_artifact_id=_source_id(source),
        source_media_type="text/html",
        source_bytes=source,
    )
    assert "secret" not in document.canonical_bytes().decode("utf-8")
    assert "hidden" not in document.canonical_bytes().decode("utf-8")
    assert any(block.text == "kept" for block in document.blocks)


def test_generic_xml_extracts_structure_figures_and_direct_doi() -> None:
    source = """<?xml version='1.0'?>
    <document xml:lang='en'>
      <title>Generic document</title>
      <section><heading>Methods</heading><paragraph>Text with units and Unicode μ.</paragraph>
        <table><caption>Measurements</caption><tr><th>Name</th><td>Value</td></tr></table>
        <figure><caption>Figure one</caption></figure>
      </section>
      <references><reference>Example citation doi:10.1234/ABC.1</reference></references>
    </document>""".encode("utf-8")
    document = DocumentParserRouter().parse(
        source_artifact_id=_source_id(source),
        source_media_type="text/xml",
        source_bytes=source,
    )
    assert document.parser_id == "xml"
    assert document.language == "en"
    assert document.title == "Generic document"
    assert any(section.title == "Methods" for section in document.sections)
    assert document.tables[0].cells[0].is_header is True
    assert document.figures[0].caption == "Figure one"
    assert document.references[0].identifier == "10.1234/abc.1"


def test_jats_table_wrap_has_deterministic_cell_coordinates_and_locators() -> None:
    source = b"""<article><front><article-meta><title-group><article-title>Table fixture</article-title></title-group></article-meta></front>
    <body><sec><title>Results</title><table-wrap><caption><title>Measured values</title></caption>
    <table><tr><th>A</th><th>B</th></tr><tr><td>1</td><td>2</td></tr></table>
    </table-wrap></sec></body></article>"""
    source_id = _source_id(source)
    document = DocumentParserRouter().parse(source_id, "application/xml", source)
    assert document.parser_id == "jats"
    table = document.tables[0]
    assert table.caption == "Measured values"
    assert [(cell.row_index, cell.column_index, cell.is_header) for cell in table.cells] == [
        (0, 0, True),
        (0, 1, True),
        (1, 0, False),
        (1, 1, False),
    ]
    assert all(cell.locator.kind == SourceLocatorKind.XML_ELEMENT.value for cell in table.cells)
    assert len({cell.locator.path for cell in table.cells}) == 4


def test_canonical_document_round_trip_is_byte_identical() -> None:
    _, document = _parse_fixture("minimal.jats.xml", "application/xml")
    restored = CanonicalDocument.from_dict(_canonical_json(document))
    assert restored.canonical_bytes() == document.canonical_bytes()
    assert restored.artifact_id == document.artifact_id


@pytest.mark.parametrize(
    ("source_name", "media_type", "expected_name"),
    (
        ("minimal.jats.xml", "application/xml", "minimal.jats.canonical.json"),
        ("minimal.html", "text/html", "minimal.html.canonical.json"),
    ),
)
def test_derived_golden_documents_match_canonical_bytes(
    source_name: str, media_type: str, expected_name: str
) -> None:
    _, document = _parse_fixture(source_name, media_type)
    expected = json.loads((FIXTURE_ROOT.parent / "expected" / expected_name).read_text(encoding="utf-8"))
    from molly.core.ids import canonical_json_bytes

    assert document.canonical_bytes() == canonical_json_bytes(expected)


def test_source_locator_is_typed_bounded_and_non_executable() -> None:
    source_id = _source_id(b"source")
    locator = SourceLocator(
        source_artifact_id=source_id,
        kind=SourceLocatorKind.XML_ELEMENT,
        path="/article[1]/body[1]/p[2]",
    )
    assert SourceLocator.from_dict(locator.to_dict()) == locator
    with pytest.raises(CoreContractError):
        SourceLocator(source_artifact_id=source_id, kind="XML_ELEMENT", path="/tmp/file")
    with pytest.raises(CoreContractError):
        SourceLocator(source_artifact_id=source_id, kind="XML_ELEMENT", path="/article[1]/../p[1]")
    with pytest.raises(CoreContractError):
        SourceLocator(source_artifact_id=source_id, kind="PDF_PAGE", page_number=0)
    with pytest.raises(CoreContractError):
        SourceLocator(
            source_artifact_id=source_id,
            kind="PDF_REGION",
            page_number=1,
            bbox=(0.0, 0.0, float("inf"), 1.0),
        )
    with pytest.raises(CoreContractError):
        SourceLocator(source_artifact_id=source_id, kind="MINERU_ELEMENT", page_number=1)


def test_structural_paths_distinguish_repeated_siblings_and_survive_restart() -> None:
    source = b"<document><p>one</p><p>two</p><p>three</p></document>"
    source_id = _source_id(source)
    first = DocumentParserRouter().parse(
        source_artifact_id=source_id,
        source_media_type="application/xml",
        source_bytes=source,
    )
    second = DocumentParserRouter().parse(
        source_artifact_id=source_id,
        source_media_type="application/xml",
        source_bytes=source,
    )
    paths = [block.locator.path for block in first.blocks if block.kind == "PARAGRAPH"]
    assert paths == ["/document[1]/p[1]", "/document[1]/p[2]", "/document[1]/p[3]"]
    assert first.canonical_bytes() == second.canonical_bytes()


def test_parser_config_digest_binds_tool_and_document_semantics(tmp_path: Path) -> None:
    source, first = _parse_fixture("minimal.jats.xml", "application/xml")
    default = DocumentParserConfig()
    changed = replace(default, max_blocks=default.max_blocks - 1)
    first_router = DocumentParserRouter(default)
    second_router = DocumentParserRouter(changed)
    first = first_router.parse(
        source_artifact_id=_source_id(source), source_media_type="application/xml", source_bytes=source
    )
    second = second_router.parse(
        source_artifact_id=_source_id(source), source_media_type="application/xml", source_bytes=source
    )
    assert default.digest != changed.digest
    assert document_tool_specs(DocumentService(artifact_store=ArtifactStore(tmp_path / "one"), router=first_router))[0].spec_digest != document_tool_specs(
        DocumentService(artifact_store=ArtifactStore(tmp_path / "two"), router=second_router)
    )[0].spec_digest
    assert first.parser_config_digest != second.parser_config_digest
    assert first.canonical_bytes() != second.canonical_bytes()


def test_xml_security_and_resource_limits_fail_closed() -> None:
    router = DocumentParserRouter()
    source_id = _source_id(b"x")
    dangerous = (
        b'<!DOCTYPE article [ <!ENTITY xxe SYSTEM "file:///etc/passwd"> ]>'
        b'<article><article-meta/><body><p>&xxe;</p></body></article>'
    )
    with pytest.raises(MalformedDocumentError):
        router.parse(source_artifact_id=source_id, source_media_type="application/xml", source_bytes=dangerous)
    xinclude = b'<document xmlns:xi="http://www.w3.org/2001/XInclude"><xi:include href="file.xml"/></document>'
    with pytest.raises(MalformedDocumentError):
        router.parse(source_artifact_id=source_id, source_media_type="application/xml", source_bytes=xinclude)
    with pytest.raises(MalformedDocumentError):
        router.parse(source_artifact_id=source_id, source_media_type="application/xml", source_bytes=b"<document>")
    with pytest.raises(MalformedDocumentError):
        router.parse(source_artifact_id=source_id, source_media_type="application/xml", source_bytes=b"\x00\xff\x00")

    limited = DocumentParserRouter(
        DocumentParserConfig(max_node_count=3, max_nesting_depth=10)
    )
    with pytest.raises(DocumentLimitError):
        limited.parse(
            source_artifact_id=source_id,
            source_media_type="application/xml",
            source_bytes=b"<document><a/><b/><c/></document>",
        )
    shallow = DocumentParserRouter(DocumentParserConfig(max_nesting_depth=2))
    with pytest.raises(DocumentLimitError):
        shallow.parse(
            source_artifact_id=source_id,
            source_media_type="application/xml",
            source_bytes=b"<document><a><b/></a></document>",
        )
    text_limited = DocumentParserRouter(DocumentParserConfig(max_normalized_text_bytes=8))
    with pytest.raises(DocumentLimitError):
        text_limited.parse(
            source_artifact_id=source_id,
            source_media_type="application/xml",
            source_bytes=b"<document><p>123456789</p></document>",
        )
    with pytest.raises(DocumentLimitError):
        DocumentParserRouter(DocumentParserConfig(max_source_bytes=4, max_normalized_text_bytes=4)).parse(
            source_artifact_id=source_id,
            source_media_type="application/xml",
            source_bytes=b"<document/>",
        )


def test_router_is_closed_and_does_not_route_json_or_accept_model_parser_choice() -> None:
    router = DocumentParserRouter()
    source = b"{}"
    with pytest.raises(UnsupportedDocumentError):
        router.parse(source_artifact_id=_source_id(source), source_media_type="application/json", source_bytes=source)
    with pytest.raises(UnsupportedDocumentError):
        router.registry.resolve("made_up_parser")
    assert router.select_parser_id("text/html; charset=utf-8", b"<p>x</p>") == "html"


def test_html_limits_and_recoverable_malformed_structure_are_bounded() -> None:
    source_id = _source_id(b"html")
    deep = "<div>" * 4 + "text" + "</div>" * 4
    with pytest.raises(DocumentLimitError):
        DocumentParserRouter(DocumentParserConfig(max_nesting_depth=3)).parse(
            source_id, "text/html", deep.encode("utf-8")
        )
    with pytest.raises(DocumentLimitError):
        DocumentParserRouter(DocumentParserConfig(max_node_count=1)).parse(
            source_id, "text/html", b"<div><p>x</p></div>"
        )
    with pytest.raises(DocumentLimitError):
        DocumentParserRouter(DocumentParserConfig(max_normalized_text_bytes=4)).parse(
            source_id, "text/html", b"<p>12345</p>"
        )
    malformed = b"<html><body><p>one<p>two"
    first = DocumentParserRouter().parse(source_id, "text/html", malformed)
    second = DocumentParserRouter().parse(source_id, "text/html", malformed)
    assert first.canonical_bytes() == second.canonical_bytes()


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdf:
    def __init__(self, pages: list[_FakePage]) -> None:
        self.pages = pages

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


class _FakeMinerU:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, source_bytes: bytes):
        self.calls += 1
        return [
            {"page_number": 1, "element_index": 0, "type": "title", "text": "Fallback title", "bbox": [0, 0, 10, 10]},
            {"page_number": 1, "element_index": 1, "type": "paragraph", "text": "Fallback text", "bbox": [0, 10, 10, 20]},
            {
                "page_number": 1,
                "element_index": 2,
                "type": "table",
                "text": "Fallback table",
                "cells": [
                    {"row_index": 0, "column_index": 0, "is_header": True, "text": "h"},
                    {"row_index": 1, "column_index": 0, "text": "v"},
                ],
            },
        ]


def test_optional_dependencies_are_not_needed_for_xml_html_and_are_lazy_for_pdf(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    backend = _FakeMinerU()
    router = DocumentParserRouter(mineru_backend=backend)
    jats = (FIXTURE_ROOT / "minimal.jats.xml").read_bytes()
    html = (FIXTURE_ROOT / "minimal.html").read_bytes()
    router.parse(source_artifact_id=_source_id(jats), source_media_type="application/xml", source_bytes=jats)
    router.parse(source_artifact_id=_source_id(html), source_media_type="text/html", source_bytes=html)
    assert backend.calls == 0


def test_pdf_text_route_isolated_and_good_quality_skips_mineru(monkeypatch) -> None:
    backend = _FakeMinerU()
    fake_module = SimpleNamespace(open=lambda stream: _FakePdf([_FakePage("A sufficiently long deterministic PDF text layer for tests.")]))
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_module)
    router = DocumentParserRouter(mineru_backend=backend)
    source = b"%PDF-synthetic"
    document = router.parse(
        source_artifact_id=_source_id(source), source_media_type="application/pdf", source_bytes=source
    )
    assert document.parser_id == "pdf_text"
    assert document.parser_quality.status == ParserQualityStatus.GOOD.value
    assert document.blocks[0].locator.kind == SourceLocatorKind.PDF_PAGE.value
    assert backend.calls == 0


def test_pdf_text_parser_accepts_runtime_generated_synthetic_pdf() -> None:
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    pytest.importorskip("pdfplumber")
    buffer = BytesIO()
    canvas = reportlab.Canvas(buffer)
    canvas.drawString(72, 720, "CORE-04 synthetic PDF text layer fixture")
    canvas.save()
    source = buffer.getvalue()
    document = DocumentParserRouter().parse(
        source_artifact_id=_source_id(source),
        source_media_type="application/pdf",
        source_bytes=source,
    )
    assert document.parser_id == "pdf_text"
    assert document.blocks[0].locator.kind == SourceLocatorKind.PDF_PAGE.value
    assert document.parser_quality.pages_with_text == (1,)


def test_pdf_quality_floor_uses_only_configured_fake_mineru_fallback(monkeypatch) -> None:
    backend = _FakeMinerU()
    fake_module = SimpleNamespace(open=lambda stream: _FakePdf([_FakePage("x")]))
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_module)
    router = DocumentParserRouter(mineru_backend=backend)
    source = b"%PDF-synthetic"
    document = router.parse(
        source_artifact_id=_source_id(source), source_media_type="application/pdf", source_bytes=source
    )
    assert document.parser_id == "mineru"
    assert document.blocks[0].locator.kind == SourceLocatorKind.MINERU_ELEMENT.value
    assert all(item.locator.source_artifact_id == _source_id(source) for item in document.blocks)
    assert backend.calls == 1


def test_pdf_unavailable_without_fallback_is_explicit(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pdfplumber", None)
    config = DocumentParserConfig(mineru_enabled=False)
    router = DocumentParserRouter(config)
    source = b"%PDF-synthetic"
    with pytest.raises(ParserUnavailableError):
        router.parse(
            source_artifact_id=_source_id(source), source_media_type="application/pdf", source_bytes=source
        )


def test_document_service_binds_exact_source_and_returns_unpublished_canonical_draft(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    source = (FIXTURE_ROOT / "minimal.jats.xml").read_bytes()
    source_record = store.put(source, media_type="application/xml")
    service = DocumentService(artifact_store=store)
    outcome = service.parse_declared_artifact(source_record.artifact_id, reader=store.read)
    assert outcome.document is not None
    assert outcome.artifact_draft is not None
    assert outcome.result.data["canonical_document_artifact_id"] == outcome.document.artifact_id
    assert outcome.artifact_draft.schema_name == CANONICAL_SCHEMA_NAME
    assert outcome.artifact_draft.schema_version == CANONICAL_SCHEMA_VERSION
    assert not store.exists(outcome.document.artifact_id)
    with pytest.raises(DocumentIntegrityError):
        service.parse_declared_artifact(source_record.artifact_id, reader=lambda _: b"tampered")


def test_document_tool_agentloop_publishes_document_and_projects_bounded_lineage(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    source = (FIXTURE_ROOT / "minimal.html").read_bytes()
    source_record = store.put(source, media_type="text/html")
    service = DocumentService(artifact_store=store)
    registry = ToolRegistry()
    spec = register_document_tools(registry, service)[0]
    policy = ToolPolicy(
        allowed_tools=(spec.name,),
        allowed_side_effect_classes=(SideEffectClass.PURE,),
    )

    class Provider:
        def __init__(self) -> None:
            self.calls = 0
            self.outcomes = []

        def next_action(self, context, tools):
            self.calls += 1
            self.outcomes.append(context.previous_tool_outcome)
            if self.calls == 1:
                return ToolCallProposal("document_parse", input_artifact_ids=(source_record.artifact_id,))
            return StopAction("document parsed")

    provider = Provider()
    ledger = RunLedger(tmp_path / "events.jsonl")
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl")
    loop = AgentLoop(
        store=store,
        ledger=ledger,
        lineage=lineage,
        registry=registry,
        policy=policy,
        decision_provider=provider,
    )
    request = RunRequest.create(
        goal="parse one offline CORE-04 fixture",
        input_artifact_ids=(source_record.artifact_id,),
        tool_policy_digest=policy.digest,
    )
    result = loop.run(request)
    assert result.status == RunStatus.STOPPED.value
    success = next(event for event in ledger.events if event.event_type == TOOL_EXECUTION_SUCCEEDED)
    canonical_id = success.output_artifact_ids[0]
    assert store.verify(canonical_id).schema_name == CANONICAL_SCHEMA_NAME
    assert provider.outcomes[1]["data"]["status"] == "PARSED"
    assert provider.outcomes[1]["data"]["canonical_document_artifact_id"] == canonical_id
    produced = [item for item in lineage.relations if item.relation_type == "PRODUCED_BY" and item.subject_id == canonical_id]
    derived = [item for item in lineage.relations if item.relation_type == "DERIVED_FROM" and item.subject_id == canonical_id]
    consumed = [item for item in lineage.relations if item.relation_type == "CONSUMED_BY" and item.subject_id == source_record.artifact_id]
    assert len(produced) == 1
    assert len(derived) == 1 and derived[0].object_id == source_record.artifact_id
    assert len(consumed) == 1
    assert "access_profile_ref" not in store.read(canonical_id).decode("utf-8")


def test_core03_offline_full_text_artifact_flows_through_document_tool(tmp_path: Path) -> None:
    """Exercise the existing CORE-03 artifact boundary without live network."""

    source = (FIXTURE_ROOT / "minimal.jats.xml").read_bytes()
    oa_config = ProviderConfig(
        provider_id="offline_oa",
        provider_class=ProviderClass.OA_RESOLUTION,
        routes=(ProviderRoute("lookup", "oa.example.org", "/v2/{doi}", path_prefix="/v2/"),),
    )
    full_text_config = ProviderConfig(
        provider_id="offline_repository",
        provider_class=ProviderClass.FULL_TEXT,
        routes=(
            ProviderRoute(
                "article",
                "repository.example.org",
                "/articles/{path}",
                path_prefix="/articles/",
                accepted_content_types=("application/xml",),
                access_status=AccessStatus.VERIFIED_OPEN_ACCESS,
                access_basis="offline-fixture-route",
                license_status="verified-open-access",
                redistribution_basis="verified-public-safe-fixture",
            ),
        ),
    )
    config = AcquisitionConfig(providers=(oa_config, full_text_config))

    class OfflineResolver:
        provider_id = "offline_oa"

        def resolve(self, doi: str) -> ProviderRequest:
            route = oa_config.route("lookup")
            return ProviderRequest(
                provider_id="offline_oa",
                provider_config=oa_config,
                route=route,
                path=route.render_path({"doi": doi}),
                query={},
                request_shape={"operation": "resolve", "doi": doi},
                canonical_identifier=doi,
            )

        def normalize_resolution(self, body: bytes) -> tuple[SourceCandidate, ...]:
            return (
                SourceCandidate(
                    url="https://repository.example.org/articles/minimal.jats.xml",
                    content_type_hint="application/xml",
                    access_status="verified",
                    license_status="cc-by",
                ),
            )

    class OfflineTransport:
        def fetch(self, url, *, route, config, headers=(), secret_values=()):
            if config.provider_id == "offline_oa":
                body = b'{"best_oa_location":{"url":"https://repository.example.org/articles/minimal.jats.xml","url_type":"xml","license":"cc-by"}}'
                return NetworkResponse(200, {"content-type": "application/json"}, body, url, url)
            return NetworkResponse(200, {"content-type": "application/xml"}, source, url, url)

    artifact_store = ArtifactStore(tmp_path / "artifacts")
    acquisition = AcquisitionService(
        artifact_store=artifact_store,
        cache=AcquisitionCache(tmp_path / "cache"),
        config=config,
        transport=OfflineTransport(),
        oa_resolvers=(OfflineResolver(),),
    )
    acquired = acquisition.acquire_full_text("10.1234/fixture")
    content_draft = next(draft for draft in acquired.artifacts if draft.media_type == "application/xml")
    source_record = artifact_store.put(content_draft.content, media_type=content_draft.media_type)
    source_artifact_id = source_record.artifact_id
    assert source_artifact_id == acquired.data["content_artifact_id"]
    source_record = artifact_store.verify(source_artifact_id)
    assert source_record.media_type == "application/xml"
    assert acquired.data["artifact_class"] == "PUBLIC_ARTIFACT"

    service = DocumentService(artifact_store=artifact_store)
    registry = ToolRegistry()
    spec = register_document_tools(registry, service)[0]
    policy = ToolPolicy(allowed_tools=(spec.name,), allowed_side_effect_classes=(SideEffectClass.PURE,))

    class Provider:
        def __init__(self) -> None:
            self.calls = 0

        def next_action(self, context, tools):
            self.calls += 1
            return (
                ToolCallProposal("document_parse", input_artifact_ids=(source_artifact_id,))
                if self.calls == 1
                else StopAction("done")
            )

    ledger = RunLedger(tmp_path / "events.jsonl")
    lineage = ArtifactLineage(tmp_path / "lineage.jsonl")
    loop = AgentLoop(
        store=artifact_store,
        ledger=ledger,
        lineage=lineage,
        registry=registry,
        policy=policy,
        decision_provider=Provider(),
    )
    request = RunRequest.create(
        goal="normalize offline acquired source",
        input_artifact_ids=(source_artifact_id,),
        tool_policy_digest=policy.digest,
    )
    result = loop.run(request)
    assert result.status == RunStatus.STOPPED.value
    success = next(event for event in ledger.events if event.event_type == TOOL_EXECUTION_SUCCEEDED)
    canonical_id = success.output_artifact_ids[0]
    document_bytes = artifact_store.read(canonical_id)
    assert json.loads(document_bytes)["source_artifact_id"] == source_artifact_id
    assert "PUBLIC_ARTIFACT" not in document_bytes.decode("utf-8")
    assert "redistribution_basis" not in document_bytes.decode("utf-8")
    assert any(
        relation.relation_type == "DERIVED_FROM"
        and relation.subject_id == canonical_id
        and relation.object_id == source_artifact_id
        for relation in lineage.relations
    )


def test_document_tool_requires_exactly_one_declared_source(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    service = DocumentService(artifact_store=store)
    spec = document_tool_specs(service)[0]
    registry = ToolRegistry()
    register_document_tools(registry, service)
    policy = ToolPolicy(allowed_tools=(spec.name,), allowed_side_effect_classes=(SideEffectClass.PURE,))
    # The tool input schema has no model arguments; the executor boundary is
    # separately tested through a direct context-free AgentLoop failure path.
    assert spec.input_schema["additionalProperties"] is False
    assert registry.resolve_exact(spec.name, spec.version, spec.spec_digest) == spec
    assert policy.allows(spec)
    first = store.put(b"<p>one</p>", media_type="text/html")
    second = store.put(b"<p>two</p>", media_type="text/html")
    for index, inputs in enumerate(((), (first.artifact_id, second.artifact_id))):
        class Provider:
            def __init__(self, declared_inputs):
                self.declared_inputs = declared_inputs
                self.calls = 0

            def next_action(self, context, tools):
                self.calls += 1
                if self.calls == 1:
                    return ToolCallProposal("document_parse", input_artifact_ids=self.declared_inputs)
                return StopAction("input boundary checked")

        ledger = RunLedger(tmp_path / f"boundary-{index}.jsonl")
        loop = AgentLoop(
            store=store,
            ledger=ledger,
            lineage=ArtifactLineage(tmp_path / f"boundary-{index}.lineage.jsonl"),
            registry=registry,
            policy=policy,
            decision_provider=Provider(inputs),
        )
        request = RunRequest.create(
            goal="check document input boundary",
            input_artifact_ids=inputs,
            tool_policy_digest=policy.digest,
        )
        assert loop.run(request).status == RunStatus.STOPPED.value
        assert any(event.event_type == TOOL_EXECUTION_FAILED for event in ledger.events)


def test_canonical_document_rejects_dangling_and_duplicate_structure() -> None:
    _, document = _parse_fixture("minimal.jats.xml", "application/xml")
    raw = _canonical_json(document)
    raw["blocks"].append(dict(raw["blocks"][0]))
    with pytest.raises(CoreContractError):
        CanonicalDocument.from_dict(raw)


def test_canonical_output_rejects_schema_source_locator_and_cell_integrity_errors() -> None:
    _, document = _parse_fixture("minimal.jats.xml", "application/xml")
    raw = _canonical_json(document)
    raw["schema_version"] = "2"
    with pytest.raises(CoreContractError):
        CanonicalDocument.from_dict(raw)
    raw = _canonical_json(document)
    raw["blocks"][0]["locator"]["source_artifact_id"] = _source_id(b"other")
    with pytest.raises(CoreContractError):
        CanonicalDocument.from_dict(raw)
    locator = SourceLocator(
        source_artifact_id=document.source_artifact_id,
        kind=SourceLocatorKind.HTML_ELEMENT,
        path="/html[1]/p[1]",
    )
    with pytest.raises(CoreContractError):
        CanonicalCell(
            row_index=-1,
            column_index=0,
            row_span=1,
            column_span=1,
            is_header=False,
            text="bad",
            locator=locator,
        )
    raw = _canonical_json(document)
    raw["blocks"][0]["section_id"] = "missing-section"
    with pytest.raises(CoreContractError):
        CanonicalDocument.from_dict(raw)


def test_import_boundary_excludes_legacy_llm_network_and_prototype_dependencies() -> None:
    forbidden = {
        "ai4s_agent",
        "prototypes",
        "molly.acquisition",
        "httpx",
        "requests",
        "urllib",
        "socket",
        "subprocess",
        "openai",
        "anthropic",
        "molly.domains",
    }
    optional_pdf_allowed = {"pdfplumber"}
    for path in (PROJECT_ROOT / "src" / "molly" / "documents").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name in optional_pdf_allowed:
                    assert path.name == "pdf_text.py"
                assert not any(name == item or name.startswith(item + ".") for item in forbidden), (path, name)


def test_optional_parser_modules_are_not_referenced_by_structured_parser_code() -> None:
    for path in (
        PROJECT_ROOT / "src" / "molly" / "documents" / "canonical.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "locators.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "router.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "parsers" / "_common.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "parsers" / "_xml_common.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "parsers" / "html.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "parsers" / "jats.py",
        PROJECT_ROOT / "src" / "molly" / "documents" / "parsers" / "xml.py",
        ):
        text = path.read_text(encoding="utf-8")
        assert "pdfplumber" not in text
