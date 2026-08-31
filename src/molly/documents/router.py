"""Closed, deterministic parser selection for CORE-04 documents.

The router is deliberately a host-owned registry.  A model can ask the
document tool to parse a declared artifact, but it cannot choose a parser,
backend, or resource profile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from molly.core.errors import CoreContractError
from molly.core.ids import (
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    thaw_json,
    validate_identifier,
)

from .canonical import DOCUMENT_MEDIA_TYPES, CanonicalDocument
from .errors import (
    DocumentContractError,
    DocumentLimitError,
    ParserQualityError,
    ParserUnavailableError,
    UnsupportedDocumentError,
)
from .parsers.html import HtmlParser
from .parsers.jats import JatsParser
from .parsers.mineru import MinerUBackend, MinerUFallbackParser
from .parsers.pdf_text import PdfTextParser
from .parsers.xml import GenericXmlParser
from .quality import ParserQuality, ParserQualityStatus


_DEFAULT_PARSER_VERSIONS = {
    "jats": "1",
    "xml": "1",
    "html": "1",
    "pdf_text": "1",
    "mineru": "1",
}
_DEFAULT_PRIORITY = ("jats", "xml", "html", "pdf_text", "mineru")
_MAX_CONFIG_PROFILE_REF = 128


def _bounded_positive(value: int, *, field: str, maximum: int = 100_000_000) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise DocumentContractError(f"{field} must be a bounded positive integer")
    return value


def _parser_id(value: str, *, field: str = "parser id") -> str:
    try:
        return validate_identifier(value, field=field)
    except CoreContractError as exc:
        raise DocumentContractError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class DocumentParserConfig:
    """Immutable, non-secret parser semantics and resource limits."""

    canonical_schema_version: str = "1"
    parser_versions: Mapping[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_PARSER_VERSIONS)
    )
    routing_priority: tuple[str, ...] = _DEFAULT_PRIORITY
    max_source_bytes: int = 25 * 1024 * 1024
    max_node_count: int = 100_000
    max_nesting_depth: int = 100
    max_sections: int = 10_000
    max_blocks: int = 50_000
    max_tables: int = 2_000
    max_table_cells: int = 100_000
    max_figures: int = 10_000
    max_references: int = 20_000
    max_text_chars: int = 1_000_000
    max_normalized_text_bytes: int = 5 * 1024 * 1024
    max_page_count: int = 10_000
    pdf_min_text_chars: int = 32
    mineru_enabled: bool = True
    mineru_profile_ref: str | None = "mineru-fallback"

    def __post_init__(self) -> None:
        if self.canonical_schema_version != "1":
            raise DocumentContractError("only canonical document schema version 1 is supported")
        versions: dict[str, str] = {}
        if not isinstance(self.parser_versions, Mapping) or not self.parser_versions:
            raise DocumentContractError("parser_versions must be a non-empty object")
        for raw_id, raw_version in self.parser_versions.items():
            parser_id = _parser_id(raw_id, field="parser version parser_id")
            version = _parser_id(raw_version, field=f"parser version for {parser_id}")
            versions[parser_id] = version
        priority = tuple(_parser_id(value, field="routing priority parser_id") for value in self.routing_priority)
        if len(priority) != len(set(priority)):
            raise DocumentContractError("routing_priority must not contain duplicates")
        if set(priority) - set(versions):
            raise DocumentContractError("routing_priority references an unconfigured parser")
        if not priority:
            raise DocumentContractError("routing_priority cannot be empty")
        object.__setattr__(self, "parser_versions", freeze_json_mapping(versions, field="parser_versions"))
        object.__setattr__(self, "routing_priority", priority)

        limits = (
            ("max_source_bytes", self.max_source_bytes, 25 * 1024 * 1024),
            ("max_node_count", self.max_node_count, 10_000_000),
            ("max_nesting_depth", self.max_nesting_depth, 10_000),
            ("max_sections", self.max_sections, 1_000_000),
            ("max_blocks", self.max_blocks, 1_000_000),
            ("max_tables", self.max_tables, 100_000),
            ("max_table_cells", self.max_table_cells, 2_000_000),
            ("max_figures", self.max_figures, 1_000_000),
            ("max_references", self.max_references, 1_000_000),
            ("max_text_chars", self.max_text_chars, 25_000_000),
            ("max_normalized_text_bytes", self.max_normalized_text_bytes, 25 * 1024 * 1024),
            ("max_page_count", self.max_page_count, 100_000),
            ("pdf_min_text_chars", self.pdf_min_text_chars, 1_000_000),
        )
        for name, value, maximum in limits:
            _bounded_positive(value, field=name, maximum=maximum)
        if self.max_normalized_text_bytes > self.max_source_bytes:
            raise DocumentContractError("normalized text limit cannot exceed source byte limit")
        if not isinstance(self.mineru_enabled, bool):
            raise DocumentContractError("mineru_enabled must be boolean")
        if self.mineru_profile_ref is not None:
            if not isinstance(self.mineru_profile_ref, str) or not self.mineru_profile_ref.strip():
                raise DocumentContractError("mineru_profile_ref must be a bounded reference")
            if len(self.mineru_profile_ref) > _MAX_CONFIG_PROFILE_REF:
                raise DocumentContractError("mineru_profile_ref is too long")
            _parser_id(self.mineru_profile_ref, field="mineru_profile_ref")

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_schema_version": self.canonical_schema_version,
            "parser_versions": thaw_json(self.parser_versions),
            "routing_priority": list(self.routing_priority),
            "normalization_policy": {
                "unicode_normal_form": "NFC",
                "line_endings": "LF",
                "whitespace": "collapse-and-trim",
                "html_non_content": ["noscript", "script", "style"],
                "xml_entities": "standard-parser-resolved-without-external-resolution",
            },
            "limits": {
                "max_source_bytes": self.max_source_bytes,
                "max_node_count": self.max_node_count,
                "max_nesting_depth": self.max_nesting_depth,
                "max_sections": self.max_sections,
                "max_blocks": self.max_blocks,
                "max_tables": self.max_tables,
                "max_table_cells": self.max_table_cells,
                "max_figures": self.max_figures,
                "max_references": self.max_references,
                "max_text_chars": self.max_text_chars,
                "max_normalized_text_bytes": self.max_normalized_text_bytes,
                "max_page_count": self.max_page_count,
                "pdf_min_text_chars": self.pdf_min_text_chars,
            },
            "pdf_fallback": {
                "mineru_enabled": self.mineru_enabled,
                "mineru_profile_ref": self.mineru_profile_ref,
            },
        }

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    @property
    def parser_config_digest(self) -> str:
        return self.digest

    def quality(
        self,
        *,
        text_char_count: int,
        block_count: int,
        table_count: int,
        page_count: int = 0,
        pages_with_text: tuple[int, ...] = (),
        status: str | ParserQualityStatus = ParserQualityStatus.GOOD,
        warning_codes: tuple[str, ...] = (),
    ) -> ParserQuality:
        return ParserQuality(
            status=status,
            text_char_count=text_char_count,
            block_count=block_count,
            table_count=table_count,
            page_count=page_count,
            pages_with_text=pages_with_text,
            warning_codes=warning_codes,
        )


class DocumentParser(Protocol):
    parser_id: str
    version: str

    def parse(
        self,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
        config: DocumentParserConfig,
    ) -> CanonicalDocument:
        ...


class DocumentParserRegistry:
    """Closed server-owned parser map used only by the router."""

    def __init__(self, parsers: Sequence[DocumentParser] = ()) -> None:
        self._parsers: dict[str, DocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        parser_id = _parser_id(getattr(parser, "parser_id", None), field="parser_id")
        version = _parser_id(getattr(parser, "version", None), field=f"{parser_id} version")
        if not callable(getattr(parser, "parse", None)):
            raise DocumentContractError("registered parser must provide parse()")
        if parser_id in self._parsers:
            raise DocumentContractError(f"duplicate document parser: {parser_id}")
        self._parsers[parser_id] = parser

    def resolve(self, parser_id: str) -> DocumentParser:
        parser_id = _parser_id(parser_id)
        try:
            return self._parsers[parser_id]
        except KeyError as exc:
            raise UnsupportedDocumentError(f"no registered parser for {parser_id}") from exc

    @property
    def parser_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._parsers))

    def validate_config(self, config: DocumentParserConfig) -> None:
        for parser_id, expected_version in config.parser_versions.items():
            parser = self.resolve(parser_id)
            if parser.version != expected_version:
                raise DocumentContractError(
                    f"configured parser version does not match registered {parser_id}"
                )

    @classmethod
    def default(
        cls,
        *,
        mineru_backend: MinerUBackend | None = None,
    ) -> "DocumentParserRegistry":
        return cls(
            (
                JatsParser(),
                GenericXmlParser(),
                HtmlParser(),
                PdfTextParser(),
                MinerUFallbackParser(mineru_backend),
            )
        )


def _declared_media_type(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UnsupportedDocumentError("source media type is required")
    media_type = value.casefold().split(";", 1)[0].strip()
    if media_type not in DOCUMENT_MEDIA_TYPES:
        raise UnsupportedDocumentError(f"unsupported document media type: {media_type}")
    return media_type


class DocumentParserRouter:
    """Select and invoke one parser from the server-owned closed registry."""

    def __init__(
        self,
        config: DocumentParserConfig | None = None,
        registry: DocumentParserRegistry | None = None,
        *,
        mineru_backend: MinerUBackend | None = None,
    ) -> None:
        self.registry = registry or DocumentParserRegistry.default(mineru_backend=mineru_backend)
        if config is None and registry is not None:
            configured_ids = self.registry.parser_ids
            versions = {
                parser_id: self.registry.resolve(parser_id).version
                for parser_id in configured_ids
            }
            priority = tuple(parser_id for parser_id in _DEFAULT_PRIORITY if parser_id in versions)
            self.config = DocumentParserConfig(
                parser_versions=versions,
                routing_priority=priority or configured_ids,
            )
        else:
            self.config = config or DocumentParserConfig()
        self.registry.validate_config(self.config)

    @property
    def parser_config_digest(self) -> str:
        return self.config.digest

    @property
    def execution_config_digest(self) -> str:
        return self.config.digest

    def _select_xml_parser(self, source_bytes: bytes) -> str:
        try:
            jats = self.registry.resolve("jats")
        except UnsupportedDocumentError:
            self.registry.resolve("xml")
            return "xml"
        if not isinstance(jats, JatsParser):
            # A custom server registry may provide an equivalent parser.  It
            # is still selected only by the host's XML classification rule.
            try:
                candidate = jats.is_jats(source_bytes, self.config)  # type: ignore[attr-defined]
            except AttributeError:
                candidate = False
            return "jats" if candidate else "xml"
        return "jats" if jats.is_jats(source_bytes, self.config) else "xml"

    def select_parser_id(self, source_media_type: str, source_bytes: bytes) -> str:
        media_type = _declared_media_type(source_media_type)
        if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
            raise DocumentContractError("document source must be bytes-like")
        payload = bytes(source_bytes)
        if len(payload) > self.config.max_source_bytes:
            raise DocumentLimitError("document source exceeds the configured byte limit")
        if media_type in {"application/xml", "text/xml"}:
            parser_id = self._select_xml_parser(payload)
        elif media_type == "text/html":
            parser_id = "html"
        elif media_type == "application/pdf":
            parser_id = "pdf_text"
        else:
            raise UnsupportedDocumentError(f"unsupported document media type: {media_type}")
        self.registry.resolve(parser_id)
        return parser_id

    def parse(
        self,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
    ) -> CanonicalDocument:
        media_type = _declared_media_type(source_media_type)
        payload = bytes(source_bytes) if isinstance(source_bytes, (bytes, bytearray, memoryview)) else source_bytes
        parser_id = self.select_parser_id(media_type, payload)
        if media_type in {"application/xml", "text/xml"}:
            if parser_id == "jats":
                try:
                    return self.registry.resolve("jats").parse(
                        source_artifact_id=source_artifact_id,
                        source_media_type=media_type,
                        source_bytes=payload,
                        config=self.config,
                    )
                except UnsupportedDocumentError:
                    # A well-formed non-JATS XML document is intentionally
                    # handled by the conservative generic XML route.
                    parser_id = "xml"
            return self.registry.resolve(parser_id).parse(
                source_artifact_id=source_artifact_id,
                source_media_type=media_type,
                source_bytes=payload,
                config=self.config,
            )
        if media_type == "text/html":
            return self.registry.resolve("html").parse(
                source_artifact_id=source_artifact_id,
                source_media_type=media_type,
                source_bytes=payload,
                config=self.config,
            )
        if media_type == "application/pdf":
            return self._parse_pdf(
                source_artifact_id=source_artifact_id,
                source_media_type=media_type,
                source_bytes=payload,
            )
        raise UnsupportedDocumentError(f"unsupported document media type: {media_type}")

    def _parse_pdf(
        self,
        *,
        source_artifact_id: str,
        source_media_type: str,
        source_bytes: bytes,
    ) -> CanonicalDocument:
        try:
            pdf_parser = self.registry.resolve("pdf_text")
        except UnsupportedDocumentError:
            pdf_parser = None
        try:
            document = (
                None
                if pdf_parser is None
                else pdf_parser.parse(
                    source_artifact_id=source_artifact_id,
                    source_media_type=source_media_type,
                    source_bytes=source_bytes,
                    config=self.config,
                )
            )
        except ParserUnavailableError:
            document = None
        if document is not None and document.parser_quality.status not in {
            ParserQualityStatus.INSUFFICIENT.value,
        }:
            return document
        if not self.config.mineru_enabled:
            if document is None:
                raise ParserUnavailableError("PDF text parser is unavailable")
            raise ParserQualityError("PDF text parser output is below the configured quality floor")
        try:
            fallback = self.registry.resolve("mineru")
        except UnsupportedDocumentError as exc:
            raise ParserUnavailableError("MinerU fallback is unavailable") from exc
        try:
            return fallback.parse(
                source_artifact_id=source_artifact_id,
                source_media_type=source_media_type,
                source_bytes=source_bytes,
                config=self.config,
            )
        except ParserUnavailableError:
            if document is None:
                raise ParserUnavailableError("PDF parser and MinerU fallback are unavailable")
            raise ParserUnavailableError("PDF text quality is insufficient and MinerU fallback is unavailable")

    # Descriptive aliases used by host callers; neither accepts a model parser.
    def route(self, *, source_media_type: str, source_bytes: bytes, source_artifact_id: str) -> CanonicalDocument:
        return self.parse(
            source_artifact_id=source_artifact_id,
            source_media_type=source_media_type,
            source_bytes=source_bytes,
        )


__all__ = [
    "DocumentParser",
    "DocumentParserConfig",
    "DocumentParserRegistry",
    "DocumentParserRouter",
]
