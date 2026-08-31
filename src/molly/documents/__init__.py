"""Source-neutral deterministic document normalization for Molly Core v2."""

from .canonical import (
    CANONICAL_SCHEMA_NAME,
    CANONICAL_SCHEMA_VERSION,
    CanonicalBlock,
    CanonicalBlockKind,
    CanonicalCell,
    CanonicalDocument,
    CanonicalFigure,
    CanonicalReference,
    CanonicalSection,
    CanonicalTable,
    deterministic_object_id,
)
from .errors import (
    DocumentContractError,
    DocumentIntegrityError,
    DocumentLimitError,
    MalformedDocumentError,
    ParserQualityError,
    ParserUnavailableError,
    UnsupportedDocumentError,
)
from .locators import SourceLocator, SourceLocatorKind
from .parsers import (
    GenericXmlParser,
    HtmlParser,
    JatsParser,
    MinerUBackend,
    MinerUCell,
    MinerUElement,
    MinerUFallbackParser,
    PdfTextParser,
)
from .quality import ParserQuality, ParserQualityStatus
from .router import (
    DocumentParser,
    DocumentParserConfig,
    DocumentParserRegistry,
    DocumentParserRouter,
)
from .service import DocumentParseOutcome, DocumentService
from .tools import document_tool_specs, register_document_tools


__all__ = [
    "CANONICAL_SCHEMA_NAME",
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalBlock",
    "CanonicalBlockKind",
    "CanonicalCell",
    "CanonicalDocument",
    "CanonicalFigure",
    "CanonicalReference",
    "CanonicalSection",
    "CanonicalTable",
    "DocumentContractError",
    "DocumentIntegrityError",
    "DocumentLimitError",
    "DocumentParseOutcome",
    "DocumentParser",
    "DocumentParserConfig",
    "DocumentParserRegistry",
    "DocumentParserRouter",
    "DocumentService",
    "GenericXmlParser",
    "HtmlParser",
    "JatsParser",
    "MalformedDocumentError",
    "MinerUBackend",
    "MinerUCell",
    "MinerUElement",
    "MinerUFallbackParser",
    "ParserQuality",
    "ParserQualityError",
    "ParserQualityStatus",
    "ParserUnavailableError",
    "PdfTextParser",
    "SourceLocator",
    "SourceLocatorKind",
    "UnsupportedDocumentError",
    "deterministic_object_id",
    "document_tool_specs",
    "register_document_tools",
]
