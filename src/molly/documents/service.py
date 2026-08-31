"""Host-owned document normalization service for CORE-04."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from molly.core.artifacts import ArtifactStore
from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    sha256_bytes,
    validate_artifact_id,
)
from molly.core.tools import ArtifactDraft, ToolResult

from .canonical import CANONICAL_SCHEMA_NAME, CANONICAL_SCHEMA_VERSION, CanonicalDocument
from .errors import DocumentIntegrityError, ParserUnavailableError
from .router import DocumentParserRouter


@dataclass(frozen=True, slots=True)
class DocumentParseOutcome:
    """The validated document result before AgentLoop publication."""

    source_artifact_id: str
    document: CanonicalDocument | None
    canonical_bytes: bytes | None
    artifact_draft: ArtifactDraft | None
    result: ToolResult

    @property
    def canonical_document_artifact_id(self) -> str | None:
        return None if self.document is None else self.document.artifact_id


def _content_family(media_type: str) -> str | None:
    normalized = media_type.casefold().split(";", 1)[0].strip()
    return {
        "application/xml": "xml",
        "text/xml": "xml",
        "text/html": "html",
        "application/pdf": "pdf",
    }.get(normalized)


class DocumentService:
    """Normalize one exact declared source artifact without owning execution."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        router: DocumentParserRouter | None = None,
    ) -> None:
        if not isinstance(artifact_store, ArtifactStore):
            raise CoreContractError("DocumentService requires an ArtifactStore")
        if router is not None and not isinstance(router, DocumentParserRouter):
            raise CoreContractError("DocumentService router must be a DocumentParserRouter")
        self.artifact_store = artifact_store
        self.router = router or DocumentParserRouter()

    @property
    def parser_config_digest(self) -> str:
        return self.router.parser_config_digest

    @staticmethod
    def _verify_reader_bytes(
        artifact_id: str,
        expected_sha256: str,
        expected_size: int,
        reader: Callable[[str], bytes],
    ) -> bytes:
        try:
            source_bytes = reader(artifact_id)
        except Exception as exc:
            raise DocumentIntegrityError("declared source artifact could not be read") from exc
        if not isinstance(source_bytes, (bytes, bytearray, memoryview)):
            raise DocumentIntegrityError("declared source reader returned non-bytes")
        payload = bytes(source_bytes)
        if len(payload) != expected_size or sha256_bytes(payload) != expected_sha256:
            raise DocumentIntegrityError("declared source bytes do not match ArtifactRecord")
        return payload

    @staticmethod
    def _unavailable_result(artifact_id: str, media_type: str) -> ToolResult:
        family = _content_family(media_type)
        data = {
            "status": "PARSER_UNAVAILABLE",
            "source_artifact_id": artifact_id,
            "source_content_family": family,
            "canonical_document_artifact_id": None,
            "parser_id": None,
            "parser_version": None,
            "quality_status": None,
            "section_count": 0,
            "block_count": 0,
            "table_count": 0,
            "figure_count": 0,
            "reference_count": 0,
        }
        return ToolResult(data=data, artifacts=())

    def parse_declared_artifact(
        self,
        artifact_id: str,
        *,
        reader: Callable[[str], bytes] | None = None,
    ) -> DocumentParseOutcome:
        """Parse only the exact artifact ID declared by the current tool call."""

        validate_artifact_id(artifact_id)
        reader = self.artifact_store.read if reader is None else reader
        if not callable(reader):
            raise CoreContractError("declared source reader must be callable")
        record = self.artifact_store.get_metadata(artifact_id)
        source_bytes = self._verify_reader_bytes(
            artifact_id,
            record.sha256,
            record.size_bytes,
            reader,
        )
        try:
            document = self.router.parse(
                source_artifact_id=artifact_id,
                source_media_type=record.media_type,
                source_bytes=source_bytes,
            )
        except ParserUnavailableError:
            return DocumentParseOutcome(
                source_artifact_id=artifact_id,
                document=None,
                canonical_bytes=None,
                artifact_draft=None,
                result=self._unavailable_result(artifact_id, record.media_type),
            )

        if not isinstance(document, CanonicalDocument):
            raise DocumentIntegrityError("parser returned a non-canonical document")
        if document.source_artifact_id != artifact_id:
            raise DocumentIntegrityError("parser output source identity does not match declaration")
        canonical_bytes = document.canonical_bytes()
        digest = sha256_bytes(canonical_bytes)
        if document.canonical_document_sha256 != digest:
            raise DocumentIntegrityError("canonical document digest is inconsistent")
        if document.artifact_id != artifact_id_for_sha256(digest):
            raise DocumentIntegrityError("canonical document artifact identity is inconsistent")
        draft = ArtifactDraft(
            content=canonical_bytes,
            media_type="application/json",
            schema_name=CANONICAL_SCHEMA_NAME,
            schema_version=CANONICAL_SCHEMA_VERSION,
        )
        result = ToolResult(
            data={
                "status": "PARSED",
                "source_artifact_id": artifact_id,
                "source_content_family": document.source_content_family,
                "canonical_document_artifact_id": document.artifact_id,
                "canonical_document_sha256": digest,
                "parser_id": document.parser_id,
                "parser_version": document.parser_version,
                "quality_status": document.parser_quality.status,
                "section_count": len(document.sections),
                "block_count": len(document.blocks),
                "table_count": len(document.tables),
                "figure_count": len(document.figures),
                "reference_count": len(document.references),
            },
            artifacts=(draft,),
        )
        return DocumentParseOutcome(
            source_artifact_id=artifact_id,
            document=document,
            canonical_bytes=canonical_bytes,
            artifact_draft=draft,
            result=result,
        )

    def parse_artifact(self, artifact_id: str) -> DocumentParseOutcome:
        """Convenience host path using the service-owned ArtifactStore reader."""

        validate_artifact_id(artifact_id)
        return self.parse_declared_artifact(artifact_id, reader=self.artifact_store.read)

    def parse(self, artifact_id: str) -> DocumentParseOutcome:
        """Descriptive alias for :meth:`parse_artifact`."""

        return self.parse_artifact(artifact_id)


__all__ = ["DocumentParseOutcome", "DocumentService"]
