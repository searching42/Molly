"""Deterministic evidence candidates extracted from CanonicalDocument only.

Candidates are structural hints, not scientific claims.  They deliberately
carry source locators and no run, timestamp, UUID, filesystem, or provider
state so the same canonical document and extractor configuration always yield
the same candidate bundle bytes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Callable

from molly.core.artifacts import ArtifactRecord
from molly.core.errors import CoreContractError
from molly.core.ids import (
    artifact_id_for_sha256,
    canonical_json_bytes,
    freeze_json_mapping,
    sha256_bytes,
    thaw_json,
    validate_artifact_id,
    validate_digest_reference,
    validate_identifier,
)
from molly.core.tools import ArtifactDraft
from molly.documents.canonical import CanonicalBlock, CanonicalDocument, CanonicalTable
from molly.documents.locators import SourceLocator

from .errors import EvidenceContractError, EvidenceIntegrityError


CANDIDATE_SCHEMA_NAME = "molly.evidence.candidate-bundle"
CANDIDATE_SCHEMA_VERSION = "1"
MAX_CANDIDATES = 10_000
MAX_CANDIDATE_TEXT_CHARS = 16_384
MAX_HINTS = 32
MAX_CONTEXT_CHARS = 8_192


class CandidateType(str, Enum):
    TEXT_EVIDENCE = "TEXT_EVIDENCE"
    TABLE_ROW = "TABLE_ROW"
    TABLE_CELL_GROUP = "TABLE_CELL_GROUP"
    CAPTION_EVIDENCE = "CAPTION_EVIDENCE"


def _candidate_type(value: str | CandidateType) -> str:
    candidate = value.value if isinstance(value, CandidateType) else value
    if not isinstance(candidate, str):
        raise EvidenceContractError("candidate_type must be text")
    try:
        return CandidateType(candidate.strip().upper()).value
    except ValueError as exc:
        raise EvidenceContractError(f"unknown candidate_type: {candidate!r}") from exc


def _bounded_text(value: str, *, field: str, maximum: int, required: bool = False) -> str:
    if not isinstance(value, str):
        raise EvidenceContractError(f"{field} must be text")
    if len(value) > maximum or "\x00" in value:
        raise EvidenceContractError(f"{field} exceeds its bounded text contract")
    if required and not value.strip():
        raise EvidenceContractError(f"{field} is required")
    return value


def _locator(value: SourceLocator | Mapping[str, Any], *, field: str) -> SourceLocator:
    if isinstance(value, SourceLocator):
        return value
    if isinstance(value, Mapping):
        try:
            return SourceLocator.from_dict(value)
        except Exception as exc:
            raise EvidenceContractError(f"{field} is malformed") from exc
    raise EvidenceContractError(f"{field} must be a SourceLocator")


def _locators(values: Sequence[SourceLocator | Mapping[str, Any]], *, source_id: str) -> tuple[SourceLocator, ...]:
    result = tuple(_locator(value, field="source_locator") for value in values)
    if not result:
        raise EvidenceContractError("source_locators must not be empty")
    if len(result) > 256:
        raise EvidenceContractError("source_locators exceed the bounded limit")
    if any(item.source_artifact_id != source_id for item in result):
        raise EvidenceIntegrityError("candidate locator source does not match source_artifact_id")
    if len({item.canonical_bytes() for item in result}) != len(result):
        raise EvidenceContractError("source_locators must not contain duplicates")
    return result


def _cell_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        raise EvidenceContractError("table_cells must contain canonical cell objects")
    allowed = {"row_index", "column_index", "row_span", "column_span", "is_header", "text", "locator"}
    if set(value) - allowed:
        raise EvidenceContractError("table cell has unknown fields")
    return {
        "row_index": value["row_index"],
        "column_index": value["column_index"],
        "row_span": value.get("row_span", 1),
        "column_span": value.get("column_span", 1),
        "is_header": bool(value.get("is_header", False)),
        "text": _bounded_text(str(value.get("text", "")), field="table cell text", maximum=MAX_CONTEXT_CHARS),
        "locator": _locator(value["locator"], field="table cell locator").to_dict(),
    }


def _source_payload(candidate: "EvidenceCandidate") -> dict[str, Any]:
    return {
        "canonical_document_artifact_id": candidate.canonical_document_artifact_id,
        "source_artifact_id": candidate.source_artifact_id,
        "candidate_type": candidate.candidate_type,
        "source_locators": [item.to_dict() for item in candidate.source_locators],
        "source_text": candidate.source_text,
        "table_cells": list(candidate.table_cells),
        "structural_context": thaw_json(candidate.structural_context),
        "field_hints": thaw_json(candidate.field_hints),
        "extractor_version": candidate.extractor_version,
        "extractor_config_digest": candidate.extractor_config_digest,
    }


@dataclass(frozen=True, slots=True)
class CandidateExtractorConfig:
    """Server-owned deterministic extractor configuration."""

    extractor_version: str = "1"
    aliases: Mapping[str, Sequence[str]] = field(default_factory=lambda: {
        "molecule_identity": ("molecule", "smiles", "inchi", "inchikey", "compound", "structure"),
        "property_id": ("property", "plqy", "photoluminescence quantum yield", "quantum yield"),
        "property_value": ("value", "plqy value", "quantum yield value"),
        "unit": ("unit", "units", "fraction", "%", "percent", "percentage"),
        "measurement_condition": ("condition", "conditions", "solvent", "medium", "host", "dopant", "temperature"),
    })
    max_candidates: int = MAX_CANDIDATES
    max_candidate_text_chars: int = MAX_CANDIDATE_TEXT_CHARS

    def __post_init__(self) -> None:
        validate_identifier(self.extractor_version, field="extractor_version")
        if isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int) or not 1 <= self.max_candidates <= MAX_CANDIDATES:
            raise EvidenceContractError("max_candidates is outside the bounded range")
        if isinstance(self.max_candidate_text_chars, bool) or not isinstance(self.max_candidate_text_chars, int) or not 64 <= self.max_candidate_text_chars <= MAX_CANDIDATE_TEXT_CHARS:
            raise EvidenceContractError("max_candidate_text_chars is outside the bounded range")
        if not isinstance(self.aliases, Mapping):
            raise EvidenceContractError("aliases must be an object")
        normalized: dict[str, tuple[str, ...]] = {}
        for key, values in self.aliases.items():
            validate_identifier(str(key), field="extractor alias field")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise EvidenceContractError("extractor aliases must be string sequences")
            clean = tuple(_bounded_text(str(item).casefold().strip(), field="extractor alias", maximum=128, required=True) for item in values)
            normalized[str(key)] = tuple(dict.fromkeys(clean))
        object.__setattr__(self, "aliases", freeze_json_mapping(normalized, field="extractor aliases"))

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "extractor_version": self.extractor_version,
            "aliases": thaw_json(self.aliases),
            "max_candidates": self.max_candidates,
            "max_candidate_text_chars": self.max_candidate_text_chars,
        }


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """One deterministic structural evidence candidate."""

    candidate_id: str
    canonical_document_artifact_id: str
    source_artifact_id: str
    candidate_type: str | CandidateType
    source_locator: SourceLocator | Mapping[str, Any]
    source_text: str = ""
    table_cells: tuple[Mapping[str, Any], ...] = ()
    structural_context: Mapping[str, Any] = field(default_factory=dict)
    field_hints: Mapping[str, Any] = field(default_factory=dict)
    extractor_version: str = "1"
    extractor_config_digest: str = ""
    source_locators: tuple[SourceLocator | Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        validate_identifier(self.candidate_id, field="candidate_id")
        validate_artifact_id(self.canonical_document_artifact_id)
        validate_artifact_id(self.source_artifact_id)
        object.__setattr__(self, "candidate_type", _candidate_type(self.candidate_type))
        primary = _locator(self.source_locator, field="source_locator")
        all_locators = self.source_locators or (primary,)
        normalized_locators = _locators(all_locators, source_id=self.source_artifact_id)
        if primary.canonical_bytes() != normalized_locators[0].canonical_bytes():
            raise EvidenceContractError("source_locator must equal the first source_locators entry")
        object.__setattr__(self, "source_locator", primary)
        object.__setattr__(self, "source_locators", normalized_locators)
        object.__setattr__(self, "source_text", _bounded_text(self.source_text, field="source_text", maximum=MAX_CANDIDATE_TEXT_CHARS))
        normalized_cells = tuple(_cell_dict(item) for item in self.table_cells)
        if len(normalized_cells) > 256:
            raise EvidenceContractError("table_cells exceed the bounded limit")
        object.__setattr__(self, "table_cells", normalized_cells)
        object.__setattr__(self, "structural_context", freeze_json_mapping(self.structural_context, field="structural_context"))
        object.__setattr__(self, "field_hints", freeze_json_mapping(self.field_hints, field="field_hints"))
        validate_identifier(self.extractor_version, field="extractor_version")
        object.__setattr__(self, "extractor_config_digest", validate_digest_reference(self.extractor_config_digest, field="extractor_config_digest"))

    @property
    def digest(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_dict(include_id=False)))

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        payload = {
            "canonical_document_artifact_id": self.canonical_document_artifact_id,
            "source_artifact_id": self.source_artifact_id,
            "candidate_type": self.candidate_type,
            "source_locator": self.source_locator.to_dict(),
            "source_locators": [item.to_dict() for item in self.source_locators],
            "source_text": self.source_text,
            "table_cells": list(self.table_cells),
            "structural_context": thaw_json(self.structural_context),
            "field_hints": thaw_json(self.field_hints),
            "extractor_version": self.extractor_version,
            "extractor_config_digest": self.extractor_config_digest,
        }
        if include_id:
            payload["candidate_id"] = self.candidate_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        canonical_document_artifact_id: str,
        source_artifact_id: str,
        candidate_type: str | CandidateType,
        source_locators: Sequence[SourceLocator],
        source_text: str = "",
        table_cells: Sequence[Mapping[str, Any]] = (),
        structural_context: Mapping[str, Any] | None = None,
        field_hints: Mapping[str, Any] | None = None,
        extractor_version: str,
        extractor_config_digest: str,
    ) -> "EvidenceCandidate":
        candidate_type_value = _candidate_type(candidate_type)
        candidate_without_id = {
            "canonical_document_artifact_id": canonical_document_artifact_id,
            "source_artifact_id": source_artifact_id,
            "candidate_type": candidate_type_value,
            "source_locator": source_locators[0].to_dict(),
            "source_locators": [item.to_dict() for item in source_locators],
            "source_text": source_text,
            "table_cells": list(table_cells),
            "structural_context": dict(structural_context or {}),
            "field_hints": dict(field_hints or {}),
            "extractor_version": extractor_version,
            "extractor_config_digest": extractor_config_digest,
        }
        candidate_id = f"candidate_{sha256_bytes(canonical_json_bytes(candidate_without_id))}"
        return cls(
            candidate_id=candidate_id,
            source_locator=source_locators[0],
            source_locators=tuple(source_locators),
            canonical_document_artifact_id=canonical_document_artifact_id,
            source_artifact_id=source_artifact_id,
            candidate_type=candidate_type_value,
            source_text=source_text,
            table_cells=tuple(table_cells),
            structural_context=structural_context or {},
            field_hints=field_hints or {},
            extractor_version=extractor_version,
            extractor_config_digest=extractor_config_digest,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceCandidate":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("candidate must be an object")
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if set(value) - allowed:
            raise EvidenceContractError(f"candidate has unknown fields: {sorted(set(value) - allowed)!r}")
        try:
            candidate = cls(
                candidate_id=str(value["candidate_id"]),
                canonical_document_artifact_id=str(value["canonical_document_artifact_id"]),
                source_artifact_id=str(value["source_artifact_id"]),
                candidate_type=value["candidate_type"],
                source_locator=SourceLocator.from_dict(value["source_locator"]),
                source_locators=tuple(SourceLocator.from_dict(item) for item in value.get("source_locators", (value["source_locator"],))),
                source_text=str(value.get("source_text", "")),
                table_cells=tuple(value.get("table_cells", ())),
                structural_context=dict(value.get("structural_context", {})),
                field_hints=dict(value.get("field_hints", {})),
                extractor_version=str(value["extractor_version"]),
                extractor_config_digest=str(value["extractor_config_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("candidate is malformed") from exc
        expected = f"candidate_{candidate.digest}"
        if candidate.candidate_id != expected:
            raise EvidenceIntegrityError("candidate ID is not deterministic for its content")
        return candidate


@dataclass(frozen=True, slots=True)
class EvidenceCandidateBundle:
    """Immutable deterministic set of candidates for one canonical document."""

    schema_version: str
    canonical_document_artifact_id: str
    source_artifact_id: str
    extractor_version: str
    extractor_config_digest: str
    candidates: tuple[EvidenceCandidate, ...]
    candidate_count: int | None = None
    schema_name: str = CANDIDATE_SCHEMA_NAME

    def __post_init__(self) -> None:
        if self.schema_name != CANDIDATE_SCHEMA_NAME or self.schema_version != CANDIDATE_SCHEMA_VERSION:
            raise EvidenceContractError("unsupported candidate bundle schema")
        validate_artifact_id(self.canonical_document_artifact_id)
        validate_artifact_id(self.source_artifact_id)
        validate_identifier(self.extractor_version, field="extractor_version")
        object.__setattr__(self, "extractor_config_digest", validate_digest_reference(self.extractor_config_digest, field="extractor_config_digest"))
        candidates = tuple(item if isinstance(item, EvidenceCandidate) else EvidenceCandidate.from_dict(item) for item in self.candidates)
        if len(candidates) > MAX_CANDIDATES:
            raise EvidenceContractError("candidate bundle exceeds the bounded candidate count")
        if self.candidate_count is not None and self.candidate_count != len(candidates):
            raise EvidenceIntegrityError("candidate_count does not match candidates")
        object.__setattr__(self, "candidate_count", len(candidates))
        ids = tuple(item.candidate_id for item in candidates)
        if len(ids) != len(set(ids)) or ids != tuple(sorted(ids)):
            raise EvidenceContractError("candidates must have unique deterministic sorted IDs")
        for candidate in candidates:
            if candidate.canonical_document_artifact_id != self.canonical_document_artifact_id or candidate.source_artifact_id != self.source_artifact_id:
                raise EvidenceIntegrityError("candidate does not belong to the bundle source")
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "canonical_document_artifact_id": self.canonical_document_artifact_id,
            "source_artifact_id": self.source_artifact_id,
            "extractor_version": self.extractor_version,
            "extractor_config_digest": self.extractor_config_digest,
            "candidate_count": self.candidate_count,
            "candidates": [item.to_dict() for item in self.candidates],
        }

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def artifact_id(self) -> str:
        return artifact_id_for_sha256(sha256_bytes(self.canonical_bytes()))

    @property
    def digest(self) -> str:
        return sha256_bytes(self.canonical_bytes())

    def to_artifact_draft(self) -> ArtifactDraft:
        return ArtifactDraft(
            content=self.canonical_bytes(),
            media_type="application/json",
            schema_name=self.schema_name,
            schema_version=self.schema_version,
        )

    def candidate_by_id(self, candidate_id: str) -> EvidenceCandidate:
        validate_identifier(candidate_id, field="candidate_id")
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise EvidenceIntegrityError("candidate ID is not present in the bundle")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceCandidateBundle":
        if not isinstance(value, Mapping):
            raise EvidenceContractError("candidate bundle must be an object")
        allowed = {"schema_name", "schema_version", "canonical_document_artifact_id", "source_artifact_id", "extractor_version", "extractor_config_digest", "candidate_count", "candidates"}
        if set(value) - allowed:
            raise EvidenceContractError("candidate bundle has unknown fields")
        try:
            return cls(
                schema_name=str(value.get("schema_name", CANDIDATE_SCHEMA_NAME)),
                schema_version=str(value["schema_version"]),
                canonical_document_artifact_id=str(value["canonical_document_artifact_id"]),
                source_artifact_id=str(value["source_artifact_id"]),
                extractor_version=str(value["extractor_version"]),
                extractor_config_digest=str(value["extractor_config_digest"]),
                candidate_count=value.get("candidate_count"),
                candidates=tuple(EvidenceCandidate.from_dict(item) for item in value.get("candidates", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceContractError("candidate bundle is malformed") from exc


_ALIAS_NORMALIZE_RE = re.compile(r"[^a-z0-9%]+")


def _normalized_alias(value: str) -> str:
    return _ALIAS_NORMALIZE_RE.sub(" ", value.casefold()).strip()


def _hints_for_cells(cells: Sequence[Any], config: CandidateExtractorConfig) -> dict[str, Any]:
    aliases = thaw_json(config.aliases)
    by_alias: dict[str, str] = {}
    for field_name, field_aliases in aliases.items():
        for alias in field_aliases:
            by_alias[_normalized_alias(alias)] = field_name
    headers = {int(cell.column_index): _normalized_alias(cell.text) for cell in cells if getattr(cell, "is_header", False)}
    hints: dict[str, Any] = {}
    for cell in cells:
        header = headers.get(int(cell.column_index))
        if header is None:
            continue
        field_name = by_alias.get(header)
        if field_name is not None and cell.text.strip():
            hints[field_name] = cell.text.strip()
    # A property cell can itself be an alias even when a malformed source did
    # not mark its header row.  This remains a hint and never a scientific fact.
    for cell in cells:
        normalized = _normalized_alias(cell.text)
        field_name = by_alias.get(normalized)
        if field_name == "property_id":
            hints.setdefault(field_name, cell.text.strip())
    return hints


class EvidenceCandidateExtractor:
    """Extract bounded, source-located structural candidates from a document."""

    def __init__(self, config: CandidateExtractorConfig | None = None) -> None:
        self.config = config or CandidateExtractorConfig()

    @property
    def config_digest(self) -> str:
        return self.config.digest

    def _candidate(
        self,
        document: CanonicalDocument,
        *,
        candidate_type: CandidateType,
        locators: Sequence[SourceLocator],
        source_text: str,
        table_cells: Sequence[Mapping[str, Any]] = (),
        structural_context: Mapping[str, Any] | None = None,
        field_hints: Mapping[str, Any] | None = None,
    ) -> EvidenceCandidate:
        return EvidenceCandidate.create(
            canonical_document_artifact_id=document.artifact_id,
            source_artifact_id=document.source_artifact_id,
            candidate_type=candidate_type,
            source_locators=locators,
            source_text=source_text[: self.config.max_candidate_text_chars],
            table_cells=table_cells,
            structural_context=structural_context,
            field_hints=field_hints,
            extractor_version=self.config.extractor_version,
            extractor_config_digest=self.config.digest,
        )

    def extract(self, document: CanonicalDocument) -> EvidenceCandidateBundle:
        if not isinstance(document, CanonicalDocument):
            raise EvidenceContractError("candidate extraction requires a CanonicalDocument")
        candidates: list[EvidenceCandidate] = []
        for table in document.tables:
            cells = tuple(sorted(table.cells, key=lambda item: (item.row_index, item.column_index, item.locator.canonical_bytes())))
            rows = sorted({cell.row_index for cell in cells})
            headers = tuple(cell for cell in cells if cell.is_header)
            header_row = min((cell.row_index for cell in headers), default=(min(rows) if rows else 0))
            for row in rows:
                if row == header_row:
                    continue
                row_cells = tuple(cell for cell in cells if cell.row_index == row)
                if not row_cells:
                    continue
                hints = _hints_for_cells(headers + row_cells, self.config)
                # Header aliases are only meaningful when values were found in
                # the row.  A row with no recognizable structure is retained as
                # a cell group only when a cell explicitly contains an alias.
                explicit_alias = any(_normalized_alias(cell.text) in {
                    _normalized_alias(alias)
                    for values in thaw_json(self.config.aliases).values()
                    for alias in values
                } for cell in row_cells)
                if not hints and not explicit_alias:
                    continue
                locators = tuple(cell.locator for cell in row_cells)
                text = " | ".join(cell.text.strip() for cell in row_cells)
                candidates.append(self._candidate(
                    document,
                    candidate_type=CandidateType.TABLE_ROW if hints else CandidateType.TABLE_CELL_GROUP,
                    locators=locators,
                    source_text=text,
                    table_cells=tuple(cell.to_dict() for cell in row_cells),
                    structural_context={"table_id": table.table_id, "row_index": row, "header_row_index": header_row},
                    field_hints=hints,
                ))
            if table.caption:
                aliases_text = " ".join(alias for values in thaw_json(self.config.aliases).values() for alias in values)
                if any(alias.casefold() in table.caption.casefold() for alias in aliases_text.split(" ")):
                    candidates.append(self._candidate(
                        document,
                        candidate_type=CandidateType.CAPTION_EVIDENCE,
                        locators=(table.locator,),
                        source_text=table.caption,
                        structural_context={"table_id": table.table_id, "caption": True},
                    ))

        alias_terms = tuple(
            alias.casefold()
            for values in thaw_json(self.config.aliases).values()
            for alias in values
            if len(alias) >= 3
        )
        for block in document.blocks:
            if any(term in block.text.casefold() for term in alias_terms):
                candidates.append(self._candidate(
                    document,
                    candidate_type=CandidateType.CAPTION_EVIDENCE if block.kind == "CAPTION" else CandidateType.TEXT_EVIDENCE,
                    locators=(block.locator,),
                    source_text=block.text,
                    structural_context={"block_id": block.block_id, "kind": block.kind, "section_id": block.section_id},
                ))

        # Candidate order is part of the deterministic extraction contract.
        unique: dict[str, EvidenceCandidate] = {item.candidate_id: item for item in candidates}
        ordered = tuple(sorted(unique.values(), key=lambda item: item.candidate_id))[: self.config.max_candidates]
        return EvidenceCandidateBundle(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            canonical_document_artifact_id=document.artifact_id,
            source_artifact_id=document.source_artifact_id,
            extractor_version=self.config.extractor_version,
            extractor_config_digest=self.config.digest,
            candidates=ordered,
        )


def extract_from_artifact(
    canonical_document_artifact_id: str,
    reader: Callable[[str], bytes],
    *,
    extractor: EvidenceCandidateExtractor | None = None,
) -> EvidenceCandidateBundle:
    """Read and verify exactly one CanonicalDocument artifact before extraction."""

    validate_artifact_id(canonical_document_artifact_id)
    if not callable(reader):
        raise EvidenceContractError("canonical document reader must be callable")
    try:
        import json
        document = CanonicalDocument.from_dict(json.loads(reader(canonical_document_artifact_id).decode("utf-8")))
    except Exception as exc:
        raise EvidenceIntegrityError("canonical document artifact is not valid UTF-8 canonical JSON") from exc
    if document.artifact_id != canonical_document_artifact_id:
        raise EvidenceIntegrityError("canonical document bytes do not match declared artifact identity")
    return (extractor or EvidenceCandidateExtractor()).extract(document)


__all__ = [
    "CANDIDATE_SCHEMA_NAME",
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateExtractorConfig",
    "CandidateType",
    "EvidenceCandidate",
    "EvidenceCandidateBundle",
    "EvidenceCandidateExtractor",
    "extract_from_artifact",
]
