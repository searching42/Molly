from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    TypeAdapter,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryLocatorMatchStatus,
    OledSupplementaryReviewTable,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    OledSupplementaryScopedCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryCellDisposition,
    OledSupplementaryCellDispositionKind,
    OledSupplementaryDatasetExclusionDisposition,
    OledSupplementaryDatasetExclusionReason,
    OledSupplementaryKnownPropertyProposal,
    OledSupplementaryOntologyReviewDisposition,
    OledSupplementaryOntologyReviewReason,
    OledSupplementaryProposalComparisonContext,
    OledSupplementaryResponseProducer,
    OledSupplementaryScopedCandidateResponseArtifact,
    OledSupplementaryScopedCandidateResponseManifest,
    OledSupplementarySourceCheckDisposition,
    OledSupplementarySourceCheckReason,
    build_oled_supplementary_scoped_candidate_response_artifact,
    validate_oled_supplementary_safe_authored_text,
)


SUPPLEMENTARY_SEMANTIC_REVIEW_PACKET_VERSION = (
    "oled_supplementary_semantic_review_packet.v1"
)
SUPPLEMENTARY_SEMANTIC_DECISION_MANIFEST_VERSION = (
    "oled_supplementary_semantic_decision_manifest.v1"
)
SUPPLEMENTARY_SEMANTIC_ADJUDICATION_ARTIFACT_VERSION = (
    "oled_supplementary_semantic_adjudication.v1"
)

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_NUMERIC_LEXEME_RE = re.compile(
    r"[+\-−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-−]?\d+)?"
)
_CELL_DISPOSITION_ADAPTER = TypeAdapter(OledSupplementaryCellDisposition)


class OledSupplementarySemanticReviewPacketStatus(str, Enum):
    READY_FOR_HUMAN_REVIEW = "ready_for_human_semantic_review"


class OledSupplementarySemanticReviewItemKind(str, Enum):
    COLUMN_MAPPING_GROUP = "column_mapping_group"
    SCOPE_SEMANTIC_NOTE = "scope_semantic_note"


class OledSupplementarySemanticDecision(str, Enum):
    ACCEPT_KNOWN_MAPPING = "accept_known_mapping"
    CONFIRM_ONTOLOGY_REVIEW = "confirm_ontology_review"
    CONFIRM_SOURCE_CHECK = "confirm_source_check"
    ACCEPT_EXCLUSION = "accept_exclusion"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    REJECT_GROUP = "reject_group"
    RESOLVE_SEMANTIC_NOTE_AS_REPORTED = "resolve_semantic_note_as_reported"
    REJECT_SCOPE = "reject_scope"


class OledSupplementarySemanticAdjudicationStatus(str, Enum):
    READY_FOR_LATER_MATERIALIZATION_REVIEW = (
        "ready_for_later_materialization_review"
    )
    REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS = (
        "review_complete_with_unresolved_items"
    )
    REVIEW_COMPLETE_NO_ELIGIBLE_MAPPINGS = (
        "review_complete_no_eligible_mappings"
    )


class OledSupplementaryKnownMappingSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: Literal[OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    proposal_note: str = ""
    property_id: str
    property_label: str
    target_layer: OledCausalLayer
    reported_unit: str
    canonical_unit: str
    comparison_context: OledSupplementaryProposalComparisonContext | None = None


class OledSupplementaryOntologyReviewSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: Literal[OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    proposal_note: str = ""
    property_label: str
    proposed_target_layer: OledCausalLayer
    reported_unit: str = ""
    ontology_review_reason: OledSupplementaryOntologyReviewReason


class OledSupplementarySourceCheckSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: Literal[OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    proposal_note: str = ""
    source_check_reason: OledSupplementarySourceCheckReason


class OledSupplementaryExclusionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    disposition: Literal[OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    proposal_note: str = ""
    exclusion_reason: OledSupplementaryDatasetExclusionReason


OledSupplementaryDispositionSummary = Annotated[
    Union[
        OledSupplementaryKnownMappingSummary,
        OledSupplementaryOntologyReviewSummary,
        OledSupplementarySourceCheckSummary,
        OledSupplementaryExclusionSummary,
    ],
    Field(discriminator="disposition"),
]


class OledSupplementarySemanticReviewCell(BaseModel):
    """One exact PR-H cell binding retained inside a compact column group."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope_id: str
    table_id: str
    table_content_digest: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    cell_value: str
    reported_value_text: str
    reported_decimal_places: Annotated[StrictInt, Field(ge=0)] | None = None
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_name: str
    reported_subject_text: str
    source_cell_digest: str
    cell_disposition_digest: str

    @field_validator(
        "scope_id",
        "table_id",
        "column_name",
        "cell_value",
        "reported_value_text",
        "subject_column_name",
        "reported_subject_text",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        if info.field_name in {"scope_id", "table_id"}:
            return _validate_bound_id(value, field_name=str(info.field_name))
        if not isinstance(value, str) or not value:
            raise ValueError(f"{info.field_name} is required")
        return value

    @field_validator(
        "table_content_digest",
        "source_cell_digest",
        "cell_disposition_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_cell_integrity(self) -> OledSupplementarySemanticReviewCell:
        if self.reported_value_text != self.cell_value:
            raise ValueError("semantic review cell reported literal mismatch")
        if _stable_hash(_source_cell_payload(self)) != self.source_cell_digest:
            raise ValueError("semantic review source cell digest mismatch")
        return self


class OledSupplementaryColumnMappingReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: Literal[OledSupplementarySemanticReviewItemKind.COLUMN_MAPPING_GROUP]
    scope_id: str
    table_id: str
    table_content_digest: str
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_name: str
    linked_semantic_note_review_item_id: str = ""
    disposition_summary: OledSupplementaryDispositionSummary
    member_cell_count: Annotated[StrictInt, Field(ge=1)]
    member_cells: list[OledSupplementarySemanticReviewCell] = Field(default_factory=list)

    @field_validator("review_item_id", "scope_id", "table_id")
    @classmethod
    def validate_bound_text(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("subject_column_name")
    @classmethod
    def validate_subject_column_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("subject_column_name is required")
        return value

    @field_validator("linked_semantic_note_review_item_id")
    @classmethod
    def validate_optional_semantic_item_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        return _validate_bound_id(
            clean,
            field_name="linked_semantic_note_review_item_id",
        )

    @field_validator("review_item_digest", "table_content_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_group_integrity(self) -> OledSupplementaryColumnMappingReviewItem:
        if self.member_cell_count != len(self.member_cells) or not self.member_cells:
            raise ValueError("semantic review group member count mismatch")
        if (
            self.disposition_summary.column_index != self.column_index
            or self.disposition_summary.column_name != self.column_name
        ):
            raise ValueError("semantic review group summary column mismatch")
        observed_order = [
            (cell.row_index, cell.source_cell_digest) for cell in self.member_cells
        ]
        if observed_order != sorted(observed_order):
            raise ValueError("semantic review group cells must be sorted")
        source_digests = [cell.source_cell_digest for cell in self.member_cells]
        if len(source_digests) != len(set(source_digests)):
            raise ValueError("semantic review group contains duplicate source cells")
        for cell in self.member_cells:
            if (
                cell.scope_id != self.scope_id
                or cell.table_id != self.table_id
                or cell.table_content_digest != self.table_content_digest
                or cell.column_index != self.column_index
                or cell.column_name != self.column_name
                or cell.subject_column_index != self.subject_column_index
                or cell.subject_column_name != self.subject_column_name
            ):
                raise ValueError("semantic review group cell binding mismatch")
            reconstructed = _reconstruct_disposition_payload(
                self.disposition_summary,
                cell,
            )
            validated = _CELL_DISPOSITION_ADAPTER.validate_python(reconstructed)
            if (
                _stable_hash(validated.model_dump(mode="json"))
                != cell.cell_disposition_digest
            ):
                raise ValueError("semantic review cell disposition digest mismatch")
        if _review_item_digest(self) != self.review_item_digest:
            raise ValueError("semantic review group digest mismatch")
        if self.review_item_id != _review_item_id(_review_item_identity(self)):
            raise ValueError("semantic review group id mismatch")
        return self


class OledSupplementaryScopeSemanticNoteReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: Literal[OledSupplementarySemanticReviewItemKind.SCOPE_SEMANTIC_NOTE]
    scope_id: str
    table_id: str
    table_content_digest: str
    semantic_note: str
    affected_mapping_review_item_ids: list[str] = Field(default_factory=list)

    @field_validator("review_item_id", "scope_id", "table_id", "semantic_note")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        if info.field_name != "semantic_note":
            return _validate_bound_id(value, field_name=str(info.field_name))
        if not isinstance(value, str) or not value:
            raise ValueError(f"{info.field_name} is required")
        return value

    @field_validator("review_item_digest", "table_content_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("affected_mapping_review_item_ids")
    @classmethod
    def validate_affected_ids(cls, value: list[str]) -> list[str]:
        if not value or value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("semantic note affected item ids must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_item_integrity(self) -> OledSupplementaryScopeSemanticNoteReviewItem:
        if _review_item_digest(self) != self.review_item_digest:
            raise ValueError("semantic note review item digest mismatch")
        if self.review_item_id != _review_item_id(_review_item_identity(self)):
            raise ValueError("semantic note review item id mismatch")
        return self


OledSupplementarySemanticReviewItem = Annotated[
    Union[
        OledSupplementaryColumnMappingReviewItem,
        OledSupplementaryScopeSemanticNoteReviewItem,
    ],
    Field(discriminator="item_kind"),
]


class OledSupplementarySemanticReviewScope(BaseModel):
    """Full PR-G evidence context shown once, not repeated for every cell."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope_id: str
    review_item_id: str
    recovery_item_id: str
    source_review_item_digest: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    parser_backend: str
    target_locator: str
    canonical_locator: str
    match_status: OledSupplementaryLocatorMatchStatus
    matched_table: OledSupplementaryReviewTable
    parser_warning_codes: list[str] = Field(default_factory=list)
    semantic_note: str = ""
    semantic_review_required: StrictBool
    mapping_review_item_ids: list[str] = Field(default_factory=list)
    semantic_note_review_item_id: str = ""

    @field_validator("scope_id", "review_item_id", "recovery_item_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("source_review_item_digest", "source_pdf_sha256", "parsed_document_sha256")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("mapping_review_item_ids", "parser_warning_codes")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        clean = [
            _validate_bound_id(item, field_name=str(info.field_name)) for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @field_validator("semantic_note_review_item_id")
    @classmethod
    def validate_optional_note_item_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        return _validate_bound_id(clean, field_name="semantic_note_review_item_id")

    @model_validator(mode="after")
    def validate_scope_shape(self) -> OledSupplementarySemanticReviewScope:
        if not self.mapping_review_item_ids:
            raise ValueError("semantic review scope requires mapping review items")
        if self.semantic_review_required != bool(self.semantic_note):
            raise ValueError("semantic review scope note flag mismatch")
        if bool(self.semantic_note_review_item_id) != self.semantic_review_required:
            raise ValueError("semantic review scope note item binding mismatch")
        if self.match_status != OledSupplementaryLocatorMatchStatus.EXACT_MATCH:
            raise ValueError("semantic review supports exact table scopes only")
        headers = self.matched_table.headers
        if (
            not headers
            or any(not header.strip() for header in headers)
            or len(headers) != len(set(headers))
            or any(set(row) != set(headers) for row in self.matched_table.rows)
        ):
            raise ValueError("semantic review scope requires a rectangular headed table")
        return self


class OledSupplementarySemanticReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = SUPPLEMENTARY_SEMANTIC_REVIEW_PACKET_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_digest: str
    ontology_snapshot_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    response_artifact_sha256: str
    response_artifact_digest: str
    producer: OledSupplementaryResponseProducer
    status: OledSupplementarySemanticReviewPacketStatus
    scope_count: Annotated[StrictInt, Field(ge=1)]
    review_item_count: Annotated[StrictInt, Field(ge=1)]
    mapping_review_item_count: Annotated[StrictInt, Field(ge=1)]
    semantic_note_review_item_count: Annotated[StrictInt, Field(ge=0)]
    source_cell_count: Annotated[StrictInt, Field(ge=1)]
    known_property_cell_count: Annotated[StrictInt, Field(ge=0)]
    ontology_review_cell_count: Annotated[StrictInt, Field(ge=0)]
    source_check_cell_count: Annotated[StrictInt, Field(ge=0)]
    exclusion_cell_count: Annotated[StrictInt, Field(ge=0)]
    scopes: list[OledSupplementarySemanticReviewScope] = Field(default_factory=list)
    review_items: list[OledSupplementarySemanticReviewItem] = Field(default_factory=list)
    review_packet_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    complete_source_context_included: StrictBool = True
    strict_cell_partition_validated: StrictBool = True
    human_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    response_authorship_exact_bound: StrictBool = True
    human_semantic_review_completed: StrictBool = False
    table_transcription_validated: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    ontology_extensions_applied: StrictBool = False
    schema_candidates_created: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    device_only_admitted: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SEMANTIC_REVIEW_PACKET_VERSION:
            raise ValueError("unexpected supplementary semantic review packet version")
        return value

    @field_validator(
        "request_artifact_sha256",
        "request_digest",
        "ontology_snapshot_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_sha256",
        "response_artifact_digest",
        "review_packet_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @model_validator(mode="after")
    def validate_packet_integrity(self) -> OledSupplementarySemanticReviewPacket:
        if self.scope_count != len(self.scopes) or not self.scopes:
            raise ValueError("semantic review packet scope count mismatch")
        scope_ids = [scope.scope_id for scope in self.scopes]
        if scope_ids != sorted(scope_ids) or len(scope_ids) != len(set(scope_ids)):
            raise ValueError("semantic review packet scopes must be sorted and unique")
        item_ids = [item.review_item_id for item in self.review_items]
        if (
            self.review_item_count != len(self.review_items)
            or item_ids != sorted(item_ids)
            or len(item_ids) != len(set(item_ids))
        ):
            raise ValueError("semantic review packet item coverage mismatch")
        mapping_items = [
            item
            for item in self.review_items
            if isinstance(item, OledSupplementaryColumnMappingReviewItem)
        ]
        semantic_items = [
            item
            for item in self.review_items
            if isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem)
        ]
        if self.mapping_review_item_count != len(mapping_items):
            raise ValueError("semantic review mapping item count mismatch")
        if self.semantic_note_review_item_count != len(semantic_items):
            raise ValueError("semantic review note item count mismatch")
        scope_by_id = {scope.scope_id: scope for scope in self.scopes}
        if {item.scope_id for item in self.review_items} != set(scope_by_id):
            raise ValueError("semantic review items do not exactly cover packet scopes")
        item_by_id = {item.review_item_id: item for item in self.review_items}
        declared_item_ids: set[str] = set()
        for scope in self.scopes:
            expected_mapping_ids = sorted(
                item.review_item_id
                for item in mapping_items
                if item.scope_id == scope.scope_id
            )
            if scope.mapping_review_item_ids != expected_mapping_ids:
                raise ValueError("semantic review scope mapping item binding mismatch")
            declared_item_ids.update(scope.mapping_review_item_ids)
            expected_note_ids = [
                item.review_item_id
                for item in semantic_items
                if item.scope_id == scope.scope_id
            ]
            if len(expected_note_ids) > 1:
                raise ValueError("semantic review scope repeats its semantic note")
            expected_note_id = expected_note_ids[0] if expected_note_ids else ""
            if scope.semantic_note_review_item_id != expected_note_id:
                raise ValueError("semantic review scope note item coverage mismatch")
            if expected_note_id:
                declared_item_ids.add(expected_note_id)
                note_item = item_by_id[expected_note_id]
                assert isinstance(note_item, OledSupplementaryScopeSemanticNoteReviewItem)
                if (
                    note_item.semantic_note != scope.semantic_note
                    or note_item.table_id != scope.matched_table.table_id
                    or note_item.table_content_digest
                    != scope.matched_table.table_content_digest
                    or note_item.affected_mapping_review_item_ids != expected_mapping_ids
                ):
                    raise ValueError("semantic review note source binding mismatch")
            for mapping_id in expected_mapping_ids:
                mapping_item = item_by_id[mapping_id]
                assert isinstance(
                    mapping_item,
                    OledSupplementaryColumnMappingReviewItem,
                )
                if (
                    mapping_item.table_id != scope.matched_table.table_id
                    or mapping_item.table_content_digest
                    != scope.matched_table.table_content_digest
                    or mapping_item.linked_semantic_note_review_item_id
                    != expected_note_id
                ):
                    raise ValueError("semantic review mapping table binding mismatch")
                headers = scope.matched_table.headers
                if (
                    mapping_item.column_index >= len(headers)
                    or headers[mapping_item.column_index] != mapping_item.column_name
                    or mapping_item.subject_column_index != 0
                    or mapping_item.subject_column_index >= len(headers)
                    or headers[mapping_item.subject_column_index]
                    != mapping_item.subject_column_name
                ):
                    raise ValueError("semantic review mapping header binding mismatch")
                for cell in mapping_item.member_cells:
                    if cell.row_index >= len(scope.matched_table.rows):
                        raise ValueError("semantic review cell row binding mismatch")
                    source_row = scope.matched_table.rows[cell.row_index]
                    if (
                        source_row[mapping_item.column_name] != cell.cell_value
                        or source_row[mapping_item.subject_column_name]
                        != cell.reported_subject_text
                    ):
                        raise ValueError("semantic review cell table literal mismatch")
            expected_roster = {
                (row_index, column_index, column_name, row[column_name])
                for row_index, row in enumerate(scope.matched_table.rows)
                for column_index, column_name in enumerate(
                    scope.matched_table.headers[1:],
                    start=1,
                )
                if _NUMERIC_LEXEME_RE.search(row[column_name]) is not None
            }
            observed_roster = {
                (
                    cell.row_index,
                    cell.column_index,
                    cell.column_name,
                    cell.cell_value,
                )
                for mapping_id in expected_mapping_ids
                for cell in item_by_id[mapping_id].member_cells
                if isinstance(
                    item_by_id[mapping_id],
                    OledSupplementaryColumnMappingReviewItem,
                )
            }
            if observed_roster != expected_roster:
                raise ValueError("semantic review packet numeric-cell roster mismatch")
        if declared_item_ids != set(item_by_id):
            raise ValueError("semantic review scopes hide or repeat review items")
        all_cells = [cell for item in mapping_items for cell in item.member_cells]
        source_digests = [cell.source_cell_digest for cell in all_cells]
        disposition_digests = [cell.cell_disposition_digest for cell in all_cells]
        if (
            self.source_cell_count != len(all_cells)
            or len(source_digests) != len(set(source_digests))
            or len(disposition_digests) != len(set(disposition_digests))
        ):
            raise ValueError("semantic review packet cells are not a strict partition")
        disposition_counts = {
            kind: sum(
                len(item.member_cells)
                for item in mapping_items
                if item.disposition_summary.disposition == kind
            )
            for kind in OledSupplementaryCellDispositionKind
        }
        expected_counts = {
            OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY:
                self.known_property_cell_count,
            OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW:
                self.ontology_review_cell_count,
            OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK:
                self.source_check_cell_count,
            OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET:
                self.exclusion_cell_count,
        }
        if disposition_counts != expected_counts:
            raise ValueError("semantic review packet disposition counts mismatch")
        fixed_true = (
            "review_only",
            "offline_only",
            "complete_source_context_included",
            "strict_cell_partition_validated",
            "human_review_required",
            "source_pdf_remains_authoritative",
            "response_authorship_exact_bound",
        )
        fixed_false = (
            "human_semantic_review_completed",
            "table_transcription_validated",
            "scientific_content_validated",
            "physical_semantics_validated",
            "material_identity_resolved",
            "ontology_extensions_applied",
            "schema_candidates_created",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("semantic review packet lost a required boundary flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("semantic review packet crossed a downstream boundary")
        if _packet_digest(self) != self.review_packet_digest:
            raise ValueError("semantic review packet digest mismatch")
        return self


class OledSupplementarySemanticDecisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: OledSupplementarySemanticReviewItemKind
    decision: OledSupplementarySemanticDecision
    review_note: str = ""

    @field_validator("review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="review_item_id")

    @field_validator("review_item_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="review_item_digest")

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="review_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_note_requirement(self) -> OledSupplementarySemanticDecisionEntry:
        if self.decision in {
            OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK,
            OledSupplementarySemanticDecision.REJECT_GROUP,
            OledSupplementarySemanticDecision.RESOLVE_SEMANTIC_NOTE_AS_REPORTED,
            OledSupplementarySemanticDecision.REJECT_SCOPE,
        } and not self.review_note:
            raise ValueError("semantic review decision requires review_note")
        return self


class OledSupplementarySemanticDecisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = SUPPLEMENTARY_SEMANTIC_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    review_packet_sha256: str
    review_packet_digest: str
    reviewed_by: str
    reviewed_at: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledSupplementarySemanticDecisionEntry] = Field(default_factory=list)

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("schema_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SEMANTIC_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary semantic decision manifest version")
        return value

    @field_validator("review_packet_sha256", "review_packet_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="reviewed_by",
            required=True,
            max_length=200,
        )

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="reviewed_at")

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledSupplementarySemanticDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("semantic review adjudication requires confirmation")
        if not self.decisions:
            raise ValueError("semantic review adjudication requires decisions")
        ids = [item.review_item_id for item in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("semantic review decisions must be unique")
        return self


class OledSupplementaryAdjudicatedGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    scope_id: str
    table_id: str
    table_content_digest: str
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_name: str
    linked_semantic_note_review_item_id: str = ""
    disposition_summary: OledSupplementaryDispositionSummary
    member_cell_count: Annotated[StrictInt, Field(ge=1)]
    member_source_cell_digests: list[str] = Field(default_factory=list)
    decision: OledSupplementarySemanticDecision
    review_note: str = ""
    blocked_by_scope_semantics: StrictBool
    eligible_for_later_materialization_review: StrictBool
    ontology_review_pending: StrictBool
    source_check_pending: StrictBool
    exclusion_confirmed: StrictBool
    rejected: StrictBool
    rejected_by_scope: StrictBool

    @field_validator("review_item_id", "scope_id", "table_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("subject_column_name")
    @classmethod
    def validate_subject_column_name(cls, value: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("subject_column_name is required")
        return value

    @field_validator("linked_semantic_note_review_item_id")
    @classmethod
    def validate_optional_semantic_item_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        return _validate_bound_id(
            clean,
            field_name="linked_semantic_note_review_item_id",
        )

    @field_validator("review_item_digest", "table_content_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("member_source_cell_digests")
    @classmethod
    def validate_member_digests(cls, value: list[str]) -> list[str]:
        clean = [_normalize_sha256(item, field_name="source_cell_digest") for item in value]
        if not clean or clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("adjudicated group member digests must be sorted and unique")
        return clean

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="review_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_group_shape(self) -> OledSupplementaryAdjudicatedGroup:
        if self.member_cell_count != len(self.member_source_cell_digests):
            raise ValueError("adjudicated semantic group member count mismatch")
        kind = self.disposition_summary.disposition
        if self.decision not in _allowed_group_decisions(kind):
            raise ValueError("adjudicated semantic group decision is incompatible")
        if self.decision in {
            OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK,
            OledSupplementarySemanticDecision.REJECT_GROUP,
        } and not self.review_note:
            raise ValueError("adjudicated semantic group decision requires review_note")
        accepted_known = (
            kind == OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY
            and self.decision == OledSupplementarySemanticDecision.ACCEPT_KNOWN_MAPPING
        )
        if self.eligible_for_later_materialization_review != (
            accepted_known and not self.blocked_by_scope_semantics
        ):
            raise ValueError("adjudicated semantic group eligibility mismatch")
        expected_ontology = not self.rejected_by_scope and (
            kind == OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW
            and self.decision
            == OledSupplementarySemanticDecision.CONFIRM_ONTOLOGY_REVIEW
        )
        expected_source = not self.rejected_by_scope and (
            self.decision == OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK
            or (
                kind == OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK
                and self.decision
                == OledSupplementarySemanticDecision.CONFIRM_SOURCE_CHECK
            )
        )
        expected_exclusion = not self.rejected_by_scope and (
            kind == OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET
            and self.decision == OledSupplementarySemanticDecision.ACCEPT_EXCLUSION
        )
        if self.ontology_review_pending != expected_ontology:
            raise ValueError("adjudicated semantic group ontology flag mismatch")
        if self.source_check_pending != expected_source:
            raise ValueError("adjudicated semantic group source-check flag mismatch")
        if self.exclusion_confirmed != expected_exclusion:
            raise ValueError("adjudicated semantic group exclusion flag mismatch")
        if self.rejected != (
            self.decision == OledSupplementarySemanticDecision.REJECT_GROUP
            or self.rejected_by_scope
        ):
            raise ValueError("adjudicated semantic group rejection flag mismatch")
        return self


class OledSupplementaryAdjudicatedSemanticNote(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    scope_id: str
    table_id: str
    table_content_digest: str
    semantic_note: str
    affected_mapping_review_item_ids: list[str] = Field(default_factory=list)
    decision: OledSupplementarySemanticDecision
    review_note: str
    semantic_note_resolved: StrictBool
    source_check_pending: StrictBool
    scope_rejected: StrictBool

    @field_validator("review_item_id", "scope_id", "table_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("review_item_digest", "table_content_digest")
    @classmethod
    def validate_digest(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("affected_mapping_review_item_ids")
    @classmethod
    def validate_affected_ids(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_bound_id(item, field_name="affected_mapping_review_item_id")
            for item in value
        ]
        if not clean or clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("adjudicated semantic note affected ids must be sorted and unique")
        return clean

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="review_note",
            required=True,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_note_shape(self) -> OledSupplementaryAdjudicatedSemanticNote:
        if self.decision not in _allowed_semantic_note_decisions():
            raise ValueError("adjudicated semantic note decision is incompatible")
        if self.semantic_note_resolved != (
            self.decision
            == OledSupplementarySemanticDecision.RESOLVE_SEMANTIC_NOTE_AS_REPORTED
        ):
            raise ValueError("adjudicated semantic note resolution flag mismatch")
        if self.source_check_pending != (
            self.decision == OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK
        ):
            raise ValueError("adjudicated semantic note source-check flag mismatch")
        if self.scope_rejected != (
            self.decision == OledSupplementarySemanticDecision.REJECT_SCOPE
        ):
            raise ValueError("adjudicated semantic note rejection flag mismatch")
        return self


class OledSupplementaryAdjudicatedCell(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_cell: OledSupplementarySemanticReviewCell
    decision_source_review_item_id: str
    decision_source_review_item_digest: str
    disposition: OledSupplementaryCellDispositionKind
    decision: OledSupplementarySemanticDecision
    blocked_by_scope_semantics: StrictBool
    eligible_for_later_materialization_review: StrictBool
    ontology_review_pending: StrictBool
    source_check_pending: StrictBool
    exclusion_confirmed: StrictBool
    rejected: StrictBool
    rejected_by_scope: StrictBool

    @field_validator("decision_source_review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="decision_source_review_item_id")

    @field_validator("decision_source_review_item_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="decision_source_review_item_digest")


class OledSupplementarySemanticAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = SUPPLEMENTARY_SEMANTIC_ADJUDICATION_ARTIFACT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    response_artifact_sha256: str
    response_artifact_digest: str
    review_packet_sha256: str
    review_packet_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    reviewed_by: str
    reviewed_at: str
    status: OledSupplementarySemanticAdjudicationStatus
    review_item_count: Annotated[StrictInt, Field(ge=1)]
    group_count: Annotated[StrictInt, Field(ge=1)]
    semantic_note_count: Annotated[StrictInt, Field(ge=0)]
    cell_count: Annotated[StrictInt, Field(ge=1)]
    later_eligible_group_count: Annotated[StrictInt, Field(ge=0)]
    later_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    ontology_review_pending_group_count: Annotated[StrictInt, Field(ge=0)]
    ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    source_check_pending_group_count: Annotated[StrictInt, Field(ge=0)]
    source_check_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    exclusion_confirmed_group_count: Annotated[StrictInt, Field(ge=0)]
    exclusion_confirmed_cell_count: Annotated[StrictInt, Field(ge=0)]
    rejected_group_count: Annotated[StrictInt, Field(ge=0)]
    rejected_cell_count: Annotated[StrictInt, Field(ge=0)]
    unresolved_review_item_count: Annotated[StrictInt, Field(ge=0)]
    adjudicated_groups: list[OledSupplementaryAdjudicatedGroup] = Field(default_factory=list)
    adjudicated_semantic_notes: list[OledSupplementaryAdjudicatedSemanticNote] = Field(
        default_factory=list
    )
    adjudicated_cells: list[OledSupplementaryAdjudicatedCell] = Field(default_factory=list)
    adjudication_artifact_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    request_response_binding_revalidated: StrictBool = True
    review_packet_binding_revalidated: StrictBool = True
    decision_binding_validated: StrictBool = True
    strict_group_partition_validated: StrictBool = True
    cell_decision_coverage_validated: StrictBool = True
    human_decisions_recorded: StrictBool = True
    human_semantic_review_completed: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    table_transcription_validated: StrictBool = False
    table_exhaustiveness_validated: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    source_values_corrected: StrictBool = False
    ontology_extensions_applied: StrictBool = False
    schema_candidates_created: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    device_only_admitted: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SEMANTIC_ADJUDICATION_ARTIFACT_VERSION:
            raise ValueError("unexpected supplementary semantic adjudication version")
        return value

    @field_validator(
        "request_artifact_sha256",
        "request_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_sha256",
        "response_artifact_digest",
        "review_packet_sha256",
        "review_packet_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "adjudication_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("generated_at", "reviewed_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="reviewed_by",
            required=True,
            max_length=200,
        )

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledSupplementarySemanticAdjudicationArtifact:
        if _parse_timestamp(self.reviewed_at) > _parse_timestamp(self.generated_at):
            raise ValueError("semantic adjudication predates the human review")
        group_ids = [group.review_item_id for group in self.adjudicated_groups]
        note_ids = [item.review_item_id for item in self.adjudicated_semantic_notes]
        if (
            self.group_count != len(self.adjudicated_groups)
            or not self.adjudicated_groups
            or group_ids != sorted(group_ids)
            or len(group_ids) != len(set(group_ids))
        ):
            raise ValueError("semantic adjudication group coverage mismatch")
        if (
            self.semantic_note_count != len(self.adjudicated_semantic_notes)
            or note_ids != sorted(note_ids)
            or len(note_ids) != len(set(note_ids))
        ):
            raise ValueError("semantic adjudication note coverage mismatch")
        if self.review_item_count != self.group_count + self.semantic_note_count:
            raise ValueError("semantic adjudication review item count mismatch")
        if set(group_ids).intersection(note_ids):
            raise ValueError("semantic adjudication repeats a review item")
        note_by_scope: dict[str, OledSupplementaryAdjudicatedSemanticNote] = {}
        note_by_id: dict[str, OledSupplementaryAdjudicatedSemanticNote] = {}
        for item in self.adjudicated_semantic_notes:
            if item.scope_id in note_by_scope:
                raise ValueError("semantic adjudication repeats a scope semantic note")
            note_by_scope[item.scope_id] = item
            note_by_id[item.review_item_id] = item
        group_scope_ids = {group.scope_id for group in self.adjudicated_groups}
        if not set(note_by_scope).issubset(group_scope_ids):
            raise ValueError("semantic adjudication contains an orphan semantic note")
        referenced_note_ids: set[str] = set()
        for group in self.adjudicated_groups:
            note = note_by_scope.get(group.scope_id)
            expected_note_id = note.review_item_id if note is not None else ""
            if group.linked_semantic_note_review_item_id != expected_note_id:
                raise ValueError("semantic adjudication group semantic-note binding mismatch")
            if expected_note_id:
                referenced_note_ids.add(expected_note_id)
                assert note is not None
                if (
                    note.table_id != group.table_id
                    or note.table_content_digest != group.table_content_digest
                ):
                    raise ValueError("semantic adjudication note table binding mismatch")
            expected_blocked = bool(note is not None and not note.semantic_note_resolved)
            if group.blocked_by_scope_semantics != expected_blocked:
                raise ValueError("semantic adjudication scope semantic blocking mismatch")
            expected_scope_rejected = bool(note is not None and note.scope_rejected)
            if group.rejected_by_scope != expected_scope_rejected:
                raise ValueError("semantic adjudication scope rejection binding mismatch")
        if referenced_note_ids != set(note_by_id):
            raise ValueError("semantic adjudication semantic-note coverage mismatch")
        group_ids_by_scope: dict[str, list[str]] = {}
        for group in self.adjudicated_groups:
            group_ids_by_scope.setdefault(group.scope_id, []).append(group.review_item_id)
        for note in self.adjudicated_semantic_notes:
            expected_group_ids = sorted(group_ids_by_scope[note.scope_id])
            if note.affected_mapping_review_item_ids != expected_group_ids:
                raise ValueError("semantic adjudication note affected-group mismatch")
            OledSupplementaryScopeSemanticNoteReviewItem(
                review_item_id=note.review_item_id,
                review_item_digest=note.review_item_digest,
                item_kind=OledSupplementarySemanticReviewItemKind.SCOPE_SEMANTIC_NOTE,
                scope_id=note.scope_id,
                table_id=note.table_id,
                table_content_digest=note.table_content_digest,
                semantic_note=note.semantic_note,
                affected_mapping_review_item_ids=note.affected_mapping_review_item_ids,
            )
        observed_cell_order = [
            (
                cell.source_cell.scope_id,
                cell.source_cell.row_index,
                cell.source_cell.column_index,
                cell.source_cell.source_cell_digest,
            )
            for cell in self.adjudicated_cells
        ]
        if observed_cell_order != sorted(observed_cell_order):
            raise ValueError("semantic adjudication cells must be sorted")
        source_digests = [
            cell.source_cell.source_cell_digest for cell in self.adjudicated_cells
        ]
        if (
            self.cell_count != len(self.adjudicated_cells)
            or len(source_digests) != len(set(source_digests))
        ):
            raise ValueError("semantic adjudication cell coverage mismatch")
        cells_by_group: dict[str, list[OledSupplementaryAdjudicatedCell]] = {}
        for cell in self.adjudicated_cells:
            cells_by_group.setdefault(cell.decision_source_review_item_id, []).append(cell)
        group_by_id = {group.review_item_id: group for group in self.adjudicated_groups}
        if set(cells_by_group) != set(group_by_id):
            raise ValueError("semantic adjudication cells do not cover every group")
        for group_id, group in group_by_id.items():
            cells = cells_by_group[group_id]
            if sorted(cell.source_cell.source_cell_digest for cell in cells) != (
                group.member_source_cell_digests
            ):
                raise ValueError("semantic adjudication group member expansion mismatch")
            for cell in cells:
                if (
                    cell.decision_source_review_item_digest != group.review_item_digest
                    or cell.disposition != group.disposition_summary.disposition
                    or cell.decision != group.decision
                    or cell.blocked_by_scope_semantics
                    != group.blocked_by_scope_semantics
                    or cell.eligible_for_later_materialization_review
                    != group.eligible_for_later_materialization_review
                    or cell.ontology_review_pending != group.ontology_review_pending
                    or cell.source_check_pending != group.source_check_pending
                    or cell.exclusion_confirmed != group.exclusion_confirmed
                    or cell.rejected != group.rejected
                    or cell.rejected_by_scope != group.rejected_by_scope
                ):
                    raise ValueError("semantic adjudication expanded cell outcome mismatch")
                source = cell.source_cell
                if (
                    source.scope_id != group.scope_id
                    or source.table_id != group.table_id
                    or source.table_content_digest != group.table_content_digest
                    or source.column_index != group.column_index
                    or source.column_name != group.column_name
                    or source.subject_column_index != group.subject_column_index
                    or source.subject_column_name != group.subject_column_name
                ):
                    raise ValueError("semantic adjudication cell source binding mismatch")
                reconstructed = _CELL_DISPOSITION_ADAPTER.validate_python(
                    _reconstruct_disposition_payload(
                        group.disposition_summary,
                        source,
                    )
                )
                if (
                    _stable_hash(reconstructed.model_dump(mode="json"))
                    != source.cell_disposition_digest
                ):
                    raise ValueError("semantic adjudication cell disposition binding mismatch")
            OledSupplementaryColumnMappingReviewItem(
                review_item_id=group.review_item_id,
                review_item_digest=group.review_item_digest,
                item_kind=OledSupplementarySemanticReviewItemKind.COLUMN_MAPPING_GROUP,
                scope_id=group.scope_id,
                table_id=group.table_id,
                table_content_digest=group.table_content_digest,
                column_index=group.column_index,
                column_name=group.column_name,
                subject_column_index=group.subject_column_index,
                subject_column_name=group.subject_column_name,
                linked_semantic_note_review_item_id=(
                    group.linked_semantic_note_review_item_id
                ),
                disposition_summary=group.disposition_summary,
                member_cell_count=group.member_cell_count,
                member_cells=sorted(
                    (cell.source_cell for cell in cells),
                    key=lambda item: (item.row_index, item.source_cell_digest),
                ),
            )
        count_pairs = (
            (
                "later_eligible",
                self.later_eligible_group_count,
                self.later_eligible_cell_count,
                "eligible_for_later_materialization_review",
            ),
            (
                "ontology_review_pending",
                self.ontology_review_pending_group_count,
                self.ontology_review_pending_cell_count,
                "ontology_review_pending",
            ),
            (
                "source_check_pending",
                self.source_check_pending_group_count,
                self.source_check_pending_cell_count,
                "source_check_pending",
            ),
            (
                "exclusion_confirmed",
                self.exclusion_confirmed_group_count,
                self.exclusion_confirmed_cell_count,
                "exclusion_confirmed",
            ),
            (
                "rejected",
                self.rejected_group_count,
                self.rejected_cell_count,
                "rejected",
            ),
        )
        for label, group_count, cell_count, field_name in count_pairs:
            if group_count != sum(
                bool(getattr(group, field_name)) for group in self.adjudicated_groups
            ) or cell_count != sum(
                bool(getattr(cell, field_name)) for cell in self.adjudicated_cells
            ):
                raise ValueError(f"semantic adjudication {label} count mismatch")
        unresolved = sum(
            group.ontology_review_pending or group.source_check_pending
            for group in self.adjudicated_groups
        ) + sum(
            item.source_check_pending for item in self.adjudicated_semantic_notes
        )
        if self.unresolved_review_item_count != unresolved:
            raise ValueError("semantic adjudication unresolved item count mismatch")
        expected_status = _adjudication_status(
            unresolved_count=unresolved,
            later_eligible_group_count=self.later_eligible_group_count,
        )
        if self.status != expected_status:
            raise ValueError("semantic adjudication status mismatch")
        fixed_true = (
            "review_only",
            "offline_only",
            "request_response_binding_revalidated",
            "review_packet_binding_revalidated",
            "decision_binding_validated",
            "strict_group_partition_validated",
            "cell_decision_coverage_validated",
            "human_decisions_recorded",
            "human_semantic_review_completed",
            "source_pdf_remains_authoritative",
        )
        fixed_false = (
            "table_transcription_validated",
            "table_exhaustiveness_validated",
            "scientific_content_validated",
            "physical_semantics_validated",
            "material_identity_resolved",
            "source_values_corrected",
            "ontology_extensions_applied",
            "schema_candidates_created",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("semantic adjudication lost a required audit flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("semantic adjudication crossed a downstream boundary")
        if _adjudication_digest(self) != self.adjudication_artifact_digest:
            raise ValueError("semantic adjudication artifact digest mismatch")
        return self


def validate_oled_supplementary_semantic_review_inputs(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryScopedCandidateResponseArtifact,
) -> None:
    """Replay PR-H and compare the full expected artifact, not just audit booleans."""

    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    response = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    artifact = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    expected = build_oled_supplementary_scoped_candidate_response_artifact(
        request_artifact=request,
        request_artifact_sha256=request_artifact_sha256,
        response_manifest=response,
        response_manifest_sha256=response_manifest_sha256,
        generated_at=artifact.generated_at,
    )
    if expected.model_dump(mode="json") != artifact.model_dump(mode="json"):
        raise ValueError("supplementary semantic review PR-H artifact binding mismatch")


def build_oled_supplementary_semantic_review_packet(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryScopedCandidateResponseArtifact,
    response_artifact_sha256: str,
    generated_at: str,
) -> OledSupplementarySemanticReviewPacket:
    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    response = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    artifact = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    validate_oled_supplementary_semantic_review_inputs(
        request_artifact=request,
        request_artifact_sha256=request_artifact_sha256,
        response_manifest=response,
        response_manifest_sha256=response_manifest_sha256,
        response_artifact=artifact,
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(generated_at_clean) < _parse_timestamp(artifact.generated_at):
        raise ValueError("supplementary semantic review packet predates PR-H")
    request_scopes = {scope.scope_id: scope for scope in request.scopes}
    semantic_note_ids_by_scope = {
        scope.scope_id: _review_item_id(
            _semantic_note_item_identity(
                scope_id=scope.scope_id,
                table_id=scope.table_id,
                table_content_digest=scope.table_content_digest,
                semantic_note=scope.semantic_note,
            )
        )
        for scope in artifact.scope_results
        if scope.semantic_note
    }
    review_items: list[
        OledSupplementaryColumnMappingReviewItem
        | OledSupplementaryScopeSemanticNoteReviewItem
    ] = []
    mapping_items_by_scope: dict[str, list[OledSupplementaryColumnMappingReviewItem]] = {}
    for response_scope in artifact.scope_results:
        grouped: dict[str, list[OledSupplementaryCellDisposition]] = {}
        summaries: dict[str, OledSupplementaryDispositionSummary] = {}
        for disposition in response_scope.cell_dispositions:
            summary = _disposition_summary(disposition)
            signature = _stable_hash(summary.model_dump(mode="json"))
            grouped.setdefault(signature, []).append(disposition)
            summaries[signature] = summary
        scope_items: list[OledSupplementaryColumnMappingReviewItem] = []
        for signature in sorted(grouped):
            summary = summaries[signature]
            cells = sorted(
                (_review_cell(item) for item in grouped[signature]),
                key=lambda item: (item.row_index, item.source_cell_digest),
            )
            identity = {
                "item_kind": OledSupplementarySemanticReviewItemKind.COLUMN_MAPPING_GROUP.value,
                "scope_id": response_scope.scope_id,
                "table_id": response_scope.table_id,
                "table_content_digest": response_scope.table_content_digest,
                "column_index": summary.column_index,
                "column_name": summary.column_name,
                "subject_column_index": cells[0].subject_column_index,
                "subject_column_name": cells[0].subject_column_name,
                "linked_semantic_note_review_item_id": semantic_note_ids_by_scope.get(
                    response_scope.scope_id,
                    "",
                ),
                "disposition_summary": summary.model_dump(mode="json"),
                "member_source_cell_digests": sorted(
                    cell.source_cell_digest for cell in cells
                ),
            }
            payload: dict[str, Any] = {
                "review_item_id": _review_item_id(identity),
                "review_item_digest": "sha256:" + "0" * 64,
                "item_kind": OledSupplementarySemanticReviewItemKind.COLUMN_MAPPING_GROUP,
                "scope_id": response_scope.scope_id,
                "table_id": response_scope.table_id,
                "table_content_digest": response_scope.table_content_digest,
                "column_index": summary.column_index,
                "column_name": summary.column_name,
                "subject_column_index": cells[0].subject_column_index,
                "subject_column_name": cells[0].subject_column_name,
                "linked_semantic_note_review_item_id": semantic_note_ids_by_scope.get(
                    response_scope.scope_id,
                    "",
                ),
                "disposition_summary": summary.model_dump(mode="json"),
                "member_cell_count": len(cells),
                "member_cells": [cell.model_dump(mode="json") for cell in cells],
            }
            payload["review_item_digest"] = _stable_hash(
                {key: value for key, value in payload.items() if key != "review_item_digest"}
            )
            scope_items.append(
                OledSupplementaryColumnMappingReviewItem.model_validate(payload)
            )
        scope_items.sort(key=lambda item: item.review_item_id)
        mapping_items_by_scope[response_scope.scope_id] = scope_items
        review_items.extend(scope_items)
    for response_scope in artifact.scope_results:
        if not response_scope.semantic_note:
            continue
        mapping_ids = sorted(
            item.review_item_id for item in mapping_items_by_scope[response_scope.scope_id]
        )
        identity = _semantic_note_item_identity(
            scope_id=response_scope.scope_id,
            table_id=response_scope.table_id,
            table_content_digest=response_scope.table_content_digest,
            semantic_note=response_scope.semantic_note,
        )
        payload = {
            "review_item_id": semantic_note_ids_by_scope[response_scope.scope_id],
            "review_item_digest": "sha256:" + "0" * 64,
            **identity,
            "affected_mapping_review_item_ids": mapping_ids,
        }
        payload["review_item_digest"] = _stable_hash(
            {key: value for key, value in payload.items() if key != "review_item_digest"}
        )
        review_items.append(
            OledSupplementaryScopeSemanticNoteReviewItem.model_validate(payload)
        )
    review_items.sort(key=lambda item: item.review_item_id)
    semantic_items_by_scope = {
        item.scope_id: item
        for item in review_items
        if isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem)
    }
    scopes = []
    for scope_id in sorted(request_scopes):
        source = request_scopes[scope_id]
        scopes.append(
            OledSupplementarySemanticReviewScope(
                scope_id=source.scope_id,
                review_item_id=source.review_item_id,
                recovery_item_id=source.recovery_item_id,
                source_review_item_digest=source.source_review_item_digest,
                source_id=source.source_id,
                source_pdf_sha256=source.source_pdf_sha256,
                parsed_document_sha256=source.parsed_document_sha256,
                parser_backend=source.parser_backend,
                target_locator=source.target_locator,
                canonical_locator=source.canonical_locator,
                match_status=source.match_status,
                matched_table=source.matched_table,
                parser_warning_codes=source.parser_warning_codes,
                semantic_note=source.semantic_note,
                semantic_review_required=source.semantic_review_required,
                mapping_review_item_ids=sorted(
                    item.review_item_id for item in mapping_items_by_scope[scope_id]
                ),
                semantic_note_review_item_id=(
                    semantic_items_by_scope[scope_id].review_item_id
                    if scope_id in semantic_items_by_scope
                    else ""
                ),
            )
        )
    mapping_items = [
        item
        for item in review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
    ]
    counts = {
        kind: sum(
            len(item.member_cells)
            for item in mapping_items
            if item.disposition_summary.disposition == kind
        )
        for kind in OledSupplementaryCellDispositionKind
    }
    payload = {
        "artifact_version": SUPPLEMENTARY_SEMANTIC_REVIEW_PACKET_VERSION,
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_digest": request.request_digest,
        "ontology_snapshot_digest": request.ontology_snapshot_digest,
        "response_manifest_sha256": _normalize_sha256(
            response_manifest_sha256,
            field_name="response_manifest_sha256",
        ),
        "response_manifest_digest": artifact.response_manifest_digest,
        "response_artifact_sha256": _normalize_sha256(
            response_artifact_sha256,
            field_name="response_artifact_sha256",
        ),
        "response_artifact_digest": artifact.response_artifact_digest,
        "producer": response.producer.model_dump(mode="json"),
        "status": OledSupplementarySemanticReviewPacketStatus.READY_FOR_HUMAN_REVIEW,
        "scope_count": len(scopes),
        "review_item_count": len(review_items),
        "mapping_review_item_count": len(mapping_items),
        "semantic_note_review_item_count": len(review_items) - len(mapping_items),
        "source_cell_count": sum(len(item.member_cells) for item in mapping_items),
        "known_property_cell_count": counts[
            OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY
        ],
        "ontology_review_cell_count": counts[
            OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW
        ],
        "source_check_cell_count": counts[
            OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK
        ],
        "exclusion_cell_count": counts[
            OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET
        ],
        "scopes": [scope.model_dump(mode="json") for scope in scopes],
        "review_items": [item.model_dump(mode="json") for item in review_items],
        "review_packet_digest": "sha256:" + "0" * 64,
        "review_only": True,
        "offline_only": True,
        "complete_source_context_included": True,
        "strict_cell_partition_validated": True,
        "human_review_required": True,
        "source_pdf_remains_authoritative": True,
        "response_authorship_exact_bound": True,
        "human_semantic_review_completed": False,
        "table_transcription_validated": False,
        "scientific_content_validated": False,
        "physical_semantics_validated": False,
        "material_identity_resolved": False,
        "ontology_extensions_applied": False,
        "schema_candidates_created": False,
        "reviewed_evidence_staging": False,
        "direct_admission_eligible": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
    }
    payload["review_packet_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "review_packet_digest"}
    )
    return OledSupplementarySemanticReviewPacket.model_validate(payload)


def validate_oled_supplementary_semantic_decision_binding(
    *,
    review_packet: OledSupplementarySemanticReviewPacket,
    review_packet_sha256: str,
    decision_manifest: OledSupplementarySemanticDecisionManifest,
) -> None:
    packet = OledSupplementarySemanticReviewPacket.model_validate(
        review_packet.model_dump(mode="json")
    )
    decisions = OledSupplementarySemanticDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    if decisions.run_id != packet.run_id or decisions.paper_id != packet.paper_id:
        raise ValueError("semantic decision identity does not match review packet")
    if decisions.review_packet_sha256 != _normalize_sha256(
        review_packet_sha256,
        field_name="review_packet_sha256",
    ):
        raise ValueError("semantic decision does not bind exact review packet bytes")
    if decisions.review_packet_digest != packet.review_packet_digest:
        raise ValueError("semantic decision does not bind canonical review packet")
    if _parse_timestamp(decisions.reviewed_at) < _parse_timestamp(packet.generated_at):
        raise ValueError("semantic decision predates its review packet")
    item_by_id = {item.review_item_id: item for item in packet.review_items}
    decision_by_id = {item.review_item_id: item for item in decisions.decisions}
    if set(item_by_id) != set(decision_by_id):
        raise ValueError("semantic decisions must exactly cover review items")
    for item_id, item in item_by_id.items():
        decision = decision_by_id[item_id]
        if (
            decision.review_item_digest != item.review_item_digest
            or decision.item_kind != item.item_kind
        ):
            raise ValueError("semantic decision item binding mismatch")
        allowed = _allowed_decisions(item)
        if decision.decision not in allowed:
            raise ValueError("semantic decision is incompatible with review item kind")


def build_oled_supplementary_semantic_adjudication_artifact(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryScopedCandidateResponseArtifact,
    response_artifact_sha256: str,
    review_packet: OledSupplementarySemanticReviewPacket,
    review_packet_sha256: str,
    decision_manifest: OledSupplementarySemanticDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledSupplementarySemanticAdjudicationArtifact:
    packet = OledSupplementarySemanticReviewPacket.model_validate(
        review_packet.model_dump(mode="json")
    )
    expected_packet = build_oled_supplementary_semantic_review_packet(
        request_artifact=request_artifact,
        request_artifact_sha256=request_artifact_sha256,
        response_manifest=response_manifest,
        response_manifest_sha256=response_manifest_sha256,
        response_artifact=response_artifact,
        response_artifact_sha256=response_artifact_sha256,
        generated_at=packet.generated_at,
    )
    if expected_packet.model_dump(mode="json") != packet.model_dump(mode="json"):
        raise ValueError("semantic adjudication review packet binding mismatch")
    decisions = OledSupplementarySemanticDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    validate_oled_supplementary_semantic_decision_binding(
        review_packet=packet,
        review_packet_sha256=review_packet_sha256,
        decision_manifest=decisions,
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(decisions.reviewed_at) > _parse_timestamp(generated_at_clean):
        raise ValueError("semantic adjudication predates its decisions")
    decision_by_id = {item.review_item_id: item for item in decisions.decisions}
    semantic_results: list[OledSupplementaryAdjudicatedSemanticNote] = []
    semantic_resolved_by_scope = {
        scope.scope_id: not scope.semantic_review_required for scope in packet.scopes
    }
    scope_rejected_by_scope = {scope.scope_id: False for scope in packet.scopes}
    for item in packet.review_items:
        if not isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem):
            continue
        decision = decision_by_id[item.review_item_id]
        result = OledSupplementaryAdjudicatedSemanticNote(
            review_item_id=item.review_item_id,
            review_item_digest=item.review_item_digest,
            scope_id=item.scope_id,
            table_id=item.table_id,
            table_content_digest=item.table_content_digest,
            semantic_note=item.semantic_note,
            affected_mapping_review_item_ids=item.affected_mapping_review_item_ids,
            decision=decision.decision,
            review_note=decision.review_note,
            semantic_note_resolved=(
                decision.decision
                == OledSupplementarySemanticDecision.RESOLVE_SEMANTIC_NOTE_AS_REPORTED
            ),
            source_check_pending=(
                decision.decision == OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK
            ),
            scope_rejected=(
                decision.decision == OledSupplementarySemanticDecision.REJECT_SCOPE
            ),
        )
        semantic_resolved_by_scope[item.scope_id] = result.semantic_note_resolved
        scope_rejected_by_scope[item.scope_id] = result.scope_rejected
        semantic_results.append(result)
    semantic_results.sort(key=lambda item: item.review_item_id)
    groups: list[OledSupplementaryAdjudicatedGroup] = []
    cells: list[OledSupplementaryAdjudicatedCell] = []
    for item in packet.review_items:
        if not isinstance(item, OledSupplementaryColumnMappingReviewItem):
            continue
        decision = decision_by_id[item.review_item_id]
        kind = item.disposition_summary.disposition
        blocked = not semantic_resolved_by_scope[item.scope_id]
        rejected_by_scope = scope_rejected_by_scope[item.scope_id]
        eligible = (
            kind == OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY
            and decision.decision
            == OledSupplementarySemanticDecision.ACCEPT_KNOWN_MAPPING
            and not blocked
        )
        ontology_pending = not rejected_by_scope and (
            kind == OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW
            and decision.decision
            == OledSupplementarySemanticDecision.CONFIRM_ONTOLOGY_REVIEW
        )
        source_pending = not rejected_by_scope and (
            decision.decision == OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK
            or (
                kind == OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK
                and decision.decision
                == OledSupplementarySemanticDecision.CONFIRM_SOURCE_CHECK
            )
        )
        exclusion_confirmed = not rejected_by_scope and (
            kind == OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET
            and decision.decision == OledSupplementarySemanticDecision.ACCEPT_EXCLUSION
        )
        rejected = (
            decision.decision == OledSupplementarySemanticDecision.REJECT_GROUP
            or rejected_by_scope
        )
        group = OledSupplementaryAdjudicatedGroup(
            review_item_id=item.review_item_id,
            review_item_digest=item.review_item_digest,
            scope_id=item.scope_id,
            table_id=item.table_id,
            table_content_digest=item.table_content_digest,
            column_index=item.column_index,
            column_name=item.column_name,
            subject_column_index=item.subject_column_index,
            subject_column_name=item.subject_column_name,
            linked_semantic_note_review_item_id=(
                item.linked_semantic_note_review_item_id
            ),
            disposition_summary=item.disposition_summary,
            member_cell_count=item.member_cell_count,
            member_source_cell_digests=sorted(
                cell.source_cell_digest for cell in item.member_cells
            ),
            decision=decision.decision,
            review_note=decision.review_note,
            blocked_by_scope_semantics=blocked,
            eligible_for_later_materialization_review=eligible,
            ontology_review_pending=ontology_pending,
            source_check_pending=source_pending,
            exclusion_confirmed=exclusion_confirmed,
            rejected=rejected,
            rejected_by_scope=rejected_by_scope,
        )
        groups.append(group)
        for source_cell in item.member_cells:
            cells.append(
                OledSupplementaryAdjudicatedCell(
                    source_cell=source_cell,
                    decision_source_review_item_id=item.review_item_id,
                    decision_source_review_item_digest=item.review_item_digest,
                    disposition=kind,
                    decision=decision.decision,
                    blocked_by_scope_semantics=blocked,
                    eligible_for_later_materialization_review=eligible,
                    ontology_review_pending=ontology_pending,
                    source_check_pending=source_pending,
                    exclusion_confirmed=exclusion_confirmed,
                    rejected=rejected,
                    rejected_by_scope=rejected_by_scope,
                )
            )
    groups.sort(key=lambda item: item.review_item_id)
    cells.sort(
        key=lambda item: (
            item.source_cell.scope_id,
            item.source_cell.row_index,
            item.source_cell.column_index,
            item.source_cell.source_cell_digest,
        )
    )
    unresolved_count = sum(
        group.ontology_review_pending or group.source_check_pending for group in groups
    ) + sum(item.source_check_pending for item in semantic_results)
    later_group_count = sum(
        group.eligible_for_later_materialization_review for group in groups
    )
    payload = {
        "artifact_version": SUPPLEMENTARY_SEMANTIC_ADJUDICATION_ARTIFACT_VERSION,
        "run_id": packet.run_id,
        "paper_id": packet.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": packet.request_artifact_sha256,
        "request_digest": packet.request_digest,
        "response_manifest_sha256": packet.response_manifest_sha256,
        "response_manifest_digest": packet.response_manifest_digest,
        "response_artifact_sha256": packet.response_artifact_sha256,
        "response_artifact_digest": packet.response_artifact_digest,
        "review_packet_sha256": _normalize_sha256(
            review_packet_sha256,
            field_name="review_packet_sha256",
        ),
        "review_packet_digest": packet.review_packet_digest,
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": _stable_hash(
            _canonical_decision_manifest_payload(decisions)
        ),
        "reviewed_by": decisions.reviewed_by,
        "reviewed_at": decisions.reviewed_at,
        "status": _adjudication_status(
            unresolved_count=unresolved_count,
            later_eligible_group_count=later_group_count,
        ),
        "review_item_count": len(packet.review_items),
        "group_count": len(groups),
        "semantic_note_count": len(semantic_results),
        "cell_count": len(cells),
        "later_eligible_group_count": later_group_count,
        "later_eligible_cell_count": sum(
            cell.eligible_for_later_materialization_review for cell in cells
        ),
        "ontology_review_pending_group_count": sum(
            group.ontology_review_pending for group in groups
        ),
        "ontology_review_pending_cell_count": sum(
            cell.ontology_review_pending for cell in cells
        ),
        "source_check_pending_group_count": sum(
            group.source_check_pending for group in groups
        ),
        "source_check_pending_cell_count": sum(
            cell.source_check_pending for cell in cells
        ),
        "exclusion_confirmed_group_count": sum(
            group.exclusion_confirmed for group in groups
        ),
        "exclusion_confirmed_cell_count": sum(
            cell.exclusion_confirmed for cell in cells
        ),
        "rejected_group_count": sum(group.rejected for group in groups),
        "rejected_cell_count": sum(cell.rejected for cell in cells),
        "unresolved_review_item_count": unresolved_count,
        "adjudicated_groups": [group.model_dump(mode="json") for group in groups],
        "adjudicated_semantic_notes": [
            item.model_dump(mode="json") for item in semantic_results
        ],
        "adjudicated_cells": [cell.model_dump(mode="json") for cell in cells],
        "adjudication_artifact_digest": "sha256:" + "0" * 64,
        "review_only": True,
        "offline_only": True,
        "request_response_binding_revalidated": True,
        "review_packet_binding_revalidated": True,
        "decision_binding_validated": True,
        "strict_group_partition_validated": True,
        "cell_decision_coverage_validated": True,
        "human_decisions_recorded": True,
        "human_semantic_review_completed": True,
        "source_pdf_remains_authoritative": True,
        "table_transcription_validated": False,
        "table_exhaustiveness_validated": False,
        "scientific_content_validated": False,
        "physical_semantics_validated": False,
        "material_identity_resolved": False,
        "source_values_corrected": False,
        "ontology_extensions_applied": False,
        "schema_candidates_created": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "direct_admission_eligible": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
    }
    payload["adjudication_artifact_digest"] = _stable_hash(
        {
            key: value
            for key, value in payload.items()
            if key != "adjudication_artifact_digest"
        }
    )
    return OledSupplementarySemanticAdjudicationArtifact.model_validate(payload)


def render_oled_supplementary_semantic_review_markdown(
    packet: OledSupplementarySemanticReviewPacket,
    *,
    review_packet_sha256: str = "",
) -> str:
    review = OledSupplementarySemanticReviewPacket.model_validate(
        packet.model_dump(mode="json")
    )
    packet_file_sha256 = (
        _normalize_sha256(review_packet_sha256, field_name="review_packet_sha256")
        if str(review_packet_sha256 or "").strip()
        else ""
    )
    lines = [
        "# Supplementary semantic review packet",
        "",
        "> Read-only evidence view. Record decisions in the exact-bound JSON decision manifest.",
        "> Accepting a mapping does not validate transcription, identity, scientific truth, or create gold data.",
        "",
        "## Boundary summary",
        "",
        f"- Paper: {_md_code(review.paper_id)}",
        f"- Run ID: {_md_code(review.run_id)}",
        f"- Scopes: {review.scope_count}",
        f"- Source cells: {review.source_cell_count}",
        f"- Mapping groups: {review.mapping_review_item_count}",
        f"- Semantic-note items: {review.semantic_note_review_item_count}",
        f"- Request SHA-256: {_md_code(review.request_artifact_sha256)}",
        f"- Response manifest SHA-256: {_md_code(review.response_manifest_sha256)}",
        f"- PR-H artifact SHA-256: {_md_code(review.response_artifact_sha256)}",
        f"- Packet digest: {_md_code(review.review_packet_digest)}",
    ]
    if packet_file_sha256:
        lines.append(f"- Exact packet-file SHA-256: {_md_code(packet_file_sha256)}")
    lines.append("")
    mapping_by_id = {
        item.review_item_id: item
        for item in review.review_items
        if isinstance(item, OledSupplementaryColumnMappingReviewItem)
    }
    semantic_by_scope = {
        item.scope_id: item
        for item in review.review_items
        if isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem)
    }
    for scope_number, scope in enumerate(review.scopes, start=1):
        table = scope.matched_table
        lines.extend(
            [
                f"## Scope {scope_number}: {_md_code(scope.canonical_locator)}",
                "",
                f"- Source: {_md_code(scope.source_id)}",
                f"- Page: {table.page}",
                f"- Table ID: {_md_code(table.table_id)}",
                f"- Caption: {_md_code(table.caption)}",
                f"- Parser backend: {_md_code(scope.parser_backend)}",
                f"- Parser warnings: {_md_code(', '.join(scope.parser_warning_codes) or 'none')}",
                "",
                "### Bound table",
                "",
                "| " + " | ".join(_md_code(header) for header in table.headers) + " |",
                "| " + " | ".join("---" for _ in table.headers) + " |",
            ]
        )
        for row in table.rows:
            lines.append(
                "| "
                + " | ".join(_md_code(row.get(header, "")) for header in table.headers)
                + " |"
            )
        lines.extend(["", "Footnotes:", ""])
        if table.footnotes:
            lines.extend(f"- {_md_code(note)}" for note in table.footnotes)
        else:
            lines.append("- none")
        note_item = semantic_by_scope.get(scope.scope_id)
        if note_item is not None:
            lines.extend(
                [
                    "",
                    "### Mandatory scope semantic note",
                    "",
                    f"- Review item: {_md_code(note_item.review_item_id)}",
                    f"- Review-item digest: {_md_code(note_item.review_item_digest)}",
                    f"- Item kind: {_md_code(note_item.item_kind.value)}",
                    f"- Note: {_md_code(note_item.semantic_note)}",
                    "- Allowed decisions: `resolve_semantic_note_as_reported`, `needs_source_check`, `reject_scope`",
                    "- A resolution note is required; until resolved, every known mapping in this scope is blocked.",
                ]
            )
        lines.extend(["", "### Column mapping groups", ""])
        lines.extend(
            [
                "Row indexes are zero-based. `dp` means decimal places in the exact reported scalar; it preserves distinctions such as `0.030` versus `0.03`.",
                "Short cell references are display aids only; the JSON packet retains every full source-cell and disposition digest.",
                "",
            ]
        )
        scope_groups = sorted(
            (mapping_by_id[item_id] for item_id in scope.mapping_review_item_ids),
            key=lambda item: (item.column_index, item.review_item_id),
        )
        for group_number, item in enumerate(scope_groups, start=1):
            summary = item.disposition_summary
            lines.extend(
                [
                    f"#### G{group_number:02d}: {_md_code(item.column_name)}",
                    "",
                    f"- Review item: {_md_code(item.review_item_id)}",
                    f"- Review-item digest: {_md_code(item.review_item_digest)}",
                    f"- Item kind: {_md_code(item.item_kind.value)}",
                    f"- Disposition: {_md_code(summary.disposition.value)}",
                    f"- Cells: {item.member_cell_count}",
                    f"- Subject column: {_md_code(item.subject_column_name)} (0-based index {item.subject_column_index})",
                    f"- Allowed decisions: {_md_code(', '.join(value.value for value in sorted(_allowed_decisions(item), key=lambda value: value.value)))}",
                    "- Exact disposition details:",
                    *(
                        f"  - {label}: {_md_code(value)}"
                        for label, value in _summary_details(summary)
                    ),
                    "",
                    "| Row (0-based) | Subject | Value as reported | dp | Cell ref |",
                    "| ---: | --- | --- | ---: | --- |",
                ]
            )
            for cell in item.member_cells:
                dp = "—" if cell.reported_decimal_places is None else str(
                    cell.reported_decimal_places
                )
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(cell.row_index),
                            _md_code(cell.reported_subject_text),
                            _md_code(cell.reported_value_text),
                            dp,
                            _md_code(cell.source_cell_digest[7:19]),
                        ]
                    )
                    + " |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _disposition_summary(
    disposition: OledSupplementaryCellDisposition,
) -> OledSupplementaryDispositionSummary:
    payload = disposition.model_dump(mode="json")
    common = {
        "disposition": payload["disposition"],
        "column_index": payload["column_index"],
        "column_name": payload["column_name"],
        "proposal_note": payload["proposal_note"],
    }
    if isinstance(disposition, OledSupplementaryKnownPropertyProposal):
        return OledSupplementaryKnownMappingSummary(
            **common,
            property_id=disposition.property_id,
            property_label=disposition.property_label,
            target_layer=disposition.target_layer,
            reported_unit=disposition.reported_unit,
            canonical_unit=disposition.canonical_unit,
            comparison_context=disposition.comparison_context,
        )
    if isinstance(disposition, OledSupplementaryOntologyReviewDisposition):
        return OledSupplementaryOntologyReviewSummary(
            **common,
            property_label=disposition.property_label,
            proposed_target_layer=disposition.proposed_target_layer,
            reported_unit=disposition.reported_unit,
            ontology_review_reason=disposition.ontology_review_reason,
        )
    if isinstance(disposition, OledSupplementarySourceCheckDisposition):
        return OledSupplementarySourceCheckSummary(
            **common,
            source_check_reason=disposition.source_check_reason,
        )
    if isinstance(disposition, OledSupplementaryDatasetExclusionDisposition):
        return OledSupplementaryExclusionSummary(
            **common,
            exclusion_reason=disposition.exclusion_reason,
        )
    raise TypeError("unsupported supplementary cell disposition")


def _review_cell(
    disposition: OledSupplementaryCellDisposition,
) -> OledSupplementarySemanticReviewCell:
    payload = disposition.model_dump(mode="json")
    source_payload = {
        key: payload[key]
        for key in (
            "scope_id",
            "table_id",
            "table_content_digest",
            "row_index",
            "column_index",
            "column_name",
            "cell_value",
            "reported_value_text",
            "reported_decimal_places",
            "subject_column_index",
            "subject_column_name",
            "reported_subject_text",
        )
    }
    return OledSupplementarySemanticReviewCell(
        **source_payload,
        source_cell_digest=_stable_hash(source_payload),
        cell_disposition_digest=_stable_hash(payload),
    )


def _source_cell_payload(cell: OledSupplementarySemanticReviewCell) -> dict[str, Any]:
    return cell.model_dump(
        mode="json",
        exclude={"source_cell_digest", "cell_disposition_digest"},
    )


def _reconstruct_disposition_payload(
    summary: OledSupplementaryDispositionSummary,
    cell: OledSupplementarySemanticReviewCell,
) -> dict[str, Any]:
    payload = _source_cell_payload(cell)
    payload.update(summary.model_dump(mode="json"))
    return payload


def _review_item_identity(
    item: OledSupplementaryColumnMappingReviewItem
    | OledSupplementaryScopeSemanticNoteReviewItem,
) -> dict[str, Any]:
    if isinstance(item, OledSupplementaryColumnMappingReviewItem):
        return {
            "item_kind": item.item_kind.value,
            "scope_id": item.scope_id,
            "table_id": item.table_id,
            "table_content_digest": item.table_content_digest,
            "column_index": item.column_index,
            "column_name": item.column_name,
            "subject_column_index": item.subject_column_index,
            "subject_column_name": item.subject_column_name,
            "linked_semantic_note_review_item_id": (
                item.linked_semantic_note_review_item_id
            ),
            "disposition_summary": item.disposition_summary.model_dump(mode="json"),
            "member_source_cell_digests": sorted(
                cell.source_cell_digest for cell in item.member_cells
            ),
        }
    return _semantic_note_item_identity(
        scope_id=item.scope_id,
        table_id=item.table_id,
        table_content_digest=item.table_content_digest,
        semantic_note=item.semantic_note,
    )


def _semantic_note_item_identity(
    *,
    scope_id: str,
    table_id: str,
    table_content_digest: str,
    semantic_note: str,
) -> dict[str, Any]:
    return {
        "item_kind": OledSupplementarySemanticReviewItemKind.SCOPE_SEMANTIC_NOTE.value,
        "scope_id": scope_id,
        "table_id": table_id,
        "table_content_digest": table_content_digest,
        "semantic_note": semantic_note,
    }


def _review_item_id(identity: dict[str, Any]) -> str:
    return f"supplementary-semantic-review:{_stable_hash(identity)[7:31]}"


def _review_item_digest(
    item: OledSupplementaryColumnMappingReviewItem
    | OledSupplementaryScopeSemanticNoteReviewItem,
) -> str:
    return _stable_hash(item.model_dump(mode="json", exclude={"review_item_digest"}))


def _packet_digest(packet: OledSupplementarySemanticReviewPacket) -> str:
    return _stable_hash(packet.model_dump(mode="json", exclude={"review_packet_digest"}))


def _adjudication_digest(
    artifact: OledSupplementarySemanticAdjudicationArtifact,
) -> str:
    return _stable_hash(
        artifact.model_dump(mode="json", exclude={"adjudication_artifact_digest"})
    )


def _canonical_decision_manifest_payload(
    manifest: OledSupplementarySemanticDecisionManifest,
) -> dict[str, Any]:
    payload = manifest.model_dump(mode="json")
    payload["decisions"] = sorted(
        payload["decisions"], key=lambda item: item["review_item_id"]
    )
    return payload


def _allowed_decisions(
    item: OledSupplementaryColumnMappingReviewItem
    | OledSupplementaryScopeSemanticNoteReviewItem,
) -> set[OledSupplementarySemanticDecision]:
    if isinstance(item, OledSupplementaryScopeSemanticNoteReviewItem):
        return _allowed_semantic_note_decisions()
    return _allowed_group_decisions(item.disposition_summary.disposition)


def _allowed_semantic_note_decisions() -> set[OledSupplementarySemanticDecision]:
    return {
        OledSupplementarySemanticDecision.RESOLVE_SEMANTIC_NOTE_AS_REPORTED,
        OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK,
        OledSupplementarySemanticDecision.REJECT_SCOPE,
    }


def _allowed_group_decisions(
    kind: OledSupplementaryCellDispositionKind,
) -> set[OledSupplementarySemanticDecision]:
    shared = {
        OledSupplementarySemanticDecision.NEEDS_SOURCE_CHECK,
        OledSupplementarySemanticDecision.REJECT_GROUP,
    }
    if kind == OledSupplementaryCellDispositionKind.PROPOSE_KNOWN_PROPERTY:
        return shared | {OledSupplementarySemanticDecision.ACCEPT_KNOWN_MAPPING}
    if kind == OledSupplementaryCellDispositionKind.NEEDS_ONTOLOGY_REVIEW:
        return shared | {OledSupplementarySemanticDecision.CONFIRM_ONTOLOGY_REVIEW}
    if kind == OledSupplementaryCellDispositionKind.NEEDS_SOURCE_CHECK:
        return {
            OledSupplementarySemanticDecision.CONFIRM_SOURCE_CHECK,
            OledSupplementarySemanticDecision.REJECT_GROUP,
        }
    if kind == OledSupplementaryCellDispositionKind.EXCLUDE_FROM_DATASET:
        return shared | {OledSupplementarySemanticDecision.ACCEPT_EXCLUSION}
    raise ValueError("unsupported semantic review disposition")


def _adjudication_status(
    *,
    unresolved_count: int,
    later_eligible_group_count: int,
) -> OledSupplementarySemanticAdjudicationStatus:
    if unresolved_count:
        return (
            OledSupplementarySemanticAdjudicationStatus.REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS
        )
    if later_eligible_group_count:
        return (
            OledSupplementarySemanticAdjudicationStatus.READY_FOR_LATER_MATERIALIZATION_REVIEW
        )
    return OledSupplementarySemanticAdjudicationStatus.REVIEW_COMPLETE_NO_ELIGIBLE_MAPPINGS


def _summary_details(
    summary: OledSupplementaryDispositionSummary,
) -> list[tuple[str, str]]:
    proposal_note = summary.proposal_note or "none"
    if isinstance(summary, OledSupplementaryKnownMappingSummary):
        comparison_context = (
            "null"
            if summary.comparison_context is None
            else json.dumps(
                summary.comparison_context.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return [
            ("reported property label", summary.property_label),
            ("property ID", summary.property_id),
            ("target layer", summary.target_layer.value),
            ("reported unit", summary.reported_unit),
            ("canonical unit", summary.canonical_unit),
            ("comparison context", comparison_context),
            ("proposal note", proposal_note),
        ]
    if isinstance(summary, OledSupplementaryOntologyReviewSummary):
        return [
            ("reported property label", summary.property_label),
            ("proposed target layer", summary.proposed_target_layer.value),
            ("reported unit", summary.reported_unit or "unitless"),
            ("ontology-review reason", summary.ontology_review_reason.value),
            ("proposal note", proposal_note),
        ]
    if isinstance(summary, OledSupplementarySourceCheckSummary):
        return [
            ("source-check reason", summary.source_check_reason.value),
            ("proposal note", proposal_note),
        ]
    return [
        ("exclusion reason", summary.exclusion_reason.value),
        ("proposal note", proposal_note),
    ]


def _validate_reviewer_text(
    value: Any,
    *,
    field_name: str,
    required: bool,
    max_length: int,
) -> str:
    clean = validate_oled_supplementary_safe_authored_text(
        value,
        field_name=field_name,
        required=required,
        max_length=max_length,
    )
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in clean
    ):
        raise ValueError(f"{field_name} contains unsafe display-control characters")
    return clean


def _validate_path_segment(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(clean) or clean in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path segment")
    return clean


def _validate_bound_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_BOUND_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if clean.startswith("sha256:"):
        digest = clean[7:]
    else:
        digest = clean
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{digest.lower()}"


def _validate_timestamp(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is required")
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return clean


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _md_code(value: Any) -> str:
    clean = str(value).replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ⏎ ")
    clean = "".join(
        f"\\u{ord(character):04X}"
        if ord(character) < 32
        or ord(character) == 127
        or unicodedata.category(character) == "Cf"
        or character in {"\u2028", "\u2029"}
        else character
        for character in clean
    )
    clean = html.escape(clean, quote=False).replace("|", "&#124;")
    return f"<code>{clean}</code>"


__all__ = [
    "SUPPLEMENTARY_SEMANTIC_ADJUDICATION_ARTIFACT_VERSION",
    "SUPPLEMENTARY_SEMANTIC_DECISION_MANIFEST_VERSION",
    "SUPPLEMENTARY_SEMANTIC_REVIEW_PACKET_VERSION",
    "OledSupplementaryAdjudicatedCell",
    "OledSupplementaryAdjudicatedGroup",
    "OledSupplementaryAdjudicatedSemanticNote",
    "OledSupplementaryColumnMappingReviewItem",
    "OledSupplementarySemanticAdjudicationArtifact",
    "OledSupplementarySemanticAdjudicationStatus",
    "OledSupplementarySemanticDecision",
    "OledSupplementarySemanticDecisionEntry",
    "OledSupplementarySemanticDecisionManifest",
    "OledSupplementarySemanticReviewCell",
    "OledSupplementarySemanticReviewItemKind",
    "OledSupplementarySemanticReviewPacket",
    "OledSupplementarySemanticReviewPacketStatus",
    "OledSupplementaryScopeSemanticNoteReviewItem",
    "build_oled_supplementary_semantic_adjudication_artifact",
    "build_oled_supplementary_semantic_review_packet",
    "render_oled_supplementary_semantic_review_markdown",
    "validate_oled_supplementary_semantic_decision_binding",
    "validate_oled_supplementary_semantic_review_inputs",
]
