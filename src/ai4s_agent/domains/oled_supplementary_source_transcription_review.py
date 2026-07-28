from __future__ import annotations

import html
import json
from enum import Enum
from typing import Annotated, Any, Literal, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryReviewTable,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    OledSupplementaryScopedCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryScopedCandidateResponseArtifact,
    OledSupplementaryScopedCandidateResponseManifest,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementarySemanticAdjudicationArtifact,
    OledSupplementarySemanticDecisionManifest,
    OledSupplementarySemanticReviewPacket,
    _md_code,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_reviewer_text,
    _validate_timestamp,
    build_oled_supplementary_semantic_adjudication_artifact,
    build_oled_supplementary_semantic_review_packet,
)


SUPPLEMENTARY_SOURCE_TRANSCRIPTION_REVIEW_PACKET_VERSION = (
    "oled_supplementary_source_transcription_review_packet.v1"
)
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_DECISION_MANIFEST_VERSION = (
    "oled_supplementary_source_transcription_decision_manifest.v1"
)
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_ADJUDICATION_ARTIFACT_VERSION = (
    "oled_supplementary_source_transcription_adjudication.v1"
)
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION = (
    "oled_supplementary_source_transcription_visual_equivalence.v1"
)
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDERER_ID = "poppler-pdftoppm"
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDER_PROFILE = (
    "png-200dpi-rgb-full-page-v1"
)
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_POPPLER_RUNTIME_TRUST_MODEL = (
    "operator-trusted-dynamic-poppler-runtime.v1"
)
SUPPLEMENTARY_SOURCE_TRANSCRIPTION_PAGE_COUNTER_ID = "poppler-pdfinfo"

SOURCE_TRANSCRIPTION_COMPONENT_NAMES = (
    "page_anchor",
    "caption",
    "headers",
    "row_structure",
    "cell_literals",
    "footnotes",
    "table_extent",
)
SOURCE_TRANSCRIPTION_COMPONENT_RESULT_FIELDS = tuple(
    f"{component_name}_check"
    for component_name in SOURCE_TRANSCRIPTION_COMPONENT_NAMES
)
SOURCE_TRANSCRIPTION_COMPONENT_RESULT_TO_COMPONENT = {
    result_field: component_name
    for result_field, component_name in zip(
        SOURCE_TRANSCRIPTION_COMPONENT_RESULT_FIELDS,
        SOURCE_TRANSCRIPTION_COMPONENT_NAMES,
        strict=True,
    )
}


class OledSupplementarySourceTranscriptionReviewItemKind(str, Enum):
    TABLE_TRANSCRIPTION_SCOPE = "table_transcription_scope"


class OledSupplementarySourceHeaderBindingKind(str, Enum):
    REPORTED_LITERAL = "reported_literal"
    PARSER_PLACEHOLDER_CANDIDATE_FOR_BLANK_HEADER = (
        "parser_placeholder_candidate_for_blank_header"
    )


class OledSupplementarySourceTranscriptionComponentResult(str, Enum):
    VERIFIED_EQUIVALENT = "verified_equivalent"
    MISMATCH = "mismatch"
    NOT_CHECKED = "not_checked"


class OledSupplementarySourceTranscriptionDecision(str, Enum):
    ACCEPT_BOUNDED_SOURCE_TRANSCRIPTION = "accept_bounded_source_transcription"
    NEEDS_REPARSE = "needs_reparse"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    REJECT_SCOPE = "reject_scope"


class OledSupplementarySourceTranscriptionReviewPacketStatus(str, Enum):
    READY_FOR_HUMAN_SOURCE_TRANSCRIPTION_REVIEW = (
        "ready_for_human_source_transcription_review"
    )


class OledSupplementarySourceTranscriptionAdjudicationStatus(str, Enum):
    READY_FOR_LATER_IDENTITY_REVIEW = "ready_for_later_identity_review"
    REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS = (
        "review_complete_with_unresolved_items"
    )
    REVIEW_COMPLETE_NO_ELIGIBLE_SCOPES = (
        "review_complete_no_eligible_scopes"
    )


class OledSupplementarySourceHeaderReviewBinding(BaseModel):
    """A parser column key kept separate from its source-visible header candidate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    column_index: Annotated[StrictInt, Field(ge=0)]
    parser_key: str
    source_visible_header_candidate: str
    binding_kind: OledSupplementarySourceHeaderBindingKind

    @field_validator("parser_key", "source_visible_header_candidate")
    @classmethod
    def validate_header_text(cls, value: str, info: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{info.field_name} must be text")
        if len(value) > 2_000 or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"{info.field_name} contains unsafe text")
        if info.field_name == "parser_key" and not value:
            raise ValueError("parser_key is required")
        return value

    @model_validator(mode="after")
    def validate_binding_shape(self) -> OledSupplementarySourceHeaderReviewBinding:
        positional_placeholder = f"column_{self.column_index + 1}"
        if (
            self.binding_kind
            == OledSupplementarySourceHeaderBindingKind.PARSER_PLACEHOLDER_CANDIDATE_FOR_BLANK_HEADER
        ):
            if (
                self.parser_key != positional_placeholder
                or self.source_visible_header_candidate != ""
            ):
                raise ValueError(
                    "blank-header candidate requires the exact positional parser "
                    "placeholder and an empty source-visible candidate"
                )
        elif (
            not self.source_visible_header_candidate
            or self.source_visible_header_candidate != self.parser_key
        ):
            raise ValueError(
                "reported-literal header candidate must equal the parser key verbatim"
            )
        return self


class OledSupplementarySourcePageAsset(BaseModel):
    """One bounded full-page rendering produced from the exact source PDF."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    asset_id: str
    asset_filename: str
    source_id: str
    source_pdf_sha256: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    renderer_id: str
    renderer_version: str
    render_profile: str
    mime_type: Literal["image/png"] = "image/png"
    rendered_asset_sha256: str
    rendered_asset_byte_size: Annotated[StrictInt, Field(ge=1)]
    pixel_width: Annotated[StrictInt, Field(ge=1)]
    pixel_height: Annotated[StrictInt, Field(ge=1)]
    page_asset_digest: str
    exact_source_pdf_bytes_used: StrictBool = True
    full_page_rendered: StrictBool = True
    source_bbox_crop_applied: StrictBool = False

    @field_validator("asset_id", "renderer_id", "renderer_version", "render_profile")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("asset_filename", "source_id")
    @classmethod
    def validate_path_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "source_pdf_sha256",
        "rendered_asset_sha256",
        "page_asset_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_asset_integrity(self) -> OledSupplementarySourcePageAsset:
        if self.renderer_id != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDERER_ID:
            raise ValueError(
                "supplementary source transcription v1 requires the bound "
                "Poppler renderer"
            )
        if self.render_profile != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDER_PROFILE:
            raise ValueError(
                "supplementary source transcription v1 requires the bound "
                "200 dpi RGB full-page render profile"
            )
        suffix = _page_asset_suffix(_page_asset_identity(self))
        if self.asset_id != f"supplementary-source-page:{suffix}":
            raise ValueError("supplementary source page asset_id mismatch")
        if self.asset_filename != f"source-page-{suffix}.png":
            raise ValueError("supplementary source page asset filename mismatch")
        if not self.exact_source_pdf_bytes_used or not self.full_page_rendered:
            raise ValueError("supplementary source page lost a required evidence flag")
        if self.source_bbox_crop_applied:
            raise ValueError(
                "supplementary source transcription v1 forbids uncalibrated bbox crops"
            )
        if _page_asset_digest(self) != self.page_asset_digest:
            raise ValueError("supplementary source page asset digest mismatch")
        return self


class OledSupplementarySourcePdfEvidence(BaseModel):
    """Exact PDF-byte evidence and its bounded full-page render assets."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    source_pdf_sha256: str
    source_pdf_byte_size: Annotated[StrictInt, Field(ge=1)]
    source_pdf_page_count: Annotated[StrictInt, Field(ge=1)]
    page_counter_id: str = SUPPLEMENTARY_SOURCE_TRANSCRIPTION_PAGE_COUNTER_ID
    page_counter_version: str
    page_counter_executable_sha256: str
    renderer_executable_sha256: str
    poppler_runtime_trust_model: str = (
        SUPPLEMENTARY_SOURCE_TRANSCRIPTION_POPPLER_RUNTIME_TRUST_MODEL
    )
    dynamic_library_closure_bound: StrictBool = False
    poppler_executable_evidence_digest: str
    page_assets: list[OledSupplementarySourcePageAsset] = Field(default_factory=list)
    page_asset_count: Annotated[StrictInt, Field(ge=1)]
    asset_bundle_digest: str
    source_pdf_evidence_digest: str
    exact_source_pdf_bytes_verified: StrictBool = True
    source_pdf_read: StrictBool = True
    page_assets_rendered_from_bound_pdf: StrictBool = True
    external_renderer_called: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    rendered_pages_are_authoritative: StrictBool = False
    network_accessed: StrictBool = False

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator(
        "page_counter_id",
        "page_counter_version",
        "poppler_runtime_trust_model",
    )
    @classmethod
    def validate_tool_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "source_pdf_sha256",
        "page_counter_executable_sha256",
        "renderer_executable_sha256",
        "poppler_executable_evidence_digest",
        "asset_bundle_digest",
        "source_pdf_evidence_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_evidence_integrity(self) -> OledSupplementarySourcePdfEvidence:
        if self.page_counter_id != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_PAGE_COUNTER_ID:
            raise ValueError(
                "supplementary source transcription v1 requires the bound "
                "Poppler page counter"
            )
        if (
            self.poppler_runtime_trust_model
            != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_POPPLER_RUNTIME_TRUST_MODEL
            or self.dynamic_library_closure_bound
        ):
            raise ValueError(
                "supplementary source transcription v1 requires the explicit "
                "operator-trusted dynamic Poppler runtime boundary"
            )
        if not self.page_assets:
            raise ValueError("supplementary source PDF evidence requires page assets")
        expected_order = sorted(
            self.page_assets,
            key=lambda asset: (asset.pdf_page_number_one_based, asset.asset_id),
        )
        if self.page_assets != expected_order:
            raise ValueError("supplementary source PDF page assets must be ordered")
        asset_ids = [asset.asset_id for asset in self.page_assets]
        asset_filenames = [asset.asset_filename for asset in self.page_assets]
        asset_pages = [asset.pdf_page_number_one_based for asset in self.page_assets]
        if len(asset_ids) != len(set(asset_ids)) or len(asset_filenames) != len(
            set(asset_filenames)
        ):
            raise ValueError("supplementary source PDF page assets must be unique")
        if len(asset_pages) != len(set(asset_pages)):
            raise ValueError(
                "supplementary source PDF evidence requires one asset per page"
            )
        if self.page_asset_count != len(self.page_assets):
            raise ValueError("supplementary source PDF page asset count mismatch")
        for asset in self.page_assets:
            if (
                asset.source_id != self.source_id
                or asset.source_pdf_sha256 != self.source_pdf_sha256
            ):
                raise ValueError("supplementary source PDF page asset binding mismatch")
            if asset.pdf_page_number_one_based > self.source_pdf_page_count:
                raise ValueError("supplementary source PDF page asset is out of range")
        renderer_versions = {asset.renderer_version for asset in self.page_assets}
        if len(renderer_versions) != 1:
            raise ValueError(
                "supplementary source PDF page assets must use one renderer version"
            )
        if renderer_versions != {self.page_counter_version}:
            raise ValueError(
                "supplementary source PDF Poppler tool versions must match"
            )
        fixed_true = (
            "exact_source_pdf_bytes_verified",
            "source_pdf_read",
            "page_assets_rendered_from_bound_pdf",
            "external_renderer_called",
            "source_pdf_remains_authoritative",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("supplementary source PDF evidence lost a required flag")
        if self.rendered_pages_are_authoritative or self.network_accessed:
            raise ValueError("supplementary source PDF evidence crossed its boundary")
        if _asset_bundle_digest(self.page_assets) != self.asset_bundle_digest:
            raise ValueError("supplementary source PDF asset bundle digest mismatch")
        if (
            _poppler_executable_evidence_digest(self)
            != self.poppler_executable_evidence_digest
        ):
            raise ValueError(
                "supplementary source PDF Poppler executable evidence digest mismatch"
            )
        if _source_pdf_evidence_digest(self) != self.source_pdf_evidence_digest:
            raise ValueError("supplementary source PDF evidence digest mismatch")
        return self


class OledSupplementarySourceTranscriptionComponentDigests(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    page_anchor: str
    caption: str
    headers: str
    row_structure: str
    cell_literals: str
    footnotes: str
    table_extent: str

    @field_validator(*SOURCE_TRANSCRIPTION_COMPONENT_NAMES)
    @classmethod
    def validate_component_digest(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))


class OledSupplementarySourceTranscriptionComponentResults(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    page_anchor_check: OledSupplementarySourceTranscriptionComponentResult
    caption_check: OledSupplementarySourceTranscriptionComponentResult
    headers_check: OledSupplementarySourceTranscriptionComponentResult
    row_structure_check: OledSupplementarySourceTranscriptionComponentResult
    cell_literals_check: OledSupplementarySourceTranscriptionComponentResult
    footnotes_check: OledSupplementarySourceTranscriptionComponentResult
    table_extent_check: OledSupplementarySourceTranscriptionComponentResult


class OledSupplementarySourceTranscriptionReviewItem(BaseModel):
    """One all-or-nothing selected-table transcription review item."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: Literal[
        OledSupplementarySourceTranscriptionReviewItemKind.TABLE_TRANSCRIPTION_SCOPE
    ] = OledSupplementarySourceTranscriptionReviewItemKind.TABLE_TRANSCRIPTION_SCOPE
    visual_equivalence_contract_version: str = (
        SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION
    )
    scope_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    matched_table: OledSupplementaryReviewTable
    header_review_bindings: list[OledSupplementarySourceHeaderReviewBinding] = Field(
        default_factory=list
    )
    component_digests: OledSupplementarySourceTranscriptionComponentDigests
    page_asset_ids: list[str] = Field(default_factory=list)
    page_asset_digests: list[str] = Field(default_factory=list)
    full_table_cell_count: Annotated[StrictInt, Field(ge=1)]
    numeric_source_cell_count: Annotated[StrictInt, Field(ge=1)]
    source_cell_digests: list[str] = Field(default_factory=list)
    upstream_later_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_later_eligible_source_cell_digests: list[str] = Field(default_factory=list)
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_source_cell_digests: list[str] = Field(
        default_factory=list
    )
    upstream_source_check_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_source_check_pending_source_cell_digests: list[str] = Field(
        default_factory=list
    )
    upstream_rejected_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_rejected_source_cell_digests: list[str] = Field(default_factory=list)
    upstream_exclusion_confirmed_cell_count: Annotated[StrictInt, Field(ge=0)] = 0
    upstream_exclusion_confirmed_source_cell_digests: list[str] = Field(
        default_factory=list
    )
    upstream_blocked_by_scope_semantics_cell_count: Annotated[
        StrictInt, Field(ge=0)
    ] = 0
    upstream_blocked_by_scope_semantics_source_cell_digests: list[str] = Field(
        default_factory=list
    )
    source_pdf_remains_authoritative: StrictBool = True
    parsed_table_is_authoritative: StrictBool = False
    selected_table_extent_included_for_review: StrictBool = True
    document_wide_table_exhaustiveness_claimed: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False

    @field_validator("review_item_id", "scope_id", "table_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("visual_equivalence_contract_version")
    @classmethod
    def validate_contract_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION:
            raise ValueError("unexpected source transcription visual-equivalence contract")
        return value

    @field_validator(
        "review_item_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "source_cell_digests",
        "upstream_later_eligible_source_cell_digests",
        "upstream_ontology_review_pending_source_cell_digests",
        "upstream_source_check_pending_source_cell_digests",
        "upstream_rejected_source_cell_digests",
        "upstream_exclusion_confirmed_source_cell_digests",
        "upstream_blocked_by_scope_semantics_source_cell_digests",
        "page_asset_digests",
    )
    @classmethod
    def validate_sorted_digests(cls, value: list[str], info: Any) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name=str(info.field_name)) for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @field_validator("page_asset_ids")
    @classmethod
    def validate_page_asset_ids(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_bound_id(item, field_name="page_asset_id") for item in value
        ]
        if not clean or clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("page_asset_ids must be non-empty, sorted, and unique")
        return clean

    @model_validator(mode="after")
    def validate_item_integrity(self) -> OledSupplementarySourceTranscriptionReviewItem:
        if self.item_kind != (
            OledSupplementarySourceTranscriptionReviewItemKind.TABLE_TRANSCRIPTION_SCOPE
        ):
            raise ValueError("unexpected source transcription review item kind")
        if self.table_id != self.matched_table.table_id:
            raise ValueError("source transcription table_id binding mismatch")
        if self.table_content_digest != self.matched_table.table_content_digest:
            raise ValueError("source transcription table digest binding mismatch")
        if self.pdf_page_number_one_based != self.matched_table.page:
            raise ValueError("source transcription PDF page binding mismatch")
        expected_header_bindings = _header_review_bindings(self.matched_table)
        if self.header_review_bindings != expected_header_bindings:
            raise ValueError(
                "source transcription header-review bindings do not match the table"
            )
        if self.full_table_cell_count != (
            len(self.matched_table.rows) * len(self.matched_table.headers)
        ):
            raise ValueError("source transcription full table cell count mismatch")
        if self.numeric_source_cell_count != len(self.source_cell_digests):
            raise ValueError("source transcription numeric cell count mismatch")
        count_pairs = (
            (
                self.upstream_later_eligible_cell_count,
                self.upstream_later_eligible_source_cell_digests,
            ),
            (
                self.upstream_ontology_review_pending_cell_count,
                self.upstream_ontology_review_pending_source_cell_digests,
            ),
            (
                self.upstream_source_check_pending_cell_count,
                self.upstream_source_check_pending_source_cell_digests,
            ),
            (
                self.upstream_rejected_cell_count,
                self.upstream_rejected_source_cell_digests,
            ),
            (
                self.upstream_exclusion_confirmed_cell_count,
                self.upstream_exclusion_confirmed_source_cell_digests,
            ),
            (
                self.upstream_blocked_by_scope_semantics_cell_count,
                self.upstream_blocked_by_scope_semantics_source_cell_digests,
            ),
        )
        source_digests = set(self.source_cell_digests)
        for count, digests in count_pairs:
            if count != len(digests):
                raise ValueError("source transcription upstream cell count mismatch")
            if not set(digests).issubset(source_digests):
                raise ValueError("source transcription upstream cell roster mismatch")
        if set(self.upstream_later_eligible_source_cell_digests).intersection(
            set(self.upstream_ontology_review_pending_source_cell_digests)
            | set(self.upstream_source_check_pending_source_cell_digests)
            | set(self.upstream_rejected_source_cell_digests)
            | set(self.upstream_exclusion_confirmed_source_cell_digests)
        ):
            raise ValueError("source transcription eligible cells overlap blocked outcomes")
        if len(self.page_asset_ids) != len(self.page_asset_digests):
            raise ValueError("source transcription page asset binding count mismatch")
        expected_components = _component_digests_for_item(self)
        if self.component_digests != expected_components:
            raise ValueError("source transcription component digest mismatch")
        expected_id = _source_transcription_review_item_id(
            _source_transcription_review_item_identity(self)
        )
        if self.review_item_id != expected_id:
            raise ValueError("source transcription review_item_id mismatch")
        if _source_transcription_review_item_digest(self) != self.review_item_digest:
            raise ValueError("source transcription review item digest mismatch")
        fixed_true = (
            "source_pdf_remains_authoritative",
            "selected_table_extent_included_for_review",
        )
        fixed_false = (
            "parsed_table_is_authoritative",
            "document_wide_table_exhaustiveness_claimed",
            "scientific_content_validated",
            "physical_semantics_validated",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("source transcription review item lost a required flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("source transcription review item crossed its boundary")
        return self


class OledSupplementarySourceTranscriptionReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = SUPPLEMENTARY_SOURCE_TRANSCRIPTION_REVIEW_PACKET_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    response_artifact_sha256: str
    response_artifact_digest: str
    semantic_review_packet_sha256: str
    semantic_review_packet_digest: str
    semantic_decision_manifest_sha256: str
    semantic_decision_manifest_digest: str
    semantic_adjudication_artifact_sha256: str
    semantic_adjudication_artifact_digest: str
    source_pdf_evidence: OledSupplementarySourcePdfEvidence
    source_pdf_evidence_digest: str
    status: OledSupplementarySourceTranscriptionReviewPacketStatus
    scope_count: Annotated[StrictInt, Field(ge=1)]
    review_item_count: Annotated[StrictInt, Field(ge=1)]
    page_asset_count: Annotated[StrictInt, Field(ge=1)]
    full_table_cell_count: Annotated[StrictInt, Field(ge=1)]
    numeric_source_cell_count: Annotated[StrictInt, Field(ge=1)]
    upstream_later_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_source_check_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_rejected_cell_count: Annotated[StrictInt, Field(ge=0)]
    review_items: list[OledSupplementarySourceTranscriptionReviewItem] = Field(
        default_factory=list
    )
    review_packet_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    exact_source_pdf_bytes_verified: StrictBool = True
    upstream_chain_replayed: StrictBool = True
    full_parsed_table_included: StrictBool = True
    source_page_assets_rendered_from_bound_pdf: StrictBool = True
    strict_scope_partition_validated: StrictBool = True
    strict_cell_partition_validated: StrictBool = True
    human_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    parsed_table_is_authoritative: StrictBool = False
    rendered_pages_are_authoritative: StrictBool = False
    human_source_transcription_review_completed: StrictBool = False
    all_reviewed_scopes_transcription_validated: StrictBool = False
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
    source_pdf_read: StrictBool = True
    external_renderer_called: StrictBool = True
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_REVIEW_PACKET_VERSION:
            raise ValueError("unexpected source transcription review packet version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator(
        "request_artifact_sha256",
        "request_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_sha256",
        "response_artifact_digest",
        "semantic_review_packet_sha256",
        "semantic_review_packet_digest",
        "semantic_decision_manifest_sha256",
        "semantic_decision_manifest_digest",
        "semantic_adjudication_artifact_sha256",
        "semantic_adjudication_artifact_digest",
        "source_pdf_evidence_digest",
        "review_packet_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_packet_integrity(self) -> OledSupplementarySourceTranscriptionReviewPacket:
        if self.status != (
            OledSupplementarySourceTranscriptionReviewPacketStatus.READY_FOR_HUMAN_SOURCE_TRANSCRIPTION_REVIEW
        ):
            raise ValueError("unexpected source transcription packet status")
        if not self.review_items:
            raise ValueError("source transcription packet requires review items")
        if self.source_pdf_evidence_digest != (
            self.source_pdf_evidence.source_pdf_evidence_digest
        ):
            raise ValueError("source transcription PDF evidence binding mismatch")
        item_ids = [item.review_item_id for item in self.review_items]
        scope_table_keys = [(item.scope_id, item.table_id) for item in self.review_items]
        if (
            item_ids != sorted(item_ids)
            or len(item_ids) != len(set(item_ids))
            or len(scope_table_keys) != len(set(scope_table_keys))
        ):
            raise ValueError("source transcription review items must be ordered and unique")
        if self.scope_count != len({item.scope_id for item in self.review_items}):
            raise ValueError("source transcription scope count mismatch")
        if self.review_item_count != len(self.review_items):
            raise ValueError("source transcription review item count mismatch")
        if self.page_asset_count != self.source_pdf_evidence.page_asset_count:
            raise ValueError("source transcription page asset count mismatch")
        count_fields = (
            "full_table_cell_count",
            "numeric_source_cell_count",
            "upstream_later_eligible_cell_count",
            "upstream_ontology_review_pending_cell_count",
            "upstream_source_check_pending_cell_count",
            "upstream_rejected_cell_count",
        )
        for field_name in count_fields:
            item_field = field_name
            if getattr(self, field_name) != sum(
                getattr(item, item_field) for item in self.review_items
            ):
                raise ValueError(f"source transcription {field_name} mismatch")
        all_source_cell_digests = [
            source_cell_digest
            for item in self.review_items
            for source_cell_digest in item.source_cell_digests
        ]
        if len(all_source_cell_digests) != len(set(all_source_cell_digests)):
            raise ValueError("source transcription packet duplicates source cells")
        asset_by_id = {
            asset.asset_id: asset for asset in self.source_pdf_evidence.page_assets
        }
        referenced_assets: set[str] = set()
        for item in self.review_items:
            if (
                item.source_id != self.source_pdf_evidence.source_id
                or item.source_pdf_sha256
                != self.source_pdf_evidence.source_pdf_sha256
            ):
                raise ValueError("source transcription item PDF binding mismatch")
            expected_asset_digests = []
            for asset_id in item.page_asset_ids:
                asset = asset_by_id.get(asset_id)
                if asset is None:
                    raise ValueError("source transcription item references unknown page asset")
                if asset.pdf_page_number_one_based != item.pdf_page_number_one_based:
                    raise ValueError("source transcription item page asset mismatch")
                expected_asset_digests.append(asset.page_asset_digest)
                referenced_assets.add(asset_id)
            if sorted(expected_asset_digests) != item.page_asset_digests:
                raise ValueError("source transcription item page asset digest mismatch")
        if referenced_assets != set(asset_by_id):
            raise ValueError("source transcription packet contains orphan page assets")
        fixed_true = (
            "review_only",
            "offline_only",
            "exact_source_pdf_bytes_verified",
            "upstream_chain_replayed",
            "full_parsed_table_included",
            "source_page_assets_rendered_from_bound_pdf",
            "strict_scope_partition_validated",
            "strict_cell_partition_validated",
            "human_review_required",
            "source_pdf_remains_authoritative",
            "source_pdf_read",
            "external_renderer_called",
        )
        fixed_false = (
            "parsed_table_is_authoritative",
            "rendered_pages_are_authoritative",
            "human_source_transcription_review_completed",
            "all_reviewed_scopes_transcription_validated",
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
            raise ValueError("source transcription packet lost a required flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("source transcription packet crossed a downstream boundary")
        if _source_transcription_review_packet_digest(self) != self.review_packet_digest:
            raise ValueError("source transcription review packet digest mismatch")
        return self


class OledSupplementarySourceTranscriptionDecisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: Literal[
        OledSupplementarySourceTranscriptionReviewItemKind.TABLE_TRANSCRIPTION_SCOPE
    ]
    decision: OledSupplementarySourceTranscriptionDecision
    component_results: OledSupplementarySourceTranscriptionComponentResults
    review_note: str = ""

    @field_validator("review_item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="review_item_id")

    @field_validator("review_item_digest")
    @classmethod
    def validate_item_digest(cls, value: str) -> str:
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
    def validate_decision_shape(self) -> OledSupplementarySourceTranscriptionDecisionEntry:
        _validate_source_transcription_decision_components(
            decision=self.decision,
            component_results=self.component_results,
            review_note=self.review_note,
        )
        return self


class OledSupplementarySourceTranscriptionDecisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = SUPPLEMENTARY_SOURCE_TRANSCRIPTION_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    review_packet_sha256: str
    review_packet_digest: str
    source_pdf_evidence_digest: str
    reviewed_by: str
    reviewed_at: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledSupplementarySourceTranscriptionDecisionEntry] = Field(
        default_factory=list
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected source transcription decision version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "review_packet_sha256",
        "review_packet_digest",
        "source_pdf_evidence_digest",
    )
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
    def validate_manifest_integrity(self) -> OledSupplementarySourceTranscriptionDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("source transcription adjudication requires confirmation")
        if not self.decisions:
            raise ValueError("source transcription adjudication requires decisions")
        ids = [decision.review_item_id for decision in self.decisions]
        if len(ids) != len(set(ids)):
            raise ValueError("source transcription decisions must be unique")
        return self


class OledSupplementaryAdjudicatedSourceTranscription(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    scope_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    header_review_bindings: list[OledSupplementarySourceHeaderReviewBinding] = Field(
        default_factory=list
    )
    component_digests: OledSupplementarySourceTranscriptionComponentDigests
    component_results: OledSupplementarySourceTranscriptionComponentResults
    page_asset_ids: list[str] = Field(default_factory=list)
    page_asset_digests: list[str] = Field(default_factory=list)
    full_table_cell_count: Annotated[StrictInt, Field(ge=1)]
    numeric_source_cell_count: Annotated[StrictInt, Field(ge=1)]
    source_cell_digests: list[str] = Field(default_factory=list)
    upstream_later_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_later_eligible_source_cell_digests: list[str] = Field(default_factory=list)
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_source_check_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_rejected_cell_count: Annotated[StrictInt, Field(ge=0)]
    decision: OledSupplementarySourceTranscriptionDecision
    review_note: str = ""
    table_transcription_validated: StrictBool
    bounded_selected_table_extent_validated: StrictBool
    reparse_required: StrictBool
    source_check_pending: StrictBool
    rejected: StrictBool
    later_identity_review_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    later_identity_review_eligible_source_cell_digests: list[str] = Field(
        default_factory=list
    )
    source_pdf_remains_authoritative: StrictBool = True
    parsed_table_is_authoritative: StrictBool = False
    table_exhaustiveness_validated: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False
    material_identity_resolved: StrictBool = False

    @field_validator("review_item_id", "scope_id", "table_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator(
        "review_item_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
        "source_cell_digests",
        "page_asset_digests",
        "upstream_later_eligible_source_cell_digests",
        "later_identity_review_eligible_source_cell_digests",
        mode="before",
    )
    @classmethod
    def validate_digest_fields(cls, value: Any, info: Any) -> Any:
        if isinstance(value, list):
            clean = [
                _normalize_sha256(item, field_name=str(info.field_name))
                for item in value
            ]
            if clean != sorted(clean) or len(clean) != len(set(clean)):
                raise ValueError(f"{info.field_name} must be sorted and unique")
            return clean
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("page_asset_ids")
    @classmethod
    def validate_page_asset_ids(cls, value: list[str]) -> list[str]:
        clean = [_validate_bound_id(item, field_name="page_asset_id") for item in value]
        if not clean or clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("adjudicated source page assets must be sorted and unique")
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
    def validate_result_integrity(self) -> OledSupplementaryAdjudicatedSourceTranscription:
        _validate_source_transcription_decision_components(
            decision=self.decision,
            component_results=self.component_results,
            review_note=self.review_note,
        )
        accepted = (
            self.decision
            == OledSupplementarySourceTranscriptionDecision.ACCEPT_BOUNDED_SOURCE_TRANSCRIPTION
        )
        if self.table_transcription_validated != accepted:
            raise ValueError("adjudicated source transcription validation flag mismatch")
        if self.bounded_selected_table_extent_validated != accepted:
            raise ValueError("adjudicated selected-table extent flag mismatch")
        if self.reparse_required != (
            self.decision == OledSupplementarySourceTranscriptionDecision.NEEDS_REPARSE
        ):
            raise ValueError("adjudicated source transcription reparse flag mismatch")
        if self.source_check_pending != (
            self.decision
            == OledSupplementarySourceTranscriptionDecision.NEEDS_SOURCE_CHECK
        ):
            raise ValueError("adjudicated source transcription source-check flag mismatch")
        if self.rejected != (
            self.decision == OledSupplementarySourceTranscriptionDecision.REJECT_SCOPE
        ):
            raise ValueError("adjudicated source transcription rejection flag mismatch")
        expected_eligible_count = (
            self.upstream_later_eligible_cell_count if accepted else 0
        )
        if self.upstream_later_eligible_cell_count != len(
            self.upstream_later_eligible_source_cell_digests
        ):
            raise ValueError("adjudicated upstream eligibility count mismatch")
        if (
            self.later_identity_review_eligible_cell_count
            != expected_eligible_count
            or self.later_identity_review_eligible_cell_count
            != len(self.later_identity_review_eligible_source_cell_digests)
        ):
            raise ValueError("adjudicated later identity-review eligibility mismatch")
        expected_eligible_digests = (
            self.upstream_later_eligible_source_cell_digests if accepted else []
        )
        if (
            self.later_identity_review_eligible_source_cell_digests
            != expected_eligible_digests
        ):
            raise ValueError("adjudicated identity-review cell binding mismatch")
        if not set(self.later_identity_review_eligible_source_cell_digests).issubset(
            set(self.source_cell_digests)
        ):
            raise ValueError("adjudicated identity-review cells are outside the table")
        if self.numeric_source_cell_count != len(self.source_cell_digests):
            raise ValueError("adjudicated source transcription cell count mismatch")
        if self.full_table_cell_count < self.numeric_source_cell_count:
            raise ValueError("adjudicated source transcription table extent mismatch")
        if len(self.page_asset_ids) != len(self.page_asset_digests):
            raise ValueError("adjudicated source transcription page asset mismatch")
        _validate_standalone_header_review_bindings(self.header_review_bindings)
        if (
            _headers_component_digest(self.header_review_bindings)
            != self.component_digests.headers
        ):
            raise ValueError(
                "adjudicated source transcription header bindings changed"
            )
        for count in (
            self.upstream_later_eligible_cell_count,
            self.upstream_ontology_review_pending_cell_count,
            self.upstream_source_check_pending_cell_count,
            self.upstream_rejected_cell_count,
        ):
            if count > self.numeric_source_cell_count:
                raise ValueError("adjudicated upstream count exceeds source cells")
        fixed_true = ("source_pdf_remains_authoritative",)
        fixed_false = (
            "parsed_table_is_authoritative",
            "table_exhaustiveness_validated",
            "scientific_content_validated",
            "physical_semantics_validated",
            "material_identity_resolved",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("adjudicated source transcription lost a required flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("adjudicated source transcription crossed its boundary")
        return self


class OledSupplementarySourceTranscriptionAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = (
        SUPPLEMENTARY_SOURCE_TRANSCRIPTION_ADJUDICATION_ARTIFACT_VERSION
    )
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    response_artifact_sha256: str
    response_artifact_digest: str
    semantic_review_packet_sha256: str
    semantic_review_packet_digest: str
    semantic_decision_manifest_sha256: str
    semantic_decision_manifest_digest: str
    semantic_adjudication_artifact_sha256: str
    semantic_adjudication_artifact_digest: str
    source_pdf_evidence_digest: str
    review_packet_sha256: str
    review_packet_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    reviewed_by: str
    reviewed_at: str
    status: OledSupplementarySourceTranscriptionAdjudicationStatus
    review_item_count: Annotated[StrictInt, Field(ge=1)]
    accepted_scope_count: Annotated[StrictInt, Field(ge=0)]
    reparse_required_scope_count: Annotated[StrictInt, Field(ge=0)]
    source_check_pending_scope_count: Annotated[StrictInt, Field(ge=0)]
    rejected_scope_count: Annotated[StrictInt, Field(ge=0)]
    unresolved_review_item_count: Annotated[StrictInt, Field(ge=0)]
    bounded_transcription_validated_cell_count: Annotated[StrictInt, Field(ge=0)]
    later_identity_review_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_source_check_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_rejected_cell_count: Annotated[StrictInt, Field(ge=0)]
    adjudicated_tables: list[OledSupplementaryAdjudicatedSourceTranscription] = Field(
        default_factory=list
    )
    adjudication_artifact_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    upstream_chain_revalidated: StrictBool = True
    review_packet_binding_revalidated: StrictBool = True
    source_pdf_evidence_binding_revalidated: StrictBool = True
    decision_binding_validated: StrictBool = True
    human_decisions_recorded: StrictBool = True
    human_source_transcription_review_completed: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    parsed_table_is_authoritative: StrictBool = False
    rendered_pages_are_authoritative: StrictBool = False
    all_reviewed_scopes_transcription_validated: StrictBool
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
    source_pdf_read: StrictBool = True
    external_renderer_called: StrictBool = True
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SOURCE_TRANSCRIPTION_ADJUDICATION_ARTIFACT_VERSION:
            raise ValueError("unexpected source transcription adjudication version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at", "reviewed_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "request_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_sha256",
        "response_artifact_digest",
        "semantic_review_packet_sha256",
        "semantic_review_packet_digest",
        "semantic_decision_manifest_sha256",
        "semantic_decision_manifest_digest",
        "semantic_adjudication_artifact_sha256",
        "semantic_adjudication_artifact_digest",
        "source_pdf_evidence_digest",
        "review_packet_sha256",
        "review_packet_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "adjudication_artifact_digest",
    )
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

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledSupplementarySourceTranscriptionAdjudicationArtifact:
        if _parse_timestamp(self.reviewed_at) > _parse_timestamp(self.generated_at):
            raise ValueError("source transcription adjudication predates its decisions")
        if self.review_item_count != len(self.adjudicated_tables):
            raise ValueError("source transcription adjudication item count mismatch")
        item_ids = [item.review_item_id for item in self.adjudicated_tables]
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("source transcription adjudicated tables must be ordered")
        expected_counts = {
            "accepted_scope_count": sum(
                item.table_transcription_validated for item in self.adjudicated_tables
            ),
            "reparse_required_scope_count": sum(
                item.reparse_required for item in self.adjudicated_tables
            ),
            "source_check_pending_scope_count": sum(
                item.source_check_pending for item in self.adjudicated_tables
            ),
            "rejected_scope_count": sum(item.rejected for item in self.adjudicated_tables),
            "bounded_transcription_validated_cell_count": sum(
                item.numeric_source_cell_count
                for item in self.adjudicated_tables
                if item.table_transcription_validated
            ),
            "later_identity_review_eligible_cell_count": sum(
                item.later_identity_review_eligible_cell_count
                for item in self.adjudicated_tables
            ),
            "upstream_ontology_review_pending_cell_count": sum(
                item.upstream_ontology_review_pending_cell_count
                for item in self.adjudicated_tables
            ),
            "upstream_source_check_pending_cell_count": sum(
                item.upstream_source_check_pending_cell_count
                for item in self.adjudicated_tables
            ),
            "upstream_rejected_cell_count": sum(
                item.upstream_rejected_cell_count
                for item in self.adjudicated_tables
            ),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"source transcription adjudication {field_name} mismatch")
        unresolved = (
            self.reparse_required_scope_count + self.source_check_pending_scope_count
        )
        if self.unresolved_review_item_count != unresolved:
            raise ValueError("source transcription unresolved item count mismatch")
        if self.all_reviewed_scopes_transcription_validated != (
            self.accepted_scope_count == self.review_item_count
        ):
            raise ValueError("source transcription aggregate validation flag mismatch")
        if self.status != _source_transcription_adjudication_status(
            accepted_count=self.accepted_scope_count,
            unresolved_count=unresolved,
            eligible_cell_count=self.later_identity_review_eligible_cell_count,
        ):
            raise ValueError("source transcription adjudication status mismatch")
        fixed_true = (
            "review_only",
            "offline_only",
            "upstream_chain_revalidated",
            "review_packet_binding_revalidated",
            "source_pdf_evidence_binding_revalidated",
            "decision_binding_validated",
            "human_decisions_recorded",
            "human_source_transcription_review_completed",
            "source_pdf_remains_authoritative",
            "source_pdf_read",
            "external_renderer_called",
        )
        fixed_false = (
            "parsed_table_is_authoritative",
            "rendered_pages_are_authoritative",
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
            raise ValueError("source transcription adjudication lost a required flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("source transcription adjudication crossed a downstream boundary")
        if (
            _source_transcription_adjudication_digest(self)
            != self.adjudication_artifact_digest
        ):
            raise ValueError("source transcription adjudication digest mismatch")
        return self


def build_oled_supplementary_source_page_asset(
    *,
    source_id: str,
    source_pdf_sha256: str,
    pdf_page_number_one_based: int,
    renderer_id: str,
    renderer_version: str,
    render_profile: str,
    rendered_asset_sha256: str,
    rendered_asset_byte_size: int,
    pixel_width: int,
    pixel_height: int,
) -> OledSupplementarySourcePageAsset:
    identity = {
        "source_id": source_id,
        "source_pdf_sha256": _normalize_sha256(
            source_pdf_sha256,
            field_name="source_pdf_sha256",
        ),
        "pdf_page_number_one_based": pdf_page_number_one_based,
        "renderer_id": renderer_id,
        "renderer_version": renderer_version,
        "render_profile": render_profile,
        "mime_type": "image/png",
        "rendered_asset_sha256": _normalize_sha256(
            rendered_asset_sha256,
            field_name="rendered_asset_sha256",
        ),
        "rendered_asset_byte_size": rendered_asset_byte_size,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "exact_source_pdf_bytes_used": True,
        "full_page_rendered": True,
        "source_bbox_crop_applied": False,
    }
    suffix = _page_asset_suffix(identity)
    payload = {
        **identity,
        "asset_id": f"supplementary-source-page:{suffix}",
        "asset_filename": f"source-page-{suffix}.png",
        "page_asset_digest": "sha256:" + "0" * 64,
    }
    provisional = OledSupplementarySourcePageAsset.model_construct(**payload)
    payload["page_asset_digest"] = _page_asset_digest(provisional)
    return OledSupplementarySourcePageAsset.model_validate(payload)


def build_oled_supplementary_source_pdf_evidence(
    *,
    source_id: str,
    source_pdf_sha256: str,
    source_pdf_byte_size: int,
    source_pdf_page_count: int,
    page_counter_version: str,
    page_counter_executable_sha256: str,
    renderer_executable_sha256: str,
    page_assets: Sequence[OledSupplementarySourcePageAsset],
) -> OledSupplementarySourcePdfEvidence:
    assets = [
        OledSupplementarySourcePageAsset.model_validate(asset.model_dump(mode="json"))
        for asset in page_assets
    ]
    assets.sort(key=lambda asset: (asset.pdf_page_number_one_based, asset.asset_id))
    payload: dict[str, Any] = {
        "source_id": source_id,
        "source_pdf_sha256": _normalize_sha256(
            source_pdf_sha256,
            field_name="source_pdf_sha256",
        ),
        "source_pdf_byte_size": source_pdf_byte_size,
        "source_pdf_page_count": source_pdf_page_count,
        "page_counter_id": SUPPLEMENTARY_SOURCE_TRANSCRIPTION_PAGE_COUNTER_ID,
        "page_counter_version": page_counter_version,
        "page_counter_executable_sha256": _normalize_sha256(
            page_counter_executable_sha256,
            field_name="page_counter_executable_sha256",
        ),
        "renderer_executable_sha256": _normalize_sha256(
            renderer_executable_sha256,
            field_name="renderer_executable_sha256",
        ),
        "poppler_runtime_trust_model": (
            SUPPLEMENTARY_SOURCE_TRANSCRIPTION_POPPLER_RUNTIME_TRUST_MODEL
        ),
        "dynamic_library_closure_bound": False,
        "poppler_executable_evidence_digest": "sha256:" + "0" * 64,
        "page_assets": assets,
        "page_asset_count": len(assets),
        "asset_bundle_digest": _asset_bundle_digest(assets),
        "source_pdf_evidence_digest": "sha256:" + "0" * 64,
        "exact_source_pdf_bytes_verified": True,
        "source_pdf_read": True,
        "page_assets_rendered_from_bound_pdf": True,
        "external_renderer_called": True,
        "source_pdf_remains_authoritative": True,
        "rendered_pages_are_authoritative": False,
        "network_accessed": False,
    }
    provisional = OledSupplementarySourcePdfEvidence.model_construct(**payload)
    payload["poppler_executable_evidence_digest"] = (
        _poppler_executable_evidence_digest(provisional)
    )
    provisional = OledSupplementarySourcePdfEvidence.model_construct(**payload)
    payload["source_pdf_evidence_digest"] = _source_pdf_evidence_digest(provisional)
    return OledSupplementarySourcePdfEvidence.model_validate(payload)


def validate_oled_supplementary_source_transcription_inputs(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryScopedCandidateResponseArtifact,
    response_artifact_sha256: str,
    semantic_review_packet: OledSupplementarySemanticReviewPacket,
    semantic_review_packet_sha256: str,
    semantic_decision_manifest: OledSupplementarySemanticDecisionManifest,
    semantic_decision_manifest_sha256: str,
    semantic_adjudication_artifact: OledSupplementarySemanticAdjudicationArtifact,
    semantic_adjudication_artifact_sha256: str,
) -> None:
    """Replay the complete PR-G/H/I chain and reject copied audit flags."""

    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    response = OledSupplementaryScopedCandidateResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    response_validation = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    semantic_packet = OledSupplementarySemanticReviewPacket.model_validate(
        semantic_review_packet.model_dump(mode="json")
    )
    semantic_decisions = OledSupplementarySemanticDecisionManifest.model_validate(
        semantic_decision_manifest.model_dump(mode="json")
    )
    semantic_adjudication = OledSupplementarySemanticAdjudicationArtifact.model_validate(
        semantic_adjudication_artifact.model_dump(mode="json")
    )
    normalized_hashes = {
        "request": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "manifest": _normalize_sha256(
            response_manifest_sha256,
            field_name="response_manifest_sha256",
        ),
        "response": _normalize_sha256(
            response_artifact_sha256,
            field_name="response_artifact_sha256",
        ),
        "packet": _normalize_sha256(
            semantic_review_packet_sha256,
            field_name="semantic_review_packet_sha256",
        ),
        "decisions": _normalize_sha256(
            semantic_decision_manifest_sha256,
            field_name="semantic_decision_manifest_sha256",
        ),
        "adjudication": _normalize_sha256(
            semantic_adjudication_artifact_sha256,
            field_name="semantic_adjudication_artifact_sha256",
        ),
    }
    expected_packet = build_oled_supplementary_semantic_review_packet(
        request_artifact=request,
        request_artifact_sha256=normalized_hashes["request"],
        response_manifest=response,
        response_manifest_sha256=normalized_hashes["manifest"],
        response_artifact=response_validation,
        response_artifact_sha256=normalized_hashes["response"],
        generated_at=semantic_packet.generated_at,
    )
    if expected_packet.model_dump(mode="json") != semantic_packet.model_dump(
        mode="json"
    ):
        raise ValueError("source transcription PR-I review packet binding mismatch")
    if semantic_decisions.review_packet_sha256 != normalized_hashes["packet"]:
        raise ValueError("source transcription semantic decision byte binding mismatch")
    expected_adjudication = build_oled_supplementary_semantic_adjudication_artifact(
        request_artifact=request,
        request_artifact_sha256=normalized_hashes["request"],
        response_manifest=response,
        response_manifest_sha256=normalized_hashes["manifest"],
        response_artifact=response_validation,
        response_artifact_sha256=normalized_hashes["response"],
        review_packet=semantic_packet,
        review_packet_sha256=normalized_hashes["packet"],
        decision_manifest=semantic_decisions,
        decision_manifest_sha256=normalized_hashes["decisions"],
        generated_at=semantic_adjudication.generated_at,
    )
    if expected_adjudication.model_dump(
        mode="json"
    ) != semantic_adjudication.model_dump(mode="json"):
        raise ValueError("source transcription PR-I adjudication binding mismatch")
    identities = {
        (request.run_id, request.paper_id),
        (response.run_id, response.paper_id),
        (response_validation.run_id, response_validation.paper_id),
        (semantic_packet.run_id, semantic_packet.paper_id),
        (semantic_decisions.run_id, semantic_decisions.paper_id),
        (semantic_adjudication.run_id, semantic_adjudication.paper_id),
    }
    if len(identities) != 1:
        raise ValueError("source transcription upstream identity mismatch")


def build_oled_supplementary_source_transcription_review_packet(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryScopedCandidateResponseArtifact,
    response_artifact_sha256: str,
    semantic_review_packet: OledSupplementarySemanticReviewPacket,
    semantic_review_packet_sha256: str,
    semantic_decision_manifest: OledSupplementarySemanticDecisionManifest,
    semantic_decision_manifest_sha256: str,
    semantic_adjudication_artifact: OledSupplementarySemanticAdjudicationArtifact,
    semantic_adjudication_artifact_sha256: str,
    source_pdf_evidence: OledSupplementarySourcePdfEvidence,
    generated_at: str,
) -> OledSupplementarySourceTranscriptionReviewPacket:
    validate_oled_supplementary_source_transcription_inputs(
        request_artifact=request_artifact,
        request_artifact_sha256=request_artifact_sha256,
        response_manifest=response_manifest,
        response_manifest_sha256=response_manifest_sha256,
        response_artifact=response_artifact,
        response_artifact_sha256=response_artifact_sha256,
        semantic_review_packet=semantic_review_packet,
        semantic_review_packet_sha256=semantic_review_packet_sha256,
        semantic_decision_manifest=semantic_decision_manifest,
        semantic_decision_manifest_sha256=semantic_decision_manifest_sha256,
        semantic_adjudication_artifact=semantic_adjudication_artifact,
        semantic_adjudication_artifact_sha256=semantic_adjudication_artifact_sha256,
    )
    request = OledSupplementaryScopedCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    response_validation = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    semantic_packet = OledSupplementarySemanticReviewPacket.model_validate(
        semantic_review_packet.model_dump(mode="json")
    )
    semantic_adjudication = OledSupplementarySemanticAdjudicationArtifact.model_validate(
        semantic_adjudication_artifact.model_dump(mode="json")
    )
    evidence = OledSupplementarySourcePdfEvidence.model_validate(
        source_pdf_evidence.model_dump(mode="json")
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(generated_at_clean) < _parse_timestamp(
        semantic_adjudication.generated_at
    ):
        raise ValueError("source transcription packet predates PR-I adjudication")
    scope_source_pairs = {
        (scope.source_id, scope.source_pdf_sha256) for scope in semantic_packet.scopes
    }
    if scope_source_pairs != {(evidence.source_id, evidence.source_pdf_sha256)}:
        raise ValueError("source transcription v1 requires one exact bound source PDF")
    assets_by_page: dict[int, list[OledSupplementarySourcePageAsset]] = {}
    for asset in evidence.page_assets:
        assets_by_page.setdefault(asset.pdf_page_number_one_based, []).append(asset)
    adjudicated_cells_by_scope: dict[str, list[Any]] = {}
    for cell in semantic_adjudication.adjudicated_cells:
        adjudicated_cells_by_scope.setdefault(
            cell.source_cell.scope_id,
            [],
        ).append(cell)
    items: list[OledSupplementarySourceTranscriptionReviewItem] = []
    for scope in semantic_packet.scopes:
        table = scope.matched_table
        if table.page < 1 or table.page > evidence.source_pdf_page_count:
            raise ValueError("source transcription table page is outside the source PDF")
        page_assets = sorted(
            assets_by_page.get(table.page, []),
            key=lambda asset: asset.asset_id,
        )
        if not page_assets:
            raise ValueError("source transcription table lacks a bound full-page asset")
        cells = adjudicated_cells_by_scope.get(scope.scope_id, [])
        if not cells:
            raise ValueError("source transcription scope lacks PR-I adjudicated cells")
        source_cell_digests = sorted(
            cell.source_cell.source_cell_digest for cell in cells
        )
        if len(source_cell_digests) != len(set(source_cell_digests)):
            raise ValueError("source transcription scope contains duplicate source cells")
        lists = {
            "later": sorted(
                cell.source_cell.source_cell_digest
                for cell in cells
                if cell.eligible_for_later_materialization_review
            ),
            "ontology": sorted(
                cell.source_cell.source_cell_digest
                for cell in cells
                if cell.ontology_review_pending
            ),
            "source": sorted(
                cell.source_cell.source_cell_digest
                for cell in cells
                if cell.source_check_pending
            ),
            "rejected": sorted(
                cell.source_cell.source_cell_digest for cell in cells if cell.rejected
            ),
            "exclusion": sorted(
                cell.source_cell.source_cell_digest
                for cell in cells
                if cell.exclusion_confirmed
            ),
            "semantic_block": sorted(
                cell.source_cell.source_cell_digest
                for cell in cells
                if cell.blocked_by_scope_semantics
            ),
        }
        header_bindings = _header_review_bindings(table)
        base_payload: dict[str, Any] = {
            "review_item_id": "supplementary-source-transcription:placeholder",
            "review_item_digest": "sha256:" + "0" * 64,
            "item_kind": OledSupplementarySourceTranscriptionReviewItemKind.TABLE_TRANSCRIPTION_SCOPE,
            "visual_equivalence_contract_version": (
                SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION
            ),
            "scope_id": scope.scope_id,
            "source_id": scope.source_id,
            "source_pdf_sha256": scope.source_pdf_sha256,
            "parsed_document_sha256": scope.parsed_document_sha256,
            "table_id": table.table_id,
            "table_content_digest": table.table_content_digest,
            "pdf_page_number_one_based": table.page,
            "matched_table": table,
            "header_review_bindings": header_bindings,
            "component_digests": {},
            "page_asset_ids": sorted(asset.asset_id for asset in page_assets),
            "page_asset_digests": sorted(
                asset.page_asset_digest for asset in page_assets
            ),
            "full_table_cell_count": len(table.rows) * len(table.headers),
            "numeric_source_cell_count": len(source_cell_digests),
            "source_cell_digests": source_cell_digests,
            "upstream_later_eligible_cell_count": len(lists["later"]),
            "upstream_later_eligible_source_cell_digests": lists["later"],
            "upstream_ontology_review_pending_cell_count": len(lists["ontology"]),
            "upstream_ontology_review_pending_source_cell_digests": lists[
                "ontology"
            ],
            "upstream_source_check_pending_cell_count": len(lists["source"]),
            "upstream_source_check_pending_source_cell_digests": lists["source"],
            "upstream_rejected_cell_count": len(lists["rejected"]),
            "upstream_rejected_source_cell_digests": lists["rejected"],
            "upstream_exclusion_confirmed_cell_count": len(lists["exclusion"]),
            "upstream_exclusion_confirmed_source_cell_digests": lists[
                "exclusion"
            ],
            "upstream_blocked_by_scope_semantics_cell_count": len(
                lists["semantic_block"]
            ),
            "upstream_blocked_by_scope_semantics_source_cell_digests": lists[
                "semantic_block"
            ],
            "source_pdf_remains_authoritative": True,
            "parsed_table_is_authoritative": False,
            "selected_table_extent_included_for_review": True,
            "document_wide_table_exhaustiveness_claimed": False,
            "scientific_content_validated": False,
            "physical_semantics_validated": False,
        }
        component_digests = _component_digests_from_payload(base_payload)
        base_payload["component_digests"] = component_digests
        provisional = OledSupplementarySourceTranscriptionReviewItem.model_construct(
            **base_payload
        )
        base_payload["review_item_id"] = _source_transcription_review_item_id(
            _source_transcription_review_item_identity(provisional)
        )
        provisional = OledSupplementarySourceTranscriptionReviewItem.model_construct(
            **base_payload
        )
        base_payload["review_item_digest"] = (
            _source_transcription_review_item_digest(provisional)
        )
        items.append(
            OledSupplementarySourceTranscriptionReviewItem.model_validate(
                base_payload
            )
        )
    request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    manifest_sha = _normalize_sha256(
        response_manifest_sha256,
        field_name="response_manifest_sha256",
    )
    response_sha = _normalize_sha256(
        response_artifact_sha256,
        field_name="response_artifact_sha256",
    )
    semantic_packet_sha = _normalize_sha256(
        semantic_review_packet_sha256,
        field_name="semantic_review_packet_sha256",
    )
    semantic_decisions_sha = _normalize_sha256(
        semantic_decision_manifest_sha256,
        field_name="semantic_decision_manifest_sha256",
    )
    semantic_adjudication_sha = _normalize_sha256(
        semantic_adjudication_artifact_sha256,
        field_name="semantic_adjudication_artifact_sha256",
    )
    items.sort(key=lambda item: item.review_item_id)
    payload = {
        "artifact_version": SUPPLEMENTARY_SOURCE_TRANSCRIPTION_REVIEW_PACKET_VERSION,
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": request_sha,
        "request_digest": request.request_digest,
        "response_manifest_sha256": manifest_sha,
        "response_manifest_digest": response_validation.response_manifest_digest,
        "response_artifact_sha256": response_sha,
        "response_artifact_digest": response_validation.response_artifact_digest,
        "semantic_review_packet_sha256": semantic_packet_sha,
        "semantic_review_packet_digest": semantic_packet.review_packet_digest,
        "semantic_decision_manifest_sha256": semantic_decisions_sha,
        "semantic_decision_manifest_digest": (
            semantic_adjudication.decision_manifest_digest
        ),
        "semantic_adjudication_artifact_sha256": semantic_adjudication_sha,
        "semantic_adjudication_artifact_digest": (
            semantic_adjudication.adjudication_artifact_digest
        ),
        "source_pdf_evidence": evidence,
        "source_pdf_evidence_digest": evidence.source_pdf_evidence_digest,
        "status": OledSupplementarySourceTranscriptionReviewPacketStatus.READY_FOR_HUMAN_SOURCE_TRANSCRIPTION_REVIEW,
        "scope_count": len({item.scope_id for item in items}),
        "review_item_count": len(items),
        "page_asset_count": evidence.page_asset_count,
        "full_table_cell_count": sum(item.full_table_cell_count for item in items),
        "numeric_source_cell_count": sum(
            item.numeric_source_cell_count for item in items
        ),
        "upstream_later_eligible_cell_count": sum(
            item.upstream_later_eligible_cell_count for item in items
        ),
        "upstream_ontology_review_pending_cell_count": sum(
            item.upstream_ontology_review_pending_cell_count for item in items
        ),
        "upstream_source_check_pending_cell_count": sum(
            item.upstream_source_check_pending_cell_count for item in items
        ),
        "upstream_rejected_cell_count": sum(
            item.upstream_rejected_cell_count for item in items
        ),
        "review_items": items,
        "review_packet_digest": "sha256:" + "0" * 64,
        "review_only": True,
        "offline_only": True,
        "exact_source_pdf_bytes_verified": True,
        "upstream_chain_replayed": True,
        "full_parsed_table_included": True,
        "source_page_assets_rendered_from_bound_pdf": True,
        "strict_scope_partition_validated": True,
        "strict_cell_partition_validated": True,
        "human_review_required": True,
        "source_pdf_remains_authoritative": True,
        "parsed_table_is_authoritative": False,
        "rendered_pages_are_authoritative": False,
        "human_source_transcription_review_completed": False,
        "all_reviewed_scopes_transcription_validated": False,
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
        "source_pdf_read": True,
        "external_renderer_called": True,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
    }
    provisional_packet = OledSupplementarySourceTranscriptionReviewPacket.model_construct(
        **payload
    )
    payload["review_packet_digest"] = _source_transcription_review_packet_digest(
        provisional_packet
    )
    return OledSupplementarySourceTranscriptionReviewPacket.model_validate(payload)


def validate_oled_supplementary_source_transcription_decision_binding(
    *,
    review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    review_packet_sha256: str,
    decision_manifest: OledSupplementarySourceTranscriptionDecisionManifest,
) -> None:
    packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        review_packet.model_dump(mode="json")
    )
    decisions = OledSupplementarySourceTranscriptionDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    if decisions.run_id != packet.run_id or decisions.paper_id != packet.paper_id:
        raise ValueError("source transcription decision identity mismatch")
    if decisions.review_packet_sha256 != _normalize_sha256(
        review_packet_sha256,
        field_name="review_packet_sha256",
    ):
        raise ValueError("source transcription decision byte binding mismatch")
    if decisions.review_packet_digest != packet.review_packet_digest:
        raise ValueError("source transcription decision content binding mismatch")
    if decisions.source_pdf_evidence_digest != packet.source_pdf_evidence_digest:
        raise ValueError("source transcription decision PDF evidence binding mismatch")
    if _parse_timestamp(decisions.reviewed_at) < _parse_timestamp(packet.generated_at):
        raise ValueError("source transcription decision predates its packet")
    item_by_id = {item.review_item_id: item for item in packet.review_items}
    decision_by_id = {item.review_item_id: item for item in decisions.decisions}
    if set(item_by_id) != set(decision_by_id):
        raise ValueError("source transcription decisions must exactly cover review items")
    for item_id, item in item_by_id.items():
        decision = decision_by_id[item_id]
        if (
            decision.review_item_digest != item.review_item_digest
            or decision.item_kind != item.item_kind
        ):
            raise ValueError("source transcription decision item binding mismatch")


def build_oled_supplementary_source_transcription_adjudication_artifact(
    *,
    request_artifact: OledSupplementaryScopedCandidateRequestArtifact,
    request_artifact_sha256: str,
    response_manifest: OledSupplementaryScopedCandidateResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryScopedCandidateResponseArtifact,
    response_artifact_sha256: str,
    semantic_review_packet: OledSupplementarySemanticReviewPacket,
    semantic_review_packet_sha256: str,
    semantic_decision_manifest: OledSupplementarySemanticDecisionManifest,
    semantic_decision_manifest_sha256: str,
    semantic_adjudication_artifact: OledSupplementarySemanticAdjudicationArtifact,
    semantic_adjudication_artifact_sha256: str,
    source_pdf_evidence: OledSupplementarySourcePdfEvidence,
    review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    review_packet_sha256: str,
    decision_manifest: OledSupplementarySourceTranscriptionDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledSupplementarySourceTranscriptionAdjudicationArtifact:
    packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        review_packet.model_dump(mode="json")
    )
    expected_packet = build_oled_supplementary_source_transcription_review_packet(
        request_artifact=request_artifact,
        request_artifact_sha256=request_artifact_sha256,
        response_manifest=response_manifest,
        response_manifest_sha256=response_manifest_sha256,
        response_artifact=response_artifact,
        response_artifact_sha256=response_artifact_sha256,
        semantic_review_packet=semantic_review_packet,
        semantic_review_packet_sha256=semantic_review_packet_sha256,
        semantic_decision_manifest=semantic_decision_manifest,
        semantic_decision_manifest_sha256=semantic_decision_manifest_sha256,
        semantic_adjudication_artifact=semantic_adjudication_artifact,
        semantic_adjudication_artifact_sha256=semantic_adjudication_artifact_sha256,
        source_pdf_evidence=source_pdf_evidence,
        generated_at=packet.generated_at,
    )
    if expected_packet.model_dump(mode="json") != packet.model_dump(mode="json"):
        raise ValueError("source transcription adjudication packet binding mismatch")
    decisions = OledSupplementarySourceTranscriptionDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    validate_oled_supplementary_source_transcription_decision_binding(
        review_packet=packet,
        review_packet_sha256=review_packet_sha256,
        decision_manifest=decisions,
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(decisions.reviewed_at) > _parse_timestamp(generated_at_clean):
        raise ValueError("source transcription adjudication predates its decisions")
    decision_by_id = {decision.review_item_id: decision for decision in decisions.decisions}
    results: list[OledSupplementaryAdjudicatedSourceTranscription] = []
    for item in packet.review_items:
        decision = decision_by_id[item.review_item_id]
        accepted = (
            decision.decision
            == OledSupplementarySourceTranscriptionDecision.ACCEPT_BOUNDED_SOURCE_TRANSCRIPTION
        )
        eligible_digests = (
            item.upstream_later_eligible_source_cell_digests if accepted else []
        )
        result = OledSupplementaryAdjudicatedSourceTranscription(
            review_item_id=item.review_item_id,
            review_item_digest=item.review_item_digest,
            scope_id=item.scope_id,
            source_id=item.source_id,
            source_pdf_sha256=item.source_pdf_sha256,
            parsed_document_sha256=item.parsed_document_sha256,
            table_id=item.table_id,
            table_content_digest=item.table_content_digest,
            pdf_page_number_one_based=item.pdf_page_number_one_based,
            header_review_bindings=item.header_review_bindings,
            component_digests=item.component_digests,
            component_results=decision.component_results,
            page_asset_ids=item.page_asset_ids,
            page_asset_digests=item.page_asset_digests,
            full_table_cell_count=item.full_table_cell_count,
            numeric_source_cell_count=item.numeric_source_cell_count,
            source_cell_digests=item.source_cell_digests,
            upstream_later_eligible_cell_count=(
                item.upstream_later_eligible_cell_count
            ),
            upstream_later_eligible_source_cell_digests=(
                item.upstream_later_eligible_source_cell_digests
            ),
            upstream_ontology_review_pending_cell_count=(
                item.upstream_ontology_review_pending_cell_count
            ),
            upstream_source_check_pending_cell_count=(
                item.upstream_source_check_pending_cell_count
            ),
            upstream_rejected_cell_count=item.upstream_rejected_cell_count,
            decision=decision.decision,
            review_note=decision.review_note,
            table_transcription_validated=accepted,
            bounded_selected_table_extent_validated=accepted,
            reparse_required=(
                decision.decision
                == OledSupplementarySourceTranscriptionDecision.NEEDS_REPARSE
            ),
            source_check_pending=(
                decision.decision
                == OledSupplementarySourceTranscriptionDecision.NEEDS_SOURCE_CHECK
            ),
            rejected=(
                decision.decision
                == OledSupplementarySourceTranscriptionDecision.REJECT_SCOPE
            ),
            later_identity_review_eligible_cell_count=len(eligible_digests),
            later_identity_review_eligible_source_cell_digests=eligible_digests,
            source_pdf_remains_authoritative=True,
            parsed_table_is_authoritative=False,
            table_exhaustiveness_validated=False,
            scientific_content_validated=False,
            physical_semantics_validated=False,
            material_identity_resolved=False,
        )
        results.append(result)
    results.sort(key=lambda item: item.review_item_id)
    accepted_count = sum(item.table_transcription_validated for item in results)
    reparse_count = sum(item.reparse_required for item in results)
    source_check_count = sum(item.source_check_pending for item in results)
    rejected_count = sum(item.rejected for item in results)
    unresolved_count = reparse_count + source_check_count
    eligible_cell_count = sum(
        item.later_identity_review_eligible_cell_count for item in results
    )
    semantic_validation = OledSupplementaryScopedCandidateResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    semantic_packet = OledSupplementarySemanticReviewPacket.model_validate(
        semantic_review_packet.model_dump(mode="json")
    )
    semantic_adjudication = OledSupplementarySemanticAdjudicationArtifact.model_validate(
        semantic_adjudication_artifact.model_dump(mode="json")
    )
    payload = {
        "artifact_version": (
            SUPPLEMENTARY_SOURCE_TRANSCRIPTION_ADJUDICATION_ARTIFACT_VERSION
        ),
        "run_id": packet.run_id,
        "paper_id": packet.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": packet.request_artifact_sha256,
        "request_digest": packet.request_digest,
        "response_manifest_sha256": packet.response_manifest_sha256,
        "response_manifest_digest": semantic_validation.response_manifest_digest,
        "response_artifact_sha256": packet.response_artifact_sha256,
        "response_artifact_digest": semantic_validation.response_artifact_digest,
        "semantic_review_packet_sha256": packet.semantic_review_packet_sha256,
        "semantic_review_packet_digest": semantic_packet.review_packet_digest,
        "semantic_decision_manifest_sha256": packet.semantic_decision_manifest_sha256,
        "semantic_decision_manifest_digest": (
            semantic_adjudication.decision_manifest_digest
        ),
        "semantic_adjudication_artifact_sha256": (
            packet.semantic_adjudication_artifact_sha256
        ),
        "semantic_adjudication_artifact_digest": (
            semantic_adjudication.adjudication_artifact_digest
        ),
        "source_pdf_evidence_digest": packet.source_pdf_evidence_digest,
        "review_packet_sha256": _normalize_sha256(
            review_packet_sha256,
            field_name="review_packet_sha256",
        ),
        "review_packet_digest": packet.review_packet_digest,
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": _source_transcription_decision_manifest_digest(
            decisions
        ),
        "reviewed_by": decisions.reviewed_by,
        "reviewed_at": decisions.reviewed_at,
        "status": _source_transcription_adjudication_status(
            accepted_count=accepted_count,
            unresolved_count=unresolved_count,
            eligible_cell_count=eligible_cell_count,
        ),
        "review_item_count": len(results),
        "accepted_scope_count": accepted_count,
        "reparse_required_scope_count": reparse_count,
        "source_check_pending_scope_count": source_check_count,
        "rejected_scope_count": rejected_count,
        "unresolved_review_item_count": unresolved_count,
        "bounded_transcription_validated_cell_count": sum(
            item.numeric_source_cell_count
            for item in results
            if item.table_transcription_validated
        ),
        "later_identity_review_eligible_cell_count": eligible_cell_count,
        "upstream_ontology_review_pending_cell_count": sum(
            item.upstream_ontology_review_pending_cell_count for item in results
        ),
        "upstream_source_check_pending_cell_count": sum(
            item.upstream_source_check_pending_cell_count for item in results
        ),
        "upstream_rejected_cell_count": sum(
            item.upstream_rejected_cell_count for item in results
        ),
        "adjudicated_tables": results,
        "adjudication_artifact_digest": "sha256:" + "0" * 64,
        "review_only": True,
        "offline_only": True,
        "upstream_chain_revalidated": True,
        "review_packet_binding_revalidated": True,
        "source_pdf_evidence_binding_revalidated": True,
        "decision_binding_validated": True,
        "human_decisions_recorded": True,
        "human_source_transcription_review_completed": True,
        "source_pdf_remains_authoritative": True,
        "parsed_table_is_authoritative": False,
        "rendered_pages_are_authoritative": False,
        "all_reviewed_scopes_transcription_validated": (
            accepted_count == len(results)
        ),
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
        "source_pdf_read": True,
        "external_renderer_called": True,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
    }
    provisional = OledSupplementarySourceTranscriptionAdjudicationArtifact.model_construct(
        **payload
    )
    payload["adjudication_artifact_digest"] = (
        _source_transcription_adjudication_digest(provisional)
    )
    return OledSupplementarySourceTranscriptionAdjudicationArtifact.model_validate(
        payload
    )


def render_oled_supplementary_source_transcription_review_markdown(
    packet: OledSupplementarySourceTranscriptionReviewPacket,
    *,
    review_packet_sha256: str = "",
) -> str:
    review = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        packet.model_dump(mode="json")
    )
    packet_sha = (
        _normalize_sha256(review_packet_sha256, field_name="review_packet_sha256")
        if str(review_packet_sha256 or "").strip()
        else "not supplied"
    )
    asset_by_id = {
        asset.asset_id: asset for asset in review.source_pdf_evidence.page_assets
    }
    lines = [
        "# Supplementary bounded source-transcription review",
        "",
        "> Compare the complete parsed table with the displayed page from the exact bound PDF.",
        "> Acceptance attests only bounded selected-table visual equivalence; it does not validate scientific truth, physical semantics, identity, ontology, or dataset admission.",
        "",
        "## Boundary summary",
        "",
        f"- Paper: {_md_code(review.paper_id)}",
        f"- Run ID: {_md_code(review.run_id)}",
        f"- Review items: {review.review_item_count}",
        f"- Full table cells: {review.full_table_cell_count}",
        f"- Numeric source cells: {review.numeric_source_cell_count}",
        f"- PR-I later-eligible cells: {review.upstream_later_eligible_cell_count}",
        f"- PR-I ontology-pending cells: {review.upstream_ontology_review_pending_cell_count}",
        f"- Source PDF SHA-256: {_md_code(review.source_pdf_evidence.source_pdf_sha256)}",
        f"- Page counter: {_md_code(review.source_pdf_evidence.page_counter_id)} {_md_code(review.source_pdf_evidence.page_counter_version)}",
        f"- Page-counter executable SHA-256: {_md_code(review.source_pdf_evidence.page_counter_executable_sha256)}",
        f"- Renderer executable SHA-256: {_md_code(review.source_pdf_evidence.renderer_executable_sha256)}",
        f"- Poppler runtime trust model: {_md_code(review.source_pdf_evidence.poppler_runtime_trust_model)}",
        f"- Dynamic-library closure bound: {_md_code(str(review.source_pdf_evidence.dynamic_library_closure_bound).lower())}",
        f"- Poppler executable evidence digest: {_md_code(review.source_pdf_evidence.poppler_executable_evidence_digest)}",
        f"- Source PDF evidence digest: {_md_code(review.source_pdf_evidence_digest)}",
        f"- Packet digest: {_md_code(review.review_packet_digest)}",
        f"- Exact packet-file SHA-256: {_md_code(packet_sha)}",
        "",
        "## Visual-equivalence contract",
        "",
        f"Contract: {_md_code(SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION)}",
        "",
        "For an accepted item, every component must be `verified_equivalent`: page anchor, caption, ordered source-visible header candidates and their parser-key bindings, row structure, all cell literals, footnotes, and selected-table extent. Line wrapping, non-semantic whitespace, and visually equivalent subscript/superscript markup are allowed. Signs, Unicode symbols, units, digits, decimal places, trailing zeros, row/column order, and footnote content must remain visually exact.",
        "A positional parser key such as `column_1` is displayed only in the separate header-binding list and is never presented as a source literal. Its empty source-header candidate may be accepted only when the corresponding source-page header is visibly blank; otherwise `headers_check` must be `mismatch`.",
        "",
    ]
    ordered_items = sorted(
        review.review_items,
        key=lambda item: (
            item.pdf_page_number_one_based,
            item.table_id,
            item.review_item_id,
        ),
    )
    for index, item in enumerate(ordered_items, start=1):
        lines.extend(
            [
                f"## T{index:02d}: {_md_code(item.table_id)}",
                "",
                f"- Review item: {_md_code(item.review_item_id)}",
                f"- Review-item digest: {_md_code(item.review_item_digest)}",
                f"- Scope: {_md_code(item.scope_id)}",
                f"- Source: {_md_code(item.source_id)}",
                f"- PDF page (one-based): {item.pdf_page_number_one_based}",
                f"- Parsed source bounding box (provenance only): {_md_code(json.dumps(item.matched_table.source_bbox, ensure_ascii=False, sort_keys=True))}",
                "- Evidence image: full PDF page; no source-bbox crop was applied",
                f"- Parsed-document SHA-256: {_md_code(item.parsed_document_sha256)}",
                f"- Table-content digest: {_md_code(item.table_content_digest)}",
                f"- Cells: {item.full_table_cell_count} total / {item.numeric_source_cell_count} numeric",
                "- Allowed decisions: `accept_bounded_source_transcription`, `needs_reparse`, `needs_source_check`, `reject_scope`",
                "",
                "### Exact bound source page",
                "",
            ]
        )
        for asset_id in item.page_asset_ids:
            asset = asset_by_id[asset_id]
            relative_source = f"assets/{asset.asset_filename}"
            lines.extend(
                [
                    f"![Exact bound source PDF page {asset.pdf_page_number_one_based}]({html.escape(relative_source, quote=True)})",
                    "",
                    f"- Page asset: {_md_code(asset.asset_id)}",
                    f"- PNG SHA-256: {_md_code(asset.rendered_asset_sha256)}",
                    f"- Renderer: {_md_code(asset.renderer_id)} {_md_code(asset.renderer_version)} / {_md_code(asset.render_profile)}",
                    "",
                ]
            )
        table = item.matched_table
        rendered_headers = [
            (
                ""
                if binding.source_visible_header_candidate == ""
                else _md_code(binding.source_visible_header_candidate)
            )
            for binding in item.header_review_bindings
        ]
        lines.extend(
            [
                "### Parsed selected table",
                "",
                f"Caption: {_md_code(table.caption)}",
                "",
                "Header bindings below are review candidates, not source claims. A blank candidate is rendered as an empty table-header cell and must be checked against the page image.",
                "",
            ]
        )
        for binding in item.header_review_bindings:
            source_candidate = (
                "<em>[blank candidate]</em>"
                if binding.source_visible_header_candidate == ""
                else _md_code(binding.source_visible_header_candidate)
            )
            lines.append(
                f"- Column {binding.column_index + 1}: source-visible candidate {source_candidate}; parser key {_md_code(binding.parser_key)}; binding {_md_code(binding.binding_kind.value)}"
            )
        lines.extend(
            [
                "",
                "Parsed row grid (headers show only the source-visible candidates):",
                "",
                "| " + " | ".join(rendered_headers) + " |",
                "| " + " | ".join("---" for _ in table.headers) + " |",
            ]
        )
        for row in table.rows:
            lines.append(
                "| "
                + " | ".join(_md_code(row[header]) for header in table.headers)
                + " |"
            )
        lines.extend(["", "Footnotes:", ""])
        if table.footnotes:
            lines.extend(f"- {_md_code(note)}" for note in table.footnotes)
        else:
            lines.append(
                "- `parsed table contains zero footnote entries; verify this against the source page`"
            )
        lines.extend(
            [
                "",
                "### Decision checklist and component bindings",
                "",
                "Record exactly one result for every `*_check` field: `verified_equivalent`, `mismatch`, or `not_checked`.",
                "",
            ]
        )
        for result_field, component_name in zip(
            SOURCE_TRANSCRIPTION_COMPONENT_RESULT_FIELDS,
            SOURCE_TRANSCRIPTION_COMPONENT_NAMES,
            strict=True,
        ):
            lines.append(
                f"- `{result_field}` (`{component_name}`): {_md_code(getattr(item.component_digests, component_name))}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _page_asset_identity(asset: OledSupplementarySourcePageAsset) -> dict[str, Any]:
    return {
        "source_id": asset.source_id,
        "source_pdf_sha256": asset.source_pdf_sha256,
        "pdf_page_number_one_based": asset.pdf_page_number_one_based,
        "renderer_id": asset.renderer_id,
        "renderer_version": asset.renderer_version,
        "render_profile": asset.render_profile,
        "mime_type": asset.mime_type,
        "rendered_asset_sha256": asset.rendered_asset_sha256,
        "rendered_asset_byte_size": asset.rendered_asset_byte_size,
        "pixel_width": asset.pixel_width,
        "pixel_height": asset.pixel_height,
        "exact_source_pdf_bytes_used": asset.exact_source_pdf_bytes_used,
        "full_page_rendered": asset.full_page_rendered,
        "source_bbox_crop_applied": asset.source_bbox_crop_applied,
    }


def _page_asset_suffix(identity: dict[str, Any]) -> str:
    return _stable_hash(identity)[7:31]


def _page_asset_digest(asset: OledSupplementarySourcePageAsset) -> str:
    return _stable_hash(
        {
            **_page_asset_identity(asset),
            "asset_id": asset.asset_id,
            "asset_filename": asset.asset_filename,
        }
    )


def _asset_bundle_digest(
    page_assets: Sequence[OledSupplementarySourcePageAsset],
) -> str:
    return _stable_hash(
        [
            asset.model_dump(mode="json")
            for asset in sorted(
                page_assets,
                key=lambda item: (item.pdf_page_number_one_based, item.asset_id),
            )
        ]
    )


def _poppler_executable_evidence_digest(
    evidence: OledSupplementarySourcePdfEvidence,
) -> str:
    return _stable_hash(
        {
            "runtime_trust_model": evidence.poppler_runtime_trust_model,
            "dynamic_library_closure_bound": evidence.dynamic_library_closure_bound,
            "page_counter": {
                "tool_id": evidence.page_counter_id,
                "tool_version": evidence.page_counter_version,
                "executable_sha256": evidence.page_counter_executable_sha256,
            },
            "renderer": {
                "tool_id": SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDERER_ID,
                "tool_version": (
                    evidence.page_assets[0].renderer_version
                    if evidence.page_assets
                    else evidence.page_counter_version
                ),
                "executable_sha256": evidence.renderer_executable_sha256,
            },
        }
    )


def _source_pdf_evidence_digest(
    evidence: OledSupplementarySourcePdfEvidence,
) -> str:
    payload = evidence.model_dump(mode="json")
    payload.pop("source_pdf_evidence_digest", None)
    return _stable_hash(payload)


def _ordered_table_cells(table: OledSupplementaryReviewTable) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(table.rows):
        if set(row) != set(table.headers):
            raise ValueError("source transcription requires a complete rectangular table")
        for column_index, column_name in enumerate(table.headers):
            cells.append(
                {
                    "row_index": row_index,
                    "column_index": column_index,
                    "column_name": column_name,
                    "cell_value": row[column_name],
                }
            )
    return cells


def _header_review_bindings(
    table: OledSupplementaryReviewTable,
) -> list[OledSupplementarySourceHeaderReviewBinding]:
    bindings: list[OledSupplementarySourceHeaderReviewBinding] = []
    for column_index, parser_key in enumerate(table.headers):
        is_positional_placeholder = parser_key == f"column_{column_index + 1}"
        bindings.append(
            OledSupplementarySourceHeaderReviewBinding(
                column_index=column_index,
                parser_key=parser_key,
                source_visible_header_candidate=(
                    "" if is_positional_placeholder else parser_key
                ),
                binding_kind=(
                    OledSupplementarySourceHeaderBindingKind.PARSER_PLACEHOLDER_CANDIDATE_FOR_BLANK_HEADER
                    if is_positional_placeholder
                    else OledSupplementarySourceHeaderBindingKind.REPORTED_LITERAL
                ),
            )
        )
    return bindings


def _validate_standalone_header_review_bindings(
    bindings: Sequence[OledSupplementarySourceHeaderReviewBinding],
) -> None:
    if not bindings:
        raise ValueError("source transcription header-review bindings are required")
    if [binding.column_index for binding in bindings] != list(range(len(bindings))):
        raise ValueError(
            "source transcription header-review bindings must cover every column in order"
        )
    parser_keys = [binding.parser_key for binding in bindings]
    if len(parser_keys) != len(set(parser_keys)):
        raise ValueError("source transcription header-review parser keys must be unique")


def _headers_component_evidence(
    bindings: Sequence[OledSupplementarySourceHeaderReviewBinding],
) -> dict[str, Any]:
    _validate_standalone_header_review_bindings(bindings)
    return {
        "parser_headers": [binding.parser_key for binding in bindings],
        "header_review_bindings": [
            binding.model_dump(mode="json") for binding in bindings
        ],
    }


def _headers_component_digest(
    bindings: Sequence[OledSupplementarySourceHeaderReviewBinding],
) -> str:
    return _stable_hash(
        {
            "contract_version": (
                SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION
            ),
            "component": "headers",
            "evidence": _headers_component_evidence(bindings),
        }
    )


def _component_digests_from_payload(
    payload: dict[str, Any],
) -> OledSupplementarySourceTranscriptionComponentDigests:
    table_value = payload["matched_table"]
    table = (
        table_value
        if isinstance(table_value, OledSupplementaryReviewTable)
        else OledSupplementaryReviewTable.model_validate(table_value)
    )
    header_bindings = [
        value
        if isinstance(value, OledSupplementarySourceHeaderReviewBinding)
        else OledSupplementarySourceHeaderReviewBinding.model_validate(value)
        for value in payload["header_review_bindings"]
    ]
    if header_bindings != _header_review_bindings(table):
        raise ValueError(
            "source transcription header-review bindings do not match the table"
        )
    page_asset_ids = list(payload["page_asset_ids"])
    page_asset_digests = list(payload["page_asset_digests"])
    component_payloads = {
        "page_anchor": {
            "source_id": payload["source_id"],
            "source_pdf_sha256": payload["source_pdf_sha256"],
            "pdf_page_number_one_based": payload["pdf_page_number_one_based"],
            "source_bbox": table.source_bbox,
            "page_asset_ids": page_asset_ids,
            "page_asset_digests": page_asset_digests,
        },
        "caption": {"caption": table.caption},
        "headers": _headers_component_evidence(header_bindings),
        "row_structure": {
            "row_count": table.row_count,
            "column_count": table.column_count,
            "row_keys": [list(table.headers) for _ in table.rows],
            "subject_column_name": table.headers[0],
            "ordered_subject_labels": [
                row[table.headers[0]] for row in table.rows
            ],
        },
        "cell_literals": {"cells": _ordered_table_cells(table)},
        "footnotes": {"footnotes": table.footnotes},
        "table_extent": {
            "table_id": table.table_id,
            "table_content_digest": table.table_content_digest,
            "row_count": table.row_count,
            "column_count": table.column_count,
            "full_table_cell_count": payload["full_table_cell_count"],
            "numeric_source_cell_count": payload["numeric_source_cell_count"],
        },
    }
    return OledSupplementarySourceTranscriptionComponentDigests(
        **{
            name: _stable_hash(
                {
                    "contract_version": (
                        SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION
                    ),
                    "component": name,
                    "evidence": component_payloads[name],
                }
            )
            for name in SOURCE_TRANSCRIPTION_COMPONENT_NAMES
        }
    )


def _component_digests_for_item(
    item: OledSupplementarySourceTranscriptionReviewItem,
) -> OledSupplementarySourceTranscriptionComponentDigests:
    return _component_digests_from_payload(item.model_dump(mode="json"))


def _source_transcription_review_item_identity(
    item: OledSupplementarySourceTranscriptionReviewItem,
) -> dict[str, Any]:
    return {
        "item_kind": item.item_kind.value,
        "visual_equivalence_contract_version": item.visual_equivalence_contract_version,
        "scope_id": item.scope_id,
        "source_id": item.source_id,
        "source_pdf_sha256": item.source_pdf_sha256,
        "parsed_document_sha256": item.parsed_document_sha256,
        "table_id": item.table_id,
        "table_content_digest": item.table_content_digest,
        "pdf_page_number_one_based": item.pdf_page_number_one_based,
        "component_digests": item.component_digests.model_dump(mode="json"),
        "page_asset_ids": item.page_asset_ids,
        "page_asset_digests": item.page_asset_digests,
        "source_cell_digests": item.source_cell_digests,
        "upstream_later_eligible_source_cell_digests": (
            item.upstream_later_eligible_source_cell_digests
        ),
        "upstream_ontology_review_pending_source_cell_digests": (
            item.upstream_ontology_review_pending_source_cell_digests
        ),
        "upstream_source_check_pending_source_cell_digests": (
            item.upstream_source_check_pending_source_cell_digests
        ),
        "upstream_rejected_source_cell_digests": (
            item.upstream_rejected_source_cell_digests
        ),
        "upstream_exclusion_confirmed_source_cell_digests": (
            item.upstream_exclusion_confirmed_source_cell_digests
        ),
        "upstream_blocked_by_scope_semantics_source_cell_digests": (
            item.upstream_blocked_by_scope_semantics_source_cell_digests
        ),
    }


def _source_transcription_review_item_id(identity: dict[str, Any]) -> str:
    return f"supplementary-source-transcription:{_stable_hash(identity)[7:31]}"


def _source_transcription_review_item_digest(
    item: OledSupplementarySourceTranscriptionReviewItem,
) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("review_item_digest", None)
    return _stable_hash(payload)


def _source_transcription_review_packet_digest(
    packet: OledSupplementarySourceTranscriptionReviewPacket,
) -> str:
    payload = packet.model_dump(mode="json")
    payload.pop("review_packet_digest", None)
    return _stable_hash(payload)


def _component_result_values(
    results: OledSupplementarySourceTranscriptionComponentResults,
) -> list[OledSupplementarySourceTranscriptionComponentResult]:
    return [
        getattr(results, field_name)
        for field_name in SOURCE_TRANSCRIPTION_COMPONENT_RESULT_FIELDS
    ]


def _validate_source_transcription_decision_components(
    *,
    decision: OledSupplementarySourceTranscriptionDecision,
    component_results: OledSupplementarySourceTranscriptionComponentResults,
    review_note: str,
) -> None:
    results = _component_result_values(component_results)
    mismatch_count = results.count(
        OledSupplementarySourceTranscriptionComponentResult.MISMATCH
    )
    not_checked_count = results.count(
        OledSupplementarySourceTranscriptionComponentResult.NOT_CHECKED
    )
    verified_count = results.count(
        OledSupplementarySourceTranscriptionComponentResult.VERIFIED_EQUIVALENT
    )
    if (
        decision
        == OledSupplementarySourceTranscriptionDecision.ACCEPT_BOUNDED_SOURCE_TRANSCRIPTION
    ):
        if verified_count != len(SOURCE_TRANSCRIPTION_COMPONENT_NAMES):
            raise ValueError(
                "accepted source transcription requires every component verified"
            )
    elif decision == OledSupplementarySourceTranscriptionDecision.NEEDS_REPARSE:
        if (
            component_results.page_anchor_check
            != OledSupplementarySourceTranscriptionComponentResult.VERIFIED_EQUIVALENT
            or not mismatch_count
            or not_checked_count
        ):
            raise ValueError(
                "needs_reparse requires a verified page anchor, a later-component "
                "mismatch, and no unchecked component"
            )
    elif decision == OledSupplementarySourceTranscriptionDecision.NEEDS_SOURCE_CHECK:
        if not not_checked_count or mismatch_count:
            raise ValueError(
                "needs_source_check requires an unchecked component and no mismatch"
            )
    if (
        decision
        != OledSupplementarySourceTranscriptionDecision.ACCEPT_BOUNDED_SOURCE_TRANSCRIPTION
        and not review_note
    ):
        raise ValueError("non-success source transcription decision requires review_note")


def _source_transcription_decision_manifest_digest(
    manifest: OledSupplementarySourceTranscriptionDecisionManifest,
) -> str:
    payload = manifest.model_dump(mode="json")
    payload["decisions"] = sorted(
        payload["decisions"],
        key=lambda decision: decision["review_item_id"],
    )
    return _stable_hash(payload)


def _source_transcription_adjudication_status(
    *,
    accepted_count: int,
    unresolved_count: int,
    eligible_cell_count: int,
) -> OledSupplementarySourceTranscriptionAdjudicationStatus:
    if unresolved_count:
        return (
            OledSupplementarySourceTranscriptionAdjudicationStatus.REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS
        )
    if accepted_count and eligible_cell_count:
        return (
            OledSupplementarySourceTranscriptionAdjudicationStatus.READY_FOR_LATER_IDENTITY_REVIEW
        )
    return (
        OledSupplementarySourceTranscriptionAdjudicationStatus.REVIEW_COMPLETE_NO_ELIGIBLE_SCOPES
    )


def _source_transcription_adjudication_digest(
    artifact: OledSupplementarySourceTranscriptionAdjudicationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("adjudication_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_ADJUDICATION_ARTIFACT_VERSION",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_DECISION_MANIFEST_VERSION",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_PAGE_COUNTER_ID",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_POPPLER_RUNTIME_TRUST_MODEL",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDERER_ID",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_RENDER_PROFILE",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_REVIEW_PACKET_VERSION",
    "SUPPLEMENTARY_SOURCE_TRANSCRIPTION_VISUAL_EQUIVALENCE_CONTRACT_VERSION",
    "SOURCE_TRANSCRIPTION_COMPONENT_NAMES",
    "SOURCE_TRANSCRIPTION_COMPONENT_RESULT_FIELDS",
    "SOURCE_TRANSCRIPTION_COMPONENT_RESULT_TO_COMPONENT",
    "OledSupplementaryAdjudicatedSourceTranscription",
    "OledSupplementarySourceHeaderBindingKind",
    "OledSupplementarySourceHeaderReviewBinding",
    "OledSupplementarySourcePageAsset",
    "OledSupplementarySourcePdfEvidence",
    "OledSupplementarySourceTranscriptionAdjudicationArtifact",
    "OledSupplementarySourceTranscriptionAdjudicationStatus",
    "OledSupplementarySourceTranscriptionComponentDigests",
    "OledSupplementarySourceTranscriptionComponentResult",
    "OledSupplementarySourceTranscriptionComponentResults",
    "OledSupplementarySourceTranscriptionDecision",
    "OledSupplementarySourceTranscriptionDecisionEntry",
    "OledSupplementarySourceTranscriptionDecisionManifest",
    "OledSupplementarySourceTranscriptionReviewItem",
    "OledSupplementarySourceTranscriptionReviewItemKind",
    "OledSupplementarySourceTranscriptionReviewPacket",
    "OledSupplementarySourceTranscriptionReviewPacketStatus",
    "build_oled_supplementary_source_page_asset",
    "build_oled_supplementary_source_pdf_evidence",
    "build_oled_supplementary_source_transcription_adjudication_artifact",
    "build_oled_supplementary_source_transcription_review_packet",
    "render_oled_supplementary_source_transcription_review_markdown",
    "validate_oled_supplementary_source_transcription_decision_binding",
    "validate_oled_supplementary_source_transcription_inputs",
]
