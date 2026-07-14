from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from datetime import datetime
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

from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityAmbiguousIdentity,
    OledSupplementaryMaterialIdentityCandidateCollisionFinding,
    OledSupplementaryMaterialIdentityEvidenceAnchor,
    OledSupplementaryMaterialIdentityEvidenceDispositionKind,
    OledSupplementaryMaterialIdentityEvidenceProducer,
    OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    OledSupplementaryMaterialIdentityExcludeIdentityGroup,
    OledSupplementaryMaterialIdentityNeedsSourceCheck,
    OledSupplementaryMaterialIdentityProposeStructureCandidate,
    OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
    OledSupplementaryMaterialIdentityValidatedEvidenceResult,
    build_oled_supplementary_material_identity_evidence_response_artifact,
    oled_supplementary_material_identity_evidence_anchor_digest,
    oled_supplementary_material_identity_evidence_response_artifact_digest,
    oled_supplementary_material_identity_evidence_response_manifest_digest,
    validate_oled_supplementary_material_identity_evidence_response_binding,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    validate_oled_supplementary_safe_authored_text,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourcePageAsset,
    OledSupplementarySourcePdfEvidence,
    OledSupplementarySourceTranscriptionReviewPacket,
)


SUPPLEMENTARY_MATERIAL_IDENTITY_REVIEW_PACKET_VERSION = (
    "oled_supplementary_material_identity_review_packet.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_DECISION_MANIFEST_VERSION = (
    "oled_supplementary_material_identity_decision_manifest.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_ADJUDICATION_ARTIFACT_VERSION = (
    "oled_supplementary_material_identity_adjudication.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_SOURCE_MATCH_CONTRACT_VERSION = (
    "oled_supplementary_material_identity_source_match.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE = (
    "rdkit_prepare_and_draw_2d_stereo_800x600_rgb_png.v1"
)

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")


class OledSupplementaryMaterialIdentityReviewItemKind(str, Enum):
    MATERIAL_IDENTITY_GROUP = "material_identity_group"


class OledSupplementaryMaterialIdentityReviewPageRole(str, Enum):
    ROW_CONTEXT = "row_context"
    IDENTITY_EVIDENCE = "identity_evidence"


class OledSupplementaryMaterialIdentityReviewPacketStatus(str, Enum):
    READY_FOR_HUMAN_MATERIAL_IDENTITY_REVIEW = (
        "ready_for_human_material_identity_review"
    )


class OledSupplementaryMaterialIdentityAnchorAssessmentResult(str, Enum):
    SUPPORTS_CLAIM = "supports_claim"
    DOES_NOT_SUPPORT_CLAIM = "does_not_support_claim"
    NOT_CHECKED = "not_checked"


class OledSupplementaryMaterialIdentityCandidateSourceMatchResult(str, Enum):
    MATCHES_SOURCE = "matches_source"
    DOES_NOT_MATCH_SOURCE = "does_not_match_source"
    NOT_CHECKED = "not_checked"
    NOT_APPLICABLE = "not_applicable"


class OledSupplementaryMaterialIdentityReviewDecision(str, Enum):
    ACCEPT_STRUCTURE_CANDIDATE = "accept_structure_candidate"
    ACCEPT_STRUCTURE_ANCHOR_ONLY = "accept_structure_anchor_only"
    CONFIRM_SOURCE_CHECK = "confirm_source_check"
    CONFIRM_AMBIGUOUS_IDENTITY = "confirm_ambiguous_identity"
    ACCEPT_IDENTITY_EXCLUSION = "accept_identity_exclusion"
    NEEDS_SOURCE_CHECK = "needs_source_check"
    REJECT_RESPONSE_EVIDENCE = "reject_response_evidence"


class OledSupplementaryMaterialIdentityAdjudicationStatus(str, Enum):
    READY_FOR_LATER_MATERIAL_IDENTITY_REGISTRY_REVIEW = (
        "ready_for_later_material_identity_registry_review"
    )
    REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS = (
        "review_complete_with_unresolved_items"
    )
    REVIEW_COMPLETE_NO_REGISTRY_ELIGIBLE_IDENTITIES = (
        "review_complete_no_registry_eligible_identities"
    )


class OledSupplementaryMaterialIdentityCandidateDepictionAsset(BaseModel):
    """A deterministic, non-authoritative review projection of one candidate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    asset_id: str
    asset_filename: str
    validated_result_id: str
    candidate_digest: str
    toolkit_id: Literal["rdkit"] = "rdkit"
    toolkit_version: str
    depiction_profile: str = SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE
    mime_type: Literal["image/png"] = "image/png"
    rendered_asset_sha256: str
    rendered_asset_byte_size: Annotated[StrictInt, Field(ge=1, le=50_000_000)]
    pixel_width: Annotated[StrictInt, Field(ge=1, le=20_000)]
    pixel_height: Annotated[StrictInt, Field(ge=1, le=20_000)]
    depiction_asset_digest: str
    exact_candidate_graph_used: StrictBool = True
    deterministic_2d_coordinates_requested: StrictBool = True
    non_authoritative_review_projection: StrictBool = True
    material_identity_resolved: StrictBool = False

    @field_validator("asset_id", "validated_result_id", "toolkit_version")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("asset_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="asset_filename")

    @field_validator(
        "candidate_digest",
        "rendered_asset_sha256",
        "depiction_asset_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("depiction_profile")
    @classmethod
    def validate_depiction_profile(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE:
            raise ValueError("unexpected material identity depiction profile")
        return value

    @model_validator(mode="after")
    def validate_asset_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityCandidateDepictionAsset:
        suffix = _depiction_asset_suffix(self)
        if self.asset_id != f"material-identity-candidate:{suffix}":
            raise ValueError("material identity depiction asset_id mismatch")
        if self.asset_filename != f"identity-candidate-{suffix}.png":
            raise ValueError("material identity depiction filename mismatch")
        if self.pixel_width != 800 or self.pixel_height != 600:
            raise ValueError("material identity depiction profile requires 800x600 PNG")
        if not (
            self.exact_candidate_graph_used
            and self.deterministic_2d_coordinates_requested
            and self.non_authoritative_review_projection
        ):
            raise ValueError("material identity depiction lost a required audit flag")
        if self.material_identity_resolved:
            raise ValueError("candidate depiction cannot resolve material identity")
        if (
            oled_supplementary_material_identity_candidate_depiction_asset_digest(
                self
            )
            != self.depiction_asset_digest
        ):
            raise ValueError("material identity depiction digest mismatch")
        return self


class OledSupplementaryMaterialIdentityReviewPageReference(BaseModel):
    """One role-specific reference to a packet-level full-page asset."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    page_role: OledSupplementaryMaterialIdentityReviewPageRole
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    page_asset_id: str
    page_asset_digest: str
    evidence_anchor_digests: list[str] = Field(default_factory=list, max_length=64)
    page_reference_digest: str

    @field_validator("page_asset_id")
    @classmethod
    def validate_asset_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="page_asset_id")

    @field_validator("page_asset_digest", "page_reference_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("evidence_anchor_digests")
    @classmethod
    def validate_anchor_digests(cls, value: list[str]) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name="evidence_anchor_digest")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("page-reference anchor digests must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_reference_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityReviewPageReference:
        if (
            self.page_role
            == OledSupplementaryMaterialIdentityReviewPageRole.ROW_CONTEXT
            and self.evidence_anchor_digests
        ):
            raise ValueError("row-context pages must not claim identity anchors")
        if (
            self.page_role
            == OledSupplementaryMaterialIdentityReviewPageRole.IDENTITY_EVIDENCE
            and not self.evidence_anchor_digests
        ):
            raise ValueError("identity-evidence pages require anchor bindings")
        if _page_reference_digest(self) != self.page_reference_digest:
            raise ValueError("material identity page-reference digest mismatch")
        return self


class OledSupplementaryMaterialIdentityReviewItem(BaseModel):
    """One exact PR-L group with row context and claimed identity evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: Literal[
        OledSupplementaryMaterialIdentityReviewItemKind.MATERIAL_IDENTITY_GROUP
    ] = OledSupplementaryMaterialIdentityReviewItemKind.MATERIAL_IDENTITY_GROUP
    source_match_contract_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_SOURCE_MATCH_CONTRACT_VERSION
    )
    validated_result: OledSupplementaryMaterialIdentityValidatedEvidenceResult
    row_context_page: OledSupplementaryMaterialIdentityReviewPageReference
    identity_evidence_pages: list[
        OledSupplementaryMaterialIdentityReviewPageReference
    ] = Field(default_factory=list, max_length=64)
    candidate_depiction_asset: (
        OledSupplementaryMaterialIdentityCandidateDepictionAsset | None
    ) = None
    related_collision_findings: list[
        OledSupplementaryMaterialIdentityCandidateCollisionFinding
    ] = Field(default_factory=list)
    source_pdf_remains_authoritative: StrictBool = True
    row_context_is_identity_evidence: StrictBool = False
    candidate_depiction_is_authoritative: StrictBool = False
    human_identity_review_completed: StrictBool = False
    source_to_candidate_match_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    automatic_candidate_merge: StrictBool = False

    @field_validator("review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="review_item_id")

    @field_validator("review_item_digest")
    @classmethod
    def validate_review_item_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="review_item_digest")

    @field_validator("source_match_contract_version")
    @classmethod
    def validate_contract_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_SOURCE_MATCH_CONTRACT_VERSION:
            raise ValueError("unexpected material identity source-match contract")
        return value

    @field_validator("identity_evidence_pages")
    @classmethod
    def validate_identity_page_order(
        cls,
        value: list[OledSupplementaryMaterialIdentityReviewPageReference],
    ) -> list[OledSupplementaryMaterialIdentityReviewPageReference]:
        order = [item.pdf_page_number_one_based for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("identity-evidence page references must be ordered and unique")
        return value

    @field_validator("related_collision_findings")
    @classmethod
    def validate_collision_order(
        cls,
        value: list[OledSupplementaryMaterialIdentityCandidateCollisionFinding],
    ) -> list[OledSupplementaryMaterialIdentityCandidateCollisionFinding]:
        order = [item.finding_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("related collision findings must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_item_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityReviewItem:
        result = self.validated_result
        group = result.bound_identity_group
        response = result.response_result
        if self.review_item_id != _review_item_id(result):
            raise ValueError("material identity review item_id mismatch")
        if (
            self.row_context_page.page_role
            != OledSupplementaryMaterialIdentityReviewPageRole.ROW_CONTEXT
            or self.row_context_page.pdf_page_number_one_based
            != group.pdf_page_number_one_based
        ):
            raise ValueError("material identity row-context page mismatch")
        anchors = _result_anchors(response)
        anchors_by_page = _anchor_digests_by_page(anchors)
        observed_by_page = {
            ref.pdf_page_number_one_based: ref.evidence_anchor_digests
            for ref in self.identity_evidence_pages
        }
        if observed_by_page != anchors_by_page:
            raise ValueError("material identity evidence-page partition mismatch")
        if any(
            ref.page_role
            != OledSupplementaryMaterialIdentityReviewPageRole.IDENTITY_EVIDENCE
            for ref in self.identity_evidence_pages
        ):
            raise ValueError("identity evidence uses the wrong page role")
        is_candidate = isinstance(
            response,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        )
        if is_candidate != (self.candidate_depiction_asset is not None):
            raise ValueError("candidate depiction presence does not match disposition")
        if self.candidate_depiction_asset is not None:
            chemistry = result.chemistry_validation
            if chemistry is None:
                raise ValueError("candidate depiction requires chemistry validation")
            if (
                self.candidate_depiction_asset.validated_result_id
                != result.validated_result_id
                or self.candidate_depiction_asset.candidate_digest
                != chemistry.candidate_digest
            ):
                raise ValueError("candidate depiction is bound to the wrong candidate")
        group_id = group.identity_group_id
        if any(
            group_id not in finding.identity_group_ids
            for finding in self.related_collision_findings
        ):
            raise ValueError("review item contains an unrelated collision finding")
        fixed_true = ("source_pdf_remains_authoritative",)
        fixed_false = (
            "row_context_is_identity_evidence",
            "candidate_depiction_is_authoritative",
            "human_identity_review_completed",
            "source_to_candidate_match_validated",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "automatic_candidate_merge",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("material identity review item crossed its boundary")
        if self.review_item_digest != _review_item_digest(self):
            raise ValueError("material identity review item digest mismatch")
        return self


class OledSupplementaryMaterialIdentityReviewPacket(BaseModel):
    """A complete, exact-bound, PDF-backed human-review packet."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = SUPPLEMENTARY_MATERIAL_IDENTITY_REVIEW_PACKET_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    material_identity_request_digest: str
    transcription_review_packet_sha256: str
    transcription_review_packet_digest: str
    upstream_source_pdf_evidence_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    response_artifact_sha256: str
    response_artifact_digest: str
    response_generated_at: str
    response_producer: OledSupplementaryMaterialIdentityEvidenceProducer
    review_source_pdf_evidence: OledSupplementarySourcePdfEvidence
    review_source_pdf_evidence_digest: str
    status: OledSupplementaryMaterialIdentityReviewPacketStatus
    identity_group_count: Annotated[StrictInt, Field(ge=1)]
    review_item_count: Annotated[StrictInt, Field(ge=1)]
    identity_dependent_cell_count: Annotated[StrictInt, Field(ge=1)]
    bounded_transcription_validated_cell_count: Annotated[StrictInt, Field(ge=1)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    structure_candidate_count: Annotated[StrictInt, Field(ge=0)]
    structure_anchor_only_count: Annotated[StrictInt, Field(ge=0)]
    source_check_count: Annotated[StrictInt, Field(ge=0)]
    ambiguous_identity_count: Annotated[StrictInt, Field(ge=0)]
    exclusion_proposal_count: Annotated[StrictInt, Field(ge=0)]
    evidence_anchor_count: Annotated[StrictInt, Field(ge=0)]
    cited_source_page_count: Annotated[StrictInt, Field(ge=1)]
    candidate_depiction_asset_count: Annotated[StrictInt, Field(ge=0)]
    collision_finding_count: Annotated[StrictInt, Field(ge=0)]
    collision_findings: list[
        OledSupplementaryMaterialIdentityCandidateCollisionFinding
    ] = Field(default_factory=list)
    review_items: list[OledSupplementaryMaterialIdentityReviewItem] = Field(
        min_length=1
    )
    review_packet_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    exact_source_pdf_bytes_verified: StrictBool = True
    source_page_assets_rendered_from_bound_pdf: StrictBool = True
    complete_identity_group_coverage_validated: StrictBool = True
    complete_dependent_cell_coverage_validated: StrictBool = True
    complete_evidence_anchor_coverage_validated: StrictBool = True
    row_context_and_identity_evidence_separated: StrictBool = True
    candidate_depiction_required_for_structure_candidates: StrictBool = True
    complete_candidate_depiction_coverage_validated: StrictBool = True
    joint_exact_input_revalidation_required: StrictBool = True
    standalone_upstream_chain_revalidation_supported: StrictBool = False
    standalone_source_pdf_bytes_revalidation_supported: StrictBool = False
    standalone_page_asset_bytes_revalidation_supported: StrictBool = False
    standalone_candidate_depiction_bytes_revalidation_supported: StrictBool = False
    human_identity_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    source_pdf_read: StrictBool = True
    external_renderer_called: StrictBool = True
    candidate_depiction_renderer_called: StrictBool
    human_identity_review_completed: StrictBool = False
    source_to_candidate_match_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    registry_written: StrictBool = False
    schema_candidates_created: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    training_eligible: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_REVIEW_PACKET_VERSION:
            raise ValueError("unexpected material identity review packet version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at", "response_generated_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "material_identity_request_digest",
        "transcription_review_packet_sha256",
        "transcription_review_packet_digest",
        "upstream_source_pdf_evidence_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_sha256",
        "response_artifact_digest",
        "review_source_pdf_evidence_digest",
        "review_packet_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("collision_findings")
    @classmethod
    def validate_collision_order(
        cls,
        value: list[OledSupplementaryMaterialIdentityCandidateCollisionFinding],
    ) -> list[OledSupplementaryMaterialIdentityCandidateCollisionFinding]:
        order = [item.finding_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("packet collision findings must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_packet_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityReviewPacket:
        if self.status != (
            OledSupplementaryMaterialIdentityReviewPacketStatus
            .READY_FOR_HUMAN_MATERIAL_IDENTITY_REVIEW
        ):
            raise ValueError("material identity review packet status mismatch")
        if _parse_timestamp(self.response_generated_at) > _parse_timestamp(
            self.generated_at
        ):
            raise ValueError("material identity review packet predates PR-L")
        if (
            self.review_source_pdf_evidence_digest
            != self.review_source_pdf_evidence.source_pdf_evidence_digest
        ):
            raise ValueError("review source PDF evidence digest mismatch")
        item_order = [
            (
                item.validated_result.bound_identity_group.scope_id,
                item.validated_result.bound_identity_group.table_id,
                item.validated_result.bound_identity_group.row_index,
                item.validated_result.bound_identity_group.identity_group_id,
            )
            for item in self.review_items
        ]
        if item_order != sorted(item_order):
            raise ValueError("material identity review items must be sorted")
        group_ids = [
            item.validated_result.bound_identity_group.identity_group_id
            for item in self.review_items
        ]
        result_ids = [
            item.validated_result.validated_result_id for item in self.review_items
        ]
        if (
            self.identity_group_count != len(self.review_items)
            or self.review_item_count != len(self.review_items)
            or len(group_ids) != len(set(group_ids))
            or len(result_ids) != len(set(result_ids))
        ):
            raise ValueError("material identity review group coverage mismatch")
        cells = [
            digest
            for item in self.review_items
            for digest in (
                item.validated_result.bound_identity_group
                .identity_dependent_source_cell_digests
            )
        ]
        if (
            self.identity_dependent_cell_count != len(cells)
            or len(cells) != len(set(cells))
        ):
            raise ValueError("material identity review cell coverage mismatch")
        if (
            self.identity_dependent_cell_count
            + self.upstream_ontology_review_pending_cell_count
            > self.bounded_transcription_validated_cell_count
        ):
            raise ValueError("material identity review cell partition is impossible")
        if self.device_only_cell_count != 0:
            raise ValueError("device-only cells must remain outside identity review")
        disposition_counts = {
            kind: sum(
                item.validated_result.response_result.disposition == kind
                for item in self.review_items
            )
            for kind in OledSupplementaryMaterialIdentityEvidenceDispositionKind
        }
        expected_counts = {
            OledSupplementaryMaterialIdentityEvidenceDispositionKind
            .PROPOSE_STRUCTURE_CANDIDATE: self.structure_candidate_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind
            .RECORD_STRUCTURE_ANCHOR_ONLY: self.structure_anchor_only_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind
            .NEEDS_SOURCE_CHECK: self.source_check_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind
            .AMBIGUOUS_IDENTITY: self.ambiguous_identity_count,
            OledSupplementaryMaterialIdentityEvidenceDispositionKind
            .EXCLUDE_IDENTITY_GROUP: self.exclusion_proposal_count,
        }
        if disposition_counts != expected_counts:
            raise ValueError("material identity review disposition counts mismatch")
        anchors = [
            anchor
            for item in self.review_items
            for anchor in _result_anchors(item.validated_result.response_result)
        ]
        if self.evidence_anchor_count != len(anchors):
            raise ValueError("material identity review anchor count mismatch")
        assets_by_id = {
            asset.asset_id: asset
            for asset in self.review_source_pdf_evidence.page_assets
        }
        if len(assets_by_id) != len(self.review_source_pdf_evidence.page_assets):
            raise ValueError("material identity review repeats a source page asset")
        expected_pages = {
            item.validated_result.bound_identity_group.pdf_page_number_one_based
            for item in self.review_items
        } | {
            anchor.pdf_page_number_one_based for anchor in anchors
        }
        observed_pages = {
            asset.pdf_page_number_one_based for asset in assets_by_id.values()
        }
        if observed_pages != expected_pages:
            raise ValueError("review page assets do not exactly cover context and anchors")
        for item in self.review_items:
            references = [item.row_context_page, *item.identity_evidence_pages]
            for reference in references:
                asset = assets_by_id.get(reference.page_asset_id)
                if asset is None or (
                    asset.page_asset_digest != reference.page_asset_digest
                    or asset.pdf_page_number_one_based
                    != reference.pdf_page_number_one_based
                ):
                    raise ValueError("review item references the wrong source page asset")
            result = item.validated_result
            if (
                result.bound_identity_group.source_id
                != self.review_source_pdf_evidence.source_id
                or result.bound_identity_group.source_pdf_sha256
                != self.review_source_pdf_evidence.source_pdf_sha256
                or result.source_pdf_page_count
                != self.review_source_pdf_evidence.source_pdf_page_count
            ):
                raise ValueError("review item source PDF binding mismatch")
        depictions = [
            item.candidate_depiction_asset
            for item in self.review_items
            if item.candidate_depiction_asset is not None
        ]
        if (
            self.candidate_depiction_asset_count != len(depictions)
            or self.candidate_depiction_asset_count != self.structure_candidate_count
            or self.candidate_depiction_renderer_called != bool(depictions)
        ):
            raise ValueError("material identity depiction coverage mismatch")
        depiction_ids = [asset.asset_id for asset in depictions]
        depiction_filenames = [asset.asset_filename for asset in depictions]
        if len(depiction_ids) != len(set(depiction_ids)) or len(
            depiction_filenames
        ) != len(set(depiction_filenames)):
            raise ValueError("material identity depiction assets must be unique")
        if self.cited_source_page_count != len(observed_pages):
            raise ValueError("material identity cited-page count mismatch")
        if self.collision_finding_count != len(self.collision_findings):
            raise ValueError("material identity collision count mismatch")
        findings_by_digest = {
            finding.finding_digest: finding for finding in self.collision_findings
        }
        for item in self.review_items:
            group_id = item.validated_result.bound_identity_group.identity_group_id
            expected = sorted(
                (
                    finding
                    for finding in self.collision_findings
                    if group_id in finding.identity_group_ids
                ),
                key=lambda finding: finding.finding_digest,
            )
            if [finding.model_dump(mode="json") for finding in expected] != [
                finding.model_dump(mode="json")
                for finding in item.related_collision_findings
            ]:
                raise ValueError("review item collision roster mismatch")
        if len(findings_by_digest) != len(self.collision_findings):
            raise ValueError("material identity packet repeats a collision finding")
        fixed_true = (
            "review_only",
            "offline_only",
            "exact_source_pdf_bytes_verified",
            "source_page_assets_rendered_from_bound_pdf",
            "complete_identity_group_coverage_validated",
            "complete_dependent_cell_coverage_validated",
            "complete_evidence_anchor_coverage_validated",
            "row_context_and_identity_evidence_separated",
            "candidate_depiction_required_for_structure_candidates",
            "complete_candidate_depiction_coverage_validated",
            "joint_exact_input_revalidation_required",
            "human_identity_review_required",
            "source_pdf_remains_authoritative",
            "source_pdf_read",
            "external_renderer_called",
        )
        fixed_false = (
            "standalone_upstream_chain_revalidation_supported",
            "standalone_source_pdf_bytes_revalidation_supported",
            "standalone_page_asset_bytes_revalidation_supported",
            "standalone_candidate_depiction_bytes_revalidation_supported",
            "human_identity_review_completed",
            "source_to_candidate_match_validated",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
            "registry_written",
            "schema_candidates_created",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "training_eligible",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("material identity review packet crossed its boundary")
        if (
            oled_supplementary_material_identity_review_packet_digest(self)
            != self.review_packet_digest
        ):
            raise ValueError("material identity review packet digest mismatch")
        return self


class OledSupplementaryMaterialIdentityAnchorAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    evidence_anchor_digest: str
    assessment: OledSupplementaryMaterialIdentityAnchorAssessmentResult
    review_note: str = ""

    @field_validator("evidence_anchor_digest")
    @classmethod
    def validate_anchor_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="evidence_anchor_digest")

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="anchor review_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_note_requirement(
        self,
    ) -> OledSupplementaryMaterialIdentityAnchorAssessment:
        if (
            self.assessment
            != OledSupplementaryMaterialIdentityAnchorAssessmentResult.SUPPORTS_CLAIM
            and not self.review_note
        ):
            raise ValueError("non-supporting anchor assessment requires review_note")
        return self


class OledSupplementaryMaterialIdentityDecisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item_id: str
    review_item_digest: str
    item_kind: Literal[
        OledSupplementaryMaterialIdentityReviewItemKind.MATERIAL_IDENTITY_GROUP
    ]
    decision: OledSupplementaryMaterialIdentityReviewDecision
    anchor_assessments: list[OledSupplementaryMaterialIdentityAnchorAssessment] = (
        Field(default_factory=list, max_length=64)
    )
    candidate_source_match: OledSupplementaryMaterialIdentityCandidateSourceMatchResult
    reviewed_collision_finding_digests: list[str] = Field(default_factory=list)
    review_note: str = ""

    @field_validator("review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="review_item_id")

    @field_validator("review_item_digest")
    @classmethod
    def validate_review_item_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="review_item_digest")

    @field_validator("anchor_assessments")
    @classmethod
    def validate_assessment_order(
        cls,
        value: list[OledSupplementaryMaterialIdentityAnchorAssessment],
    ) -> list[OledSupplementaryMaterialIdentityAnchorAssessment]:
        order = [item.evidence_anchor_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("anchor assessments must be sorted and unique")
        return value

    @field_validator("reviewed_collision_finding_digests")
    @classmethod
    def validate_collision_digests(cls, value: list[str]) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name="collision_finding_digest")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("reviewed collision digests must be sorted and unique")
        return clean

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="review_note",
            required=False,
            max_length=4_000,
        )


class OledSupplementaryMaterialIdentityDecisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = SUPPLEMENTARY_MATERIAL_IDENTITY_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    review_packet_sha256: str
    review_packet_digest: str
    review_source_pdf_evidence_digest: str
    reviewed_by: str
    reviewed_at: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledSupplementaryMaterialIdentityDecisionEntry] = Field(
        min_length=1
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected material identity decision manifest")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "review_packet_sha256",
        "review_packet_digest",
        "review_source_pdf_evidence_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
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
    def validate_manifest_shape(
        self,
    ) -> OledSupplementaryMaterialIdentityDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("material identity adjudication must be confirmed")
        item_ids = [entry.review_item_id for entry in self.decisions]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("material identity decision manifest repeats an item")
        return self


class OledSupplementaryAdjudicatedMaterialIdentityGroup(BaseModel):
    """One human decision with all truth values derived from that decision."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_item: OledSupplementaryMaterialIdentityReviewItem
    decision_entry: OledSupplementaryMaterialIdentityDecisionEntry
    adjudicated_group_digest: str
    source_anchors_human_validated: StrictBool
    source_to_candidate_match_human_validated: StrictBool
    paper_local_structure_candidate_accepted: StrictBool
    structure_anchor_only_confirmed: StrictBool
    source_check_pending: StrictBool
    ambiguous_identity_pending: StrictBool
    identity_exclusion_confirmed: StrictBool
    response_evidence_rejected: StrictBool
    eligible_for_later_registry_review: StrictBool
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    registry_written: StrictBool = False

    @field_validator("adjudicated_group_digest")
    @classmethod
    def validate_group_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="adjudicated_group_digest")

    @model_validator(mode="after")
    def validate_group_integrity(
        self,
    ) -> OledSupplementaryAdjudicatedMaterialIdentityGroup:
        if (
            self.decision_entry.review_item_id != self.review_item.review_item_id
            or self.decision_entry.review_item_digest
            != self.review_item.review_item_digest
        ):
            raise ValueError("adjudicated material identity item binding mismatch")
        _validate_decision_for_item(self.review_item, self.decision_entry)
        expected = _derived_adjudicated_group_flags(
            self.review_item,
            self.decision_entry,
        )
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ValueError(
                    f"adjudicated material identity {field_name} mismatch"
                )
        fixed_false = (
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
            "registry_written",
        )
        if any(getattr(self, name) for name in fixed_false):
            raise ValueError("adjudicated identity group crossed its boundary")
        if self.adjudicated_group_digest != _adjudicated_group_digest(self):
            raise ValueError("adjudicated material identity group digest mismatch")
        return self


class OledSupplementaryMaterialIdentityAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_ADJUDICATION_ARTIFACT_VERSION
    )
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    material_identity_request_digest: str
    transcription_review_packet_sha256: str
    transcription_review_packet_digest: str
    upstream_source_pdf_evidence_digest: str
    response_manifest_sha256: str
    response_manifest_digest: str
    response_artifact_sha256: str
    response_artifact_digest: str
    response_producer: OledSupplementaryMaterialIdentityEvidenceProducer
    review_source_pdf_evidence_digest: str
    review_packet_sha256: str
    review_packet_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    reviewed_by: str
    reviewed_at: str
    status: OledSupplementaryMaterialIdentityAdjudicationStatus
    review_item_count: Annotated[StrictInt, Field(ge=1)]
    accepted_structure_candidate_count: Annotated[StrictInt, Field(ge=0)]
    confirmed_structure_anchor_only_count: Annotated[StrictInt, Field(ge=0)]
    source_check_pending_group_count: Annotated[StrictInt, Field(ge=0)]
    ambiguous_identity_pending_group_count: Annotated[StrictInt, Field(ge=0)]
    identity_exclusion_confirmed_group_count: Annotated[StrictInt, Field(ge=0)]
    response_evidence_rejected_group_count: Annotated[StrictInt, Field(ge=0)]
    unresolved_review_item_count: Annotated[StrictInt, Field(ge=0)]
    later_registry_review_eligible_group_count: Annotated[StrictInt, Field(ge=0)]
    later_registry_review_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    adjudicated_groups: list[OledSupplementaryAdjudicatedMaterialIdentityGroup] = (
        Field(min_length=1)
    )
    adjudication_artifact_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    joint_exact_input_revalidation_required: StrictBool = True
    joint_upstream_and_pr_l_binding_revalidated: StrictBool = True
    review_packet_binding_revalidated: StrictBool = True
    source_pdf_and_page_assets_revalidated: StrictBool = True
    candidate_depictions_revalidated: StrictBool = True
    decision_binding_validated: StrictBool = True
    complete_group_decision_coverage_validated: StrictBool = True
    complete_anchor_assessment_coverage_validated: StrictBool = True
    human_decisions_recorded: StrictBool = True
    human_identity_review_completed: StrictBool = True
    standalone_upstream_chain_revalidation_supported: StrictBool = False
    standalone_source_pdf_bytes_revalidation_supported: StrictBool = False
    standalone_page_asset_bytes_revalidation_supported: StrictBool = False
    standalone_candidate_depiction_bytes_revalidation_supported: StrictBool = False
    source_pdf_remains_authoritative: StrictBool = True
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    registry_written: StrictBool = False
    schema_candidates_created: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    training_eligible: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_ADJUDICATION_ARTIFACT_VERSION:
            raise ValueError("unexpected material identity adjudication artifact")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at", "reviewed_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="reviewed_by",
            required=True,
            max_length=200,
        )

    @field_validator(
        "request_artifact_sha256",
        "material_identity_request_digest",
        "transcription_review_packet_sha256",
        "transcription_review_packet_digest",
        "upstream_source_pdf_evidence_digest",
        "response_manifest_sha256",
        "response_manifest_digest",
        "response_artifact_sha256",
        "response_artifact_digest",
        "review_source_pdf_evidence_digest",
        "review_packet_sha256",
        "review_packet_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "adjudication_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_adjudication_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityAdjudicationArtifact:
        if _parse_timestamp(self.reviewed_at) > _parse_timestamp(self.generated_at):
            raise ValueError("material identity adjudication predates review")
        item_ids = [
            item.review_item.review_item_id for item in self.adjudicated_groups
        ]
        if (
            self.review_item_count != len(self.adjudicated_groups)
            or len(item_ids) != len(set(item_ids))
        ):
            raise ValueError("material identity adjudication group coverage mismatch")
        counts = _adjudication_counts(self.adjudicated_groups)
        expected_fields = {
            "accepted_structure_candidate_count": "accepted_candidate_count",
            "confirmed_structure_anchor_only_count": "anchor_only_count",
            "source_check_pending_group_count": "source_check_count",
            "ambiguous_identity_pending_group_count": "ambiguous_count",
            "identity_exclusion_confirmed_group_count": "exclusion_count",
            "response_evidence_rejected_group_count": "rejected_count",
            "unresolved_review_item_count": "unresolved_count",
            "later_registry_review_eligible_group_count": "eligible_group_count",
            "later_registry_review_eligible_cell_count": "eligible_cell_count",
        }
        for field_name, count_name in expected_fields.items():
            if getattr(self, field_name) != counts[count_name]:
                raise ValueError(f"material identity {field_name} mismatch")
        expected_status = _adjudication_status(
            unresolved_count=counts["unresolved_count"],
            eligible_group_count=counts["eligible_group_count"],
        )
        if self.status != expected_status:
            raise ValueError("material identity adjudication status mismatch")
        if self.device_only_cell_count != 0:
            raise ValueError("device-only cells must remain outside adjudication")
        fixed_true = (
            "review_only",
            "offline_only",
            "joint_exact_input_revalidation_required",
            "joint_upstream_and_pr_l_binding_revalidated",
            "review_packet_binding_revalidated",
            "source_pdf_and_page_assets_revalidated",
            "candidate_depictions_revalidated",
            "decision_binding_validated",
            "complete_group_decision_coverage_validated",
            "complete_anchor_assessment_coverage_validated",
            "human_decisions_recorded",
            "human_identity_review_completed",
            "source_pdf_remains_authoritative",
        )
        fixed_false = (
            "standalone_upstream_chain_revalidation_supported",
            "standalone_source_pdf_bytes_revalidation_supported",
            "standalone_page_asset_bytes_revalidation_supported",
            "standalone_candidate_depiction_bytes_revalidation_supported",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
            "registry_written",
            "schema_candidates_created",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "training_eligible",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("material identity adjudication crossed its boundary")
        if (
            oled_supplementary_material_identity_adjudication_artifact_digest(
                self
            )
            != self.adjudication_artifact_digest
        ):
            raise ValueError("material identity adjudication digest mismatch")
        return self


def build_oled_supplementary_material_identity_candidate_depiction_asset(
    *,
    validated_result_id: str,
    candidate_digest: str,
    toolkit_version: str,
    rendered_asset_sha256: str,
    rendered_asset_byte_size: int,
    pixel_width: int,
    pixel_height: int,
) -> OledSupplementaryMaterialIdentityCandidateDepictionAsset:
    payload: dict[str, Any] = {
        "asset_id": "placeholder",
        "asset_filename": "placeholder.png",
        "validated_result_id": _validate_bound_id(
            validated_result_id,
            field_name="validated_result_id",
        ),
        "candidate_digest": _normalize_sha256(
            candidate_digest,
            field_name="candidate_digest",
        ),
        "toolkit_id": "rdkit",
        "toolkit_version": _validate_bound_id(
            toolkit_version,
            field_name="toolkit_version",
        ),
        "depiction_profile": SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE,
        "mime_type": "image/png",
        "rendered_asset_sha256": _normalize_sha256(
            rendered_asset_sha256,
            field_name="rendered_asset_sha256",
        ),
        "rendered_asset_byte_size": rendered_asset_byte_size,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "depiction_asset_digest": "sha256:" + "0" * 64,
        "exact_candidate_graph_used": True,
        "deterministic_2d_coordinates_requested": True,
        "non_authoritative_review_projection": True,
        "material_identity_resolved": False,
    }
    provisional = (
        OledSupplementaryMaterialIdentityCandidateDepictionAsset.model_construct(
            **payload
        )
    )
    suffix = _depiction_asset_suffix(provisional)
    payload["asset_id"] = f"material-identity-candidate:{suffix}"
    payload["asset_filename"] = f"identity-candidate-{suffix}.png"
    provisional = (
        OledSupplementaryMaterialIdentityCandidateDepictionAsset.model_construct(
            **payload
        )
    )
    payload["depiction_asset_digest"] = (
        oled_supplementary_material_identity_candidate_depiction_asset_digest(
            provisional
        )
    )
    return OledSupplementaryMaterialIdentityCandidateDepictionAsset.model_validate(
        payload
    )


def validate_oled_supplementary_material_identity_review_inputs(
    *,
    request_artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    request_artifact_sha256: str,
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    response_manifest: OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    response_artifact_sha256: str,
    review_source_pdf_evidence: OledSupplementarySourcePdfEvidence,
    candidate_depiction_assets: Sequence[
        OledSupplementaryMaterialIdentityCandidateDepictionAsset
    ],
) -> None:
    """Jointly replay PR-L and validate the exact PR-M review evidence roster."""

    request = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    transcription = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        transcription_review_packet.model_dump(mode="json")
    )
    manifest = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    response = OledSupplementaryMaterialIdentityEvidenceResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    source_evidence = OledSupplementarySourcePdfEvidence.model_validate(
        review_source_pdf_evidence.model_dump(mode="json")
    )
    depictions = [
        OledSupplementaryMaterialIdentityCandidateDepictionAsset.model_validate(
            item.model_dump(mode="json")
        )
        for item in candidate_depiction_assets
    ]
    request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    transcription_sha = _normalize_sha256(
        transcription_review_packet_sha256,
        field_name="transcription_review_packet_sha256",
    )
    manifest_sha = _normalize_sha256(
        response_manifest_sha256,
        field_name="response_manifest_sha256",
    )
    _normalize_sha256(
        response_artifact_sha256,
        field_name="response_artifact_sha256",
    )
    validate_oled_supplementary_material_identity_evidence_response_binding(
        request_artifact=request,
        request_artifact_sha256=request_sha,
        transcription_review_packet=transcription,
        transcription_review_packet_sha256=transcription_sha,
        response_manifest=manifest,
    )
    expected_response = (
        build_oled_supplementary_material_identity_evidence_response_artifact(
            request_artifact=request,
            request_artifact_sha256=request_sha,
            transcription_review_packet=transcription,
            transcription_review_packet_sha256=transcription_sha,
            response_manifest=manifest,
            response_manifest_sha256=manifest_sha,
            generated_at=response.generated_at,
        )
    )
    if expected_response.model_dump(mode="json") != response.model_dump(mode="json"):
        raise ValueError("material identity review requires an exact PR-L replay")
    if (
        response.request_artifact_sha256 != request_sha
        or response.transcription_review_packet_sha256 != transcription_sha
        or response.response_manifest_sha256 != manifest_sha
    ):
        raise ValueError("material identity review PR-L byte binding mismatch")
    upstream_source = transcription.source_pdf_evidence
    source_identity = (
        source_evidence.source_id,
        source_evidence.source_pdf_sha256,
        source_evidence.source_pdf_byte_size,
        source_evidence.source_pdf_page_count,
    )
    expected_source_identity = (
        upstream_source.source_id,
        upstream_source.source_pdf_sha256,
        upstream_source.source_pdf_byte_size,
        upstream_source.source_pdf_page_count,
    )
    if source_identity != expected_source_identity:
        raise ValueError("material identity review source PDF identity mismatch")
    anchors = [
        anchor
        for result in response.validated_results
        for anchor in _result_anchors(result.response_result)
    ]
    expected_pages = {
        result.bound_identity_group.pdf_page_number_one_based
        for result in response.validated_results
    } | {anchor.pdf_page_number_one_based for anchor in anchors}
    observed_pages = {
        asset.pdf_page_number_one_based for asset in source_evidence.page_assets
    }
    if observed_pages != expected_pages:
        missing = sorted(expected_pages - observed_pages)
        extra = sorted(observed_pages - expected_pages)
        raise ValueError(
            "material identity review page roster mismatch: "
            f"missing={missing}, extra={extra}"
        )
    candidates = {
        result.validated_result_id: result
        for result in response.validated_results
        if isinstance(
            result.response_result,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        )
    }
    depictions_by_result = {item.validated_result_id: item for item in depictions}
    if len(depictions_by_result) != len(depictions) or set(depictions_by_result) != set(
        candidates
    ):
        raise ValueError("material identity review candidate depiction coverage mismatch")
    asset_ids = [item.asset_id for item in depictions]
    filenames = [item.asset_filename for item in depictions]
    if len(asset_ids) != len(set(asset_ids)) or len(filenames) != len(set(filenames)):
        raise ValueError("material identity review candidate depictions repeat an asset")
    for result_id, result in candidates.items():
        chemistry = result.chemistry_validation
        if chemistry is None:
            raise ValueError("PR-L structure candidate lacks chemistry validation")
        depiction = depictions_by_result[result_id]
        if (
            depiction.candidate_digest != chemistry.candidate_digest
            or depiction.toolkit_version != response.rdkit_version
        ):
            raise ValueError("candidate depiction does not bind exact PR-L chemistry")


def build_oled_supplementary_material_identity_review_packet(
    *,
    request_artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    request_artifact_sha256: str,
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    response_manifest: OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    response_artifact_sha256: str,
    review_source_pdf_evidence: OledSupplementarySourcePdfEvidence,
    candidate_depiction_assets: Sequence[
        OledSupplementaryMaterialIdentityCandidateDepictionAsset
    ],
    generated_at: str,
) -> OledSupplementaryMaterialIdentityReviewPacket:
    request = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        request_artifact.model_dump(mode="json")
    )
    transcription = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        transcription_review_packet.model_dump(mode="json")
    )
    manifest = OledSupplementaryMaterialIdentityEvidenceResponseManifest.model_validate(
        response_manifest.model_dump(mode="json")
    )
    response = OledSupplementaryMaterialIdentityEvidenceResponseArtifact.model_validate(
        response_artifact.model_dump(mode="json")
    )
    source_evidence = OledSupplementarySourcePdfEvidence.model_validate(
        review_source_pdf_evidence.model_dump(mode="json")
    )
    depictions = [
        OledSupplementaryMaterialIdentityCandidateDepictionAsset.model_validate(
            item.model_dump(mode="json")
        )
        for item in candidate_depiction_assets
    ]
    validate_oled_supplementary_material_identity_review_inputs(
        request_artifact=request,
        request_artifact_sha256=request_artifact_sha256,
        transcription_review_packet=transcription,
        transcription_review_packet_sha256=transcription_review_packet_sha256,
        response_manifest=manifest,
        response_manifest_sha256=response_manifest_sha256,
        response_artifact=response,
        response_artifact_sha256=response_artifact_sha256,
        review_source_pdf_evidence=source_evidence,
        candidate_depiction_assets=depictions,
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(generated_at_clean) < _parse_timestamp(response.generated_at):
        raise ValueError("material identity review packet predates PR-L")
    page_assets = {
        asset.pdf_page_number_one_based: asset for asset in source_evidence.page_assets
    }
    depictions_by_result = {item.validated_result_id: item for item in depictions}
    collision_findings = sorted(
        response.collision_findings,
        key=lambda item: item.finding_digest,
    )
    review_items: list[OledSupplementaryMaterialIdentityReviewItem] = []
    for result in response.validated_results:
        group = result.bound_identity_group
        anchors = _result_anchors(result.response_result)
        row_reference = _build_page_reference(
            role=OledSupplementaryMaterialIdentityReviewPageRole.ROW_CONTEXT,
            asset=page_assets[group.pdf_page_number_one_based],
            anchor_digests=[],
        )
        evidence_references = [
            _build_page_reference(
                role=(
                    OledSupplementaryMaterialIdentityReviewPageRole
                    .IDENTITY_EVIDENCE
                ),
                asset=page_assets[page_number],
                anchor_digests=anchor_digests,
            )
            for page_number, anchor_digests in _anchor_digests_by_page(anchors).items()
        ]
        related = sorted(
            (
                finding
                for finding in collision_findings
                if group.identity_group_id in finding.identity_group_ids
            ),
            key=lambda item: item.finding_digest,
        )
        payload: dict[str, Any] = {
            "review_item_id": _review_item_id(result),
            "review_item_digest": "sha256:" + "0" * 64,
            "item_kind": (
                OledSupplementaryMaterialIdentityReviewItemKind
                .MATERIAL_IDENTITY_GROUP
            ),
            "source_match_contract_version": (
                SUPPLEMENTARY_MATERIAL_IDENTITY_SOURCE_MATCH_CONTRACT_VERSION
            ),
            "validated_result": result,
            "row_context_page": row_reference,
            "identity_evidence_pages": evidence_references,
            "candidate_depiction_asset": depictions_by_result.get(
                result.validated_result_id
            ),
            "related_collision_findings": related,
        }
        provisional = OledSupplementaryMaterialIdentityReviewItem.model_construct(
            **payload
        )
        payload["review_item_digest"] = _review_item_digest(provisional)
        review_items.append(
            OledSupplementaryMaterialIdentityReviewItem.model_validate(payload)
        )
    review_items.sort(
        key=lambda item: (
            item.validated_result.bound_identity_group.scope_id,
            item.validated_result.bound_identity_group.table_id,
            item.validated_result.bound_identity_group.row_index,
            item.validated_result.bound_identity_group.identity_group_id,
        )
    )
    payload = {
        "artifact_version": SUPPLEMENTARY_MATERIAL_IDENTITY_REVIEW_PACKET_VERSION,
        "run_id": response.run_id,
        "paper_id": response.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "material_identity_request_digest": request.material_identity_request_digest,
        "transcription_review_packet_sha256": _normalize_sha256(
            transcription_review_packet_sha256,
            field_name="transcription_review_packet_sha256",
        ),
        "transcription_review_packet_digest": transcription.review_packet_digest,
        "upstream_source_pdf_evidence_digest": (
            transcription.source_pdf_evidence_digest
        ),
        "response_manifest_sha256": _normalize_sha256(
            response_manifest_sha256,
            field_name="response_manifest_sha256",
        ),
        "response_manifest_digest": (
            oled_supplementary_material_identity_evidence_response_manifest_digest(
                manifest
            )
        ),
        "response_artifact_sha256": _normalize_sha256(
            response_artifact_sha256,
            field_name="response_artifact_sha256",
        ),
        "response_artifact_digest": (
            oled_supplementary_material_identity_evidence_response_artifact_digest(
                response
            )
        ),
        "response_generated_at": response.generated_at,
        "response_producer": response.producer,
        "review_source_pdf_evidence": source_evidence,
        "review_source_pdf_evidence_digest": source_evidence.source_pdf_evidence_digest,
        "status": (
            OledSupplementaryMaterialIdentityReviewPacketStatus
            .READY_FOR_HUMAN_MATERIAL_IDENTITY_REVIEW
        ),
        "identity_group_count": response.identity_group_count,
        "review_item_count": len(review_items),
        "identity_dependent_cell_count": response.identity_dependent_cell_count,
        "bounded_transcription_validated_cell_count": (
            response.bounded_transcription_validated_cell_count
        ),
        "upstream_ontology_review_pending_cell_count": (
            response.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "structure_candidate_count": response.structure_candidate_count,
        "structure_anchor_only_count": response.structure_anchor_only_count,
        "source_check_count": response.source_check_count,
        "ambiguous_identity_count": response.ambiguous_identity_count,
        "exclusion_proposal_count": response.exclusion_proposal_count,
        "evidence_anchor_count": response.evidence_anchor_count,
        "cited_source_page_count": len(source_evidence.page_assets),
        "candidate_depiction_asset_count": len(depictions),
        "collision_finding_count": len(collision_findings),
        "collision_findings": collision_findings,
        "review_items": review_items,
        "review_packet_digest": "sha256:" + "0" * 64,
        "candidate_depiction_renderer_called": bool(depictions),
    }
    provisional = OledSupplementaryMaterialIdentityReviewPacket.model_construct(
        **payload
    )
    payload["review_packet_digest"] = (
        oled_supplementary_material_identity_review_packet_digest(provisional)
    )
    return OledSupplementaryMaterialIdentityReviewPacket.model_validate(payload)


def validate_oled_supplementary_material_identity_decision_binding(
    *,
    review_packet: OledSupplementaryMaterialIdentityReviewPacket,
    review_packet_sha256: str,
    decision_manifest: OledSupplementaryMaterialIdentityDecisionManifest,
) -> None:
    packet = OledSupplementaryMaterialIdentityReviewPacket.model_validate(
        review_packet.model_dump(mode="json")
    )
    decisions = OledSupplementaryMaterialIdentityDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    packet_sha = _normalize_sha256(
        review_packet_sha256,
        field_name="review_packet_sha256",
    )
    if (
        decisions.run_id != packet.run_id
        or decisions.paper_id != packet.paper_id
        or decisions.review_packet_sha256 != packet_sha
        or decisions.review_packet_digest != packet.review_packet_digest
        or decisions.review_source_pdf_evidence_digest
        != packet.review_source_pdf_evidence_digest
    ):
        raise ValueError("material identity decision packet binding mismatch")
    if _parse_timestamp(decisions.reviewed_at) < _parse_timestamp(packet.generated_at):
        raise ValueError("material identity decision predates its review packet")
    items = {item.review_item_id: item for item in packet.review_items}
    entries = {entry.review_item_id: entry for entry in decisions.decisions}
    if set(items) != set(entries):
        missing = sorted(set(items) - set(entries))
        unknown = sorted(set(entries) - set(items))
        raise ValueError(
            "material identity decision coverage mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    for item_id, item in items.items():
        entry = entries[item_id]
        if (
            entry.review_item_digest != item.review_item_digest
            or entry.item_kind != item.item_kind
        ):
            raise ValueError("material identity decision item binding mismatch")
        _validate_decision_for_item(item, entry)


def build_oled_supplementary_material_identity_adjudication_artifact(
    *,
    request_artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    request_artifact_sha256: str,
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    response_manifest: OledSupplementaryMaterialIdentityEvidenceResponseManifest,
    response_manifest_sha256: str,
    response_artifact: OledSupplementaryMaterialIdentityEvidenceResponseArtifact,
    response_artifact_sha256: str,
    review_source_pdf_evidence: OledSupplementarySourcePdfEvidence,
    candidate_depiction_assets: Sequence[
        OledSupplementaryMaterialIdentityCandidateDepictionAsset
    ],
    review_packet: OledSupplementaryMaterialIdentityReviewPacket,
    review_packet_sha256: str,
    decision_manifest: OledSupplementaryMaterialIdentityDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledSupplementaryMaterialIdentityAdjudicationArtifact:
    packet = OledSupplementaryMaterialIdentityReviewPacket.model_validate(
        review_packet.model_dump(mode="json")
    )
    decisions = OledSupplementaryMaterialIdentityDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    expected_packet = build_oled_supplementary_material_identity_review_packet(
        request_artifact=request_artifact,
        request_artifact_sha256=request_artifact_sha256,
        transcription_review_packet=transcription_review_packet,
        transcription_review_packet_sha256=transcription_review_packet_sha256,
        response_manifest=response_manifest,
        response_manifest_sha256=response_manifest_sha256,
        response_artifact=response_artifact,
        response_artifact_sha256=response_artifact_sha256,
        review_source_pdf_evidence=review_source_pdf_evidence,
        candidate_depiction_assets=candidate_depiction_assets,
        generated_at=packet.generated_at,
    )
    if expected_packet.model_dump(mode="json") != packet.model_dump(mode="json"):
        raise ValueError("material identity adjudication requires an exact packet replay")
    validate_oled_supplementary_material_identity_decision_binding(
        review_packet=packet,
        review_packet_sha256=review_packet_sha256,
        decision_manifest=decisions,
    )
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(generated_at_clean) < _parse_timestamp(decisions.reviewed_at):
        raise ValueError("material identity adjudication predates human review")
    decisions_by_item = {
        entry.review_item_id: entry for entry in decisions.decisions
    }
    adjudicated: list[OledSupplementaryAdjudicatedMaterialIdentityGroup] = []
    for item in packet.review_items:
        entry = decisions_by_item[item.review_item_id]
        flags = _derived_adjudicated_group_flags(item, entry)
        payload: dict[str, Any] = {
            "review_item": item,
            "decision_entry": entry,
            "adjudicated_group_digest": "sha256:" + "0" * 64,
            **flags,
        }
        provisional = (
            OledSupplementaryAdjudicatedMaterialIdentityGroup.model_construct(
                **payload
            )
        )
        payload["adjudicated_group_digest"] = _adjudicated_group_digest(
            provisional
        )
        adjudicated.append(
            OledSupplementaryAdjudicatedMaterialIdentityGroup.model_validate(payload)
        )
    counts = _adjudication_counts(adjudicated)
    payload = {
        "artifact_version": (
            SUPPLEMENTARY_MATERIAL_IDENTITY_ADJUDICATION_ARTIFACT_VERSION
        ),
        "run_id": packet.run_id,
        "paper_id": packet.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": packet.request_artifact_sha256,
        "material_identity_request_digest": packet.material_identity_request_digest,
        "transcription_review_packet_sha256": (
            packet.transcription_review_packet_sha256
        ),
        "transcription_review_packet_digest": (
            packet.transcription_review_packet_digest
        ),
        "upstream_source_pdf_evidence_digest": (
            packet.upstream_source_pdf_evidence_digest
        ),
        "response_manifest_sha256": packet.response_manifest_sha256,
        "response_manifest_digest": packet.response_manifest_digest,
        "response_artifact_sha256": packet.response_artifact_sha256,
        "response_artifact_digest": packet.response_artifact_digest,
        "response_producer": packet.response_producer,
        "review_source_pdf_evidence_digest": (
            packet.review_source_pdf_evidence_digest
        ),
        "review_packet_sha256": _normalize_sha256(
            review_packet_sha256,
            field_name="review_packet_sha256",
        ),
        "review_packet_digest": packet.review_packet_digest,
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": (
            oled_supplementary_material_identity_decision_manifest_digest(decisions)
        ),
        "reviewed_by": decisions.reviewed_by,
        "reviewed_at": decisions.reviewed_at,
        "status": _adjudication_status(
            unresolved_count=counts["unresolved_count"],
            eligible_group_count=counts["eligible_group_count"],
        ),
        "review_item_count": len(adjudicated),
        "accepted_structure_candidate_count": counts["accepted_candidate_count"],
        "confirmed_structure_anchor_only_count": counts["anchor_only_count"],
        "source_check_pending_group_count": counts["source_check_count"],
        "ambiguous_identity_pending_group_count": counts["ambiguous_count"],
        "identity_exclusion_confirmed_group_count": counts["exclusion_count"],
        "response_evidence_rejected_group_count": counts["rejected_count"],
        "unresolved_review_item_count": counts["unresolved_count"],
        "later_registry_review_eligible_group_count": counts[
            "eligible_group_count"
        ],
        "later_registry_review_eligible_cell_count": counts["eligible_cell_count"],
        "upstream_ontology_review_pending_cell_count": (
            packet.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "adjudicated_groups": adjudicated,
        "adjudication_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledSupplementaryMaterialIdentityAdjudicationArtifact.model_construct(
        **payload
    )
    payload["adjudication_artifact_digest"] = (
        oled_supplementary_material_identity_adjudication_artifact_digest(
            provisional
        )
    )
    return OledSupplementaryMaterialIdentityAdjudicationArtifact.model_validate(
        payload
    )


def render_oled_supplementary_material_identity_review_markdown(
    packet: OledSupplementaryMaterialIdentityReviewPacket,
    *,
    review_packet_sha256: str = "",
) -> str:
    review = OledSupplementaryMaterialIdentityReviewPacket.model_validate(
        packet.model_dump(mode="json")
    )
    packet_sha = (
        _normalize_sha256(review_packet_sha256, field_name="review_packet_sha256")
        if str(review_packet_sha256 or "").strip()
        else "not supplied"
    )
    assets_by_id = {
        asset.asset_id: asset for asset in review.review_source_pdf_evidence.page_assets
    }
    first_page_asset = review.review_source_pdf_evidence.page_assets[0]
    page_usage: dict[int, list[str]] = {
        asset.pdf_page_number_one_based: []
        for asset in review.review_source_pdf_evidence.page_assets
    }
    for item in review.review_items:
        group = item.validated_result.bound_identity_group
        page_usage[item.row_context_page.pdf_page_number_one_based].append(
            "table context only for " + group.reported_subject_text
        )
        anchors_by_digest = {
            oled_supplementary_material_identity_evidence_anchor_digest(anchor): anchor
            for anchor in _result_anchors(item.validated_result.response_result)
        }
        for reference in item.identity_evidence_pages:
            for digest in reference.evidence_anchor_digests:
                anchor = anchors_by_digest[digest]
                page_usage[reference.pdf_page_number_one_based].append(
                    "identity evidence for "
                    + group.reported_subject_text
                    + ": "
                    + anchor.singleton_locator
                    + " / panel "
                    + (anchor.panel_label or "not asserted")
                )
    lines = [
        "# Supplementary material-identity review",
        "",
        "> Compare each exact paper-local row, every claimed source anchor, and any candidate depiction.",
        "> Candidate depictions are non-authoritative review projections. The bound PDF remains authoritative.",
        "> Acceptance does not write a Registry, assign a canonical identity, merge aliases, or admit dataset records.",
        "",
        "## Exact binding",
        "",
        f"- Paper: `{_md_code(review.paper_id)}`",
        f"- Run ID: `{_md_code(review.run_id)}`",
        f"- Review packet file SHA-256: `{packet_sha}`",
        f"- Review packet digest: `{review.review_packet_digest}`",
        f"- PR-K request SHA-256 / digest: `{review.request_artifact_sha256}` / `{review.material_identity_request_digest}`",
        f"- PR-J packet SHA-256 / digest: `{review.transcription_review_packet_sha256}` / `{review.transcription_review_packet_digest}`",
        f"- Upstream PR-J PDF-evidence digest: `{review.upstream_source_pdf_evidence_digest}`",
        f"- PR-L manifest SHA-256 / digest: `{review.response_manifest_sha256}` / `{review.response_manifest_digest}`",
        f"- PR-L artifact SHA-256: `{review.response_artifact_sha256}`",
        f"- PR-L artifact digest: `{review.response_artifact_digest}`",
        f"- Response producer kind: `{review.response_producer.kind.value}`",
        f"- Execution client: `{_md_code(review.response_producer.client_id)}`",
        f"- Model provider / snapshot: `{_md_code(review.response_producer.model_provider_id or 'not supplied')}` / `{_md_code(review.response_producer.model_snapshot_id or 'not supplied')}`",
        f"- Source ID: `{_md_code(review.review_source_pdf_evidence.source_id)}`",
        f"- Source PDF SHA-256: `{review.review_source_pdf_evidence.source_pdf_sha256}`",
        f"- Source PDF pages / bytes: {review.review_source_pdf_evidence.source_pdf_page_count} / {review.review_source_pdf_evidence.source_pdf_byte_size}",
        f"- Review render evidence digest: `{review.review_source_pdf_evidence_digest}`",
        f"- Renderer profile: `{first_page_asset.renderer_id}` / `{first_page_asset.render_profile}`",
        f"- Groups / dependent cells: {review.identity_group_count} / {review.identity_dependent_cell_count}",
        f"- Ontology-pending / device-only cells: {review.upstream_ontology_review_pending_cell_count} / {review.device_only_cell_count}",
        "",
        "## Source-match contract",
        "",
        f"Contract: `{review.review_items[0].source_match_contract_version}`",
        "",
        "For an anchor, `supports_claim` means its page, locator, panel label, and every declared evidence role are supported by the source. For a structure candidate, `matches_source` additionally means that the exact candidate graph shown in the bound depiction matches the authoritative source representation.",
        "",
        "## Bound source pages",
        "",
    ]
    for asset in review.review_source_pdf_evidence.page_assets:
        lines.extend(
            [
                f"### PDF page {asset.pdf_page_number_one_based}",
                "",
                f"- Page asset ID: `{asset.asset_id}`",
                f"- Page asset digest: `{asset.page_asset_digest}`",
                f"![Bound source page {asset.pdf_page_number_one_based}]({_md_url(asset.asset_filename)})",
                "",
            ]
        )
        for usage in sorted(page_usage[asset.pdf_page_number_one_based]):
            lines.append(f"- Use: `{_md_code(usage)}`")
        lines.append("")
    lines.extend(
        [
            "## Group overview",
            "",
            "| Item | Reported subject | PR-L disposition | Anchors | Candidate depiction | Chemistry/collision warning | Allowed outcomes |",
            "| --- | --- | --- | ---: | --- | --- | --- |",
        ]
    )
    for index, item in enumerate(review.review_items, start=1):
        result = item.validated_result
        chemistry = result.chemistry_validation
        warning_parts: list[str] = []
        if chemistry is not None:
            warning_parts.extend(code.value for code in chemistry.finding_codes)
        if item.related_collision_findings:
            warning_parts.append("candidate identifier collision")
        warning = ", ".join(warning_parts) if warning_parts else "none recorded"
        allowed_outcomes = ", ".join(
            value.value
            for value in sorted(
                _allowed_decisions_for_disposition(result.response_result.disposition),
                key=lambda value: value.value,
            )
        )
        lines.append(
            "| "
            + f"I{index:02d} | {_md_code(result.bound_identity_group.reported_subject_text)} | "
            + f"{result.response_result.disposition.value} | "
            + f"{len(_result_anchors(result.response_result))} | "
            + ("yes" if item.candidate_depiction_asset is not None else "no")
            + f" | {_md_code(warning)} | {_md_code(allowed_outcomes)} |"
        )
    lines.append("")
    for index, item in enumerate(review.review_items, start=1):
        result = item.validated_result
        group = result.bound_identity_group
        response = result.response_result
        lines.extend(
            [
                f"## {index}. `{_md_code(group.reported_subject_text)}`",
                "",
                f"- Review item ID: `{item.review_item_id}`",
                f"- Review item digest: `{item.review_item_digest}`",
                f"- Identity group ID: `{group.identity_group_id}`",
                f"- Validated result ID: `{result.validated_result_id}`",
                f"- Response disposition: `{response.disposition.value}`",
                f"- Table / row: `{_md_code(group.table_id)}` / {group.row_index}",
                f"- Reported subject literal: `{_md_code(group.reported_subject_text)}`",
                f"- Dependent cells: {group.identity_dependent_cell_count}",
                "- Dependent property headers: "
                + ", ".join(
                    f"`{_md_code(cell.column_name)}`"
                    for cell in group.identity_dependent_cells
                ),
                "",
            ]
        )
        for label, value in _response_review_details(response):
            lines.append(f"- {label}: `{_md_code(value)}`")
        allowed_decisions = sorted(
            _allowed_decisions_for_disposition(response.disposition),
            key=lambda value: value.value,
        )
        lines.extend(
            [
                "- Allowed decisions: "
                + ", ".join(f"`{value.value}`" for value in allowed_decisions),
                "",
                "### Row context (not identity evidence)",
                "",
            ]
        )
        row_asset = assets_by_id[item.row_context_page.page_asset_id]
        lines.extend(
            [
                f"- PDF page: {row_asset.pdf_page_number_one_based}",
                f"- Page asset ID: `{row_asset.asset_id}`",
                f"- Page asset digest: `{row_asset.page_asset_digest}`",
                "",
                "### Identity evidence",
                "",
            ]
        )
        anchors = _result_anchors(response)
        anchors_by_digest = {
            oled_supplementary_material_identity_evidence_anchor_digest(anchor): anchor
            for anchor in anchors
        }
        if not anchors:
            lines.extend(["No source anchor was proposed for this outcome.", ""])
        for reference in item.identity_evidence_pages:
            asset = assets_by_id[reference.page_asset_id]
            lines.extend(
                [
                    f"#### PDF page {asset.pdf_page_number_one_based}",
                    "",
                    f"- Page asset ID: `{asset.asset_id}`",
                    f"- Page asset digest: `{asset.page_asset_digest}`",
                    "",
                ]
            )
            for digest in reference.evidence_anchor_digests:
                anchor = anchors_by_digest[digest]
                roles = ", ".join(role.value for role in anchor.evidence_roles)
                lines.extend(
                    [
                        f"- Anchor digest: `{digest}`",
                        f"  - source page: [PDF page {asset.pdf_page_number_one_based}](#pdf-page-{asset.pdf_page_number_one_based})",
                        f"  - anchor kind: `{anchor.anchor_kind.value}`",
                        f"  - locator: `{_md_code(anchor.singleton_locator)}`",
                        f"  - panel label: `{_md_code(anchor.panel_label or 'none')}`",
                        f"  - roles: `{_md_code(roles)}`",
                        f"  - representation kind: `{anchor.source_representation_kind.value}`",
                        f"  - representation: `{_md_code(anchor.source_representation)}`",
                    ]
                )
                if anchor.source_excerpt:
                    lines.append(
                        f"  - source excerpt: `{_md_code(anchor.source_excerpt)}`"
                    )
            lines.append("")
        if item.candidate_depiction_asset is not None:
            candidate = response.structure_candidate
            depiction = item.candidate_depiction_asset
            chemistry = result.chemistry_validation
            assert chemistry is not None
            lines.extend(
                [
                    "### Untrusted candidate depiction - not source evidence",
                    "",
                    f"- Candidate origin: `{candidate.candidate_origin.value}`",
                    f"- Candidate literal: `{_md_code(candidate.structure_candidate_text)}`",
                    f"- Candidate canonical isomeric SMILES: `{_md_code(candidate.canonical_isomeric_smiles_candidate)}`",
                    f"- Candidate InChIKey: `{candidate.inchikey_candidate}`",
                    f"- Chemistry-validation digest: `{chemistry.chemistry_validation_digest}`",
                    f"- Fragment count / net charge: {chemistry.fragment_count} / {chemistry.net_formal_charge}",
                    "- Deterministic chemistry findings: `"
                    + _md_code(
                        ", ".join(code.value for code in chemistry.finding_codes)
                        or "none recorded"
                    )
                    + "`",
                    f"- Depiction asset digest: `{depiction.depiction_asset_digest}`",
                    f"![Untrusted candidate graph review projection]({_md_url(depiction.asset_filename)})",
                    "",
                ]
            )
        if item.related_collision_findings:
            lines.extend(["### Collision findings (no merge performed)", ""])
            for finding in item.related_collision_findings:
                lines.append(
                    f"- `{finding.finding_digest}` — `{finding.finding_kind.value}` — groups: "
                    + ", ".join(f"`{group_id}`" for group_id in finding.identity_group_ids)
                )
            lines.append("")
        lines.extend(
            [
                "### Decision manifest values",
                "",
                f"- `review_item_id`: `{item.review_item_id}`",
                f"- `review_item_digest`: `{item.review_item_digest}`",
                f"- `item_kind`: `{item.item_kind.value}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def oled_supplementary_material_identity_candidate_depiction_asset_digest(
    asset: OledSupplementaryMaterialIdentityCandidateDepictionAsset,
) -> str:
    payload = asset.model_dump(mode="json")
    payload.pop("depiction_asset_digest", None)
    return _stable_hash(payload)


def oled_supplementary_material_identity_review_packet_digest(
    packet: OledSupplementaryMaterialIdentityReviewPacket,
) -> str:
    payload = packet.model_dump(mode="json")
    payload.pop("review_packet_digest", None)
    return _stable_hash(payload)


def oled_supplementary_material_identity_decision_manifest_digest(
    manifest: OledSupplementaryMaterialIdentityDecisionManifest,
) -> str:
    payload = manifest.model_dump(mode="json")
    payload["decisions"] = sorted(
        payload["decisions"],
        key=lambda item: item["review_item_id"],
    )
    return _stable_hash(payload)


def oled_supplementary_material_identity_adjudication_artifact_digest(
    artifact: OledSupplementaryMaterialIdentityAdjudicationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("adjudication_artifact_digest", None)
    return _stable_hash(payload)


def _result_anchors(response: Any) -> list[OledSupplementaryMaterialIdentityEvidenceAnchor]:
    if isinstance(
        response,
        (
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
            OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
        ),
    ):
        return list(response.evidence_anchors)
    return []


def _response_review_details(response: Any) -> list[tuple[str, str]]:
    if isinstance(
        response,
        (
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
            OledSupplementaryMaterialIdentityRecordStructureAnchorOnly,
        ),
    ):
        return [("Proposal note", response.proposal_note or "none")]
    if isinstance(response, OledSupplementaryMaterialIdentityNeedsSourceCheck):
        return [
            ("Source-check reason", response.source_check_reason.value),
            ("Response review note", response.review_note),
        ]
    if isinstance(response, OledSupplementaryMaterialIdentityAmbiguousIdentity):
        return [
            ("Ambiguity reason", response.ambiguity_reason.value),
            ("Response review note", response.review_note),
        ]
    if isinstance(response, OledSupplementaryMaterialIdentityExcludeIdentityGroup):
        return [
            ("Exclusion reason", response.exclusion_reason.value),
            ("Response review note", response.review_note),
        ]
    raise ValueError("unsupported material identity response disposition")


def _anchor_digests_by_page(
    anchors: Sequence[OledSupplementaryMaterialIdentityEvidenceAnchor],
) -> dict[int, list[str]]:
    grouped: dict[int, list[str]] = {}
    for anchor in anchors:
        grouped.setdefault(anchor.pdf_page_number_one_based, []).append(
            oled_supplementary_material_identity_evidence_anchor_digest(anchor)
        )
    return {
        page: sorted(digests) for page, digests in sorted(grouped.items())
    }


def _build_page_reference(
    *,
    role: OledSupplementaryMaterialIdentityReviewPageRole,
    asset: OledSupplementarySourcePageAsset,
    anchor_digests: Sequence[str],
) -> OledSupplementaryMaterialIdentityReviewPageReference:
    payload = {
        "page_role": role,
        "pdf_page_number_one_based": asset.pdf_page_number_one_based,
        "page_asset_id": asset.asset_id,
        "page_asset_digest": asset.page_asset_digest,
        "evidence_anchor_digests": sorted(anchor_digests),
        "page_reference_digest": "sha256:" + "0" * 64,
    }
    provisional = OledSupplementaryMaterialIdentityReviewPageReference.model_construct(
        **payload
    )
    payload["page_reference_digest"] = _page_reference_digest(provisional)
    return OledSupplementaryMaterialIdentityReviewPageReference.model_validate(payload)


def _page_reference_digest(
    reference: OledSupplementaryMaterialIdentityReviewPageReference,
) -> str:
    payload = reference.model_dump(mode="json")
    payload.pop("page_reference_digest", None)
    return _stable_hash(payload)


def _depiction_asset_suffix(
    asset: OledSupplementaryMaterialIdentityCandidateDepictionAsset,
) -> str:
    identity = {
        "validated_result_id": asset.validated_result_id,
        "candidate_digest": asset.candidate_digest,
        "toolkit_id": asset.toolkit_id,
        "toolkit_version": asset.toolkit_version,
        "depiction_profile": asset.depiction_profile,
        "rendered_asset_sha256": asset.rendered_asset_sha256,
        "rendered_asset_byte_size": asset.rendered_asset_byte_size,
        "pixel_width": asset.pixel_width,
        "pixel_height": asset.pixel_height,
    }
    return _stable_hash(identity).removeprefix("sha256:")[:24]


def _review_item_id(
    result: OledSupplementaryMaterialIdentityValidatedEvidenceResult,
) -> str:
    suffix = _stable_hash(
        {
            "source_match_contract_version": (
                SUPPLEMENTARY_MATERIAL_IDENTITY_SOURCE_MATCH_CONTRACT_VERSION
            ),
            "validated_result_id": result.validated_result_id,
            "validated_result_digest": result.validated_result_digest,
        }
    ).removeprefix("sha256:")[:24]
    return f"material-identity-review:{suffix}"


def _review_item_digest(item: OledSupplementaryMaterialIdentityReviewItem) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("review_item_digest", None)
    return _stable_hash(payload)


def _adjudicated_group_digest(
    group: OledSupplementaryAdjudicatedMaterialIdentityGroup,
) -> str:
    payload = group.model_dump(mode="json")
    payload.pop("adjudicated_group_digest", None)
    return _stable_hash(payload)


def _allowed_decisions_for_disposition(
    disposition: OledSupplementaryMaterialIdentityEvidenceDispositionKind,
) -> set[OledSupplementaryMaterialIdentityReviewDecision]:
    decision = OledSupplementaryMaterialIdentityReviewDecision
    if disposition == (
        OledSupplementaryMaterialIdentityEvidenceDispositionKind
        .PROPOSE_STRUCTURE_CANDIDATE
    ):
        return {
            decision.ACCEPT_STRUCTURE_CANDIDATE,
            decision.NEEDS_SOURCE_CHECK,
            decision.REJECT_RESPONSE_EVIDENCE,
        }
    if disposition == (
        OledSupplementaryMaterialIdentityEvidenceDispositionKind
        .RECORD_STRUCTURE_ANCHOR_ONLY
    ):
        return {
            decision.ACCEPT_STRUCTURE_ANCHOR_ONLY,
            decision.NEEDS_SOURCE_CHECK,
            decision.REJECT_RESPONSE_EVIDENCE,
        }
    if disposition == (
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.NEEDS_SOURCE_CHECK
    ):
        return {decision.CONFIRM_SOURCE_CHECK, decision.REJECT_RESPONSE_EVIDENCE}
    if disposition == (
        OledSupplementaryMaterialIdentityEvidenceDispositionKind.AMBIGUOUS_IDENTITY
    ):
        return {
            decision.CONFIRM_AMBIGUOUS_IDENTITY,
            decision.REJECT_RESPONSE_EVIDENCE,
        }
    if disposition == (
        OledSupplementaryMaterialIdentityEvidenceDispositionKind
        .EXCLUDE_IDENTITY_GROUP
    ):
        return {
            decision.ACCEPT_IDENTITY_EXCLUSION,
            decision.NEEDS_SOURCE_CHECK,
            decision.REJECT_RESPONSE_EVIDENCE,
        }
    raise ValueError("unsupported material identity evidence disposition")


def _validate_decision_for_item(
    item: OledSupplementaryMaterialIdentityReviewItem,
    entry: OledSupplementaryMaterialIdentityDecisionEntry,
) -> None:
    response = item.validated_result.response_result
    if entry.decision not in _allowed_decisions_for_disposition(response.disposition):
        raise ValueError("material identity decision is incompatible with disposition")
    anchors = _result_anchors(response)
    expected_anchor_digests = sorted(
        oled_supplementary_material_identity_evidence_anchor_digest(anchor)
        for anchor in anchors
    )
    observed_anchor_digests = [
        assessment.evidence_anchor_digest
        for assessment in entry.anchor_assessments
    ]
    if observed_anchor_digests != expected_anchor_digests:
        raise ValueError("material identity anchor-assessment coverage mismatch")
    expected_collisions = [
        finding.finding_digest for finding in item.related_collision_findings
    ]
    if entry.reviewed_collision_finding_digests != expected_collisions:
        raise ValueError("material identity collision-review coverage mismatch")
    is_candidate = isinstance(
        response,
        OledSupplementaryMaterialIdentityProposeStructureCandidate,
    )
    if is_candidate:
        if (
            entry.candidate_source_match
            == OledSupplementaryMaterialIdentityCandidateSourceMatchResult
            .NOT_APPLICABLE
        ):
            raise ValueError("structure candidate requires a source-match result")
    elif (
        entry.candidate_source_match
        != OledSupplementaryMaterialIdentityCandidateSourceMatchResult.NOT_APPLICABLE
    ):
        raise ValueError("non-candidate outcome requires not_applicable source match")
    assessment_values = {
        assessment.assessment for assessment in entry.anchor_assessments
    }
    has_mismatch = (
        OledSupplementaryMaterialIdentityAnchorAssessmentResult
        .DOES_NOT_SUPPORT_CLAIM
        in assessment_values
    ) or (
        entry.candidate_source_match
        == OledSupplementaryMaterialIdentityCandidateSourceMatchResult
        .DOES_NOT_MATCH_SOURCE
    )
    has_unchecked = (
        OledSupplementaryMaterialIdentityAnchorAssessmentResult.NOT_CHECKED
        in assessment_values
    ) or (
        entry.candidate_source_match
        == OledSupplementaryMaterialIdentityCandidateSourceMatchResult.NOT_CHECKED
    )
    all_anchors_support = bool(anchors) and all(
        assessment.assessment
        == OledSupplementaryMaterialIdentityAnchorAssessmentResult.SUPPORTS_CLAIM
        for assessment in entry.anchor_assessments
    )
    decision = entry.decision
    if decision == (
        OledSupplementaryMaterialIdentityReviewDecision.ACCEPT_STRUCTURE_CANDIDATE
    ) and not (
        all_anchors_support
        and entry.candidate_source_match
        == OledSupplementaryMaterialIdentityCandidateSourceMatchResult.MATCHES_SOURCE
    ):
        raise ValueError("candidate acceptance requires complete supporting evidence")
    if decision == (
        OledSupplementaryMaterialIdentityReviewDecision
        .ACCEPT_STRUCTURE_ANCHOR_ONLY
    ) and not all_anchors_support:
        raise ValueError("anchor-only acceptance requires every anchor to support")
    if decision == OledSupplementaryMaterialIdentityReviewDecision.NEEDS_SOURCE_CHECK:
        unresolved_without_claimed_anchors = isinstance(
            response,
            OledSupplementaryMaterialIdentityExcludeIdentityGroup,
        ) and not anchors
        if (
            has_mismatch
            or (not has_unchecked and not unresolved_without_claimed_anchors)
            or not entry.review_note
        ):
            raise ValueError(
                "needs_source_check requires unchecked evidence, no mismatch, and review_note"
            )
    if decision == (
        OledSupplementaryMaterialIdentityReviewDecision.REJECT_RESPONSE_EVIDENCE
    ):
        if anchors and not has_mismatch:
            raise ValueError("positive response rejection requires a known mismatch")
        if not entry.review_note:
            raise ValueError("response evidence rejection requires review_note")
    if decision in {
        OledSupplementaryMaterialIdentityReviewDecision.CONFIRM_SOURCE_CHECK,
        OledSupplementaryMaterialIdentityReviewDecision
        .CONFIRM_AMBIGUOUS_IDENTITY,
    } and (entry.anchor_assessments or has_mismatch or has_unchecked):
        raise ValueError("unresolved response confirmation must not invent evidence checks")


def _derived_adjudicated_group_flags(
    item: OledSupplementaryMaterialIdentityReviewItem,
    entry: OledSupplementaryMaterialIdentityDecisionEntry,
) -> dict[str, bool]:
    decision = entry.decision
    accepted_candidate = decision == (
        OledSupplementaryMaterialIdentityReviewDecision.ACCEPT_STRUCTURE_CANDIDATE
    )
    anchor_only = decision == (
        OledSupplementaryMaterialIdentityReviewDecision
        .ACCEPT_STRUCTURE_ANCHOR_ONLY
    )
    source_check = decision in {
        OledSupplementaryMaterialIdentityReviewDecision.CONFIRM_SOURCE_CHECK,
        OledSupplementaryMaterialIdentityReviewDecision.NEEDS_SOURCE_CHECK,
    }
    ambiguous = decision == (
        OledSupplementaryMaterialIdentityReviewDecision
        .CONFIRM_AMBIGUOUS_IDENTITY
    )
    exclusion = decision == (
        OledSupplementaryMaterialIdentityReviewDecision.ACCEPT_IDENTITY_EXCLUSION
    )
    rejected = decision == (
        OledSupplementaryMaterialIdentityReviewDecision.REJECT_RESPONSE_EVIDENCE
    )
    return {
        "source_anchors_human_validated": accepted_candidate or anchor_only,
        "source_to_candidate_match_human_validated": accepted_candidate,
        "paper_local_structure_candidate_accepted": accepted_candidate,
        "structure_anchor_only_confirmed": anchor_only,
        "source_check_pending": source_check,
        "ambiguous_identity_pending": ambiguous,
        "identity_exclusion_confirmed": exclusion,
        "response_evidence_rejected": rejected,
        "eligible_for_later_registry_review": accepted_candidate,
    }


def _adjudication_counts(
    groups: Sequence[OledSupplementaryAdjudicatedMaterialIdentityGroup],
) -> dict[str, int]:
    accepted = sum(item.paper_local_structure_candidate_accepted for item in groups)
    anchor_only = sum(item.structure_anchor_only_confirmed for item in groups)
    source_check = sum(item.source_check_pending for item in groups)
    ambiguous = sum(item.ambiguous_identity_pending for item in groups)
    exclusion = sum(item.identity_exclusion_confirmed for item in groups)
    rejected = sum(item.response_evidence_rejected for item in groups)
    # Rejecting the supplied response evidence does not exclude or resolve the
    # paper-local material. It leaves that identity group unresolved for a new
    # evidence response.
    unresolved = anchor_only + source_check + ambiguous + rejected
    eligible = sum(item.eligible_for_later_registry_review for item in groups)
    eligible_cells = sum(
        item.review_item.validated_result.bound_identity_group
        .identity_dependent_cell_count
        for item in groups
        if item.eligible_for_later_registry_review
    )
    return {
        "accepted_candidate_count": accepted,
        "anchor_only_count": anchor_only,
        "source_check_count": source_check,
        "ambiguous_count": ambiguous,
        "exclusion_count": exclusion,
        "rejected_count": rejected,
        "unresolved_count": unresolved,
        "eligible_group_count": eligible,
        "eligible_cell_count": eligible_cells,
    }


def _adjudication_status(
    *,
    unresolved_count: int,
    eligible_group_count: int,
) -> OledSupplementaryMaterialIdentityAdjudicationStatus:
    if unresolved_count:
        return (
            OledSupplementaryMaterialIdentityAdjudicationStatus
            .REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS
        )
    if eligible_group_count:
        return (
            OledSupplementaryMaterialIdentityAdjudicationStatus
            .READY_FOR_LATER_MATERIAL_IDENTITY_REGISTRY_REVIEW
        )
    return (
        OledSupplementaryMaterialIdentityAdjudicationStatus
        .REVIEW_COMPLETE_NO_REGISTRY_ELIGIBLE_IDENTITIES
    )


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
    decoded = clean
    for _ in range(4):
        next_decoded = html.unescape(decoded)
        if next_decoded == decoded:
            break
        decoded = validate_oled_supplementary_safe_authored_text(
            next_decoded,
            field_name=field_name,
            required=required,
            max_length=max_length,
        )
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        for character in decoded
    ):
        raise ValueError(f"{field_name} contains unsafe display controls")
    return clean


def _validate_path_segment(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_PATH_SEGMENT_RE.fullmatch(clean) or clean in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path segment")
    return clean


def _validate_bound_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean or not _SAFE_BOUND_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    match = _SHA256_RE.fullmatch(str(value or "").strip())
    if match is None:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_timestamp(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    parsed = _parse_timestamp(clean)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return clean


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid material identity review timestamp") from exc


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _md_text(value: Any) -> str:
    visible = "".join(
        _visible_unicode_escape(character)
        if unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        else character
        for character in str(value)
    )
    return html.escape(visible, quote=True)


def _visible_unicode_escape(character: str) -> str:
    codepoint = ord(character)
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04X}"
    return f"\\U{codepoint:08X}"


def _md_code(value: Any) -> str:
    return _md_text(value).replace("`", "&#96;").replace("|", "&#124;")


def _md_url(value: str) -> str:
    filename = _validate_path_segment(value, field_name="asset_filename")
    return f"assets/{filename}"


__all__ = [
    "SUPPLEMENTARY_MATERIAL_IDENTITY_ADJUDICATION_ARTIFACT_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_DECISION_MANIFEST_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_DEPICTION_PROFILE",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_REVIEW_PACKET_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_SOURCE_MATCH_CONTRACT_VERSION",
    "OledSupplementaryAdjudicatedMaterialIdentityGroup",
    "OledSupplementaryMaterialIdentityAdjudicationArtifact",
    "OledSupplementaryMaterialIdentityAdjudicationStatus",
    "OledSupplementaryMaterialIdentityAnchorAssessment",
    "OledSupplementaryMaterialIdentityAnchorAssessmentResult",
    "OledSupplementaryMaterialIdentityCandidateDepictionAsset",
    "OledSupplementaryMaterialIdentityCandidateSourceMatchResult",
    "OledSupplementaryMaterialIdentityDecisionEntry",
    "OledSupplementaryMaterialIdentityDecisionManifest",
    "OledSupplementaryMaterialIdentityReviewDecision",
    "OledSupplementaryMaterialIdentityReviewItem",
    "OledSupplementaryMaterialIdentityReviewItemKind",
    "OledSupplementaryMaterialIdentityReviewPacket",
    "OledSupplementaryMaterialIdentityReviewPacketStatus",
    "OledSupplementaryMaterialIdentityReviewPageReference",
    "OledSupplementaryMaterialIdentityReviewPageRole",
    "build_oled_supplementary_material_identity_adjudication_artifact",
    "build_oled_supplementary_material_identity_candidate_depiction_asset",
    "build_oled_supplementary_material_identity_review_packet",
    "oled_supplementary_material_identity_adjudication_artifact_digest",
    "oled_supplementary_material_identity_candidate_depiction_asset_digest",
    "oled_supplementary_material_identity_decision_manifest_digest",
    "oled_supplementary_material_identity_review_packet_digest",
    "render_oled_supplementary_material_identity_review_markdown",
    "validate_oled_supplementary_material_identity_decision_binding",
    "validate_oled_supplementary_material_identity_review_inputs",
]
