from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_supplementary_scoped_candidate_request import (
    OledSupplementaryScopedCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    OledSupplementaryScopedCandidateResponseArtifact,
    OledSupplementaryScopedCandidateResponseManifest,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementaryAdjudicatedCell,
    OledSupplementarySemanticAdjudicationArtifact,
    OledSupplementarySemanticDecisionManifest,
    OledSupplementarySemanticReviewPacket,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceHeaderReviewBinding,
    OledSupplementarySourceTranscriptionAdjudicationArtifact,
    OledSupplementarySourceTranscriptionAdjudicationStatus,
    OledSupplementarySourceTranscriptionDecisionManifest,
    OledSupplementarySourceTranscriptionReviewPacket,
    build_oled_supplementary_source_transcription_adjudication_artifact,
)


SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_ARTIFACT_VERSION = (
    "oled_supplementary_material_identity_candidate_request.v1"
)
SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_CONTRACT_VERSION = (
    "oled_supplementary_material_identity_candidate_request_contract.v1"
)

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")

_IDENTITY_PROPOSAL_INSTRUCTIONS = (
    "Treat every group as one paper-local table row, not as a canonical material.",
    "Preserve the reported subject literal exactly and do not normalize aliases.",
    "Do not merge groups by spelling, case, punctuation, abbreviation, or chemical similarity.",
    (
        "Use the source PDF as authoritative and cite exact page and figure, "
        "scheme, table, or text evidence."
    ),
    (
        "Do not infer a structure, SMILES, InChIKey, material role, or entity "
        "type from a row label alone."
    ),
    (
        "Return an unresolved outcome when exact source-located identity "
        "evidence is unavailable or ambiguous."
    ),
    "Do not create Registry, Schema, Gold, training, or dataset records from this request.",
    "Return data only; do not return executable code, credentials, URLs, or local paths.",
)


class OledSupplementaryMaterialIdentityCandidateRequestStatus(str, Enum):
    READY_FOR_MATERIAL_IDENTITY_EVIDENCE_PROPOSAL = (
        "ready_for_material_identity_evidence_proposal"
    )


class OledSupplementaryMaterialIdentityDependentCell(BaseModel):
    """One exact PR-I cell whose later use depends on this row identity."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    row_index: StrictInt = Field(ge=0)
    column_index: StrictInt = Field(ge=0)
    column_name: str
    source_cell_digest: str
    cell_disposition_digest: str
    semantic_review_item_id: str
    semantic_review_item_digest: str

    @field_validator("column_name")
    @classmethod
    def validate_column_name(cls, value: str) -> str:
        return _validate_source_literal(value, field_name="column_name")

    @field_validator("semantic_review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="semantic_review_item_id")

    @field_validator(
        "source_cell_digest",
        "cell_disposition_digest",
        "semantic_review_item_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))


class OledSupplementaryMaterialIdentityCandidateGroup(BaseModel):
    """One exact paper-local row; never a resolved or cross-paper identity."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    identity_group_id: str
    identity_group_digest: str
    contract_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_CONTRACT_VERSION
    )
    scope_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    pdf_page_number_one_based: StrictInt = Field(ge=1)
    source_transcription_review_item_id: str
    source_transcription_review_item_digest: str
    row_index: StrictInt = Field(ge=0)
    subject_column_index: StrictInt = Field(ge=0)
    subject_header_binding: OledSupplementarySourceHeaderReviewBinding
    reported_subject_text: str
    identity_dependent_cell_count: StrictInt = Field(ge=1)
    identity_dependent_source_cell_digests: list[str] = Field(default_factory=list)
    identity_dependent_cells: list[OledSupplementaryMaterialIdentityDependentCell] = (
        Field(default_factory=list)
    )
    identity_resolution_scope: Literal["paper_local_table_row"] = (
        "paper_local_table_row"
    )
    identity_evidence_required: StrictBool = True
    source_structure_evidence_included: StrictBool = False
    human_identity_review_required: StrictBool = True
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False

    @field_validator(
        "identity_group_id",
        "scope_id",
        "table_id",
        "source_transcription_review_item_id",
    )
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("reported_subject_text")
    @classmethod
    def validate_reported_subject_text(cls, value: str) -> str:
        return _validate_source_literal(value, field_name="reported_subject_text")

    @field_validator(
        "identity_group_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
        "source_transcription_review_item_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("contract_version")
    @classmethod
    def validate_contract_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_CONTRACT_VERSION:
            raise ValueError("unexpected material identity candidate request contract")
        return value

    @field_validator("identity_dependent_source_cell_digests")
    @classmethod
    def validate_source_cell_digests(cls, value: list[str]) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name="source_cell_digest") for item in value
        ]
        if not clean or clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(
                "identity dependent source cell digests must be non-empty, sorted, and unique"
            )
        return clean

    @model_validator(mode="after")
    def validate_group_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityCandidateGroup:
        if self.subject_header_binding.column_index != self.subject_column_index:
            raise ValueError("material identity subject header binding mismatch")
        observed_order = [
            (cell.column_index, cell.source_cell_digest)
            for cell in self.identity_dependent_cells
        ]
        if observed_order != sorted(observed_order):
            raise ValueError("material identity dependent cells must be sorted")
        observed_digests = sorted(
            cell.source_cell_digest for cell in self.identity_dependent_cells
        )
        column_indexes = [
            cell.column_index for cell in self.identity_dependent_cells
        ]
        source_coordinates = [
            (cell.row_index, cell.column_index, cell.column_name)
            for cell in self.identity_dependent_cells
        ]
        if (
            self.identity_dependent_cell_count != len(self.identity_dependent_cells)
            or self.identity_dependent_cell_count
            != len(self.identity_dependent_source_cell_digests)
            or observed_digests != self.identity_dependent_source_cell_digests
        ):
            raise ValueError("material identity dependent cell roster mismatch")
        if (
            len(column_indexes) != len(set(column_indexes))
            or len(source_coordinates) != len(set(source_coordinates))
        ):
            raise ValueError("material identity dependent source coordinates repeat")
        if self.subject_column_index in column_indexes:
            raise ValueError("material identity subject column cannot be a dependent cell")
        if any(cell.row_index != self.row_index for cell in self.identity_dependent_cells):
            raise ValueError("material identity dependent cell moved to another row")
        if self.identity_group_id != _material_identity_group_id(self):
            raise ValueError("material identity group id mismatch")
        if self.identity_group_digest != _material_identity_group_digest(self):
            raise ValueError("material identity group digest mismatch")
        fixed_true = (
            "identity_evidence_required",
            "human_identity_review_required",
        )
        fixed_false = (
            "source_structure_evidence_included",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("material identity group lost a required boundary flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("material identity group crossed the request boundary")
        return self


class OledSupplementaryMaterialIdentityCandidateRequestArtifact(BaseModel):
    """Exact-bound request only; it contains no identity candidate or resolution."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_ARTIFACT_VERSION
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
    transcription_review_packet_sha256: str
    transcription_review_packet_digest: str
    transcription_decision_manifest_sha256: str
    transcription_decision_manifest_digest: str
    transcription_adjudication_artifact_sha256: str
    transcription_adjudication_artifact_digest: str
    source_pdf_evidence_digest: str
    status: OledSupplementaryMaterialIdentityCandidateRequestStatus
    source_count: StrictInt = Field(ge=1)
    scope_count: StrictInt = Field(ge=1)
    accepted_transcription_scope_count: StrictInt = Field(ge=1)
    identity_group_count: StrictInt = Field(ge=1)
    identity_dependent_cell_count: StrictInt = Field(ge=1)
    bounded_transcription_validated_cell_count: StrictInt = Field(ge=1)
    upstream_ontology_review_pending_cell_count: StrictInt = Field(ge=0)
    device_only_cell_count: Literal[0] = 0
    proposal_instructions: list[str] = Field(
        default_factory=lambda: list(_IDENTITY_PROPOSAL_INSTRUCTIONS)
    )
    identity_groups: list[OledSupplementaryMaterialIdentityCandidateGroup] = Field(
        default_factory=list
    )
    material_identity_request_digest: str
    request_only: StrictBool = True
    offline_only: StrictBool = True
    upstream_chain_replayed: StrictBool = True
    source_transcription_adjudication_replayed: StrictBool = True
    strict_eligible_cell_intersection_validated: StrictBool = True
    strict_row_partition_validated: StrictBool = True
    paper_local_row_scope_only: StrictBool = True
    reported_subject_literals_preserved: StrictBool = True
    bounded_source_transcription_accepted: StrictBool = True
    human_identity_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    material_identity_evidence_proposal_requested: StrictBool = True
    response_received: StrictBool = False
    source_pdf_read: StrictBool = False
    raw_parsed_document_read: StrictBool = False
    source_structure_evidence_included: StrictBool = False
    identity_evidence_validated: StrictBool = False
    material_identity_resolved: StrictBool = False
    canonical_smiles_assigned: StrictBool = False
    inchikey_assigned: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    table_exhaustiveness_validated: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False
    schema_candidates_created: StrictBool = False
    registry_written: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    training_eligible: StrictBool = False
    device_only_admitted: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_ARTIFACT_VERSION:
            raise ValueError("unexpected material identity candidate request version")
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
        "transcription_review_packet_sha256",
        "transcription_review_packet_digest",
        "transcription_decision_manifest_sha256",
        "transcription_decision_manifest_digest",
        "transcription_adjudication_artifact_sha256",
        "transcription_adjudication_artifact_digest",
        "source_pdf_evidence_digest",
        "material_identity_request_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledSupplementaryMaterialIdentityCandidateRequestArtifact:
        if self.status != (
            OledSupplementaryMaterialIdentityCandidateRequestStatus
            .READY_FOR_MATERIAL_IDENTITY_EVIDENCE_PROPOSAL
        ):
            raise ValueError("material identity request status mismatch")
        if self.proposal_instructions != list(_IDENTITY_PROPOSAL_INSTRUCTIONS):
            raise ValueError("material identity proposal instructions changed")
        group_order = [
            (group.scope_id, group.table_id, group.row_index, group.identity_group_id)
            for group in self.identity_groups
        ]
        if group_order != sorted(group_order):
            raise ValueError("material identity groups must be sorted")
        group_ids = [group.identity_group_id for group in self.identity_groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("material identity group ids must be unique")
        logical_rows = [
            (
                group.scope_id,
                group.table_id,
                group.table_content_digest,
                group.row_index,
            )
            for group in self.identity_groups
        ]
        if len(logical_rows) != len(set(logical_rows)):
            raise ValueError("material identity logical rows must be unique")
        cell_digests = [
            digest
            for group in self.identity_groups
            for digest in group.identity_dependent_source_cell_digests
        ]
        if (
            self.identity_group_count != len(self.identity_groups)
            or self.identity_dependent_cell_count != len(cell_digests)
            or len(cell_digests) != len(set(cell_digests))
        ):
            raise ValueError("material identity aggregate coverage mismatch")
        observed_scopes = {group.scope_id for group in self.identity_groups}
        source_provenance_by_id: dict[str, tuple[str, str]] = {}
        scope_provenance: dict[str, tuple[Any, ...]] = {}
        transcription_item_provenance: dict[str, tuple[Any, ...]] = {}
        column_provenance: dict[tuple[Any, ...], tuple[Any, ...]] = {}
        semantic_item_provenance: dict[str, tuple[Any, ...]] = {}
        for group in self.identity_groups:
            source_provenance = (
                group.source_pdf_sha256,
                group.parsed_document_sha256,
            )
            previous_source = source_provenance_by_id.setdefault(
                group.source_id,
                source_provenance,
            )
            if previous_source != source_provenance:
                raise ValueError(
                    "material identity source id binds inconsistent source provenance"
                )
            provenance = (
                group.source_id,
                group.source_pdf_sha256,
                group.parsed_document_sha256,
                group.table_id,
                group.table_content_digest,
                group.pdf_page_number_one_based,
                group.source_transcription_review_item_id,
                group.source_transcription_review_item_digest,
                group.subject_column_index,
                group.subject_header_binding.model_dump(mode="json"),
            )
            previous_provenance = scope_provenance.setdefault(
                group.scope_id,
                provenance,
            )
            if previous_provenance != provenance:
                raise ValueError("material identity scope provenance is inconsistent")
            transcription_provenance = (
                group.source_transcription_review_item_digest,
                group.scope_id,
                group.source_id,
                group.source_pdf_sha256,
                group.parsed_document_sha256,
                group.table_id,
                group.table_content_digest,
                group.pdf_page_number_one_based,
            )
            previous_transcription = transcription_item_provenance.setdefault(
                group.source_transcription_review_item_id,
                transcription_provenance,
            )
            if previous_transcription != transcription_provenance:
                raise ValueError(
                    "material identity transcription review item provenance is inconsistent"
                )
            for cell in group.identity_dependent_cells:
                column_key = (
                    group.scope_id,
                    group.table_id,
                    group.table_content_digest,
                    cell.column_index,
                )
                column_value = (
                    cell.column_name,
                    cell.semantic_review_item_id,
                    cell.semantic_review_item_digest,
                )
                previous_column = column_provenance.setdefault(
                    column_key,
                    column_value,
                )
                if previous_column != column_value:
                    raise ValueError(
                        "material identity column provenance is inconsistent"
                    )
                semantic_value = (
                    cell.semantic_review_item_digest,
                    group.scope_id,
                    group.table_id,
                    group.table_content_digest,
                    cell.column_index,
                    cell.column_name,
                )
                previous_semantic = semantic_item_provenance.setdefault(
                    cell.semantic_review_item_id,
                    semantic_value,
                )
                if previous_semantic != semantic_value:
                    raise ValueError(
                        "material identity semantic review item provenance is inconsistent"
                    )
        if self.scope_count != len(observed_scopes) or self.source_count != len(
            source_provenance_by_id
        ):
            raise ValueError("material identity source or scope count mismatch")
        if self.accepted_transcription_scope_count < self.scope_count:
            raise ValueError("material identity accepted scope count mismatch")
        if (
            self.identity_dependent_cell_count
            > self.bounded_transcription_validated_cell_count
        ):
            raise ValueError("identity-dependent cells exceed bounded transcription")
        if (
            self.identity_dependent_cell_count
            + self.upstream_ontology_review_pending_cell_count
            > self.bounded_transcription_validated_cell_count
        ):
            raise ValueError("material identity cell partition exceeds bounded transcription")
        fixed_true = (
            "request_only",
            "offline_only",
            "upstream_chain_replayed",
            "source_transcription_adjudication_replayed",
            "strict_eligible_cell_intersection_validated",
            "strict_row_partition_validated",
            "paper_local_row_scope_only",
            "reported_subject_literals_preserved",
            "bounded_source_transcription_accepted",
            "human_identity_review_required",
            "source_pdf_remains_authoritative",
            "material_identity_evidence_proposal_requested",
        )
        fixed_false = (
            "response_received",
            "source_pdf_read",
            "raw_parsed_document_read",
            "source_structure_evidence_included",
            "identity_evidence_validated",
            "material_identity_resolved",
            "canonical_smiles_assigned",
            "inchikey_assigned",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
            "table_exhaustiveness_validated",
            "scientific_content_validated",
            "physical_semantics_validated",
            "schema_candidates_created",
            "registry_written",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "training_eligible",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true):
            raise ValueError("material identity request lost a required audit flag")
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("material identity request crossed a downstream boundary")
        if self.material_identity_request_digest != _material_identity_request_digest(
            self
        ):
            raise ValueError("material identity candidate request digest mismatch")
        return self


def build_oled_supplementary_material_identity_candidate_request_artifact(
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
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    transcription_decision_manifest: OledSupplementarySourceTranscriptionDecisionManifest,
    transcription_decision_manifest_sha256: str,
    transcription_adjudication_artifact: OledSupplementarySourceTranscriptionAdjudicationArtifact,
    transcription_adjudication_artifact_sha256: str,
    generated_at: str,
) -> OledSupplementaryMaterialIdentityCandidateRequestArtifact:
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
    transcription_packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        transcription_review_packet.model_dump(mode="json")
    )
    transcription_decisions = OledSupplementarySourceTranscriptionDecisionManifest.model_validate(
        transcription_decision_manifest.model_dump(mode="json")
    )
    transcription_adjudication = (
        OledSupplementarySourceTranscriptionAdjudicationArtifact.model_validate(
            transcription_adjudication_artifact.model_dump(mode="json")
        )
    )
    hashes = {
        "request": _normalize_sha256(request_artifact_sha256, field_name="request_artifact_sha256"),
        "response_manifest": _normalize_sha256(
            response_manifest_sha256,
            field_name="response_manifest_sha256",
        ),
        "response_artifact": _normalize_sha256(
            response_artifact_sha256,
            field_name="response_artifact_sha256",
        ),
        "semantic_packet": _normalize_sha256(
            semantic_review_packet_sha256,
            field_name="semantic_review_packet_sha256",
        ),
        "semantic_decisions": _normalize_sha256(
            semantic_decision_manifest_sha256,
            field_name="semantic_decision_manifest_sha256",
        ),
        "semantic_adjudication": _normalize_sha256(
            semantic_adjudication_artifact_sha256,
            field_name="semantic_adjudication_artifact_sha256",
        ),
        "transcription_packet": _normalize_sha256(
            transcription_review_packet_sha256,
            field_name="transcription_review_packet_sha256",
        ),
        "transcription_decisions": _normalize_sha256(
            transcription_decision_manifest_sha256,
            field_name="transcription_decision_manifest_sha256",
        ),
        "transcription_adjudication": _normalize_sha256(
            transcription_adjudication_artifact_sha256,
            field_name="transcription_adjudication_artifact_sha256",
        ),
    }
    expected_transcription_adjudication = (
        build_oled_supplementary_source_transcription_adjudication_artifact(
            request_artifact=request,
            request_artifact_sha256=hashes["request"],
            response_manifest=response,
            response_manifest_sha256=hashes["response_manifest"],
            response_artifact=response_validation,
            response_artifact_sha256=hashes["response_artifact"],
            semantic_review_packet=semantic_packet,
            semantic_review_packet_sha256=hashes["semantic_packet"],
            semantic_decision_manifest=semantic_decisions,
            semantic_decision_manifest_sha256=hashes["semantic_decisions"],
            semantic_adjudication_artifact=semantic_adjudication,
            semantic_adjudication_artifact_sha256=hashes["semantic_adjudication"],
            source_pdf_evidence=transcription_packet.source_pdf_evidence,
            review_packet=transcription_packet,
            review_packet_sha256=hashes["transcription_packet"],
            decision_manifest=transcription_decisions,
            decision_manifest_sha256=hashes["transcription_decisions"],
            generated_at=transcription_adjudication.generated_at,
        )
    )
    if expected_transcription_adjudication.model_dump(
        mode="json"
    ) != transcription_adjudication.model_dump(mode="json"):
        raise ValueError("material identity PR-J adjudication replay mismatch")
    if transcription_adjudication.status != (
        OledSupplementarySourceTranscriptionAdjudicationStatus.READY_FOR_LATER_IDENTITY_REVIEW
    ):
        raise ValueError("material identity request requires PR-J identity readiness")
    if (
        transcription_adjudication.unresolved_review_item_count
        or not transcription_adjudication.all_reviewed_scopes_transcription_validated
        or transcription_adjudication.accepted_scope_count
        != transcription_adjudication.review_item_count
        or transcription_adjudication.later_identity_review_eligible_cell_count <= 0
    ):
        raise ValueError("material identity request requires fully accepted PR-J scopes")
    generated_at_clean = _validate_timestamp(generated_at, field_name="generated_at")
    if _parse_timestamp(generated_at_clean) < _parse_timestamp(
        transcription_adjudication.generated_at
    ):
        raise ValueError("material identity request predates PR-J adjudication")

    eligible_semantic_cells = [
        cell
        for cell in semantic_adjudication.adjudicated_cells
        if cell.eligible_for_later_materialization_review
    ]
    _validate_eligible_semantic_cells(eligible_semantic_cells)
    semantic_eligible_by_digest = {
        cell.source_cell.source_cell_digest: cell for cell in eligible_semantic_cells
    }
    if len(semantic_eligible_by_digest) != len(eligible_semantic_cells):
        raise ValueError("material identity semantic eligible cells are duplicated")

    accepted_table_by_key: dict[tuple[str, str], Any] = {}
    transcription_item_by_key = {
        (item.scope_id, item.table_id): item for item in transcription_packet.review_items
    }
    accepted_identity_digests: list[str] = []
    for adjudicated_table in transcription_adjudication.adjudicated_tables:
        if not adjudicated_table.table_transcription_validated:
            continue
        key = (adjudicated_table.scope_id, adjudicated_table.table_id)
        item = transcription_item_by_key.get(key)
        if item is None or key in accepted_table_by_key:
            raise ValueError("material identity accepted table binding mismatch")
        if (
            item.review_item_id != adjudicated_table.review_item_id
            or item.review_item_digest != adjudicated_table.review_item_digest
            or item.table_content_digest != adjudicated_table.table_content_digest
        ):
            raise ValueError("material identity accepted table changed")
        accepted_table_by_key[key] = item
        accepted_identity_digests.extend(
            adjudicated_table.later_identity_review_eligible_source_cell_digests
        )
    if len(accepted_identity_digests) != len(set(accepted_identity_digests)):
        raise ValueError("material identity PR-J eligible cells are duplicated")
    if set(accepted_identity_digests) != set(semantic_eligible_by_digest):
        raise ValueError("material identity PR-I and PR-J eligible rosters differ")
    if len(accepted_identity_digests) != (
        transcription_adjudication.later_identity_review_eligible_cell_count
    ):
        raise ValueError("material identity PR-J eligible count mismatch")

    semantic_scope_by_key = {
        (scope.scope_id, scope.matched_table.table_id): scope
        for scope in semantic_packet.scopes
    }
    grouped_cells: dict[tuple[Any, ...], list[OledSupplementaryAdjudicatedCell]] = {}
    group_context: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
    for digest in sorted(accepted_identity_digests):
        cell = semantic_eligible_by_digest[digest]
        source = cell.source_cell
        table_key = (source.scope_id, source.table_id)
        item = accepted_table_by_key.get(table_key)
        scope = semantic_scope_by_key.get(table_key)
        if item is None or scope is None:
            raise ValueError("material identity eligible cell is outside an accepted table")
        _validate_cell_against_table(cell=cell, item=item, scope=scope)
        header_binding = _subject_header_binding(item, source.subject_column_index)
        key = (
            source.scope_id,
            source.table_id,
            source.table_content_digest,
            source.row_index,
            source.reported_subject_text,
            source.subject_column_index,
            header_binding.parser_key,
            header_binding.source_visible_header_candidate,
            header_binding.binding_kind.value,
        )
        grouped_cells.setdefault(key, []).append(cell)
        group_context[key] = (item, scope, header_binding)

    groups: list[OledSupplementaryMaterialIdentityCandidateGroup] = []
    for key in sorted(grouped_cells, key=lambda item: (item[0], item[1], item[3], item[4])):
        cells = sorted(
            grouped_cells[key],
            key=lambda cell: (
                cell.source_cell.column_index,
                cell.source_cell.source_cell_digest,
            ),
        )
        item, scope, header_binding = group_context[key]
        source = cells[0].source_cell
        dependent_cells = [
            OledSupplementaryMaterialIdentityDependentCell(
                row_index=cell.source_cell.row_index,
                column_index=cell.source_cell.column_index,
                column_name=cell.source_cell.column_name,
                source_cell_digest=cell.source_cell.source_cell_digest,
                cell_disposition_digest=cell.source_cell.cell_disposition_digest,
                semantic_review_item_id=cell.decision_source_review_item_id,
                semantic_review_item_digest=cell.decision_source_review_item_digest,
            )
            for cell in cells
        ]
        base: dict[str, Any] = {
            "identity_group_id": "material-identity-placeholder",
            "identity_group_digest": "sha256:" + "0" * 64,
            "contract_version": SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_CONTRACT_VERSION,
            "scope_id": source.scope_id,
            "source_id": scope.source_id,
            "source_pdf_sha256": scope.source_pdf_sha256,
            "parsed_document_sha256": scope.parsed_document_sha256,
            "table_id": source.table_id,
            "table_content_digest": source.table_content_digest,
            "pdf_page_number_one_based": item.pdf_page_number_one_based,
            "source_transcription_review_item_id": item.review_item_id,
            "source_transcription_review_item_digest": item.review_item_digest,
            "row_index": source.row_index,
            "subject_column_index": source.subject_column_index,
            "subject_header_binding": header_binding,
            "reported_subject_text": source.reported_subject_text,
            "identity_dependent_cell_count": len(dependent_cells),
            "identity_dependent_source_cell_digests": sorted(
                cell.source_cell_digest for cell in dependent_cells
            ),
            "identity_dependent_cells": dependent_cells,
        }
        provisional = OledSupplementaryMaterialIdentityCandidateGroup.model_construct(
            **base
        )
        base["identity_group_id"] = _material_identity_group_id(provisional)
        provisional = OledSupplementaryMaterialIdentityCandidateGroup.model_construct(
            **base
        )
        base["identity_group_digest"] = _material_identity_group_digest(provisional)
        groups.append(
            OledSupplementaryMaterialIdentityCandidateGroup.model_validate(base)
        )

    groups.sort(
        key=lambda group: (
            group.scope_id,
            group.table_id,
            group.row_index,
            group.identity_group_id,
        )
    )
    identities = {
        (artifact.run_id, artifact.paper_id)
        for artifact in (
            request,
            response,
            response_validation,
            semantic_packet,
            semantic_decisions,
            semantic_adjudication,
            transcription_packet,
            transcription_decisions,
            transcription_adjudication,
        )
    }
    if len(identities) != 1:
        raise ValueError("material identity upstream run or paper mismatch")
    ontology_digests = {
        digest
        for item in transcription_packet.review_items
        for digest in item.upstream_ontology_review_pending_source_cell_digests
    }
    if ontology_digests.intersection(accepted_identity_digests):
        raise ValueError("ontology-pending cells entered material identity groups")
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_ARTIFACT_VERSION,
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "generated_at": generated_at_clean,
        "request_artifact_sha256": hashes["request"],
        "request_digest": request.request_digest,
        "response_manifest_sha256": hashes["response_manifest"],
        "response_manifest_digest": response_validation.response_manifest_digest,
        "response_artifact_sha256": hashes["response_artifact"],
        "response_artifact_digest": response_validation.response_artifact_digest,
        "semantic_review_packet_sha256": hashes["semantic_packet"],
        "semantic_review_packet_digest": semantic_packet.review_packet_digest,
        "semantic_decision_manifest_sha256": hashes["semantic_decisions"],
        "semantic_decision_manifest_digest": semantic_adjudication.decision_manifest_digest,
        "semantic_adjudication_artifact_sha256": hashes["semantic_adjudication"],
        "semantic_adjudication_artifact_digest": semantic_adjudication.adjudication_artifact_digest,
        "transcription_review_packet_sha256": hashes["transcription_packet"],
        "transcription_review_packet_digest": transcription_packet.review_packet_digest,
        "transcription_decision_manifest_sha256": hashes["transcription_decisions"],
        "transcription_decision_manifest_digest": (
            transcription_adjudication.decision_manifest_digest
        ),
        "transcription_adjudication_artifact_sha256": hashes["transcription_adjudication"],
        "transcription_adjudication_artifact_digest": (
            transcription_adjudication.adjudication_artifact_digest
        ),
        "source_pdf_evidence_digest": transcription_adjudication.source_pdf_evidence_digest,
        "status": (
            OledSupplementaryMaterialIdentityCandidateRequestStatus
            .READY_FOR_MATERIAL_IDENTITY_EVIDENCE_PROPOSAL
        ),
        "source_count": len({group.source_id for group in groups}),
        "scope_count": len({group.scope_id for group in groups}),
        "accepted_transcription_scope_count": transcription_adjudication.accepted_scope_count,
        "identity_group_count": len(groups),
        "identity_dependent_cell_count": len(accepted_identity_digests),
        "bounded_transcription_validated_cell_count": (
            transcription_adjudication.bounded_transcription_validated_cell_count
        ),
        "upstream_ontology_review_pending_cell_count": (
            transcription_adjudication.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "proposal_instructions": list(_IDENTITY_PROPOSAL_INSTRUCTIONS),
        "identity_groups": groups,
        "material_identity_request_digest": "sha256:" + "0" * 64,
    }
    provisional_artifact = (
        OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_construct(
            **payload
        )
    )
    payload["material_identity_request_digest"] = _material_identity_request_digest(
        provisional_artifact
    )
    return OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        payload
    )


def render_oled_supplementary_material_identity_candidate_request_markdown(
    artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    *,
    request_artifact_sha256: str,
) -> str:
    request = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        artifact.model_dump(mode="json")
    )
    request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    lines = [
        "# Supplementary material-identity evidence request",
        "",
        f"- Paper: `{_markdown_code(request.paper_id)}`",
        f"- Run: `{_markdown_code(request.run_id)}`",
        f"- Request file SHA-256: `{request_sha}`",
        f"- Request digest: `{request.material_identity_request_digest}`",
        f"- Status: `{request.status.value}`",
        f"- Identity groups: **{request.identity_group_count}**",
        f"- Identity-dependent cells: **{request.identity_dependent_cell_count}**",
        "- Ontology-pending cells excluded: "
        f"**{request.upstream_ontology_review_pending_cell_count}**",
        "",
        "## Boundary",
        "",
        (
            "Every item below is one paper-local table row. A reported row label "
            "is not structure evidence, a canonical identity, or permission to "
            "merge records."
        ),
        "",
        "| Item | Reported subject | Row | Subject header | Page | Dependent columns | Cells |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for index, group in enumerate(request.identity_groups, start=1):
        binding = group.subject_header_binding
        if not binding.source_visible_header_candidate:
            header = (
                "no explicit source header (parser key "
                f"{_markdown_code(binding.parser_key)})"
            )
        else:
            header = binding.source_visible_header_candidate
        columns = ", ".join(
            cell.column_name for cell in group.identity_dependent_cells
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    f"I{index:02d}",
                    f"`{_markdown_code(group.reported_subject_text)}`",
                    str(group.row_index),
                    _markdown_table_text(header),
                    str(group.pdf_page_number_one_based),
                    _markdown_table_text(columns),
                    str(group.identity_dependent_cell_count),
                )
            )
            + " |"
        )
    lines.extend(("", "## Exact group bindings", ""))
    for index, group in enumerate(request.identity_groups, start=1):
        lines.extend(
            (
                f"### I{index:02d}: `{_markdown_code(group.reported_subject_text)}`",
                "",
                f"- Group id: `{group.identity_group_id}`",
                f"- Group digest: `{group.identity_group_digest}`",
                f"- Scope: `{group.scope_id}`",
                f"- Table: `{group.table_id}`",
                f"- Table digest: `{group.table_content_digest}`",
                f"- Source PDF SHA-256: `{group.source_pdf_sha256}`",
                f"- Zero-based row: `{group.row_index}`",
                "- Structure evidence included: `false`",
                "- Material identity resolved: `false`",
                "- Dependent source-cell digests:",
                "",
            )
        )
        lines.extend(
            f"  - `{digest}`"
            for digest in group.identity_dependent_source_cell_digests
        )
        lines.append("")
    lines.extend(("## Proposal instructions", ""))
    lines.extend(
        f"{index}. {_markdown_text(instruction)}"
        for index, instruction in enumerate(request.proposal_instructions, start=1)
    )
    lines.extend(
        (
            "",
            (
                "A later response remains a proposal. It cannot create canonical "
                "identity, Registry, Schema, Gold, training, or dataset records "
                "without independent validation and human adjudication."
            ),
            "",
        )
    )
    return "\n".join(lines)


def _validate_eligible_semantic_cells(
    cells: list[OledSupplementaryAdjudicatedCell],
) -> None:
    if not cells:
        raise ValueError("material identity request has no PR-I eligible cells")
    for cell in cells:
        if (
            cell.blocked_by_scope_semantics
            or cell.ontology_review_pending
            or cell.source_check_pending
            or cell.exclusion_confirmed
            or cell.rejected
            or cell.rejected_by_scope
        ):
            raise ValueError("blocked PR-I cell entered material identity eligibility")


def _validate_cell_against_table(*, cell: Any, item: Any, scope: Any) -> None:
    source = cell.source_cell
    table = item.matched_table
    if (
        source.table_content_digest != item.table_content_digest
        or scope.matched_table.model_dump(mode="json") != table.model_dump(mode="json")
        or source.row_index >= len(table.rows)
        or source.column_index >= len(table.headers)
        or source.subject_column_index >= len(table.headers)
        or table.headers[source.column_index] != source.column_name
        or table.headers[source.subject_column_index] != source.subject_column_name
    ):
        raise ValueError("material identity cell table binding mismatch")
    row = table.rows[source.row_index]
    if (
        row[source.column_name] != source.cell_value
        or row[source.subject_column_name] != source.reported_subject_text
    ):
        raise ValueError("material identity cell row or subject binding mismatch")


def _subject_header_binding(
    item: Any,
    subject_column_index: int,
) -> OledSupplementarySourceHeaderReviewBinding:
    bindings = [
        binding
        for binding in item.header_review_bindings
        if binding.column_index == subject_column_index
    ]
    if len(bindings) != 1:
        raise ValueError("material identity subject header binding is unavailable")
    binding = bindings[0]
    if binding.parser_key != item.matched_table.headers[subject_column_index]:
        raise ValueError("material identity subject parser key mismatch")
    return binding


def _material_identity_group_id(
    group: OledSupplementaryMaterialIdentityCandidateGroup,
) -> str:
    identity = {
        "contract_version": group.contract_version,
        "scope_id": group.scope_id,
        "source_id": group.source_id,
        "source_pdf_sha256": group.source_pdf_sha256,
        "table_id": group.table_id,
        "table_content_digest": group.table_content_digest,
        "row_index": group.row_index,
        "subject_column_index": group.subject_column_index,
        "subject_header_binding": group.subject_header_binding.model_dump(mode="json")
        if isinstance(group.subject_header_binding, BaseModel)
        else group.subject_header_binding,
        "reported_subject_text": group.reported_subject_text,
    }
    return f"supplementary-material-identity:{_stable_hash(identity)[7:31]}"


def _material_identity_group_digest(
    group: OledSupplementaryMaterialIdentityCandidateGroup,
) -> str:
    payload = group.model_dump(mode="json")
    payload.pop("identity_group_digest", None)
    return _stable_hash(payload)


def _material_identity_request_digest(
    artifact: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("material_identity_request_digest", None)
    return _stable_hash(payload)


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _normalize_sha256(value: str, *, field_name: str) -> str:
    match = _SHA256_RE.fullmatch(str(value or ""))
    if match is None:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_path_segment(value: str, *, field_name: str) -> str:
    clean = str(value or "")
    if not clean or not _SAFE_PATH_SEGMENT_RE.fullmatch(clean):
        raise ValueError(f"{field_name} must be a safe path segment")
    return clean


def _validate_bound_id(value: str, *, field_name: str) -> str:
    clean = str(value or "")
    if not clean or not _SAFE_BOUND_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} must be a safe bound id")
    return clean


def _validate_source_literal(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 4_000:
        raise ValueError(f"{field_name} is required")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{field_name} contains unsafe control text")
    return value


def _validate_timestamp(value: str, *, field_name: str) -> str:
    clean = str(value or "")
    parsed = _parse_timestamp(clean)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return clean


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid material identity timestamp") from exc


def _markdown_text(value: str) -> str:
    visible = "".join(
        _visible_unicode_escape(character)
        if unicodedata.category(character) in {"Cc", "Cf", "Zl", "Zp"}
        else character
        for character in value
    )
    return html.escape(visible, quote=True)


def _visible_unicode_escape(character: str) -> str:
    codepoint = ord(character)
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04X}"
    return f"\\U{codepoint:08X}"


def _markdown_code(value: str) -> str:
    return _markdown_text(value).replace("`", "&#96;").replace("|", "&#124;")


def _markdown_table_text(value: str) -> str:
    return _markdown_text(value).replace("|", "&#124;").replace("`", "&#96;")


__all__ = [
    "OledSupplementaryMaterialIdentityCandidateGroup",
    "OledSupplementaryMaterialIdentityCandidateRequestArtifact",
    "OledSupplementaryMaterialIdentityCandidateRequestStatus",
    "OledSupplementaryMaterialIdentityDependentCell",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_ARTIFACT_VERSION",
    "SUPPLEMENTARY_MATERIAL_IDENTITY_CANDIDATE_REQUEST_CONTRACT_VERSION",
    "build_oled_supplementary_material_identity_candidate_request_artifact",
    "render_oled_supplementary_material_identity_candidate_request_markdown",
]
