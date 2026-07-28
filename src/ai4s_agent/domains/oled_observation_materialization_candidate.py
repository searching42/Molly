from __future__ import annotations

import hashlib
import json
import math
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_layered_schema import (
    OledComparisonContextStatus,
    OledDeviceLayer,
    OledEvidenceSource,
    OledEvidenceType,
    OledInteractionLayer,
    OledLayeredCanonicalObservation,
    OledLayeredRecord,
    OledMeasurementCondition,
    OledMeasurementLayer,
    OledMolecularLayer,
    OledPropertyObservation,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
)
from ai4s_agent.domains.oled_observation_staging_preflight import (
    OledObservationStagingPreflightArtifact,
    oled_observation_staging_preflight_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityCandidateGroup,
    OledSupplementaryMaterialIdentityCandidateRequestArtifact,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)
from ai4s_agent.domains.oled_supplementary_semantic_review import (
    OledSupplementaryAdjudicatedCell,
    OledSupplementaryAdjudicatedGroup,
    OledSupplementaryKnownMappingSummary,
    OledSupplementarySemanticAdjudicationArtifact,
    OledSupplementarySemanticReviewCell,
)
from ai4s_agent.domains.oled_supplementary_source_transcription_review import (
    OledSupplementarySourceTranscriptionAdjudicationArtifact,
    OledSupplementarySourceTranscriptionReviewItem,
    OledSupplementarySourceTranscriptionReviewPacket,
)


OLED_OBSERVATION_MATERIALIZATION_CANDIDATE_VERSION = (
    "oled_observation_materialization_candidate.v1"
)


class OledObservationMaterializationCandidateStatus(str, Enum):
    READY_FOR_REVIEWED_EVIDENCE_STAGING_PREFLIGHT = (
        "ready_for_reviewed_evidence_staging_preflight"
    )
    NO_RESOLVED_OBSERVATION_CANDIDATES = "no_resolved_observation_candidates"


class OledObservationMaterializationCandidateItem(BaseModel):
    """One exact, resolved material/property observation candidate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    staging_item_id: str
    staging_item_digest: str
    identity_group_id: str
    identity_group_digest: str
    selected_existing_material_id: str
    selected_registry_entry: OledMaterialRegistryEntry
    reported_subject_text: str
    scope_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    row_index: Annotated[StrictInt, Field(ge=0)]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    source_cell_digest: str
    cell_disposition_digest: str
    semantic_review_item_id: str
    semantic_review_item_digest: str
    source_transcription_review_item_id: str
    source_transcription_review_item_digest: str
    semantic_source_cell: OledSupplementarySemanticReviewCell
    mapping_summary: OledSupplementaryKnownMappingSummary
    property_observation: OledPropertyObservation
    canonical_observation: OledLayeredCanonicalObservation
    comparison_context_status: OledComparisonContextStatus
    comparison_context_required_fields: list[str] = Field(default_factory=list)
    comparison_context_missing_fields: list[str] = Field(default_factory=list)
    comparison_ready: StrictBool
    candidate_digest: str
    exact_reported_literal_replayed: StrictBool = True
    exact_reported_precision_replayed: StrictBool = True
    exact_reported_unit_replayed: StrictBool = True
    human_property_mapping_confirmed: StrictBool = True
    human_source_transcription_confirmed: StrictBool = True
    human_registry_mapping_confirmed: StrictBool = True
    material_id_attached_to_observation: StrictBool = True
    observation_candidate_materialized: StrictBool = True
    reviewed_evidence_staged: StrictBool = False
    direct_admission_eligible: StrictBool = False
    gold_record_created: StrictBool = False

    @field_validator(
        "candidate_id",
        "staging_item_id",
        "identity_group_id",
        "selected_existing_material_id",
        "scope_id",
        "table_id",
        "semantic_review_item_id",
        "source_transcription_review_item_id",
    )
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("reported_subject_text", "column_name")
    @classmethod
    def validate_source_literals(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > 20_000:
            raise ValueError(f"{info.field_name} is required and bounded")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError(f"{info.field_name} contains control text")
        return value

    @field_validator(
        "staging_item_digest",
        "identity_group_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
        "source_cell_digest",
        "cell_disposition_digest",
        "semantic_review_item_digest",
        "source_transcription_review_item_digest",
        "candidate_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "comparison_context_required_fields",
        "comparison_context_missing_fields",
    )
    @classmethod
    def validate_context_fields(cls, value: list[str], info: Any) -> list[str]:
        clean = [str(item).strip() for item in value]
        if any(not item for item in clean) or clean != sorted(clean) or len(clean) != len(
            set(clean)
        ):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_candidate_integrity(
        self,
    ) -> OledObservationMaterializationCandidateItem:
        if self.candidate_id != f"observation-candidate:{self.source_cell_digest[7:]}":
            raise ValueError("observation candidate_id mismatch")
        if self.selected_registry_entry.material_id != self.selected_existing_material_id:
            raise ValueError("observation candidate Registry entry mismatch")
        source_cell = self.semantic_source_cell
        expected_source = {
            "scope_id": self.scope_id,
            "table_id": self.table_id,
            "table_content_digest": self.table_content_digest,
            "row_index": self.row_index,
            "column_index": self.column_index,
            "column_name": self.column_name,
            "reported_subject_text": self.reported_subject_text,
            "source_cell_digest": self.source_cell_digest,
            "cell_disposition_digest": self.cell_disposition_digest,
        }
        for field_name, expected in expected_source.items():
            if getattr(source_cell, field_name) != expected:
                raise ValueError(f"observation candidate source {field_name} mismatch")
        mapping = self.mapping_summary
        observation = self.property_observation
        canonical = self.canonical_observation
        if (
            mapping.column_index != self.column_index
            or mapping.column_name != self.column_name
            or observation.property_label != mapping.property_id
            or observation.reported_value_text != source_cell.reported_value_text
            or observation.reported_decimal_places != source_cell.reported_decimal_places
            or observation.unit != mapping.reported_unit
            or canonical.layer != mapping.target_layer
            or canonical.property_id != mapping.property_id
            or canonical.raw_property_label != mapping.property_id
            or canonical.unit != mapping.reported_unit
            or canonical.normalized_unit != mapping.canonical_unit
            or canonical.reported_value_text != source_cell.reported_value_text
            or canonical.reported_decimal_places != source_cell.reported_decimal_places
        ):
            raise ValueError("observation candidate mapping or reported value mismatch")
        if canonical.value != observation.value or canonical.condition != observation.condition:
            raise ValueError("observation candidate canonical replay mismatch")
        if len(observation.evidence_sources) != 1:
            raise ValueError("observation candidate requires one exact source-cell evidence")
        evidence = observation.evidence_sources[0]
        if evidence.layer != mapping.target_layer or evidence.source_type != (
            OledEvidenceType.SUPPLEMENTARY
        ):
            raise ValueError("observation candidate evidence layer mismatch")
        expected_context = _condition_from_mapping(mapping)
        if observation.condition != expected_context:
            raise ValueError("observation candidate comparison context mismatch")
        if (
            self.comparison_context_status != canonical.comparison_context_status
            or self.comparison_context_required_fields
            != sorted(canonical.comparison_context_required_fields)
            or self.comparison_context_missing_fields
            != sorted(canonical.comparison_context_missing_fields)
            or self.comparison_ready != canonical.is_comparison_ready
        ):
            raise ValueError("observation candidate comparison assessment mismatch")
        fixed_true = (
            "exact_reported_literal_replayed",
            "exact_reported_precision_replayed",
            "exact_reported_unit_replayed",
            "human_property_mapping_confirmed",
            "human_source_transcription_confirmed",
            "human_registry_mapping_confirmed",
            "material_id_attached_to_observation",
            "observation_candidate_materialized",
        )
        fixed_false = (
            "reviewed_evidence_staged",
            "direct_admission_eligible",
            "gold_record_created",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true) or any(
            getattr(self, field_name) for field_name in fixed_false
        ):
            raise ValueError("observation candidate crossed its materialization boundary")
        if _observation_candidate_item_digest(self) != self.candidate_digest:
            raise ValueError("observation candidate item digest mismatch")
        return self


class OledObservationMaterializationCandidateArtifact(BaseModel):
    """Self-contained exact-chain candidates; never a dataset or Gold write."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_OBSERVATION_MATERIALIZATION_CANDIDATE_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    staging_preflight_sha256: str
    staging_preflight_digest: str
    material_identity_request_sha256: str
    material_identity_request_digest: str
    semantic_adjudication_sha256: str
    semantic_adjudication_digest: str
    transcription_review_packet_sha256: str
    transcription_review_packet_digest: str
    transcription_adjudication_sha256: str
    transcription_adjudication_digest: str
    staging_preflight: OledObservationStagingPreflightArtifact
    material_identity_request: OledSupplementaryMaterialIdentityCandidateRequestArtifact
    semantic_adjudication: OledSupplementarySemanticAdjudicationArtifact
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket
    transcription_adjudication: OledSupplementarySourceTranscriptionAdjudicationArtifact
    status: OledObservationMaterializationCandidateStatus
    source_staging_item_count: Annotated[StrictInt, Field(ge=0)]
    source_staging_cell_count: Annotated[StrictInt, Field(ge=0)]
    observation_candidate_count: Annotated[StrictInt, Field(ge=0)]
    comparison_ready_candidate_count: Annotated[StrictInt, Field(ge=0)]
    comparison_context_not_required_count: Annotated[StrictInt, Field(ge=0)]
    comparison_context_complete_count: Annotated[StrictInt, Field(ge=0)]
    comparison_context_incomplete_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    observation_candidates: list[OledObservationMaterializationCandidateItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    artifact_digest: str
    candidate_only: StrictBool = True
    offline_only: StrictBool = True
    exact_staging_preflight_bytes_bound: StrictBool = True
    exact_material_identity_request_bytes_bound: StrictBool = True
    exact_semantic_adjudication_bytes_bound: StrictBool = True
    exact_transcription_review_packet_bytes_bound: StrictBool = True
    exact_transcription_adjudication_bytes_bound: StrictBool = True
    complete_exact_chain_rejoined: StrictBool = True
    source_property_values_present: StrictBool
    reported_literals_and_precision_replayed: StrictBool = True
    property_mapping_and_context_replayed: StrictBool = True
    bounded_source_transcription_replayed: StrictBool = True
    material_id_attached_to_observations: StrictBool
    observations_materialized: StrictBool
    comparison_readiness_assessed: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    source_pdf_read: StrictBool = False
    raw_parsed_document_read: StrictBool = False
    source_values_corrected: StrictBool = False
    ontology_extensions_applied: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    registry_written: StrictBool = False
    aliases_mutated: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_OBSERVATION_MATERIALIZATION_CANDIDATE_VERSION:
            raise ValueError("unexpected observation materialization candidate version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator(
        "staging_preflight_sha256",
        "staging_preflight_digest",
        "material_identity_request_sha256",
        "material_identity_request_digest",
        "semantic_adjudication_sha256",
        "semantic_adjudication_digest",
        "transcription_review_packet_sha256",
        "transcription_review_packet_digest",
        "transcription_adjudication_sha256",
        "transcription_adjudication_digest",
        "artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("observation_candidates")
    @classmethod
    def validate_candidate_order(
        cls,
        value: list[OledObservationMaterializationCandidateItem],
    ) -> list[OledObservationMaterializationCandidateItem]:
        order = [item.candidate_id for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("observation candidates must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledObservationMaterializationCandidateArtifact:
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.staging_preflight.generated_at
        ):
            raise ValueError("observation materialization predates PR-P")
        expected_bindings = {
            "run_id": self.staging_preflight.run_id,
            "paper_id": self.staging_preflight.paper_id,
            "staging_preflight_digest": self.staging_preflight.preflight_artifact_digest,
            "material_identity_request_digest": (
                self.material_identity_request.material_identity_request_digest
            ),
            "semantic_adjudication_digest": (
                self.semantic_adjudication.adjudication_artifact_digest
            ),
            "transcription_review_packet_digest": (
                self.transcription_review_packet.review_packet_digest
            ),
            "transcription_adjudication_digest": (
                self.transcription_adjudication.adjudication_artifact_digest
            ),
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"observation materialization {field_name} mismatch")
        if oled_observation_staging_preflight_artifact_digest(
            self.staging_preflight
        ) != self.staging_preflight_digest:
            raise ValueError("observation materialization embedded PR-P digest mismatch")
        _validate_exact_chain(
            staging=self.staging_preflight,
            material_identity_request=self.material_identity_request,
            material_identity_request_sha256=self.material_identity_request_sha256,
            semantic_adjudication=self.semantic_adjudication,
            semantic_adjudication_sha256=self.semantic_adjudication_sha256,
            transcription_packet=self.transcription_review_packet,
            transcription_packet_sha256=self.transcription_review_packet_sha256,
            transcription_adjudication=self.transcription_adjudication,
            transcription_adjudication_sha256=self.transcription_adjudication_sha256,
        )
        expected_candidates = _derive_observation_candidates(
            staging=self.staging_preflight,
            material_identity_request=self.material_identity_request,
            semantic_adjudication=self.semantic_adjudication,
            transcription_packet=self.transcription_review_packet,
            transcription_adjudication=self.transcription_adjudication,
        )
        if [item.model_dump(mode="json") for item in self.observation_candidates] != [
            item.model_dump(mode="json") for item in expected_candidates
        ]:
            raise ValueError("observation candidate derivation mismatch")
        context_counts = _comparison_context_counts(expected_candidates)
        expected_counts = {
            "source_staging_item_count": self.staging_preflight.staging_item_count,
            "source_staging_cell_count": self.staging_preflight.staging_cell_count,
            "observation_candidate_count": len(expected_candidates),
            "comparison_ready_candidate_count": sum(
                item.comparison_ready for item in expected_candidates
            ),
            "comparison_context_not_required_count": context_counts[
                OledComparisonContextStatus.NOT_REQUIRED
            ],
            "comparison_context_complete_count": context_counts[
                OledComparisonContextStatus.COMPLETE
            ],
            "comparison_context_incomplete_count": context_counts[
                OledComparisonContextStatus.INCOMPLETE
            ],
            "upstream_ontology_review_pending_cell_count": (
                self.staging_preflight.upstream_ontology_review_pending_cell_count
            ),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"observation materialization {field_name} mismatch")
        expected_status = (
            OledObservationMaterializationCandidateStatus
            .READY_FOR_REVIEWED_EVIDENCE_STAGING_PREFLIGHT
            if expected_candidates
            else OledObservationMaterializationCandidateStatus
            .NO_RESOLVED_OBSERVATION_CANDIDATES
        )
        if self.status != expected_status or self.device_only_cell_count != 0:
            raise ValueError("observation materialization status or device boundary mismatch")
        has_candidates = bool(expected_candidates)
        if any(
            getattr(self, field_name) != has_candidates
            for field_name in (
                "source_property_values_present",
                "material_id_attached_to_observations",
                "observations_materialized",
            )
        ):
            raise ValueError("observation materialization presence flags mismatch")
        fixed_true = (
            "candidate_only",
            "offline_only",
            "exact_staging_preflight_bytes_bound",
            "exact_material_identity_request_bytes_bound",
            "exact_semantic_adjudication_bytes_bound",
            "exact_transcription_review_packet_bytes_bound",
            "exact_transcription_adjudication_bytes_bound",
            "complete_exact_chain_rejoined",
            "reported_literals_and_precision_replayed",
            "property_mapping_and_context_replayed",
            "bounded_source_transcription_replayed",
            "comparison_readiness_assessed",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "source_pdf_read",
            "raw_parsed_document_read",
            "source_values_corrected",
            "ontology_extensions_applied",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "registry_written",
            "aliases_mutated",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true) or any(
            getattr(self, field_name) for field_name in fixed_false
        ):
            raise ValueError("observation materialization crossed its candidate boundary")
        if oled_observation_materialization_candidate_artifact_digest(self) != (
            self.artifact_digest
        ):
            raise ValueError("observation materialization artifact digest mismatch")
        return self


def build_oled_observation_materialization_candidate_artifact(
    *,
    staging_preflight: OledObservationStagingPreflightArtifact,
    staging_preflight_sha256: str,
    material_identity_request: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    material_identity_request_sha256: str,
    semantic_adjudication: OledSupplementarySemanticAdjudicationArtifact,
    semantic_adjudication_sha256: str,
    transcription_review_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_review_packet_sha256: str,
    transcription_adjudication: OledSupplementarySourceTranscriptionAdjudicationArtifact,
    transcription_adjudication_sha256: str,
    generated_at: str,
) -> OledObservationMaterializationCandidateArtifact:
    staging = OledObservationStagingPreflightArtifact.model_validate(
        staging_preflight.model_dump(mode="json")
    )
    identity_request = OledSupplementaryMaterialIdentityCandidateRequestArtifact.model_validate(
        material_identity_request.model_dump(mode="json")
    )
    semantic = OledSupplementarySemanticAdjudicationArtifact.model_validate(
        semantic_adjudication.model_dump(mode="json")
    )
    transcription_packet = OledSupplementarySourceTranscriptionReviewPacket.model_validate(
        transcription_review_packet.model_dump(mode="json")
    )
    transcription = OledSupplementarySourceTranscriptionAdjudicationArtifact.model_validate(
        transcription_adjudication.model_dump(mode="json")
    )
    normalized_hashes = {
        "staging": _normalize_sha256(
            staging_preflight_sha256,
            field_name="staging_preflight_sha256",
        ),
        "identity_request": _normalize_sha256(
            material_identity_request_sha256,
            field_name="material_identity_request_sha256",
        ),
        "semantic": _normalize_sha256(
            semantic_adjudication_sha256,
            field_name="semantic_adjudication_sha256",
        ),
        "transcription_packet": _normalize_sha256(
            transcription_review_packet_sha256,
            field_name="transcription_review_packet_sha256",
        ),
        "transcription": _normalize_sha256(
            transcription_adjudication_sha256,
            field_name="transcription_adjudication_sha256",
        ),
    }
    _validate_exact_chain(
        staging=staging,
        material_identity_request=identity_request,
        material_identity_request_sha256=normalized_hashes["identity_request"],
        semantic_adjudication=semantic,
        semantic_adjudication_sha256=normalized_hashes["semantic"],
        transcription_packet=transcription_packet,
        transcription_packet_sha256=normalized_hashes["transcription_packet"],
        transcription_adjudication=transcription,
        transcription_adjudication_sha256=normalized_hashes["transcription"],
    )
    candidates = _derive_observation_candidates(
        staging=staging,
        material_identity_request=identity_request,
        semantic_adjudication=semantic,
        transcription_packet=transcription_packet,
        transcription_adjudication=transcription,
    )
    context_counts = _comparison_context_counts(candidates)
    payload: dict[str, Any] = {
        "artifact_version": OLED_OBSERVATION_MATERIALIZATION_CANDIDATE_VERSION,
        "run_id": staging.run_id,
        "paper_id": staging.paper_id,
        "generated_at": generated_at,
        "staging_preflight_sha256": normalized_hashes["staging"],
        "staging_preflight_digest": staging.preflight_artifact_digest,
        "material_identity_request_sha256": normalized_hashes["identity_request"],
        "material_identity_request_digest": identity_request.material_identity_request_digest,
        "semantic_adjudication_sha256": normalized_hashes["semantic"],
        "semantic_adjudication_digest": semantic.adjudication_artifact_digest,
        "transcription_review_packet_sha256": normalized_hashes["transcription_packet"],
        "transcription_review_packet_digest": transcription_packet.review_packet_digest,
        "transcription_adjudication_sha256": normalized_hashes["transcription"],
        "transcription_adjudication_digest": transcription.adjudication_artifact_digest,
        "staging_preflight": staging,
        "material_identity_request": identity_request,
        "semantic_adjudication": semantic,
        "transcription_review_packet": transcription_packet,
        "transcription_adjudication": transcription,
        "status": (
            OledObservationMaterializationCandidateStatus
            .READY_FOR_REVIEWED_EVIDENCE_STAGING_PREFLIGHT
            if candidates
            else OledObservationMaterializationCandidateStatus
            .NO_RESOLVED_OBSERVATION_CANDIDATES
        ),
        "source_staging_item_count": staging.staging_item_count,
        "source_staging_cell_count": staging.staging_cell_count,
        "observation_candidate_count": len(candidates),
        "comparison_ready_candidate_count": sum(
            item.comparison_ready for item in candidates
        ),
        "comparison_context_not_required_count": context_counts[
            OledComparisonContextStatus.NOT_REQUIRED
        ],
        "comparison_context_complete_count": context_counts[
            OledComparisonContextStatus.COMPLETE
        ],
        "comparison_context_incomplete_count": context_counts[
            OledComparisonContextStatus.INCOMPLETE
        ],
        "upstream_ontology_review_pending_cell_count": (
            staging.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "observation_candidates": candidates,
        "artifact_digest": "sha256:" + "0" * 64,
        "source_property_values_present": bool(candidates),
        "material_id_attached_to_observations": bool(candidates),
        "observations_materialized": bool(candidates),
    }
    provisional = OledObservationMaterializationCandidateArtifact.model_construct(
        **payload
    )
    payload["artifact_digest"] = (
        oled_observation_materialization_candidate_artifact_digest(provisional)
    )
    return OledObservationMaterializationCandidateArtifact.model_validate(payload)


def oled_observation_materialization_candidate_artifact_digest(
    artifact: OledObservationMaterializationCandidateArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("artifact_digest", None)
    return _stable_hash(payload)


def _validate_exact_chain(
    *,
    staging: OledObservationStagingPreflightArtifact,
    material_identity_request: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    material_identity_request_sha256: str,
    semantic_adjudication: OledSupplementarySemanticAdjudicationArtifact,
    semantic_adjudication_sha256: str,
    transcription_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_packet_sha256: str,
    transcription_adjudication: OledSupplementarySourceTranscriptionAdjudicationArtifact,
    transcription_adjudication_sha256: str,
) -> None:
    identity_request_sha = _normalize_sha256(
        material_identity_request_sha256,
        field_name="material_identity_request_sha256",
    )
    semantic_sha = _normalize_sha256(
        semantic_adjudication_sha256,
        field_name="semantic_adjudication_sha256",
    )
    transcription_packet_sha = _normalize_sha256(
        transcription_packet_sha256,
        field_name="transcription_review_packet_sha256",
    )
    transcription_sha = _normalize_sha256(
        transcription_adjudication_sha256,
        field_name="transcription_adjudication_sha256",
    )
    pr_m = staging.resolution_request.source_adjudication
    if (
        pr_m.material_identity_request_digest
        != material_identity_request.material_identity_request_digest
    ):
        raise ValueError("PR-M is not semantically bound to the supplied PR-K request")
    expected = {
        "run_id": staging.run_id,
        "paper_id": staging.paper_id,
    }
    for artifact_name, artifact in (
        ("PR-K", material_identity_request),
        ("PR-I", semantic_adjudication),
        ("PR-J packet", transcription_packet),
        ("PR-J adjudication", transcription_adjudication),
    ):
        for field_name, value in expected.items():
            if getattr(artifact, field_name) != value:
                raise ValueError(f"{artifact_name} {field_name} differs from PR-P")
    exact_hash_bindings = {
        "PR-K": (pr_m.request_artifact_sha256, identity_request_sha),
        "PR-I": (
            material_identity_request.semantic_adjudication_artifact_sha256,
            semantic_sha,
        ),
        "PR-J packet to PR-I": (
            transcription_packet.semantic_adjudication_artifact_sha256,
            semantic_sha,
        ),
        "PR-J adjudication to PR-I": (
            transcription_adjudication.semantic_adjudication_artifact_sha256,
            semantic_sha,
        ),
        "PR-J packet": (
            material_identity_request.transcription_review_packet_sha256,
            transcription_packet_sha,
        ),
        "PR-J adjudication": (
            material_identity_request.transcription_adjudication_artifact_sha256,
            transcription_sha,
        ),
    }
    for label, (bound, actual) in exact_hash_bindings.items():
        if bound != actual:
            raise ValueError(f"{label} is not the exact file bound by the downstream chain")
    semantic_digest = semantic_adjudication.adjudication_artifact_digest
    packet_digest = transcription_packet.review_packet_digest
    transcription_digest = transcription_adjudication.adjudication_artifact_digest
    digest_bindings = {
        "PR-I": (
            material_identity_request.semantic_adjudication_artifact_digest,
            semantic_digest,
        ),
        "PR-J packet": (
            material_identity_request.transcription_review_packet_digest,
            packet_digest,
        ),
        "PR-J adjudication": (
            material_identity_request.transcription_adjudication_artifact_digest,
            transcription_digest,
        ),
        "PR-J packet to PR-I": (
            transcription_packet.semantic_adjudication_artifact_digest,
            semantic_digest,
        ),
        "PR-J adjudication to PR-I": (
            transcription_adjudication.semantic_adjudication_artifact_digest,
            semantic_digest,
        ),
        "PR-J adjudication to packet": (
            transcription_adjudication.review_packet_digest,
            packet_digest,
        ),
    }
    for label, (bound, actual) in digest_bindings.items():
        if bound != actual:
            raise ValueError(f"{label} semantic binding mismatch")
    if transcription_adjudication.review_packet_sha256 != transcription_packet_sha:
        raise ValueError("PR-J adjudication is not bound to the supplied packet bytes")
    if not (
        material_identity_request.source_pdf_evidence_digest
        == transcription_packet.source_pdf_evidence_digest
        == transcription_adjudication.source_pdf_evidence_digest
    ):
        raise ValueError("PR-K/PR-J source PDF evidence binding mismatch")
    if _parse_timestamp(transcription_packet.generated_at) < _parse_timestamp(
        semantic_adjudication.generated_at
    ):
        raise ValueError("PR-J packet predates the exact PR-I adjudication")
    if _parse_timestamp(transcription_adjudication.reviewed_at) < _parse_timestamp(
        transcription_packet.generated_at
    ):
        raise ValueError("PR-J human review predates the exact PR-J packet")
    if (
        _parse_timestamp(material_identity_request.generated_at)
        > _parse_timestamp(pr_m.reviewed_at)
        or _parse_timestamp(semantic_adjudication.generated_at)
        > _parse_timestamp(material_identity_request.generated_at)
        or _parse_timestamp(transcription_adjudication.generated_at)
        > _parse_timestamp(material_identity_request.generated_at)
    ):
        raise ValueError("observation materialization upstream causal order mismatch")


def _derive_observation_candidates(
    *,
    staging: OledObservationStagingPreflightArtifact,
    material_identity_request: OledSupplementaryMaterialIdentityCandidateRequestArtifact,
    semantic_adjudication: OledSupplementarySemanticAdjudicationArtifact,
    transcription_packet: OledSupplementarySourceTranscriptionReviewPacket,
    transcription_adjudication: OledSupplementarySourceTranscriptionAdjudicationArtifact,
) -> list[OledObservationMaterializationCandidateItem]:
    identity_groups = {
        group.identity_group_id: group
        for group in material_identity_request.identity_groups
    }
    semantic_cells = {
        item.source_cell.source_cell_digest: item
        for item in semantic_adjudication.adjudicated_cells
        if item.eligible_for_later_materialization_review
    }
    semantic_groups = {
        item.review_item_id: item
        for item in semantic_adjudication.adjudicated_groups
        if item.eligible_for_later_materialization_review
    }
    transcription_items = {
        (item.scope_id, item.table_id): item
        for item in transcription_packet.review_items
    }
    accepted_tables = {
        (item.scope_id, item.table_id): item
        for item in transcription_adjudication.adjudicated_tables
        if item.table_transcription_validated
    }
    candidates: list[OledObservationMaterializationCandidateItem] = []
    seen_source_cells: set[str] = set()
    for staging_item in staging.staging_items:
        identity_group = identity_groups.get(staging_item.identity_group_id)
        if identity_group is None:
            raise ValueError("PR-P identity group is absent from exact PR-K")
        _validate_staging_group(staging_item, identity_group)
        table_key = (staging_item.scope_id, staging_item.table_id)
        transcription_item = transcription_items.get(table_key)
        accepted_table = accepted_tables.get(table_key)
        if transcription_item is None or accepted_table is None:
            raise ValueError("PR-P staging table lacks an accepted exact PR-J table")
        if (
            accepted_table.review_item_id != transcription_item.review_item_id
            or accepted_table.review_item_digest != transcription_item.review_item_digest
            or accepted_table.table_content_digest != transcription_item.table_content_digest
        ):
            raise ValueError("accepted PR-J table differs from its exact review packet")
        accepted_cell_digests = set(
            accepted_table.later_identity_review_eligible_source_cell_digests
        )
        for cell_ref in staging_item.identity_dependent_cells:
            if cell_ref.source_cell_digest in seen_source_cells:
                raise ValueError("observation materialization source cell is duplicated")
            seen_source_cells.add(cell_ref.source_cell_digest)
            adjudicated_cell = semantic_cells.get(cell_ref.source_cell_digest)
            group = semantic_groups.get(cell_ref.semantic_review_item_id)
            if adjudicated_cell is None or group is None:
                raise ValueError("PR-P cell is not eligible in exact PR-I adjudication")
            _validate_semantic_binding(cell_ref, adjudicated_cell, group)
            if cell_ref.source_cell_digest not in accepted_cell_digests:
                raise ValueError("PR-P cell is not eligible in accepted PR-J table")
            source_cell = adjudicated_cell.source_cell
            _validate_transcribed_source_cell(
                source_cell=source_cell,
                staging_item=staging_item,
                transcription_item=transcription_item,
            )
            summary = group.disposition_summary
            if not isinstance(summary, OledSupplementaryKnownMappingSummary):
                raise ValueError("observation candidate lacks an accepted known mapping")
            candidates.append(
                _build_candidate_item(
                    staging_item=staging_item,
                    identity_group=identity_group,
                    cell=source_cell,
                    mapping=summary,
                    transcription_item=transcription_item,
                )
            )
    candidates.sort(key=lambda item: item.candidate_id)
    if len(candidates) != staging.staging_cell_count:
        raise ValueError("observation candidate coverage differs from PR-P staging cells")
    return candidates


def _validate_staging_group(
    staging_item: Any,
    identity_group: OledSupplementaryMaterialIdentityCandidateGroup,
) -> None:
    expected = {
        "identity_group_digest": identity_group.identity_group_digest,
        "reported_subject_text": identity_group.reported_subject_text,
        "scope_id": identity_group.scope_id,
        "source_id": identity_group.source_id,
        "source_pdf_sha256": identity_group.source_pdf_sha256,
        "parsed_document_sha256": identity_group.parsed_document_sha256,
        "table_id": identity_group.table_id,
        "table_content_digest": identity_group.table_content_digest,
        "pdf_page_number_one_based": identity_group.pdf_page_number_one_based,
        "row_index": identity_group.row_index,
        "subject_column_index": identity_group.subject_column_index,
        "identity_dependent_cell_count": identity_group.identity_dependent_cell_count,
    }
    for field_name, value in expected.items():
        if getattr(staging_item, field_name) != value:
            raise ValueError(f"PR-P/PR-K identity group {field_name} mismatch")
    if [cell.model_dump(mode="json") for cell in staging_item.identity_dependent_cells] != [
        cell.model_dump(mode="json") for cell in identity_group.identity_dependent_cells
    ]:
        raise ValueError("PR-P/PR-K dependent-cell roster mismatch")


def _validate_semantic_binding(
    cell_ref: Any,
    adjudicated_cell: OledSupplementaryAdjudicatedCell,
    group: OledSupplementaryAdjudicatedGroup,
) -> None:
    source_cell = adjudicated_cell.source_cell
    expected = {
        "row_index": source_cell.row_index,
        "column_index": source_cell.column_index,
        "column_name": source_cell.column_name,
        "source_cell_digest": source_cell.source_cell_digest,
        "cell_disposition_digest": source_cell.cell_disposition_digest,
        "semantic_review_item_id": adjudicated_cell.decision_source_review_item_id,
        "semantic_review_item_digest": adjudicated_cell.decision_source_review_item_digest,
    }
    for field_name, value in expected.items():
        if getattr(cell_ref, field_name) != value:
            raise ValueError(f"PR-P/PR-I cell {field_name} mismatch")
    if (
        group.review_item_digest != cell_ref.semantic_review_item_digest
        or source_cell.source_cell_digest not in group.member_source_cell_digests
    ):
        raise ValueError("PR-I group does not contain the exact staged cell")


def _validate_transcribed_source_cell(
    *,
    source_cell: OledSupplementarySemanticReviewCell,
    staging_item: Any,
    transcription_item: OledSupplementarySourceTranscriptionReviewItem,
) -> None:
    table = transcription_item.matched_table
    if (
        transcription_item.source_id != staging_item.source_id
        or transcription_item.source_pdf_sha256 != staging_item.source_pdf_sha256
        or transcription_item.parsed_document_sha256 != staging_item.parsed_document_sha256
        or transcription_item.pdf_page_number_one_based
        != staging_item.pdf_page_number_one_based
        or source_cell.row_index >= len(table.rows)
        or source_cell.column_index >= len(table.headers)
        or source_cell.subject_column_index >= len(table.headers)
    ):
        raise ValueError("PR-J table provenance or source coordinate mismatch")
    property_key = table.headers[source_cell.column_index]
    subject_key = table.headers[source_cell.subject_column_index]
    row = table.rows[source_cell.row_index]
    if (
        property_key != source_cell.column_name
        or subject_key != source_cell.subject_column_name
        or row[property_key] != source_cell.reported_value_text
        or row[subject_key] != source_cell.reported_subject_text
    ):
        raise ValueError("PR-I reported literal differs from exact PR-J table")


def _build_candidate_item(
    *,
    staging_item: Any,
    identity_group: OledSupplementaryMaterialIdentityCandidateGroup,
    cell: OledSupplementarySemanticReviewCell,
    mapping: OledSupplementaryKnownMappingSummary,
    transcription_item: OledSupplementarySourceTranscriptionReviewItem,
) -> OledObservationMaterializationCandidateItem:
    condition = _condition_from_mapping(mapping)
    evidence = OledEvidenceSource(
        source_id=f"source-cell:{cell.source_cell_digest[7:]}",
        source_type=OledEvidenceType.SUPPLEMENTARY,
        layer=mapping.target_layer,
        locator=(
            f"{transcription_item.table_id} row {cell.row_index} "
            f"column {cell.column_index}"
        ),
        metadata={
            "scope_id": cell.scope_id,
            "source_id": staging_item.source_id,
            "source_pdf_sha256": staging_item.source_pdf_sha256,
            "parsed_document_sha256": staging_item.parsed_document_sha256,
            "table_content_digest": cell.table_content_digest,
            "source_cell_digest": cell.source_cell_digest,
            "cell_disposition_digest": cell.cell_disposition_digest,
            "semantic_review_item_id": staging_item.identity_dependent_cells[
                [
                    item.source_cell_digest
                    for item in staging_item.identity_dependent_cells
                ].index(cell.source_cell_digest)
            ].semantic_review_item_id,
            "source_transcription_review_item_id": transcription_item.review_item_id,
            "pdf_page_number_one_based": staging_item.pdf_page_number_one_based,
            "row_index": cell.row_index,
            "column_index": cell.column_index,
        },
    )
    observation = OledPropertyObservation(
        property_label=mapping.property_id,
        value=_numeric_value(cell.reported_value_text, cell.reported_decimal_places),
        unit=mapping.reported_unit,
        reported_value_text=cell.reported_value_text,
        reported_decimal_places=cell.reported_decimal_places,
        condition=condition,
        evidence_sources=[evidence],
        metadata={
            "property_id": mapping.property_id,
            "reported_property_label": mapping.property_label,
            "canonical_unit": mapping.canonical_unit,
            "selected_existing_material_id": staging_item.selected_existing_material_id,
            "reported_subject_text": staging_item.reported_subject_text,
        },
    )
    canonical = _canonicalize_observation(mapping.target_layer, observation)
    cell_ref = next(
        item
        for item in identity_group.identity_dependent_cells
        if item.source_cell_digest == cell.source_cell_digest
    )
    payload: dict[str, Any] = {
        "candidate_id": f"observation-candidate:{cell.source_cell_digest[7:]}",
        "staging_item_id": staging_item.staging_item_id,
        "staging_item_digest": staging_item.staging_item_digest,
        "identity_group_id": identity_group.identity_group_id,
        "identity_group_digest": identity_group.identity_group_digest,
        "selected_existing_material_id": staging_item.selected_existing_material_id,
        "selected_registry_entry": staging_item.selected_registry_entry,
        "reported_subject_text": staging_item.reported_subject_text,
        "scope_id": staging_item.scope_id,
        "source_id": staging_item.source_id,
        "source_pdf_sha256": staging_item.source_pdf_sha256,
        "parsed_document_sha256": staging_item.parsed_document_sha256,
        "table_id": staging_item.table_id,
        "table_content_digest": staging_item.table_content_digest,
        "pdf_page_number_one_based": staging_item.pdf_page_number_one_based,
        "row_index": cell.row_index,
        "column_index": cell.column_index,
        "column_name": cell.column_name,
        "source_cell_digest": cell.source_cell_digest,
        "cell_disposition_digest": cell.cell_disposition_digest,
        "semantic_review_item_id": cell_ref.semantic_review_item_id,
        "semantic_review_item_digest": cell_ref.semantic_review_item_digest,
        "source_transcription_review_item_id": transcription_item.review_item_id,
        "source_transcription_review_item_digest": transcription_item.review_item_digest,
        "semantic_source_cell": cell,
        "mapping_summary": mapping,
        "property_observation": observation,
        "canonical_observation": canonical,
        "comparison_context_status": canonical.comparison_context_status,
        "comparison_context_required_fields": sorted(
            canonical.comparison_context_required_fields
        ),
        "comparison_context_missing_fields": sorted(
            canonical.comparison_context_missing_fields
        ),
        "comparison_ready": canonical.is_comparison_ready,
        "candidate_digest": "sha256:" + "0" * 64,
    }
    provisional = OledObservationMaterializationCandidateItem.model_construct(**payload)
    payload["candidate_digest"] = _observation_candidate_item_digest(provisional)
    return OledObservationMaterializationCandidateItem.model_validate(payload)


def _condition_from_mapping(
    mapping: OledSupplementaryKnownMappingSummary,
) -> OledMeasurementCondition | None:
    if mapping.comparison_context is None:
        return None
    return OledMeasurementCondition.model_validate(
        mapping.comparison_context.model_dump(mode="json")
    )


def _canonicalize_observation(
    layer: OledCausalLayer,
    observation: OledPropertyObservation,
) -> OledLayeredCanonicalObservation:
    if layer == OledCausalLayer.MOLECULE:
        record = OledLayeredRecord(molecule=OledMolecularLayer(properties=[observation]))
    elif layer == OledCausalLayer.INTERACTION:
        record = OledLayeredRecord(
            interaction=OledInteractionLayer(properties=[observation])
        )
    elif layer == OledCausalLayer.DEVICE:
        record = OledLayeredRecord(device=OledDeviceLayer(properties=[observation]))
    elif layer == OledCausalLayer.MEASUREMENT:
        record = OledLayeredRecord(
            measurement=OledMeasurementLayer(measurements=[observation])
        )
    else:
        raise ValueError("unsupported OLED observation target layer")
    report = record.validate_schema()
    if not report.is_valid or len(report.observations) != 1:
        error_codes = ",".join(sorted(report.error_codes)) or "missing_observation"
        raise ValueError(f"materialized observation fails layered schema: {error_codes}")
    return report.observations[0]


def _numeric_value(reported: str, decimal_places: int | None) -> int | float:
    clean = reported.strip().replace("−", "-")
    try:
        value = Decimal(clean)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("known-property reported value is not a strict scalar") from exc
    if not value.is_finite():
        raise ValueError("known-property reported value must be finite")
    if decimal_places == 0 and "." not in clean and "e" not in clean.lower():
        return int(value)
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("known-property reported value exceeds finite numeric range")
    return numeric


def _comparison_context_counts(
    candidates: list[OledObservationMaterializationCandidateItem],
) -> dict[OledComparisonContextStatus, int]:
    return {
        status: sum(item.comparison_context_status == status for item in candidates)
        for status in OledComparisonContextStatus
    }


def _observation_candidate_item_digest(
    item: OledObservationMaterializationCandidateItem,
) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("candidate_digest", None)
    return _stable_hash(payload)


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "OLED_OBSERVATION_MATERIALIZATION_CANDIDATE_VERSION",
    "OledObservationMaterializationCandidateArtifact",
    "OledObservationMaterializationCandidateItem",
    "OledObservationMaterializationCandidateStatus",
    "build_oled_observation_materialization_candidate_artifact",
    "oled_observation_materialization_candidate_artifact_digest",
]
