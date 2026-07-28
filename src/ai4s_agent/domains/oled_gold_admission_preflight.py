from __future__ import annotations

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
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
)
from ai4s_agent.domains.oled_reviewed_evidence_facet_adjudication import (
    OledConfidenceSufficiencyDecision,
    OledReviewedEvidenceFacetAdjudicatedObservation,
    OledReviewedEvidenceFacetAdjudicationArtifact,
    OledScientificConsistencyDecision,
    oled_reviewed_evidence_facet_adjudication_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledComparisonContextStatus,
    _id_hash,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_GOLD_ADMISSION_PREFLIGHT_VERSION = "oled_gold_admission_preflight.v1"


class OledGoldAdmissionPreflightStatus(str, Enum):
    CANDIDATES_READY = "gold_admission_candidates_ready_for_publication_preflight"
    PARTIAL_WITH_BLOCKED_EVIDENCE = (
        "gold_admission_candidates_ready_with_blocked_evidence"
    )
    NO_ELIGIBLE_EVIDENCE = "no_gold_admission_eligible_evidence"


class OledGoldAdmissionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    candidate_id: str
    source_entry_id: str
    source_projection_id: str
    source_claim_id: str
    source_adjudicated_observation_digest: str
    selected_material_id: str
    selected_registry_entry: OledMaterialRegistryEntry
    registry_entry_payload_digest: str
    registry_entry_digest: str
    reported_subject_text: str
    property_id: str
    property_label: str
    target_layer: OledCausalLayer
    reported_value: float | int | str | None = None
    reported_value_text: str
    reported_decimal_places: Annotated[StrictInt, Field(ge=0)] | None = None
    reported_unit: str
    normalized_value: float | int | str | None = None
    normalized_unit: str | None = None
    comparison_context_status: OledComparisonContextStatus
    comparison_context: dict[str, Any] | None = None
    comparison_context_hash: str | None = None
    source_id: str
    source_pdf_sha256: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    table_id: str
    table_content_digest: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    column_index: Annotated[StrictInt, Field(ge=0)]
    column_name: str
    source_cell_digest: str
    scientific_consistency: Literal["consistent"] = "consistent"
    confidence_sufficiency: Literal["sufficient"] = "sufficient"
    facet_reviewed_by: str
    facet_reviewed_at: str
    facet_review_note: str
    evidence_refs: list[str]
    gold_admission_candidate: StrictBool = True
    eligible_for_gold_publication_preflight: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_record_created: StrictBool = False
    candidate_digest: str

    @field_validator(
        "candidate_id",
        "source_entry_id",
        "source_projection_id",
        "source_claim_id",
        "selected_material_id",
        "property_id",
        "table_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("facet_reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="facet_reviewed_at")

    @field_validator(
        "source_adjudicated_observation_digest",
        "registry_entry_payload_digest",
        "registry_entry_digest",
        "source_pdf_sha256",
        "table_content_digest",
        "source_cell_digest",
        "candidate_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("comparison_context_hash")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_sha256(value, field_name="comparison_context_hash")

    @field_validator("evidence_refs")
    @classmethod
    def validate_evidence_refs(cls, value: list[str]) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)) or not value:
            raise ValueError("Gold admission evidence refs must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_candidate(self) -> OledGoldAdmissionCandidate:
        if self.target_layer == OledCausalLayer.DEVICE:
            raise ValueError("device evidence cannot enter Gold admission")
        if self.selected_registry_entry.material_id != self.selected_material_id:
            raise ValueError("Gold admission Registry entry binding mismatch")
        if self.selected_registry_entry.entry_digest != self.registry_entry_digest:
            raise ValueError("Gold admission Registry entry digest mismatch")
        if _stable_hash(
            self.selected_registry_entry.model_dump(mode="json")
        ) != self.registry_entry_payload_digest:
            raise ValueError("Gold admission Registry payload digest mismatch")
        if self.comparison_context_status == OledComparisonContextStatus.INCOMPLETE:
            raise ValueError("incomplete context cannot enter Gold admission")
        if self.comparison_context_status == OledComparisonContextStatus.NOT_REQUIRED:
            if self.comparison_context is not None or self.comparison_context_hash is not None:
                raise ValueError("not-required Gold context must remain absent")
        elif self.comparison_context is None or self.comparison_context_hash is None:
            raise ValueError("complete Gold context must remain bound")
        fixed_true = (
            "gold_admission_candidate",
            "eligible_for_gold_publication_preflight",
            "categorical_confidence_only",
        )
        fixed_false = (
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "gold_record_created",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("Gold admission candidate crossed its boundary")
        if _gold_admission_candidate_digest(self) != self.candidate_digest:
            raise ValueError("Gold admission candidate digest mismatch")
        return self


class OledGoldAdmissionPreflightArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_GOLD_ADMISSION_PREFLIGHT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    facet_adjudication_sha256: str
    facet_adjudication_digest: str
    facet_adjudication: OledReviewedEvidenceFacetAdjudicationArtifact
    status: OledGoldAdmissionPreflightStatus
    source_review_group_count: Annotated[StrictInt, Field(ge=0)]
    source_reviewed_observation_count: Annotated[StrictInt, Field(ge=0)]
    eligible_candidate_count: Annotated[StrictInt, Field(ge=0)]
    blocked_observation_count: Annotated[StrictInt, Field(ge=0)]
    blocked_scientific_inconsistent_count: Annotated[StrictInt, Field(ge=0)]
    blocked_scientific_source_check_count: Annotated[StrictInt, Field(ge=0)]
    blocked_confidence_insufficient_count: Annotated[StrictInt, Field(ge=0)]
    blocked_confidence_source_check_count: Annotated[StrictInt, Field(ge=0)]
    device_only_count: Literal[0] = 0
    candidates: list[OledGoldAdmissionCandidate] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    preflight_artifact_digest: str
    exact_facet_adjudication_bytes_bound: StrictBool = True
    complete_adjudication_roster_replayed: StrictBool = True
    eligible_pair_replayed: StrictBool = True
    source_and_registry_provenance_preserved: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_invented: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_records_created: StrictBool = False
    gold_published: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    reviewed_evidence_mutated: StrictBool = False
    registry_written: StrictBool = False
    aliases_mutated: StrictBool = False
    standalone_input_bytes_revalidation_supported: StrictBool = False
    source_pdf_read: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_GOLD_ADMISSION_PREFLIGHT_VERSION:
            raise ValueError("unexpected Gold admission preflight version")
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
        "facet_adjudication_sha256",
        "facet_adjudication_digest",
        "preflight_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("candidates")
    @classmethod
    def validate_candidate_order(
        cls,
        value: list[OledGoldAdmissionCandidate],
    ) -> list[OledGoldAdmissionCandidate]:
        candidate_ids = [candidate.candidate_id for candidate in value]
        if candidate_ids != sorted(candidate_ids) or len(candidate_ids) != len(
            set(candidate_ids)
        ):
            raise ValueError("Gold admission candidates must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledGoldAdmissionPreflightArtifact:
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.facet_adjudication.generated_at
        ):
            raise ValueError("Gold admission preflight predates facet adjudication")
        if (
            self.run_id != self.facet_adjudication.run_id
            or self.paper_id != self.facet_adjudication.paper_id
            or self.facet_adjudication_digest
            != self.facet_adjudication.adjudication_artifact_digest
            or oled_reviewed_evidence_facet_adjudication_artifact_digest(
                self.facet_adjudication
            )
            != self.facet_adjudication_digest
        ):
            raise ValueError("Gold admission facet adjudication binding mismatch")
        expected_candidates = _derive_gold_admission_candidates(
            self.facet_adjudication
        )
        if [candidate.model_dump(mode="json") for candidate in self.candidates] != [
            candidate.model_dump(mode="json") for candidate in expected_candidates
        ]:
            raise ValueError("Gold admission candidate derivation mismatch")
        expected_counts = _preflight_counts(self.facet_adjudication)
        count_fields = {
            "source_review_group_count": "groups",
            "source_reviewed_observation_count": "total",
            "eligible_candidate_count": "eligible",
            "blocked_observation_count": "blocked",
            "blocked_scientific_inconsistent_count": "scientific_inconsistent",
            "blocked_scientific_source_check_count": "scientific_source_check",
            "blocked_confidence_insufficient_count": "confidence_insufficient",
            "blocked_confidence_source_check_count": "confidence_source_check",
        }
        for field_name, count_name in count_fields.items():
            if getattr(self, field_name) != expected_counts[count_name]:
                raise ValueError(f"Gold admission {field_name} mismatch")
        if self.status != _preflight_status(expected_counts) or self.device_only_count != 0:
            raise ValueError("Gold admission status or device boundary mismatch")
        fixed_true = (
            "exact_facet_adjudication_bytes_bound",
            "complete_adjudication_roster_replayed",
            "eligible_pair_replayed",
            "source_and_registry_provenance_preserved",
            "categorical_confidence_only",
        )
        fixed_false = (
            "numeric_confidence_score_invented",
            "legacy_numeric_confidence_record_constructed",
            "gold_records_created",
            "gold_published",
            "dataset_written",
            "training_eligible",
            "reviewed_evidence_mutated",
            "registry_written",
            "aliases_mutated",
            "standalone_input_bytes_revalidation_supported",
            "source_pdf_read",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("Gold admission preflight crossed its boundary")
        if oled_gold_admission_preflight_artifact_digest(self) != (
            self.preflight_artifact_digest
        ):
            raise ValueError("Gold admission preflight artifact digest mismatch")
        return self


def build_oled_gold_admission_preflight_artifact(
    *,
    facet_adjudication: OledReviewedEvidenceFacetAdjudicationArtifact,
    facet_adjudication_sha256: str,
    generated_at: str,
) -> OledGoldAdmissionPreflightArtifact:
    adjudication = OledReviewedEvidenceFacetAdjudicationArtifact.model_validate(
        facet_adjudication.model_dump(mode="json")
    )
    candidates = _derive_gold_admission_candidates(adjudication)
    counts = _preflight_counts(adjudication)
    payload: dict[str, Any] = {
        "artifact_version": OLED_GOLD_ADMISSION_PREFLIGHT_VERSION,
        "run_id": adjudication.run_id,
        "paper_id": adjudication.paper_id,
        "generated_at": generated_at,
        "facet_adjudication_sha256": _normalize_sha256(
            facet_adjudication_sha256,
            field_name="facet_adjudication_sha256",
        ),
        "facet_adjudication_digest": adjudication.adjudication_artifact_digest,
        "facet_adjudication": adjudication,
        "status": _preflight_status(counts),
        "source_review_group_count": counts["groups"],
        "source_reviewed_observation_count": counts["total"],
        "eligible_candidate_count": counts["eligible"],
        "blocked_observation_count": counts["blocked"],
        "blocked_scientific_inconsistent_count": counts[
            "scientific_inconsistent"
        ],
        "blocked_scientific_source_check_count": counts[
            "scientific_source_check"
        ],
        "blocked_confidence_insufficient_count": counts[
            "confidence_insufficient"
        ],
        "blocked_confidence_source_check_count": counts[
            "confidence_source_check"
        ],
        "device_only_count": 0,
        "candidates": candidates,
        "preflight_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledGoldAdmissionPreflightArtifact.model_construct(**payload)
    payload["preflight_artifact_digest"] = (
        oled_gold_admission_preflight_artifact_digest(provisional)
    )
    return OledGoldAdmissionPreflightArtifact.model_validate(payload)


def oled_gold_admission_preflight_artifact_digest(
    artifact: OledGoldAdmissionPreflightArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("preflight_artifact_digest", None)
    return _stable_hash(payload)


def _derive_gold_admission_candidates(
    adjudication: OledReviewedEvidenceFacetAdjudicationArtifact,
) -> list[OledGoldAdmissionCandidate]:
    candidates = [
        _build_gold_admission_candidate(item, adjudication)
        for item in adjudication.adjudicated_observations
        if item.eligible_for_gold_admission_preflight
    ]
    return sorted(candidates, key=lambda candidate: candidate.candidate_id)


def _build_gold_admission_candidate(
    item: OledReviewedEvidenceFacetAdjudicatedObservation,
    adjudication: OledReviewedEvidenceFacetAdjudicationArtifact,
) -> OledGoldAdmissionCandidate:
    if (
        item.decision_entry.scientific_consistency
        != OledScientificConsistencyDecision.CONSISTENT
        or item.decision_entry.confidence_sufficiency
        != OledConfidenceSufficiencyDecision.SUFFICIENT
        or item.retained_gold_blocker_codes
        or not item.eligible_for_gold_admission_preflight
    ):
        raise ValueError("blocked evidence cannot become a Gold admission candidate")
    observation = item.request_observation
    evidence_refs = sorted(
        {
            f"source:{observation.source_id}",
            f"pdf:{observation.source_pdf_sha256}",
            (
                f"table:{observation.table_id}:page:"
                f"{observation.pdf_page_number_one_based}:row:{observation.row_index}:"
                f"column:{observation.column_index}"
            ),
            f"cell:{observation.source_cell_digest}",
            f"ledger-entry:{observation.entry_id}",
        }
    )
    candidate_id = _id_hash(
        "oled-gold-admission-candidate",
        {
            "entry_id": observation.entry_id,
            "adjudicated_observation_digest": (
                item.adjudicated_observation_digest
            ),
            "facet_adjudication_digest": (
                adjudication.adjudication_artifact_digest
            ),
        },
    )
    payload: dict[str, Any] = {
        "candidate_id": candidate_id,
        "source_entry_id": observation.entry_id,
        "source_projection_id": observation.projection_id,
        "source_claim_id": observation.source_claim_id,
        "source_adjudicated_observation_digest": (
            item.adjudicated_observation_digest
        ),
        "selected_material_id": observation.selected_material_id,
        "selected_registry_entry": observation.selected_registry_entry,
        "registry_entry_payload_digest": observation.registry_entry_digest,
        "registry_entry_digest": observation.selected_registry_entry.entry_digest,
        "reported_subject_text": observation.reported_subject_text,
        "property_id": observation.property_id,
        "property_label": observation.property_label,
        "target_layer": observation.target_layer,
        "reported_value": observation.reported_value,
        "reported_value_text": observation.reported_value_text,
        "reported_decimal_places": observation.reported_decimal_places,
        "reported_unit": observation.reported_unit,
        "normalized_value": observation.normalized_value,
        "normalized_unit": observation.normalized_unit,
        "comparison_context_status": observation.comparison_context_status,
        "comparison_context": observation.comparison_context,
        "comparison_context_hash": observation.comparison_context_hash,
        "source_id": observation.source_id,
        "source_pdf_sha256": observation.source_pdf_sha256,
        "pdf_page_number_one_based": observation.pdf_page_number_one_based,
        "table_id": observation.table_id,
        "table_content_digest": observation.table_content_digest,
        "row_index": observation.row_index,
        "column_index": observation.column_index,
        "column_name": observation.column_name,
        "source_cell_digest": observation.source_cell_digest,
        "scientific_consistency": "consistent",
        "confidence_sufficiency": "sufficient",
        "facet_reviewed_by": adjudication.reviewed_by,
        "facet_reviewed_at": adjudication.reviewed_at,
        "facet_review_note": item.decision_entry.review_note,
        "evidence_refs": evidence_refs,
        "gold_admission_candidate": True,
        "eligible_for_gold_publication_preflight": True,
        "categorical_confidence_only": True,
        "numeric_confidence_score_assigned": False,
        "legacy_numeric_confidence_record_constructed": False,
        "gold_record_created": False,
        "candidate_digest": "sha256:" + "0" * 64,
    }
    provisional = OledGoldAdmissionCandidate.model_construct(**payload)
    payload["candidate_digest"] = _gold_admission_candidate_digest(provisional)
    return OledGoldAdmissionCandidate.model_validate(payload)


def _gold_admission_candidate_digest(
    candidate: OledGoldAdmissionCandidate,
) -> str:
    payload = candidate.model_dump(mode="json")
    payload.pop("candidate_digest", None)
    return _stable_hash(payload)


def _preflight_counts(
    adjudication: OledReviewedEvidenceFacetAdjudicationArtifact,
) -> dict[str, int]:
    return {
        "groups": adjudication.review_group_count,
        "total": adjudication.reviewed_observation_count,
        "eligible": adjudication.gold_admission_preflight_eligible_count,
        "blocked": adjudication.blocked_observation_count,
        "scientific_inconsistent": adjudication.scientific_inconsistent_count,
        "scientific_source_check": adjudication.scientific_source_check_count,
        "confidence_insufficient": adjudication.confidence_insufficient_count,
        "confidence_source_check": adjudication.confidence_source_check_count,
    }


def _preflight_status(counts: dict[str, int]) -> OledGoldAdmissionPreflightStatus:
    if counts["eligible"] == 0:
        return OledGoldAdmissionPreflightStatus.NO_ELIGIBLE_EVIDENCE
    if counts["blocked"]:
        return OledGoldAdmissionPreflightStatus.PARTIAL_WITH_BLOCKED_EVIDENCE
    return OledGoldAdmissionPreflightStatus.CANDIDATES_READY


__all__ = [
    "OLED_GOLD_ADMISSION_PREFLIGHT_VERSION",
    "OledGoldAdmissionCandidate",
    "OledGoldAdmissionPreflightArtifact",
    "OledGoldAdmissionPreflightStatus",
    "build_oled_gold_admission_preflight_artifact",
    "oled_gold_admission_preflight_artifact_digest",
]
