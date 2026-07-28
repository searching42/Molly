from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, field_validator, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
)
from ai4s_agent.domains.oled_reviewed_evidence_ledger_postwrite_verifier import (
    OledReviewedEvidenceLedgerPostwriteVerificationArtifact,
    OledReviewedEvidenceLedgerPostwriteVerificationStatus,
    oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    OledComparisonContextStatus,
    OledReviewedEvidenceLedgerEntry,
    OledReviewedEvidenceLedgerEntryStatus,
    OledReviewedEvidencePreflightItem,
    OledReviewedEvidenceSourceRowGroup,
    _id_hash,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_REVIEWED_EVIDENCE_FACET_REVIEW_REQUEST_VERSION = (
    "oled_reviewed_evidence_facet_review_request.v1"
)

_REQUIRED_BLOCKERS = [
    "missing_confidence_assessment",
    "scientific_consistency_not_reviewed",
]
_SCIENTIFIC_DISPOSITIONS = [
    "consistent",
    "inconsistent",
    "needs_source_check",
]
_CONFIDENCE_DISPOSITIONS = [
    "sufficient",
    "insufficient",
    "needs_source_check",
]
_SCIENTIFIC_CONSISTENCY_QUESTION = (
    "Are the material, property, layer, reported literal/unit, "
    "normalized value/unit, and comparison context mutually "
    "consistent with the exact source-row evidence?"
)
_CONFIDENCE_SUFFICIENCY_QUESTION = (
    "Is the exact reviewed evidence sufficiently supported for later "
    "Gold consideration, without interpreting this as a calibrated probability?"
)


class OledReviewedEvidenceFacetReviewRequestStatus(str, Enum):
    READY = "ready_for_bounded_human_facet_review"
    NO_ELIGIBLE_EVIDENCE = "no_eligible_reviewed_evidence"


class OledReviewedEvidenceFacetReviewContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_facets: list[str]
    scientific_consistency_question: str
    scientific_consistency_allowed_dispositions: list[str]
    confidence_sufficiency_question: str
    confidence_sufficiency_allowed_dispositions: list[str]
    confidence_is_calibrated_probability: StrictBool = False
    source_pdf_remains_authoritative: StrictBool = True
    numeric_confidence_score_requested: StrictBool = False
    one_decision_per_observation_required: StrictBool = True

    @model_validator(mode="after")
    def validate_contract(self) -> OledReviewedEvidenceFacetReviewContract:
        if self.requested_facets != [
            "confidence_sufficiency",
            "scientific_consistency",
        ]:
            raise ValueError("unexpected reviewed-evidence facet roster")
        if self.scientific_consistency_allowed_dispositions != (
            _SCIENTIFIC_DISPOSITIONS
        ) or self.confidence_sufficiency_allowed_dispositions != (
            _CONFIDENCE_DISPOSITIONS
        ):
            raise ValueError("unexpected reviewed-evidence facet dispositions")
        if (
            self.confidence_is_calibrated_probability
            or not self.source_pdf_remains_authoritative
            or self.numeric_confidence_score_requested
            or not self.one_decision_per_observation_required
        ):
            raise ValueError("reviewed-evidence facet contract boundary mismatch")
        return self


def _build_expected_review_contract() -> OledReviewedEvidenceFacetReviewContract:
    return OledReviewedEvidenceFacetReviewContract.model_validate(
        {
            "requested_facets": [
                "confidence_sufficiency",
                "scientific_consistency",
            ],
            "scientific_consistency_question": _SCIENTIFIC_CONSISTENCY_QUESTION,
            "scientific_consistency_allowed_dispositions": _SCIENTIFIC_DISPOSITIONS,
            "confidence_sufficiency_question": _CONFIDENCE_SUFFICIENCY_QUESTION,
            "confidence_sufficiency_allowed_dispositions": _CONFIDENCE_DISPOSITIONS,
            "confidence_is_calibrated_probability": False,
            "source_pdf_remains_authoritative": True,
            "numeric_confidence_score_requested": False,
            "one_decision_per_observation_required": True,
        }
    )


class OledReviewedEvidenceFacetReviewObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    entry_id: str
    projection_id: str
    source_claim_id: str
    source_candidate_id: str
    source_candidate_digest: str
    entry_digest: str
    selected_material_id: str
    selected_registry_entry: OledMaterialRegistryEntry
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
    gold_blocker_codes: list[str]
    observation_digest: str

    @field_validator(
        "entry_id",
        "projection_id",
        "source_claim_id",
        "source_candidate_id",
        "selected_material_id",
        "property_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("table_id")
    @classmethod
    def validate_table_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="table_id")

    @field_validator(
        "source_candidate_digest",
        "entry_digest",
        "registry_entry_digest",
        "source_pdf_sha256",
        "table_content_digest",
        "source_cell_digest",
        "observation_digest",
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

    @field_validator("gold_blocker_codes")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != _REQUIRED_BLOCKERS:
            raise ValueError("review request must contain only the two facet blockers")
        return value

    @model_validator(mode="after")
    def validate_observation(self) -> OledReviewedEvidenceFacetReviewObservation:
        if self.target_layer == OledCausalLayer.DEVICE:
            raise ValueError("device-only evidence cannot enter facet review")
        if self.comparison_context_status == OledComparisonContextStatus.INCOMPLETE:
            raise ValueError("incomplete comparison context cannot enter facet review")
        if self.comparison_context_status == OledComparisonContextStatus.NOT_REQUIRED:
            if self.comparison_context is not None or self.comparison_context_hash is not None:
                raise ValueError("not-required comparison context must remain absent")
        elif self.comparison_context is None or self.comparison_context_hash is None:
            raise ValueError("complete comparison context must remain explicitly bound")
        if _facet_observation_digest(self) != self.observation_digest:
            raise ValueError("reviewed-evidence facet observation digest mismatch")
        return self


class OledReviewedEvidenceFacetReviewGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_group_id: str
    source_row_group_id: str
    staging_item_id: str
    identity_group_id: str
    selected_material_id: str
    source_pdf_sha256: str
    table_id: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    observation_count: Annotated[StrictInt, Field(ge=1)]
    observations: list[OledReviewedEvidenceFacetReviewObservation] = Field(
        min_length=1
    )
    group_digest: str

    @field_validator(
        "review_group_id",
        "source_row_group_id",
        "staging_item_id",
        "identity_group_id",
        "selected_material_id",
        "table_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_pdf_sha256", "group_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("observations")
    @classmethod
    def validate_observation_order(
        cls,
        value: list[OledReviewedEvidenceFacetReviewObservation],
    ) -> list[OledReviewedEvidenceFacetReviewObservation]:
        order = [item.entry_id for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("facet review observations must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_group(self) -> OledReviewedEvidenceFacetReviewGroup:
        if self.observation_count != len(self.observations):
            raise ValueError("facet review group count mismatch")
        if any(
            observation.selected_material_id != self.selected_material_id
            or observation.source_pdf_sha256 != self.source_pdf_sha256
            or observation.table_id != self.table_id
            or observation.row_index != self.row_index
            for observation in self.observations
        ):
            raise ValueError("facet review source-row binding mismatch")
        if _facet_group_digest(self) != self.group_digest:
            raise ValueError("facet review group digest mismatch")
        return self


class OledReviewedEvidenceFacetReviewRequestArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_REVIEWED_EVIDENCE_FACET_REVIEW_REQUEST_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    postwrite_verification_sha256: str
    postwrite_verification_digest: str
    postwrite_verification: OledReviewedEvidenceLedgerPostwriteVerificationArtifact
    review_contract: OledReviewedEvidenceFacetReviewContract
    status: OledReviewedEvidenceFacetReviewRequestStatus
    eligible_observation_count: Annotated[StrictInt, Field(ge=0)]
    review_group_count: Annotated[StrictInt, Field(ge=0)]
    excluded_quarantined_count: Annotated[StrictInt, Field(ge=0)]
    excluded_incomplete_context_count: Annotated[StrictInt, Field(ge=0)]
    excluded_unplanned_blocker_count: Annotated[StrictInt, Field(ge=0)]
    device_only_count: Literal[0] = 0
    review_groups: list[OledReviewedEvidenceFacetReviewGroup] = Field(
        default_factory=list
    )
    request_artifact_digest: str
    exact_postwrite_bytes_bound: StrictBool = True
    active_only: StrictBool = True
    comparison_ready_only: StrictBool = True
    source_row_grouped: StrictBool = True
    numeric_confidence_score_invented: StrictBool = False
    human_decisions_recorded: StrictBool = False
    reviewed_evidence_mutated: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    registry_written: StrictBool = False
    aliases_mutated: StrictBool = False
    standalone_input_bytes_revalidation_supported: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_REVIEWED_EVIDENCE_FACET_REVIEW_REQUEST_VERSION:
            raise ValueError("unexpected facet review request version")
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
        "postwrite_verification_sha256",
        "postwrite_verification_digest",
        "request_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("review_groups")
    @classmethod
    def validate_group_order(
        cls,
        value: list[OledReviewedEvidenceFacetReviewGroup],
    ) -> list[OledReviewedEvidenceFacetReviewGroup]:
        order = [group.review_group_id for group in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("facet review groups must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledReviewedEvidenceFacetReviewRequestArtifact:
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.postwrite_verification.generated_at
        ):
            raise ValueError("facet review request predates PR-T")
        if self.run_id != self.postwrite_verification.run_id or self.paper_id != (
            self.postwrite_verification.paper_id
        ):
            raise ValueError("facet review request source identity mismatch")
        if self.postwrite_verification.status != (
            OledReviewedEvidenceLedgerPostwriteVerificationStatus.VERIFIED
        ) or self.postwrite_verification_digest != (
            self.postwrite_verification.verification_artifact_digest
        ) or oled_reviewed_evidence_ledger_postwrite_verification_artifact_digest(
            self.postwrite_verification
        ) != self.postwrite_verification_digest:
            raise ValueError("facet review request PR-T binding mismatch")
        if self.review_contract != _build_expected_review_contract():
            raise ValueError("facet review request contract mismatch")
        expected = _derive_facet_review_request(self.postwrite_verification)
        if [group.model_dump(mode="json") for group in self.review_groups] != [
            group.model_dump(mode="json") for group in expected["groups"]
        ]:
            raise ValueError("facet review group derivation mismatch")
        expected_counts = {
            "eligible_observation_count": sum(
                group.observation_count for group in expected["groups"]
            ),
            "review_group_count": len(expected["groups"]),
            "excluded_quarantined_count": expected["excluded_quarantined_count"],
            "excluded_incomplete_context_count": expected[
                "excluded_incomplete_context_count"
            ],
            "excluded_unplanned_blocker_count": expected[
                "excluded_unplanned_blocker_count"
            ],
        }
        for field_name, expected_value in expected_counts.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"facet review {field_name} mismatch")
        expected_status = (
            OledReviewedEvidenceFacetReviewRequestStatus.READY
            if expected["groups"]
            else OledReviewedEvidenceFacetReviewRequestStatus.NO_ELIGIBLE_EVIDENCE
        )
        if self.status != expected_status or self.device_only_count != 0:
            raise ValueError("facet review status or device boundary mismatch")
        fixed_true = (
            "exact_postwrite_bytes_bound",
            "active_only",
            "comparison_ready_only",
            "source_row_grouped",
        )
        fixed_false = (
            "numeric_confidence_score_invented",
            "human_decisions_recorded",
            "reviewed_evidence_mutated",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "registry_written",
            "aliases_mutated",
            "standalone_input_bytes_revalidation_supported",
            "network_accessed",
            "external_service_called",
            "llm_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("facet review request crossed its boundary")
        if oled_reviewed_evidence_facet_review_request_artifact_digest(self) != (
            self.request_artifact_digest
        ):
            raise ValueError("facet review request artifact digest mismatch")
        return self


def build_oled_reviewed_evidence_facet_review_request_artifact(
    *,
    postwrite_verification: OledReviewedEvidenceLedgerPostwriteVerificationArtifact,
    postwrite_verification_sha256: str,
    generated_at: str,
) -> OledReviewedEvidenceFacetReviewRequestArtifact:
    verification = OledReviewedEvidenceLedgerPostwriteVerificationArtifact.model_validate(
        postwrite_verification.model_dump(mode="json")
    )
    derived = _derive_facet_review_request(verification)
    contract = _build_expected_review_contract()
    payload: dict[str, Any] = {
        "artifact_version": OLED_REVIEWED_EVIDENCE_FACET_REVIEW_REQUEST_VERSION,
        "run_id": verification.run_id,
        "paper_id": verification.paper_id,
        "generated_at": generated_at,
        "postwrite_verification_sha256": _normalize_sha256(
            postwrite_verification_sha256,
            field_name="postwrite_verification_sha256",
        ),
        "postwrite_verification_digest": verification.verification_artifact_digest,
        "postwrite_verification": verification,
        "review_contract": contract,
        "status": (
            OledReviewedEvidenceFacetReviewRequestStatus.READY
            if derived["groups"]
            else OledReviewedEvidenceFacetReviewRequestStatus.NO_ELIGIBLE_EVIDENCE
        ),
        "eligible_observation_count": sum(
            group.observation_count for group in derived["groups"]
        ),
        "review_group_count": len(derived["groups"]),
        "excluded_quarantined_count": derived["excluded_quarantined_count"],
        "excluded_incomplete_context_count": derived[
            "excluded_incomplete_context_count"
        ],
        "excluded_unplanned_blocker_count": derived[
            "excluded_unplanned_blocker_count"
        ],
        "device_only_count": 0,
        "review_groups": derived["groups"],
        "request_artifact_digest": "sha256:" + "0" * 64,
        "exact_postwrite_bytes_bound": True,
        "active_only": True,
        "comparison_ready_only": True,
        "source_row_grouped": True,
        "numeric_confidence_score_invented": False,
        "human_decisions_recorded": False,
        "reviewed_evidence_mutated": False,
        "gold_records_created": False,
        "dataset_written": False,
        "training_eligible": False,
        "registry_written": False,
        "aliases_mutated": False,
        "standalone_input_bytes_revalidation_supported": False,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
    }
    provisional = OledReviewedEvidenceFacetReviewRequestArtifact.model_construct(
        **payload
    )
    payload["request_artifact_digest"] = (
        oled_reviewed_evidence_facet_review_request_artifact_digest(provisional)
    )
    return OledReviewedEvidenceFacetReviewRequestArtifact.model_validate(payload)


def _derive_facet_review_request(
    verification: OledReviewedEvidenceLedgerPostwriteVerificationArtifact,
) -> dict[str, Any]:
    receipt = verification.write_artifact
    preflight = receipt.preflight_artifact
    entries_by_projection = {
        entry.projection_id: entry
        for entry in verification.published_snapshot.entries
    }
    row_groups = {
        group.source_row_group_id: group for group in preflight.source_row_groups
    }
    grouped: dict[str, list[OledReviewedEvidenceFacetReviewObservation]] = {}
    excluded_quarantined = 0
    excluded_incomplete = 0
    excluded_blockers = 0
    for item in preflight.preflight_items:
        entry = entries_by_projection.get(item.projection_id)
        if entry is None:
            continue
        if entry.status != OledReviewedEvidenceLedgerEntryStatus.ACTIVE:
            if entry.status == OledReviewedEvidenceLedgerEntryStatus.QUARANTINED:
                excluded_quarantined += 1
            continue
        if not item.comparison_ready or (
            item.source_candidate.comparison_context_status
            == OledComparisonContextStatus.INCOMPLETE
        ):
            excluded_incomplete += 1
            continue
        if item.gold_blocker_codes != _REQUIRED_BLOCKERS:
            excluded_blockers += 1
            continue
        observation = _build_facet_observation(entry, item)
        grouped.setdefault(item.source_row_group_id, []).append(observation)
    groups = [
        _build_facet_group(
            source_group=row_groups[group_id],
            observations=observations,
            verification_digest=verification.verification_artifact_digest,
        )
        for group_id, observations in sorted(grouped.items())
    ]
    return {
        "groups": sorted(groups, key=lambda group: group.review_group_id),
        "excluded_quarantined_count": excluded_quarantined,
        "excluded_incomplete_context_count": excluded_incomplete,
        "excluded_unplanned_blocker_count": excluded_blockers,
    }


def _build_facet_observation(
    entry: OledReviewedEvidenceLedgerEntry,
    item: OledReviewedEvidencePreflightItem,
) -> OledReviewedEvidenceFacetReviewObservation:
    candidate = item.source_candidate
    payload: dict[str, Any] = {
        "entry_id": entry.entry_id,
        "projection_id": entry.projection_id,
        "source_claim_id": entry.source_claim_id,
        "source_candidate_id": entry.source_candidate_id,
        "source_candidate_digest": entry.source_candidate_digest,
        "entry_digest": entry.entry_digest,
        "selected_material_id": entry.selected_material_id,
        "selected_registry_entry": candidate.selected_registry_entry,
        "registry_entry_digest": entry.registry_entry_digest,
        "reported_subject_text": candidate.reported_subject_text,
        "property_id": entry.property_id,
        "property_label": candidate.mapping_summary.property_label,
        "target_layer": entry.target_layer,
        "reported_value": entry.reported_value,
        "reported_value_text": entry.reported_value_text,
        "reported_decimal_places": entry.reported_decimal_places,
        "reported_unit": entry.reported_unit,
        "normalized_value": entry.normalized_value,
        "normalized_unit": entry.normalized_unit,
        "comparison_context_status": entry.comparison_context_status,
        "comparison_context": (
            candidate.mapping_summary.comparison_context.model_dump(mode="json")
            if candidate.mapping_summary.comparison_context is not None
            else None
        ),
        "comparison_context_hash": entry.comparison_context_hash,
        "source_id": candidate.source_id,
        "source_pdf_sha256": entry.source_pdf_sha256,
        "pdf_page_number_one_based": candidate.pdf_page_number_one_based,
        "table_id": entry.table_id,
        "table_content_digest": candidate.table_content_digest,
        "row_index": entry.row_index,
        "column_index": entry.column_index,
        "column_name": candidate.column_name,
        "source_cell_digest": entry.source_cell_digest,
        "gold_blocker_codes": item.gold_blocker_codes,
        "observation_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidenceFacetReviewObservation.model_construct(
        **payload
    )
    payload["observation_digest"] = _facet_observation_digest(provisional)
    return OledReviewedEvidenceFacetReviewObservation.model_validate(payload)


def _build_facet_group(
    *,
    source_group: OledReviewedEvidenceSourceRowGroup,
    observations: list[OledReviewedEvidenceFacetReviewObservation],
    verification_digest: str,
) -> OledReviewedEvidenceFacetReviewGroup:
    ordered = sorted(observations, key=lambda item: item.entry_id)
    group_id = _id_hash(
        "reviewed-evidence-facet-review-group",
        {
            "source_row_group_id": source_group.source_row_group_id,
            "entry_ids": [item.entry_id for item in ordered],
            "postwrite_verification_digest": verification_digest,
        },
    )
    payload: dict[str, Any] = {
        "review_group_id": group_id,
        "source_row_group_id": source_group.source_row_group_id,
        "staging_item_id": source_group.staging_item_id,
        "identity_group_id": source_group.identity_group_id,
        "selected_material_id": source_group.selected_material_id,
        "source_pdf_sha256": source_group.source_pdf_sha256,
        "table_id": source_group.table_id,
        "row_index": source_group.row_index,
        "observation_count": len(ordered),
        "observations": ordered,
        "group_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidenceFacetReviewGroup.model_construct(**payload)
    payload["group_digest"] = _facet_group_digest(provisional)
    return OledReviewedEvidenceFacetReviewGroup.model_validate(payload)


def _facet_observation_digest(
    observation: OledReviewedEvidenceFacetReviewObservation,
) -> str:
    payload = observation.model_dump(mode="json")
    payload.pop("observation_digest", None)
    return _stable_hash(payload)


def _facet_group_digest(group: OledReviewedEvidenceFacetReviewGroup) -> str:
    payload = group.model_dump(mode="json")
    payload.pop("group_digest", None)
    return _stable_hash(payload)


def oled_reviewed_evidence_facet_review_request_artifact_digest(
    artifact: OledReviewedEvidenceFacetReviewRequestArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("request_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "OLED_REVIEWED_EVIDENCE_FACET_REVIEW_REQUEST_VERSION",
    "OledReviewedEvidenceFacetReviewRequestArtifact",
    "OledReviewedEvidenceFacetReviewRequestStatus",
    "build_oled_reviewed_evidence_facet_review_request_artifact",
    "oled_reviewed_evidence_facet_review_request_artifact_digest",
]
