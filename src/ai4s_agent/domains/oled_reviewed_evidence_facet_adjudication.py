from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_reviewed_evidence_facet_review_request import (
    OledReviewedEvidenceFacetReviewObservation,
    OledReviewedEvidenceFacetReviewRequestArtifact,
    OledReviewedEvidenceFacetReviewRequestStatus,
    oled_reviewed_evidence_facet_review_request_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _validate_reviewer_text,
)


OLED_REVIEWED_EVIDENCE_FACET_DECISION_MANIFEST_VERSION = (
    "oled_reviewed_evidence_facet_decision_manifest.v1"
)
OLED_REVIEWED_EVIDENCE_FACET_ADJUDICATION_VERSION = (
    "oled_reviewed_evidence_facet_adjudication.v1"
)


class OledScientificConsistencyDecision(str, Enum):
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    NEEDS_SOURCE_CHECK = "needs_source_check"


class OledConfidenceSufficiencyDecision(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    NEEDS_SOURCE_CHECK = "needs_source_check"


class OledReviewedEvidenceFacetAdjudicationStatus(str, Enum):
    READY_FOR_GOLD_ADMISSION_PREFLIGHT = (
        "facet_review_complete_ready_for_gold_admission_preflight"
    )
    REVIEW_COMPLETE_WITH_BLOCKED_EVIDENCE = (
        "facet_review_complete_with_blocked_evidence"
    )
    NO_ELIGIBLE_EVIDENCE = "no_eligible_reviewed_evidence"


class OledReviewedEvidenceFacetDecisionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_group_id: str
    group_digest: str
    entry_id: str
    observation_digest: str
    scientific_consistency: OledScientificConsistencyDecision
    confidence_sufficiency: OledConfidenceSufficiencyDecision
    review_note: str

    @field_validator("review_group_id", "entry_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("group_digest", "observation_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="review_note",
            required=True,
            max_length=4_000,
        )


class OledReviewedEvidenceFacetDecisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = OLED_REVIEWED_EVIDENCE_FACET_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    request_artifact_sha256: str
    request_artifact_digest: str
    postwrite_verification_sha256: str
    postwrite_verification_digest: str
    reviewed_by: str
    reviewed_at: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledReviewedEvidenceFacetDecisionEntry] = Field(
        default_factory=list,
        max_length=1_000_000,
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != OLED_REVIEWED_EVIDENCE_FACET_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected reviewed-evidence facet manifest version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "request_artifact_digest",
        "postwrite_verification_sha256",
        "postwrite_verification_digest",
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

    @field_validator("decisions")
    @classmethod
    def validate_decision_order(
        cls,
        value: list[OledReviewedEvidenceFacetDecisionEntry],
    ) -> list[OledReviewedEvidenceFacetDecisionEntry]:
        entry_ids = [decision.entry_id for decision in value]
        if entry_ids != sorted(entry_ids) or len(entry_ids) != len(set(entry_ids)):
            raise ValueError("facet decisions must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> OledReviewedEvidenceFacetDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("reviewed-evidence facet adjudication must be confirmed")
        return self


class OledReviewedEvidenceFacetAdjudicatedObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    review_group_id: str
    group_digest: str
    request_observation: OledReviewedEvidenceFacetReviewObservation
    decision_entry: OledReviewedEvidenceFacetDecisionEntry
    retained_gold_blocker_codes: list[str]
    scientific_consistency_reviewed: StrictBool = True
    confidence_sufficiency_reviewed: StrictBool = True
    scientific_consistency_accepted: StrictBool
    confidence_sufficiency_accepted: StrictBool
    source_check_required: StrictBool
    eligible_for_gold_admission_preflight: StrictBool
    gold_record_created: StrictBool = False
    adjudicated_observation_digest: str

    @field_validator("review_group_id")
    @classmethod
    def validate_group_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="review_group_id")

    @field_validator("group_digest", "adjudicated_observation_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("retained_gold_blocker_codes")
    @classmethod
    def validate_blockers(cls, value: list[str]) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("retained Gold blockers must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_observation(
        self,
    ) -> OledReviewedEvidenceFacetAdjudicatedObservation:
        if (
            self.decision_entry.review_group_id != self.review_group_id
            or self.decision_entry.group_digest != self.group_digest
            or self.decision_entry.entry_id != self.request_observation.entry_id
            or self.decision_entry.observation_digest
            != self.request_observation.observation_digest
        ):
            raise ValueError("facet adjudicated observation binding mismatch")
        expected = _decision_outcome(self.decision_entry)
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"facet adjudicated {field_name} mismatch")
        if (
            not self.scientific_consistency_reviewed
            or not self.confidence_sufficiency_reviewed
            or self.gold_record_created
        ):
            raise ValueError("facet adjudication crossed its review boundary")
        if _adjudicated_observation_digest(self) != (
            self.adjudicated_observation_digest
        ):
            raise ValueError("facet adjudicated observation digest mismatch")
        return self


class OledReviewedEvidenceFacetAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_REVIEWED_EVIDENCE_FACET_ADJUDICATION_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_artifact_digest: str
    request_artifact: OledReviewedEvidenceFacetReviewRequestArtifact
    postwrite_verification_sha256: str
    postwrite_verification_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    reviewed_by: str
    reviewed_at: str
    status: OledReviewedEvidenceFacetAdjudicationStatus
    review_group_count: Annotated[StrictInt, Field(ge=0)]
    reviewed_observation_count: Annotated[StrictInt, Field(ge=0)]
    scientific_consistent_count: Annotated[StrictInt, Field(ge=0)]
    scientific_inconsistent_count: Annotated[StrictInt, Field(ge=0)]
    scientific_source_check_count: Annotated[StrictInt, Field(ge=0)]
    confidence_sufficient_count: Annotated[StrictInt, Field(ge=0)]
    confidence_insufficient_count: Annotated[StrictInt, Field(ge=0)]
    confidence_source_check_count: Annotated[StrictInt, Field(ge=0)]
    gold_admission_preflight_eligible_count: Annotated[StrictInt, Field(ge=0)]
    blocked_observation_count: Annotated[StrictInt, Field(ge=0)]
    device_only_count: Annotated[StrictInt, Field(ge=0)] = 0
    adjudicated_observations: list[
        OledReviewedEvidenceFacetAdjudicatedObservation
    ] = Field(default_factory=list, max_length=1_000_000)
    adjudication_artifact_digest: str
    exact_request_bytes_bound: StrictBool = True
    exact_decision_manifest_bytes_bound: StrictBool = True
    complete_group_roster_replayed: StrictBool = True
    complete_observation_roster_replayed: StrictBool = True
    human_facet_review_completed: StrictBool = True
    numeric_confidence_score_invented: StrictBool = False
    reviewed_evidence_mutated: StrictBool = False
    direct_gold_admission_eligible: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
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
        if value != OLED_REVIEWED_EVIDENCE_FACET_ADJUDICATION_VERSION:
            raise ValueError("unexpected reviewed-evidence facet adjudication version")
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
        "request_artifact_digest",
        "postwrite_verification_sha256",
        "postwrite_verification_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "adjudication_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("adjudicated_observations")
    @classmethod
    def validate_observation_order(
        cls,
        value: list[OledReviewedEvidenceFacetAdjudicatedObservation],
    ) -> list[OledReviewedEvidenceFacetAdjudicatedObservation]:
        entry_ids = [
            observation.request_observation.entry_id for observation in value
        ]
        if entry_ids != sorted(entry_ids) or len(entry_ids) != len(set(entry_ids)):
            raise ValueError("facet adjudicated observations must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(
        self,
    ) -> OledReviewedEvidenceFacetAdjudicationArtifact:
        if _parse_timestamp(self.reviewed_at) < _parse_timestamp(
            self.request_artifact.generated_at
        ) or _parse_timestamp(self.generated_at) < _parse_timestamp(self.reviewed_at):
            raise ValueError("facet adjudication timestamp order mismatch")
        if (
            self.run_id != self.request_artifact.run_id
            or self.paper_id != self.request_artifact.paper_id
            or self.request_artifact_digest
            != self.request_artifact.request_artifact_digest
            or oled_reviewed_evidence_facet_review_request_artifact_digest(
                self.request_artifact
            )
            != self.request_artifact_digest
            or self.postwrite_verification_sha256
            != self.request_artifact.postwrite_verification_sha256
            or self.postwrite_verification_digest
            != self.request_artifact.postwrite_verification_digest
        ):
            raise ValueError("facet adjudication request binding mismatch")
        expected_roster = _request_observation_roster(self.request_artifact)
        actual_roster = {
            observation.request_observation.entry_id: (
                observation.review_group_id,
                observation.group_digest,
                observation.request_observation.observation_digest,
            )
            for observation in self.adjudicated_observations
        }
        expected_roster_projection = {
            entry_id: (group_id, group_digest, observation.observation_digest)
            for entry_id, (group_id, group_digest, observation) in (
                expected_roster.items()
            )
        }
        if actual_roster != expected_roster_projection:
            raise ValueError("facet adjudication observation roster mismatch")
        counts = _adjudication_counts(self.adjudicated_observations)
        expected_counts = {
            "review_group_count": self.request_artifact.review_group_count,
            "reviewed_observation_count": "total",
            "scientific_consistent_count": "scientific_consistent",
            "scientific_inconsistent_count": "scientific_inconsistent",
            "scientific_source_check_count": "scientific_source_check",
            "confidence_sufficient_count": "confidence_sufficient",
            "confidence_insufficient_count": "confidence_insufficient",
            "confidence_source_check_count": "confidence_source_check",
            "gold_admission_preflight_eligible_count": "eligible",
            "blocked_observation_count": "blocked",
        }
        for field_name, count_name in expected_counts.items():
            expected_value = (
                count_name
                if isinstance(count_name, int)
                else counts[count_name]
            )
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"facet adjudication {field_name} mismatch")
        if (
            self.reviewed_observation_count
            != self.request_artifact.eligible_observation_count
            or self.device_only_count != 0
            or self.status != _adjudication_status(counts)
        ):
            raise ValueError("facet adjudication coverage or status mismatch")
        reconstructed_manifest = OledReviewedEvidenceFacetDecisionManifest(
            run_id=self.run_id,
            paper_id=self.paper_id,
            request_artifact_sha256=self.request_artifact_sha256,
            request_artifact_digest=self.request_artifact_digest,
            postwrite_verification_sha256=self.postwrite_verification_sha256,
            postwrite_verification_digest=self.postwrite_verification_digest,
            reviewed_by=self.reviewed_by,
            reviewed_at=self.reviewed_at,
            adjudication_confirmed=True,
            decisions=[
                observation.decision_entry
                for observation in self.adjudicated_observations
            ],
        )
        if oled_reviewed_evidence_facet_decision_manifest_digest(
            reconstructed_manifest
        ) != self.decision_manifest_digest:
            raise ValueError("facet decision manifest digest mismatch")
        fixed_true = (
            "exact_request_bytes_bound",
            "exact_decision_manifest_bytes_bound",
            "complete_group_roster_replayed",
            "complete_observation_roster_replayed",
            "human_facet_review_completed",
        )
        fixed_false = (
            "numeric_confidence_score_invented",
            "reviewed_evidence_mutated",
            "direct_gold_admission_eligible",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
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
            raise ValueError("facet adjudication crossed its boundary")
        if oled_reviewed_evidence_facet_adjudication_artifact_digest(self) != (
            self.adjudication_artifact_digest
        ):
            raise ValueError("facet adjudication artifact digest mismatch")
        return self


def validate_oled_reviewed_evidence_facet_decisions(
    *,
    request: OledReviewedEvidenceFacetReviewRequestArtifact,
    request_artifact_sha256: str,
    decision_manifest: OledReviewedEvidenceFacetDecisionManifest,
) -> None:
    review_request = OledReviewedEvidenceFacetReviewRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    manifest = OledReviewedEvidenceFacetDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    expected_bindings = {
        "run_id": review_request.run_id,
        "paper_id": review_request.paper_id,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_artifact_digest": review_request.request_artifact_digest,
        "postwrite_verification_sha256": (
            review_request.postwrite_verification_sha256
        ),
        "postwrite_verification_digest": (
            review_request.postwrite_verification_digest
        ),
    }
    for field_name, expected in expected_bindings.items():
        if getattr(manifest, field_name) != expected:
            raise ValueError(f"facet decision manifest {field_name} mismatch")
    if _parse_timestamp(manifest.reviewed_at) < _parse_timestamp(
        review_request.generated_at
    ):
        raise ValueError("facet decision manifest predates its request")
    roster = _request_observation_roster(review_request)
    decisions = {decision.entry_id: decision for decision in manifest.decisions}
    if decisions.keys() != roster.keys():
        raise ValueError("facet decision coverage does not match the request")
    for entry_id, (group_id, group_digest, observation) in roster.items():
        decision = decisions[entry_id]
        if (
            decision.review_group_id != group_id
            or decision.group_digest != group_digest
            or decision.observation_digest != observation.observation_digest
        ):
            raise ValueError("facet decision observation binding mismatch")


def build_oled_reviewed_evidence_facet_adjudication_artifact(
    *,
    request: OledReviewedEvidenceFacetReviewRequestArtifact,
    request_artifact_sha256: str,
    decision_manifest: OledReviewedEvidenceFacetDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledReviewedEvidenceFacetAdjudicationArtifact:
    review_request = OledReviewedEvidenceFacetReviewRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    manifest = OledReviewedEvidenceFacetDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    validate_oled_reviewed_evidence_facet_decisions(
        request=review_request,
        request_artifact_sha256=request_artifact_sha256,
        decision_manifest=manifest,
    )
    roster = _request_observation_roster(review_request)
    decisions = {decision.entry_id: decision for decision in manifest.decisions}
    adjudicated = [
        _build_adjudicated_observation(
            group_id=roster[entry_id][0],
            group_digest=roster[entry_id][1],
            observation=roster[entry_id][2],
            decision=decisions[entry_id],
        )
        for entry_id in sorted(roster)
    ]
    counts = _adjudication_counts(adjudicated)
    payload: dict[str, Any] = {
        "artifact_version": OLED_REVIEWED_EVIDENCE_FACET_ADJUDICATION_VERSION,
        "run_id": review_request.run_id,
        "paper_id": review_request.paper_id,
        "generated_at": generated_at,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_artifact_digest": review_request.request_artifact_digest,
        "request_artifact": review_request,
        "postwrite_verification_sha256": (
            review_request.postwrite_verification_sha256
        ),
        "postwrite_verification_digest": (
            review_request.postwrite_verification_digest
        ),
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": (
            oled_reviewed_evidence_facet_decision_manifest_digest(manifest)
        ),
        "reviewed_by": manifest.reviewed_by,
        "reviewed_at": manifest.reviewed_at,
        "status": _adjudication_status(counts),
        "review_group_count": review_request.review_group_count,
        "reviewed_observation_count": counts["total"],
        "scientific_consistent_count": counts["scientific_consistent"],
        "scientific_inconsistent_count": counts["scientific_inconsistent"],
        "scientific_source_check_count": counts["scientific_source_check"],
        "confidence_sufficient_count": counts["confidence_sufficient"],
        "confidence_insufficient_count": counts["confidence_insufficient"],
        "confidence_source_check_count": counts["confidence_source_check"],
        "gold_admission_preflight_eligible_count": counts["eligible"],
        "blocked_observation_count": counts["blocked"],
        "device_only_count": 0,
        "adjudicated_observations": adjudicated,
        "adjudication_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidenceFacetAdjudicationArtifact.model_construct(
        **payload
    )
    payload["adjudication_artifact_digest"] = (
        oled_reviewed_evidence_facet_adjudication_artifact_digest(provisional)
    )
    return OledReviewedEvidenceFacetAdjudicationArtifact.model_validate(payload)


def oled_reviewed_evidence_facet_decision_manifest_digest(
    manifest: OledReviewedEvidenceFacetDecisionManifest,
) -> str:
    payload = manifest.model_dump(mode="json")
    payload["decisions"] = sorted(
        payload["decisions"],
        key=lambda decision: decision["entry_id"],
    )
    return _stable_hash(payload)


def oled_reviewed_evidence_facet_adjudication_artifact_digest(
    artifact: OledReviewedEvidenceFacetAdjudicationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("adjudication_artifact_digest", None)
    return _stable_hash(payload)


def _request_observation_roster(
    request: OledReviewedEvidenceFacetReviewRequestArtifact,
) -> dict[
    str,
    tuple[str, str, OledReviewedEvidenceFacetReviewObservation],
]:
    roster: dict[
        str,
        tuple[str, str, OledReviewedEvidenceFacetReviewObservation],
    ] = {}
    for group in request.review_groups:
        for observation in group.observations:
            if observation.entry_id in roster:
                raise ValueError("facet request repeats an observation entry")
            roster[observation.entry_id] = (
                group.review_group_id,
                group.group_digest,
                observation,
            )
    return roster


def _decision_outcome(
    decision: OledReviewedEvidenceFacetDecisionEntry,
) -> dict[str, Any]:
    scientific_accepted = (
        decision.scientific_consistency
        == OledScientificConsistencyDecision.CONSISTENT
    )
    confidence_accepted = (
        decision.confidence_sufficiency
        == OledConfidenceSufficiencyDecision.SUFFICIENT
    )
    source_check = (
        decision.scientific_consistency
        == OledScientificConsistencyDecision.NEEDS_SOURCE_CHECK
        or decision.confidence_sufficiency
        == OledConfidenceSufficiencyDecision.NEEDS_SOURCE_CHECK
    )
    blockers: list[str] = []
    if (
        decision.scientific_consistency
        == OledScientificConsistencyDecision.INCONSISTENT
    ):
        blockers.append("scientific_consistency_inconsistent")
    elif (
        decision.scientific_consistency
        == OledScientificConsistencyDecision.NEEDS_SOURCE_CHECK
    ):
        blockers.append("scientific_consistency_source_check_required")
    if (
        decision.confidence_sufficiency
        == OledConfidenceSufficiencyDecision.INSUFFICIENT
    ):
        blockers.append("confidence_evidence_insufficient")
    elif (
        decision.confidence_sufficiency
        == OledConfidenceSufficiencyDecision.NEEDS_SOURCE_CHECK
    ):
        blockers.append("confidence_source_check_required")
    return {
        "retained_gold_blocker_codes": sorted(blockers),
        "scientific_consistency_accepted": scientific_accepted,
        "confidence_sufficiency_accepted": confidence_accepted,
        "source_check_required": source_check,
        "eligible_for_gold_admission_preflight": (
            scientific_accepted and confidence_accepted
        ),
    }


def _build_adjudicated_observation(
    *,
    group_id: str,
    group_digest: str,
    observation: OledReviewedEvidenceFacetReviewObservation,
    decision: OledReviewedEvidenceFacetDecisionEntry,
) -> OledReviewedEvidenceFacetAdjudicatedObservation:
    payload: dict[str, Any] = {
        "review_group_id": group_id,
        "group_digest": group_digest,
        "request_observation": observation,
        "decision_entry": decision,
        **_decision_outcome(decision),
        "scientific_consistency_reviewed": True,
        "confidence_sufficiency_reviewed": True,
        "gold_record_created": False,
        "adjudicated_observation_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidenceFacetAdjudicatedObservation.model_construct(
        **payload
    )
    payload["adjudicated_observation_digest"] = (
        _adjudicated_observation_digest(provisional)
    )
    return OledReviewedEvidenceFacetAdjudicatedObservation.model_validate(payload)


def _adjudicated_observation_digest(
    observation: OledReviewedEvidenceFacetAdjudicatedObservation,
) -> str:
    payload = observation.model_dump(mode="json")
    payload.pop("adjudicated_observation_digest", None)
    return _stable_hash(payload)


def _adjudication_counts(
    observations: list[OledReviewedEvidenceFacetAdjudicatedObservation],
) -> dict[str, int]:
    return {
        "total": len(observations),
        "scientific_consistent": sum(
            item.decision_entry.scientific_consistency
            == OledScientificConsistencyDecision.CONSISTENT
            for item in observations
        ),
        "scientific_inconsistent": sum(
            item.decision_entry.scientific_consistency
            == OledScientificConsistencyDecision.INCONSISTENT
            for item in observations
        ),
        "scientific_source_check": sum(
            item.decision_entry.scientific_consistency
            == OledScientificConsistencyDecision.NEEDS_SOURCE_CHECK
            for item in observations
        ),
        "confidence_sufficient": sum(
            item.decision_entry.confidence_sufficiency
            == OledConfidenceSufficiencyDecision.SUFFICIENT
            for item in observations
        ),
        "confidence_insufficient": sum(
            item.decision_entry.confidence_sufficiency
            == OledConfidenceSufficiencyDecision.INSUFFICIENT
            for item in observations
        ),
        "confidence_source_check": sum(
            item.decision_entry.confidence_sufficiency
            == OledConfidenceSufficiencyDecision.NEEDS_SOURCE_CHECK
            for item in observations
        ),
        "eligible": sum(
            item.eligible_for_gold_admission_preflight for item in observations
        ),
        "blocked": sum(
            not item.eligible_for_gold_admission_preflight for item in observations
        ),
    }


def _adjudication_status(
    counts: dict[str, int],
) -> OledReviewedEvidenceFacetAdjudicationStatus:
    if counts["total"] == 0:
        return OledReviewedEvidenceFacetAdjudicationStatus.NO_ELIGIBLE_EVIDENCE
    if counts["eligible"] == counts["total"]:
        return (
            OledReviewedEvidenceFacetAdjudicationStatus
            .READY_FOR_GOLD_ADMISSION_PREFLIGHT
        )
    return (
        OledReviewedEvidenceFacetAdjudicationStatus
        .REVIEW_COMPLETE_WITH_BLOCKED_EVIDENCE
    )


__all__ = [
    "OLED_REVIEWED_EVIDENCE_FACET_ADJUDICATION_VERSION",
    "OLED_REVIEWED_EVIDENCE_FACET_DECISION_MANIFEST_VERSION",
    "OledConfidenceSufficiencyDecision",
    "OledReviewedEvidenceFacetAdjudicatedObservation",
    "OledReviewedEvidenceFacetAdjudicationArtifact",
    "OledReviewedEvidenceFacetAdjudicationStatus",
    "OledReviewedEvidenceFacetDecisionEntry",
    "OledReviewedEvidenceFacetDecisionManifest",
    "OledScientificConsistencyDecision",
    "build_oled_reviewed_evidence_facet_adjudication_artifact",
    "oled_reviewed_evidence_facet_adjudication_artifact_digest",
    "oled_reviewed_evidence_facet_decision_manifest_digest",
    "validate_oled_reviewed_evidence_facet_decisions",
]
