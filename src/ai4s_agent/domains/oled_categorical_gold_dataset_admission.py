from __future__ import annotations

import hashlib
import json
from collections import Counter
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

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_dataset_views import OledDatasetViewKind
from ai4s_agent.domains.oled_gold_successor_postwrite_verifier import (
    OledGoldSuccessorPostwriteVerificationArtifact,
    OledGoldSuccessorPostwriteVerificationStatus,
    independently_replay_gold_successor_postwrite,
    oled_gold_successor_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldEntry,
    OledCategoricalGoldSnapshot,
    categorical_gold_snapshot_publication_bytes,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    _id_hash,
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_CATEGORICAL_GOLD_DATASET_ADMISSION_VERSION = (
    "oled_categorical_gold_dataset_admission.v1"
)


class OledCategoricalGoldDatasetAdmissionStatus(str, Enum):
    COMPLETE = "categorical_gold_dataset_admission_complete"


class OledCategoricalGoldDatasetAdmissionDecisionStatus(str, Enum):
    ADMITTED = "admitted_to_explicit_views"
    NOT_ADMITTED = "not_admitted_to_any_current_view"


class OledCategoricalGoldViewEligibility(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    view_kind: OledDatasetViewKind
    eligible: StrictBool
    reason_code: str

    @field_validator("reason_code")
    @classmethod
    def validate_reason_code(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="reason_code")


class OledCategoricalGoldDatasetAdmissionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    admission_decision_id: str
    gold_entry_id: str
    gold_entry_digest: str
    source_candidate_id: str
    source_candidate_digest: str
    selected_material_id: str
    property_id: str
    target_layer: OledCausalLayer
    comparison_context_status: str
    comparison_context_hash: str | None = None
    status: OledCategoricalGoldDatasetAdmissionDecisionStatus
    eligible_view_kinds: list[OledDatasetViewKind] = Field(default_factory=list)
    view_eligibility: list[OledCategoricalGoldViewEligibility] = Field(
        min_length=len(OledDatasetViewKind),
        max_length=len(OledDatasetViewKind),
    )
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    dataset_row_materialized: StrictBool = False
    decision_digest: str

    @field_validator(
        "admission_decision_id",
        "gold_entry_id",
        "source_candidate_id",
        "selected_material_id",
        "property_id",
        "comparison_context_status",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "gold_entry_digest",
        "source_candidate_digest",
        "decision_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("comparison_context_hash")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_sha256(value, field_name="comparison_context_hash")

    @field_validator("eligible_view_kinds")
    @classmethod
    def validate_eligible_views(
        cls,
        value: list[OledDatasetViewKind],
    ) -> list[OledDatasetViewKind]:
        if value != sorted(value, key=lambda item: item.value) or len(value) != len(
            set(value)
        ):
            raise ValueError("eligible dataset views must be sorted and unique")
        return value

    @field_validator("view_eligibility")
    @classmethod
    def validate_view_roster(
        cls,
        value: list[OledCategoricalGoldViewEligibility],
    ) -> list[OledCategoricalGoldViewEligibility]:
        kinds = [item.view_kind for item in value]
        expected = sorted(OledDatasetViewKind, key=lambda item: item.value)
        if kinds != expected:
            raise ValueError("dataset-view eligibility roster must be complete and ordered")
        return value

    @model_validator(mode="after")
    def validate_decision(self) -> OledCategoricalGoldDatasetAdmissionDecision:
        eligible = [
            item.view_kind for item in self.view_eligibility if item.eligible
        ]
        if eligible != self.eligible_view_kinds:
            raise ValueError("eligible dataset-view summary mismatch")
        expected_status = (
            OledCategoricalGoldDatasetAdmissionDecisionStatus.ADMITTED
            if eligible
            else OledCategoricalGoldDatasetAdmissionDecisionStatus.NOT_ADMITTED
        )
        if self.status != expected_status:
            raise ValueError("dataset admission decision status mismatch")
        if (
            not self.categorical_confidence_only
            or self.numeric_confidence_score_assigned
            or self.legacy_numeric_confidence_record_constructed
            or self.dataset_row_materialized
        ):
            raise ValueError("dataset admission decision crossed its boundary")
        if (
            oled_categorical_gold_dataset_admission_decision_digest(self)
            != self.decision_digest
        ):
            raise ValueError("dataset admission decision digest mismatch")
        return self


class OledCategoricalGoldDatasetAdmissionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_CATEGORICAL_GOLD_DATASET_ADMISSION_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    verification_artifact_sha256: str
    verification_artifact_digest: str
    published_snapshot_sha256: str
    published_snapshot_digest: str
    verification_artifact: OledGoldSuccessorPostwriteVerificationArtifact
    published_snapshot: OledCategoricalGoldSnapshot
    status: OledCategoricalGoldDatasetAdmissionStatus
    gold_registry_id: str
    admitted_snapshot_id: str
    admitted_generation: Annotated[StrictInt, Field(ge=1)]
    input_entry_count: Annotated[StrictInt, Field(ge=1)]
    admitted_entry_count: Annotated[StrictInt, Field(ge=0)]
    not_admitted_entry_count: Annotated[StrictInt, Field(ge=0)]
    view_eligible_entry_counts: dict[OledDatasetViewKind, StrictInt]
    decisions: list[OledCategoricalGoldDatasetAdmissionDecision] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    admission_artifact_digest: str
    exact_verification_artifact_bytes_bound: StrictBool = True
    exact_published_snapshot_bytes_bound: StrictBool = True
    pr_ag_verification_replayed: StrictBool = True
    complete_snapshot_roster_replayed: StrictBool = True
    causal_layer_view_policy_replayed: StrictBool = True
    comparison_context_preserved: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    dataset_view_rows_written: StrictBool = False
    dataset_materialized: StrictBool = False
    split_assignments_created: StrictBool = False
    features_materialized: StrictBool = False
    training_package_created: StrictBool = False
    training_eligible: StrictBool = False
    gold_snapshot_written: StrictBool = False
    gold_head_activated: StrictBool = False
    reviewed_evidence_mutated: StrictBool = False
    registry_written: StrictBool = False
    source_pdf_read: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False
    standalone_input_bytes_revalidation_supported: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_CATEGORICAL_GOLD_DATASET_ADMISSION_VERSION:
            raise ValueError("unexpected categorical Gold dataset admission version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator("gold_registry_id", "admitted_snapshot_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "verification_artifact_sha256",
        "verification_artifact_digest",
        "published_snapshot_sha256",
        "published_snapshot_digest",
        "admission_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("decisions")
    @classmethod
    def validate_decision_order(
        cls,
        value: list[OledCategoricalGoldDatasetAdmissionDecision],
    ) -> list[OledCategoricalGoldDatasetAdmissionDecision]:
        ids = [item.gold_entry_id for item in value]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("dataset admission decisions must be Gold-entry ordered")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledCategoricalGoldDatasetAdmissionArtifact:
        verification = self.verification_artifact
        snapshot = self.published_snapshot
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            verification.generated_at
        ):
            raise ValueError("dataset admission predates PR-AG")
        if (
            verification.status
            != OledGoldSuccessorPostwriteVerificationStatus.VERIFIED
            or not verification.eligible_for_explicit_dataset_admission_input
        ):
            raise ValueError("PR-AG artifact is not eligible for dataset admission")
        if (
            oled_gold_successor_postwrite_verification_artifact_digest(verification)
            != self.verification_artifact_digest
            or verification.verification_artifact_digest
            != self.verification_artifact_digest
        ):
            raise ValueError("PR-AG verification digest mismatch")
        if self.verification_artifact_sha256 != _sha256_bytes(
            gold_successor_postwrite_verification_publication_bytes(verification)
        ):
            raise ValueError("published PR-AG verification file SHA-256 mismatch")
        if self.published_snapshot_sha256 != _sha256_bytes(
            categorical_gold_snapshot_publication_bytes(snapshot)
        ):
            raise ValueError("published categorical Gold snapshot file SHA-256 mismatch")
        if (
            snapshot.model_dump(mode="json")
            != verification.published_snapshot.model_dump(mode="json")
            or snapshot.snapshot_digest != self.published_snapshot_digest
        ):
            raise ValueError("dataset admission snapshot differs from PR-AG")
        independently_replay_gold_successor_postwrite(
            verification.write_artifact,
            snapshot,
            write_artifact_sha256=verification.write_artifact_sha256,
            published_snapshot_sha256=self.published_snapshot_sha256,
        )
        expected_decisions = [
            build_oled_categorical_gold_dataset_admission_decision(entry)
            for entry in snapshot.entries
        ]
        if [item.model_dump(mode="json") for item in self.decisions] != [
            item.model_dump(mode="json") for item in expected_decisions
        ]:
            raise ValueError("dataset admission decision roster mismatch")
        counts = Counter(
            view
            for decision in expected_decisions
            for view in decision.eligible_view_kinds
        )
        expected_view_counts = {
            view: counts.get(view, 0)
            for view in sorted(OledDatasetViewKind, key=lambda item: item.value)
        }
        admitted = sum(bool(item.eligible_view_kinds) for item in expected_decisions)
        expected = {
            "run_id": verification.run_id,
            "paper_id": verification.paper_id,
            "published_snapshot_digest": snapshot.snapshot_digest,
            "gold_registry_id": snapshot.gold_registry_id,
            "admitted_snapshot_id": snapshot.snapshot_id,
            "admitted_generation": snapshot.generation,
            "input_entry_count": len(snapshot.entries),
            "admitted_entry_count": admitted,
            "not_admitted_entry_count": len(snapshot.entries) - admitted,
            "view_eligible_entry_counts": expected_view_counts,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"dataset admission {name} mismatch")
        fixed_true = (
            "exact_verification_artifact_bytes_bound",
            "exact_published_snapshot_bytes_bound",
            "pr_ag_verification_replayed",
            "complete_snapshot_roster_replayed",
            "causal_layer_view_policy_replayed",
            "comparison_context_preserved",
            "categorical_confidence_only",
        )
        fixed_false = (
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "dataset_view_rows_written",
            "dataset_materialized",
            "split_assignments_created",
            "features_materialized",
            "training_package_created",
            "training_eligible",
            "gold_snapshot_written",
            "gold_head_activated",
            "reviewed_evidence_mutated",
            "registry_written",
            "source_pdf_read",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "standalone_input_bytes_revalidation_supported",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("categorical Gold dataset admission crossed its boundary")
        if (
            oled_categorical_gold_dataset_admission_artifact_digest(self)
            != self.admission_artifact_digest
        ):
            raise ValueError("dataset admission artifact digest mismatch")
        return self


def build_oled_categorical_gold_dataset_admission_decision(
    entry: OledCategoricalGoldEntry,
) -> OledCategoricalGoldDatasetAdmissionDecision:
    candidate = entry.candidate
    eligibility = _view_eligibility(entry)
    eligible_views = sorted(
        [item.view_kind for item in eligibility if item.eligible],
        key=lambda item: item.value,
    )
    payload: dict[str, Any] = {
        "admission_decision_id": _id_hash(
            "oled-categorical-gold-dataset-admission",
            {
                "gold_entry_digest": entry.entry_digest,
                "eligible_view_kinds": [item.value for item in eligible_views],
            },
        ),
        "gold_entry_id": entry.gold_entry_id,
        "gold_entry_digest": entry.entry_digest,
        "source_candidate_id": entry.source_candidate_id,
        "source_candidate_digest": entry.source_candidate_digest,
        "selected_material_id": entry.selected_material_id,
        "property_id": entry.property_id,
        "target_layer": candidate.target_layer,
        "comparison_context_status": candidate.comparison_context_status.value,
        "comparison_context_hash": candidate.comparison_context_hash,
        "status": (
            OledCategoricalGoldDatasetAdmissionDecisionStatus.ADMITTED
            if eligible_views
            else OledCategoricalGoldDatasetAdmissionDecisionStatus.NOT_ADMITTED
        ),
        "eligible_view_kinds": eligible_views,
        "view_eligibility": eligibility,
        "decision_digest": "sha256:" + "0" * 64,
    }
    provisional = OledCategoricalGoldDatasetAdmissionDecision.model_construct(
        **payload
    )
    payload["decision_digest"] = (
        oled_categorical_gold_dataset_admission_decision_digest(provisional)
    )
    return OledCategoricalGoldDatasetAdmissionDecision.model_validate(payload)


def build_oled_categorical_gold_dataset_admission_artifact(
    *,
    verification_artifact: OledGoldSuccessorPostwriteVerificationArtifact,
    verification_artifact_sha256: str,
    published_snapshot: OledCategoricalGoldSnapshot,
    published_snapshot_sha256: str,
    generated_at: str,
) -> OledCategoricalGoldDatasetAdmissionArtifact:
    verification = OledGoldSuccessorPostwriteVerificationArtifact.model_validate(
        verification_artifact.model_dump(mode="json")
    )
    snapshot = OledCategoricalGoldSnapshot.model_validate(
        published_snapshot.model_dump(mode="json")
    )
    if snapshot.model_dump(mode="json") != (
        verification.published_snapshot.model_dump(mode="json")
    ):
        raise ValueError("dataset admission snapshot differs from PR-AG")
    decisions = [
        build_oled_categorical_gold_dataset_admission_decision(entry)
        for entry in snapshot.entries
    ]
    counts = Counter(
        view for decision in decisions for view in decision.eligible_view_kinds
    )
    admitted = sum(bool(item.eligible_view_kinds) for item in decisions)
    payload: dict[str, Any] = {
        "artifact_version": OLED_CATEGORICAL_GOLD_DATASET_ADMISSION_VERSION,
        "run_id": verification.run_id,
        "paper_id": verification.paper_id,
        "generated_at": generated_at,
        "verification_artifact_sha256": _normalize_sha256(
            verification_artifact_sha256,
            field_name="verification_artifact_sha256",
        ),
        "verification_artifact_digest": verification.verification_artifact_digest,
        "published_snapshot_sha256": _normalize_sha256(
            published_snapshot_sha256,
            field_name="published_snapshot_sha256",
        ),
        "published_snapshot_digest": snapshot.snapshot_digest,
        "verification_artifact": verification,
        "published_snapshot": snapshot,
        "status": OledCategoricalGoldDatasetAdmissionStatus.COMPLETE,
        "gold_registry_id": snapshot.gold_registry_id,
        "admitted_snapshot_id": snapshot.snapshot_id,
        "admitted_generation": snapshot.generation,
        "input_entry_count": len(snapshot.entries),
        "admitted_entry_count": admitted,
        "not_admitted_entry_count": len(snapshot.entries) - admitted,
        "view_eligible_entry_counts": {
            view: counts.get(view, 0)
            for view in sorted(OledDatasetViewKind, key=lambda item: item.value)
        },
        "decisions": decisions,
        "admission_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledCategoricalGoldDatasetAdmissionArtifact.model_construct(
        **payload
    )
    payload["admission_artifact_digest"] = (
        oled_categorical_gold_dataset_admission_artifact_digest(provisional)
    )
    return OledCategoricalGoldDatasetAdmissionArtifact.model_validate(payload)


def gold_successor_postwrite_verification_publication_bytes(
    artifact: OledGoldSuccessorPostwriteVerificationArtifact,
) -> bytes:
    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def oled_categorical_gold_dataset_admission_decision_digest(
    decision: OledCategoricalGoldDatasetAdmissionDecision,
) -> str:
    payload = decision.model_dump(mode="json")
    payload.pop("decision_digest", None)
    return _stable_hash(payload)


def oled_categorical_gold_dataset_admission_artifact_digest(
    artifact: OledCategoricalGoldDatasetAdmissionArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("admission_artifact_digest", None)
    return _stable_hash(payload)


def _view_eligibility(
    entry: OledCategoricalGoldEntry,
) -> list[OledCategoricalGoldViewEligibility]:
    layer = entry.candidate.target_layer
    result: dict[OledDatasetViewKind, tuple[bool, str]] = {
        OledDatasetViewKind.RAW_ALL_MEASUREMENTS: (
            layer == OledCausalLayer.MEASUREMENT,
            (
                "measurement_layer_admitted"
                if layer == OledCausalLayer.MEASUREMENT
                else "requires_measurement_layer"
            ),
        ),
        OledDatasetViewKind.CURATED_INTRINSIC: (
            layer == OledCausalLayer.MOLECULE,
            (
                "molecular_layer_admitted"
                if layer == OledCausalLayer.MOLECULE
                else "requires_molecular_layer"
            ),
        ),
        OledDatasetViewKind.CURATED_DEVICE_BASELINE: (
            False,
            "requires_device_and_confounder_semantics_not_in_categorical_gold",
        ),
        OledDatasetViewKind.BEST_REPORTED: (
            False,
            "requires_explicit_best_reported_semantics_not_in_categorical_gold",
        ),
    }
    return [
        OledCategoricalGoldViewEligibility(
            view_kind=view,
            eligible=result[view][0],
            reason_code=result[view][1],
        )
        for view in sorted(OledDatasetViewKind, key=lambda item: item.value)
    ]


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "OLED_CATEGORICAL_GOLD_DATASET_ADMISSION_VERSION",
    "OledCategoricalGoldDatasetAdmissionArtifact",
    "OledCategoricalGoldDatasetAdmissionDecision",
    "OledCategoricalGoldDatasetAdmissionDecisionStatus",
    "OledCategoricalGoldDatasetAdmissionStatus",
    "OledCategoricalGoldViewEligibility",
    "build_oled_categorical_gold_dataset_admission_artifact",
    "build_oled_categorical_gold_dataset_admission_decision",
    "gold_successor_postwrite_verification_publication_bytes",
    "oled_categorical_gold_dataset_admission_artifact_digest",
    "oled_categorical_gold_dataset_admission_decision_digest",
]
