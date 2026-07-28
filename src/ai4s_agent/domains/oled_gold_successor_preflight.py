from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from ai4s_agent.domains.oled_gold_admission_preflight import (
    OledGoldAdmissionCandidate,
)
from ai4s_agent.domains.oled_gold_candidate_postwrite_verifier import (
    OledGoldCandidatePostwriteVerificationArtifact,
    OledGoldCandidatePostwriteVerificationStatus,
    independently_replay_gold_candidate_postwrite,
    oled_gold_candidate_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    OledGoldCandidateSnapshot,
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


OLED_CATEGORICAL_GOLD_SNAPSHOT_VERSION = "oled_categorical_gold_snapshot.v1"
OLED_GOLD_SUCCESSOR_PREFLIGHT_VERSION = "oled_gold_successor_preflight.v1"


class OledGoldSuccessorPreflightStatus(str, Enum):
    READY_FOR_SUCCESSOR_WRITE = "ready_for_gold_successor_write"


class OledCategoricalGoldEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    gold_entry_id: str
    source_candidate_snapshot_id: str
    source_candidate_snapshot_digest: str
    source_candidate_id: str
    source_candidate_digest: str
    source_adjudicated_observation_digest: str
    source_cell_digest: str
    selected_material_id: str
    property_id: str
    comparison_context_hash: str | None = None
    candidate: OledGoldAdmissionCandidate
    scientific_consistency: str = "consistent"
    confidence_sufficiency: str = "sufficient"
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    entry_digest: str

    @field_validator(
        "gold_entry_id",
        "source_candidate_snapshot_id",
        "source_candidate_id",
        "selected_material_id",
        "property_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator(
        "source_candidate_snapshot_digest",
        "source_candidate_digest",
        "source_adjudicated_observation_digest",
        "source_cell_digest",
        "entry_digest",
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

    @model_validator(mode="after")
    def validate_entry(self) -> OledCategoricalGoldEntry:
        candidate = self.candidate
        expected_gold_entry_id = _id_hash(
            "oled-categorical-gold-entry",
            {"candidate_digest": candidate.candidate_digest},
        )
        if self.gold_entry_id != expected_gold_entry_id:
            raise ValueError("categorical Gold entry ID is not deterministic")
        expected = {
            "source_candidate_id": candidate.candidate_id,
            "source_candidate_digest": candidate.candidate_digest,
            "source_adjudicated_observation_digest": (
                candidate.source_adjudicated_observation_digest
            ),
            "source_cell_digest": candidate.source_cell_digest,
            "selected_material_id": candidate.selected_material_id,
            "property_id": candidate.property_id,
            "comparison_context_hash": candidate.comparison_context_hash,
            "scientific_consistency": candidate.scientific_consistency,
            "confidence_sufficiency": candidate.confidence_sufficiency,
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ValueError(f"categorical Gold entry {field_name} mismatch")
        if (
            not self.categorical_confidence_only
            or self.numeric_confidence_score_assigned
            or self.legacy_numeric_confidence_record_constructed
        ):
            raise ValueError("categorical Gold entry crossed its confidence boundary")
        if oled_categorical_gold_entry_digest(self) != self.entry_digest:
            raise ValueError("categorical Gold entry digest mismatch")
        return self


class OledCategoricalGoldSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    snapshot_version: str = OLED_CATEGORICAL_GOLD_SNAPSHOT_VERSION
    gold_registry_id: str
    snapshot_id: str
    generated_at: str
    generation: Annotated[StrictInt, Field(ge=0)]
    parent_snapshot_digest: str | None = None
    source_verification_digest: str | None = None
    entry_count: Annotated[StrictInt, Field(ge=0)]
    entries: list[OledCategoricalGoldEntry] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    snapshot_digest: str
    immutable_snapshot: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    curated_dataset_written: StrictBool = False
    training_eligible: StrictBool = False

    @field_validator("snapshot_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_CATEGORICAL_GOLD_SNAPSHOT_VERSION:
            raise ValueError("unexpected categorical Gold snapshot version")
        return value

    @field_validator("gold_registry_id", "snapshot_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator(
        "parent_snapshot_digest",
        "source_verification_digest",
        mode="before",
    )
    @classmethod
    def validate_optional_digests(
        cls,
        value: str | None,
        info: Any,
    ) -> str | None:
        if value is None:
            return None
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("snapshot_digest")
    @classmethod
    def validate_snapshot_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="snapshot_digest")

    @field_validator("entries")
    @classmethod
    def validate_entry_order(
        cls,
        value: list[OledCategoricalGoldEntry],
    ) -> list[OledCategoricalGoldEntry]:
        entry_ids = [entry.gold_entry_id for entry in value]
        if entry_ids != sorted(entry_ids) or len(entry_ids) != len(set(entry_ids)):
            raise ValueError("categorical Gold entries must be ID ordered and unique")
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> OledCategoricalGoldSnapshot:
        if self.entry_count != len(self.entries):
            raise ValueError("categorical Gold snapshot entry count mismatch")
        _validate_snapshot_internal_uniqueness(self.entries)
        if self.generation == 0:
            if self.entries or self.parent_snapshot_digest is not None:
                raise ValueError("categorical Gold genesis snapshot must be empty")
            if self.source_verification_digest is not None:
                raise ValueError("categorical Gold genesis cannot bind a verification")
        else:
            if self.parent_snapshot_digest is None:
                raise ValueError("categorical Gold successor requires a parent digest")
            if self.source_verification_digest is None:
                raise ValueError("categorical Gold successor requires verification lineage")
        if (
            not self.immutable_snapshot
            or not self.categorical_confidence_only
            or self.numeric_confidence_score_assigned
            or self.legacy_numeric_confidence_record_constructed
            or self.curated_dataset_written
            or self.training_eligible
        ):
            raise ValueError("categorical Gold snapshot crossed its boundary")
        if oled_categorical_gold_snapshot_digest(self) != self.snapshot_digest:
            raise ValueError("categorical Gold snapshot digest mismatch")
        return self


class OledGoldPlannedAddition(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_candidate_id: str
    source_candidate_digest: str
    gold_entry: OledCategoricalGoldEntry
    planned_addition_digest: str

    @field_validator("source_candidate_id")
    @classmethod
    def validate_source_candidate_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="source_candidate_id")

    @field_validator("source_candidate_digest", "planned_addition_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_addition(self) -> OledGoldPlannedAddition:
        if (
            self.source_candidate_id != self.gold_entry.source_candidate_id
            or self.source_candidate_digest
            != self.gold_entry.source_candidate_digest
        ):
            raise ValueError("Gold planned addition source mismatch")
        if oled_gold_planned_addition_digest(self) != self.planned_addition_digest:
            raise ValueError("Gold planned addition digest mismatch")
        return self


class OledGoldSuccessorPreflightArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_GOLD_SUCCESSOR_PREFLIGHT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    verification_artifact_sha256: str
    verification_artifact_digest: str
    candidate_snapshot_sha256: str
    candidate_snapshot_digest: str
    current_gold_snapshot_sha256: str
    current_gold_snapshot_digest: str
    verification_artifact: OledGoldCandidatePostwriteVerificationArtifact
    candidate_snapshot: OledGoldCandidateSnapshot
    current_gold_snapshot: OledCategoricalGoldSnapshot
    status: OledGoldSuccessorPreflightStatus
    candidate_count: Annotated[StrictInt, Field(ge=1)]
    planned_addition_count: Annotated[StrictInt, Field(ge=0)]
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    expected_entry_count: Annotated[StrictInt, Field(ge=0)]
    planned_additions: list[OledGoldPlannedAddition] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    expected_successor_snapshot_digest: str = ""
    expected_successor_snapshot: OledCategoricalGoldSnapshot | None = None
    preflight_artifact_digest: str
    offline_only: StrictBool = True
    plan_only: StrictBool = True
    exact_verification_bytes_bound_at_construction: StrictBool = True
    exact_candidate_snapshot_bytes_bound_at_construction: StrictBool = True
    exact_current_snapshot_bytes_bound_at_construction: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    candidate_publication_independently_replayed: StrictBool = True
    current_snapshot_parent_bound: StrictBool = True
    current_snapshot_lineage_receipt_bound: StrictBool = False
    current_snapshot_conflicts_rechecked: StrictBool = True
    deterministic_successor_planned: StrictBool = True
    append_only_plan_verified: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_snapshot_written: StrictBool = False
    gold_head_activated: StrictBool = False
    curated_dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    reviewed_evidence_mutated: StrictBool = False
    registry_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_GOLD_SUCCESSOR_PREFLIGHT_VERSION:
            raise ValueError("unexpected Gold successor preflight version")
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
        "verification_artifact_sha256",
        "verification_artifact_digest",
        "candidate_snapshot_sha256",
        "candidate_snapshot_digest",
        "current_gold_snapshot_sha256",
        "current_gold_snapshot_digest",
        "preflight_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("expected_successor_snapshot_digest")
    @classmethod
    def validate_optional_successor_digest(cls, value: str) -> str:
        if not value:
            return ""
        return _normalize_sha256(
            value,
            field_name="expected_successor_snapshot_digest",
        )

    @field_validator("planned_additions")
    @classmethod
    def validate_addition_order(
        cls,
        value: list[OledGoldPlannedAddition],
    ) -> list[OledGoldPlannedAddition]:
        ids = [item.gold_entry.gold_entry_id for item in value]
        if ids != sorted(ids) or len(ids) != len(set(ids)):
            raise ValueError("Gold planned additions must be entry-ID ordered")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledGoldSuccessorPreflightArtifact:
        verification = self.verification_artifact
        current = self.current_gold_snapshot
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            verification.generated_at
        ) or _parse_timestamp(self.generated_at) < _parse_timestamp(
            current.generated_at
        ):
            raise ValueError("Gold successor preflight timestamp reversal")
        if (
            verification.status
            != OledGoldCandidatePostwriteVerificationStatus.VERIFIED
            or not verification.eligible_for_explicit_gold_publication_input
        ):
            raise ValueError("Gold successor input is not publication eligible")
        expected_bindings = {
            "run_id": verification.run_id,
            "paper_id": verification.paper_id,
            "verification_artifact_digest": verification.verification_artifact_digest,
            "candidate_snapshot_digest": self.candidate_snapshot.snapshot_digest,
            "current_gold_snapshot_digest": current.snapshot_digest,
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Gold successor preflight {field_name} mismatch")
        if (
            oled_gold_candidate_postwrite_verification_artifact_digest(verification)
            != verification.verification_artifact_digest
        ):
            raise ValueError("Gold successor embedded PR-AD changed")
        replay = independently_replay_gold_candidate_postwrite(
            verification.write_artifact,
            self.candidate_snapshot,
            write_artifact_sha256=verification.write_artifact_sha256,
            published_snapshot_sha256=self.candidate_snapshot_sha256,
        )
        if replay["candidate_ids"] != verification.verified_candidate_ids:
            raise ValueError("Gold successor candidate replay mismatch")
        derived = _derive_gold_successor_preflight(
            verification=verification,
            candidate_snapshot=self.candidate_snapshot,
            current_snapshot=current,
            generated_at=self.generated_at,
        )
        expected_fields = {
            "status": derived["status"],
            "candidate_count": len(self.candidate_snapshot.candidates),
            "planned_addition_count": len(derived["planned_additions"]),
            "prior_entry_count": len(current.entries),
            "expected_entry_count": derived["expected_entry_count"],
            "expected_successor_snapshot_digest": derived[
                "expected_successor_snapshot_digest"
            ],
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Gold successor preflight {field_name} mismatch")
        if [item.model_dump(mode="json") for item in self.planned_additions] != [
            item.model_dump(mode="json") for item in derived["planned_additions"]
        ]:
            raise ValueError("Gold successor planned additions changed")
        expected_snapshot = derived["expected_successor_snapshot"]
        if (
            self.expected_successor_snapshot.model_dump(mode="json")
            if self.expected_successor_snapshot is not None
            else None
        ) != (
            expected_snapshot.model_dump(mode="json")
            if expected_snapshot is not None
            else None
        ):
            raise ValueError("Gold expected successor snapshot changed")
        fixed_true = (
            "offline_only",
            "plan_only",
            "exact_verification_bytes_bound_at_construction",
            "exact_candidate_snapshot_bytes_bound_at_construction",
            "exact_current_snapshot_bytes_bound_at_construction",
            "candidate_publication_independently_replayed",
            "current_snapshot_parent_bound",
            "current_snapshot_conflicts_rechecked",
            "deterministic_successor_planned",
            "append_only_plan_verified",
            "categorical_confidence_only",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "current_snapshot_lineage_receipt_bound",
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "gold_snapshot_written",
            "gold_head_activated",
            "curated_dataset_written",
            "training_eligible",
            "reviewed_evidence_mutated",
            "registry_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("Gold successor preflight crossed its boundary")
        if oled_gold_successor_preflight_artifact_digest(self) != (
            self.preflight_artifact_digest
        ):
            raise ValueError("Gold successor preflight artifact digest mismatch")
        return self


def build_oled_categorical_gold_genesis_snapshot(
    *,
    gold_registry_id: str,
    generated_at: str,
) -> OledCategoricalGoldSnapshot:
    payload: dict[str, Any] = {
        "snapshot_version": OLED_CATEGORICAL_GOLD_SNAPSHOT_VERSION,
        "gold_registry_id": gold_registry_id,
        "snapshot_id": _id_hash(
            "oled-categorical-gold-genesis",
            {"gold_registry_id": gold_registry_id},
        ),
        "generated_at": generated_at,
        "generation": 0,
        "parent_snapshot_digest": None,
        "source_verification_digest": None,
        "entry_count": 0,
        "entries": [],
        "snapshot_digest": "sha256:" + "0" * 64,
    }
    provisional = OledCategoricalGoldSnapshot.model_construct(**payload)
    payload["snapshot_digest"] = oled_categorical_gold_snapshot_digest(provisional)
    return OledCategoricalGoldSnapshot.model_validate(payload)


def build_oled_gold_successor_preflight_artifact(
    *,
    verification_artifact: OledGoldCandidatePostwriteVerificationArtifact,
    verification_artifact_sha256: str,
    candidate_snapshot: OledGoldCandidateSnapshot,
    candidate_snapshot_sha256: str,
    current_gold_snapshot: OledCategoricalGoldSnapshot,
    current_gold_snapshot_sha256: str,
    generated_at: str,
) -> OledGoldSuccessorPreflightArtifact:
    verification = OledGoldCandidatePostwriteVerificationArtifact.model_validate(
        verification_artifact.model_dump(mode="json")
    )
    candidate = OledGoldCandidateSnapshot.model_validate(
        candidate_snapshot.model_dump(mode="json")
    )
    current = OledCategoricalGoldSnapshot.model_validate(
        current_gold_snapshot.model_dump(mode="json")
    )
    normalized_candidate_sha = _normalize_sha256(
        candidate_snapshot_sha256,
        field_name="candidate_snapshot_sha256",
    )
    independently_replay_gold_candidate_postwrite(
        verification.write_artifact,
        candidate,
        write_artifact_sha256=verification.write_artifact_sha256,
        published_snapshot_sha256=normalized_candidate_sha,
    )
    derived = _derive_gold_successor_preflight(
        verification=verification,
        candidate_snapshot=candidate,
        current_snapshot=current,
        generated_at=generated_at,
    )
    payload: dict[str, Any] = {
        "artifact_version": OLED_GOLD_SUCCESSOR_PREFLIGHT_VERSION,
        "run_id": verification.run_id,
        "paper_id": verification.paper_id,
        "generated_at": generated_at,
        "verification_artifact_sha256": _normalize_sha256(
            verification_artifact_sha256,
            field_name="verification_artifact_sha256",
        ),
        "verification_artifact_digest": verification.verification_artifact_digest,
        "candidate_snapshot_sha256": normalized_candidate_sha,
        "candidate_snapshot_digest": candidate.snapshot_digest,
        "current_gold_snapshot_sha256": _normalize_sha256(
            current_gold_snapshot_sha256,
            field_name="current_gold_snapshot_sha256",
        ),
        "current_gold_snapshot_digest": current.snapshot_digest,
        "verification_artifact": verification,
        "candidate_snapshot": candidate,
        "current_gold_snapshot": current,
        "status": derived["status"],
        "candidate_count": len(candidate.candidates),
        "planned_addition_count": len(derived["planned_additions"]),
        "prior_entry_count": len(current.entries),
        "expected_entry_count": derived["expected_entry_count"],
        "planned_additions": derived["planned_additions"],
        "expected_successor_snapshot_digest": derived[
            "expected_successor_snapshot_digest"
        ],
        "expected_successor_snapshot": derived["expected_successor_snapshot"],
        "preflight_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledGoldSuccessorPreflightArtifact.model_construct(**payload)
    payload["preflight_artifact_digest"] = (
        oled_gold_successor_preflight_artifact_digest(provisional)
    )
    return OledGoldSuccessorPreflightArtifact.model_validate(payload)


def _derive_gold_successor_preflight(
    *,
    verification: OledGoldCandidatePostwriteVerificationArtifact,
    candidate_snapshot: OledGoldCandidateSnapshot,
    current_snapshot: OledCategoricalGoldSnapshot,
    generated_at: str,
) -> dict[str, Any]:
    if _parse_timestamp(generated_at) < _parse_timestamp(
        verification.generated_at
    ) or _parse_timestamp(generated_at) < _parse_timestamp(
        current_snapshot.generated_at
    ):
        raise ValueError("Gold successor preflight timestamp reversal")
    if (
        candidate_snapshot.snapshot_id != verification.snapshot_id
        or candidate_snapshot.snapshot_digest
        != verification.published_snapshot_digest
    ):
        raise ValueError("Gold successor candidate snapshot binding mismatch")

    additions = sorted(
        (
            _build_gold_planned_addition(
                candidate,
                candidate_snapshot=candidate_snapshot,
            )
            for candidate in candidate_snapshot.candidates
        ),
        key=lambda item: item.gold_entry.gold_entry_id,
    )
    _recheck_gold_conflicts(
        current_snapshot=current_snapshot,
        planned_additions=additions,
    )
    successor = _build_successor_snapshot(
        current_snapshot=current_snapshot,
        verification=verification,
        planned_additions=additions,
        generated_at=generated_at,
    )
    return {
        "status": OledGoldSuccessorPreflightStatus.READY_FOR_SUCCESSOR_WRITE,
        "planned_additions": additions,
        "expected_entry_count": len(successor.entries),
        "expected_successor_snapshot_digest": successor.snapshot_digest,
        "expected_successor_snapshot": successor,
    }


def _build_gold_planned_addition(
    candidate: OledGoldAdmissionCandidate,
    *,
    candidate_snapshot: OledGoldCandidateSnapshot,
) -> OledGoldPlannedAddition:
    entry_payload: dict[str, Any] = {
        "gold_entry_id": _id_hash(
            "oled-categorical-gold-entry",
            {"candidate_digest": candidate.candidate_digest},
        ),
        "source_candidate_snapshot_id": candidate_snapshot.snapshot_id,
        "source_candidate_snapshot_digest": candidate_snapshot.snapshot_digest,
        "source_candidate_id": candidate.candidate_id,
        "source_candidate_digest": candidate.candidate_digest,
        "source_adjudicated_observation_digest": (
            candidate.source_adjudicated_observation_digest
        ),
        "source_cell_digest": candidate.source_cell_digest,
        "selected_material_id": candidate.selected_material_id,
        "property_id": candidate.property_id,
        "comparison_context_hash": candidate.comparison_context_hash,
        "candidate": candidate,
        "scientific_consistency": candidate.scientific_consistency,
        "confidence_sufficiency": candidate.confidence_sufficiency,
        "entry_digest": "sha256:" + "0" * 64,
    }
    provisional_entry = OledCategoricalGoldEntry.model_construct(**entry_payload)
    entry_payload["entry_digest"] = oled_categorical_gold_entry_digest(
        provisional_entry
    )
    entry = OledCategoricalGoldEntry.model_validate(entry_payload)
    addition_payload: dict[str, Any] = {
        "source_candidate_id": candidate.candidate_id,
        "source_candidate_digest": candidate.candidate_digest,
        "gold_entry": entry,
        "planned_addition_digest": "sha256:" + "0" * 64,
    }
    provisional_addition = OledGoldPlannedAddition.model_construct(
        **addition_payload
    )
    addition_payload["planned_addition_digest"] = (
        oled_gold_planned_addition_digest(provisional_addition)
    )
    return OledGoldPlannedAddition.model_validate(addition_payload)


def _recheck_gold_conflicts(
    *,
    current_snapshot: OledCategoricalGoldSnapshot,
    planned_additions: Sequence[OledGoldPlannedAddition],
) -> None:
    existing = current_snapshot.entries
    dimensions = {
        "Gold entry ID": ({entry.gold_entry_id for entry in existing}, set()),
        "candidate ID": ({entry.source_candidate_id for entry in existing}, set()),
        "candidate digest": (
            {entry.source_candidate_digest for entry in existing},
            set(),
        ),
        "observation digest": (
            {entry.source_adjudicated_observation_digest for entry in existing},
            set(),
        ),
        "source cell digest": (
            {entry.source_cell_digest for entry in existing},
            set(),
        ),
    }
    existing_semantics = {
        _semantic_key(entry.candidate): entry.source_candidate_digest
        for entry in existing
    }
    batch_semantics: dict[tuple[Any, ...], str] = {}
    for addition in planned_additions:
        entry = addition.gold_entry
        values = {
            "Gold entry ID": entry.gold_entry_id,
            "candidate ID": entry.source_candidate_id,
            "candidate digest": entry.source_candidate_digest,
            "observation digest": entry.source_adjudicated_observation_digest,
            "source cell digest": entry.source_cell_digest,
        }
        for label, value in values.items():
            existing_values, batch_values = dimensions[label]
            if value in existing_values:
                raise ValueError(f"Gold successor {label} already exists")
            if value in batch_values:
                raise ValueError(f"Gold successor {label} repeats in candidate batch")
            batch_values.add(value)
        semantic_key = _semantic_key(entry.candidate)
        if semantic_key in existing_semantics:
            raise ValueError("Gold successor semantic observation conflicts with current Gold")
        if semantic_key in batch_semantics:
            raise ValueError("Gold successor semantic observation repeats in candidate batch")
        batch_semantics[semantic_key] = entry.source_candidate_digest


def _validate_snapshot_internal_uniqueness(
    entries: Sequence[OledCategoricalGoldEntry],
) -> None:
    dimensions = {
        "source candidate ID": [entry.source_candidate_id for entry in entries],
        "source candidate digest": [
            entry.source_candidate_digest for entry in entries
        ],
        "adjudicated observation digest": [
            entry.source_adjudicated_observation_digest for entry in entries
        ],
        "source cell digest": [entry.source_cell_digest for entry in entries],
        "semantic observation": [_semantic_key(entry.candidate) for entry in entries],
    }
    for label, values in dimensions.items():
        if len(values) != len(set(values)):
            raise ValueError(
                f"categorical Gold snapshot contains duplicate {label}"
            )


def _semantic_key(candidate: OledGoldAdmissionCandidate) -> tuple[Any, ...]:
    return (
        candidate.selected_material_id,
        candidate.property_id,
        candidate.target_layer.value,
        candidate.normalized_value,
        candidate.normalized_unit,
        candidate.comparison_context_hash,
    )


def _build_successor_snapshot(
    *,
    current_snapshot: OledCategoricalGoldSnapshot,
    verification: OledGoldCandidatePostwriteVerificationArtifact,
    planned_additions: Sequence[OledGoldPlannedAddition],
    generated_at: str,
) -> OledCategoricalGoldSnapshot:
    entries = sorted(
        [
            *current_snapshot.entries,
            *(addition.gold_entry for addition in planned_additions),
        ],
        key=lambda entry: entry.gold_entry_id,
    )
    payload: dict[str, Any] = {
        "snapshot_version": OLED_CATEGORICAL_GOLD_SNAPSHOT_VERSION,
        "gold_registry_id": current_snapshot.gold_registry_id,
        "snapshot_id": _id_hash(
            "oled-categorical-gold-successor",
            {
                "parent_snapshot_digest": current_snapshot.snapshot_digest,
                "verification_digest": verification.verification_artifact_digest,
                "planned_addition_digests": [
                    addition.planned_addition_digest
                    for addition in planned_additions
                ],
            },
        ),
        "generated_at": generated_at,
        "generation": current_snapshot.generation + 1,
        "parent_snapshot_digest": current_snapshot.snapshot_digest,
        "source_verification_digest": verification.verification_artifact_digest,
        "entry_count": len(entries),
        "entries": entries,
        "snapshot_digest": "sha256:" + "0" * 64,
    }
    provisional = OledCategoricalGoldSnapshot.model_construct(**payload)
    payload["snapshot_digest"] = oled_categorical_gold_snapshot_digest(provisional)
    return OledCategoricalGoldSnapshot.model_validate(payload)


def oled_categorical_gold_entry_digest(entry: OledCategoricalGoldEntry) -> str:
    payload = entry.model_dump(mode="json")
    payload.pop("entry_digest", None)
    return _stable_hash(payload)


def oled_gold_planned_addition_digest(addition: OledGoldPlannedAddition) -> str:
    payload = addition.model_dump(mode="json")
    payload.pop("planned_addition_digest", None)
    return _stable_hash(payload)


def oled_categorical_gold_snapshot_digest(
    snapshot: OledCategoricalGoldSnapshot,
) -> str:
    payload = snapshot.model_dump(mode="json")
    payload.pop("snapshot_digest", None)
    return _stable_hash(payload)


def oled_gold_successor_preflight_artifact_digest(
    artifact: OledGoldSuccessorPreflightArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("preflight_artifact_digest", None)
    return _stable_hash(payload)


def categorical_gold_snapshot_publication_bytes(
    snapshot: OledCategoricalGoldSnapshot,
) -> bytes:
    return (
        json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


__all__ = [
    "OLED_CATEGORICAL_GOLD_SNAPSHOT_VERSION",
    "OLED_GOLD_SUCCESSOR_PREFLIGHT_VERSION",
    "OledCategoricalGoldEntry",
    "OledCategoricalGoldSnapshot",
    "OledGoldPlannedAddition",
    "OledGoldSuccessorPreflightArtifact",
    "OledGoldSuccessorPreflightStatus",
    "build_oled_categorical_gold_genesis_snapshot",
    "build_oled_gold_successor_preflight_artifact",
    "categorical_gold_snapshot_publication_bytes",
    "oled_categorical_gold_entry_digest",
    "oled_categorical_gold_snapshot_digest",
    "oled_gold_planned_addition_digest",
    "oled_gold_successor_preflight_artifact_digest",
]
