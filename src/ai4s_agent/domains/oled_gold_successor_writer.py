from __future__ import annotations

import hashlib
import json
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

from ai4s_agent.domains.oled_gold_candidate_postwrite_verifier import (
    OledGoldCandidatePostwriteVerificationArtifact,
    oled_gold_candidate_postwrite_verification_artifact_digest,
)
from ai4s_agent.domains.oled_gold_candidate_writer import (
    OledGoldCandidateSnapshot,
    oled_gold_candidate_snapshot_digest,
)
from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
    OledGoldSuccessorPreflightArtifact,
    OledGoldSuccessorPreflightStatus,
    categorical_gold_snapshot_publication_bytes,
    oled_categorical_gold_snapshot_digest,
    oled_gold_successor_preflight_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_GOLD_SUCCESSOR_WRITE_VERSION = "oled_gold_successor_write.v1"
GOLD_SUCCESSOR_WRITE_FILENAME = "categorical_gold_successor_write.json"
GOLD_SUCCESSOR_SNAPSHOT_FILENAME = "categorical_gold_snapshot.json"


class OledGoldSuccessorWriteStatus(str, Enum):
    PUBLISHED_AND_ACTIVATED = (
        "categorical_gold_successor_snapshot_published_and_activated"
    )


class OledGoldSuccessorWriteArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_GOLD_SUCCESSOR_WRITE_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    preflight_artifact_sha256: str
    preflight_artifact_digest: str
    verification_artifact_sha256: str
    verification_artifact_digest: str
    candidate_snapshot_sha256: str
    candidate_snapshot_digest: str
    prior_gold_snapshot_sha256: str
    prior_gold_snapshot_digest: str
    preflight_artifact: OledGoldSuccessorPreflightArtifact
    verification_artifact: OledGoldCandidatePostwriteVerificationArtifact
    candidate_snapshot: OledGoldCandidateSnapshot
    prior_gold_snapshot: OledCategoricalGoldSnapshot
    published_successor_snapshot: OledCategoricalGoldSnapshot
    status: OledGoldSuccessorWriteStatus
    publication_receipt_filename: Literal[
        "categorical_gold_successor_write.json"
    ] = GOLD_SUCCESSOR_WRITE_FILENAME
    published_snapshot_filename: Literal[
        "categorical_gold_snapshot.json"
    ] = GOLD_SUCCESSOR_SNAPSHOT_FILENAME
    published_snapshot_file_sha256: str
    prior_generation: Annotated[StrictInt, Field(ge=0)]
    published_generation: Annotated[StrictInt, Field(ge=1)]
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    added_entry_count: Annotated[StrictInt, Field(ge=1)]
    published_entry_count: Annotated[StrictInt, Field(ge=1)]
    added_gold_entry_ids: list[str] = Field(min_length=1)
    added_entry_digests: list[str] = Field(min_length=1)
    successor_snapshot_id: str
    successor_snapshot_digest: str
    activated_snapshot_id: str
    activated_snapshot_digest: str
    write_artifact_digest: str
    publication_receipt_created: StrictBool = True
    activation_receipt_created: StrictBool = True
    immutable_successor_snapshot_published: StrictBool = True
    categorical_gold_snapshot_published: StrictBool = True
    categorical_gold_snapshot_activated: StrictBool = True
    gold_head_activated: StrictBool = False
    mutable_gold_head_pointer_written: StrictBool = False
    compare_and_swap_parent_bytes_verified: StrictBool = True
    preflight_bytes_rechecked_before_publication: StrictBool = True
    verification_bytes_rechecked_before_publication: StrictBool = True
    candidate_snapshot_bytes_rechecked_before_publication: StrictBool = True
    prior_snapshot_bytes_rechecked_before_publication: StrictBool = True
    append_only_transition_verified: StrictBool = True
    atomic_noreplace_directory_publication: StrictBool = True
    fsync_files_and_directories_completed: StrictBool = True
    published_directory_inode_bound: StrictBool = True
    published_payloads_revalidated: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    prior_snapshot_mutated: StrictBool = False
    curated_dataset_written: StrictBool = False
    training_eligible: StrictBool = False
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
        if value != OLED_GOLD_SUCCESSOR_WRITE_VERSION:
            raise ValueError("unexpected Gold successor write version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("successor_snapshot_id", "activated_snapshot_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator(
        "preflight_artifact_sha256",
        "preflight_artifact_digest",
        "verification_artifact_sha256",
        "verification_artifact_digest",
        "candidate_snapshot_sha256",
        "candidate_snapshot_digest",
        "prior_gold_snapshot_sha256",
        "prior_gold_snapshot_digest",
        "published_snapshot_file_sha256",
        "successor_snapshot_digest",
        "activated_snapshot_digest",
        "write_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("added_gold_entry_ids", "added_entry_digests")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledGoldSuccessorWriteArtifact:
        preflight = self.preflight_artifact
        verification = self.verification_artifact
        candidate = self.candidate_snapshot
        prior = self.prior_gold_snapshot
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            preflight.generated_at
        ):
            raise ValueError("Gold successor write timestamp reversal")
        if (
            oled_gold_successor_preflight_artifact_digest(preflight)
            != preflight.preflight_artifact_digest
        ):
            raise ValueError("Gold successor write embedded preflight changed")
        if (
            oled_gold_candidate_postwrite_verification_artifact_digest(
                verification
            )
            != verification.verification_artifact_digest
        ):
            raise ValueError("Gold successor write embedded verification changed")
        if (
            oled_gold_candidate_snapshot_digest(candidate)
            != candidate.snapshot_digest
        ):
            raise ValueError("Gold successor write candidate snapshot changed")
        if (
            oled_categorical_gold_snapshot_digest(prior)
            != prior.snapshot_digest
        ):
            raise ValueError("Gold successor write prior snapshot changed")
        expected_bindings = {
            "run_id": preflight.run_id,
            "paper_id": preflight.paper_id,
            "preflight_artifact_digest": preflight.preflight_artifact_digest,
            "verification_artifact_sha256": (
                preflight.verification_artifact_sha256
            ),
            "verification_artifact_digest": (
                preflight.verification_artifact_digest
            ),
            "candidate_snapshot_sha256": preflight.candidate_snapshot_sha256,
            "candidate_snapshot_digest": preflight.candidate_snapshot_digest,
            "prior_gold_snapshot_sha256": (
                preflight.current_gold_snapshot_sha256
            ),
            "prior_gold_snapshot_digest": (
                preflight.current_gold_snapshot_digest
            ),
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Gold successor write {field_name} mismatch")
        if (
            verification.model_dump(mode="json")
            != preflight.verification_artifact.model_dump(mode="json")
            or candidate.model_dump(mode="json")
            != preflight.candidate_snapshot.model_dump(mode="json")
            or prior.model_dump(mode="json")
            != preflight.current_gold_snapshot.model_dump(mode="json")
        ):
            raise ValueError("Gold successor write exact input model mismatch")
        derived = _derive_gold_successor_write(preflight, prior)
        successor = derived["successor"]
        if self.published_successor_snapshot.model_dump(mode="json") != (
            successor.model_dump(mode="json")
        ):
            raise ValueError("published categorical Gold successor changed")
        expected_fields = {
            "status": OledGoldSuccessorWriteStatus.PUBLISHED_AND_ACTIVATED,
            "published_snapshot_file_sha256": derived["snapshot_file_sha256"],
            "prior_generation": prior.generation,
            "published_generation": successor.generation,
            "prior_entry_count": len(prior.entries),
            "added_entry_count": len(derived["added_gold_entry_ids"]),
            "published_entry_count": len(successor.entries),
            "added_gold_entry_ids": derived["added_gold_entry_ids"],
            "added_entry_digests": derived["added_entry_digests"],
            "successor_snapshot_id": successor.snapshot_id,
            "successor_snapshot_digest": successor.snapshot_digest,
            "activated_snapshot_id": successor.snapshot_id,
            "activated_snapshot_digest": successor.snapshot_digest,
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Gold successor write {field_name} mismatch")
        fixed_true = (
            "publication_receipt_created",
            "activation_receipt_created",
            "immutable_successor_snapshot_published",
            "categorical_gold_snapshot_published",
            "categorical_gold_snapshot_activated",
            "compare_and_swap_parent_bytes_verified",
            "preflight_bytes_rechecked_before_publication",
            "verification_bytes_rechecked_before_publication",
            "candidate_snapshot_bytes_rechecked_before_publication",
            "prior_snapshot_bytes_rechecked_before_publication",
            "append_only_transition_verified",
            "atomic_noreplace_directory_publication",
            "fsync_files_and_directories_completed",
            "published_directory_inode_bound",
            "published_payloads_revalidated",
            "categorical_confidence_only",
        )
        fixed_false = (
            "mutable_gold_head_pointer_written",
            "gold_head_activated",
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "prior_snapshot_mutated",
            "curated_dataset_written",
            "training_eligible",
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
            raise ValueError("Gold successor writer crossed its boundary")
        if oled_gold_successor_write_artifact_digest(self) != (
            self.write_artifact_digest
        ):
            raise ValueError("Gold successor write artifact digest mismatch")
        return self


def build_oled_gold_successor_write_artifact(
    *,
    preflight_artifact: OledGoldSuccessorPreflightArtifact,
    preflight_artifact_sha256: str,
    verification_artifact: OledGoldCandidatePostwriteVerificationArtifact,
    verification_artifact_sha256: str,
    candidate_snapshot: OledGoldCandidateSnapshot,
    candidate_snapshot_sha256: str,
    prior_gold_snapshot: OledCategoricalGoldSnapshot,
    prior_gold_snapshot_sha256: str,
    generated_at: str,
) -> OledGoldSuccessorWriteArtifact:
    preflight = OledGoldSuccessorPreflightArtifact.model_validate(
        preflight_artifact.model_dump(mode="json")
    )
    verification = OledGoldCandidatePostwriteVerificationArtifact.model_validate(
        verification_artifact.model_dump(mode="json")
    )
    candidate = OledGoldCandidateSnapshot.model_validate(
        candidate_snapshot.model_dump(mode="json")
    )
    prior = OledCategoricalGoldSnapshot.model_validate(
        prior_gold_snapshot.model_dump(mode="json")
    )
    derived = _derive_gold_successor_write(preflight, prior)
    successor = derived["successor"]
    payload: dict[str, Any] = {
        "artifact_version": OLED_GOLD_SUCCESSOR_WRITE_VERSION,
        "run_id": preflight.run_id,
        "paper_id": preflight.paper_id,
        "generated_at": generated_at,
        "preflight_artifact_sha256": _normalize_sha256(
            preflight_artifact_sha256,
            field_name="preflight_artifact_sha256",
        ),
        "preflight_artifact_digest": preflight.preflight_artifact_digest,
        "verification_artifact_sha256": _normalize_sha256(
            verification_artifact_sha256,
            field_name="verification_artifact_sha256",
        ),
        "verification_artifact_digest": verification.verification_artifact_digest,
        "candidate_snapshot_sha256": _normalize_sha256(
            candidate_snapshot_sha256,
            field_name="candidate_snapshot_sha256",
        ),
        "candidate_snapshot_digest": candidate.snapshot_digest,
        "prior_gold_snapshot_sha256": _normalize_sha256(
            prior_gold_snapshot_sha256,
            field_name="prior_gold_snapshot_sha256",
        ),
        "prior_gold_snapshot_digest": prior.snapshot_digest,
        "preflight_artifact": preflight,
        "verification_artifact": verification,
        "candidate_snapshot": candidate,
        "prior_gold_snapshot": prior,
        "published_successor_snapshot": successor,
        "status": OledGoldSuccessorWriteStatus.PUBLISHED_AND_ACTIVATED,
        "published_snapshot_file_sha256": derived["snapshot_file_sha256"],
        "prior_generation": prior.generation,
        "published_generation": successor.generation,
        "prior_entry_count": len(prior.entries),
        "added_entry_count": len(derived["added_gold_entry_ids"]),
        "published_entry_count": len(successor.entries),
        "added_gold_entry_ids": derived["added_gold_entry_ids"],
        "added_entry_digests": derived["added_entry_digests"],
        "successor_snapshot_id": successor.snapshot_id,
        "successor_snapshot_digest": successor.snapshot_digest,
        "activated_snapshot_id": successor.snapshot_id,
        "activated_snapshot_digest": successor.snapshot_digest,
        "write_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledGoldSuccessorWriteArtifact.model_construct(**payload)
    payload["write_artifact_digest"] = (
        oled_gold_successor_write_artifact_digest(provisional)
    )
    return OledGoldSuccessorWriteArtifact.model_validate(payload)


def _derive_gold_successor_write(
    preflight: OledGoldSuccessorPreflightArtifact,
    prior: OledCategoricalGoldSnapshot,
) -> dict[str, Any]:
    if (
        preflight.status
        != OledGoldSuccessorPreflightStatus.READY_FOR_SUCCESSOR_WRITE
        or preflight.expected_successor_snapshot is None
    ):
        raise ValueError("Gold successor preflight has no snapshot to publish")
    if (
        preflight.current_gold_snapshot_digest != prior.snapshot_digest
        or preflight.current_gold_snapshot.model_dump(mode="json")
        != prior.model_dump(mode="json")
    ):
        raise ValueError("compare-and-swap prior Gold does not match PR-AE")
    successor = OledCategoricalGoldSnapshot.model_validate(
        preflight.expected_successor_snapshot.model_dump(mode="json")
    )
    if (
        successor.snapshot_digest
        != preflight.expected_successor_snapshot_digest
    ):
        raise ValueError("PR-AE expected successor digest mismatch")
    prior_by_id = {entry.gold_entry_id: entry for entry in prior.entries}
    successor_by_id = {
        entry.gold_entry_id: entry for entry in successor.entries
    }
    for entry_id, prior_entry in prior_by_id.items():
        successor_entry = successor_by_id.get(entry_id)
        if successor_entry is None or successor_entry.model_dump(
            mode="json"
        ) != prior_entry.model_dump(mode="json"):
            raise ValueError("Gold successor changed a prior entry")
    planned_by_id = {
        item.gold_entry.gold_entry_id: item.gold_entry
        for item in preflight.planned_additions
    }
    if set(successor_by_id) != set(prior_by_id) | set(planned_by_id):
        raise ValueError("Gold successor entry coverage differs from PR-AE")
    for entry_id, planned_entry in planned_by_id.items():
        if successor_by_id[entry_id].model_dump(mode="json") != (
            planned_entry.model_dump(mode="json")
        ):
            raise ValueError("Gold successor planned entry changed")
    if (
        len(planned_by_id) != preflight.planned_addition_count
        or len(successor.entries) != preflight.expected_entry_count
    ):
        raise ValueError("Gold successor entry count mismatch")
    if (
        successor.parent_snapshot_digest != prior.snapshot_digest
        or successor.generation != prior.generation + 1
        or successor.source_verification_digest
        != preflight.verification_artifact_digest
    ):
        raise ValueError("Gold successor lineage mismatch")
    snapshot_bytes = categorical_gold_snapshot_publication_bytes(successor)
    return {
        "successor": successor,
        "snapshot_file_sha256": (
            f"sha256:{hashlib.sha256(snapshot_bytes).hexdigest()}"
        ),
        "added_gold_entry_ids": sorted(planned_by_id),
        "added_entry_digests": sorted(
            entry.entry_digest for entry in planned_by_id.values()
        ),
    }


def gold_successor_write_receipt_publication_bytes(
    artifact: OledGoldSuccessorWriteArtifact,
) -> bytes:
    return (
        json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def oled_gold_successor_write_artifact_digest(
    artifact: OledGoldSuccessorWriteArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("write_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "GOLD_SUCCESSOR_SNAPSHOT_FILENAME",
    "GOLD_SUCCESSOR_WRITE_FILENAME",
    "OLED_GOLD_SUCCESSOR_WRITE_VERSION",
    "OledGoldSuccessorWriteArtifact",
    "OledGoldSuccessorWriteStatus",
    "build_oled_gold_successor_write_artifact",
    "gold_successor_write_receipt_publication_bytes",
    "oled_gold_successor_write_artifact_digest",
]
