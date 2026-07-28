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

from ai4s_agent.domains.oled_gold_admission_preflight import (
    OledGoldAdmissionCandidate,
    OledGoldAdmissionPreflightArtifact,
    OledGoldAdmissionPreflightStatus,
    oled_gold_admission_preflight_artifact_digest,
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


OLED_GOLD_CANDIDATE_SNAPSHOT_VERSION = "oled_gold_candidate_snapshot.v1"
OLED_GOLD_CANDIDATE_WRITE_VERSION = "oled_gold_candidate_write.v1"
GOLD_CANDIDATE_WRITE_FILENAME = "gold_candidate_write.json"
GOLD_CANDIDATE_SNAPSHOT_FILENAME = "gold_candidate_snapshot.json"


class OledGoldCandidateWriteStatus(str, Enum):
    PUBLISHED = "immutable_gold_candidate_snapshot_published"


class OledGoldCandidateSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    snapshot_version: str = OLED_GOLD_CANDIDATE_SNAPSHOT_VERSION
    snapshot_id: str
    generated_at: str
    source_preflight_digest: str
    entry_count: Annotated[StrictInt, Field(ge=1)]
    candidates: list[OledGoldAdmissionCandidate] = Field(
        min_length=1,
        max_length=1_000_000,
    )
    snapshot_digest: str
    immutable_candidate_only: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_records_created: StrictBool = False
    curated_dataset_written: StrictBool = False
    training_eligible: StrictBool = False

    @field_validator("snapshot_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_GOLD_CANDIDATE_SNAPSHOT_VERSION:
            raise ValueError("unexpected Gold candidate snapshot version")
        return value

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="snapshot_id")

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator("source_preflight_digest", "snapshot_digest")
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
            raise ValueError("Gold snapshot candidates must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_snapshot(self) -> OledGoldCandidateSnapshot:
        if self.entry_count != len(self.candidates):
            raise ValueError("Gold candidate snapshot count mismatch")
        fixed_true = ("immutable_candidate_only", "categorical_confidence_only")
        fixed_false = (
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "gold_records_created",
            "curated_dataset_written",
            "training_eligible",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("Gold candidate snapshot crossed its boundary")
        if oled_gold_candidate_snapshot_digest(self) != self.snapshot_digest:
            raise ValueError("Gold candidate snapshot digest mismatch")
        return self


class OledGoldCandidateWriteArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_GOLD_CANDIDATE_WRITE_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    preflight_artifact_sha256: str
    preflight_artifact_digest: str
    preflight_artifact: OledGoldAdmissionPreflightArtifact
    published_snapshot: OledGoldCandidateSnapshot
    status: OledGoldCandidateWriteStatus
    publication_receipt_filename: Literal[
        "gold_candidate_write.json"
    ] = GOLD_CANDIDATE_WRITE_FILENAME
    published_snapshot_filename: Literal[
        "gold_candidate_snapshot.json"
    ] = GOLD_CANDIDATE_SNAPSHOT_FILENAME
    published_snapshot_file_sha256: str
    published_candidate_count: Annotated[StrictInt, Field(ge=1)]
    published_candidate_ids: list[str] = Field(min_length=1)
    published_candidate_digests: list[str] = Field(min_length=1)
    snapshot_id: str
    snapshot_digest: str
    write_artifact_digest: str
    publication_receipt_created: StrictBool = True
    immutable_candidate_snapshot_published: StrictBool = True
    preflight_bytes_rechecked_before_publication: StrictBool = True
    exact_candidate_roster_replayed: StrictBool = True
    atomic_noreplace_directory_publication: StrictBool = True
    fsync_files_and_directories_completed: StrictBool = True
    published_directory_inode_bound: StrictBool = True
    published_payloads_revalidated: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_records_created: StrictBool = False
    gold_head_activated: StrictBool = False
    curated_dataset_written: StrictBool = False
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
        if value != OLED_GOLD_CANDIDATE_WRITE_VERSION:
            raise ValueError("unexpected Gold candidate write version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="snapshot_id")

    @field_validator(
        "preflight_artifact_sha256",
        "preflight_artifact_digest",
        "published_snapshot_file_sha256",
        "snapshot_digest",
        "write_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("published_candidate_ids", "published_candidate_digests")
    @classmethod
    def validate_sorted_unique_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(self) -> OledGoldCandidateWriteArtifact:
        preflight = self.preflight_artifact
        snapshot = self.published_snapshot
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            preflight.generated_at
        ) or snapshot.generated_at != self.generated_at:
            raise ValueError("Gold candidate publication timestamp mismatch")
        if (
            preflight.status
            not in {
                OledGoldAdmissionPreflightStatus.CANDIDATES_READY,
                OledGoldAdmissionPreflightStatus.PARTIAL_WITH_BLOCKED_EVIDENCE,
            }
            or not preflight.candidates
        ):
            raise ValueError("Gold candidate publication requires eligible candidates")
        if (
            oled_gold_admission_preflight_artifact_digest(preflight)
            != preflight.preflight_artifact_digest
            or self.preflight_artifact_digest
            != preflight.preflight_artifact_digest
            or self.run_id != preflight.run_id
            or self.paper_id != preflight.paper_id
        ):
            raise ValueError("Gold candidate write preflight binding mismatch")
        expected_snapshot = build_oled_gold_candidate_snapshot(
            preflight=preflight,
            generated_at=self.generated_at,
        )
        if snapshot.model_dump(mode="json") != expected_snapshot.model_dump(
            mode="json"
        ):
            raise ValueError("published Gold candidate snapshot changed")
        snapshot_bytes = gold_candidate_snapshot_publication_bytes(snapshot)
        expected_fields = {
            "status": OledGoldCandidateWriteStatus.PUBLISHED,
            "published_snapshot_file_sha256": (
                f"sha256:{hashlib.sha256(snapshot_bytes).hexdigest()}"
            ),
            "published_candidate_count": len(snapshot.candidates),
            "published_candidate_ids": sorted(
                candidate.candidate_id for candidate in snapshot.candidates
            ),
            "published_candidate_digests": sorted(
                candidate.candidate_digest for candidate in snapshot.candidates
            ),
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_digest": snapshot.snapshot_digest,
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Gold candidate write {field_name} mismatch")
        fixed_true = (
            "publication_receipt_created",
            "immutable_candidate_snapshot_published",
            "preflight_bytes_rechecked_before_publication",
            "exact_candidate_roster_replayed",
            "atomic_noreplace_directory_publication",
            "fsync_files_and_directories_completed",
            "published_directory_inode_bound",
            "published_payloads_revalidated",
            "categorical_confidence_only",
        )
        fixed_false = (
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "gold_records_created",
            "gold_head_activated",
            "curated_dataset_written",
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
            raise ValueError("Gold candidate writer crossed its boundary")
        if oled_gold_candidate_write_artifact_digest(self) != (
            self.write_artifact_digest
        ):
            raise ValueError("Gold candidate write artifact digest mismatch")
        return self


def build_oled_gold_candidate_snapshot(
    *,
    preflight: OledGoldAdmissionPreflightArtifact,
    generated_at: str,
) -> OledGoldCandidateSnapshot:
    candidates = [
        OledGoldAdmissionCandidate.model_validate(
            candidate.model_dump(mode="json")
        )
        for candidate in preflight.candidates
    ]
    if not candidates:
        raise ValueError("Gold candidate publication cannot publish an empty roster")
    payload: dict[str, Any] = {
        "snapshot_version": OLED_GOLD_CANDIDATE_SNAPSHOT_VERSION,
        "snapshot_id": _id_hash(
            "oled-gold-candidate-snapshot",
            {
                "preflight_digest": preflight.preflight_artifact_digest,
                "candidate_ids": [candidate.candidate_id for candidate in candidates],
            },
        ),
        "generated_at": generated_at,
        "source_preflight_digest": preflight.preflight_artifact_digest,
        "entry_count": len(candidates),
        "candidates": sorted(candidates, key=lambda item: item.candidate_id),
        "snapshot_digest": "sha256:" + "0" * 64,
        "immutable_candidate_only": True,
        "categorical_confidence_only": True,
        "numeric_confidence_score_assigned": False,
        "legacy_numeric_confidence_record_constructed": False,
        "gold_records_created": False,
        "curated_dataset_written": False,
        "training_eligible": False,
    }
    provisional = OledGoldCandidateSnapshot.model_construct(**payload)
    payload["snapshot_digest"] = oled_gold_candidate_snapshot_digest(provisional)
    return OledGoldCandidateSnapshot.model_validate(payload)


def build_oled_gold_candidate_write_artifact(
    *,
    preflight: OledGoldAdmissionPreflightArtifact,
    preflight_artifact_sha256: str,
    generated_at: str,
) -> OledGoldCandidateWriteArtifact:
    validated = OledGoldAdmissionPreflightArtifact.model_validate(
        preflight.model_dump(mode="json")
    )
    snapshot = build_oled_gold_candidate_snapshot(
        preflight=validated,
        generated_at=generated_at,
    )
    snapshot_bytes = gold_candidate_snapshot_publication_bytes(snapshot)
    payload: dict[str, Any] = {
        "artifact_version": OLED_GOLD_CANDIDATE_WRITE_VERSION,
        "run_id": validated.run_id,
        "paper_id": validated.paper_id,
        "generated_at": generated_at,
        "preflight_artifact_sha256": _normalize_sha256(
            preflight_artifact_sha256,
            field_name="preflight_artifact_sha256",
        ),
        "preflight_artifact_digest": validated.preflight_artifact_digest,
        "preflight_artifact": validated,
        "published_snapshot": snapshot,
        "status": OledGoldCandidateWriteStatus.PUBLISHED,
        "published_snapshot_file_sha256": (
            f"sha256:{hashlib.sha256(snapshot_bytes).hexdigest()}"
        ),
        "published_candidate_count": len(snapshot.candidates),
        "published_candidate_ids": sorted(
            candidate.candidate_id for candidate in snapshot.candidates
        ),
        "published_candidate_digests": sorted(
            candidate.candidate_digest for candidate in snapshot.candidates
        ),
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_digest": snapshot.snapshot_digest,
        "write_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledGoldCandidateWriteArtifact.model_construct(**payload)
    payload["write_artifact_digest"] = (
        oled_gold_candidate_write_artifact_digest(provisional)
    )
    return OledGoldCandidateWriteArtifact.model_validate(payload)


def gold_candidate_snapshot_publication_bytes(
    snapshot: OledGoldCandidateSnapshot,
) -> bytes:
    return (
        json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def oled_gold_candidate_snapshot_digest(
    snapshot: OledGoldCandidateSnapshot,
) -> str:
    payload = snapshot.model_dump(mode="json")
    payload.pop("snapshot_digest", None)
    return _stable_hash(payload)


def oled_gold_candidate_write_artifact_digest(
    artifact: OledGoldCandidateWriteArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("write_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "GOLD_CANDIDATE_SNAPSHOT_FILENAME",
    "GOLD_CANDIDATE_WRITE_FILENAME",
    "OLED_GOLD_CANDIDATE_SNAPSHOT_VERSION",
    "OLED_GOLD_CANDIDATE_WRITE_VERSION",
    "OledGoldCandidateSnapshot",
    "OledGoldCandidateWriteArtifact",
    "OledGoldCandidateWriteStatus",
    "build_oled_gold_candidate_snapshot",
    "build_oled_gold_candidate_write_artifact",
    "gold_candidate_snapshot_publication_bytes",
    "oled_gold_candidate_snapshot_digest",
    "oled_gold_candidate_write_artifact_digest",
]
