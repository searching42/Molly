from __future__ import annotations

import hashlib
import json
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

from ai4s_agent.domains.oled_gold_candidate_writer import (
    OledGoldCandidateSnapshot,
    OledGoldCandidateWriteArtifact,
    build_oled_gold_candidate_snapshot,
    gold_candidate_snapshot_publication_bytes,
    oled_gold_candidate_snapshot_digest,
    oled_gold_candidate_write_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_GOLD_CANDIDATE_POSTWRITE_VERIFIER_VERSION = (
    "oled_gold_candidate_postwrite_verifier.v1"
)


class OledGoldCandidatePostwriteVerificationStatus(str, Enum):
    VERIFIED = "gold_candidate_publication_verified"


class OledGoldCandidatePostwriteVerificationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_GOLD_CANDIDATE_POSTWRITE_VERIFIER_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    write_artifact_sha256: str
    write_artifact_digest: str
    published_snapshot_sha256: str
    published_snapshot_digest: str
    write_artifact: OledGoldCandidateWriteArtifact
    published_snapshot: OledGoldCandidateSnapshot
    status: OledGoldCandidatePostwriteVerificationStatus
    snapshot_id: str
    verified_candidate_count: Annotated[StrictInt, Field(ge=1)]
    verified_candidate_ids: list[str] = Field(min_length=1)
    verified_candidate_digests: list[str] = Field(min_length=1)
    verification_artifact_digest: str
    exact_write_artifact_bytes_bound: StrictBool = True
    exact_published_snapshot_bytes_bound: StrictBool = True
    write_artifact_integrity_replayed: StrictBool = True
    published_snapshot_file_sha_replayed: StrictBool = True
    exact_preflight_candidate_roster_replayed: StrictBool = True
    candidate_payloads_and_digests_replayed: StrictBool = True
    snapshot_ordering_and_counts_replayed: StrictBool = True
    snapshot_lineage_replayed: StrictBool = True
    published_gold_candidate_snapshot_verified: StrictBool = True
    eligible_for_explicit_gold_publication_input: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_records_created: StrictBool = False
    gold_head_activated: StrictBool = False
    curated_dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    reviewed_evidence_mutated: StrictBool = False
    registry_written: StrictBool = False
    standalone_input_bytes_revalidation_supported: StrictBool = False
    source_pdf_read: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_GOLD_CANDIDATE_POSTWRITE_VERIFIER_VERSION:
            raise ValueError("unexpected Gold candidate verifier version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="snapshot_id")

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator(
        "write_artifact_sha256",
        "write_artifact_digest",
        "published_snapshot_sha256",
        "published_snapshot_digest",
        "verification_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("verified_candidate_ids", "verified_candidate_digests")
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(
        self,
    ) -> OledGoldCandidatePostwriteVerificationArtifact:
        receipt = self.write_artifact
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            receipt.generated_at
        ):
            raise ValueError("Gold candidate verification predates PR-AC")
        if self.run_id != receipt.run_id or self.paper_id != receipt.paper_id:
            raise ValueError("Gold candidate verification identity mismatch")
        if (
            self.write_artifact_digest != receipt.write_artifact_digest
            or oled_gold_candidate_write_artifact_digest(receipt)
            != self.write_artifact_digest
        ):
            raise ValueError("Gold candidate receipt digest mismatch")
        if (
            self.published_snapshot_digest
            != self.published_snapshot.snapshot_digest
            or oled_gold_candidate_snapshot_digest(self.published_snapshot)
            != self.published_snapshot_digest
        ):
            raise ValueError("Gold candidate snapshot digest mismatch")
        replay = independently_replay_gold_candidate_postwrite(
            receipt,
            self.published_snapshot,
            write_artifact_sha256=self.write_artifact_sha256,
            published_snapshot_sha256=self.published_snapshot_sha256,
        )
        expected = {
            "status": OledGoldCandidatePostwriteVerificationStatus.VERIFIED,
            "snapshot_id": self.published_snapshot.snapshot_id,
            "verified_candidate_count": len(replay["candidate_ids"]),
            "verified_candidate_ids": replay["candidate_ids"],
            "verified_candidate_digests": replay["candidate_digests"],
        }
        for field_name, value in expected.items():
            if getattr(self, field_name) != value:
                raise ValueError(f"Gold candidate verification {field_name} mismatch")
        fixed_true = (
            "exact_write_artifact_bytes_bound",
            "exact_published_snapshot_bytes_bound",
            "write_artifact_integrity_replayed",
            "published_snapshot_file_sha_replayed",
            "exact_preflight_candidate_roster_replayed",
            "candidate_payloads_and_digests_replayed",
            "snapshot_ordering_and_counts_replayed",
            "snapshot_lineage_replayed",
            "published_gold_candidate_snapshot_verified",
            "eligible_for_explicit_gold_publication_input",
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
            raise ValueError("Gold candidate verifier crossed its boundary")
        if oled_gold_candidate_postwrite_verification_artifact_digest(self) != (
            self.verification_artifact_digest
        ):
            raise ValueError("Gold candidate verification artifact digest mismatch")
        return self


def build_oled_gold_candidate_postwrite_verification_artifact(
    *,
    write_artifact: OledGoldCandidateWriteArtifact,
    write_artifact_sha256: str,
    published_snapshot: OledGoldCandidateSnapshot,
    published_snapshot_sha256: str,
    generated_at: str,
) -> OledGoldCandidatePostwriteVerificationArtifact:
    receipt = OledGoldCandidateWriteArtifact.model_validate(
        write_artifact.model_dump(mode="json")
    )
    snapshot = OledGoldCandidateSnapshot.model_validate(
        published_snapshot.model_dump(mode="json")
    )
    receipt_sha = _normalize_sha256(
        write_artifact_sha256,
        field_name="write_artifact_sha256",
    )
    snapshot_sha = _normalize_sha256(
        published_snapshot_sha256,
        field_name="published_snapshot_sha256",
    )
    replay = independently_replay_gold_candidate_postwrite(
        receipt,
        snapshot,
        write_artifact_sha256=receipt_sha,
        published_snapshot_sha256=snapshot_sha,
    )
    payload: dict[str, Any] = {
        "artifact_version": OLED_GOLD_CANDIDATE_POSTWRITE_VERIFIER_VERSION,
        "run_id": receipt.run_id,
        "paper_id": receipt.paper_id,
        "generated_at": generated_at,
        "write_artifact_sha256": receipt_sha,
        "write_artifact_digest": receipt.write_artifact_digest,
        "published_snapshot_sha256": snapshot_sha,
        "published_snapshot_digest": snapshot.snapshot_digest,
        "write_artifact": receipt,
        "published_snapshot": snapshot,
        "status": OledGoldCandidatePostwriteVerificationStatus.VERIFIED,
        "snapshot_id": snapshot.snapshot_id,
        "verified_candidate_count": len(replay["candidate_ids"]),
        "verified_candidate_ids": replay["candidate_ids"],
        "verified_candidate_digests": replay["candidate_digests"],
        "verification_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledGoldCandidatePostwriteVerificationArtifact.model_construct(
        **payload
    )
    payload["verification_artifact_digest"] = (
        oled_gold_candidate_postwrite_verification_artifact_digest(provisional)
    )
    return OledGoldCandidatePostwriteVerificationArtifact.model_validate(payload)


def independently_replay_gold_candidate_postwrite(
    receipt: OledGoldCandidateWriteArtifact,
    published: OledGoldCandidateSnapshot,
    *,
    write_artifact_sha256: str,
    published_snapshot_sha256: str,
) -> dict[str, list[str]]:
    if write_artifact_sha256 != _sha256_bytes(
        gold_candidate_write_receipt_publication_bytes(receipt)
    ):
        raise ValueError("published PR-AC receipt file SHA-256 mismatch")
    expected_snapshot_sha = _sha256_bytes(
        gold_candidate_snapshot_publication_bytes(published)
    )
    if (
        published_snapshot_sha256 != expected_snapshot_sha
        or published_snapshot_sha256 != receipt.published_snapshot_file_sha256
    ):
        raise ValueError("published Gold candidate snapshot file SHA-256 mismatch")
    expected = build_oled_gold_candidate_snapshot(
        preflight=receipt.preflight_artifact,
        generated_at=receipt.generated_at,
    )
    if (
        published.model_dump(mode="json")
        != receipt.published_snapshot.model_dump(mode="json")
        or published.model_dump(mode="json") != expected.model_dump(mode="json")
    ):
        raise ValueError("published Gold candidate snapshot differs from PR-AB/PR-AC")
    candidate_ids = [candidate.candidate_id for candidate in published.candidates]
    candidate_digests = sorted(
        candidate.candidate_digest for candidate in published.candidates
    )
    expected_by_id = {
        candidate.candidate_id: candidate
        for candidate in receipt.preflight_artifact.candidates
    }
    published_by_id = {
        candidate.candidate_id: candidate for candidate in published.candidates
    }
    if (
        candidate_ids != sorted(candidate_ids)
        or set(published_by_id) != set(expected_by_id)
    ):
        raise ValueError("published Gold candidate roster mismatch")
    for candidate_id, candidate in expected_by_id.items():
        if published_by_id[candidate_id].model_dump(mode="json") != (
            candidate.model_dump(mode="json")
        ):
            raise ValueError("published Gold candidate payload mismatch")
    if (
        published.entry_count != len(candidate_ids)
        or receipt.published_candidate_count != len(candidate_ids)
        or receipt.published_candidate_ids != sorted(candidate_ids)
        or receipt.published_candidate_digests != candidate_digests
    ):
        raise ValueError("published Gold candidate counts or digests mismatch")
    if (
        published.source_preflight_digest
        != receipt.preflight_artifact.preflight_artifact_digest
        or published.snapshot_id != receipt.snapshot_id
        or published.snapshot_digest != receipt.snapshot_digest
        or published.generated_at != receipt.generated_at
    ):
        raise ValueError("published Gold candidate snapshot lineage mismatch")
    return {
        "candidate_ids": sorted(candidate_ids),
        "candidate_digests": candidate_digests,
    }


def gold_candidate_write_receipt_publication_bytes(
    receipt: OledGoldCandidateWriteArtifact,
) -> bytes:
    return (
        json.dumps(
            receipt.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def oled_gold_candidate_postwrite_verification_artifact_digest(
    artifact: OledGoldCandidatePostwriteVerificationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("verification_artifact_digest", None)
    return _stable_hash(payload)


__all__ = [
    "OLED_GOLD_CANDIDATE_POSTWRITE_VERIFIER_VERSION",
    "OledGoldCandidatePostwriteVerificationArtifact",
    "OledGoldCandidatePostwriteVerificationStatus",
    "build_oled_gold_candidate_postwrite_verification_artifact",
    "gold_candidate_write_receipt_publication_bytes",
    "independently_replay_gold_candidate_postwrite",
    "oled_gold_candidate_postwrite_verification_artifact_digest",
]
