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

from ai4s_agent.domains.oled_gold_successor_preflight import (
    OledCategoricalGoldSnapshot,
    categorical_gold_snapshot_publication_bytes,
    oled_categorical_gold_snapshot_digest,
)
from ai4s_agent.domains.oled_gold_successor_writer import (
    OledGoldSuccessorWriteArtifact,
    gold_successor_write_receipt_publication_bytes,
    oled_gold_successor_write_artifact_digest,
)
from ai4s_agent.domains.oled_reviewed_evidence_staging_preflight import (
    _normalize_sha256,
    _parse_timestamp,
    _stable_hash,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_GOLD_SUCCESSOR_POSTWRITE_VERIFIER_VERSION = (
    "oled_gold_successor_postwrite_verifier.v1"
)


class OledGoldSuccessorPostwriteVerificationStatus(str, Enum):
    VERIFIED = "categorical_gold_successor_publication_verified"


class OledGoldSuccessorPostwriteVerificationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_GOLD_SUCCESSOR_POSTWRITE_VERIFIER_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    write_artifact_sha256: str
    write_artifact_digest: str
    published_snapshot_sha256: str
    published_snapshot_digest: str
    write_artifact: OledGoldSuccessorWriteArtifact
    published_snapshot: OledCategoricalGoldSnapshot
    status: OledGoldSuccessorPostwriteVerificationStatus
    gold_registry_id: str
    prior_snapshot_id: str
    verified_successor_snapshot_id: str
    verified_activated_snapshot_id: str
    prior_generation: Annotated[StrictInt, Field(ge=0)]
    verified_generation: Annotated[StrictInt, Field(ge=1)]
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_added_entry_count: Annotated[StrictInt, Field(ge=1)]
    published_entry_count: Annotated[StrictInt, Field(ge=1)]
    verified_added_gold_entry_ids: list[str] = Field(min_length=1)
    verified_added_entry_digests: list[str] = Field(min_length=1)
    verification_artifact_digest: str
    exact_write_artifact_bytes_bound: StrictBool = True
    exact_published_snapshot_bytes_bound: StrictBool = True
    write_artifact_integrity_replayed: StrictBool = True
    published_snapshot_file_sha_replayed: StrictBool = True
    compare_and_swap_parent_replayed: StrictBool = True
    append_only_prior_preservation_replayed: StrictBool = True
    exact_planned_additions_replayed: StrictBool = True
    deterministic_entry_identity_replayed: StrictBool = True
    snapshot_internal_uniqueness_replayed: StrictBool = True
    snapshot_ordering_replayed: StrictBool = True
    snapshot_lineage_replayed: StrictBool = True
    publication_counts_replayed: StrictBool = True
    activation_binding_replayed: StrictBool = True
    published_categorical_gold_snapshot_verified: StrictBool = True
    snapshot_activation_receipt_verified: StrictBool = True
    eligible_for_explicit_dataset_admission_input: StrictBool = True
    categorical_confidence_only: StrictBool = True
    numeric_confidence_score_assigned: StrictBool = False
    legacy_numeric_confidence_record_constructed: StrictBool = False
    gold_snapshot_written: StrictBool = False
    categorical_gold_snapshot_activated: StrictBool = False
    gold_head_activated: StrictBool = False
    mutable_gold_head_pointer_written: StrictBool = False
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
        if value != OLED_GOLD_SUCCESSOR_POSTWRITE_VERIFIER_VERSION:
            raise ValueError("unexpected Gold successor verifier version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "gold_registry_id",
        "prior_snapshot_id",
        "verified_successor_snapshot_id",
        "verified_activated_snapshot_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

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

    @field_validator(
        "verified_added_gold_entry_ids",
        "verified_added_entry_digests",
    )
    @classmethod
    def validate_sorted_unique(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact(
        self,
    ) -> OledGoldSuccessorPostwriteVerificationArtifact:
        receipt = self.write_artifact
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            receipt.generated_at
        ):
            raise ValueError("Gold successor verification predates PR-AF")
        if self.run_id != receipt.run_id or self.paper_id != receipt.paper_id:
            raise ValueError("Gold successor verification identity mismatch")
        if (
            self.write_artifact_digest != receipt.write_artifact_digest
            or oled_gold_successor_write_artifact_digest(receipt)
            != self.write_artifact_digest
        ):
            raise ValueError("Gold successor receipt digest mismatch")
        if (
            self.published_snapshot_digest
            != self.published_snapshot.snapshot_digest
            or oled_categorical_gold_snapshot_digest(self.published_snapshot)
            != self.published_snapshot_digest
        ):
            raise ValueError("Gold successor snapshot digest mismatch")
        replay = independently_replay_gold_successor_postwrite(
            receipt,
            self.published_snapshot,
            write_artifact_sha256=self.write_artifact_sha256,
            published_snapshot_sha256=self.published_snapshot_sha256,
        )
        expected_fields = {
            "status": OledGoldSuccessorPostwriteVerificationStatus.VERIFIED,
            "gold_registry_id": self.published_snapshot.gold_registry_id,
            "prior_snapshot_id": receipt.prior_gold_snapshot.snapshot_id,
            "verified_successor_snapshot_id": self.published_snapshot.snapshot_id,
            "verified_activated_snapshot_id": self.published_snapshot.snapshot_id,
            "prior_generation": receipt.prior_gold_snapshot.generation,
            "verified_generation": self.published_snapshot.generation,
            "prior_entry_count": len(receipt.prior_gold_snapshot.entries),
            "verified_added_entry_count": len(replay["added_gold_entry_ids"]),
            "published_entry_count": len(self.published_snapshot.entries),
            "verified_added_gold_entry_ids": replay["added_gold_entry_ids"],
            "verified_added_entry_digests": replay["added_entry_digests"],
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Gold successor verification {field_name} mismatch")
        fixed_true = (
            "exact_write_artifact_bytes_bound",
            "exact_published_snapshot_bytes_bound",
            "write_artifact_integrity_replayed",
            "published_snapshot_file_sha_replayed",
            "compare_and_swap_parent_replayed",
            "append_only_prior_preservation_replayed",
            "exact_planned_additions_replayed",
            "deterministic_entry_identity_replayed",
            "snapshot_internal_uniqueness_replayed",
            "snapshot_ordering_replayed",
            "snapshot_lineage_replayed",
            "publication_counts_replayed",
            "activation_binding_replayed",
            "published_categorical_gold_snapshot_verified",
            "snapshot_activation_receipt_verified",
            "eligible_for_explicit_dataset_admission_input",
            "categorical_confidence_only",
        )
        fixed_false = (
            "numeric_confidence_score_assigned",
            "legacy_numeric_confidence_record_constructed",
            "gold_snapshot_written",
            "categorical_gold_snapshot_activated",
            "gold_head_activated",
            "mutable_gold_head_pointer_written",
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
            raise ValueError("Gold successor verifier crossed its boundary")
        if oled_gold_successor_postwrite_verification_artifact_digest(self) != (
            self.verification_artifact_digest
        ):
            raise ValueError("Gold successor verification digest mismatch")
        return self


def build_oled_gold_successor_postwrite_verification_artifact(
    *,
    write_artifact: OledGoldSuccessorWriteArtifact,
    write_artifact_sha256: str,
    published_snapshot: OledCategoricalGoldSnapshot,
    published_snapshot_sha256: str,
    generated_at: str,
) -> OledGoldSuccessorPostwriteVerificationArtifact:
    receipt = OledGoldSuccessorWriteArtifact.model_validate(
        write_artifact.model_dump(mode="json")
    )
    snapshot = OledCategoricalGoldSnapshot.model_validate(
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
    replay = independently_replay_gold_successor_postwrite(
        receipt,
        snapshot,
        write_artifact_sha256=receipt_sha,
        published_snapshot_sha256=snapshot_sha,
    )
    payload: dict[str, Any] = {
        "artifact_version": OLED_GOLD_SUCCESSOR_POSTWRITE_VERIFIER_VERSION,
        "run_id": receipt.run_id,
        "paper_id": receipt.paper_id,
        "generated_at": generated_at,
        "write_artifact_sha256": receipt_sha,
        "write_artifact_digest": receipt.write_artifact_digest,
        "published_snapshot_sha256": snapshot_sha,
        "published_snapshot_digest": snapshot.snapshot_digest,
        "write_artifact": receipt,
        "published_snapshot": snapshot,
        "status": OledGoldSuccessorPostwriteVerificationStatus.VERIFIED,
        "gold_registry_id": snapshot.gold_registry_id,
        "prior_snapshot_id": receipt.prior_gold_snapshot.snapshot_id,
        "verified_successor_snapshot_id": snapshot.snapshot_id,
        "verified_activated_snapshot_id": snapshot.snapshot_id,
        "prior_generation": receipt.prior_gold_snapshot.generation,
        "verified_generation": snapshot.generation,
        "prior_entry_count": len(receipt.prior_gold_snapshot.entries),
        "verified_added_entry_count": len(replay["added_gold_entry_ids"]),
        "published_entry_count": len(snapshot.entries),
        "verified_added_gold_entry_ids": replay["added_gold_entry_ids"],
        "verified_added_entry_digests": replay["added_entry_digests"],
        "verification_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = (
        OledGoldSuccessorPostwriteVerificationArtifact.model_construct(**payload)
    )
    payload["verification_artifact_digest"] = (
        oled_gold_successor_postwrite_verification_artifact_digest(provisional)
    )
    return OledGoldSuccessorPostwriteVerificationArtifact.model_validate(payload)


def independently_replay_gold_successor_postwrite(
    receipt: OledGoldSuccessorWriteArtifact,
    published: OledCategoricalGoldSnapshot,
    *,
    write_artifact_sha256: str,
    published_snapshot_sha256: str,
) -> dict[str, list[str]]:
    if write_artifact_sha256 != _sha256_bytes(
        gold_successor_write_receipt_publication_bytes(receipt)
    ):
        raise ValueError("published PR-AF receipt file SHA-256 mismatch")
    expected_snapshot_sha = _sha256_bytes(
        categorical_gold_snapshot_publication_bytes(published)
    )
    if (
        published_snapshot_sha256 != expected_snapshot_sha
        or published_snapshot_sha256 != receipt.published_snapshot_file_sha256
    ):
        raise ValueError("published categorical Gold snapshot file SHA-256 mismatch")
    preflight = receipt.preflight_artifact
    expected_successor = preflight.expected_successor_snapshot
    if expected_successor is None:
        raise ValueError("PR-AE has no expected Gold successor")
    published_payload = published.model_dump(mode="json")
    if (
        published_payload
        != receipt.published_successor_snapshot.model_dump(mode="json")
        or published_payload != expected_successor.model_dump(mode="json")
    ):
        raise ValueError("published categorical Gold snapshot differs from PR-AE/PR-AF")

    prior = receipt.prior_gold_snapshot
    if (
        prior.model_dump(mode="json")
        != preflight.current_gold_snapshot.model_dump(mode="json")
        or prior.snapshot_digest != receipt.prior_gold_snapshot_digest
        or prior.snapshot_digest != preflight.current_gold_snapshot_digest
        or receipt.prior_gold_snapshot_sha256
        != preflight.current_gold_snapshot_sha256
    ):
        raise ValueError("published Gold compare-and-swap parent mismatch")
    prior_by_id = {entry.gold_entry_id: entry for entry in prior.entries}
    published_by_id = {entry.gold_entry_id: entry for entry in published.entries}
    planned_by_id = {
        item.gold_entry.gold_entry_id: item.gold_entry
        for item in preflight.planned_additions
    }
    for entry_id, prior_entry in prior_by_id.items():
        current = published_by_id.get(entry_id)
        if current is None or current.model_dump(mode="json") != (
            prior_entry.model_dump(mode="json")
        ):
            raise ValueError("published Gold rewrites or removes a prior entry")
    if set(published_by_id) != set(prior_by_id) | set(planned_by_id):
        raise ValueError("published Gold addition coverage mismatch")
    for entry_id, planned_entry in planned_by_id.items():
        if published_by_id[entry_id].model_dump(mode="json") != (
            planned_entry.model_dump(mode="json")
        ):
            raise ValueError("published Gold entry differs from PR-AE plan")
    published_order = [entry.gold_entry_id for entry in published.entries]
    if published_order != sorted(published_order):
        raise ValueError("published Gold entry order mismatch")
    added_ids = sorted(set(published_by_id) - set(prior_by_id))
    added_digests = sorted(
        published_by_id[entry_id].entry_digest for entry_id in added_ids
    )
    if (
        added_ids != sorted(planned_by_id)
        or added_ids != receipt.added_gold_entry_ids
    ):
        raise ValueError("published Gold added entry IDs mismatch")
    if added_digests != receipt.added_entry_digests:
        raise ValueError("published Gold added entry digests mismatch")
    if (
        len(prior_by_id) != receipt.prior_entry_count
        or len(added_ids) != receipt.added_entry_count
        or len(published_by_id) != receipt.published_entry_count
        or len(added_ids) != preflight.planned_addition_count
        or len(published_by_id) != preflight.expected_entry_count
    ):
        raise ValueError("published Gold counts mismatch")
    if (
        published.gold_registry_id != prior.gold_registry_id
        or published.generation != prior.generation + 1
        or published.generation != receipt.published_generation
        or published.parent_snapshot_digest != prior.snapshot_digest
        or published.source_verification_digest
        != preflight.verification_artifact_digest
        or published.snapshot_id != receipt.successor_snapshot_id
        or published.snapshot_digest != receipt.successor_snapshot_digest
        or published.snapshot_digest
        != preflight.expected_successor_snapshot_digest
    ):
        raise ValueError("published Gold snapshot lineage mismatch")
    if (
        receipt.activated_snapshot_id != published.snapshot_id
        or receipt.activated_snapshot_digest != published.snapshot_digest
    ):
        raise ValueError("published Gold activation binding mismatch")
    if _parse_timestamp(published.generated_at) < _parse_timestamp(
        prior.generated_at
    ) or _parse_timestamp(receipt.generated_at) < _parse_timestamp(
        published.generated_at
    ):
        raise ValueError("published Gold timestamp lineage mismatch")
    return {
        "added_gold_entry_ids": added_ids,
        "added_entry_digests": added_digests,
    }


def oled_gold_successor_postwrite_verification_artifact_digest(
    artifact: OledGoldSuccessorPostwriteVerificationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("verification_artifact_digest", None)
    return _stable_hash(payload)


def _sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


__all__ = [
    "OLED_GOLD_SUCCESSOR_POSTWRITE_VERIFIER_VERSION",
    "OledGoldSuccessorPostwriteVerificationArtifact",
    "OledGoldSuccessorPostwriteVerificationStatus",
    "build_oled_gold_successor_postwrite_verification_artifact",
    "independently_replay_gold_successor_postwrite",
    "oled_gold_successor_postwrite_verification_artifact_digest",
]
