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

from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistrySnapshot,
    oled_material_registry_snapshot_digest,
)
from ai4s_agent.domains.oled_material_registry_successor_writer import (
    OledMaterialRegistrySuccessorWriteArtifact,
    material_registry_snapshot_publication_bytes,
    oled_material_registry_successor_write_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_MATERIAL_REGISTRY_SUCCESSOR_POSTWRITE_VERIFIER_VERSION = (
    "oled_material_registry_successor_postwrite_verifier.v1"
)


class OledMaterialRegistrySuccessorPostwriteVerificationStatus(str, Enum):
    VERIFIED = "registry_successor_publication_verified"


class OledMaterialRegistrySuccessorPostwriteVerificationArtifact(BaseModel):
    """Read-only independent replay of one exact PR-Y publication."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = (
        OLED_MATERIAL_REGISTRY_SUCCESSOR_POSTWRITE_VERIFIER_VERSION
    )
    run_id: str
    paper_id: str
    generated_at: str
    write_artifact_sha256: str
    write_artifact_digest: str
    published_snapshot_sha256: str
    published_snapshot_digest: str
    write_artifact: OledMaterialRegistrySuccessorWriteArtifact
    published_snapshot: OledMaterialRegistrySnapshot
    status: OledMaterialRegistrySuccessorPostwriteVerificationStatus
    registry_id: str
    prior_registry_version: str
    verified_successor_registry_version: str
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    verified_added_entry_count: Annotated[StrictInt, Field(ge=1)]
    verified_added_entry_cell_count: Annotated[StrictInt, Field(ge=0)]
    published_entry_count: Annotated[StrictInt, Field(ge=1)]
    verified_added_material_ids: list[str] = Field(default_factory=list)
    verified_added_entry_digests: list[str] = Field(default_factory=list)
    verification_artifact_digest: str
    exact_write_artifact_bytes_bound: StrictBool = True
    exact_published_snapshot_bytes_bound: StrictBool = True
    write_artifact_integrity_replayed: StrictBool = True
    published_snapshot_file_sha_replayed: StrictBool = True
    append_only_prior_preservation_replayed: StrictBool = True
    exact_planned_additions_replayed: StrictBool = True
    entry_identity_and_chemistry_replayed: StrictBool = True
    snapshot_ordering_replayed: StrictBool = True
    snapshot_lineage_replayed: StrictBool = True
    publication_counts_replayed: StrictBool = True
    published_registry_snapshot_verified: StrictBool = True
    eligible_for_explicit_pr_n_input: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    registry_written: StrictBool = False
    registry_head_activated: StrictBool = False
    activation_receipt_created: StrictBool = False
    existing_registry_mutated: StrictBool = False
    observations_materialized: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    device_only_records_admitted: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_SUCCESSOR_POSTWRITE_VERIFIER_VERSION:
            raise ValueError("unexpected Registry post-write verifier version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "registry_id",
        "prior_registry_version",
        "verified_successor_registry_version",
    )
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
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
        "verified_added_material_ids",
        "verified_added_entry_digests",
    )
    @classmethod
    def validate_sorted_unique_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_integrity(
        self,
    ) -> OledMaterialRegistrySuccessorPostwriteVerificationArtifact:
        receipt = self.write_artifact
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            receipt.generated_at
        ):
            raise ValueError("Registry post-write verification predates PR-Y")
        if self.run_id != receipt.run_id or self.paper_id != receipt.paper_id:
            raise ValueError("Registry post-write source identity mismatch")
        if self.write_artifact_digest != receipt.write_artifact_digest or (
            oled_material_registry_successor_write_artifact_digest(receipt)
            != self.write_artifact_digest
        ):
            raise ValueError("Registry post-write receipt digest mismatch")
        if self.published_snapshot_digest != self.published_snapshot.snapshot_digest or (
            oled_material_registry_snapshot_digest(self.published_snapshot)
            != self.published_snapshot_digest
        ):
            raise ValueError("Registry post-write published snapshot digest mismatch")
        replay = _independently_replay_registry_postwrite(
            receipt,
            self.published_snapshot,
            write_artifact_sha256=self.write_artifact_sha256,
            published_snapshot_sha256=self.published_snapshot_sha256,
        )
        expected_fields = {
            "status": (
                OledMaterialRegistrySuccessorPostwriteVerificationStatus.VERIFIED
            ),
            "registry_id": self.published_snapshot.registry_id,
            "prior_registry_version": receipt.prior_registry_snapshot.registry_version,
            "verified_successor_registry_version": (
                self.published_snapshot.registry_version
            ),
            "prior_entry_count": len(receipt.prior_registry_snapshot.entries),
            "verified_added_entry_count": len(replay["added_material_ids"]),
            "verified_added_entry_cell_count": (
                receipt.preflight_artifact.planned_addition_cell_count
            ),
            "published_entry_count": len(self.published_snapshot.entries),
            "verified_added_material_ids": replay["added_material_ids"],
            "verified_added_entry_digests": replay["added_entry_digests"],
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry post-write {field_name} mismatch")
        fixed_true = (
            "exact_write_artifact_bytes_bound",
            "exact_published_snapshot_bytes_bound",
            "write_artifact_integrity_replayed",
            "published_snapshot_file_sha_replayed",
            "append_only_prior_preservation_replayed",
            "exact_planned_additions_replayed",
            "entry_identity_and_chemistry_replayed",
            "snapshot_ordering_replayed",
            "snapshot_lineage_replayed",
            "publication_counts_replayed",
            "published_registry_snapshot_verified",
            "eligible_for_explicit_pr_n_input",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "registry_written",
            "registry_head_activated",
            "activation_receipt_created",
            "existing_registry_mutated",
            "observations_materialized",
            "reviewed_evidence_staging",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "device_only_records_admitted",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("Registry post-write verifier crossed its boundary")
        if oled_material_registry_successor_postwrite_verification_artifact_digest(
            self
        ) != self.verification_artifact_digest:
            raise ValueError("Registry post-write verification digest mismatch")
        return self


def build_oled_material_registry_successor_postwrite_verification_artifact(
    *,
    write_artifact: OledMaterialRegistrySuccessorWriteArtifact,
    write_artifact_sha256: str,
    published_snapshot: OledMaterialRegistrySnapshot,
    published_snapshot_sha256: str,
    generated_at: str,
) -> OledMaterialRegistrySuccessorPostwriteVerificationArtifact:
    receipt = OledMaterialRegistrySuccessorWriteArtifact.model_validate(
        write_artifact.model_dump(mode="json")
    )
    snapshot = OledMaterialRegistrySnapshot.model_validate(
        published_snapshot.model_dump(mode="json")
    )
    normalized_write_sha = _normalize_sha256(
        write_artifact_sha256,
        field_name="write_artifact_sha256",
    )
    normalized_snapshot_sha = _normalize_sha256(
        published_snapshot_sha256,
        field_name="published_snapshot_sha256",
    )
    replay = _independently_replay_registry_postwrite(
        receipt,
        snapshot,
        write_artifact_sha256=normalized_write_sha,
        published_snapshot_sha256=normalized_snapshot_sha,
    )
    payload: dict[str, Any] = {
        "artifact_version": (
            OLED_MATERIAL_REGISTRY_SUCCESSOR_POSTWRITE_VERIFIER_VERSION
        ),
        "run_id": receipt.run_id,
        "paper_id": receipt.paper_id,
        "generated_at": generated_at,
        "write_artifact_sha256": normalized_write_sha,
        "write_artifact_digest": receipt.write_artifact_digest,
        "published_snapshot_sha256": normalized_snapshot_sha,
        "published_snapshot_digest": snapshot.snapshot_digest,
        "write_artifact": receipt,
        "published_snapshot": snapshot,
        "status": OledMaterialRegistrySuccessorPostwriteVerificationStatus.VERIFIED,
        "registry_id": snapshot.registry_id,
        "prior_registry_version": receipt.prior_registry_snapshot.registry_version,
        "verified_successor_registry_version": snapshot.registry_version,
        "prior_entry_count": len(receipt.prior_registry_snapshot.entries),
        "verified_added_entry_count": len(replay["added_material_ids"]),
        "verified_added_entry_cell_count": (
            receipt.preflight_artifact.planned_addition_cell_count
        ),
        "published_entry_count": len(snapshot.entries),
        "verified_added_material_ids": replay["added_material_ids"],
        "verified_added_entry_digests": replay["added_entry_digests"],
        "verification_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = (
        OledMaterialRegistrySuccessorPostwriteVerificationArtifact.model_construct(
            **payload
        )
    )
    payload["verification_artifact_digest"] = (
        oled_material_registry_successor_postwrite_verification_artifact_digest(
            provisional
        )
    )
    return OledMaterialRegistrySuccessorPostwriteVerificationArtifact.model_validate(
        payload
    )


def _independently_replay_registry_postwrite(
    receipt: OledMaterialRegistrySuccessorWriteArtifact,
    published: OledMaterialRegistrySnapshot,
    *,
    write_artifact_sha256: str,
    published_snapshot_sha256: str,
) -> dict[str, Any]:
    expected_receipt_sha = _sha256_bytes(
        material_registry_write_receipt_publication_bytes(receipt)
    )
    if write_artifact_sha256 != expected_receipt_sha:
        raise ValueError("published PR-Y receipt file SHA-256 mismatch")
    expected_snapshot_sha = _sha256_bytes(
        material_registry_snapshot_publication_bytes(published)
    )
    if published_snapshot_sha256 != expected_snapshot_sha or (
        published_snapshot_sha256 != receipt.published_snapshot_file_sha256
    ):
        raise ValueError("published Registry snapshot file SHA-256 mismatch")
    if published.model_dump(mode="json") != (
        receipt.published_successor_snapshot.model_dump(mode="json")
    ) or published.model_dump(mode="json") != (
        receipt.preflight_artifact.expected_successor_snapshot.model_dump(mode="json")
    ):
        raise ValueError("published Registry snapshot differs from PR-X/PR-Y")

    prior = receipt.prior_registry_snapshot
    prior_by_id = {entry.material_id: entry for entry in prior.entries}
    published_by_id = {entry.material_id: entry for entry in published.entries}
    planned_by_id = {
        item.registry_entry.material_id: item.registry_entry
        for item in receipt.preflight_artifact.planned_additions
    }
    for material_id, prior_entry in prior_by_id.items():
        current = published_by_id.get(material_id)
        if current is None or current.model_dump(mode="json") != (
            prior_entry.model_dump(mode="json")
        ):
            raise ValueError("published Registry rewrites or removes a prior entry")
    if set(published_by_id) != set(prior_by_id) | set(planned_by_id):
        raise ValueError("published Registry addition coverage mismatch")
    for material_id, planned_entry in planned_by_id.items():
        if published_by_id[material_id].model_dump(mode="json") != (
            planned_entry.model_dump(mode="json")
        ):
            raise ValueError("published Registry entry differs from PR-X plan")
    published_order = [entry.material_id for entry in published.entries]
    if published_order != sorted(published_order):
        raise ValueError("published Registry material-ID order mismatch")
    added_ids = sorted(set(published_by_id) - set(prior_by_id))
    added_digests = sorted(
        published_by_id[material_id].entry_digest for material_id in added_ids
    )
    if added_ids != sorted(planned_by_id) or added_ids != receipt.added_material_ids:
        raise ValueError("published Registry added material IDs mismatch")
    if added_digests != receipt.added_entry_digests:
        raise ValueError("published Registry added entry digests mismatch")
    if (
        len(prior_by_id) != receipt.prior_entry_count
        or len(added_ids) != receipt.added_entry_count
        or len(published_by_id) != receipt.published_entry_count
        or len(added_ids) != receipt.preflight_artifact.planned_addition_count
    ):
        raise ValueError("published Registry counts mismatch")
    if receipt.added_entry_cell_count != (
        receipt.preflight_artifact.planned_addition_cell_count
    ):
        raise ValueError("published Registry dependent-cell count mismatch")
    if (
        published.registry_id != prior.registry_id
        or published.registry_version != receipt.successor_registry_version
        or published.registry_version
        != receipt.preflight_artifact.successor_registry_version
        or published.snapshot_digest != receipt.successor_snapshot_digest
        or published.snapshot_digest
        != receipt.preflight_artifact.expected_successor_snapshot_digest
        or published.generated_at
        != receipt.preflight_artifact.expected_successor_snapshot.generated_at
    ):
        raise ValueError("published Registry snapshot lineage mismatch")
    if _parse_timestamp(published.generated_at) < _parse_timestamp(
        prior.generated_at
    ) or _parse_timestamp(receipt.generated_at) < _parse_timestamp(
        published.generated_at
    ):
        raise ValueError("published Registry timestamp lineage mismatch")
    return {
        "added_material_ids": added_ids,
        "added_entry_digests": added_digests,
    }


def material_registry_write_receipt_publication_bytes(
    receipt: OledMaterialRegistrySuccessorWriteArtifact,
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


def oled_material_registry_successor_postwrite_verification_artifact_digest(
    artifact: OledMaterialRegistrySuccessorPostwriteVerificationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("verification_artifact_digest", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "OLED_MATERIAL_REGISTRY_SUCCESSOR_POSTWRITE_VERIFIER_VERSION",
    "OledMaterialRegistrySuccessorPostwriteVerificationArtifact",
    "OledMaterialRegistrySuccessorPostwriteVerificationStatus",
    "build_oled_material_registry_successor_postwrite_verification_artifact",
    "material_registry_write_receipt_publication_bytes",
    "oled_material_registry_successor_postwrite_verification_artifact_digest",
]
