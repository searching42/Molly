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

from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistrySnapshot,
    oled_material_registry_snapshot_digest,
)
from ai4s_agent.domains.oled_material_registry_successor_preflight import (
    OledMaterialRegistrySuccessorPreflightArtifact,
    OledMaterialRegistrySuccessorPreflightStatus,
    oled_material_registry_successor_preflight_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_MATERIAL_REGISTRY_SUCCESSOR_WRITE_VERSION = (
    "oled_material_registry_successor_write.v1"
)
MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME = (
    "material_registry_successor_write.json"
)
MATERIAL_REGISTRY_SNAPSHOT_FILENAME = "material_registry_snapshot.json"


class OledMaterialRegistrySuccessorWriteStatus(str, Enum):
    SUCCESSOR_SNAPSHOT_PUBLISHED = "registry_successor_snapshot_published"


class OledMaterialRegistrySuccessorWriteArtifact(BaseModel):
    """Receipt for one exact immutable Registry successor publication."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_MATERIAL_REGISTRY_SUCCESSOR_WRITE_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    preflight_artifact_sha256: str
    preflight_artifact_digest: str
    prior_registry_snapshot_sha256: str
    prior_registry_snapshot_digest: str
    preflight_artifact: OledMaterialRegistrySuccessorPreflightArtifact
    prior_registry_snapshot: OledMaterialRegistrySnapshot
    published_successor_snapshot: OledMaterialRegistrySnapshot
    status: OledMaterialRegistrySuccessorWriteStatus
    publication_receipt_filename: Literal[
        "material_registry_successor_write.json"
    ] = MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME
    published_snapshot_filename: Literal[
        "material_registry_snapshot.json"
    ] = MATERIAL_REGISTRY_SNAPSHOT_FILENAME
    published_snapshot_file_sha256: str
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    added_entry_count: Annotated[StrictInt, Field(ge=1)]
    added_entry_cell_count: Annotated[StrictInt, Field(ge=0)]
    published_entry_count: Annotated[StrictInt, Field(ge=1)]
    added_material_ids: list[str] = Field(default_factory=list)
    added_entry_digests: list[str] = Field(default_factory=list)
    successor_registry_version: str
    successor_snapshot_digest: str
    write_artifact_digest: str
    publication_receipt_created: StrictBool = True
    immutable_successor_snapshot_published: StrictBool = True
    compare_and_swap_parent_bytes_verified: StrictBool = True
    preflight_bytes_rechecked_before_publication: StrictBool = True
    prior_snapshot_bytes_rechecked_before_publication: StrictBool = True
    append_only_transition_verified: StrictBool = True
    atomic_noreplace_directory_publication: StrictBool = True
    fsync_files_and_directories_completed: StrictBool = True
    published_directory_inode_bound: StrictBool = True
    published_payloads_revalidated: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    material_id_reserved: StrictBool = True
    material_id_assigned: StrictBool = True
    canonical_names_assigned: StrictBool = True
    aliases_assigned: StrictBool = True
    registry_entry_created: StrictBool = True
    registry_snapshot_published: StrictBool = True
    registry_written: StrictBool = True
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
        if value != OLED_MATERIAL_REGISTRY_SUCCESSOR_WRITE_VERSION:
            raise ValueError("unexpected Registry successor write version")
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
        "preflight_artifact_sha256",
        "preflight_artifact_digest",
        "prior_registry_snapshot_sha256",
        "prior_registry_snapshot_digest",
        "published_snapshot_file_sha256",
        "successor_snapshot_digest",
        "write_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("added_material_ids", "added_entry_digests")
    @classmethod
    def validate_sorted_unique_lists(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> OledMaterialRegistrySuccessorWriteArtifact:
        preflight = self.preflight_artifact
        prior = self.prior_registry_snapshot
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            preflight.generated_at
        ) or _parse_timestamp(self.generated_at) < _parse_timestamp(
            prior.generated_at
        ):
            raise ValueError("Registry successor write timestamp reversal")
        if oled_material_registry_successor_preflight_artifact_digest(
            preflight
        ) != preflight.preflight_artifact_digest:
            raise ValueError("Registry successor write embedded preflight changed")
        if oled_material_registry_snapshot_digest(prior) != prior.snapshot_digest:
            raise ValueError("Registry successor write prior snapshot changed")
        expected_bindings = {
            "run_id": preflight.run_id,
            "paper_id": preflight.paper_id,
            "preflight_artifact_digest": preflight.preflight_artifact_digest,
            "prior_registry_snapshot_sha256": (
                preflight.current_registry_snapshot_sha256
            ),
            "prior_registry_snapshot_digest": (
                preflight.current_registry_snapshot_digest
            ),
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry successor write {field_name} mismatch")
        derived = _derive_successor_write(preflight, prior)
        if self.published_successor_snapshot.model_dump(mode="json") != (
            derived["successor"].model_dump(mode="json")
        ):
            raise ValueError("Registry published successor snapshot changed")
        expected_fields = {
            "status": (
                OledMaterialRegistrySuccessorWriteStatus
                .SUCCESSOR_SNAPSHOT_PUBLISHED
            ),
            "published_snapshot_file_sha256": derived["snapshot_file_sha256"],
            "prior_entry_count": len(prior.entries),
            "added_entry_count": len(derived["added_material_ids"]),
            "added_entry_cell_count": preflight.planned_addition_cell_count,
            "published_entry_count": len(derived["successor"].entries),
            "added_material_ids": derived["added_material_ids"],
            "added_entry_digests": derived["added_entry_digests"],
            "successor_registry_version": derived["successor"].registry_version,
            "successor_snapshot_digest": derived["successor"].snapshot_digest,
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry successor write {field_name} mismatch")
        fixed_true = (
            "publication_receipt_created",
            "immutable_successor_snapshot_published",
            "compare_and_swap_parent_bytes_verified",
            "preflight_bytes_rechecked_before_publication",
            "prior_snapshot_bytes_rechecked_before_publication",
            "append_only_transition_verified",
            "atomic_noreplace_directory_publication",
            "fsync_files_and_directories_completed",
            "published_directory_inode_bound",
            "published_payloads_revalidated",
            "material_id_reserved",
            "material_id_assigned",
            "canonical_names_assigned",
            "aliases_assigned",
            "registry_entry_created",
            "registry_snapshot_published",
            "registry_written",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
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
            raise ValueError("Registry successor writer crossed its boundary")
        if oled_material_registry_successor_write_artifact_digest(self) != (
            self.write_artifact_digest
        ):
            raise ValueError("Registry successor write artifact digest mismatch")
        return self


def build_oled_material_registry_successor_write_artifact(
    *,
    preflight_artifact: OledMaterialRegistrySuccessorPreflightArtifact,
    preflight_artifact_sha256: str,
    prior_registry_snapshot: OledMaterialRegistrySnapshot,
    prior_registry_snapshot_sha256: str,
    generated_at: str,
) -> OledMaterialRegistrySuccessorWriteArtifact:
    preflight = OledMaterialRegistrySuccessorPreflightArtifact.model_validate(
        preflight_artifact.model_dump(mode="json")
    )
    prior = OledMaterialRegistrySnapshot.model_validate(
        prior_registry_snapshot.model_dump(mode="json")
    )
    derived = _derive_successor_write(preflight, prior)
    successor = derived["successor"]
    payload: dict[str, Any] = {
        "artifact_version": OLED_MATERIAL_REGISTRY_SUCCESSOR_WRITE_VERSION,
        "run_id": preflight.run_id,
        "paper_id": preflight.paper_id,
        "generated_at": generated_at,
        "preflight_artifact_sha256": _normalize_sha256(
            preflight_artifact_sha256,
            field_name="preflight_artifact_sha256",
        ),
        "preflight_artifact_digest": preflight.preflight_artifact_digest,
        "prior_registry_snapshot_sha256": _normalize_sha256(
            prior_registry_snapshot_sha256,
            field_name="prior_registry_snapshot_sha256",
        ),
        "prior_registry_snapshot_digest": prior.snapshot_digest,
        "preflight_artifact": preflight,
        "prior_registry_snapshot": prior,
        "published_successor_snapshot": successor,
        "status": (
            OledMaterialRegistrySuccessorWriteStatus.SUCCESSOR_SNAPSHOT_PUBLISHED
        ),
        "published_snapshot_file_sha256": derived["snapshot_file_sha256"],
        "prior_entry_count": len(prior.entries),
        "added_entry_count": len(derived["added_material_ids"]),
        "added_entry_cell_count": preflight.planned_addition_cell_count,
        "published_entry_count": len(successor.entries),
        "added_material_ids": derived["added_material_ids"],
        "added_entry_digests": derived["added_entry_digests"],
        "successor_registry_version": successor.registry_version,
        "successor_snapshot_digest": successor.snapshot_digest,
        "write_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistrySuccessorWriteArtifact.model_construct(
        **payload
    )
    payload["write_artifact_digest"] = (
        oled_material_registry_successor_write_artifact_digest(provisional)
    )
    return OledMaterialRegistrySuccessorWriteArtifact.model_validate(payload)


def _derive_successor_write(
    preflight: OledMaterialRegistrySuccessorPreflightArtifact,
    prior: OledMaterialRegistrySnapshot,
) -> dict[str, Any]:
    if preflight.status != (
        OledMaterialRegistrySuccessorPreflightStatus.READY_FOR_SUCCESSOR_WRITE
    ) or preflight.expected_successor_snapshot is None:
        raise ValueError("Registry successor preflight has no snapshot to publish")
    if preflight.current_registry_snapshot_digest != prior.snapshot_digest or (
        preflight.current_registry_snapshot.model_dump(mode="json")
        != prior.model_dump(mode="json")
    ):
        raise ValueError("compare-and-swap prior Registry does not match PR-X")
    successor = OledMaterialRegistrySnapshot.model_validate(
        preflight.expected_successor_snapshot.model_dump(mode="json")
    )
    if successor.snapshot_digest != preflight.expected_successor_snapshot_digest:
        raise ValueError("PR-X expected successor digest mismatch")
    prior_by_id = {entry.material_id: entry for entry in prior.entries}
    successor_by_id = {entry.material_id: entry for entry in successor.entries}
    for material_id, prior_entry in prior_by_id.items():
        successor_entry = successor_by_id.get(material_id)
        if successor_entry is None or successor_entry.model_dump(mode="json") != (
            prior_entry.model_dump(mode="json")
        ):
            raise ValueError("Registry successor changed a prior entry")
    planned_by_id = {
        item.registry_entry.material_id: item.registry_entry
        for item in preflight.planned_additions
    }
    if set(successor_by_id) != set(prior_by_id) | set(planned_by_id):
        raise ValueError("Registry successor entry coverage differs from PR-X")
    for material_id, planned_entry in planned_by_id.items():
        if successor_by_id[material_id].model_dump(mode="json") != (
            planned_entry.model_dump(mode="json")
        ):
            raise ValueError("Registry successor planned entry changed")
    if len(planned_by_id) != preflight.planned_addition_count:
        raise ValueError("Registry successor planned-addition count mismatch")
    snapshot_bytes = material_registry_snapshot_publication_bytes(successor)
    return {
        "successor": successor,
        "snapshot_file_sha256": (
            f"sha256:{hashlib.sha256(snapshot_bytes).hexdigest()}"
        ),
        "added_material_ids": sorted(planned_by_id),
        "added_entry_digests": sorted(
            entry.entry_digest for entry in planned_by_id.values()
        ),
    }


def material_registry_snapshot_publication_bytes(
    snapshot: OledMaterialRegistrySnapshot,
) -> bytes:
    return (
        json.dumps(
            snapshot.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def oled_material_registry_successor_write_artifact_digest(
    artifact: OledMaterialRegistrySuccessorWriteArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("write_artifact_digest", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "MATERIAL_REGISTRY_SNAPSHOT_FILENAME",
    "MATERIAL_REGISTRY_SUCCESSOR_WRITE_FILENAME",
    "OLED_MATERIAL_REGISTRY_SUCCESSOR_WRITE_VERSION",
    "OledMaterialRegistrySuccessorWriteArtifact",
    "OledMaterialRegistrySuccessorWriteStatus",
    "build_oled_material_registry_successor_write_artifact",
    "material_registry_snapshot_publication_bytes",
    "oled_material_registry_successor_write_artifact_digest",
]
