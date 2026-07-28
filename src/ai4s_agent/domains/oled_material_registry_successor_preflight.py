from __future__ import annotations

import hashlib
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

from ai4s_agent.domains.oled_material_registry_entry_adjudication import (
    OledMaterialRegistryApprovedEntryCandidate,
    OledMaterialRegistryEntryAdjudicationArtifact,
    oled_material_registry_entry_adjudication_artifact_digest,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    OledMaterialRegistrySnapshot,
    build_oled_material_registry_snapshot,
    oled_material_registry_snapshot_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_MATERIAL_REGISTRY_SUCCESSOR_PREFLIGHT_VERSION = (
    "oled_material_registry_successor_preflight.v1"
)


class OledMaterialRegistrySuccessorPreflightStatus(str, Enum):
    READY_FOR_SUCCESSOR_WRITE = "ready_for_registry_successor_write"
    NO_CHANGES_REQUIRED = "no_registry_changes_required"


class OledMaterialRegistryPlannedAddition(BaseModel):
    """One exact PR-W candidate planned for an append-only successor."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_entry_review_item_id: str
    source_adjudicated_item_digest: str
    source_candidate_digest: str
    identity_dependent_cell_count: Annotated[StrictInt, Field(ge=0)]
    registry_entry: OledMaterialRegistryEntry
    planned_addition_digest: str

    @field_validator("source_entry_review_item_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="source_entry_review_item_id")

    @field_validator(
        "source_adjudicated_item_digest",
        "source_candidate_digest",
        "planned_addition_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_integrity(self) -> OledMaterialRegistryPlannedAddition:
        if _planned_addition_digest(self) != self.planned_addition_digest:
            raise ValueError("Registry planned-addition digest mismatch")
        return self


class OledMaterialRegistrySuccessorPreflightArtifact(BaseModel):
    """A read-only plan for one append-only Registry successor snapshot."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_MATERIAL_REGISTRY_SUCCESSOR_PREFLIGHT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    entry_adjudication_sha256: str
    entry_adjudication_digest: str
    current_registry_snapshot_sha256: str
    current_registry_snapshot_digest: str
    entry_adjudication: OledMaterialRegistryEntryAdjudicationArtifact
    current_registry_snapshot: OledMaterialRegistrySnapshot
    status: OledMaterialRegistrySuccessorPreflightStatus
    eligible_candidate_count: Annotated[StrictInt, Field(ge=0)]
    eligible_candidate_cell_count: Annotated[StrictInt, Field(ge=0)]
    planned_addition_count: Annotated[StrictInt, Field(ge=0)]
    planned_addition_cell_count: Annotated[StrictInt, Field(ge=0)]
    prior_entry_count: Annotated[StrictInt, Field(ge=0)]
    expected_entry_count: Annotated[StrictInt, Field(ge=0)]
    planned_additions: list[OledMaterialRegistryPlannedAddition] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    successor_registry_version: str = ""
    expected_successor_snapshot_digest: str = ""
    expected_successor_snapshot: OledMaterialRegistrySnapshot | None = None
    preflight_artifact_digest: str
    offline_only: StrictBool = True
    plan_only: StrictBool = True
    exact_entry_adjudication_bytes_bound_at_construction: StrictBool = True
    exact_current_snapshot_bytes_bound_at_construction: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    embedded_models_revalidated: StrictBool = True
    current_snapshot_parent_bound: StrictBool = True
    current_snapshot_lineage_receipt_bound: StrictBool = False
    current_snapshot_conflicts_rechecked: StrictBool = True
    approved_candidate_chemistry_replayed: StrictBool = True
    deterministic_successor_planned: StrictBool = True
    append_only_plan_verified: StrictBool = True
    material_id_reserved: StrictBool = False
    material_id_assigned: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    registry_head_activated: StrictBool = False
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
        if value != OLED_MATERIAL_REGISTRY_SUCCESSOR_PREFLIGHT_VERSION:
            raise ValueError("unexpected Registry successor preflight version")
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
        "entry_adjudication_sha256",
        "entry_adjudication_digest",
        "current_registry_snapshot_sha256",
        "current_registry_snapshot_digest",
        "preflight_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("successor_registry_version")
    @classmethod
    def validate_optional_successor_version(cls, value: str) -> str:
        if not value:
            return ""
        return _validate_bound_id(value, field_name="successor_registry_version")

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
        value: list[OledMaterialRegistryPlannedAddition],
    ) -> list[OledMaterialRegistryPlannedAddition]:
        material_ids = [item.registry_entry.material_id for item in value]
        if material_ids != sorted(material_ids) or len(material_ids) != len(
            set(material_ids)
        ):
            raise ValueError("Registry planned additions must be material-ID ordered")
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> OledMaterialRegistrySuccessorPreflightArtifact:
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.entry_adjudication.generated_at
        ) or _parse_timestamp(self.generated_at) < _parse_timestamp(
            self.current_registry_snapshot.generated_at
        ):
            raise ValueError("Registry successor preflight timestamp reversal")
        if oled_material_registry_entry_adjudication_artifact_digest(
            self.entry_adjudication
        ) != self.entry_adjudication.adjudication_artifact_digest:
            raise ValueError("Registry successor preflight embedded PR-W changed")
        if oled_material_registry_snapshot_digest(
            self.current_registry_snapshot
        ) != self.current_registry_snapshot.snapshot_digest:
            raise ValueError("Registry successor preflight current snapshot changed")
        expected_bindings = {
            "run_id": self.entry_adjudication.run_id,
            "paper_id": self.entry_adjudication.paper_id,
            "entry_adjudication_digest": (
                self.entry_adjudication.adjudication_artifact_digest
            ),
            "current_registry_snapshot_digest": (
                self.current_registry_snapshot.snapshot_digest
            ),
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry successor preflight {field_name} mismatch")
        derived = _derive_successor_preflight(
            self.entry_adjudication,
            self.current_registry_snapshot,
            generated_at=self.generated_at,
        )
        expected_fields = {
            "status": derived["status"],
            "eligible_candidate_count": derived["candidate_count"],
            "eligible_candidate_cell_count": derived["candidate_cell_count"],
            "planned_addition_count": len(derived["planned_additions"]),
            "planned_addition_cell_count": derived["candidate_cell_count"],
            "prior_entry_count": len(self.current_registry_snapshot.entries),
            "expected_entry_count": derived["expected_entry_count"],
            "successor_registry_version": derived["successor_registry_version"],
            "expected_successor_snapshot_digest": derived[
                "expected_successor_snapshot_digest"
            ],
        }
        for field_name, expected in expected_fields.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry successor preflight {field_name} mismatch")
        if [item.model_dump(mode="json") for item in self.planned_additions] != [
            item.model_dump(mode="json") for item in derived["planned_additions"]
        ]:
            raise ValueError("Registry successor planned additions changed")
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
            raise ValueError("Registry expected successor snapshot changed")
        fixed_true = (
            "offline_only",
            "plan_only",
            "exact_entry_adjudication_bytes_bound_at_construction",
            "exact_current_snapshot_bytes_bound_at_construction",
            "embedded_models_revalidated",
            "current_snapshot_parent_bound",
            "current_snapshot_conflicts_rechecked",
            "approved_candidate_chemistry_replayed",
            "deterministic_successor_planned",
            "append_only_plan_verified",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "current_snapshot_lineage_receipt_bound",
            "material_id_reserved",
            "material_id_assigned",
            "registry_entry_created",
            "registry_written",
            "registry_head_activated",
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
            raise ValueError("Registry successor preflight crossed its boundary")
        if oled_material_registry_successor_preflight_artifact_digest(self) != (
            self.preflight_artifact_digest
        ):
            raise ValueError("Registry successor preflight artifact digest mismatch")
        return self


def build_oled_material_registry_successor_preflight_artifact(
    *,
    entry_adjudication: OledMaterialRegistryEntryAdjudicationArtifact,
    entry_adjudication_sha256: str,
    current_registry_snapshot: OledMaterialRegistrySnapshot,
    current_registry_snapshot_sha256: str,
    generated_at: str,
) -> OledMaterialRegistrySuccessorPreflightArtifact:
    adjudication = OledMaterialRegistryEntryAdjudicationArtifact.model_validate(
        entry_adjudication.model_dump(mode="json")
    )
    snapshot = OledMaterialRegistrySnapshot.model_validate(
        current_registry_snapshot.model_dump(mode="json")
    )
    derived = _derive_successor_preflight(
        adjudication,
        snapshot,
        generated_at=generated_at,
    )
    payload: dict[str, Any] = {
        "artifact_version": OLED_MATERIAL_REGISTRY_SUCCESSOR_PREFLIGHT_VERSION,
        "run_id": adjudication.run_id,
        "paper_id": adjudication.paper_id,
        "generated_at": generated_at,
        "entry_adjudication_sha256": _normalize_sha256(
            entry_adjudication_sha256,
            field_name="entry_adjudication_sha256",
        ),
        "entry_adjudication_digest": adjudication.adjudication_artifact_digest,
        "current_registry_snapshot_sha256": _normalize_sha256(
            current_registry_snapshot_sha256,
            field_name="current_registry_snapshot_sha256",
        ),
        "current_registry_snapshot_digest": snapshot.snapshot_digest,
        "entry_adjudication": adjudication,
        "current_registry_snapshot": snapshot,
        "status": derived["status"],
        "eligible_candidate_count": derived["candidate_count"],
        "eligible_candidate_cell_count": derived["candidate_cell_count"],
        "planned_addition_count": len(derived["planned_additions"]),
        "planned_addition_cell_count": derived["candidate_cell_count"],
        "prior_entry_count": len(snapshot.entries),
        "expected_entry_count": derived["expected_entry_count"],
        "planned_additions": derived["planned_additions"],
        "successor_registry_version": derived["successor_registry_version"],
        "expected_successor_snapshot_digest": derived[
            "expected_successor_snapshot_digest"
        ],
        "expected_successor_snapshot": derived["expected_successor_snapshot"],
        "preflight_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistrySuccessorPreflightArtifact.model_construct(
        **payload
    )
    payload["preflight_artifact_digest"] = (
        oled_material_registry_successor_preflight_artifact_digest(provisional)
    )
    return OledMaterialRegistrySuccessorPreflightArtifact.model_validate(payload)


def _derive_successor_preflight(
    adjudication: OledMaterialRegistryEntryAdjudicationArtifact,
    snapshot: OledMaterialRegistrySnapshot,
    *,
    generated_at: str,
) -> dict[str, Any]:
    original_snapshot = adjudication.request.resolution_request.registry_snapshot
    if snapshot.registry_id != original_snapshot.registry_id:
        raise ValueError("current Registry ID differs from the PR-W Registry")
    if _parse_timestamp(snapshot.generated_at) < _parse_timestamp(
        original_snapshot.generated_at
    ):
        raise ValueError("current Registry snapshot predates the PR-W snapshot")
    if _parse_timestamp(generated_at) < _parse_timestamp(
        adjudication.generated_at
    ) or _parse_timestamp(generated_at) < _parse_timestamp(snapshot.generated_at):
        raise ValueError("Registry successor preflight timestamp reversal")

    candidate_rows = []
    for item in adjudication.adjudicated_items:
        candidate = item.approved_entry_candidate
        if candidate is None:
            continue
        if not item.eligible_for_registry_write_preflight:
            raise ValueError("approved Registry candidate lost preflight eligibility")
        candidate_rows.append(
            (
                candidate,
                item.adjudicated_item_digest,
                item.request_item.identity_dependent_cell_count,
            )
        )
    if len(candidate_rows) != adjudication.registry_write_preflight_eligible_count:
        raise ValueError("Registry successor candidate coverage mismatch")
    candidate_cell_count = sum(row[2] for row in candidate_rows)
    if candidate_cell_count != (
        adjudication.registry_write_preflight_eligible_cell_count
    ):
        raise ValueError("Registry successor candidate cell coverage mismatch")

    _recheck_current_snapshot_conflicts(
        snapshot=snapshot,
        candidates=[row[0] for row in candidate_rows],
    )
    planned = sorted(
        (
            _build_planned_addition(
                candidate,
                source_adjudicated_item_digest=item_digest,
                identity_dependent_cell_count=cell_count,
            )
            for candidate, item_digest, cell_count in candidate_rows
        ),
        key=lambda item: item.registry_entry.material_id,
    )
    if not planned:
        return {
            "status": OledMaterialRegistrySuccessorPreflightStatus.NO_CHANGES_REQUIRED,
            "candidate_count": 0,
            "candidate_cell_count": 0,
            "planned_additions": [],
            "successor_registry_version": "",
            "expected_successor_snapshot_digest": "",
            "expected_successor_snapshot": None,
            "expected_entry_count": len(snapshot.entries),
        }

    successor_version = _successor_registry_version(
        current_snapshot_digest=snapshot.snapshot_digest,
        adjudication_digest=adjudication.adjudication_artifact_digest,
        planned_additions=planned,
    )
    successor = build_oled_material_registry_snapshot(
        registry_id=snapshot.registry_id,
        registry_version=successor_version,
        generated_at=generated_at,
        entries=[
            *snapshot.entries,
            *(item.registry_entry for item in planned),
        ],
    )
    if [
        entry.model_dump(mode="json")
        for entry in successor.entries
        if entry.material_id in {item.registry_entry.material_id for item in planned}
    ] != [item.registry_entry.model_dump(mode="json") for item in planned]:
        raise ValueError("Registry successor did not preserve planned additions")
    return {
        "status": OledMaterialRegistrySuccessorPreflightStatus.READY_FOR_SUCCESSOR_WRITE,
        "candidate_count": len(planned),
        "candidate_cell_count": candidate_cell_count,
        "planned_additions": planned,
        "successor_registry_version": successor_version,
        "expected_successor_snapshot_digest": successor.snapshot_digest,
        "expected_successor_snapshot": successor,
        "expected_entry_count": len(successor.entries),
    }


def _recheck_current_snapshot_conflicts(
    *,
    snapshot: OledMaterialRegistrySnapshot,
    candidates: Sequence[OledMaterialRegistryApprovedEntryCandidate],
) -> None:
    existing_ids = {entry.material_id for entry in snapshot.entries}
    existing_names = {
        name
        for entry in snapshot.entries
        for name in (entry.canonical_name, *entry.aliases)
    }
    existing_smiles = {entry.canonical_isomeric_smiles for entry in snapshot.entries}
    existing_inchi = {entry.standard_inchi for entry in snapshot.entries}
    existing_inchikey = {entry.inchikey for entry in snapshot.entries}
    batch_ids: set[str] = set()
    batch_names: set[str] = set()
    batch_smiles: set[str] = set()
    batch_inchi: set[str] = set()
    batch_inchikey: set[str] = set()
    for candidate in candidates:
        entry = candidate.registry_entry
        if entry.material_id in existing_ids:
            raise ValueError("Registry successor material ID is already occupied")
        if entry.material_id in batch_ids:
            raise ValueError("Registry successor candidate material ID is duplicated")
        names = {entry.canonical_name, *entry.aliases}
        if names & existing_names:
            raise ValueError("Registry successor candidate name conflicts with current Registry")
        if names & batch_names:
            raise ValueError("Registry successor candidate names conflict within the batch")
        if entry.canonical_isomeric_smiles in existing_smiles:
            raise ValueError("Registry successor canonical structure already exists")
        if entry.standard_inchi in existing_inchi:
            raise ValueError("Registry successor standard InChI already exists")
        if entry.inchikey in existing_inchikey:
            raise ValueError("Registry successor InChIKey already exists")
        if entry.canonical_isomeric_smiles in batch_smiles:
            raise ValueError("Registry successor canonical structure repeats in the batch")
        if entry.standard_inchi in batch_inchi:
            raise ValueError("Registry successor standard InChI repeats in the batch")
        if entry.inchikey in batch_inchikey:
            raise ValueError("Registry successor InChIKey repeats in the batch")
        batch_ids.add(entry.material_id)
        batch_names.update(names)
        batch_smiles.add(entry.canonical_isomeric_smiles)
        batch_inchi.add(entry.standard_inchi)
        batch_inchikey.add(entry.inchikey)


def _build_planned_addition(
    candidate: OledMaterialRegistryApprovedEntryCandidate,
    *,
    source_adjudicated_item_digest: str,
    identity_dependent_cell_count: int,
) -> OledMaterialRegistryPlannedAddition:
    payload: dict[str, Any] = {
        "source_entry_review_item_id": candidate.source_entry_review_item_id,
        "source_adjudicated_item_digest": source_adjudicated_item_digest,
        "source_candidate_digest": candidate.candidate_digest,
        "identity_dependent_cell_count": identity_dependent_cell_count,
        "registry_entry": candidate.registry_entry,
        "planned_addition_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryPlannedAddition.model_construct(**payload)
    payload["planned_addition_digest"] = _planned_addition_digest(provisional)
    return OledMaterialRegistryPlannedAddition.model_validate(payload)


def _successor_registry_version(
    *,
    current_snapshot_digest: str,
    adjudication_digest: str,
    planned_additions: Sequence[OledMaterialRegistryPlannedAddition],
) -> str:
    digest = _stable_hash(
        {
            "parent_snapshot_digest": current_snapshot_digest,
            "entry_adjudication_digest": adjudication_digest,
            "planned_addition_digests": [
                item.planned_addition_digest for item in planned_additions
            ],
        }
    )
    return f"successor-{digest.removeprefix('sha256:')[:24]}"


def _planned_addition_digest(item: OledMaterialRegistryPlannedAddition) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("planned_addition_digest", None)
    return _stable_hash(payload)


def oled_material_registry_successor_preflight_artifact_digest(
    artifact: OledMaterialRegistrySuccessorPreflightArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("preflight_artifact_digest", None)
    return _stable_hash(payload)


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "OLED_MATERIAL_REGISTRY_SUCCESSOR_PREFLIGHT_VERSION",
    "OledMaterialRegistryPlannedAddition",
    "OledMaterialRegistrySuccessorPreflightArtifact",
    "OledMaterialRegistrySuccessorPreflightStatus",
    "build_oled_material_registry_successor_preflight_artifact",
    "oled_material_registry_successor_preflight_artifact_digest",
]
