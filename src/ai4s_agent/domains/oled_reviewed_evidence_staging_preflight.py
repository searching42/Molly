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

from ai4s_agent.domains.oled_contracts import (
    DEFAULT_OLED_REPRESENTATION_CONTRACT,
    OledCausalLayer,
)
from ai4s_agent.domains.oled_layered_schema import OledComparisonContextStatus
from ai4s_agent.domains.oled_observation_materialization_candidate import (
    OledObservationMaterializationCandidateArtifact,
    OledObservationMaterializationCandidateItem,
    oled_observation_materialization_candidate_artifact_digest,
)
from ai4s_agent.domains.oled_property_ontology import (
    DEFAULT_OLED_PROPERTY_ONTOLOGY,
    OLED_PHOTOPHYSICAL_COMPARISON_CONTEXT_FIELDS,
    OLED_PHOTOPHYSICAL_CONTEXT_POLICY,
)
from ai4s_agent.domains.oled_reported_values import validate_reported_value_contract
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)
from ai4s_agent.domains.oled_units import (
    _CONDITION_UNIT_RULES,
    _PROPERTY_UNIT_RULES,
)


OLED_REVIEWED_EVIDENCE_LEDGER_SNAPSHOT_VERSION = (
    "oled_reviewed_evidence_ledger_snapshot.v1"
)
OLED_REVIEWED_EVIDENCE_STAGING_PREFLIGHT_VERSION = (
    "oled_reviewed_evidence_staging_preflight.v1"
)
OLED_REVIEWED_EVIDENCE_SEMANTIC_CONTRACT_VERSION = (
    "oled_reviewed_evidence_semantic_contract.v1"
)


class OledReviewedEvidenceLedgerEntryStatus(str, Enum):
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"


class OledReviewedEvidencePreflightDisposition(str, Enum):
    NEW_CLAIM_READY = "new_claim_ready"
    EXACT_REPLAY = "exact_replay"
    CONSISTENT_DUPLICATE_READY = "consistent_duplicate_ready"
    VALUE_CONFLICT_QUARANTINE = "value_conflict_quarantine"
    INCOMPLETE_CONTEXT_QUARANTINE = "incomplete_context_quarantine"
    REVISION_REQUIRES_REVIEW = "revision_requires_review"
    SEMANTIC_CONTRACT_MIGRATION_REQUIRED = "semantic_contract_migration_required"


class OledReviewedEvidenceStagingPreflightStatus(str, Enum):
    READY_FOR_LEDGER_WRITE = "ready_for_reviewed_evidence_ledger_write"
    MANUAL_EXCEPTION_REVIEW_REQUIRED = "manual_exception_review_required"
    NO_LEDGER_CHANGES_REQUIRED = "no_ledger_changes_required"


class OledReviewedEvidenceVerificationFacets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exact_source_bytes_bound: StrictBool = True
    source_transcription_confirmed: StrictBool = True
    property_mapping_confirmed: StrictBool = True
    material_identity_confirmed: StrictBool = True
    comparison_context_assessed: StrictBool = True
    scientific_consistency_reviewed: StrictBool = False
    confidence_score_assigned: StrictBool = False

    @model_validator(mode="after")
    def validate_boundary(self) -> OledReviewedEvidenceVerificationFacets:
        fixed_true = (
            "exact_source_bytes_bound",
            "source_transcription_confirmed",
            "property_mapping_confirmed",
            "material_identity_confirmed",
            "comparison_context_assessed",
        )
        fixed_false = (
            "scientific_consistency_reviewed",
            "confidence_score_assigned",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("reviewed-evidence verification boundary mismatch")
        return self


class OledReviewedEvidenceSemanticContractSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contract_version: str = OLED_REVIEWED_EVIDENCE_SEMANTIC_CONTRACT_VERSION
    ontology_definitions: list[dict[str, Any]]
    representation_contract: dict[str, Any]
    property_unit_rules: dict[str, Any]
    condition_unit_rules: dict[str, Any]
    comparison_context_fields: list[str]
    comparison_context_policy: str
    contract_digest: str

    @field_validator("contract_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_REVIEWED_EVIDENCE_SEMANTIC_CONTRACT_VERSION:
            raise ValueError("unexpected reviewed-evidence semantic contract version")
        return value

    @field_validator("contract_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="contract_digest")

    @field_validator("comparison_context_fields")
    @classmethod
    def validate_context_fields(cls, value: list[str]) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("semantic contract context fields must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_integrity(self) -> OledReviewedEvidenceSemanticContractSnapshot:
        if _semantic_contract_digest(self) != self.contract_digest:
            raise ValueError("reviewed-evidence semantic contract digest mismatch")
        return self


class OledReviewedEvidenceLedgerEntry(BaseModel):
    """One immutable source claim plus one versioned semantic projection."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    entry_id: str
    source_claim_id: str
    projection_id: str
    conflict_key: str
    source_materialization_artifact_digest: str
    source_candidate_id: str
    source_candidate_digest: str
    cell_disposition_digest: str
    source_pdf_sha256: str
    source_cell_digest: str
    table_id: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    column_index: Annotated[StrictInt, Field(ge=0)]
    selected_material_id: str
    registry_entry_digest: str
    property_id: str
    target_layer: OledCausalLayer
    reported_value: float | int | str | None = None
    reported_value_text: str
    reported_decimal_places: Annotated[StrictInt, Field(ge=0)] | None = None
    reported_unit: str
    normalized_value: float | int | str | None = None
    normalized_unit: str | None = None
    comparison_context_status: OledComparisonContextStatus
    comparison_context_hash: str | None = None
    comparison_context_missing_fields: list[str] = Field(default_factory=list)
    semantic_contract_digest: str
    projection_payload_digest: str
    status: OledReviewedEvidenceLedgerEntryStatus
    created_at: str
    supersedes_projection_id: str | None = None
    entry_digest: str

    @field_validator(
        "entry_id",
        "source_claim_id",
        "projection_id",
        "conflict_key",
        "source_candidate_id",
        "selected_material_id",
        "property_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("table_id")
    @classmethod
    def validate_table_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="table_id")

    @field_validator("reported_value_text", "reported_unit")
    @classmethod
    def validate_reported_text(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or len(value) > 20_000:
            raise ValueError(f"{info.field_name} must be bounded text")
        if info.field_name == "reported_value_text" and not value:
            raise ValueError("reported_value_text is required")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError(f"{info.field_name} contains control text")
        return value

    @field_validator(
        "source_materialization_artifact_digest",
        "source_candidate_digest",
        "cell_disposition_digest",
        "source_pdf_sha256",
        "source_cell_digest",
        "registry_entry_digest",
        "semantic_contract_digest",
        "projection_payload_digest",
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

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="created_at")

    @field_validator("supersedes_projection_id")
    @classmethod
    def validate_supersedes_projection_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_bound_id(value, field_name="supersedes_projection_id")

    @field_validator("comparison_context_missing_fields")
    @classmethod
    def validate_missing_fields(cls, value: list[str]) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("ledger context missing fields must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_entry_integrity(self) -> OledReviewedEvidenceLedgerEntry:
        if self.entry_id != f"reviewed-evidence:{self.projection_id.split(':', 1)[-1]}":
            raise ValueError("reviewed-evidence ledger entry_id mismatch")
        expected_source_claim_id = _source_claim_id_from_fields(
            source_pdf_sha256=self.source_pdf_sha256,
            source_cell_digest=self.source_cell_digest,
        )
        if self.source_claim_id != expected_source_claim_id:
            raise ValueError("reviewed-evidence source claim_id mismatch")
        expected_projection_id = _projection_id_from_fields(
            source_claim_id=self.source_claim_id,
            source_candidate_digest=self.source_candidate_digest,
            selected_material_id=self.selected_material_id,
            registry_entry_digest=self.registry_entry_digest,
            cell_disposition_digest=self.cell_disposition_digest,
            semantic_contract_digest=self.semantic_contract_digest,
        )
        if self.projection_id != expected_projection_id:
            raise ValueError("reviewed-evidence projection_id mismatch")
        expected_projection_payload_digest = _projection_payload_digest(
            _ledger_projection_payload(self)
        )
        if self.projection_payload_digest != expected_projection_payload_digest:
            raise ValueError("reviewed-evidence projection payload digest mismatch")
        if self.conflict_key != _conflict_key_from_fields(
            selected_material_id=self.selected_material_id,
            property_id=self.property_id,
            target_layer=self.target_layer,
            comparison_context_status=self.comparison_context_status,
            comparison_context_hash=self.comparison_context_hash,
            comparison_context_missing_fields=self.comparison_context_missing_fields,
        ):
            raise ValueError("reviewed-evidence conflict key mismatch")
        if self.target_layer == OledCausalLayer.DEVICE:
            raise ValueError("device-only reviewed evidence is forbidden")
        if self.supersedes_projection_id == self.projection_id:
            raise ValueError("reviewed-evidence projection cannot supersede itself")
        validate_reported_value_contract(
            value=self.reported_value,
            reported_value_text=self.reported_value_text,
            reported_decimal_places_value=self.reported_decimal_places,
            label="reviewed-evidence reported value",
        )
        if self.comparison_context_status == OledComparisonContextStatus.INCOMPLETE:
            if not self.comparison_context_missing_fields or self.comparison_context_hash is not None:
                raise ValueError("incomplete ledger context boundary mismatch")
        elif self.comparison_context_missing_fields:
            raise ValueError("complete ledger context cannot list missing fields")
        if _ledger_entry_digest(self) != self.entry_digest:
            raise ValueError("reviewed-evidence ledger entry digest mismatch")
        return self


class OledReviewedEvidenceLedgerSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    snapshot_version: str = OLED_REVIEWED_EVIDENCE_LEDGER_SNAPSHOT_VERSION
    snapshot_id: str
    generated_at: str
    semantic_contracts: list[OledReviewedEvidenceSemanticContractSnapshot] = Field(
        default_factory=list,
        max_length=10_000,
    )
    entries: list[OledReviewedEvidenceLedgerEntry] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    entry_count: Annotated[StrictInt, Field(ge=0)]
    snapshot_digest: str
    append_only: StrictBool = True
    device_only_entry_count: Literal[0] = 0

    @field_validator("snapshot_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_REVIEWED_EVIDENCE_LEDGER_SNAPSHOT_VERSION:
            raise ValueError("unexpected reviewed-evidence ledger snapshot version")
        return value

    @field_validator("snapshot_id")
    @classmethod
    def validate_snapshot_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="snapshot_id")

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator("snapshot_digest")
    @classmethod
    def validate_snapshot_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="snapshot_digest")

    @field_validator("entries")
    @classmethod
    def validate_entry_order(
        cls,
        value: list[OledReviewedEvidenceLedgerEntry],
    ) -> list[OledReviewedEvidenceLedgerEntry]:
        order = [item.entry_id for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("reviewed-evidence ledger entries must be sorted and unique")
        return value

    @field_validator("semantic_contracts")
    @classmethod
    def validate_semantic_contract_order(
        cls,
        value: list[OledReviewedEvidenceSemanticContractSnapshot],
    ) -> list[OledReviewedEvidenceSemanticContractSnapshot]:
        order = [item.contract_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("ledger semantic contracts must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_snapshot_integrity(self) -> OledReviewedEvidenceLedgerSnapshot:
        if self.entry_count != len(self.entries) or self.device_only_entry_count != 0:
            raise ValueError("reviewed-evidence ledger snapshot count mismatch")
        known_contracts = {
            contract.contract_digest for contract in self.semantic_contracts
        }
        if any(
            entry.semantic_contract_digest not in known_contracts
            for entry in self.entries
        ):
            raise ValueError("ledger entry semantic contract is not embedded in snapshot")
        live_counts: dict[str, int] = {}
        for entry in self.entries:
            if _parse_timestamp(entry.created_at) > _parse_timestamp(self.generated_at):
                raise ValueError("ledger entry postdates its snapshot")
            if entry.status in {
                OledReviewedEvidenceLedgerEntryStatus.ACTIVE,
                OledReviewedEvidenceLedgerEntryStatus.QUARANTINED,
            }:
                live_counts[entry.source_claim_id] = live_counts.get(entry.source_claim_id, 0) + 1
        if any(count > 1 for count in live_counts.values()):
            raise ValueError("ledger snapshot has multiple live projections for one source claim")
        if not self.append_only:
            raise ValueError("reviewed-evidence ledger must be append-only")
        if oled_reviewed_evidence_ledger_snapshot_digest(self) != self.snapshot_digest:
            raise ValueError("reviewed-evidence ledger snapshot digest mismatch")
        return self


class OledReviewedEvidencePreflightItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    preflight_item_id: str
    source_candidate: OledObservationMaterializationCandidateItem
    source_claim_id: str
    projection_id: str
    conflict_key: str
    source_row_group_id: str
    semantic_contract_digest: str
    disposition: OledReviewedEvidencePreflightDisposition
    matching_entry_ids: list[str] = Field(default_factory=list)
    gold_blocker_codes: list[str] = Field(default_factory=list)
    verification_facets: OledReviewedEvidenceVerificationFacets
    comparison_ready: StrictBool
    ledger_write_required: StrictBool
    quarantine_on_write: StrictBool
    manual_exception_review_required: StrictBool
    candidate_id_collision_detected: StrictBool
    reviewed_evidence_staged: StrictBool = False
    direct_admission_eligible: StrictBool = False
    gold_record_created: StrictBool = False
    preflight_item_digest: str

    @field_validator(
        "preflight_item_id",
        "source_claim_id",
        "projection_id",
        "conflict_key",
        "source_row_group_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("semantic_contract_digest", "preflight_item_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("matching_entry_ids", "gold_blocker_codes")
    @classmethod
    def validate_sorted_unique_text(cls, value: list[str], info: Any) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_item_integrity(self) -> OledReviewedEvidencePreflightItem:
        if self.preflight_item_id != f"reviewed-evidence-preflight:{self.projection_id.split(':', 1)[-1]}":
            raise ValueError("reviewed-evidence preflight item_id mismatch")
        if self.source_candidate.canonical_observation.layer == OledCausalLayer.DEVICE:
            raise ValueError("device-only candidate crossed reviewed-evidence preflight")
        if self.comparison_ready != self.source_candidate.comparison_ready:
            raise ValueError("reviewed-evidence comparison readiness mismatch")
        expected_flags = _disposition_flags(self.disposition)
        for field_name, expected in expected_flags.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"reviewed-evidence {field_name} mismatch")
        if "missing_confidence_assessment" not in self.gold_blocker_codes:
            raise ValueError("reviewed-evidence preflight must preserve missing confidence")
        if self.reviewed_evidence_staged or self.direct_admission_eligible or self.gold_record_created:
            raise ValueError("reviewed-evidence item crossed its preflight boundary")
        if _preflight_item_digest(self) != self.preflight_item_digest:
            raise ValueError("reviewed-evidence preflight item digest mismatch")
        return self


class OledReviewedEvidenceSourceRowGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_row_group_id: str
    staging_item_id: str
    identity_group_id: str
    selected_material_id: str
    source_pdf_sha256: str
    table_id: str
    row_index: Annotated[StrictInt, Field(ge=0)]
    preflight_item_ids: list[str] = Field(min_length=1)
    observation_count: Annotated[StrictInt, Field(ge=1)]
    group_digest: str

    @field_validator(
        "source_row_group_id",
        "staging_item_id",
        "identity_group_id",
        "selected_material_id",
    )
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_pdf_sha256", "group_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("table_id")
    @classmethod
    def validate_table_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="table_id")

    @field_validator("preflight_item_ids")
    @classmethod
    def validate_item_ids(cls, value: list[str]) -> list[str]:
        if value != sorted(value) or len(value) != len(set(value)):
            raise ValueError("source-row item ids must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_group_integrity(self) -> OledReviewedEvidenceSourceRowGroup:
        if self.observation_count != len(self.preflight_item_ids):
            raise ValueError("reviewed-evidence source-row count mismatch")
        expected_id = _source_row_group_id(
            staging_item_id=self.staging_item_id,
            identity_group_id=self.identity_group_id,
            selected_material_id=self.selected_material_id,
            source_pdf_sha256=self.source_pdf_sha256,
            table_id=self.table_id,
            row_index=self.row_index,
        )
        if self.source_row_group_id != expected_id:
            raise ValueError("reviewed-evidence source-row group_id mismatch")
        if _source_row_group_digest(self) != self.group_digest:
            raise ValueError("reviewed-evidence source-row group digest mismatch")
        return self


class OledReviewedEvidenceStagingPreflightArtifact(BaseModel):
    """Exact PR-Q plus ledger snapshot classification; publishes no evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_REVIEWED_EVIDENCE_STAGING_PREFLIGHT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    materialization_artifact_sha256: str
    materialization_artifact_digest: str
    ledger_snapshot_sha256: str
    ledger_snapshot_digest: str
    materialization_artifact: OledObservationMaterializationCandidateArtifact
    ledger_snapshot: OledReviewedEvidenceLedgerSnapshot
    semantic_contract: OledReviewedEvidenceSemanticContractSnapshot
    status: OledReviewedEvidenceStagingPreflightStatus
    source_candidate_count: Annotated[StrictInt, Field(ge=0)]
    source_row_group_count: Annotated[StrictInt, Field(ge=0)]
    ledger_write_count: Annotated[StrictInt, Field(ge=0)]
    exact_replay_count: Annotated[StrictInt, Field(ge=0)]
    consistent_duplicate_count: Annotated[StrictInt, Field(ge=0)]
    conflict_quarantine_count: Annotated[StrictInt, Field(ge=0)]
    incomplete_context_quarantine_count: Annotated[StrictInt, Field(ge=0)]
    revision_review_count: Annotated[StrictInt, Field(ge=0)]
    semantic_contract_migration_count: Annotated[StrictInt, Field(ge=0)]
    candidate_id_collision_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    preflight_items: list[OledReviewedEvidencePreflightItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    source_row_groups: list[OledReviewedEvidenceSourceRowGroup] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    preflight_artifact_digest: str
    preflight_only: StrictBool = True
    offline_only: StrictBool = True
    exact_materialization_bytes_bound: StrictBool = True
    exact_ledger_snapshot_bytes_bound: StrictBool = True
    source_claims_and_semantic_projections_separated: StrictBool = True
    semantic_contract_pinned: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    reviewed_evidence_staged: StrictBool = False
    ledger_written: StrictBool = False
    source_values_corrected: StrictBool = False
    confidence_score_invented: StrictBool = False
    direct_admission_eligible: StrictBool = False
    registry_written: StrictBool = False
    aliases_mutated: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value != OLED_REVIEWED_EVIDENCE_STAGING_PREFLIGHT_VERSION:
            raise ValueError("unexpected reviewed-evidence staging preflight version")
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
        "materialization_artifact_sha256",
        "materialization_artifact_digest",
        "ledger_snapshot_sha256",
        "ledger_snapshot_digest",
        "preflight_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("preflight_items")
    @classmethod
    def validate_item_order(
        cls,
        value: list[OledReviewedEvidencePreflightItem],
    ) -> list[OledReviewedEvidencePreflightItem]:
        order = [item.preflight_item_id for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("reviewed-evidence preflight items must be sorted and unique")
        return value

    @field_validator("source_row_groups")
    @classmethod
    def validate_group_order(
        cls,
        value: list[OledReviewedEvidenceSourceRowGroup],
    ) -> list[OledReviewedEvidenceSourceRowGroup]:
        order = [item.source_row_group_id for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("reviewed-evidence source-row groups must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledReviewedEvidenceStagingPreflightArtifact:
        generated = _parse_timestamp(self.generated_at)
        if generated < _parse_timestamp(self.materialization_artifact.generated_at):
            raise ValueError("reviewed-evidence preflight predates PR-Q")
        if generated < _parse_timestamp(self.ledger_snapshot.generated_at):
            raise ValueError("reviewed-evidence preflight predates ledger snapshot")
        current_semantic_contract = (
            build_oled_reviewed_evidence_semantic_contract_snapshot()
        )
        if self.semantic_contract.model_dump(mode="json") != (
            current_semantic_contract.model_dump(mode="json")
        ):
            raise ValueError("reviewed-evidence semantic contract does not match v1 implementation")
        if self.run_id != self.materialization_artifact.run_id or self.paper_id != (
            self.materialization_artifact.paper_id
        ):
            raise ValueError("reviewed-evidence preflight source identity mismatch")
        if self.materialization_artifact_digest != (
            self.materialization_artifact.artifact_digest
        ) or oled_observation_materialization_candidate_artifact_digest(
            self.materialization_artifact
        ) != self.materialization_artifact_digest:
            raise ValueError("reviewed-evidence embedded PR-Q digest mismatch")
        if self.ledger_snapshot_digest != self.ledger_snapshot.snapshot_digest or (
            oled_reviewed_evidence_ledger_snapshot_digest(self.ledger_snapshot)
            != self.ledger_snapshot_digest
        ):
            raise ValueError("reviewed-evidence embedded ledger snapshot mismatch")
        expected_items = _derive_preflight_items(
            self.materialization_artifact,
            self.ledger_snapshot,
            self.semantic_contract,
        )
        expected_groups = _derive_source_row_groups(expected_items)
        if [item.model_dump(mode="json") for item in self.preflight_items] != [
            item.model_dump(mode="json") for item in expected_items
        ]:
            raise ValueError("reviewed-evidence preflight item derivation mismatch")
        if [item.model_dump(mode="json") for item in self.source_row_groups] != [
            item.model_dump(mode="json") for item in expected_groups
        ]:
            raise ValueError("reviewed-evidence source-row derivation mismatch")
        expected_counts = _preflight_counts(expected_items, expected_groups)
        expected_counts.update(
            {
                "source_candidate_count": len(
                    self.materialization_artifact.observation_candidates
                ),
                "upstream_ontology_review_pending_cell_count": (
                    self.materialization_artifact.upstream_ontology_review_pending_cell_count
                ),
            }
        )
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"reviewed-evidence {field_name} mismatch")
        if self.status != _preflight_status(expected_items) or self.device_only_cell_count != 0:
            raise ValueError("reviewed-evidence preflight status or device boundary mismatch")
        fixed_true = (
            "preflight_only",
            "offline_only",
            "exact_materialization_bytes_bound",
            "exact_ledger_snapshot_bytes_bound",
            "source_claims_and_semantic_projections_separated",
            "semantic_contract_pinned",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "reviewed_evidence_staged",
            "ledger_written",
            "source_values_corrected",
            "confidence_score_invented",
            "direct_admission_eligible",
            "registry_written",
            "aliases_mutated",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("reviewed-evidence artifact crossed its preflight boundary")
        if oled_reviewed_evidence_staging_preflight_artifact_digest(self) != (
            self.preflight_artifact_digest
        ):
            raise ValueError("reviewed-evidence staging preflight digest mismatch")
        return self


def build_oled_reviewed_evidence_semantic_contract_snapshot(
) -> OledReviewedEvidenceSemanticContractSnapshot:
    payload: dict[str, Any] = {
        "contract_version": OLED_REVIEWED_EVIDENCE_SEMANTIC_CONTRACT_VERSION,
        "ontology_definitions": _canonicalize(
            [
                definition.model_dump(mode="python")
                for definition in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties()
            ]
        ),
        "representation_contract": _canonicalize(
            DEFAULT_OLED_REPRESENTATION_CONTRACT.model_dump(mode="python")
        ),
        "property_unit_rules": _canonicalize(
            {
                key: rule.model_dump(mode="python")
                for key, rule in sorted(_PROPERTY_UNIT_RULES.items())
            }
        ),
        "condition_unit_rules": _canonicalize(
            {
                key: rule.model_dump(mode="python")
                for key, rule in sorted(_CONDITION_UNIT_RULES.items())
            }
        ),
        "comparison_context_fields": sorted(
            OLED_PHOTOPHYSICAL_COMPARISON_CONTEXT_FIELDS
        ),
        "comparison_context_policy": OLED_PHOTOPHYSICAL_CONTEXT_POLICY,
        "contract_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidenceSemanticContractSnapshot.model_construct(
        **payload
    )
    payload["contract_digest"] = _semantic_contract_digest(provisional)
    return OledReviewedEvidenceSemanticContractSnapshot.model_validate(payload)


def build_empty_oled_reviewed_evidence_ledger_snapshot(
    *,
    generated_at: str,
    snapshot_id: str = "reviewed-evidence-ledger:genesis",
) -> OledReviewedEvidenceLedgerSnapshot:
    payload: dict[str, Any] = {
        "snapshot_version": OLED_REVIEWED_EVIDENCE_LEDGER_SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id,
        "generated_at": generated_at,
        "semantic_contracts": [],
        "entries": [],
        "entry_count": 0,
        "snapshot_digest": "sha256:" + "0" * 64,
        "append_only": True,
        "device_only_entry_count": 0,
    }
    provisional = OledReviewedEvidenceLedgerSnapshot.model_construct(**payload)
    payload["snapshot_digest"] = oled_reviewed_evidence_ledger_snapshot_digest(
        provisional
    )
    return OledReviewedEvidenceLedgerSnapshot.model_validate(payload)


def build_oled_reviewed_evidence_ledger_entry_from_candidate(
    *,
    candidate: OledObservationMaterializationCandidateItem,
    source_materialization_artifact_digest: str,
    semantic_contract: OledReviewedEvidenceSemanticContractSnapshot,
    status: OledReviewedEvidenceLedgerEntryStatus,
    created_at: str,
    supersedes_projection_id: str | None = None,
) -> OledReviewedEvidenceLedgerEntry:
    if candidate.canonical_observation.layer == OledCausalLayer.DEVICE:
        raise ValueError("device-only reviewed evidence is forbidden")
    source_claim_id = _source_claim_id(candidate)
    projection_id = _projection_id(
        candidate,
        source_claim_id,
        semantic_contract.contract_digest,
    )
    canonical = candidate.canonical_observation
    projection_payload = _candidate_projection_payload(
        candidate=candidate,
        source_materialization_artifact_digest=source_materialization_artifact_digest,
        semantic_contract_digest=semantic_contract.contract_digest,
    )
    payload: dict[str, Any] = {
        "entry_id": f"reviewed-evidence:{projection_id.split(':', 1)[-1]}",
        "source_claim_id": source_claim_id,
        "projection_id": projection_id,
        "conflict_key": _conflict_key(candidate),
        "source_materialization_artifact_digest": _normalize_sha256(
            source_materialization_artifact_digest,
            field_name="source_materialization_artifact_digest",
        ),
        "source_candidate_id": candidate.candidate_id,
        "source_candidate_digest": candidate.candidate_digest,
        "cell_disposition_digest": candidate.cell_disposition_digest,
        "source_pdf_sha256": candidate.source_pdf_sha256,
        "source_cell_digest": candidate.source_cell_digest,
        "table_id": candidate.table_id,
        "row_index": candidate.row_index,
        "column_index": candidate.column_index,
        "selected_material_id": candidate.selected_existing_material_id,
        "registry_entry_digest": _stable_hash(
            candidate.selected_registry_entry.model_dump(mode="json")
        ),
        "property_id": canonical.property_id,
        "target_layer": canonical.layer,
        "reported_value": candidate.property_observation.value,
        "reported_value_text": candidate.property_observation.reported_value_text,
        "reported_decimal_places": (
            candidate.property_observation.reported_decimal_places
        ),
        "reported_unit": candidate.property_observation.unit or "",
        "normalized_value": canonical.normalized_value,
        "normalized_unit": canonical.normalized_unit,
        "comparison_context_status": canonical.comparison_context_status,
        "comparison_context_hash": canonical.comparison_context_hash,
        "comparison_context_missing_fields": sorted(
            canonical.comparison_context_missing_fields
        ),
        "semantic_contract_digest": semantic_contract.contract_digest,
        "projection_payload_digest": _projection_payload_digest(
            projection_payload
        ),
        "status": status,
        "created_at": created_at,
        "supersedes_projection_id": supersedes_projection_id,
        "entry_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidenceLedgerEntry.model_construct(**payload)
    payload["entry_digest"] = _ledger_entry_digest(provisional)
    return OledReviewedEvidenceLedgerEntry.model_validate(payload)


def build_oled_reviewed_evidence_ledger_snapshot(
    *,
    entries: list[OledReviewedEvidenceLedgerEntry],
    generated_at: str,
    snapshot_id: str,
    semantic_contracts: list[
        OledReviewedEvidenceSemanticContractSnapshot
    ] | None = None,
) -> OledReviewedEvidenceLedgerSnapshot:
    clean_entries = sorted(
        [
            OledReviewedEvidenceLedgerEntry.model_validate(
                entry.model_dump(mode="json")
            )
            for entry in entries
        ],
        key=lambda entry: entry.entry_id,
    )
    contracts = list(semantic_contracts or [])
    if clean_entries and not contracts:
        current = build_oled_reviewed_evidence_semantic_contract_snapshot()
        if any(
            entry.semantic_contract_digest != current.contract_digest
            for entry in clean_entries
        ):
            raise ValueError("ledger snapshot requires every historical semantic contract")
        contracts = [current]
    clean_contracts = sorted(
        [
            OledReviewedEvidenceSemanticContractSnapshot.model_validate(
                contract.model_dump(mode="json")
            )
            for contract in contracts
        ],
        key=lambda contract: contract.contract_digest,
    )
    payload: dict[str, Any] = {
        "snapshot_version": OLED_REVIEWED_EVIDENCE_LEDGER_SNAPSHOT_VERSION,
        "snapshot_id": snapshot_id,
        "generated_at": generated_at,
        "semantic_contracts": clean_contracts,
        "entries": clean_entries,
        "entry_count": len(clean_entries),
        "snapshot_digest": "sha256:" + "0" * 64,
        "append_only": True,
        "device_only_entry_count": 0,
    }
    provisional = OledReviewedEvidenceLedgerSnapshot.model_construct(**payload)
    payload["snapshot_digest"] = oled_reviewed_evidence_ledger_snapshot_digest(
        provisional
    )
    return OledReviewedEvidenceLedgerSnapshot.model_validate(payload)


def build_oled_reviewed_evidence_staging_preflight_artifact(
    *,
    materialization_artifact: OledObservationMaterializationCandidateArtifact,
    materialization_artifact_sha256: str,
    ledger_snapshot: OledReviewedEvidenceLedgerSnapshot,
    ledger_snapshot_sha256: str,
    generated_at: str,
) -> OledReviewedEvidenceStagingPreflightArtifact:
    materialization = OledObservationMaterializationCandidateArtifact.model_validate(
        materialization_artifact.model_dump(mode="json")
    )
    ledger = OledReviewedEvidenceLedgerSnapshot.model_validate(
        ledger_snapshot.model_dump(mode="json")
    )
    semantic_contract = build_oled_reviewed_evidence_semantic_contract_snapshot()
    normalized_materialization_sha = _normalize_sha256(
        materialization_artifact_sha256,
        field_name="materialization_artifact_sha256",
    )
    normalized_ledger_sha = _normalize_sha256(
        ledger_snapshot_sha256,
        field_name="ledger_snapshot_sha256",
    )
    items = _derive_preflight_items(materialization, ledger, semantic_contract)
    groups = _derive_source_row_groups(items)
    payload: dict[str, Any] = {
        "artifact_version": OLED_REVIEWED_EVIDENCE_STAGING_PREFLIGHT_VERSION,
        "run_id": materialization.run_id,
        "paper_id": materialization.paper_id,
        "generated_at": generated_at,
        "materialization_artifact_sha256": normalized_materialization_sha,
        "materialization_artifact_digest": materialization.artifact_digest,
        "ledger_snapshot_sha256": normalized_ledger_sha,
        "ledger_snapshot_digest": ledger.snapshot_digest,
        "materialization_artifact": materialization,
        "ledger_snapshot": ledger,
        "semantic_contract": semantic_contract,
        "status": _preflight_status(items),
        "source_candidate_count": len(materialization.observation_candidates),
        "upstream_ontology_review_pending_cell_count": (
            materialization.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "preflight_items": items,
        "source_row_groups": groups,
        "preflight_artifact_digest": "sha256:" + "0" * 64,
        **_preflight_counts(items, groups),
    }
    provisional = OledReviewedEvidenceStagingPreflightArtifact.model_construct(
        **payload
    )
    payload["preflight_artifact_digest"] = (
        oled_reviewed_evidence_staging_preflight_artifact_digest(provisional)
    )
    return OledReviewedEvidenceStagingPreflightArtifact.model_validate(payload)


def oled_reviewed_evidence_ledger_snapshot_digest(
    snapshot: OledReviewedEvidenceLedgerSnapshot,
) -> str:
    payload = snapshot.model_dump(mode="json")
    payload.pop("snapshot_digest", None)
    return _stable_hash(payload)


def oled_reviewed_evidence_staging_preflight_artifact_digest(
    artifact: OledReviewedEvidenceStagingPreflightArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("preflight_artifact_digest", None)
    return _stable_hash(payload)


def _derive_preflight_items(
    materialization: OledObservationMaterializationCandidateArtifact,
    ledger: OledReviewedEvidenceLedgerSnapshot,
    semantic_contract: OledReviewedEvidenceSemanticContractSnapshot,
) -> list[OledReviewedEvidencePreflightItem]:
    if materialization.device_only_cell_count != 0 or any(
        candidate.canonical_observation.layer == OledCausalLayer.DEVICE
        for candidate in materialization.observation_candidates
    ):
        raise ValueError("PR-Q contains device-only observation candidates")
    items = [
        _derive_preflight_item(candidate, materialization, ledger, semantic_contract)
        for candidate in materialization.observation_candidates
    ]
    return sorted(items, key=lambda item: item.preflight_item_id)


def _derive_preflight_item(
    candidate: OledObservationMaterializationCandidateItem,
    materialization: OledObservationMaterializationCandidateArtifact,
    ledger: OledReviewedEvidenceLedgerSnapshot,
    semantic_contract: OledReviewedEvidenceSemanticContractSnapshot,
) -> OledReviewedEvidencePreflightItem:
    source_claim_id = _source_claim_id(candidate)
    projection_id = _projection_id(candidate, source_claim_id, semantic_contract.contract_digest)
    conflict_key = _conflict_key(candidate)
    row_group_id = _source_row_group_id(
        staging_item_id=candidate.staging_item_id,
        identity_group_id=candidate.identity_group_id,
        selected_material_id=candidate.selected_existing_material_id,
        source_pdf_sha256=candidate.source_pdf_sha256,
        table_id=candidate.table_id,
        row_index=candidate.row_index,
    )
    live_entries = [
        entry
        for entry in ledger.entries
        if entry.status in {
            OledReviewedEvidenceLedgerEntryStatus.ACTIVE,
            OledReviewedEvidenceLedgerEntryStatus.QUARANTINED,
        }
    ]
    exact = [entry for entry in ledger.entries if entry.projection_id == projection_id]
    expected_projection_payload = _candidate_projection_payload(
        candidate=candidate,
        source_materialization_artifact_digest=materialization.artifact_digest,
        semantic_contract_digest=semantic_contract.contract_digest,
    )
    for entry in exact:
        if _ledger_projection_payload(entry) != expected_projection_payload:
            raise ValueError(
                "exact replay ledger projection payload does not match "
                "the exact PR-Q candidate"
            )
    same_claim = [entry for entry in live_entries if entry.source_claim_id == source_claim_id]
    same_conflict_key = [entry for entry in live_entries if entry.conflict_key == conflict_key]
    same_contract_conflicts = [
        entry
        for entry in same_conflict_key
        if entry.semantic_contract_digest == semantic_contract.contract_digest
    ]
    other_contract_conflicts = [
        entry
        for entry in same_conflict_key
        if entry.semantic_contract_digest != semantic_contract.contract_digest
    ]
    candidate_id_collisions = [
        entry
        for entry in ledger.entries
        if entry.source_candidate_id == candidate.candidate_id
        and entry.source_claim_id != source_claim_id
    ]
    if exact:
        disposition = OledReviewedEvidencePreflightDisposition.EXACT_REPLAY
        matching = exact
    elif same_claim:
        disposition = OledReviewedEvidencePreflightDisposition.REVISION_REQUIRES_REVIEW
        matching = same_claim
    elif candidate.comparison_context_status == OledComparisonContextStatus.INCOMPLETE:
        disposition = (
            OledReviewedEvidencePreflightDisposition.INCOMPLETE_CONTEXT_QUARANTINE
        )
        matching = []
    elif other_contract_conflicts:
        disposition = (
            OledReviewedEvidencePreflightDisposition
            .SEMANTIC_CONTRACT_MIGRATION_REQUIRED
        )
        matching = other_contract_conflicts
    elif same_contract_conflicts and all(
        _entry_value_signature(entry) == _candidate_value_signature(candidate)
        for entry in same_contract_conflicts
    ):
        disposition = OledReviewedEvidencePreflightDisposition.CONSISTENT_DUPLICATE_READY
        matching = same_contract_conflicts
    elif same_contract_conflicts:
        disposition = OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE
        matching = same_contract_conflicts
    else:
        disposition = OledReviewedEvidencePreflightDisposition.NEW_CLAIM_READY
        matching = []
    blocker_codes = {
        "missing_confidence_assessment",
        "scientific_consistency_not_reviewed",
    }
    if disposition == OledReviewedEvidencePreflightDisposition.INCOMPLETE_CONTEXT_QUARANTINE:
        blocker_codes.add("incomplete_comparison_context")
    if disposition == OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE:
        blocker_codes.add("unresolved_value_conflict")
    if disposition == OledReviewedEvidencePreflightDisposition.REVISION_REQUIRES_REVIEW:
        blocker_codes.add("unreviewed_projection_revision")
    if disposition == (
        OledReviewedEvidencePreflightDisposition.SEMANTIC_CONTRACT_MIGRATION_REQUIRED
    ):
        blocker_codes.add("semantic_contract_migration_required")
    payload: dict[str, Any] = {
        "preflight_item_id": f"reviewed-evidence-preflight:{projection_id.split(':', 1)[-1]}",
        "source_candidate": candidate,
        "source_claim_id": source_claim_id,
        "projection_id": projection_id,
        "conflict_key": conflict_key,
        "source_row_group_id": row_group_id,
        "semantic_contract_digest": semantic_contract.contract_digest,
        "disposition": disposition,
        "matching_entry_ids": sorted(entry.entry_id for entry in matching),
        "gold_blocker_codes": sorted(blocker_codes),
        "verification_facets": OledReviewedEvidenceVerificationFacets(),
        "comparison_ready": candidate.comparison_ready,
        **_disposition_flags(disposition),
        "candidate_id_collision_detected": bool(candidate_id_collisions),
        "reviewed_evidence_staged": False,
        "direct_admission_eligible": False,
        "gold_record_created": False,
        "preflight_item_digest": "sha256:" + "0" * 64,
    }
    provisional = OledReviewedEvidencePreflightItem.model_construct(**payload)
    payload["preflight_item_digest"] = _preflight_item_digest(provisional)
    return OledReviewedEvidencePreflightItem.model_validate(payload)


def _derive_source_row_groups(
    items: list[OledReviewedEvidencePreflightItem],
) -> list[OledReviewedEvidenceSourceRowGroup]:
    grouped: dict[str, list[OledReviewedEvidencePreflightItem]] = {}
    for item in items:
        grouped.setdefault(item.source_row_group_id, []).append(item)
    output: list[OledReviewedEvidenceSourceRowGroup] = []
    for group_id, members in sorted(grouped.items()):
        first = members[0].source_candidate
        if any(
            (
                member.source_candidate.staging_item_id,
                member.source_candidate.identity_group_id,
                member.source_candidate.selected_existing_material_id,
                member.source_candidate.source_pdf_sha256,
                member.source_candidate.table_id,
                member.source_candidate.row_index,
            )
            != (
                first.staging_item_id,
                first.identity_group_id,
                first.selected_existing_material_id,
                first.source_pdf_sha256,
                first.table_id,
                first.row_index,
            )
            for member in members
        ):
            raise ValueError("reviewed-evidence source-row grouping collision")
        payload: dict[str, Any] = {
            "source_row_group_id": group_id,
            "staging_item_id": first.staging_item_id,
            "identity_group_id": first.identity_group_id,
            "selected_material_id": first.selected_existing_material_id,
            "source_pdf_sha256": first.source_pdf_sha256,
            "table_id": first.table_id,
            "row_index": first.row_index,
            "preflight_item_ids": sorted(member.preflight_item_id for member in members),
            "observation_count": len(members),
            "group_digest": "sha256:" + "0" * 64,
        }
        provisional = OledReviewedEvidenceSourceRowGroup.model_construct(**payload)
        payload["group_digest"] = _source_row_group_digest(provisional)
        output.append(OledReviewedEvidenceSourceRowGroup.model_validate(payload))
    return output


def _preflight_counts(
    items: list[OledReviewedEvidencePreflightItem],
    groups: list[OledReviewedEvidenceSourceRowGroup],
) -> dict[str, int]:
    return {
        "source_row_group_count": len(groups),
        "ledger_write_count": sum(item.ledger_write_required for item in items),
        "exact_replay_count": sum(
            item.disposition == OledReviewedEvidencePreflightDisposition.EXACT_REPLAY
            for item in items
        ),
        "consistent_duplicate_count": sum(
            item.disposition
            == OledReviewedEvidencePreflightDisposition.CONSISTENT_DUPLICATE_READY
            for item in items
        ),
        "conflict_quarantine_count": sum(
            item.disposition
            == OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE
            for item in items
        ),
        "incomplete_context_quarantine_count": sum(
            item.disposition
            == OledReviewedEvidencePreflightDisposition.INCOMPLETE_CONTEXT_QUARANTINE
            for item in items
        ),
        "revision_review_count": sum(
            item.disposition
            == OledReviewedEvidencePreflightDisposition.REVISION_REQUIRES_REVIEW
            for item in items
        ),
        "semantic_contract_migration_count": sum(
            item.disposition
            == OledReviewedEvidencePreflightDisposition
            .SEMANTIC_CONTRACT_MIGRATION_REQUIRED
            for item in items
        ),
        "candidate_id_collision_count": sum(
            item.candidate_id_collision_detected for item in items
        ),
    }


def _preflight_status(
    items: list[OledReviewedEvidencePreflightItem],
) -> OledReviewedEvidenceStagingPreflightStatus:
    if any(item.manual_exception_review_required for item in items):
        return OledReviewedEvidenceStagingPreflightStatus.MANUAL_EXCEPTION_REVIEW_REQUIRED
    if any(item.ledger_write_required for item in items):
        return OledReviewedEvidenceStagingPreflightStatus.READY_FOR_LEDGER_WRITE
    return OledReviewedEvidenceStagingPreflightStatus.NO_LEDGER_CHANGES_REQUIRED


def _disposition_flags(
    disposition: OledReviewedEvidencePreflightDisposition,
) -> dict[str, bool]:
    return {
        "ledger_write_required": disposition
        in {
            OledReviewedEvidencePreflightDisposition.NEW_CLAIM_READY,
            OledReviewedEvidencePreflightDisposition.CONSISTENT_DUPLICATE_READY,
            OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE,
            OledReviewedEvidencePreflightDisposition.INCOMPLETE_CONTEXT_QUARANTINE,
        },
        "quarantine_on_write": disposition
        in {
            OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE,
            OledReviewedEvidencePreflightDisposition.INCOMPLETE_CONTEXT_QUARANTINE,
        },
        "manual_exception_review_required": disposition
        in {
            OledReviewedEvidencePreflightDisposition.VALUE_CONFLICT_QUARANTINE,
            OledReviewedEvidencePreflightDisposition.REVISION_REQUIRES_REVIEW,
            OledReviewedEvidencePreflightDisposition
            .SEMANTIC_CONTRACT_MIGRATION_REQUIRED,
        },
    }


def _source_claim_id(candidate: OledObservationMaterializationCandidateItem) -> str:
    return _source_claim_id_from_fields(
        source_pdf_sha256=candidate.source_pdf_sha256,
        source_cell_digest=candidate.source_cell_digest,
    )


def _projection_id(
    candidate: OledObservationMaterializationCandidateItem,
    source_claim_id: str,
    semantic_contract_digest: str,
) -> str:
    return _projection_id_from_fields(
        source_claim_id=source_claim_id,
        source_candidate_digest=candidate.candidate_digest,
        selected_material_id=candidate.selected_existing_material_id,
        registry_entry_digest=_stable_hash(
            candidate.selected_registry_entry.model_dump(mode="json")
        ),
        cell_disposition_digest=candidate.cell_disposition_digest,
        semantic_contract_digest=semantic_contract_digest,
    )


def _conflict_key(candidate: OledObservationMaterializationCandidateItem) -> str:
    canonical = candidate.canonical_observation
    return _conflict_key_from_fields(
        selected_material_id=candidate.selected_existing_material_id,
        property_id=canonical.property_id,
        target_layer=canonical.layer,
        comparison_context_status=canonical.comparison_context_status,
        comparison_context_hash=canonical.comparison_context_hash,
        comparison_context_missing_fields=sorted(
            canonical.comparison_context_missing_fields
        ),
    )


def _source_claim_id_from_fields(
    *,
    source_pdf_sha256: str,
    source_cell_digest: str,
) -> str:
    return _id_hash(
        "source-claim",
        {
            "source_pdf_sha256": source_pdf_sha256,
            "source_cell_digest": source_cell_digest,
        },
    )


def _projection_id_from_fields(
    *,
    source_claim_id: str,
    source_candidate_digest: str,
    selected_material_id: str,
    registry_entry_digest: str,
    cell_disposition_digest: str,
    semantic_contract_digest: str,
) -> str:
    return _id_hash(
        "semantic-projection",
        {
            "source_claim_id": source_claim_id,
            "source_candidate_digest": source_candidate_digest,
            "selected_material_id": selected_material_id,
            "registry_entry_digest": registry_entry_digest,
            "cell_disposition_digest": cell_disposition_digest,
            "semantic_contract_digest": semantic_contract_digest,
        },
    )


def _conflict_key_from_fields(
    *,
    selected_material_id: str,
    property_id: str,
    target_layer: OledCausalLayer,
    comparison_context_status: OledComparisonContextStatus,
    comparison_context_hash: str | None,
    comparison_context_missing_fields: list[str],
) -> str:
    return _id_hash(
        "observation-conflict",
        {
            "selected_material_id": selected_material_id,
            "property_id": property_id,
            "target_layer": target_layer.value,
            "comparison_context_status": comparison_context_status.value,
            "comparison_context_hash": comparison_context_hash,
            "comparison_context_missing_fields": sorted(
                comparison_context_missing_fields
            ),
        },
    )


def _source_row_group_id(
    *,
    staging_item_id: str,
    identity_group_id: str,
    selected_material_id: str,
    source_pdf_sha256: str,
    table_id: str,
    row_index: int,
) -> str:
    return _id_hash(
        "source-row-group",
        {
            "staging_item_id": staging_item_id,
            "identity_group_id": identity_group_id,
            "selected_material_id": selected_material_id,
            "source_pdf_sha256": source_pdf_sha256,
            "table_id": table_id,
            "row_index": row_index,
        },
    )


def _candidate_value_signature(
    candidate: OledObservationMaterializationCandidateItem,
) -> str:
    return _stable_json(
        {
            "normalized_value": candidate.canonical_observation.normalized_value,
            "normalized_unit": candidate.canonical_observation.normalized_unit,
        }
    )


def _entry_value_signature(entry: OledReviewedEvidenceLedgerEntry) -> str:
    return _stable_json(
        {
            "normalized_value": entry.normalized_value,
            "normalized_unit": entry.normalized_unit,
        }
    )


def _candidate_projection_payload(
    *,
    candidate: OledObservationMaterializationCandidateItem,
    source_materialization_artifact_digest: str,
    semantic_contract_digest: str,
) -> dict[str, Any]:
    canonical = candidate.canonical_observation
    return {
        "source_materialization_artifact_digest": _normalize_sha256(
            source_materialization_artifact_digest,
            field_name="source_materialization_artifact_digest",
        ),
        "source_candidate_id": candidate.candidate_id,
        "source_candidate_digest": candidate.candidate_digest,
        "cell_disposition_digest": candidate.cell_disposition_digest,
        "source_pdf_sha256": candidate.source_pdf_sha256,
        "source_cell_digest": candidate.source_cell_digest,
        "table_id": candidate.table_id,
        "row_index": candidate.row_index,
        "column_index": candidate.column_index,
        "selected_material_id": candidate.selected_existing_material_id,
        "registry_entry_digest": _stable_hash(
            candidate.selected_registry_entry.model_dump(mode="json")
        ),
        "property_id": canonical.property_id,
        "target_layer": canonical.layer.value,
        "reported_value": candidate.property_observation.value,
        "reported_value_text": candidate.property_observation.reported_value_text,
        "reported_decimal_places": (
            candidate.property_observation.reported_decimal_places
        ),
        "reported_unit": candidate.property_observation.unit or "",
        "normalized_value": canonical.normalized_value,
        "normalized_unit": canonical.normalized_unit,
        "comparison_context_status": canonical.comparison_context_status.value,
        "comparison_context_hash": canonical.comparison_context_hash,
        "comparison_context_missing_fields": sorted(
            canonical.comparison_context_missing_fields
        ),
        "semantic_contract_digest": semantic_contract_digest,
    }


def _ledger_projection_payload(
    entry: OledReviewedEvidenceLedgerEntry,
) -> dict[str, Any]:
    return {
        "source_materialization_artifact_digest": (
            entry.source_materialization_artifact_digest
        ),
        "source_candidate_id": entry.source_candidate_id,
        "source_candidate_digest": entry.source_candidate_digest,
        "cell_disposition_digest": entry.cell_disposition_digest,
        "source_pdf_sha256": entry.source_pdf_sha256,
        "source_cell_digest": entry.source_cell_digest,
        "table_id": entry.table_id,
        "row_index": entry.row_index,
        "column_index": entry.column_index,
        "selected_material_id": entry.selected_material_id,
        "registry_entry_digest": entry.registry_entry_digest,
        "property_id": entry.property_id,
        "target_layer": entry.target_layer.value,
        "reported_value": entry.reported_value,
        "reported_value_text": entry.reported_value_text,
        "reported_decimal_places": entry.reported_decimal_places,
        "reported_unit": entry.reported_unit,
        "normalized_value": entry.normalized_value,
        "normalized_unit": entry.normalized_unit,
        "comparison_context_status": entry.comparison_context_status.value,
        "comparison_context_hash": entry.comparison_context_hash,
        "comparison_context_missing_fields": sorted(
            entry.comparison_context_missing_fields
        ),
        "semantic_contract_digest": entry.semantic_contract_digest,
    }


def _projection_payload_digest(payload: dict[str, Any]) -> str:
    return _stable_hash(payload)


def _semantic_contract_digest(
    snapshot: OledReviewedEvidenceSemanticContractSnapshot,
) -> str:
    payload = snapshot.model_dump(mode="json")
    payload.pop("contract_digest", None)
    return _stable_hash(payload)


def _ledger_entry_digest(entry: OledReviewedEvidenceLedgerEntry) -> str:
    payload = entry.model_dump(mode="json")
    payload.pop("entry_digest", None)
    return _stable_hash(payload)


def _preflight_item_digest(item: OledReviewedEvidencePreflightItem) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("preflight_item_digest", None)
    return _stable_hash(payload)


def _source_row_group_digest(group: OledReviewedEvidenceSourceRowGroup) -> str:
    payload = group.model_dump(mode="json")
    payload.pop("group_digest", None)
    return _stable_hash(payload)


def _id_hash(prefix: str, payload: Any) -> str:
    return f"{prefix}:{hashlib.sha256(_stable_json(payload).encode('utf-8')).hexdigest()}"


def _stable_hash(payload: Any) -> str:
    return f"sha256:{hashlib.sha256(_stable_json(payload).encode('utf-8')).hexdigest()}"


def _stable_json(payload: Any) -> str:
    return json.dumps(
        _canonicalize(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        converted = [_canonicalize(item) for item in value]
        return sorted(converted, key=_stable_json)
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


__all__ = [
    "OLED_REVIEWED_EVIDENCE_LEDGER_SNAPSHOT_VERSION",
    "OLED_REVIEWED_EVIDENCE_STAGING_PREFLIGHT_VERSION",
    "OledReviewedEvidenceLedgerEntry",
    "OledReviewedEvidenceLedgerEntryStatus",
    "OledReviewedEvidenceLedgerSnapshot",
    "OledReviewedEvidencePreflightDisposition",
    "OledReviewedEvidencePreflightItem",
    "OledReviewedEvidenceSemanticContractSnapshot",
    "OledReviewedEvidenceSourceRowGroup",
    "OledReviewedEvidenceStagingPreflightArtifact",
    "OledReviewedEvidenceStagingPreflightStatus",
    "OledReviewedEvidenceVerificationFacets",
    "build_empty_oled_reviewed_evidence_ledger_snapshot",
    "build_oled_reviewed_evidence_ledger_entry_from_candidate",
    "build_oled_reviewed_evidence_ledger_snapshot",
    "build_oled_reviewed_evidence_semantic_contract_snapshot",
    "build_oled_reviewed_evidence_staging_preflight_artifact",
    "oled_reviewed_evidence_ledger_snapshot_digest",
    "oled_reviewed_evidence_staging_preflight_artifact_digest",
]
