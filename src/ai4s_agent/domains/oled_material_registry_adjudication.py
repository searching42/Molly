from __future__ import annotations

import hashlib
import html
import json
from enum import Enum
from typing import Annotated, Any, Literal, Sequence

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
    OledMaterialRegistryConflictFinding,
    OledMaterialRegistryConflictKind,
    OledMaterialRegistryEntry,
    OledMaterialRegistryMatchStatus,
    OledMaterialRegistryResolutionRequestArtifact,
    OledMaterialRegistryResolutionRequestItem,
    render_oled_material_registry_resolution_request_markdown,
)
from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    OledSupplementaryMaterialIdentityProposeStructureCandidate,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_reviewer_text,
    _validate_timestamp,
)


OLED_MATERIAL_REGISTRY_DECISION_MANIFEST_VERSION = (
    "oled_material_registry_decision_manifest.v1"
)
OLED_MATERIAL_REGISTRY_ADJUDICATION_VERSION = (
    "oled_material_registry_adjudication.v1"
)


class OledMaterialRegistryDecision(str, Enum):
    MAP_TO_EXISTING_ENTITY = "map_to_existing_entity"
    PROPOSE_NEW_ENTITY = "propose_new_entity"
    KEEP_UNRESOLVED = "keep_unresolved"
    DEFER_CONFLICT = "defer_conflict"


class OledMaterialRegistryConflictReason(str, Enum):
    NONE = "none"
    DUPLICATE_STRUCTURAL_KEY = "duplicate_structural_key"
    STRUCTURAL_KEY_DISAGREEMENT = "structural_key_disagreement"
    REPORTED_NAME_COLLISION = "reported_name_collision"
    ENTITY_SCOPE_OR_CHEMISTRY_CONFLICT = "entity_scope_or_chemistry_conflict"


class OledMaterialRegistryAdjudicationStatus(str, Enum):
    EXISTING_ENTITY_MAPPINGS_READY_FOR_LATER_STAGING = (
        "existing_entity_mappings_ready_for_later_staging"
    )
    REVIEW_COMPLETE_WITH_PENDING_NEW_ENTITY_PROPOSALS = (
        "review_complete_with_pending_new_entity_proposals"
    )
    REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS = (
        "review_complete_with_unresolved_items"
    )
    NO_REGISTRY_ELIGIBLE_CANDIDATES = "no_registry_eligible_candidates"


class OledMaterialRegistryDecisionEntry(BaseModel):
    """One human decision bound to the complete PR-N lookup evidence roster."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    resolution_item_id: str
    resolution_item_digest: str
    decision: OledMaterialRegistryDecision
    selected_existing_material_id: str = ""
    conflict_reason: OledMaterialRegistryConflictReason = (
        OledMaterialRegistryConflictReason.NONE
    )
    reviewed_structural_candidate_material_ids: list[str] = Field(
        default_factory=list,
        max_length=100_000,
    )
    reviewed_alias_hit_digests: list[str] = Field(
        default_factory=list,
        max_length=100_000,
    )
    reviewed_registry_conflict_digests: list[str] = Field(
        default_factory=list,
        max_length=100_000,
    )
    review_note: str

    @field_validator("resolution_item_id")
    @classmethod
    def validate_resolution_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="resolution_item_id")

    @field_validator("resolution_item_digest")
    @classmethod
    def validate_resolution_item_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="resolution_item_digest")

    @field_validator("selected_existing_material_id")
    @classmethod
    def validate_optional_material_id(cls, value: str) -> str:
        if not value:
            return ""
        return _validate_bound_id(value, field_name="selected_existing_material_id")

    @field_validator("reviewed_structural_candidate_material_ids")
    @classmethod
    def validate_candidate_ids(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_bound_id(item, field_name="candidate material_id")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("reviewed structural candidate IDs must be sorted and unique")
        return clean

    @field_validator(
        "reviewed_alias_hit_digests",
        "reviewed_registry_conflict_digests",
    )
    @classmethod
    def validate_reviewed_digests(
        cls,
        value: list[str],
        info: Any,
    ) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name=str(info.field_name))
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="review_note",
            required=True,
            max_length=4_000,
        )

    @model_validator(mode="after")
    def validate_decision_shape(self) -> OledMaterialRegistryDecisionEntry:
        is_mapping = self.decision == (
            OledMaterialRegistryDecision.MAP_TO_EXISTING_ENTITY
        )
        is_deferred = self.decision == OledMaterialRegistryDecision.DEFER_CONFLICT
        if is_mapping != bool(self.selected_existing_material_id):
            raise ValueError(
                "existing-entity mapping requires exactly one selected material_id"
            )
        if is_deferred != (
            self.conflict_reason != OledMaterialRegistryConflictReason.NONE
        ):
            raise ValueError(
                "deferred conflict requires one conflict_reason and other decisions do not"
            )
        return self


class OledMaterialRegistryDecisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = OLED_MATERIAL_REGISTRY_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    request_artifact_sha256: str
    request_artifact_digest: str
    source_adjudication_sha256: str
    source_adjudication_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    reviewed_by: str
    reviewed_at: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledMaterialRegistryDecisionEntry] = Field(
        default_factory=list,
        max_length=1_000_000,
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected material Registry decision manifest version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "request_artifact_digest",
        "source_adjudication_sha256",
        "source_adjudication_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="reviewed_by",
            required=True,
            max_length=200,
        )

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="reviewed_at")

    @field_validator("decisions")
    @classmethod
    def validate_decision_order(
        cls,
        value: list[OledMaterialRegistryDecisionEntry],
    ) -> list[OledMaterialRegistryDecisionEntry]:
        item_ids = [entry.resolution_item_id for entry in value]
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("Registry decisions must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledMaterialRegistryDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("material Registry adjudication must be confirmed")
        return self


class OledMaterialRegistryNewEntityProposal(BaseModel):
    """A reviewed graph proposal without a Registry ID, name, alias, or write."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_resolution_item_id: str
    source_resolution_item_digest: str
    reported_subject_literal: str
    candidate_digest: str
    canonical_isomeric_smiles_candidate: str
    standard_inchi_candidate: str
    inchikey_candidate: str
    proposal_digest: str
    human_new_entity_proposal_recorded: StrictBool = True
    material_id_assigned: StrictBool = False
    canonical_name_assigned: StrictBool = False
    aliases_assigned: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False

    @field_validator("source_resolution_item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="source_resolution_item_id")

    @field_validator(
        "source_resolution_item_digest",
        "candidate_digest",
        "proposal_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "reported_subject_literal",
        "canonical_isomeric_smiles_candidate",
        "standard_inchi_candidate",
        "inchikey_candidate",
    )
    @classmethod
    def validate_bounded_literals(cls, value: str, info: Any) -> str:
        if not isinstance(value, str) or not value or len(value) > 20_000:
            raise ValueError(f"{info.field_name} is required and bounded")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError(f"{info.field_name} contains control text")
        return value

    @model_validator(mode="after")
    def validate_proposal_integrity(self) -> OledMaterialRegistryNewEntityProposal:
        fixed_false = (
            "material_id_assigned",
            "canonical_name_assigned",
            "aliases_assigned",
            "registry_entry_created",
            "registry_written",
        )
        if not self.human_new_entity_proposal_recorded or any(
            getattr(self, field_name) for field_name in fixed_false
        ):
            raise ValueError("new Registry entity proposal crossed its boundary")
        if _new_entity_proposal_digest(self) != self.proposal_digest:
            raise ValueError("new Registry entity proposal digest mismatch")
        return self


class OledMaterialRegistryAdjudicatedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_item: OledMaterialRegistryResolutionRequestItem
    decision_entry: OledMaterialRegistryDecisionEntry
    reviewed_registry_conflict_findings: list[
        OledMaterialRegistryConflictFinding
    ] = Field(default_factory=list, max_length=100_000)
    selected_registry_entry: OledMaterialRegistryEntry | None = None
    new_entity_proposal: OledMaterialRegistryNewEntityProposal | None = None
    adjudicated_item_digest: str
    existing_registry_entity_mapped: StrictBool
    new_registry_entity_proposed: StrictBool
    kept_unresolved: StrictBool
    conflict_deferred: StrictBool
    material_identity_resolved: StrictBool
    canonical_material_id_assigned: StrictBool
    cross_paper_identity_mapping_human_confirmed: StrictBool
    eligible_for_later_observation_staging: StrictBool
    automatic_candidate_merge: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    observations_materialized: StrictBool = False

    @field_validator("adjudicated_item_digest")
    @classmethod
    def validate_item_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="adjudicated_item_digest")

    @field_validator("reviewed_registry_conflict_findings")
    @classmethod
    def validate_conflict_finding_order(
        cls,
        value: list[OledMaterialRegistryConflictFinding],
    ) -> list[OledMaterialRegistryConflictFinding]:
        digests = [finding.finding_digest for finding in value]
        if digests != sorted(digests) or len(digests) != len(set(digests)):
            raise ValueError("reviewed Registry conflict findings must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_item_integrity(self) -> OledMaterialRegistryAdjudicatedItem:
        if (
            self.decision_entry.resolution_item_id
            != self.request_item.resolution_item_id
            or self.decision_entry.resolution_item_digest
            != self.request_item.resolution_item_digest
        ):
            raise ValueError("Registry adjudicated item binding mismatch")
        _validate_decision_for_request_item(
            self.request_item,
            self.decision_entry,
            conflict_findings=self.reviewed_registry_conflict_findings,
        )
        expected = _derived_item_flags(self.decision_entry.decision)
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"Registry adjudicated {field_name} mismatch")
        is_mapping = self.decision_entry.decision == (
            OledMaterialRegistryDecision.MAP_TO_EXISTING_ENTITY
        )
        if is_mapping != (self.selected_registry_entry is not None):
            raise ValueError("Registry mapped item selected-entry presence mismatch")
        if self.selected_registry_entry is not None:
            _validate_selected_entry_for_request_item(
                self.request_item,
                self.decision_entry.selected_existing_material_id,
                self.selected_registry_entry,
            )
        is_new = self.decision_entry.decision == (
            OledMaterialRegistryDecision.PROPOSE_NEW_ENTITY
        )
        if is_new != (self.new_entity_proposal is not None):
            raise ValueError("new Registry proposal presence mismatch")
        if self.new_entity_proposal is not None:
            _validate_new_entity_proposal_for_item(
                self.request_item,
                self.new_entity_proposal,
            )
        fixed_false = (
            "automatic_candidate_merge",
            "registry_entry_created",
            "registry_written",
            "observations_materialized",
        )
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("Registry adjudicated item crossed its write boundary")
        if _adjudicated_item_digest(self) != self.adjudicated_item_digest:
            raise ValueError("Registry adjudicated item digest mismatch")
        return self


class OledMaterialRegistryAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_MATERIAL_REGISTRY_ADJUDICATION_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_artifact_digest: str
    source_adjudication_sha256: str
    source_adjudication_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    reviewed_by: str
    reviewed_at: str
    status: OledMaterialRegistryAdjudicationStatus
    upstream_pr_m_review_item_count: Annotated[StrictInt, Field(ge=1)]
    source_registry_eligible_group_count: Annotated[StrictInt, Field(ge=0)]
    source_registry_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[
        StrictInt,
        Field(ge=0),
    ]
    review_item_count: Annotated[StrictInt, Field(ge=0)]
    existing_entity_mapping_count: Annotated[StrictInt, Field(ge=0)]
    new_entity_proposal_count: Annotated[StrictInt, Field(ge=0)]
    kept_unresolved_count: Annotated[StrictInt, Field(ge=0)]
    conflict_deferred_count: Annotated[StrictInt, Field(ge=0)]
    existing_entity_mapping_cell_count: Annotated[StrictInt, Field(ge=0)]
    new_entity_proposal_cell_count: Annotated[StrictInt, Field(ge=0)]
    kept_unresolved_cell_count: Annotated[StrictInt, Field(ge=0)]
    conflict_deferred_cell_count: Annotated[StrictInt, Field(ge=0)]
    later_observation_staging_eligible_group_count: Annotated[
        StrictInt,
        Field(ge=0),
    ]
    later_observation_staging_eligible_cell_count: Annotated[
        StrictInt,
        Field(ge=0),
    ]
    device_only_cell_count: Literal[0] = 0
    adjudicated_items: list[OledMaterialRegistryAdjudicatedItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    adjudication_artifact_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    joint_request_and_decision_binding_validated: StrictBool = True
    complete_decision_coverage_validated: StrictBool = True
    complete_structural_candidate_acknowledgement_validated: StrictBool = True
    complete_alias_hit_acknowledgement_validated: StrictBool = True
    complete_registry_conflict_acknowledgement_validated: StrictBool = True
    selected_existing_entries_replayed_from_snapshot: StrictBool = True
    new_entity_proposals_derived_from_accepted_candidates: StrictBool = True
    human_registry_review_completed: StrictBool = True
    standalone_request_bytes_revalidation_supported: StrictBool = False
    pr_m_upstream_chain_revalidated: StrictBool = False
    source_pdf_read: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    aliases_mutated: StrictBool = False
    observations_materialized: StrictBool = False
    schema_candidates_created: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    direct_admission_eligible: StrictBool = False
    training_eligible: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_ADJUDICATION_VERSION:
            raise ValueError("unexpected material Registry adjudication version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at", "reviewed_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="reviewed_by",
            required=True,
            max_length=200,
        )

    @field_validator(
        "request_artifact_sha256",
        "request_artifact_digest",
        "source_adjudication_sha256",
        "source_adjudication_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "adjudication_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("adjudicated_items")
    @classmethod
    def validate_item_order(
        cls,
        value: list[OledMaterialRegistryAdjudicatedItem],
    ) -> list[OledMaterialRegistryAdjudicatedItem]:
        item_ids = [item.request_item.resolution_item_id for item in value]
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("Registry adjudicated items must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledMaterialRegistryAdjudicationArtifact:
        if _parse_timestamp(self.reviewed_at) > _parse_timestamp(self.generated_at):
            raise ValueError("Registry adjudication predates human review")
        counts = _adjudication_counts(self.adjudicated_items)
        expected_counts = {
            "review_item_count": "item_count",
            "existing_entity_mapping_count": "mapping_count",
            "new_entity_proposal_count": "new_count",
            "kept_unresolved_count": "unresolved_count",
            "conflict_deferred_count": "deferred_count",
            "existing_entity_mapping_cell_count": "mapping_cells",
            "new_entity_proposal_cell_count": "new_cells",
            "kept_unresolved_cell_count": "unresolved_cells",
            "conflict_deferred_cell_count": "deferred_cells",
            "later_observation_staging_eligible_group_count": "mapping_count",
            "later_observation_staging_eligible_cell_count": "mapping_cells",
        }
        for field_name, count_name in expected_counts.items():
            if getattr(self, field_name) != counts[count_name]:
                raise ValueError(f"Registry adjudication {field_name} mismatch")
        partitioned_cells = (
            counts["mapping_cells"]
            + counts["new_cells"]
            + counts["unresolved_cells"]
            + counts["deferred_cells"]
        )
        if (
            self.source_registry_eligible_group_count != counts["item_count"]
            or self.source_registry_eligible_group_count != self.review_item_count
            or self.source_registry_eligible_cell_count != partitioned_cells
            or self.upstream_pr_m_review_item_count
            < self.source_registry_eligible_group_count
        ):
            raise ValueError("Registry adjudication upstream coverage mismatch")
        if self.status != _adjudication_status(counts):
            raise ValueError("Registry adjudication status mismatch")
        if self.device_only_cell_count != 0:
            raise ValueError("device-only cells must remain outside Registry adjudication")
        fixed_true = (
            "review_only",
            "offline_only",
            "joint_request_and_decision_binding_validated",
            "complete_decision_coverage_validated",
            "complete_structural_candidate_acknowledgement_validated",
            "complete_alias_hit_acknowledgement_validated",
            "complete_registry_conflict_acknowledgement_validated",
            "selected_existing_entries_replayed_from_snapshot",
            "new_entity_proposals_derived_from_accepted_candidates",
            "human_registry_review_completed",
        )
        fixed_false = (
            "standalone_request_bytes_revalidation_supported",
            "pr_m_upstream_chain_revalidated",
            "source_pdf_read",
            "automatic_candidate_merge",
            "registry_entry_created",
            "registry_written",
            "aliases_mutated",
            "observations_materialized",
            "schema_candidates_created",
            "reviewed_evidence_staging",
            "direct_admission_eligible",
            "training_eligible",
            "gold_records_created",
            "dataset_written",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true) or any(
            getattr(self, field_name) for field_name in fixed_false
        ):
            raise ValueError("Registry adjudication crossed its boundary")
        reconstructed_manifest = OledMaterialRegistryDecisionManifest(
            schema_version=OLED_MATERIAL_REGISTRY_DECISION_MANIFEST_VERSION,
            run_id=self.run_id,
            paper_id=self.paper_id,
            request_artifact_sha256=self.request_artifact_sha256,
            request_artifact_digest=self.request_artifact_digest,
            source_adjudication_sha256=self.source_adjudication_sha256,
            source_adjudication_digest=self.source_adjudication_digest,
            registry_snapshot_sha256=self.registry_snapshot_sha256,
            registry_snapshot_digest=self.registry_snapshot_digest,
            reviewed_by=self.reviewed_by,
            reviewed_at=self.reviewed_at,
            adjudication_confirmed=True,
            decisions=[item.decision_entry for item in self.adjudicated_items],
        )
        if oled_material_registry_decision_manifest_digest(
            reconstructed_manifest
        ) != self.decision_manifest_digest:
            raise ValueError("Registry adjudication decision manifest digest mismatch")
        if oled_material_registry_adjudication_artifact_digest(self) != (
            self.adjudication_artifact_digest
        ):
            raise ValueError("Registry adjudication artifact digest mismatch")
        return self


def validate_oled_material_registry_decisions(
    *,
    request: OledMaterialRegistryResolutionRequestArtifact,
    request_artifact_sha256: str,
    decision_manifest: OledMaterialRegistryDecisionManifest,
) -> None:
    resolution_request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    manifest = OledMaterialRegistryDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    expected_request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    expected_bindings = {
        "run_id": resolution_request.run_id,
        "paper_id": resolution_request.paper_id,
        "request_artifact_sha256": expected_request_sha,
        "request_artifact_digest": resolution_request.request_artifact_digest,
        "source_adjudication_sha256": resolution_request.source_adjudication_sha256,
        "source_adjudication_digest": resolution_request.source_adjudication_digest,
        "registry_snapshot_sha256": resolution_request.registry_snapshot_sha256,
        "registry_snapshot_digest": resolution_request.registry_snapshot_digest,
    }
    for field_name, expected in expected_bindings.items():
        if getattr(manifest, field_name) != expected:
            raise ValueError(f"Registry decision manifest {field_name} mismatch")
    if _parse_timestamp(manifest.reviewed_at) < _parse_timestamp(
        resolution_request.generated_at
    ):
        raise ValueError("Registry decision manifest predates its request")
    items_by_id = {
        item.resolution_item_id: item for item in resolution_request.resolution_items
    }
    conflict_findings_by_digest = {
        finding.finding_digest: finding
        for finding in resolution_request.registry_conflict_findings
    }
    decisions_by_id = {
        entry.resolution_item_id: entry for entry in manifest.decisions
    }
    if decisions_by_id.keys() != items_by_id.keys():
        raise ValueError("Registry decision coverage does not match the request")
    for item_id, item in items_by_id.items():
        entry = decisions_by_id[item_id]
        if entry.resolution_item_digest != item.resolution_item_digest:
            raise ValueError("Registry decision item digest mismatch")
        _validate_decision_for_request_item(
            item,
            entry,
            conflict_findings=[
                conflict_findings_by_digest[digest]
                for digest in item.related_registry_conflict_digests
            ],
        )


def build_oled_material_registry_adjudication_artifact(
    *,
    request: OledMaterialRegistryResolutionRequestArtifact,
    request_artifact_sha256: str,
    decision_manifest: OledMaterialRegistryDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledMaterialRegistryAdjudicationArtifact:
    resolution_request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    manifest = OledMaterialRegistryDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    validate_oled_material_registry_decisions(
        request=resolution_request,
        request_artifact_sha256=request_artifact_sha256,
        decision_manifest=manifest,
    )
    decisions_by_id = {
        entry.resolution_item_id: entry for entry in manifest.decisions
    }
    entries_by_id = {
        entry.material_id: entry for entry in resolution_request.registry_snapshot.entries
    }
    conflict_findings_by_digest = {
        finding.finding_digest: finding
        for finding in resolution_request.registry_conflict_findings
    }
    adjudicated_items = [
        _build_adjudicated_item(
            item,
            decisions_by_id[item.resolution_item_id],
            registry_entries_by_id=entries_by_id,
            registry_conflict_findings_by_digest=conflict_findings_by_digest,
        )
        for item in sorted(
            resolution_request.resolution_items,
            key=lambda request_item: request_item.resolution_item_id,
        )
    ]
    counts = _adjudication_counts(adjudicated_items)
    payload: dict[str, Any] = {
        "artifact_version": OLED_MATERIAL_REGISTRY_ADJUDICATION_VERSION,
        "run_id": resolution_request.run_id,
        "paper_id": resolution_request.paper_id,
        "generated_at": generated_at,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_artifact_digest": resolution_request.request_artifact_digest,
        "source_adjudication_sha256": resolution_request.source_adjudication_sha256,
        "source_adjudication_digest": resolution_request.source_adjudication_digest,
        "registry_snapshot_sha256": resolution_request.registry_snapshot_sha256,
        "registry_snapshot_digest": resolution_request.registry_snapshot_digest,
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": oled_material_registry_decision_manifest_digest(
            manifest
        ),
        "reviewed_by": manifest.reviewed_by,
        "reviewed_at": manifest.reviewed_at,
        "status": _adjudication_status(counts),
        "upstream_pr_m_review_item_count": resolution_request.source_review_item_count,
        "source_registry_eligible_group_count": (
            resolution_request.registry_eligible_group_count
        ),
        "source_registry_eligible_cell_count": (
            resolution_request.registry_eligible_cell_count
        ),
        "upstream_ontology_review_pending_cell_count": (
            resolution_request.source_adjudication
            .upstream_ontology_review_pending_cell_count
        ),
        "review_item_count": counts["item_count"],
        "existing_entity_mapping_count": counts["mapping_count"],
        "new_entity_proposal_count": counts["new_count"],
        "kept_unresolved_count": counts["unresolved_count"],
        "conflict_deferred_count": counts["deferred_count"],
        "existing_entity_mapping_cell_count": counts["mapping_cells"],
        "new_entity_proposal_cell_count": counts["new_cells"],
        "kept_unresolved_cell_count": counts["unresolved_cells"],
        "conflict_deferred_cell_count": counts["deferred_cells"],
        "later_observation_staging_eligible_group_count": counts["mapping_count"],
        "later_observation_staging_eligible_cell_count": counts["mapping_cells"],
        "device_only_cell_count": 0,
        "adjudicated_items": adjudicated_items,
        "adjudication_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryAdjudicationArtifact.model_construct(**payload)
    payload["adjudication_artifact_digest"] = (
        oled_material_registry_adjudication_artifact_digest(provisional)
    )
    return OledMaterialRegistryAdjudicationArtifact.model_validate(payload)


def validate_oled_material_registry_request_adjudication_chain(
    request: OledMaterialRegistryResolutionRequestArtifact,
    adjudication: OledMaterialRegistryAdjudicationArtifact,
) -> None:
    """Replay the semantic PR-N -> PR-O chain shared by downstream branches."""

    resolution_request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    registry_adjudication = OledMaterialRegistryAdjudicationArtifact.model_validate(
        adjudication.model_dump(mode="json")
    )
    if _parse_timestamp(registry_adjudication.reviewed_at) < _parse_timestamp(
        resolution_request.generated_at
    ):
        raise ValueError("PR-O human review predates the exact PR-N request")
    expected_bindings = {
        "run_id": resolution_request.run_id,
        "paper_id": resolution_request.paper_id,
        "request_artifact_digest": resolution_request.request_artifact_digest,
        "source_adjudication_sha256": (
            resolution_request.source_adjudication_sha256
        ),
        "source_adjudication_digest": (
            resolution_request.source_adjudication_digest
        ),
        "registry_snapshot_sha256": resolution_request.registry_snapshot_sha256,
        "registry_snapshot_digest": resolution_request.registry_snapshot_digest,
    }
    for field_name, expected in expected_bindings.items():
        if getattr(registry_adjudication, field_name) != expected:
            raise ValueError(f"PR-N/PR-O {field_name} binding mismatch")
    request_items = {
        item.resolution_item_id: item
        for item in resolution_request.resolution_items
    }
    adjudicated_items = {
        item.request_item.resolution_item_id: item
        for item in registry_adjudication.adjudicated_items
    }
    if request_items.keys() != adjudicated_items.keys():
        raise ValueError("PR-N/PR-O resolution item coverage mismatch")
    for item_id, request_item in request_items.items():
        embedded = adjudicated_items[item_id].request_item
        if embedded.model_dump(mode="json") != request_item.model_dump(mode="json"):
            raise ValueError("PR-O embedded request item differs from PR-N")
    snapshot_entries = {
        entry.material_id: entry
        for entry in resolution_request.registry_snapshot.entries
    }
    for adjudicated_item in registry_adjudication.adjudicated_items:
        selected = adjudicated_item.selected_registry_entry
        if selected is None:
            continue
        expected = snapshot_entries.get(selected.material_id)
        if expected is None or expected.model_dump(mode="json") != (
            selected.model_dump(mode="json")
        ):
            raise ValueError("PR-O selected Registry entry differs from PR-N snapshot")


def render_oled_material_registry_adjudication_review_markdown(
    request: OledMaterialRegistryResolutionRequestArtifact,
    *,
    request_artifact_sha256: str,
) -> str:
    resolution_request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    base = render_oled_material_registry_resolution_request_markdown(
        resolution_request,
        request_artifact_sha256=request_artifact_sha256,
    ).rstrip()
    lines = [
        base,
        "",
        "# PR-O human Registry decision instructions",
        "",
        "> Every structural candidate ID, alias-hit digest, and related conflict digest",
        "> listed for an item must be copied into the decision entry exactly. This is",
        "> acknowledgement coverage, not evidence that the aliases are scientifically valid.",
        "",
        "## Decision rules",
        "",
        "- `map_to_existing_entity`: select exactly one surfaced structural candidate;",
        "  unavailable for no-hit, duplicate-key, or conflicting-key items.",
        "- `propose_new_entity`: available only when there is no exact structural hit;",
        "  it does not assign a material ID, canonical name, alias, or Registry entry.",
        "- `keep_unresolved`: records insufficient evidence without excluding the source row.",
        "- `defer_conflict`: requires one explicit conflict reason and performs no merge.",
        "- Every decision requires a reviewer note. Registry and observation writes "
        "remain disabled.",
        "",
        "## Manifest binding values",
        "",
        f"- `schema_version`: `{OLED_MATERIAL_REGISTRY_DECISION_MANIFEST_VERSION}`",
        f"- `run_id`: `{_md(resolution_request.run_id)}`",
        f"- `paper_id`: `{_md(resolution_request.paper_id)}`",
        f"- `request_artifact_sha256`: "
        f"`{_normalize_sha256(request_artifact_sha256, field_name='request_artifact_sha256')}`",
        f"- `request_artifact_digest`: `{resolution_request.request_artifact_digest}`",
        f"- `source_adjudication_sha256`: "
        f"`{resolution_request.source_adjudication_sha256}`",
        f"- `source_adjudication_digest`: "
        f"`{resolution_request.source_adjudication_digest}`",
        f"- `registry_snapshot_sha256`: `{resolution_request.registry_snapshot_sha256}`",
        f"- `registry_snapshot_digest`: `{resolution_request.registry_snapshot_digest}`",
        "- `adjudication_confirmed`: `true`",
        "",
    ]
    conflict_findings_by_digest = {
        finding.finding_digest: finding
        for finding in resolution_request.registry_conflict_findings
    }
    for index, item in enumerate(resolution_request.resolution_items, start=1):
        candidate_ids = _structural_candidate_ids(item)
        alias_digests = sorted(
            hit.alias_hit_digest for hit in item.exact_alias_literal_hits
        )
        conflict_digests = item.related_registry_conflict_digests
        conflict_findings = [
            conflict_findings_by_digest[digest] for digest in conflict_digests
        ]
        allowed = sorted(decision.value for decision in _allowed_decisions(item))
        allowed_conflict_reasons = sorted(
            reason.value
            for reason in _allowed_conflict_reasons(item, conflict_findings)
        )
        lines.extend(
            [
                f"### D{index:02d}: `{_md(item.resolution_item_id)}`",
                "",
                f"- `resolution_item_digest`: `{item.resolution_item_digest}`",
                "- allowed `decision`: "
                + ", ".join(f"`{value}`" for value in allowed),
                "- `reviewed_structural_candidate_material_ids`: "
                + _md_list(candidate_ids),
                "- `reviewed_alias_hit_digests`: " + _md_list(alias_digests),
                "- `reviewed_registry_conflict_digests`: "
                + _md_list(conflict_digests),
                "- `selected_existing_material_id`: required only for "
                "`map_to_existing_entity`; otherwise empty string",
                "- allowed `conflict_reason` for `defer_conflict`: "
                + ", ".join(f"`{value}`" for value in allowed_conflict_reasons),
                "- `conflict_reason`: otherwise `none`",
                "- `review_note`: required",
                "",
            ]
        )
    if not resolution_request.resolution_items:
        lines.extend(
            [
                "No decision entries are required. Submit an empty `decisions` list with",
                "the exact bindings above and `adjudication_confirmed=true`.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def oled_material_registry_decision_manifest_digest(
    manifest: OledMaterialRegistryDecisionManifest,
) -> str:
    payload = manifest.model_dump(mode="json")
    payload["decisions"] = sorted(
        payload["decisions"],
        key=lambda item: item["resolution_item_id"],
    )
    return _stable_hash(payload)


def oled_material_registry_adjudication_artifact_digest(
    artifact: OledMaterialRegistryAdjudicationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("adjudication_artifact_digest", None)
    return _stable_hash(payload)


def _validate_decision_for_request_item(
    item: OledMaterialRegistryResolutionRequestItem,
    decision: OledMaterialRegistryDecisionEntry,
    *,
    conflict_findings: Sequence[OledMaterialRegistryConflictFinding] = (),
) -> None:
    if decision.decision not in _allowed_decisions(item):
        raise ValueError("Registry decision is not allowed for this lookup outcome")
    expected_candidate_ids = _structural_candidate_ids(item)
    if decision.reviewed_structural_candidate_material_ids != expected_candidate_ids:
        raise ValueError("Registry structural candidate acknowledgement mismatch")
    expected_alias_digests = sorted(
        hit.alias_hit_digest for hit in item.exact_alias_literal_hits
    )
    if decision.reviewed_alias_hit_digests != expected_alias_digests:
        raise ValueError("Registry alias-hit acknowledgement mismatch")
    if decision.reviewed_registry_conflict_digests != (
        item.related_registry_conflict_digests
    ):
        raise ValueError("Registry conflict acknowledgement mismatch")
    observed_conflict_digests = [finding.finding_digest for finding in conflict_findings]
    if observed_conflict_digests != item.related_registry_conflict_digests:
        raise ValueError("Registry conflict finding replay mismatch")
    if decision.decision == OledMaterialRegistryDecision.MAP_TO_EXISTING_ENTITY and (
        decision.selected_existing_material_id not in expected_candidate_ids
    ):
        raise ValueError("selected existing material_id is not a structural candidate")
    _validate_conflict_reason(item, decision, conflict_findings)


def _allowed_decisions(
    item: OledMaterialRegistryResolutionRequestItem,
) -> set[OledMaterialRegistryDecision]:
    always = {
        OledMaterialRegistryDecision.KEEP_UNRESOLVED,
        OledMaterialRegistryDecision.DEFER_CONFLICT,
    }
    if item.match_status == OledMaterialRegistryMatchStatus.NO_EXACT_STRUCTURAL_CANDIDATE:
        return always | {OledMaterialRegistryDecision.PROPOSE_NEW_ENTITY}
    if item.match_status in {
        OledMaterialRegistryMatchStatus.PARTIAL_STRUCTURAL_KEY_MATCH,
        OledMaterialRegistryMatchStatus.ONE_CONSISTENT_EXACT_STRUCTURAL_CANDIDATE,
    }:
        return always | {OledMaterialRegistryDecision.MAP_TO_EXISTING_ENTITY}
    return always


def _structural_candidate_ids(
    item: OledMaterialRegistryResolutionRequestItem,
) -> list[str]:
    return sorted(
        {
            *item.canonical_smiles_candidate_material_ids,
            *item.inchikey_candidate_material_ids,
        }
    )


def _build_adjudicated_item(
    item: OledMaterialRegistryResolutionRequestItem,
    decision: OledMaterialRegistryDecisionEntry,
    *,
    registry_entries_by_id: dict[str, OledMaterialRegistryEntry],
    registry_conflict_findings_by_digest: dict[
        str,
        OledMaterialRegistryConflictFinding,
    ],
) -> OledMaterialRegistryAdjudicatedItem:
    selected_entry = None
    if decision.decision == OledMaterialRegistryDecision.MAP_TO_EXISTING_ENTITY:
        selected_entry = registry_entries_by_id.get(
            decision.selected_existing_material_id
        )
        if selected_entry is None:
            raise ValueError("selected existing Registry entry is unavailable")
    new_proposal = None
    if decision.decision == OledMaterialRegistryDecision.PROPOSE_NEW_ENTITY:
        new_proposal = _build_new_entity_proposal(item)
    payload: dict[str, Any] = {
        "request_item": item,
        "decision_entry": decision,
        "reviewed_registry_conflict_findings": [
            registry_conflict_findings_by_digest[digest]
            for digest in item.related_registry_conflict_digests
        ],
        "selected_registry_entry": selected_entry,
        "new_entity_proposal": new_proposal,
        "adjudicated_item_digest": "sha256:" + "0" * 64,
        **_derived_item_flags(decision.decision),
    }
    provisional = OledMaterialRegistryAdjudicatedItem.model_construct(**payload)
    payload["adjudicated_item_digest"] = _adjudicated_item_digest(provisional)
    return OledMaterialRegistryAdjudicatedItem.model_validate(payload)


def _validate_conflict_reason(
    item: OledMaterialRegistryResolutionRequestItem,
    decision: OledMaterialRegistryDecisionEntry,
    conflict_findings: Sequence[OledMaterialRegistryConflictFinding],
) -> None:
    if decision.decision != OledMaterialRegistryDecision.DEFER_CONFLICT:
        return
    reason = decision.conflict_reason
    finding_kinds = {finding.finding_kind for finding in conflict_findings}
    if reason == OledMaterialRegistryConflictReason.DUPLICATE_STRUCTURAL_KEY:
        if not (
            item.match_status
            == OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY
            and finding_kinds
            & {
                OledMaterialRegistryConflictKind.DUPLICATE_CANONICAL_SMILES,
                OledMaterialRegistryConflictKind.DUPLICATE_INCHIKEY,
            }
        ):
            raise ValueError("duplicate-structural-key conflict reason lacks evidence")
    elif reason == OledMaterialRegistryConflictReason.STRUCTURAL_KEY_DISAGREEMENT:
        if item.match_status != (
            OledMaterialRegistryMatchStatus.CONFLICTING_STRUCTURAL_KEY_MATCHES
        ):
            raise ValueError("structural-key-disagreement reason lacks evidence")
    elif reason == OledMaterialRegistryConflictReason.REPORTED_NAME_COLLISION:
        if (
            OledMaterialRegistryConflictKind.DUPLICATE_REPORTED_NAME_LITERAL
            not in finding_kinds
        ):
            raise ValueError("reported-name-collision reason lacks evidence")


def _allowed_conflict_reasons(
    item: OledMaterialRegistryResolutionRequestItem,
    conflict_findings: Sequence[OledMaterialRegistryConflictFinding],
) -> set[OledMaterialRegistryConflictReason]:
    reasons = {
        OledMaterialRegistryConflictReason.ENTITY_SCOPE_OR_CHEMISTRY_CONFLICT
    }
    finding_kinds = {finding.finding_kind for finding in conflict_findings}
    if (
        item.match_status
        == OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY
        and finding_kinds
        & {
            OledMaterialRegistryConflictKind.DUPLICATE_CANONICAL_SMILES,
            OledMaterialRegistryConflictKind.DUPLICATE_INCHIKEY,
        }
    ):
        reasons.add(OledMaterialRegistryConflictReason.DUPLICATE_STRUCTURAL_KEY)
    if item.match_status == (
        OledMaterialRegistryMatchStatus.CONFLICTING_STRUCTURAL_KEY_MATCHES
    ):
        reasons.add(OledMaterialRegistryConflictReason.STRUCTURAL_KEY_DISAGREEMENT)
    if (
        OledMaterialRegistryConflictKind.DUPLICATE_REPORTED_NAME_LITERAL
        in finding_kinds
    ):
        reasons.add(OledMaterialRegistryConflictReason.REPORTED_NAME_COLLISION)
    return reasons


def _build_new_entity_proposal(
    item: OledMaterialRegistryResolutionRequestItem,
) -> OledMaterialRegistryNewEntityProposal:
    result = item.adjudicated_group.review_item.validated_result
    response = result.response_result
    chemistry = result.chemistry_validation
    if not isinstance(
        response,
        OledSupplementaryMaterialIdentityProposeStructureCandidate,
    ) or chemistry is None:
        raise ValueError("new Registry proposal lacks an accepted graph candidate")
    candidate = response.structure_candidate
    payload: dict[str, Any] = {
        "source_resolution_item_id": item.resolution_item_id,
        "source_resolution_item_digest": item.resolution_item_digest,
        "reported_subject_literal": result.bound_identity_group.reported_subject_text,
        "candidate_digest": item.candidate_digest,
        "canonical_isomeric_smiles_candidate": (
            candidate.canonical_isomeric_smiles_candidate
        ),
        "standard_inchi_candidate": chemistry.standard_inchi_candidate,
        "inchikey_candidate": candidate.inchikey_candidate,
        "proposal_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryNewEntityProposal.model_construct(**payload)
    payload["proposal_digest"] = _new_entity_proposal_digest(provisional)
    return OledMaterialRegistryNewEntityProposal.model_validate(payload)


def _validate_selected_entry_for_request_item(
    item: OledMaterialRegistryResolutionRequestItem,
    selected_material_id: str,
    selected_entry: OledMaterialRegistryEntry,
) -> None:
    if selected_entry.material_id != selected_material_id:
        raise ValueError("selected Registry entry material_id mismatch")
    result = item.adjudicated_group.review_item.validated_result
    response = result.response_result
    if not isinstance(
        response,
        OledSupplementaryMaterialIdentityProposeStructureCandidate,
    ):
        raise ValueError("mapped Registry item lacks a graph candidate")
    candidate = response.structure_candidate
    matched = False
    if selected_material_id in item.canonical_smiles_candidate_material_ids:
        matched = matched or (
            selected_entry.canonical_isomeric_smiles
            == candidate.canonical_isomeric_smiles_candidate
        )
    if selected_material_id in item.inchikey_candidate_material_ids:
        matched = matched or selected_entry.inchikey == candidate.inchikey_candidate
    if not matched:
        raise ValueError("selected Registry entry does not replay its structural hit")


def _validate_new_entity_proposal_for_item(
    item: OledMaterialRegistryResolutionRequestItem,
    proposal: OledMaterialRegistryNewEntityProposal,
) -> None:
    expected = _build_new_entity_proposal(item)
    if proposal.model_dump(mode="json") != expected.model_dump(mode="json"):
        raise ValueError("new Registry entity proposal changed")


def _derived_item_flags(
    decision: OledMaterialRegistryDecision,
) -> dict[str, bool]:
    mapped = decision == OledMaterialRegistryDecision.MAP_TO_EXISTING_ENTITY
    proposed = decision == OledMaterialRegistryDecision.PROPOSE_NEW_ENTITY
    unresolved = decision == OledMaterialRegistryDecision.KEEP_UNRESOLVED
    deferred = decision == OledMaterialRegistryDecision.DEFER_CONFLICT
    return {
        "existing_registry_entity_mapped": mapped,
        "new_registry_entity_proposed": proposed,
        "kept_unresolved": unresolved,
        "conflict_deferred": deferred,
        "material_identity_resolved": mapped,
        "canonical_material_id_assigned": mapped,
        "cross_paper_identity_mapping_human_confirmed": mapped,
        "eligible_for_later_observation_staging": mapped,
    }


def _adjudication_counts(
    items: Sequence[OledMaterialRegistryAdjudicatedItem],
) -> dict[str, int]:
    counts = {
        "item_count": len(items),
        "mapping_count": sum(item.existing_registry_entity_mapped for item in items),
        "new_count": sum(item.new_registry_entity_proposed for item in items),
        "unresolved_count": sum(item.kept_unresolved for item in items),
        "deferred_count": sum(item.conflict_deferred for item in items),
    }
    for key, flag in (
        ("mapping_cells", "existing_registry_entity_mapped"),
        ("new_cells", "new_registry_entity_proposed"),
        ("unresolved_cells", "kept_unresolved"),
        ("deferred_cells", "conflict_deferred"),
    ):
        counts[key] = sum(
            item.request_item.adjudicated_group.review_item.validated_result
            .bound_identity_group.identity_dependent_cell_count
            for item in items
            if getattr(item, flag)
        )
    if (
        counts["mapping_count"]
        + counts["new_count"]
        + counts["unresolved_count"]
        + counts["deferred_count"]
        != counts["item_count"]
    ):
        raise ValueError("Registry adjudication decision partition mismatch")
    return counts


def _adjudication_status(
    counts: dict[str, int],
) -> OledMaterialRegistryAdjudicationStatus:
    if counts["item_count"] == 0:
        return OledMaterialRegistryAdjudicationStatus.NO_REGISTRY_ELIGIBLE_CANDIDATES
    if counts["unresolved_count"] or counts["deferred_count"]:
        return (
            OledMaterialRegistryAdjudicationStatus
            .REVIEW_COMPLETE_WITH_UNRESOLVED_ITEMS
        )
    if counts["new_count"]:
        return (
            OledMaterialRegistryAdjudicationStatus
            .REVIEW_COMPLETE_WITH_PENDING_NEW_ENTITY_PROPOSALS
        )
    return (
        OledMaterialRegistryAdjudicationStatus
        .EXISTING_ENTITY_MAPPINGS_READY_FOR_LATER_STAGING
    )


def _new_entity_proposal_digest(
    proposal: OledMaterialRegistryNewEntityProposal,
) -> str:
    payload = proposal.model_dump(mode="json")
    payload.pop("proposal_digest", None)
    return _stable_hash(payload)


def _adjudicated_item_digest(item: OledMaterialRegistryAdjudicatedItem) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("adjudicated_item_digest", None)
    return _stable_hash(payload)


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _md(value: str) -> str:
    return html.escape(value, quote=True).replace("`", "&#96;")


def _md_list(values: Sequence[str]) -> str:
    if not values:
        return "`[]`"
    return "[" + ", ".join(f"`{_md(value)}`" for value in values) + "]"


__all__ = [
    "OLED_MATERIAL_REGISTRY_ADJUDICATION_VERSION",
    "OLED_MATERIAL_REGISTRY_DECISION_MANIFEST_VERSION",
    "OledMaterialRegistryAdjudicatedItem",
    "OledMaterialRegistryAdjudicationArtifact",
    "OledMaterialRegistryAdjudicationStatus",
    "OledMaterialRegistryConflictReason",
    "OledMaterialRegistryDecision",
    "OledMaterialRegistryDecisionEntry",
    "OledMaterialRegistryDecisionManifest",
    "OledMaterialRegistryNewEntityProposal",
    "build_oled_material_registry_adjudication_artifact",
    "oled_material_registry_adjudication_artifact_digest",
    "oled_material_registry_decision_manifest_digest",
    "render_oled_material_registry_adjudication_review_markdown",
    "validate_oled_material_registry_request_adjudication_chain",
    "validate_oled_material_registry_decisions",
]
