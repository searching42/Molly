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

from ai4s_agent.domains.oled_material_registry_entry_proposal_request import (
    OledMaterialRegistryEntryProposalRequestArtifact,
    OledMaterialRegistryEntryProposalRequestItem,
    OledMaterialRegistryEntryReviewDecision,
    oled_material_registry_entry_proposal_request_artifact_digest,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    _validate_registry_name,
    build_oled_material_registry_entry,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_reviewer_text,
    _validate_timestamp,
)


OLED_MATERIAL_REGISTRY_ENTRY_DECISION_MANIFEST_VERSION = (
    "oled_material_registry_entry_decision_manifest.v1"
)
OLED_MATERIAL_REGISTRY_ENTRY_ADJUDICATION_VERSION = (
    "oled_material_registry_entry_adjudication.v1"
)


class OledMaterialRegistryEntryAdjudicationStatus(str, Enum):
    APPROVED_ENTRY_CANDIDATES_READY_FOR_WRITE_PREFLIGHT = (
        "approved_entry_candidates_ready_for_write_preflight"
    )
    REVIEW_COMPLETE_WITH_MIXED_DISPOSITIONS = (
        "review_complete_with_mixed_dispositions"
    )
    REVIEW_COMPLETE_WITHOUT_WRITE_CANDIDATES = (
        "review_complete_without_write_candidates"
    )
    NO_ENTRY_REVIEW_ITEMS = "no_entry_review_items"


class OledMaterialRegistryEntryDecisionEntry(BaseModel):
    """One human disposition bound to one exact PR-V review item."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    entry_review_item_id: str
    entry_review_item_digest: str
    review_contract_digest: str
    decision: OledMaterialRegistryEntryReviewDecision
    approved_material_id: str = ""
    approved_canonical_name: str = ""
    approved_aliases: list[str] = Field(default_factory=list, max_length=10_000)
    reviewed_existing_name_hit_digests: list[str] = Field(
        default_factory=list,
        max_length=100_000,
    )
    reviewed_snapshot_conflict_finding_digests: list[str] = Field(
        default_factory=list,
        max_length=100_000,
    )
    reviewed_batch_conflict_finding_digests: list[str] = Field(
        default_factory=list,
        max_length=100_000,
    )
    single_entity_scope_confirmed: StrictBool = False
    material_id_approved: StrictBool = False
    canonical_name_approved: StrictBool = False
    aliases_approved: StrictBool = False
    review_contract_acknowledged: StrictBool = False
    review_note: str

    @field_validator("entry_review_item_id")
    @classmethod
    def validate_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="entry_review_item_id")

    @field_validator("approved_material_id")
    @classmethod
    def validate_optional_material_id(cls, value: str) -> str:
        if not value:
            return ""
        return _validate_bound_id(value, field_name="approved_material_id")

    @field_validator("approved_canonical_name")
    @classmethod
    def validate_optional_canonical_name(cls, value: str) -> str:
        return _validate_registry_name(
            value,
            field_name="approved_canonical_name",
            required=False,
        )

    @field_validator("approved_aliases")
    @classmethod
    def validate_aliases(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_registry_name(item, field_name="approved_alias", required=True)
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("approved aliases must be sorted and unique")
        return clean

    @field_validator(
        "entry_review_item_digest",
        "review_contract_digest",
    )
    @classmethod
    def validate_bound_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "reviewed_existing_name_hit_digests",
        "reviewed_snapshot_conflict_finding_digests",
        "reviewed_batch_conflict_finding_digests",
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
    def validate_decision_shape(self) -> OledMaterialRegistryEntryDecisionEntry:
        if not self.review_contract_acknowledged:
            raise ValueError("Registry-entry review contract must be acknowledged")
        approved = self.decision == (
            OledMaterialRegistryEntryReviewDecision
            .APPROVE_LOCAL_REGISTRY_ENTRY_CANDIDATE
        )
        if approved:
            if not self.approved_material_id or not self.approved_canonical_name:
                raise ValueError("approved Registry-entry candidate lacks ID or name")
            if not all(
                (
                    self.single_entity_scope_confirmed,
                    self.material_id_approved,
                    self.canonical_name_approved,
                    self.aliases_approved,
                )
            ):
                raise ValueError("approved Registry-entry candidate lacks confirmations")
        elif any(
            (
                self.approved_material_id,
                self.approved_canonical_name,
                self.approved_aliases,
                self.single_entity_scope_confirmed,
                self.material_id_approved,
                self.canonical_name_approved,
                self.aliases_approved,
            )
        ):
            raise ValueError("non-approved Registry-entry decision carries approvals")
        return self


class OledMaterialRegistryEntryDecisionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = OLED_MATERIAL_REGISTRY_ENTRY_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    request_artifact_sha256: str
    request_artifact_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    review_contract_digest: str
    reviewed_by: str
    reviewed_at: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledMaterialRegistryEntryDecisionEntry] = Field(
        default_factory=list,
        max_length=1_000_000,
    )

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_ENTRY_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected Registry-entry decision manifest version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "request_artifact_sha256",
        "request_artifact_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
        "review_contract_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
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
    def validate_unique_decisions(
        cls,
        value: list[OledMaterialRegistryEntryDecisionEntry],
    ) -> list[OledMaterialRegistryEntryDecisionEntry]:
        item_ids = [item.entry_review_item_id for item in value]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("Registry-entry decision manifest repeats an item")
        return value

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledMaterialRegistryEntryDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("Registry-entry adjudication must be confirmed")
        return self


class OledMaterialRegistryApprovedEntryCandidate(BaseModel):
    """A human-approved candidate that is not yet assigned or written."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_entry_review_item_id: str
    source_entry_review_item_digest: str
    registry_entry: OledMaterialRegistryEntry
    candidate_digest: str
    human_single_entity_scope_confirmed: StrictBool = True
    material_id_human_approved: StrictBool = True
    canonical_name_human_approved: StrictBool = True
    aliases_human_approved: StrictBool = True
    material_id_reserved: StrictBool = False
    material_id_assigned: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False

    @field_validator("source_entry_review_item_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="source_entry_review_item_id")

    @field_validator(
        "source_entry_review_item_digest",
        "candidate_digest",
    )
    @classmethod
    def validate_candidate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_candidate_integrity(
        self,
    ) -> OledMaterialRegistryApprovedEntryCandidate:
        if self.candidate_digest != _approved_entry_candidate_digest(self):
            raise ValueError("approved Registry-entry candidate digest mismatch")
        fixed_true = (
            "human_single_entity_scope_confirmed",
            "material_id_human_approved",
            "canonical_name_human_approved",
            "aliases_human_approved",
        )
        fixed_false = (
            "material_id_reserved",
            "material_id_assigned",
            "registry_entry_created",
            "registry_written",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true) or any(
            getattr(self, field_name) for field_name in fixed_false
        ):
            raise ValueError("approved Registry-entry candidate crossed its boundary")
        return self


class OledMaterialRegistryEntryAdjudicatedItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    request_item: OledMaterialRegistryEntryProposalRequestItem
    decision_entry: OledMaterialRegistryEntryDecisionEntry
    approved_entry_candidate: OledMaterialRegistryApprovedEntryCandidate | None = None
    adjudicated_item_digest: str
    approved_local_registry_entry_candidate: StrictBool
    kept_unresolved: StrictBool
    entity_policy_deferred: StrictBool
    routed_to_existing_registry_resolution: StrictBool
    review_contract_acknowledged: StrictBool = True
    eligible_for_registry_write_preflight: StrictBool
    material_id_reserved: StrictBool = False
    material_id_assigned: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    observations_materialized: StrictBool = False

    @field_validator("adjudicated_item_digest")
    @classmethod
    def validate_item_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="adjudicated_item_digest")

    @model_validator(mode="after")
    def validate_item_integrity(
        self,
    ) -> OledMaterialRegistryEntryAdjudicatedItem:
        if (
            self.decision_entry.entry_review_item_id
            != self.request_item.entry_review_item_id
            or self.decision_entry.entry_review_item_digest
            != self.request_item.entry_review_item_digest
            or self.decision_entry.review_contract_digest
            != self.request_item.review_contract_digest
        ):
            raise ValueError("Registry-entry adjudicated item binding mismatch")
        expected_flags = _derived_item_flags(self.decision_entry.decision)
        for field_name, expected in expected_flags.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry-entry adjudicated {field_name} mismatch")
        if not self.review_contract_acknowledged:
            raise ValueError("Registry-entry adjudicated item lost contract acknowledgement")
        approved = self.approved_local_registry_entry_candidate
        if approved != (self.approved_entry_candidate is not None):
            raise ValueError("approved Registry-entry candidate presence mismatch")
        if self.approved_entry_candidate is not None:
            expected = _build_approved_entry_candidate(
                self.request_item,
                self.decision_entry,
            )
            if self.approved_entry_candidate.model_dump(mode="json") != (
                expected.model_dump(mode="json")
            ):
                raise ValueError("approved Registry-entry candidate derivation changed")
        fixed_false = (
            "material_id_reserved",
            "material_id_assigned",
            "registry_entry_created",
            "registry_written",
            "observations_materialized",
        )
        if any(getattr(self, field_name) for field_name in fixed_false):
            raise ValueError("Registry-entry adjudicated item crossed its write boundary")
        if _adjudicated_item_digest(self) != self.adjudicated_item_digest:
            raise ValueError("Registry-entry adjudicated item digest mismatch")
        return self


class OledMaterialRegistryEntryAdjudicationArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_MATERIAL_REGISTRY_ENTRY_ADJUDICATION_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_artifact_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    review_contract_digest: str
    request: OledMaterialRegistryEntryProposalRequestArtifact
    reviewed_by: str
    reviewed_at: str
    status: OledMaterialRegistryEntryAdjudicationStatus
    review_item_count: Annotated[StrictInt, Field(ge=0)]
    review_cell_count: Annotated[StrictInt, Field(ge=0)]
    approved_entry_candidate_count: Annotated[StrictInt, Field(ge=0)]
    approved_entry_candidate_cell_count: Annotated[StrictInt, Field(ge=0)]
    kept_unresolved_count: Annotated[StrictInt, Field(ge=0)]
    kept_unresolved_cell_count: Annotated[StrictInt, Field(ge=0)]
    entity_policy_deferred_count: Annotated[StrictInt, Field(ge=0)]
    entity_policy_deferred_cell_count: Annotated[StrictInt, Field(ge=0)]
    routed_existing_resolution_count: Annotated[StrictInt, Field(ge=0)]
    routed_existing_resolution_cell_count: Annotated[StrictInt, Field(ge=0)]
    registry_write_preflight_eligible_count: Annotated[StrictInt, Field(ge=0)]
    registry_write_preflight_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Annotated[StrictInt, Field(ge=0, le=0)] = 0
    adjudicated_items: list[OledMaterialRegistryEntryAdjudicatedItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    adjudication_artifact_digest: str
    review_only: StrictBool = True
    offline_only: StrictBool = True
    exact_request_bytes_bound_at_construction: StrictBool = True
    exact_decision_manifest_bytes_bound_at_construction: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    request_model_embedded_and_revalidated: StrictBool = True
    joint_request_and_decision_binding_validated: StrictBool = True
    complete_decision_coverage_validated: StrictBool = True
    complete_review_contract_acknowledgement_validated: StrictBool = True
    complete_existing_name_hint_acknowledgement_validated: StrictBool = True
    complete_snapshot_conflict_acknowledgement_validated: StrictBool = True
    complete_batch_conflict_acknowledgement_validated: StrictBool = True
    approved_candidates_deterministically_built: StrictBool = True
    human_registry_entry_review_completed: StrictBool = True
    material_id_reserved: StrictBool = False
    material_id_assigned: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    existing_registry_mutated: StrictBool = False
    observations_materialized: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_ENTRY_ADJUDICATION_VERSION:
            raise ValueError("unexpected Registry-entry adjudication version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_path_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("generated_at", "reviewed_at")
    @classmethod
    def validate_timestamps(cls, value: str, info: Any) -> str:
        return _validate_timestamp(value, field_name=str(info.field_name))

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        return _validate_reviewer_text(
            value,
            field_name="reviewed_by",
            required=True,
            max_length=200,
        )

    @field_validator(
        "request_artifact_sha256",
        "request_artifact_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
        "review_contract_digest",
        "adjudication_artifact_digest",
    )
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("adjudicated_items")
    @classmethod
    def validate_item_order(
        cls,
        value: list[OledMaterialRegistryEntryAdjudicatedItem],
    ) -> list[OledMaterialRegistryEntryAdjudicatedItem]:
        keys = [_adjudicated_item_source_sort_key(item) for item in value]
        item_ids = [item.request_item.entry_review_item_id for item in value]
        if keys != sorted(keys) or len(item_ids) != len(set(item_ids)):
            raise ValueError("Registry-entry adjudicated items are not source ordered")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledMaterialRegistryEntryAdjudicationArtifact:
        request = self.request
        if _parse_timestamp(self.reviewed_at) < _parse_timestamp(request.generated_at):
            raise ValueError("Registry-entry human review predates PR-V request")
        if _parse_timestamp(self.generated_at) < _parse_timestamp(self.reviewed_at):
            raise ValueError("Registry-entry adjudication predates human review")
        if (
            oled_material_registry_entry_proposal_request_artifact_digest(request)
            != request.proposal_request_artifact_digest
        ):
            raise ValueError("Registry-entry adjudication embedded request changed")
        expected_bindings = {
            "run_id": request.run_id,
            "paper_id": request.paper_id,
            "request_artifact_digest": request.proposal_request_artifact_digest,
            "registry_snapshot_sha256": request.registry_snapshot_sha256,
            "registry_snapshot_digest": request.registry_snapshot_digest,
            "review_contract_digest": request.review_contract.contract_digest,
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"Registry-entry adjudication {field_name} mismatch")
        reconstructed_manifest = _reconstruct_decision_manifest(self)
        if oled_material_registry_entry_decision_manifest_digest(
            reconstructed_manifest
        ) != self.decision_manifest_digest:
            raise ValueError("Registry-entry decision manifest digest mismatch")
        validate_oled_material_registry_entry_decisions(
            request=request,
            request_artifact_sha256=self.request_artifact_sha256,
            decision_manifest=reconstructed_manifest,
        )
        expected_items = _build_adjudicated_items(request, reconstructed_manifest)
        if [item.model_dump(mode="json") for item in self.adjudicated_items] != [
            item.model_dump(mode="json") for item in expected_items
        ]:
            raise ValueError("Registry-entry adjudicated item derivation changed")
        counts = _adjudication_counts(expected_items)
        expected_counts = {
            "review_item_count": "item_count",
            "review_cell_count": "cell_count",
            "approved_entry_candidate_count": "approved_count",
            "approved_entry_candidate_cell_count": "approved_cells",
            "kept_unresolved_count": "unresolved_count",
            "kept_unresolved_cell_count": "unresolved_cells",
            "entity_policy_deferred_count": "deferred_count",
            "entity_policy_deferred_cell_count": "deferred_cells",
            "routed_existing_resolution_count": "routed_count",
            "routed_existing_resolution_cell_count": "routed_cells",
            "registry_write_preflight_eligible_count": "approved_count",
            "registry_write_preflight_eligible_cell_count": "approved_cells",
        }
        for field_name, count_name in expected_counts.items():
            if getattr(self, field_name) != counts[count_name]:
                raise ValueError(f"Registry-entry adjudication {field_name} mismatch")
        if self.upstream_ontology_review_pending_cell_count != (
            request.upstream_ontology_review_pending_cell_count
        ):
            raise ValueError("Registry-entry upstream ontology count mismatch")
        if self.status != _adjudication_status(counts) or self.device_only_cell_count:
            raise ValueError("Registry-entry adjudication status or device boundary mismatch")
        fixed_true = (
            "review_only",
            "offline_only",
            "exact_request_bytes_bound_at_construction",
            "exact_decision_manifest_bytes_bound_at_construction",
            "request_model_embedded_and_revalidated",
            "joint_request_and_decision_binding_validated",
            "complete_decision_coverage_validated",
            "complete_review_contract_acknowledgement_validated",
            "complete_existing_name_hint_acknowledgement_validated",
            "complete_snapshot_conflict_acknowledgement_validated",
            "complete_batch_conflict_acknowledgement_validated",
            "approved_candidates_deterministically_built",
            "human_registry_entry_review_completed",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "material_id_reserved",
            "material_id_assigned",
            "registry_entry_created",
            "registry_written",
            "existing_registry_mutated",
            "observations_materialized",
            "reviewed_evidence_staging",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true) or any(
            getattr(self, field_name) for field_name in fixed_false
        ):
            raise ValueError("Registry-entry adjudication crossed its boundary")
        if oled_material_registry_entry_adjudication_artifact_digest(self) != (
            self.adjudication_artifact_digest
        ):
            raise ValueError("Registry-entry adjudication artifact digest mismatch")
        return self


def validate_oled_material_registry_entry_decisions(
    *,
    request: OledMaterialRegistryEntryProposalRequestArtifact,
    request_artifact_sha256: str,
    decision_manifest: OledMaterialRegistryEntryDecisionManifest,
) -> None:
    proposal_request = OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    manifest = OledMaterialRegistryEntryDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    expected_bindings = {
        "run_id": proposal_request.run_id,
        "paper_id": proposal_request.paper_id,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_artifact_digest": proposal_request.proposal_request_artifact_digest,
        "registry_snapshot_sha256": proposal_request.registry_snapshot_sha256,
        "registry_snapshot_digest": proposal_request.registry_snapshot_digest,
        "review_contract_digest": proposal_request.review_contract.contract_digest,
    }
    for field_name, expected in expected_bindings.items():
        if getattr(manifest, field_name) != expected:
            raise ValueError(f"Registry-entry decision manifest {field_name} mismatch")
    if _parse_timestamp(manifest.reviewed_at) < _parse_timestamp(
        proposal_request.generated_at
    ):
        raise ValueError("Registry-entry decision manifest predates its request")
    request_ids = [item.entry_review_item_id for item in proposal_request.entry_review_items]
    decision_ids = [item.entry_review_item_id for item in manifest.decisions]
    if decision_ids != request_ids:
        raise ValueError("Registry-entry decision coverage or source order mismatch")
    findings_by_item = _batch_finding_digests_by_item(proposal_request)
    for request_item, decision in zip(
        proposal_request.entry_review_items,
        manifest.decisions,
        strict=True,
    ):
        if (
            decision.entry_review_item_digest != request_item.entry_review_item_digest
            or decision.review_contract_digest != request_item.review_contract_digest
        ):
            raise ValueError("Registry-entry decision item binding mismatch")
        expected_name_hits = sorted(
            hit.alias_hit_digest
            for hit in request_item.source_adjudicated_item.request_item
            .exact_alias_literal_hits
        )
        expected_snapshot_conflicts = sorted(
            finding.finding_digest
            for finding in request_item.source_adjudicated_item
            .reviewed_registry_conflict_findings
        )
        expected_batch_findings = findings_by_item[request_item.entry_review_item_id]
        if decision.reviewed_existing_name_hit_digests != expected_name_hits:
            raise ValueError("Registry-entry name-hint acknowledgement mismatch")
        if (
            decision.reviewed_snapshot_conflict_finding_digests
            != expected_snapshot_conflicts
        ):
            raise ValueError("Registry-entry snapshot-conflict acknowledgement mismatch")
        if decision.reviewed_batch_conflict_finding_digests != expected_batch_findings:
            raise ValueError("Registry-entry batch-conflict acknowledgement mismatch")
        _validate_decision_for_request_item(request_item, decision)
        if expected_batch_findings and decision.decision == (
            OledMaterialRegistryEntryReviewDecision
            .APPROVE_LOCAL_REGISTRY_ENTRY_CANDIDATE
        ):
            raise ValueError("Registry-entry candidate with batch conflicts cannot be approved")


def build_oled_material_registry_entry_adjudication_artifact(
    *,
    request: OledMaterialRegistryEntryProposalRequestArtifact,
    request_artifact_sha256: str,
    decision_manifest: OledMaterialRegistryEntryDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledMaterialRegistryEntryAdjudicationArtifact:
    proposal_request = OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    manifest = OledMaterialRegistryEntryDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    validate_oled_material_registry_entry_decisions(
        request=proposal_request,
        request_artifact_sha256=request_artifact_sha256,
        decision_manifest=manifest,
    )
    items = _build_adjudicated_items(proposal_request, manifest)
    counts = _adjudication_counts(items)
    payload: dict[str, Any] = {
        "artifact_version": OLED_MATERIAL_REGISTRY_ENTRY_ADJUDICATION_VERSION,
        "run_id": proposal_request.run_id,
        "paper_id": proposal_request.paper_id,
        "generated_at": generated_at,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_artifact_digest": proposal_request.proposal_request_artifact_digest,
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": (
            oled_material_registry_entry_decision_manifest_digest(manifest)
        ),
        "registry_snapshot_sha256": proposal_request.registry_snapshot_sha256,
        "registry_snapshot_digest": proposal_request.registry_snapshot_digest,
        "review_contract_digest": proposal_request.review_contract.contract_digest,
        "request": proposal_request,
        "reviewed_by": manifest.reviewed_by,
        "reviewed_at": manifest.reviewed_at,
        "status": _adjudication_status(counts),
        "review_item_count": counts["item_count"],
        "review_cell_count": counts["cell_count"],
        "approved_entry_candidate_count": counts["approved_count"],
        "approved_entry_candidate_cell_count": counts["approved_cells"],
        "kept_unresolved_count": counts["unresolved_count"],
        "kept_unresolved_cell_count": counts["unresolved_cells"],
        "entity_policy_deferred_count": counts["deferred_count"],
        "entity_policy_deferred_cell_count": counts["deferred_cells"],
        "routed_existing_resolution_count": counts["routed_count"],
        "routed_existing_resolution_cell_count": counts["routed_cells"],
        "registry_write_preflight_eligible_count": counts["approved_count"],
        "registry_write_preflight_eligible_cell_count": counts["approved_cells"],
        "upstream_ontology_review_pending_cell_count": (
            proposal_request.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "adjudicated_items": items,
        "adjudication_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryEntryAdjudicationArtifact.model_construct(
        **payload
    )
    payload["adjudication_artifact_digest"] = (
        oled_material_registry_entry_adjudication_artifact_digest(provisional)
    )
    return OledMaterialRegistryEntryAdjudicationArtifact.model_validate(payload)


def oled_material_registry_entry_decision_manifest_digest(
    manifest: OledMaterialRegistryEntryDecisionManifest,
) -> str:
    return _stable_hash(manifest.model_dump(mode="json"))


def oled_material_registry_entry_adjudication_artifact_digest(
    artifact: OledMaterialRegistryEntryAdjudicationArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("adjudication_artifact_digest", None)
    return _stable_hash(payload)


def _build_adjudicated_items(
    request: OledMaterialRegistryEntryProposalRequestArtifact,
    manifest: OledMaterialRegistryEntryDecisionManifest,
) -> list[OledMaterialRegistryEntryAdjudicatedItem]:
    return [
        _build_adjudicated_item(request_item, decision)
        for request_item, decision in zip(
            request.entry_review_items,
            manifest.decisions,
            strict=True,
        )
    ]


def _build_adjudicated_item(
    request_item: OledMaterialRegistryEntryProposalRequestItem,
    decision: OledMaterialRegistryEntryDecisionEntry,
) -> OledMaterialRegistryEntryAdjudicatedItem:
    candidate = None
    if decision.decision == (
        OledMaterialRegistryEntryReviewDecision.APPROVE_LOCAL_REGISTRY_ENTRY_CANDIDATE
    ):
        candidate = _build_approved_entry_candidate(request_item, decision)
    flags = _derived_item_flags(decision.decision)
    payload: dict[str, Any] = {
        "request_item": request_item,
        "decision_entry": decision,
        "approved_entry_candidate": candidate,
        "adjudicated_item_digest": "sha256:" + "0" * 64,
        **flags,
    }
    provisional = OledMaterialRegistryEntryAdjudicatedItem.model_construct(**payload)
    payload["adjudicated_item_digest"] = _adjudicated_item_digest(provisional)
    return OledMaterialRegistryEntryAdjudicatedItem.model_validate(payload)


def _build_approved_entry_candidate(
    request_item: OledMaterialRegistryEntryProposalRequestItem,
    decision: OledMaterialRegistryEntryDecisionEntry,
) -> OledMaterialRegistryApprovedEntryCandidate:
    proposal = request_item.source_adjudicated_item.new_entity_proposal
    if proposal is None:
        raise ValueError("approved Registry-entry request item lacks a graph proposal")
    entry = build_oled_material_registry_entry(
        material_id=decision.approved_material_id,
        canonical_name=decision.approved_canonical_name,
        aliases=decision.approved_aliases,
        canonical_isomeric_smiles=proposal.canonical_isomeric_smiles_candidate,
    )
    if (
        entry.standard_inchi != proposal.standard_inchi_candidate
        or entry.inchikey != proposal.inchikey_candidate
    ):
        raise ValueError("approved Registry-entry chemistry differs from PR-O proposal")
    payload: dict[str, Any] = {
        "source_entry_review_item_id": request_item.entry_review_item_id,
        "source_entry_review_item_digest": request_item.entry_review_item_digest,
        "registry_entry": entry,
        "candidate_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryApprovedEntryCandidate.model_construct(**payload)
    payload["candidate_digest"] = _approved_entry_candidate_digest(provisional)
    return OledMaterialRegistryApprovedEntryCandidate.model_validate(payload)


def _validate_decision_for_request_item(
    request_item: OledMaterialRegistryEntryProposalRequestItem,
    decision: OledMaterialRegistryEntryDecisionEntry,
) -> None:
    if decision.decision != (
        OledMaterialRegistryEntryReviewDecision.APPROVE_LOCAL_REGISTRY_ENTRY_CANDIDATE
    ):
        return
    if decision.approved_material_id != request_item.proposed_material_id:
        raise ValueError("approved material ID differs from the exact PR-V proposal")
    if decision.approved_canonical_name != request_item.proposed_canonical_name:
        raise ValueError("approved canonical name differs from the exact PR-V proposal")
    if decision.approved_aliases != request_item.proposed_aliases:
        raise ValueError("approved aliases differ from the exact PR-V proposal")


def _derived_item_flags(
    decision: OledMaterialRegistryEntryReviewDecision,
) -> dict[str, bool]:
    approved = decision == (
        OledMaterialRegistryEntryReviewDecision.APPROVE_LOCAL_REGISTRY_ENTRY_CANDIDATE
    )
    return {
        "approved_local_registry_entry_candidate": approved,
        "kept_unresolved": decision
        == OledMaterialRegistryEntryReviewDecision.KEEP_UNRESOLVED,
        "entity_policy_deferred": decision
        == OledMaterialRegistryEntryReviewDecision.DEFER_ENTITY_POLICY,
        "routed_to_existing_registry_resolution": decision
        == (
            OledMaterialRegistryEntryReviewDecision
            .ROUTE_TO_EXISTING_REGISTRY_RESOLUTION
        ),
        "review_contract_acknowledged": True,
        "eligible_for_registry_write_preflight": approved,
    }


def _batch_finding_digests_by_item(
    request: OledMaterialRegistryEntryProposalRequestArtifact,
) -> dict[str, list[str]]:
    result = {item.entry_review_item_id: [] for item in request.entry_review_items}
    for finding in request.batch_conflict_findings:
        for item_id in finding.affected_entry_review_item_ids:
            if item_id not in result:
                raise ValueError("batch conflict references an unknown review item")
            result[item_id].append(finding.finding_digest)
    return {item_id: sorted(digests) for item_id, digests in result.items()}


def _adjudication_counts(
    items: Sequence[OledMaterialRegistryEntryAdjudicatedItem],
) -> dict[str, int]:
    result = {
        "item_count": len(items),
        "cell_count": 0,
        "approved_count": 0,
        "approved_cells": 0,
        "unresolved_count": 0,
        "unresolved_cells": 0,
        "deferred_count": 0,
        "deferred_cells": 0,
        "routed_count": 0,
        "routed_cells": 0,
    }
    for item in items:
        cells = item.request_item.identity_dependent_cell_count
        result["cell_count"] += cells
        if item.approved_local_registry_entry_candidate:
            result["approved_count"] += 1
            result["approved_cells"] += cells
        elif item.kept_unresolved:
            result["unresolved_count"] += 1
            result["unresolved_cells"] += cells
        elif item.entity_policy_deferred:
            result["deferred_count"] += 1
            result["deferred_cells"] += cells
        elif item.routed_to_existing_registry_resolution:
            result["routed_count"] += 1
            result["routed_cells"] += cells
        else:
            raise ValueError("Registry-entry adjudicated item lacks a disposition")
    return result


def _adjudication_status(
    counts: dict[str, int],
) -> OledMaterialRegistryEntryAdjudicationStatus:
    if counts["item_count"] == 0:
        return OledMaterialRegistryEntryAdjudicationStatus.NO_ENTRY_REVIEW_ITEMS
    if counts["approved_count"] == counts["item_count"]:
        return (
            OledMaterialRegistryEntryAdjudicationStatus
            .APPROVED_ENTRY_CANDIDATES_READY_FOR_WRITE_PREFLIGHT
        )
    if counts["approved_count"]:
        return (
            OledMaterialRegistryEntryAdjudicationStatus
            .REVIEW_COMPLETE_WITH_MIXED_DISPOSITIONS
        )
    return (
        OledMaterialRegistryEntryAdjudicationStatus
        .REVIEW_COMPLETE_WITHOUT_WRITE_CANDIDATES
    )


def _reconstruct_decision_manifest(
    artifact: OledMaterialRegistryEntryAdjudicationArtifact,
) -> OledMaterialRegistryEntryDecisionManifest:
    return OledMaterialRegistryEntryDecisionManifest(
        schema_version=OLED_MATERIAL_REGISTRY_ENTRY_DECISION_MANIFEST_VERSION,
        run_id=artifact.run_id,
        paper_id=artifact.paper_id,
        request_artifact_sha256=artifact.request_artifact_sha256,
        request_artifact_digest=artifact.request_artifact_digest,
        registry_snapshot_sha256=artifact.registry_snapshot_sha256,
        registry_snapshot_digest=artifact.registry_snapshot_digest,
        review_contract_digest=artifact.review_contract_digest,
        reviewed_by=artifact.reviewed_by,
        reviewed_at=artifact.reviewed_at,
        adjudication_confirmed=True,
        decisions=[item.decision_entry for item in artifact.adjudicated_items],
    )


def _approved_entry_candidate_digest(
    candidate: OledMaterialRegistryApprovedEntryCandidate,
) -> str:
    payload = candidate.model_dump(mode="json")
    payload.pop("candidate_digest", None)
    return _stable_hash(payload)


def _adjudicated_item_digest(
    item: OledMaterialRegistryEntryAdjudicatedItem,
) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("adjudicated_item_digest", None)
    return _stable_hash(payload)


def _adjudicated_item_source_sort_key(
    item: OledMaterialRegistryEntryAdjudicatedItem,
) -> tuple[str, str, int, str]:
    bound = (
        item.request_item.source_adjudicated_item.request_item.adjudicated_group
        .review_item.validated_result.bound_identity_group
    )
    return (
        bound.scope_id,
        bound.table_id,
        bound.row_index,
        bound.identity_group_id,
    )


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


__all__ = [
    "OLED_MATERIAL_REGISTRY_ENTRY_ADJUDICATION_VERSION",
    "OLED_MATERIAL_REGISTRY_ENTRY_DECISION_MANIFEST_VERSION",
    "OledMaterialRegistryApprovedEntryCandidate",
    "OledMaterialRegistryEntryAdjudicatedItem",
    "OledMaterialRegistryEntryAdjudicationArtifact",
    "OledMaterialRegistryEntryAdjudicationStatus",
    "OledMaterialRegistryEntryDecisionEntry",
    "OledMaterialRegistryEntryDecisionManifest",
    "build_oled_material_registry_entry_adjudication_artifact",
    "oled_material_registry_entry_adjudication_artifact_digest",
    "oled_material_registry_entry_decision_manifest_digest",
    "validate_oled_material_registry_entry_decisions",
]
