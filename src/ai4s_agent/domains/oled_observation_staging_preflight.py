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

from ai4s_agent.domains.oled_material_registry_adjudication import (
    OledMaterialRegistryAdjudicatedItem,
    OledMaterialRegistryAdjudicationArtifact,
    oled_material_registry_adjudication_artifact_digest,
    validate_oled_material_registry_request_adjudication_chain,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    OledMaterialRegistryResolutionRequestArtifact,
    oled_material_registry_resolution_request_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_candidate_request import (
    OledSupplementaryMaterialIdentityDependentCell,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)


OLED_OBSERVATION_STAGING_PREFLIGHT_VERSION = (
    "oled_observation_staging_preflight.v1"
)


class OledObservationStagingPreflightStatus(str, Enum):
    READY_FOR_EXACT_SOURCE_VALUE_REPLAY = "ready_for_exact_source_value_replay"
    NO_EXISTING_ENTITY_MAPPINGS = "no_existing_entity_mappings"


class OledObservationStagingPreflightItem(BaseModel):
    """One resolved material plus exact reviewed-cell references, not observations."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    staging_item_id: str
    resolution_item_id: str
    resolution_item_digest: str
    registry_adjudicated_item_digest: str
    identity_group_id: str
    identity_group_digest: str
    selected_existing_material_id: str
    selected_registry_entry: OledMaterialRegistryEntry
    reported_subject_text: str
    scope_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    table_id: str
    table_content_digest: str
    pdf_page_number_one_based: Annotated[StrictInt, Field(ge=1)]
    row_index: Annotated[StrictInt, Field(ge=0)]
    subject_column_index: Annotated[StrictInt, Field(ge=0)]
    identity_dependent_cell_count: Annotated[StrictInt, Field(ge=1)]
    identity_dependent_cells: list[
        OledSupplementaryMaterialIdentityDependentCell
    ] = Field(min_length=1, max_length=1_000_000)
    staging_item_digest: str
    eligible_for_exact_source_value_replay: StrictBool = True
    human_registry_mapping_confirmed: StrictBool = True
    selected_existing_registry_entry_replayed: StrictBool = True
    source_property_values_present: StrictBool = False
    material_id_attached_to_observations: StrictBool = False
    observations_materialized: StrictBool = False
    schema_candidates_created: StrictBool = False
    gold_records_created: StrictBool = False

    @field_validator(
        "staging_item_id",
        "resolution_item_id",
        "identity_group_id",
        "selected_existing_material_id",
        "scope_id",
        "table_id",
    )
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("reported_subject_text")
    @classmethod
    def validate_reported_subject_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 20_000:
            raise ValueError("reported_subject_text is required and bounded")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("reported_subject_text contains control text")
        return value

    @field_validator(
        "resolution_item_digest",
        "registry_adjudicated_item_digest",
        "identity_group_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
        "table_content_digest",
        "staging_item_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("identity_dependent_cells")
    @classmethod
    def validate_cell_order(
        cls,
        value: list[OledSupplementaryMaterialIdentityDependentCell],
    ) -> list[OledSupplementaryMaterialIdentityDependentCell]:
        order = [(cell.column_index, cell.source_cell_digest) for cell in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("observation staging cells must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_item_integrity(self) -> OledObservationStagingPreflightItem:
        if self.staging_item_id != f"observation-staging:{self.resolution_item_id}":
            raise ValueError("observation staging item_id mismatch")
        if (
            self.selected_registry_entry.material_id
            != self.selected_existing_material_id
        ):
            raise ValueError("observation staging selected Registry entry mismatch")
        if self.identity_dependent_cell_count != len(self.identity_dependent_cells):
            raise ValueError("observation staging dependent-cell count mismatch")
        if any(cell.row_index != self.row_index for cell in self.identity_dependent_cells):
            raise ValueError("observation staging cell moved to another row")
        if self.subject_column_index in {
            cell.column_index for cell in self.identity_dependent_cells
        }:
            raise ValueError("subject column cannot be staged as a property cell")
        fixed_true = (
            "eligible_for_exact_source_value_replay",
            "human_registry_mapping_confirmed",
            "selected_existing_registry_entry_replayed",
        )
        fixed_false = (
            "source_property_values_present",
            "material_id_attached_to_observations",
            "observations_materialized",
            "schema_candidates_created",
            "gold_records_created",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("observation staging item crossed its preflight boundary")
        if _observation_staging_item_digest(self) != self.staging_item_digest:
            raise ValueError("observation staging item digest mismatch")
        return self


class OledObservationStagingPreflightArtifact(BaseModel):
    """Exact PR-N + PR-O join that emits no property values or observations."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_OBSERVATION_STAGING_PREFLIGHT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    request_artifact_sha256: str
    request_artifact_digest: str
    registry_adjudication_sha256: str
    registry_adjudication_digest: str
    source_adjudication_sha256: str
    source_adjudication_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    resolution_request: OledMaterialRegistryResolutionRequestArtifact
    registry_adjudication: OledMaterialRegistryAdjudicationArtifact
    status: OledObservationStagingPreflightStatus
    source_resolution_item_count: Annotated[StrictInt, Field(ge=0)]
    source_adjudicated_item_count: Annotated[StrictInt, Field(ge=0)]
    staging_item_count: Annotated[StrictInt, Field(ge=0)]
    staging_cell_count: Annotated[StrictInt, Field(ge=0)]
    new_entity_proposal_excluded_group_count: Annotated[StrictInt, Field(ge=0)]
    new_entity_proposal_excluded_cell_count: Annotated[StrictInt, Field(ge=0)]
    unresolved_excluded_group_count: Annotated[StrictInt, Field(ge=0)]
    unresolved_excluded_cell_count: Annotated[StrictInt, Field(ge=0)]
    conflict_deferred_excluded_group_count: Annotated[StrictInt, Field(ge=0)]
    conflict_deferred_excluded_cell_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    staging_items: list[OledObservationStagingPreflightItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    preflight_artifact_digest: str
    preflight_only: StrictBool = True
    offline_only: StrictBool = True
    exact_request_bytes_bound: StrictBool = True
    exact_registry_adjudication_bytes_bound: StrictBool = True
    request_model_embedded_and_revalidated: StrictBool = True
    registry_adjudication_model_embedded_and_revalidated: StrictBool = True
    joint_request_and_adjudication_binding_revalidated: StrictBool = True
    complete_resolution_item_coverage_revalidated: StrictBool = True
    complete_mapping_coverage_validated: StrictBool = True
    complete_dependent_cell_coverage_validated: StrictBool = True
    selected_existing_entries_replayed: StrictBool = True
    source_value_replay_required: StrictBool = True
    standalone_input_bytes_revalidation_supported: StrictBool = False
    source_pdf_read: StrictBool = False
    raw_parsed_document_read: StrictBool = False
    source_property_values_present: StrictBool = False
    material_id_attached_to_observations: StrictBool = False
    observations_materialized: StrictBool = False
    schema_candidates_created: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
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
    def validate_artifact_version(cls, value: str) -> str:
        if value != OLED_OBSERVATION_STAGING_PREFLIGHT_VERSION:
            raise ValueError("unexpected observation staging preflight version")
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
        "request_artifact_sha256",
        "request_artifact_digest",
        "registry_adjudication_sha256",
        "registry_adjudication_digest",
        "source_adjudication_sha256",
        "source_adjudication_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
        "preflight_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("staging_items")
    @classmethod
    def validate_item_order(
        cls,
        value: list[OledObservationStagingPreflightItem],
    ) -> list[OledObservationStagingPreflightItem]:
        order = [item.staging_item_id for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("observation staging items must be sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledObservationStagingPreflightArtifact:
        request = self.resolution_request
        adjudication = self.registry_adjudication
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            adjudication.generated_at
        ):
            raise ValueError("observation staging preflight predates PR-O")
        expected_bindings = {
            "run_id": request.run_id,
            "paper_id": request.paper_id,
            "request_artifact_sha256": adjudication.request_artifact_sha256,
            "request_artifact_digest": request.request_artifact_digest,
            "registry_adjudication_digest": (
                adjudication.adjudication_artifact_digest
            ),
            "source_adjudication_sha256": request.source_adjudication_sha256,
            "source_adjudication_digest": request.source_adjudication_digest,
            "registry_snapshot_sha256": request.registry_snapshot_sha256,
            "registry_snapshot_digest": request.registry_snapshot_digest,
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(
                    f"observation staging preflight {field_name} mismatch"
                )
        if (
            oled_material_registry_resolution_request_artifact_digest(request)
            != request.request_artifact_digest
            or oled_material_registry_adjudication_artifact_digest(adjudication)
            != adjudication.adjudication_artifact_digest
        ):
            raise ValueError("observation staging embedded artifact digest mismatch")
        validate_oled_material_registry_request_adjudication_chain(
            request,
            adjudication,
        )
        expected_items = [
            _build_observation_staging_item(item)
            for item in adjudication.adjudicated_items
            if item.eligible_for_later_observation_staging
        ]
        if [item.model_dump(mode="json") for item in self.staging_items] != [
            item.model_dump(mode="json") for item in expected_items
        ]:
            raise ValueError("observation staging item derivation mismatch")
        expected_counts = {
            "source_resolution_item_count": request.resolution_item_count,
            "source_adjudicated_item_count": adjudication.review_item_count,
            "staging_item_count": adjudication.existing_entity_mapping_count,
            "staging_cell_count": adjudication.existing_entity_mapping_cell_count,
            "new_entity_proposal_excluded_group_count": (
                adjudication.new_entity_proposal_count
            ),
            "new_entity_proposal_excluded_cell_count": (
                adjudication.new_entity_proposal_cell_count
            ),
            "unresolved_excluded_group_count": adjudication.kept_unresolved_count,
            "unresolved_excluded_cell_count": (
                adjudication.kept_unresolved_cell_count
            ),
            "conflict_deferred_excluded_group_count": (
                adjudication.conflict_deferred_count
            ),
            "conflict_deferred_excluded_cell_count": (
                adjudication.conflict_deferred_cell_count
            ),
            "upstream_ontology_review_pending_cell_count": (
                adjudication.upstream_ontology_review_pending_cell_count
            ),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"observation staging {field_name} mismatch")
        if self.staging_item_count != len(self.staging_items) or (
            self.staging_cell_count
            != sum(item.identity_dependent_cell_count for item in self.staging_items)
        ):
            raise ValueError("observation staging aggregate count mismatch")
        expected_status = (
            OledObservationStagingPreflightStatus.READY_FOR_EXACT_SOURCE_VALUE_REPLAY
            if self.staging_items
            else OledObservationStagingPreflightStatus.NO_EXISTING_ENTITY_MAPPINGS
        )
        if self.status != expected_status or self.device_only_cell_count != 0:
            raise ValueError("observation staging status or device boundary mismatch")
        fixed_true = (
            "preflight_only",
            "offline_only",
            "exact_request_bytes_bound",
            "exact_registry_adjudication_bytes_bound",
            "request_model_embedded_and_revalidated",
            "registry_adjudication_model_embedded_and_revalidated",
            "joint_request_and_adjudication_binding_revalidated",
            "complete_resolution_item_coverage_revalidated",
            "complete_mapping_coverage_validated",
            "complete_dependent_cell_coverage_validated",
            "selected_existing_entries_replayed",
            "source_value_replay_required",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "source_pdf_read",
            "raw_parsed_document_read",
            "source_property_values_present",
            "material_id_attached_to_observations",
            "observations_materialized",
            "schema_candidates_created",
            "reviewed_evidence_staging",
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
            raise ValueError("observation staging crossed its preflight boundary")
        if oled_observation_staging_preflight_artifact_digest(self) != (
            self.preflight_artifact_digest
        ):
            raise ValueError("observation staging preflight artifact digest mismatch")
        return self


def validate_oled_observation_staging_preflight_inputs(
    *,
    request: OledMaterialRegistryResolutionRequestArtifact,
    request_artifact_sha256: str,
    registry_adjudication: OledMaterialRegistryAdjudicationArtifact,
) -> None:
    resolution_request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    adjudication = OledMaterialRegistryAdjudicationArtifact.model_validate(
        registry_adjudication.model_dump(mode="json")
    )
    actual_request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    if adjudication.request_artifact_sha256 != actual_request_sha:
        raise ValueError("PR-O is not bound to the exact supplied PR-N file")
    validate_oled_material_registry_request_adjudication_chain(
        resolution_request,
        adjudication,
    )


def build_oled_observation_staging_preflight_artifact(
    *,
    request: OledMaterialRegistryResolutionRequestArtifact,
    request_artifact_sha256: str,
    registry_adjudication: OledMaterialRegistryAdjudicationArtifact,
    registry_adjudication_sha256: str,
    generated_at: str,
) -> OledObservationStagingPreflightArtifact:
    resolution_request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        request.model_dump(mode="json")
    )
    adjudication = OledMaterialRegistryAdjudicationArtifact.model_validate(
        registry_adjudication.model_dump(mode="json")
    )
    validate_oled_observation_staging_preflight_inputs(
        request=resolution_request,
        request_artifact_sha256=request_artifact_sha256,
        registry_adjudication=adjudication,
    )
    items = [
        _build_observation_staging_item(item)
        for item in adjudication.adjudicated_items
        if item.eligible_for_later_observation_staging
    ]
    payload: dict[str, Any] = {
        "artifact_version": OLED_OBSERVATION_STAGING_PREFLIGHT_VERSION,
        "run_id": resolution_request.run_id,
        "paper_id": resolution_request.paper_id,
        "generated_at": generated_at,
        "request_artifact_sha256": _normalize_sha256(
            request_artifact_sha256,
            field_name="request_artifact_sha256",
        ),
        "request_artifact_digest": resolution_request.request_artifact_digest,
        "registry_adjudication_sha256": _normalize_sha256(
            registry_adjudication_sha256,
            field_name="registry_adjudication_sha256",
        ),
        "registry_adjudication_digest": adjudication.adjudication_artifact_digest,
        "source_adjudication_sha256": resolution_request.source_adjudication_sha256,
        "source_adjudication_digest": resolution_request.source_adjudication_digest,
        "registry_snapshot_sha256": resolution_request.registry_snapshot_sha256,
        "registry_snapshot_digest": resolution_request.registry_snapshot_digest,
        "resolution_request": resolution_request,
        "registry_adjudication": adjudication,
        "status": (
            OledObservationStagingPreflightStatus.READY_FOR_EXACT_SOURCE_VALUE_REPLAY
            if items
            else OledObservationStagingPreflightStatus.NO_EXISTING_ENTITY_MAPPINGS
        ),
        "source_resolution_item_count": resolution_request.resolution_item_count,
        "source_adjudicated_item_count": adjudication.review_item_count,
        "staging_item_count": adjudication.existing_entity_mapping_count,
        "staging_cell_count": adjudication.existing_entity_mapping_cell_count,
        "new_entity_proposal_excluded_group_count": (
            adjudication.new_entity_proposal_count
        ),
        "new_entity_proposal_excluded_cell_count": (
            adjudication.new_entity_proposal_cell_count
        ),
        "unresolved_excluded_group_count": adjudication.kept_unresolved_count,
        "unresolved_excluded_cell_count": adjudication.kept_unresolved_cell_count,
        "conflict_deferred_excluded_group_count": (
            adjudication.conflict_deferred_count
        ),
        "conflict_deferred_excluded_cell_count": (
            adjudication.conflict_deferred_cell_count
        ),
        "upstream_ontology_review_pending_cell_count": (
            adjudication.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "staging_items": items,
        "preflight_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledObservationStagingPreflightArtifact.model_construct(**payload)
    payload["preflight_artifact_digest"] = (
        oled_observation_staging_preflight_artifact_digest(provisional)
    )
    return OledObservationStagingPreflightArtifact.model_validate(payload)


def oled_observation_staging_preflight_artifact_digest(
    artifact: OledObservationStagingPreflightArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("preflight_artifact_digest", None)
    return _stable_hash(payload)


def _build_observation_staging_item(
    item: OledMaterialRegistryAdjudicatedItem,
) -> OledObservationStagingPreflightItem:
    if not item.eligible_for_later_observation_staging:
        raise ValueError("Registry adjudicated item is not staging eligible")
    selected = item.selected_registry_entry
    if selected is None:
        raise ValueError("staging-eligible item lacks a selected Registry entry")
    request_item = item.request_item
    group = request_item.adjudicated_group.review_item.validated_result.bound_identity_group
    payload: dict[str, Any] = {
        "staging_item_id": (
            f"observation-staging:{request_item.resolution_item_id}"
        ),
        "resolution_item_id": request_item.resolution_item_id,
        "resolution_item_digest": request_item.resolution_item_digest,
        "registry_adjudicated_item_digest": item.adjudicated_item_digest,
        "identity_group_id": group.identity_group_id,
        "identity_group_digest": group.identity_group_digest,
        "selected_existing_material_id": selected.material_id,
        "selected_registry_entry": selected,
        "reported_subject_text": group.reported_subject_text,
        "scope_id": group.scope_id,
        "source_id": group.source_id,
        "source_pdf_sha256": group.source_pdf_sha256,
        "parsed_document_sha256": group.parsed_document_sha256,
        "table_id": group.table_id,
        "table_content_digest": group.table_content_digest,
        "pdf_page_number_one_based": group.pdf_page_number_one_based,
        "row_index": group.row_index,
        "subject_column_index": group.subject_column_index,
        "identity_dependent_cell_count": group.identity_dependent_cell_count,
        "identity_dependent_cells": group.identity_dependent_cells,
        "staging_item_digest": "sha256:" + "0" * 64,
    }
    provisional = OledObservationStagingPreflightItem.model_construct(**payload)
    payload["staging_item_digest"] = _observation_staging_item_digest(provisional)
    return OledObservationStagingPreflightItem.model_validate(payload)


def _observation_staging_item_digest(
    item: OledObservationStagingPreflightItem,
) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("staging_item_digest", None)
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
    "OLED_OBSERVATION_STAGING_PREFLIGHT_VERSION",
    "OledObservationStagingPreflightArtifact",
    "OledObservationStagingPreflightItem",
    "OledObservationStagingPreflightStatus",
    "build_oled_observation_staging_preflight_artifact",
    "oled_observation_staging_preflight_artifact_digest",
    "validate_oled_observation_staging_preflight_inputs",
]
