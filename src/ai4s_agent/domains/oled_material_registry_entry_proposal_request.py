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

from ai4s_agent.domains.oled_material_registry_adjudication import (
    OledMaterialRegistryAdjudicatedItem,
    OledMaterialRegistryAdjudicationArtifact,
    OledMaterialRegistryDecision,
    oled_material_registry_adjudication_artifact_digest,
    validate_oled_material_registry_request_adjudication_chain,
)
from ai4s_agent.domains.oled_material_registry_resolution_request import (
    OledMaterialRegistryEntry,
    OledMaterialRegistryMatchStatus,
    OledMaterialRegistryResolutionRequestArtifact,
    oled_material_registry_resolution_request_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    validate_oled_supplementary_safe_authored_text,
)


OLED_MATERIAL_REGISTRY_ENTRY_PROPOSAL_REQUEST_VERSION = (
    "oled_material_registry_entry_proposal_request.v1"
)
OLED_MATERIAL_REGISTRY_ENTRY_REVIEW_CONTRACT_VERSION = (
    "oled_material_registry_entry_review_contract.v1"
)
OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION = (
    "oled_material_registry_opaque_proposal_allocation.v1"
)

_ENTITY_SCOPE_QUESTION = (
    "Does the PR-M-accepted exact molecular graph represent one local Registry "
    "entity at the intended stereochemistry, charge or protonation, salt, "
    "mixture, complex, and source scope?"
)
_PREFERRED_NAME_QUESTION = (
    "Should the exact paper-local reported subject literal be approved as the "
    "preferred local Registry name, or replaced only by another source-supported "
    "literal?"
)
_ALIAS_QUESTION = (
    "Which exact source-supported literals, if any, should be approved as aliases "
    "without case folding, fuzzy matching, or automatic synonym generation?"
)
_LOCAL_ONLY_NOTICE = (
    "No exact structural candidate was found only in the bound Molly Registry "
    "snapshot; global chemical novelty, literature prior art, patent novelty, and "
    "external database coverage were not assessed."
)


class OledMaterialRegistryEntryReviewDecision(str, Enum):
    APPROVE_LOCAL_REGISTRY_ENTRY_CANDIDATE = (
        "approve_local_registry_entry_candidate"
    )
    KEEP_UNRESOLVED = "keep_unresolved"
    DEFER_ENTITY_POLICY = "defer_entity_policy"
    ROUTE_TO_EXISTING_REGISTRY_RESOLUTION = (
        "route_to_existing_registry_resolution"
    )


class OledMaterialRegistryEntryProposalRequestStatus(str, Enum):
    READY_FOR_HUMAN_REGISTRY_ENTRY_REVIEW = (
        "ready_for_human_registry_entry_review"
    )
    BATCH_CONFLICTS_REQUIRE_HUMAN_REVIEW = (
        "batch_conflicts_require_human_review"
    )
    NO_NEW_ENTITY_PROPOSALS = "no_new_entity_proposals"


class OledMaterialRegistryEntryProposalBatchFindingKind(str, Enum):
    DUPLICATE_PROPOSED_MATERIAL_ID = "duplicate_proposed_material_id"
    DUPLICATE_CANONICAL_SMILES = "duplicate_canonical_smiles"
    DUPLICATE_INCHIKEY = "duplicate_inchikey"
    DUPLICATE_PROPOSED_CANONICAL_NAME = "duplicate_proposed_canonical_name"


class OledMaterialRegistryEntryReviewContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    contract_version: str = OLED_MATERIAL_REGISTRY_ENTRY_REVIEW_CONTRACT_VERSION
    allowed_decisions: list[OledMaterialRegistryEntryReviewDecision]
    entity_scope_question: str
    preferred_name_question: str
    alias_question: str
    local_snapshot_only_notice: str
    material_id_allocation_contract_version: str
    single_molecular_entity_policy_only: StrictBool = True
    reported_subject_auto_approved_as_canonical_name: StrictBool = False
    reported_subject_auto_added_as_alias: StrictBool = False
    aliases_require_exact_source_support: StrictBool = True
    exact_structure_and_name_collision_review_required: StrictBool = True
    review_contract_acknowledgement_required: StrictBool = True
    global_chemical_novelty_assessed: StrictBool = False
    literature_prior_art_assessed: StrictBool = False
    patent_novelty_assessed: StrictBool = False
    external_database_search_performed: StrictBool = False
    registry_write_requested: StrictBool = False
    contract_digest: str

    @field_validator("contract_version")
    @classmethod
    def validate_contract_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_ENTRY_REVIEW_CONTRACT_VERSION:
            raise ValueError("unexpected material Registry entry review contract")
        return value

    @field_validator("material_id_allocation_contract_version")
    @classmethod
    def validate_allocation_contract(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION:
            raise ValueError("unexpected material ID allocation contract")
        return value

    @field_validator("contract_digest")
    @classmethod
    def validate_contract_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="contract_digest")

    @model_validator(mode="after")
    def validate_contract_integrity(self) -> OledMaterialRegistryEntryReviewContract:
        expected_decisions = list(OledMaterialRegistryEntryReviewDecision)
        if self.allowed_decisions != expected_decisions:
            raise ValueError("material Registry entry review decisions changed")
        expected_text = {
            "entity_scope_question": _ENTITY_SCOPE_QUESTION,
            "preferred_name_question": _PREFERRED_NAME_QUESTION,
            "alias_question": _ALIAS_QUESTION,
            "local_snapshot_only_notice": _LOCAL_ONLY_NOTICE,
        }
        for field_name, expected in expected_text.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"material Registry entry {field_name} changed")
        fixed_true = (
            "single_molecular_entity_policy_only",
            "aliases_require_exact_source_support",
            "exact_structure_and_name_collision_review_required",
            "review_contract_acknowledgement_required",
        )
        fixed_false = (
            "reported_subject_auto_approved_as_canonical_name",
            "reported_subject_auto_added_as_alias",
            "global_chemical_novelty_assessed",
            "literature_prior_art_assessed",
            "patent_novelty_assessed",
            "external_database_search_performed",
            "registry_write_requested",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("material Registry entry review contract crossed its boundary")
        if oled_material_registry_entry_review_contract_digest(self) != (
            self.contract_digest
        ):
            raise ValueError("material Registry entry review contract digest mismatch")
        return self


class OledMaterialRegistryEntryProposalBatchFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    finding_kind: OledMaterialRegistryEntryProposalBatchFindingKind
    key_digest: str
    affected_entry_review_item_ids: list[str] = Field(min_length=2)
    finding_digest: str
    blocks_automatic_approval: StrictBool = True
    automatic_merge_performed: StrictBool = False

    @field_validator("key_digest", "finding_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("affected_entry_review_item_ids")
    @classmethod
    def validate_item_ids(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_bound_id(item, field_name="entry_review_item_id")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("batch finding item IDs must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_finding_integrity(
        self,
    ) -> OledMaterialRegistryEntryProposalBatchFinding:
        if not self.blocks_automatic_approval or self.automatic_merge_performed:
            raise ValueError("batch conflict crossed its review boundary")
        if _batch_finding_digest(self) != self.finding_digest:
            raise ValueError("material Registry entry batch finding digest mismatch")
        return self


class OledMaterialRegistryEntryProposalRequestItem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    entry_review_item_id: str
    source_adjudicated_item: OledMaterialRegistryAdjudicatedItem
    registry_id: str
    registry_version: str
    registry_snapshot_digest: str
    material_id_allocation_contract_version: str
    material_id_allocation_digest: str
    proposed_material_id: str
    proposed_canonical_name: str
    proposed_aliases: list[str] = Field(default_factory=list)
    review_contract_digest: str
    entry_review_item_digest: str
    identity_dependent_cell_count: Annotated[StrictInt, Field(ge=1)]
    requires_human_entity_scope_review: StrictBool = True
    requires_human_name_alias_review: StrictBool = True
    requires_review_contract_acknowledgement: StrictBool = True
    local_snapshot_match_replayed: StrictBool = True
    source_graph_human_accepted: StrictBool = True
    material_id_reserved: StrictBool = False
    material_id_assigned: StrictBool = False
    canonical_name_approved: StrictBool = False
    aliases_approved: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    global_chemical_novelty_assessed: StrictBool = False

    @field_validator(
        "entry_review_item_id",
        "registry_id",
        "registry_version",
        "proposed_material_id",
    )
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("material_id_allocation_contract_version")
    @classmethod
    def validate_allocation_contract(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION:
            raise ValueError("unexpected material ID allocation contract")
        return value

    @field_validator(
        "registry_snapshot_digest",
        "material_id_allocation_digest",
        "review_contract_digest",
        "entry_review_item_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("proposed_canonical_name")
    @classmethod
    def validate_proposed_name(cls, value: str) -> str:
        return _validate_proposed_registry_name(value, field_name="proposed_canonical_name")

    @field_validator("proposed_aliases")
    @classmethod
    def validate_proposed_aliases(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_proposed_registry_name(item, field_name="proposed_alias")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("proposed aliases must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_item_integrity(
        self,
    ) -> OledMaterialRegistryEntryProposalRequestItem:
        source = self.source_adjudicated_item
        request_item = source.request_item
        proposal = source.new_entity_proposal
        if (
            source.decision_entry.decision
            != OledMaterialRegistryDecision.PROPOSE_NEW_ENTITY
            or not source.new_registry_entity_proposed
            or proposal is None
        ):
            raise ValueError("entry review item is not a PR-O new-entity proposal")
        if request_item.match_status != (
            OledMaterialRegistryMatchStatus.NO_EXACT_STRUCTURAL_CANDIDATE
        ):
            raise ValueError("entry review item did not replay a local snapshot no-hit")
        expected_item_id = _entry_review_item_id(request_item.resolution_item_id)
        if self.entry_review_item_id != expected_item_id:
            raise ValueError("material Registry entry review item_id mismatch")
        expected_allocation_digest = _material_id_allocation_digest(
            registry_id=self.registry_id,
            proposal_digest=proposal.proposal_digest,
        )
        if (
            self.material_id_allocation_digest != expected_allocation_digest
            or self.proposed_material_id
            != _proposed_material_id(expected_allocation_digest)
        ):
            raise ValueError("material Registry opaque ID proposal changed")
        if self.proposed_canonical_name != proposal.reported_subject_literal:
            raise ValueError("reported subject label proposal changed")
        if self.proposed_aliases:
            raise ValueError("aliases must remain empty until human review")
        group = request_item.adjudicated_group.review_item.validated_result
        chemistry = group.chemistry_validation
        depiction = request_item.adjudicated_group.review_item.candidate_depiction_asset
        if chemistry is None or depiction is None:
            raise ValueError("entry review item lacks validated chemistry or depiction")
        if (
            chemistry.candidate_digest != proposal.candidate_digest
            or depiction.candidate_digest != proposal.candidate_digest
        ):
            raise ValueError("entry review chemistry or depiction binding mismatch")
        dependent_count = group.bound_identity_group.identity_dependent_cell_count
        if self.identity_dependent_cell_count != dependent_count:
            raise ValueError("entry review dependent-cell count mismatch")
        fixed_true = (
            "requires_human_entity_scope_review",
            "requires_human_name_alias_review",
            "requires_review_contract_acknowledgement",
            "local_snapshot_match_replayed",
            "source_graph_human_accepted",
        )
        fixed_false = (
            "material_id_reserved",
            "material_id_assigned",
            "canonical_name_approved",
            "aliases_approved",
            "registry_entry_created",
            "registry_written",
            "global_chemical_novelty_assessed",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("material Registry entry request item crossed its boundary")
        if oled_material_registry_entry_proposal_request_item_digest(self) != (
            self.entry_review_item_digest
        ):
            raise ValueError("material Registry entry request item digest mismatch")
        return self


class OledMaterialRegistryEntryProposalRequestArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_MATERIAL_REGISTRY_ENTRY_PROPOSAL_REQUEST_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    resolution_request_sha256: str
    resolution_request_digest: str
    registry_adjudication_sha256: str
    registry_adjudication_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    resolution_request: OledMaterialRegistryResolutionRequestArtifact
    registry_adjudication: OledMaterialRegistryAdjudicationArtifact
    review_contract: OledMaterialRegistryEntryReviewContract
    status: OledMaterialRegistryEntryProposalRequestStatus
    source_resolution_item_count: Annotated[StrictInt, Field(ge=0)]
    source_adjudicated_item_count: Annotated[StrictInt, Field(ge=0)]
    entry_review_item_count: Annotated[StrictInt, Field(ge=0)]
    entry_review_cell_count: Annotated[StrictInt, Field(ge=0)]
    existing_entity_mapping_excluded_count: Annotated[StrictInt, Field(ge=0)]
    unresolved_excluded_count: Annotated[StrictInt, Field(ge=0)]
    conflict_deferred_excluded_count: Annotated[StrictInt, Field(ge=0)]
    exact_name_hint_item_count: Annotated[StrictInt, Field(ge=0)]
    batch_conflict_finding_count: Annotated[StrictInt, Field(ge=0)]
    upstream_ontology_review_pending_cell_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    batch_conflict_findings: list[
        OledMaterialRegistryEntryProposalBatchFinding
    ] = Field(default_factory=list)
    entry_review_items: list[OledMaterialRegistryEntryProposalRequestItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    proposal_request_artifact_digest: str
    request_only: StrictBool = True
    offline_only: StrictBool = True
    exact_resolution_request_bytes_bound: StrictBool = True
    exact_registry_adjudication_bytes_bound: StrictBool = True
    # File entry records hashes; embedded models cannot recover original JSON bytes.
    standalone_input_bytes_revalidation_supported: StrictBool = False
    embedded_models_revalidated: StrictBool = True
    joint_pr_n_pr_o_chain_revalidated: StrictBool = True
    complete_new_entity_proposal_coverage_validated: StrictBool = True
    accepted_graph_chemistry_replayed: StrictBool = True
    material_id_candidates_opaque_and_deterministic: StrictBool = True
    fixed_review_contract_bound: StrictBool = True
    source_pdf_read: StrictBool = False
    material_id_reserved: StrictBool = False
    material_id_assigned: StrictBool = False
    canonical_name_approved: StrictBool = False
    aliases_approved: StrictBool = False
    registry_entry_created: StrictBool = False
    registry_written: StrictBool = False
    existing_registry_mutated: StrictBool = False
    observations_materialized: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False
    training_eligible: StrictBool = False
    global_chemical_novelty_assessed: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_ENTRY_PROPOSAL_REQUEST_VERSION:
            raise ValueError("unexpected material Registry entry proposal request version")
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
        "resolution_request_sha256",
        "resolution_request_digest",
        "registry_adjudication_sha256",
        "registry_adjudication_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
        "proposal_request_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("batch_conflict_findings")
    @classmethod
    def validate_finding_order(
        cls,
        value: list[OledMaterialRegistryEntryProposalBatchFinding],
    ) -> list[OledMaterialRegistryEntryProposalBatchFinding]:
        order = [item.finding_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("entry proposal batch findings must be sorted and unique")
        return value

    @field_validator("entry_review_items")
    @classmethod
    def validate_item_order(
        cls,
        value: list[OledMaterialRegistryEntryProposalRequestItem],
    ) -> list[OledMaterialRegistryEntryProposalRequestItem]:
        order = [_entry_review_item_source_sort_key(item) for item in value]
        item_ids = [item.entry_review_item_id for item in value]
        if order != sorted(order) or len(item_ids) != len(set(item_ids)):
            raise ValueError("entry review items must be source-sorted and unique")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledMaterialRegistryEntryProposalRequestArtifact:
        request = self.resolution_request
        adjudication = self.registry_adjudication
        if _parse_timestamp(self.generated_at) < _parse_timestamp(
            adjudication.generated_at
        ):
            raise ValueError("entry proposal request predates PR-O")
        expected_bindings = {
            "run_id": request.run_id,
            "paper_id": request.paper_id,
            "resolution_request_sha256": adjudication.request_artifact_sha256,
            "resolution_request_digest": request.request_artifact_digest,
            "registry_adjudication_digest": adjudication.adjudication_artifact_digest,
            "registry_snapshot_sha256": request.registry_snapshot_sha256,
            "registry_snapshot_digest": request.registry_snapshot_digest,
        }
        for field_name, expected in expected_bindings.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"entry proposal request {field_name} mismatch")
        if (
            oled_material_registry_resolution_request_artifact_digest(request)
            != request.request_artifact_digest
            or oled_material_registry_adjudication_artifact_digest(adjudication)
            != adjudication.adjudication_artifact_digest
        ):
            raise ValueError("entry proposal embedded artifact digest mismatch")
        validate_oled_material_registry_request_adjudication_chain(
            request,
            adjudication,
        )
        expected_contract = build_oled_material_registry_entry_review_contract()
        if self.review_contract.model_dump(mode="json") != expected_contract.model_dump(
            mode="json"
        ):
            raise ValueError("entry proposal review contract changed")
        expected_items = _build_entry_review_items(
            request=request,
            adjudication=adjudication,
            review_contract=expected_contract,
        )
        _validate_proposed_material_id_availability(
            expected_items,
            request.registry_snapshot.entries,
        )
        if [item.model_dump(mode="json") for item in self.entry_review_items] != [
            item.model_dump(mode="json") for item in expected_items
        ]:
            raise ValueError("entry proposal request item derivation changed")
        expected_findings = _build_batch_conflict_findings(expected_items)
        if [item.model_dump(mode="json") for item in self.batch_conflict_findings] != [
            item.model_dump(mode="json") for item in expected_findings
        ]:
            raise ValueError("entry proposal batch conflict derivation changed")
        expected_counts = {
            "source_resolution_item_count": request.resolution_item_count,
            "source_adjudicated_item_count": adjudication.review_item_count,
            "entry_review_item_count": adjudication.new_entity_proposal_count,
            "entry_review_cell_count": adjudication.new_entity_proposal_cell_count,
            "existing_entity_mapping_excluded_count": (
                adjudication.existing_entity_mapping_count
            ),
            "unresolved_excluded_count": adjudication.kept_unresolved_count,
            "conflict_deferred_excluded_count": adjudication.conflict_deferred_count,
            "exact_name_hint_item_count": sum(
                bool(item.source_adjudicated_item.request_item.exact_alias_literal_hits)
                for item in expected_items
            ),
            "batch_conflict_finding_count": len(expected_findings),
            "upstream_ontology_review_pending_cell_count": (
                adjudication.upstream_ontology_review_pending_cell_count
            ),
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"entry proposal request {field_name} mismatch")
        if self.entry_review_item_count != len(self.entry_review_items) or (
            self.entry_review_cell_count
            != sum(item.identity_dependent_cell_count for item in self.entry_review_items)
        ):
            raise ValueError("entry proposal request aggregate count mismatch")
        expected_status = _proposal_request_status(expected_items, expected_findings)
        if self.status != expected_status or self.device_only_cell_count != 0:
            raise ValueError("entry proposal request status or device boundary mismatch")
        fixed_true = (
            "request_only",
            "offline_only",
            "exact_resolution_request_bytes_bound",
            "exact_registry_adjudication_bytes_bound",
            "embedded_models_revalidated",
            "joint_pr_n_pr_o_chain_revalidated",
            "complete_new_entity_proposal_coverage_validated",
            "accepted_graph_chemistry_replayed",
            "material_id_candidates_opaque_and_deterministic",
            "fixed_review_contract_bound",
        )
        fixed_false = (
            "standalone_input_bytes_revalidation_supported",
            "source_pdf_read",
            "material_id_reserved",
            "material_id_assigned",
            "canonical_name_approved",
            "aliases_approved",
            "registry_entry_created",
            "registry_written",
            "existing_registry_mutated",
            "observations_materialized",
            "reviewed_evidence_staging",
            "gold_records_created",
            "dataset_written",
            "training_eligible",
            "global_chemical_novelty_assessed",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
        )
        if any(not getattr(self, name) for name in fixed_true) or any(
            getattr(self, name) for name in fixed_false
        ):
            raise ValueError("entry proposal request crossed its boundary")
        if oled_material_registry_entry_proposal_request_artifact_digest(self) != (
            self.proposal_request_artifact_digest
        ):
            raise ValueError("entry proposal request artifact digest mismatch")
        return self


def build_oled_material_registry_entry_review_contract(
) -> OledMaterialRegistryEntryReviewContract:
    payload: dict[str, Any] = {
        "contract_version": OLED_MATERIAL_REGISTRY_ENTRY_REVIEW_CONTRACT_VERSION,
        "allowed_decisions": list(OledMaterialRegistryEntryReviewDecision),
        "entity_scope_question": _ENTITY_SCOPE_QUESTION,
        "preferred_name_question": _PREFERRED_NAME_QUESTION,
        "alias_question": _ALIAS_QUESTION,
        "local_snapshot_only_notice": _LOCAL_ONLY_NOTICE,
        "material_id_allocation_contract_version": (
            OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION
        ),
        "single_molecular_entity_policy_only": True,
        "reported_subject_auto_approved_as_canonical_name": False,
        "reported_subject_auto_added_as_alias": False,
        "aliases_require_exact_source_support": True,
        "exact_structure_and_name_collision_review_required": True,
        "review_contract_acknowledgement_required": True,
        "global_chemical_novelty_assessed": False,
        "literature_prior_art_assessed": False,
        "patent_novelty_assessed": False,
        "external_database_search_performed": False,
        "registry_write_requested": False,
        "contract_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryEntryReviewContract.model_construct(**payload)
    payload["contract_digest"] = (
        oled_material_registry_entry_review_contract_digest(provisional)
    )
    return OledMaterialRegistryEntryReviewContract.model_validate(payload)


def validate_oled_material_registry_entry_proposal_request_inputs(
    *,
    resolution_request: OledMaterialRegistryResolutionRequestArtifact,
    resolution_request_sha256: str,
    registry_adjudication: OledMaterialRegistryAdjudicationArtifact,
) -> None:
    request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        resolution_request.model_dump(mode="json")
    )
    adjudication = OledMaterialRegistryAdjudicationArtifact.model_validate(
        registry_adjudication.model_dump(mode="json")
    )
    actual_request_sha = _normalize_sha256(
        resolution_request_sha256,
        field_name="resolution_request_sha256",
    )
    if adjudication.request_artifact_sha256 != actual_request_sha:
        raise ValueError("PR-O is not bound to the exact supplied PR-N file")
    validate_oled_material_registry_request_adjudication_chain(
        request,
        adjudication,
    )


def build_oled_material_registry_entry_proposal_request_artifact(
    *,
    resolution_request: OledMaterialRegistryResolutionRequestArtifact,
    resolution_request_sha256: str,
    registry_adjudication: OledMaterialRegistryAdjudicationArtifact,
    registry_adjudication_sha256: str,
    generated_at: str,
) -> OledMaterialRegistryEntryProposalRequestArtifact:
    request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        resolution_request.model_dump(mode="json")
    )
    adjudication = OledMaterialRegistryAdjudicationArtifact.model_validate(
        registry_adjudication.model_dump(mode="json")
    )
    validate_oled_material_registry_entry_proposal_request_inputs(
        resolution_request=request,
        resolution_request_sha256=resolution_request_sha256,
        registry_adjudication=adjudication,
    )
    review_contract = build_oled_material_registry_entry_review_contract()
    items = _build_entry_review_items(
        request=request,
        adjudication=adjudication,
        review_contract=review_contract,
    )
    _validate_proposed_material_id_availability(
        items,
        request.registry_snapshot.entries,
    )
    findings = _build_batch_conflict_findings(items)
    payload: dict[str, Any] = {
        "artifact_version": OLED_MATERIAL_REGISTRY_ENTRY_PROPOSAL_REQUEST_VERSION,
        "run_id": request.run_id,
        "paper_id": request.paper_id,
        "generated_at": generated_at,
        "resolution_request_sha256": _normalize_sha256(
            resolution_request_sha256,
            field_name="resolution_request_sha256",
        ),
        "resolution_request_digest": request.request_artifact_digest,
        "registry_adjudication_sha256": _normalize_sha256(
            registry_adjudication_sha256,
            field_name="registry_adjudication_sha256",
        ),
        "registry_adjudication_digest": adjudication.adjudication_artifact_digest,
        "registry_snapshot_sha256": request.registry_snapshot_sha256,
        "registry_snapshot_digest": request.registry_snapshot_digest,
        "resolution_request": request,
        "registry_adjudication": adjudication,
        "review_contract": review_contract,
        "status": _proposal_request_status(items, findings),
        "source_resolution_item_count": request.resolution_item_count,
        "source_adjudicated_item_count": adjudication.review_item_count,
        "entry_review_item_count": adjudication.new_entity_proposal_count,
        "entry_review_cell_count": adjudication.new_entity_proposal_cell_count,
        "existing_entity_mapping_excluded_count": (
            adjudication.existing_entity_mapping_count
        ),
        "unresolved_excluded_count": adjudication.kept_unresolved_count,
        "conflict_deferred_excluded_count": adjudication.conflict_deferred_count,
        "exact_name_hint_item_count": sum(
            bool(item.source_adjudicated_item.request_item.exact_alias_literal_hits)
            for item in items
        ),
        "batch_conflict_finding_count": len(findings),
        "upstream_ontology_review_pending_cell_count": (
            adjudication.upstream_ontology_review_pending_cell_count
        ),
        "device_only_cell_count": 0,
        "batch_conflict_findings": findings,
        "entry_review_items": items,
        "proposal_request_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryEntryProposalRequestArtifact.model_construct(
        **payload
    )
    payload["proposal_request_artifact_digest"] = (
        oled_material_registry_entry_proposal_request_artifact_digest(provisional)
    )
    return OledMaterialRegistryEntryProposalRequestArtifact.model_validate(payload)


def render_oled_material_registry_entry_proposal_request_markdown(
    artifact: OledMaterialRegistryEntryProposalRequestArtifact,
    *,
    artifact_sha256: str,
) -> str:
    request = OledMaterialRegistryEntryProposalRequestArtifact.model_validate(
        artifact.model_dump(mode="json")
    )
    exact_sha = _normalize_sha256(artifact_sha256, field_name="artifact_sha256")
    lines = [
        "# OLED Local Material Registry Entry Review Request",
        "",
        "> This packet proposes entries only for the exact bound Molly Registry snapshot.",
        "> It does not claim that any molecule or material is globally novel.",
        "",
        "## Exact bindings",
        "",
        f"- request artifact SHA-256: `{exact_sha}`",
        f"- request artifact digest: `{request.proposal_request_artifact_digest}`",
        f"- PR-N SHA-256 / digest: `{request.resolution_request_sha256}` / "
        f"`{request.resolution_request_digest}`",
        f"- PR-O SHA-256 / digest: `{request.registry_adjudication_sha256}` / "
        f"`{request.registry_adjudication_digest}`",
        f"- Registry snapshot SHA-256 / digest: `{request.registry_snapshot_sha256}` / "
        f"`{request.registry_snapshot_digest}`",
        f"- Registry: `{_md(request.resolution_request.registry_snapshot.registry_id)}` "
        f"version `{_md(request.resolution_request.registry_snapshot.registry_version)}`",
        f"- review contract digest: `{request.review_contract.contract_digest}`",
        f"- status: `{request.status.value}`",
        "",
        "## Scope boundary",
        "",
        "- PR-N and PR-O file SHA-256 values were recorded from the files supplied",
        "  to the controlled builder; the original JSON bytes are not embedded.",
        "- Standalone validation replays embedded models and semantic digests, but",
        "  cannot independently recover or revalidate either original input file.",
        "- `standalone_input_bytes_revalidation_supported=false`.",
        f"- {_md(request.review_contract.local_snapshot_only_notice)}",
        "- The proposed material ID is opaque and deterministic for this exact PR-O proposal,",
        "  but it is not reserved or assigned until a later approved write.",
        "- The paper-local reported name is shown only as an unapproved preferred-label proposal.",
        "- Aliases start empty and require exact source support plus human approval.",
        "- Version 1 accepts only one exact single-molecular-entity graph; unresolved salts,",
        "  mixtures, complexes, polymers, or source-scope conflicts must be deferred.",
        "- No Registry, observation, Gold, dataset, or training write occurs here.",
        "",
        "## Counts",
        "",
        f"- entry review items: `{request.entry_review_item_count}`",
        f"- identity-dependent cells held behind Registry review: `{request.entry_review_cell_count}`",
        f"- exact existing-name hint items: `{request.exact_name_hint_item_count}`",
        f"- batch conflict findings: `{request.batch_conflict_finding_count}`",
        f"- device-only cells admitted: `{request.device_only_cell_count}`",
        "",
        "## Fixed review questions",
        "",
        f"1. {_md(request.review_contract.entity_scope_question)}",
        f"2. {_md(request.review_contract.preferred_name_question)}",
        f"3. {_md(request.review_contract.alias_question)}",
        "",
        "Allowed decisions: "
        + ", ".join(
            f"`{decision.value}`"
            for decision in request.review_contract.allowed_decisions
        ),
        "",
    ]
    if request.batch_conflict_findings:
        lines.extend(["## Batch conflicts", ""])
        for finding in request.batch_conflict_findings:
            lines.append(
                f"- `{finding.finding_kind.value}` affects "
                + ", ".join(
                    f"`{_md(item_id)}`"
                    for item_id in finding.affected_entry_review_item_ids
                )
            )
        lines.append("")
    for index, item in enumerate(request.entry_review_items, start=1):
        source = item.source_adjudicated_item
        request_item = source.request_item
        group = request_item.adjudicated_group.review_item.validated_result
        bound = group.bound_identity_group
        proposal = source.new_entity_proposal
        chemistry = group.chemistry_validation
        depiction = request_item.adjudicated_group.review_item.candidate_depiction_asset
        assert proposal is not None
        assert chemistry is not None
        assert depiction is not None
        lines.extend(
            [
                f"## E{index:02d}: `{_md(proposal.reported_subject_literal)}`",
                "",
                f"- entry review item ID: `{item.entry_review_item_id}`",
                f"- item digest: `{item.entry_review_item_digest}`",
                f"- source resolution item ID: `{request_item.resolution_item_id}`",
                f"- source new-entity proposal digest: `{proposal.proposal_digest}`",
                f"- source table / row / PDF page: `{_md(bound.table_id)}` / "
                f"`{bound.row_index}` / `{bound.pdf_page_number_one_based}`",
                f"- dependent property cells held: `{item.identity_dependent_cell_count}`",
                "",
                "### Accepted graph and automatic chemistry facts",
                "",
                f"- canonical isomeric SMILES: `{_md(proposal.canonical_isomeric_smiles_candidate)}`",
                f"- standard InChI: `{_md(proposal.standard_inchi_candidate)}`",
                f"- InChIKey: `{_md(proposal.inchikey_candidate)}`",
                f"- fragments / charged atoms / net formal charge: "
                f"`{chemistry.fragment_count}` / `{chemistry.charged_atom_count}` / "
                f"`{chemistry.net_formal_charge}`",
                f"- unassigned atom / bond stereochemistry: "
                f"`{chemistry.unassigned_atom_stereochemistry_count}` / "
                f"`{chemistry.unassigned_bond_stereochemistry_count}`",
                f"- chemistry findings: `{', '.join(code.value for code in chemistry.finding_codes) or 'none'}`",
                f"- depiction asset: `{_md(depiction.asset_filename)}` / "
                f"`{depiction.rendered_asset_sha256}`",
                "",
                "### Unapproved local Registry entry proposal",
                "",
                f"- proposed opaque material ID: `{item.proposed_material_id}`",
                f"- allocation digest: `{item.material_id_allocation_digest}`",
                f"- proposed preferred name: `{_md(item.proposed_canonical_name)}`",
                "- proposed aliases: `[]`",
                f"- exact existing name/alias hints: `{len(request_item.exact_alias_literal_hits)}`",
                f"- related snapshot conflict findings: `{len(source.reviewed_registry_conflict_findings)}`",
                "",
                "### Existing-snapshot name hints (never identity evidence)",
                "",
            ]
        )
        if request_item.exact_alias_literal_hits:
            for hit in request_item.exact_alias_literal_hits:
                lines.append(
                    f"- `{_md(hit.material_id)}` / `{hit.matched_field.value}` / "
                    f"`{_md(hit.matched_literal)}`"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "### Existing-snapshot duplicate-key findings",
                "",
            ]
        )
        if source.reviewed_registry_conflict_findings:
            for finding in source.reviewed_registry_conflict_findings:
                material_ids = ", ".join(
                    f"`{_md(material_id)}`" for material_id in finding.material_ids
                )
                lines.append(
                    f"- `{finding.finding_kind.value}` / "
                    f"`{_md(finding.key_literal)}` / {material_ids}"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "### Human response required",
                "",
                "- select one allowed decision and acknowledge the exact review contract;",
                "- if approving, confirm the single-entity graph scope and explicitly approve",
                "  the material ID, preferred name, and exact alias list;",
                "- route any newly discovered existing-entity match back to Registry resolution; and",
                "- do not treat the local snapshot no-hit as global novelty evidence.",
                "",
            ]
        )
    if not request.entry_review_items:
        lines.extend(
            [
                "## Entry review items",
                "",
                "No PR-O new-entity proposal is eligible for local Registry entry review.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def oled_material_registry_entry_review_contract_digest(
    contract: OledMaterialRegistryEntryReviewContract,
) -> str:
    payload = contract.model_dump(mode="json")
    payload.pop("contract_digest", None)
    return _stable_hash(payload)


def oled_material_registry_entry_proposal_request_item_digest(
    item: OledMaterialRegistryEntryProposalRequestItem,
) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("entry_review_item_digest", None)
    return _stable_hash(payload)


def oled_material_registry_entry_proposal_request_artifact_digest(
    artifact: OledMaterialRegistryEntryProposalRequestArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("proposal_request_artifact_digest", None)
    return _stable_hash(payload)


def _build_entry_review_items(
    *,
    request: OledMaterialRegistryResolutionRequestArtifact,
    adjudication: OledMaterialRegistryAdjudicationArtifact,
    review_contract: OledMaterialRegistryEntryReviewContract,
) -> list[OledMaterialRegistryEntryProposalRequestItem]:
    items = [
        _build_entry_review_item(
            item,
            registry_id=request.registry_snapshot.registry_id,
            registry_version=request.registry_snapshot.registry_version,
            registry_snapshot_digest=request.registry_snapshot.snapshot_digest,
            review_contract_digest=review_contract.contract_digest,
        )
        for item in adjudication.adjudicated_items
        if item.new_registry_entity_proposed
    ]
    return sorted(items, key=_entry_review_item_source_sort_key)


def _build_entry_review_item(
    source: OledMaterialRegistryAdjudicatedItem,
    *,
    registry_id: str,
    registry_version: str,
    registry_snapshot_digest: str,
    review_contract_digest: str,
) -> OledMaterialRegistryEntryProposalRequestItem:
    proposal = source.new_entity_proposal
    if proposal is None:
        raise ValueError("new Registry entity proposal is missing")
    allocation_digest = _material_id_allocation_digest(
        registry_id=registry_id,
        proposal_digest=proposal.proposal_digest,
    )
    group = source.request_item.adjudicated_group.review_item.validated_result
    payload: dict[str, Any] = {
        "entry_review_item_id": _entry_review_item_id(
            source.request_item.resolution_item_id
        ),
        "source_adjudicated_item": source,
        "registry_id": registry_id,
        "registry_version": registry_version,
        "registry_snapshot_digest": registry_snapshot_digest,
        "material_id_allocation_contract_version": (
            OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION
        ),
        "material_id_allocation_digest": allocation_digest,
        "proposed_material_id": _proposed_material_id(allocation_digest),
        "proposed_canonical_name": proposal.reported_subject_literal,
        "proposed_aliases": [],
        "review_contract_digest": review_contract_digest,
        "entry_review_item_digest": "sha256:" + "0" * 64,
        "identity_dependent_cell_count": (
            group.bound_identity_group.identity_dependent_cell_count
        ),
    }
    provisional = OledMaterialRegistryEntryProposalRequestItem.model_construct(
        **payload
    )
    payload["entry_review_item_digest"] = (
        oled_material_registry_entry_proposal_request_item_digest(provisional)
    )
    return OledMaterialRegistryEntryProposalRequestItem.model_validate(payload)


def _build_batch_conflict_findings(
    items: Sequence[OledMaterialRegistryEntryProposalRequestItem],
) -> list[OledMaterialRegistryEntryProposalBatchFinding]:
    keys: dict[
        tuple[OledMaterialRegistryEntryProposalBatchFindingKind, str],
        list[str],
    ] = {}
    for item in items:
        proposal = item.source_adjudicated_item.new_entity_proposal
        assert proposal is not None
        values = (
            (
                OledMaterialRegistryEntryProposalBatchFindingKind
                .DUPLICATE_PROPOSED_MATERIAL_ID,
                item.proposed_material_id,
            ),
            (
                OledMaterialRegistryEntryProposalBatchFindingKind
                .DUPLICATE_CANONICAL_SMILES,
                proposal.canonical_isomeric_smiles_candidate,
            ),
            (
                OledMaterialRegistryEntryProposalBatchFindingKind.DUPLICATE_INCHIKEY,
                proposal.inchikey_candidate,
            ),
            (
                OledMaterialRegistryEntryProposalBatchFindingKind
                .DUPLICATE_PROPOSED_CANONICAL_NAME,
                item.proposed_canonical_name,
            ),
        )
        for kind, value in values:
            keys.setdefault((kind, value), []).append(item.entry_review_item_id)
    findings: list[OledMaterialRegistryEntryProposalBatchFinding] = []
    for (kind, value), item_ids in keys.items():
        if len(item_ids) < 2:
            continue
        payload: dict[str, Any] = {
            "finding_kind": kind,
            "key_digest": _stable_hash({"kind": kind.value, "value": value}),
            "affected_entry_review_item_ids": sorted(item_ids),
            "finding_digest": "sha256:" + "0" * 64,
            "blocks_automatic_approval": True,
            "automatic_merge_performed": False,
        }
        provisional = OledMaterialRegistryEntryProposalBatchFinding.model_construct(
            **payload
        )
        payload["finding_digest"] = _batch_finding_digest(provisional)
        findings.append(
            OledMaterialRegistryEntryProposalBatchFinding.model_validate(payload)
        )
    return sorted(findings, key=lambda finding: finding.finding_digest)


def _proposal_request_status(
    items: Sequence[OledMaterialRegistryEntryProposalRequestItem],
    findings: Sequence[OledMaterialRegistryEntryProposalBatchFinding],
) -> OledMaterialRegistryEntryProposalRequestStatus:
    if not items:
        return (
            OledMaterialRegistryEntryProposalRequestStatus.NO_NEW_ENTITY_PROPOSALS
        )
    if findings:
        return (
            OledMaterialRegistryEntryProposalRequestStatus
            .BATCH_CONFLICTS_REQUIRE_HUMAN_REVIEW
        )
    return (
        OledMaterialRegistryEntryProposalRequestStatus
        .READY_FOR_HUMAN_REGISTRY_ENTRY_REVIEW
    )


def _entry_review_item_id(resolution_item_id: str) -> str:
    return f"material-registry-entry-review:{resolution_item_id}"


def _entry_review_item_source_sort_key(
    item: OledMaterialRegistryEntryProposalRequestItem,
) -> tuple[str, str, int, str]:
    bound = (
        item.source_adjudicated_item.request_item.adjudicated_group.review_item
        .validated_result.bound_identity_group
    )
    return (
        bound.scope_id,
        bound.table_id,
        bound.row_index,
        bound.identity_group_id,
    )


def _validate_proposed_material_id_availability(
    items: Sequence[OledMaterialRegistryEntryProposalRequestItem],
    registry_entries: Sequence[OledMaterialRegistryEntry],
) -> None:
    occupied_ids = {entry.material_id for entry in registry_entries}
    collisions = sorted(
        item.proposed_material_id
        for item in items
        if item.proposed_material_id in occupied_ids
    )
    if collisions:
        raise ValueError(
            "proposed material ID is already occupied in the bound Registry snapshot"
        )


def _material_id_allocation_digest(
    *,
    registry_id: str,
    proposal_digest: str,
) -> str:
    return _stable_hash(
        {
            "allocation_contract_version": (
                OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION
            ),
            "registry_id": registry_id,
            "new_entity_proposal_digest": proposal_digest,
        }
    )


def _proposed_material_id(allocation_digest: str) -> str:
    normalized = _normalize_sha256(
        allocation_digest,
        field_name="material_id_allocation_digest",
    )
    return f"material:{normalized[7:39]}"


def _batch_finding_digest(
    finding: OledMaterialRegistryEntryProposalBatchFinding,
) -> str:
    payload = finding.model_dump(mode="json")
    payload.pop("finding_digest", None)
    return _stable_hash(payload)


def _validate_proposed_registry_name(value: Any, *, field_name: str) -> str:
    clean = validate_oled_supplementary_safe_authored_text(
        value,
        field_name=field_name,
        required=True,
        max_length=4_000,
    )
    if html.unescape(clean) != clean:
        raise ValueError(f"{field_name} must not contain HTML entities")
    return clean


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


__all__ = [
    "OLED_MATERIAL_REGISTRY_ENTRY_PROPOSAL_REQUEST_VERSION",
    "OLED_MATERIAL_REGISTRY_ENTRY_REVIEW_CONTRACT_VERSION",
    "OLED_MATERIAL_REGISTRY_MATERIAL_ID_ALLOCATION_CONTRACT_VERSION",
    "OledMaterialRegistryEntryProposalBatchFinding",
    "OledMaterialRegistryEntryProposalBatchFindingKind",
    "OledMaterialRegistryEntryProposalRequestArtifact",
    "OledMaterialRegistryEntryProposalRequestItem",
    "OledMaterialRegistryEntryProposalRequestStatus",
    "OledMaterialRegistryEntryReviewContract",
    "OledMaterialRegistryEntryReviewDecision",
    "build_oled_material_registry_entry_proposal_request_artifact",
    "build_oled_material_registry_entry_review_contract",
    "oled_material_registry_entry_proposal_request_artifact_digest",
    "oled_material_registry_entry_proposal_request_item_digest",
    "oled_material_registry_entry_review_contract_digest",
    "render_oled_material_registry_entry_proposal_request_markdown",
    "validate_oled_material_registry_entry_proposal_request_inputs",
]
