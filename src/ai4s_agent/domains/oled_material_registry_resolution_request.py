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

from ai4s_agent.domains.oled_supplementary_material_identity_evidence_response import (
    SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION,
    OledSupplementaryMaterialIdentityProposeStructureCandidate,
    OledSupplementaryMaterialIdentityStructureEncodingKind,
    _rdkit_chemistry_observation,
    _rdkit_runtime_versions,
)
from ai4s_agent.domains.oled_supplementary_material_identity_review import (
    OledSupplementaryAdjudicatedMaterialIdentityGroup,
    OledSupplementaryMaterialIdentityAdjudicationArtifact,
    _normalize_sha256,
    _parse_timestamp,
    _validate_bound_id,
    _validate_path_segment,
    _validate_timestamp,
    oled_supplementary_material_identity_adjudication_artifact_digest,
)
from ai4s_agent.domains.oled_supplementary_scoped_candidate_response import (
    validate_oled_supplementary_safe_authored_text,
)


OLED_MATERIAL_REGISTRY_ENTRY_VERSION = "oled_material_registry_entry.v1"
OLED_MATERIAL_REGISTRY_SNAPSHOT_VERSION = "oled_material_registry_snapshot.v1"
OLED_MATERIAL_REGISTRY_RESOLUTION_REQUEST_VERSION = (
    "oled_material_registry_resolution_request.v1"
)


class OledMaterialRegistryMatchStatus(str, Enum):
    NO_EXACT_STRUCTURAL_CANDIDATE = "no_exact_structural_candidate"
    PARTIAL_STRUCTURAL_KEY_MATCH = "partial_structural_key_match"
    ONE_CONSISTENT_EXACT_STRUCTURAL_CANDIDATE = (
        "one_consistent_exact_structural_candidate"
    )
    AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY = "ambiguous_duplicate_structural_key"
    CONFLICTING_STRUCTURAL_KEY_MATCHES = "conflicting_structural_key_matches"


class OledMaterialRegistryResolutionRequestStatus(str, Enum):
    READY_FOR_HUMAN_REGISTRY_RESOLUTION_REVIEW = (
        "ready_for_human_registry_resolution_review"
    )
    REGISTRY_CONFLICTS_REQUIRE_HUMAN_REVIEW = (
        "registry_conflicts_require_human_review"
    )
    NO_REGISTRY_ELIGIBLE_CANDIDATES = "no_registry_eligible_candidates"


class OledMaterialRegistryAliasField(str, Enum):
    CANONICAL_NAME = "canonical_name"
    ALIAS = "alias"


class OledMaterialRegistryConflictKind(str, Enum):
    DUPLICATE_CANONICAL_SMILES = "duplicate_canonical_smiles"
    DUPLICATE_INCHIKEY = "duplicate_inchikey"
    DUPLICATE_REPORTED_NAME_LITERAL = "duplicate_reported_name_literal"


class OledMaterialRegistryEntry(BaseModel):
    """One current Registry entity supplied as part of an immutable snapshot."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    entry_version: str = OLED_MATERIAL_REGISTRY_ENTRY_VERSION
    material_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list, max_length=10_000)
    canonical_isomeric_smiles: str
    standard_inchi: str
    inchikey: str
    entry_digest: str

    @field_validator("entry_version")
    @classmethod
    def validate_entry_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_ENTRY_VERSION:
            raise ValueError("unexpected OLED material Registry entry version")
        return value

    @field_validator("material_id")
    @classmethod
    def validate_material_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="material_id")

    @field_validator("canonical_name")
    @classmethod
    def validate_canonical_name(cls, value: str) -> str:
        return _validate_registry_name(
            value,
            field_name="canonical_name",
            required=True,
        )

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, value: list[str]) -> list[str]:
        clean = [
            _validate_registry_name(item, field_name="alias", required=True)
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("Registry aliases must be sorted and unique")
        return clean

    @field_validator("canonical_isomeric_smiles", "standard_inchi", "inchikey")
    @classmethod
    def validate_chemical_literals(cls, value: str, info: Any) -> str:
        return _validate_exact_chemical_literal(value, field_name=str(info.field_name))

    @field_validator("entry_digest")
    @classmethod
    def validate_entry_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="entry_digest")

    @model_validator(mode="after")
    def validate_entry_integrity(self) -> OledMaterialRegistryEntry:
        if self.canonical_name in self.aliases:
            raise ValueError("canonical_name must not be repeated in aliases")
        observation = _rdkit_chemistry_observation(
            encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
            structure_text=self.canonical_isomeric_smiles,
        )
        if (
            observation["canonical_isomeric_smiles"]
            != self.canonical_isomeric_smiles
            or observation["standard_inchi"] != self.standard_inchi
            or observation["inchikey"] != self.inchikey
        ):
            raise ValueError("Registry entry chemical identifiers are inconsistent")
        if oled_material_registry_entry_digest(self) != self.entry_digest:
            raise ValueError("Registry entry digest mismatch")
        return self


class OledMaterialRegistrySnapshot(BaseModel):
    """A complete, read-only Registry snapshot used for deterministic lookup."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    snapshot_version: str = OLED_MATERIAL_REGISTRY_SNAPSHOT_VERSION
    registry_id: str
    registry_version: str
    generated_at: str
    chemistry_profile_version: str = (
        SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
    )
    toolkit_id: Literal["rdkit"] = "rdkit"
    toolkit_version: str
    inchi_backend_version: str
    entry_count: Annotated[StrictInt, Field(ge=0, le=1_000_000)]
    entries: list[OledMaterialRegistryEntry] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    snapshot_digest: str
    read_only_snapshot: StrictBool = True

    @field_validator("snapshot_version")
    @classmethod
    def validate_snapshot_version(cls, value: str) -> str:
        if value != OLED_MATERIAL_REGISTRY_SNAPSHOT_VERSION:
            raise ValueError("unexpected OLED material Registry snapshot version")
        return value

    @field_validator("registry_id", "registry_version", "toolkit_version", "inchi_backend_version")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        return _validate_timestamp(value, field_name="generated_at")

    @field_validator("chemistry_profile_version")
    @classmethod
    def validate_chemistry_profile(cls, value: str) -> str:
        if value != (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
        ):
            raise ValueError("unexpected material Registry chemistry profile")
        return value

    @field_validator("entries")
    @classmethod
    def validate_entry_order(
        cls,
        value: list[OledMaterialRegistryEntry],
    ) -> list[OledMaterialRegistryEntry]:
        material_ids = [entry.material_id for entry in value]
        if material_ids != sorted(material_ids) or len(material_ids) != len(
            set(material_ids)
        ):
            raise ValueError("Registry entries must be sorted by unique material_id")
        return value

    @field_validator("snapshot_digest")
    @classmethod
    def validate_snapshot_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="snapshot_digest")

    @model_validator(mode="after")
    def validate_snapshot_integrity(self) -> OledMaterialRegistrySnapshot:
        if self.entry_count != len(self.entries):
            raise ValueError("Registry snapshot entry count mismatch")
        runtime = _rdkit_runtime_versions()
        if (self.toolkit_version, self.inchi_backend_version) != runtime:
            raise ValueError("Registry snapshot chemistry runtime does not match execution")
        if not self.read_only_snapshot:
            raise ValueError("Registry snapshot must remain read-only")
        if oled_material_registry_snapshot_digest(self) != self.snapshot_digest:
            raise ValueError("Registry snapshot digest mismatch")
        return self


class OledMaterialRegistryAliasLiteralHit(BaseModel):
    """One codepoint-exact name hit; it is never identity evidence by itself."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    material_id: str
    matched_field: OledMaterialRegistryAliasField
    matched_literal: str
    alias_hit_digest: str
    exact_codepoint_match_only: StrictBool = True
    identity_evidence: StrictBool = False

    @field_validator("material_id")
    @classmethod
    def validate_material_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="material_id")

    @field_validator("matched_literal")
    @classmethod
    def validate_matched_literal(cls, value: str) -> str:
        return _validate_registry_name(
            value,
            field_name="matched_literal",
            required=True,
        )

    @field_validator("alias_hit_digest")
    @classmethod
    def validate_alias_hit_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="alias_hit_digest")

    @model_validator(mode="after")
    def validate_alias_hit_integrity(self) -> OledMaterialRegistryAliasLiteralHit:
        if not self.exact_codepoint_match_only or self.identity_evidence:
            raise ValueError("Registry alias hit crossed its evidence boundary")
        if _alias_hit_digest(self) != self.alias_hit_digest:
            raise ValueError("Registry alias hit digest mismatch")
        return self


class OledMaterialRegistryConflictFinding(BaseModel):
    """A duplicate Registry key that must remain visible to the human reviewer."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    finding_kind: OledMaterialRegistryConflictKind
    key_literal: str
    material_ids: list[str] = Field(min_length=2, max_length=100_000)
    finding_digest: str
    automatic_merge_performed: StrictBool = False

    @field_validator("key_literal")
    @classmethod
    def validate_key_literal(cls, value: str) -> str:
        if not isinstance(value, str) or not value or len(value) > 20_000:
            raise ValueError("Registry conflict key_literal is required and bounded")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("Registry conflict key_literal contains control text")
        return value

    @field_validator("material_ids")
    @classmethod
    def validate_material_ids(cls, value: list[str]) -> list[str]:
        clean = [_validate_bound_id(item, field_name="material_id") for item in value]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("Registry conflict material_ids must be sorted and unique")
        return clean

    @field_validator("finding_digest")
    @classmethod
    def validate_finding_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="finding_digest")

    @model_validator(mode="after")
    def validate_finding_integrity(self) -> OledMaterialRegistryConflictFinding:
        if self.automatic_merge_performed:
            raise ValueError("Registry conflict cannot perform an automatic merge")
        if _registry_conflict_finding_digest(self) != self.finding_digest:
            raise ValueError("Registry conflict finding digest mismatch")
        return self


class OledMaterialRegistryResolutionRequestItem(BaseModel):
    """One PR-M accepted paper-local graph compared with one Registry snapshot."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    resolution_item_id: str
    resolution_item_digest: str
    adjudicated_group: OledSupplementaryAdjudicatedMaterialIdentityGroup
    candidate_digest: str
    match_status: OledMaterialRegistryMatchStatus
    canonical_smiles_candidate_material_ids: list[str] = Field(default_factory=list)
    inchikey_candidate_material_ids: list[str] = Field(default_factory=list)
    consistent_exact_candidate_material_id: str = ""
    exact_alias_literal_hits: list[OledMaterialRegistryAliasLiteralHit] = Field(
        default_factory=list,
        max_length=100_000,
    )
    related_registry_conflict_digests: list[str] = Field(default_factory=list)
    registry_resolution_required: StrictBool = True
    material_identity_resolved: StrictBool = False
    canonical_material_id_assigned: StrictBool = False
    automatic_candidate_merge: StrictBool = False

    @field_validator("resolution_item_id")
    @classmethod
    def validate_resolution_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="resolution_item_id")

    @field_validator(
        "resolution_item_digest",
        "candidate_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator(
        "canonical_smiles_candidate_material_ids",
        "inchikey_candidate_material_ids",
    )
    @classmethod
    def validate_candidate_material_ids(
        cls,
        value: list[str],
        info: Any,
    ) -> list[str]:
        clean = [_validate_bound_id(item, field_name="material_id") for item in value]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @field_validator("consistent_exact_candidate_material_id")
    @classmethod
    def validate_optional_material_id(cls, value: str) -> str:
        if not value:
            return ""
        return _validate_bound_id(value, field_name="consistent_exact_candidate_material_id")

    @field_validator("exact_alias_literal_hits")
    @classmethod
    def validate_alias_hit_order(
        cls,
        value: list[OledMaterialRegistryAliasLiteralHit],
    ) -> list[OledMaterialRegistryAliasLiteralHit]:
        order = [item.alias_hit_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("Registry alias hits must be sorted and unique")
        return value

    @field_validator("related_registry_conflict_digests")
    @classmethod
    def validate_conflict_digests(cls, value: list[str]) -> list[str]:
        clean = [
            _normalize_sha256(item, field_name="registry_conflict_digest")
            for item in value
        ]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("related Registry conflict digests must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_item_integrity(self) -> OledMaterialRegistryResolutionRequestItem:
        group = self.adjudicated_group
        if not (
            group.eligible_for_later_registry_review
            and group.paper_local_structure_candidate_accepted
            and group.source_anchors_human_validated
            and group.source_to_candidate_match_human_validated
        ):
            raise ValueError("Registry request item is not PR-M eligible")
        result = group.review_item.validated_result
        if not isinstance(
            result.response_result,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        ) or result.chemistry_validation is None:
            raise ValueError("Registry request item lacks an accepted graph candidate")
        if self.candidate_digest != result.chemistry_validation.candidate_digest:
            raise ValueError("Registry request candidate digest mismatch")
        expected_item_id = _resolution_item_id(group)
        if self.resolution_item_id != expected_item_id:
            raise ValueError("Registry resolution item_id mismatch")
        expected_status, expected_consistent = _classify_match(
            self.canonical_smiles_candidate_material_ids,
            self.inchikey_candidate_material_ids,
        )
        if (
            self.match_status != expected_status
            or self.consistent_exact_candidate_material_id != expected_consistent
        ):
            raise ValueError("Registry resolution match classification mismatch")
        if not self.registry_resolution_required or any(
            (
                self.material_identity_resolved,
                self.canonical_material_id_assigned,
                self.automatic_candidate_merge,
            )
        ):
            raise ValueError("Registry request item crossed its resolution boundary")
        if oled_material_registry_resolution_request_item_digest(self) != (
            self.resolution_item_digest
        ):
            raise ValueError("Registry resolution item digest mismatch")
        return self


class OledMaterialRegistryResolutionRequestArtifact(BaseModel):
    """Self-contained PR-N request; neither an adjudication nor a Registry write."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = OLED_MATERIAL_REGISTRY_RESOLUTION_REQUEST_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    source_adjudication_sha256: str
    source_adjudication_digest: str
    registry_snapshot_sha256: str
    registry_snapshot_digest: str
    source_adjudication: OledSupplementaryMaterialIdentityAdjudicationArtifact
    registry_snapshot: OledMaterialRegistrySnapshot
    status: OledMaterialRegistryResolutionRequestStatus
    source_review_item_count: Annotated[StrictInt, Field(ge=1)]
    registry_eligible_group_count: Annotated[StrictInt, Field(ge=0)]
    registry_eligible_cell_count: Annotated[StrictInt, Field(ge=0)]
    resolution_item_count: Annotated[StrictInt, Field(ge=0)]
    no_exact_structural_candidate_count: Annotated[StrictInt, Field(ge=0)]
    partial_structural_key_match_count: Annotated[StrictInt, Field(ge=0)]
    consistent_exact_structural_candidate_count: Annotated[StrictInt, Field(ge=0)]
    ambiguous_duplicate_structural_key_count: Annotated[StrictInt, Field(ge=0)]
    conflicting_structural_key_match_count: Annotated[StrictInt, Field(ge=0)]
    registry_conflict_finding_count: Annotated[StrictInt, Field(ge=0)]
    device_only_cell_count: Literal[0] = 0
    registry_conflict_findings: list[OledMaterialRegistryConflictFinding] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    resolution_items: list[OledMaterialRegistryResolutionRequestItem] = Field(
        default_factory=list,
        max_length=1_000_000,
    )
    request_artifact_digest: str
    request_only: StrictBool = True
    offline_only: StrictBool = True
    exact_source_adjudication_bytes_bound: StrictBool = True
    exact_registry_snapshot_bytes_bound: StrictBool = True
    source_adjudication_model_revalidated: StrictBool = True
    registry_snapshot_model_revalidated: StrictBool = True
    complete_registry_eligible_group_coverage_validated: StrictBool = True
    complete_registry_eligible_cell_coverage_validated: StrictBool = True
    deterministic_exact_structural_lookup_performed: StrictBool = True
    exact_alias_literal_hits_reported: StrictBool = True
    alias_hit_used_as_identity_evidence: StrictBool = False
    pr_m_upstream_chain_revalidated: StrictBool = False
    source_pdf_read: StrictBool = False
    human_registry_resolution_completed: StrictBool = False
    material_identity_resolved: StrictBool = False
    canonical_material_id_assigned: StrictBool = False
    registry_written: StrictBool = False
    alias_normalized: StrictBool = False
    cross_paper_identity_merge: StrictBool = False
    automatic_candidate_merge: StrictBool = False
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
        if value != OLED_MATERIAL_REGISTRY_RESOLUTION_REQUEST_VERSION:
            raise ValueError("unexpected Registry resolution request version")
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
        "source_adjudication_sha256",
        "source_adjudication_digest",
        "registry_snapshot_sha256",
        "registry_snapshot_digest",
        "request_artifact_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("registry_conflict_findings")
    @classmethod
    def validate_conflict_order(
        cls,
        value: list[OledMaterialRegistryConflictFinding],
    ) -> list[OledMaterialRegistryConflictFinding]:
        order = [item.finding_digest for item in value]
        if order != sorted(order) or len(order) != len(set(order)):
            raise ValueError("Registry conflict findings must be sorted and unique")
        return value

    @field_validator("resolution_items")
    @classmethod
    def validate_item_order(
        cls,
        value: list[OledMaterialRegistryResolutionRequestItem],
    ) -> list[OledMaterialRegistryResolutionRequestItem]:
        order = [_item_sort_key(item) for item in value]
        if order != sorted(order):
            raise ValueError("Registry resolution items must be source ordered")
        ids = [item.resolution_item_id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("Registry resolution request repeats an item")
        return value

    @model_validator(mode="after")
    def validate_artifact_integrity(
        self,
    ) -> OledMaterialRegistryResolutionRequestArtifact:
        if self.run_id != self.source_adjudication.run_id or self.paper_id != (
            self.source_adjudication.paper_id
        ):
            raise ValueError("Registry resolution request source identity mismatch")
        if _parse_timestamp(self.generated_at) < max(
            _parse_timestamp(self.source_adjudication.generated_at),
            _parse_timestamp(self.registry_snapshot.generated_at),
        ):
            raise ValueError("Registry resolution request predates an input")
        if self.source_adjudication_digest != (
            self.source_adjudication.adjudication_artifact_digest
        ) or self.source_adjudication_digest != (
            oled_supplementary_material_identity_adjudication_artifact_digest(
                self.source_adjudication
            )
        ):
            raise ValueError("Registry request source adjudication digest mismatch")
        if self.registry_snapshot_digest != self.registry_snapshot.snapshot_digest:
            raise ValueError("Registry request snapshot digest mismatch")

        eligible_groups = _eligible_groups(self.source_adjudication)
        expected_conflicts = _relevant_registry_conflict_findings(
            self.registry_snapshot.entries,
            eligible_groups,
        )
        if [item.model_dump(mode="json") for item in self.registry_conflict_findings] != [
            item.model_dump(mode="json") for item in expected_conflicts
        ]:
            raise ValueError("Registry request conflict findings changed")

        expected_items = [
            _build_resolution_item(
                group,
                registry_entries=self.registry_snapshot.entries,
                conflict_findings=expected_conflicts,
            )
            for group in eligible_groups
        ]
        if [item.model_dump(mode="json") for item in self.resolution_items] != [
            item.model_dump(mode="json") for item in expected_items
        ]:
            raise ValueError("Registry resolution request item derivation changed")

        eligible_cell_count = sum(
            group.review_item.validated_result.bound_identity_group
            .identity_dependent_cell_count
            for group in eligible_groups
        )
        if (
            self.source_review_item_count
            != self.source_adjudication.review_item_count
            or self.registry_eligible_group_count != len(eligible_groups)
            or self.registry_eligible_group_count
            != self.source_adjudication.later_registry_review_eligible_group_count
            or self.registry_eligible_cell_count != eligible_cell_count
            or self.registry_eligible_cell_count
            != self.source_adjudication.later_registry_review_eligible_cell_count
            or self.resolution_item_count != len(expected_items)
            or self.registry_conflict_finding_count != len(expected_conflicts)
        ):
            raise ValueError("Registry resolution request coverage mismatch")
        expected_counts = _match_counts(expected_items)
        for field_name, status in (
            (
                "no_exact_structural_candidate_count",
                OledMaterialRegistryMatchStatus.NO_EXACT_STRUCTURAL_CANDIDATE,
            ),
            (
                "partial_structural_key_match_count",
                OledMaterialRegistryMatchStatus.PARTIAL_STRUCTURAL_KEY_MATCH,
            ),
            (
                "consistent_exact_structural_candidate_count",
                OledMaterialRegistryMatchStatus
                .ONE_CONSISTENT_EXACT_STRUCTURAL_CANDIDATE,
            ),
            (
                "ambiguous_duplicate_structural_key_count",
                OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY,
            ),
            (
                "conflicting_structural_key_match_count",
                OledMaterialRegistryMatchStatus.CONFLICTING_STRUCTURAL_KEY_MATCHES,
            ),
        ):
            if getattr(self, field_name) != expected_counts[status]:
                raise ValueError(f"Registry resolution {field_name} mismatch")
        if self.status != _request_status(expected_items, expected_conflicts):
            raise ValueError("Registry resolution request status mismatch")
        if self.device_only_cell_count != 0:
            raise ValueError("device-only cells must remain outside Registry review")

        fixed_true = (
            "request_only",
            "offline_only",
            "exact_source_adjudication_bytes_bound",
            "exact_registry_snapshot_bytes_bound",
            "source_adjudication_model_revalidated",
            "registry_snapshot_model_revalidated",
            "complete_registry_eligible_group_coverage_validated",
            "complete_registry_eligible_cell_coverage_validated",
            "deterministic_exact_structural_lookup_performed",
            "exact_alias_literal_hits_reported",
        )
        fixed_false = (
            "alias_hit_used_as_identity_evidence",
            "pr_m_upstream_chain_revalidated",
            "source_pdf_read",
            "human_registry_resolution_completed",
            "material_identity_resolved",
            "canonical_material_id_assigned",
            "registry_written",
            "alias_normalized",
            "cross_paper_identity_merge",
            "automatic_candidate_merge",
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
            raise ValueError("Registry resolution request crossed its boundary")
        if oled_material_registry_resolution_request_artifact_digest(self) != (
            self.request_artifact_digest
        ):
            raise ValueError("Registry resolution request digest mismatch")
        return self


def build_oled_material_registry_entry(
    *,
    material_id: str,
    canonical_name: str,
    aliases: Sequence[str] = (),
    canonical_isomeric_smiles: str,
) -> OledMaterialRegistryEntry:
    observation = _rdkit_chemistry_observation(
        encoding_kind=OledSupplementaryMaterialIdentityStructureEncodingKind.SMILES,
        structure_text=canonical_isomeric_smiles,
    )
    payload: dict[str, Any] = {
        "entry_version": OLED_MATERIAL_REGISTRY_ENTRY_VERSION,
        "material_id": material_id,
        "canonical_name": canonical_name,
        "aliases": sorted(aliases),
        "canonical_isomeric_smiles": observation["canonical_isomeric_smiles"],
        "standard_inchi": observation["standard_inchi"],
        "inchikey": observation["inchikey"],
        "entry_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryEntry.model_construct(**payload)
    payload["entry_digest"] = oled_material_registry_entry_digest(provisional)
    return OledMaterialRegistryEntry.model_validate(payload)


def build_oled_material_registry_snapshot(
    *,
    registry_id: str,
    registry_version: str,
    generated_at: str,
    entries: Sequence[OledMaterialRegistryEntry],
) -> OledMaterialRegistrySnapshot:
    toolkit_version, inchi_backend_version = _rdkit_runtime_versions()
    validated_entries = sorted(
        (
            OledMaterialRegistryEntry.model_validate(item.model_dump(mode="json"))
            for item in entries
        ),
        key=lambda item: item.material_id,
    )
    payload: dict[str, Any] = {
        "snapshot_version": OLED_MATERIAL_REGISTRY_SNAPSHOT_VERSION,
        "registry_id": registry_id,
        "registry_version": registry_version,
        "generated_at": generated_at,
        "chemistry_profile_version": (
            SUPPLEMENTARY_MATERIAL_IDENTITY_EVIDENCE_RESPONSE_CHEMISTRY_PROFILE_VERSION
        ),
        "toolkit_id": "rdkit",
        "toolkit_version": toolkit_version,
        "inchi_backend_version": inchi_backend_version,
        "entry_count": len(validated_entries),
        "entries": validated_entries,
        "snapshot_digest": "sha256:" + "0" * 64,
        "read_only_snapshot": True,
    }
    provisional = OledMaterialRegistrySnapshot.model_construct(**payload)
    payload["snapshot_digest"] = oled_material_registry_snapshot_digest(provisional)
    return OledMaterialRegistrySnapshot.model_validate(payload)


def build_oled_material_registry_resolution_request_artifact(
    *,
    source_adjudication: OledSupplementaryMaterialIdentityAdjudicationArtifact,
    source_adjudication_sha256: str,
    registry_snapshot: OledMaterialRegistrySnapshot,
    registry_snapshot_sha256: str,
    generated_at: str,
) -> OledMaterialRegistryResolutionRequestArtifact:
    adjudication = OledSupplementaryMaterialIdentityAdjudicationArtifact.model_validate(
        source_adjudication.model_dump(mode="json")
    )
    snapshot = OledMaterialRegistrySnapshot.model_validate(
        registry_snapshot.model_dump(mode="json")
    )
    eligible_groups = _eligible_groups(adjudication)
    conflicts = _relevant_registry_conflict_findings(
        snapshot.entries,
        eligible_groups,
    )
    items = [
        _build_resolution_item(
            group,
            registry_entries=snapshot.entries,
            conflict_findings=conflicts,
        )
        for group in eligible_groups
    ]
    counts = _match_counts(items)
    payload: dict[str, Any] = {
        "artifact_version": OLED_MATERIAL_REGISTRY_RESOLUTION_REQUEST_VERSION,
        "run_id": adjudication.run_id,
        "paper_id": adjudication.paper_id,
        "generated_at": generated_at,
        "source_adjudication_sha256": _normalize_sha256(
            source_adjudication_sha256,
            field_name="source_adjudication_sha256",
        ),
        "source_adjudication_digest": adjudication.adjudication_artifact_digest,
        "registry_snapshot_sha256": _normalize_sha256(
            registry_snapshot_sha256,
            field_name="registry_snapshot_sha256",
        ),
        "registry_snapshot_digest": snapshot.snapshot_digest,
        "source_adjudication": adjudication,
        "registry_snapshot": snapshot,
        "status": _request_status(items, conflicts),
        "source_review_item_count": adjudication.review_item_count,
        "registry_eligible_group_count": len(eligible_groups),
        "registry_eligible_cell_count": sum(
            group.review_item.validated_result.bound_identity_group
            .identity_dependent_cell_count
            for group in eligible_groups
        ),
        "resolution_item_count": len(items),
        "no_exact_structural_candidate_count": counts[
            OledMaterialRegistryMatchStatus.NO_EXACT_STRUCTURAL_CANDIDATE
        ],
        "partial_structural_key_match_count": counts[
            OledMaterialRegistryMatchStatus.PARTIAL_STRUCTURAL_KEY_MATCH
        ],
        "consistent_exact_structural_candidate_count": counts[
            OledMaterialRegistryMatchStatus.ONE_CONSISTENT_EXACT_STRUCTURAL_CANDIDATE
        ],
        "ambiguous_duplicate_structural_key_count": counts[
            OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY
        ],
        "conflicting_structural_key_match_count": counts[
            OledMaterialRegistryMatchStatus.CONFLICTING_STRUCTURAL_KEY_MATCHES
        ],
        "registry_conflict_finding_count": len(conflicts),
        "device_only_cell_count": 0,
        "registry_conflict_findings": conflicts,
        "resolution_items": items,
        "request_artifact_digest": "sha256:" + "0" * 64,
    }
    provisional = OledMaterialRegistryResolutionRequestArtifact.model_construct(
        **payload
    )
    payload["request_artifact_digest"] = (
        oled_material_registry_resolution_request_artifact_digest(provisional)
    )
    return OledMaterialRegistryResolutionRequestArtifact.model_validate(payload)


def render_oled_material_registry_resolution_request_markdown(
    artifact: OledMaterialRegistryResolutionRequestArtifact,
    *,
    request_artifact_sha256: str,
) -> str:
    request = OledMaterialRegistryResolutionRequestArtifact.model_validate(
        artifact.model_dump(mode="json")
    )
    request_sha = _normalize_sha256(
        request_artifact_sha256,
        field_name="request_artifact_sha256",
    )
    lines = [
        "# OLED Material Registry Resolution Review Request",
        "",
        "> This packet is a read-only lookup request. A structural-key hit is a",
        "> candidate for human resolution, not a resolved identity or Registry write.",
        "> Reported-name/alias hits are exact literal hints only and are not identity evidence.",
        "",
        "## Exact bindings",
        "",
        f"- request artifact SHA-256: `{request_sha}`",
        f"- request digest: `{request.request_artifact_digest}`",
        f"- PR-M adjudication SHA-256: `{request.source_adjudication_sha256}`",
        f"- PR-M adjudication digest: `{request.source_adjudication_digest}`",
        f"- Registry snapshot SHA-256: `{request.registry_snapshot_sha256}`",
        f"- Registry snapshot digest: `{request.registry_snapshot_digest}`",
        f"- Registry: `{_md(request.registry_snapshot.registry_id)}` version "
        f"`{_md(request.registry_snapshot.registry_version)}`",
        f"- Registry entry count: `{request.registry_snapshot.entry_count}`",
        f"- chemistry profile: "
        f"`{_md(request.registry_snapshot.chemistry_profile_version)}`",
        f"- RDKit / InChI runtime: `{_md(request.registry_snapshot.toolkit_version)}` / "
        f"`{_md(request.registry_snapshot.inchi_backend_version)}`",
        f"- status: `{request.status.value}`",
        "",
        "## Boundary",
        "",
        "- PR-M accepted paper-local graphs only; unresolved, rejected, anchor-only,",
        "  excluded, ontology-pending, and device-only records are not resolution items.",
        "- Exact SMILES and InChIKey lookup is deterministic but not dispositive.",
        "- No canonical material ID has been assigned; no alias has been normalized;",
        "  no cross-paper merge, Registry write, observation, Gold, dataset, or "
        "training write occurred.",
        "- This stage did not reopen the source PDF or replay the full PR-M upstream chain.",
        "",
        "## Counts",
        "",
        f"- PR-M review items: `{request.source_review_item_count}`",
        f"- Registry-eligible groups: `{request.registry_eligible_group_count}`",
        f"- Registry-eligible property cells: `{request.registry_eligible_cell_count}`",
        f"- no exact structural candidate: `{request.no_exact_structural_candidate_count}`",
        f"- partial structural-key match: `{request.partial_structural_key_match_count}`",
        f"- one consistent exact structural candidate: "
        f"`{request.consistent_exact_structural_candidate_count}`",
        f"- ambiguous duplicate structural key: "
        f"`{request.ambiguous_duplicate_structural_key_count}`",
        f"- conflicting structural-key matches: "
        f"`{request.conflicting_structural_key_match_count}`",
        f"- Registry conflict findings: `{request.registry_conflict_finding_count}`",
        "- device-only cells admitted: `0`",
        "",
    ]
    if request.registry_conflict_findings:
        lines.extend(["## Snapshot conflicts", ""])
        for finding in request.registry_conflict_findings:
            lines.extend(
                [
                    f"- `{finding.finding_kind.value}` / `{_md(finding.key_literal)}`",
                    "  - material IDs: "
                    + ", ".join(
                        f"`{_md(material_id)}`"
                        for material_id in finding.material_ids
                    ),
                    f"  - finding digest: `{finding.finding_digest}`",
                    "  - automatic merge: `false`",
                ]
            )
        lines.append("")

    registry_entries_by_id = {
        entry.material_id: entry for entry in request.registry_snapshot.entries
    }
    for index, item in enumerate(request.resolution_items, start=1):
        group = item.adjudicated_group
        bound = group.review_item.validated_result.bound_identity_group
        result = group.review_item.validated_result
        assert isinstance(
            result.response_result,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        )
        candidate = result.response_result.structure_candidate
        structural_candidate_ids = sorted(
            {
                *item.canonical_smiles_candidate_material_ids,
                *item.inchikey_candidate_material_ids,
            }
        )
        lines.extend(
            [
                f"## R{index:02d}: `{_md(bound.reported_subject_text)}`",
                "",
                "### PR-M accepted paper-local evidence",
                "",
                f"- resolution item ID: `{item.resolution_item_id}`",
                f"- resolution item digest: `{item.resolution_item_digest}`",
                f"- identity group ID: `{bound.identity_group_id}`",
                f"- adjudicated group digest: `{group.adjudicated_group_digest}`",
                f"- table / row: `{_md(bound.table_id)}` / `{bound.row_index}`",
                f"- reported subject literal: `{_md(bound.reported_subject_text)}`",
                f"- dependent property cells: `{bound.identity_dependent_cell_count}`",
                f"- candidate origin: `{candidate.candidate_origin.value}`",
                f"- candidate encoding: `{candidate.structure_encoding_kind.value}`",
                f"- accepted candidate literal: `{_md(candidate.structure_candidate_text)}`",
                f"- candidate canonical isomeric SMILES: "
                f"`{_md(candidate.canonical_isomeric_smiles_candidate)}`",
                f"- candidate InChIKey: `{_md(candidate.inchikey_candidate)}`",
                f"- candidate digest: `{item.candidate_digest}`",
                "",
                "### Deterministic Registry lookup (not a decision)",
                "",
                f"- match status: `{item.match_status.value}`",
                "- canonical-SMILES material IDs: "
                + _md_id_list(item.canonical_smiles_candidate_material_ids),
                "- InChIKey material IDs: "
                + _md_id_list(item.inchikey_candidate_material_ids),
                "- consistent exact candidate material ID: "
                + (
                    f"`{_md(item.consistent_exact_candidate_material_id)}`"
                    if item.consistent_exact_candidate_material_id
                    else "`none`"
                ),
                "- material identity resolved: `false`",
                "",
                "### Registry candidate entry projections (snapshot data, not source evidence)",
                "",
            ]
        )
        if structural_candidate_ids:
            for material_id in structural_candidate_ids:
                entry = registry_entries_by_id[material_id]
                aliases = (
                    ", ".join(f"`{_md(alias)}`" for alias in entry.aliases)
                    if entry.aliases
                    else "`none`"
                )
                lines.extend(
                    [
                        f"- material ID: `{_md(entry.material_id)}`",
                        f"  - preferred name: `{_md(entry.canonical_name)}`",
                        f"  - aliases: {aliases}",
                        f"  - canonical isomeric SMILES: "
                        f"`{_md(entry.canonical_isomeric_smiles)}`",
                        f"  - InChIKey: `{_md(entry.inchikey)}`",
                        f"  - entry digest: `{entry.entry_digest}`",
                    ]
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "### Exact reported-name hints (never identity evidence)",
                "",
            ]
        )
        if item.exact_alias_literal_hits:
            for hit in item.exact_alias_literal_hits:
                lines.append(
                    f"- `{_md(hit.material_id)}` / `{hit.matched_field.value}` / "
                    f"`{_md(hit.matched_literal)}`"
                )
        else:
            lines.append("- none")
        lines.extend(
            [
                "",
                "### Human decision required in the next stage",
                "",
                "- decide whether the paper-local candidate maps to one existing Registry entity,",
                "  requires a new entity, or remains unresolved;",
                "- review stereochemistry, charge/protonation state, mixtures, salts, "
                "and source scope;",
                "- do not infer identity from the reported name alone; and",
                "- preserve every conflict rather than merging automatically.",
                "",
            ]
        )
    if not request.resolution_items:
        lines.extend(
            [
                "## Resolution items",
                "",
                "No PR-M group is eligible for Registry resolution in this artifact.",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def oled_material_registry_entry_digest(entry: OledMaterialRegistryEntry) -> str:
    payload = entry.model_dump(mode="json")
    payload.pop("entry_digest", None)
    return _stable_hash(payload)


def oled_material_registry_snapshot_digest(
    snapshot: OledMaterialRegistrySnapshot,
) -> str:
    payload = snapshot.model_dump(mode="json")
    payload.pop("snapshot_digest", None)
    return _stable_hash(payload)


def oled_material_registry_resolution_request_item_digest(
    item: OledMaterialRegistryResolutionRequestItem,
) -> str:
    payload = item.model_dump(mode="json")
    payload.pop("resolution_item_digest", None)
    return _stable_hash(payload)


def oled_material_registry_resolution_request_artifact_digest(
    artifact: OledMaterialRegistryResolutionRequestArtifact,
) -> str:
    payload = artifact.model_dump(mode="json")
    payload.pop("request_artifact_digest", None)
    return _stable_hash(payload)


def _eligible_groups(
    adjudication: OledSupplementaryMaterialIdentityAdjudicationArtifact,
) -> list[OledSupplementaryAdjudicatedMaterialIdentityGroup]:
    return sorted(
        (
            group
            for group in adjudication.adjudicated_groups
            if group.eligible_for_later_registry_review
        ),
        key=_group_sort_key,
    )


def _group_sort_key(
    group: OledSupplementaryAdjudicatedMaterialIdentityGroup,
) -> tuple[str, str, int, str]:
    bound = group.review_item.validated_result.bound_identity_group
    return (bound.scope_id, bound.table_id, bound.row_index, bound.identity_group_id)


def _item_sort_key(
    item: OledMaterialRegistryResolutionRequestItem,
) -> tuple[str, str, int, str]:
    return _group_sort_key(item.adjudicated_group)


def _build_resolution_item(
    group: OledSupplementaryAdjudicatedMaterialIdentityGroup,
    *,
    registry_entries: Sequence[OledMaterialRegistryEntry],
    conflict_findings: Sequence[OledMaterialRegistryConflictFinding],
) -> OledMaterialRegistryResolutionRequestItem:
    result = group.review_item.validated_result
    if not isinstance(
        result.response_result,
        OledSupplementaryMaterialIdentityProposeStructureCandidate,
    ) or result.chemistry_validation is None:
        raise ValueError("PR-M eligible group lacks a structure candidate")
    candidate = result.response_result.structure_candidate
    smiles_ids = sorted(
        entry.material_id
        for entry in registry_entries
        if entry.canonical_isomeric_smiles
        == candidate.canonical_isomeric_smiles_candidate
    )
    inchikey_ids = sorted(
        entry.material_id
        for entry in registry_entries
        if entry.inchikey == candidate.inchikey_candidate
    )
    match_status, consistent_id = _classify_match(smiles_ids, inchikey_ids)
    reported = result.bound_identity_group.reported_subject_text
    alias_hits = _alias_hits(reported, registry_entries)
    related_keys = {
        candidate.canonical_isomeric_smiles_candidate,
        candidate.inchikey_candidate,
        reported,
    }
    related_conflicts = sorted(
        finding.finding_digest
        for finding in conflict_findings
        if finding.key_literal in related_keys
    )
    payload: dict[str, Any] = {
        "resolution_item_id": _resolution_item_id(group),
        "resolution_item_digest": "sha256:" + "0" * 64,
        "adjudicated_group": group,
        "candidate_digest": result.chemistry_validation.candidate_digest,
        "match_status": match_status,
        "canonical_smiles_candidate_material_ids": smiles_ids,
        "inchikey_candidate_material_ids": inchikey_ids,
        "consistent_exact_candidate_material_id": consistent_id,
        "exact_alias_literal_hits": alias_hits,
        "related_registry_conflict_digests": related_conflicts,
    }
    provisional = OledMaterialRegistryResolutionRequestItem.model_construct(**payload)
    payload["resolution_item_digest"] = (
        oled_material_registry_resolution_request_item_digest(provisional)
    )
    return OledMaterialRegistryResolutionRequestItem.model_validate(payload)


def _classify_match(
    smiles_ids: Sequence[str],
    inchikey_ids: Sequence[str],
) -> tuple[OledMaterialRegistryMatchStatus, str]:
    smiles = set(smiles_ids)
    inchikey = set(inchikey_ids)
    if not smiles and not inchikey:
        return OledMaterialRegistryMatchStatus.NO_EXACT_STRUCTURAL_CANDIDATE, ""
    if len(smiles) > 1 or len(inchikey) > 1:
        return OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY, ""
    if smiles and inchikey:
        if smiles == inchikey:
            material_id = next(iter(smiles))
            return (
                OledMaterialRegistryMatchStatus
                .ONE_CONSISTENT_EXACT_STRUCTURAL_CANDIDATE,
                material_id,
            )
        return OledMaterialRegistryMatchStatus.CONFLICTING_STRUCTURAL_KEY_MATCHES, ""
    return OledMaterialRegistryMatchStatus.PARTIAL_STRUCTURAL_KEY_MATCH, ""


def _alias_hits(
    reported_subject_text: str,
    registry_entries: Sequence[OledMaterialRegistryEntry],
) -> list[OledMaterialRegistryAliasLiteralHit]:
    hits: list[OledMaterialRegistryAliasLiteralHit] = []
    for entry in registry_entries:
        fields: list[tuple[OledMaterialRegistryAliasField, str]] = [
            (OledMaterialRegistryAliasField.CANONICAL_NAME, entry.canonical_name),
            *((OledMaterialRegistryAliasField.ALIAS, alias) for alias in entry.aliases),
        ]
        for matched_field, literal in fields:
            if literal != reported_subject_text:
                continue
            payload: dict[str, Any] = {
                "material_id": entry.material_id,
                "matched_field": matched_field,
                "matched_literal": literal,
                "alias_hit_digest": "sha256:" + "0" * 64,
                "exact_codepoint_match_only": True,
                "identity_evidence": False,
            }
            provisional = OledMaterialRegistryAliasLiteralHit.model_construct(**payload)
            payload["alias_hit_digest"] = _alias_hit_digest(provisional)
            hits.append(OledMaterialRegistryAliasLiteralHit.model_validate(payload))
    return sorted(hits, key=lambda item: item.alias_hit_digest)


def _registry_conflict_findings(
    entries: Sequence[OledMaterialRegistryEntry],
    *,
    relevant_keys: set[str] | None = None,
) -> list[OledMaterialRegistryConflictFinding]:
    keys: dict[tuple[OledMaterialRegistryConflictKind, str], set[str]] = {}
    for entry in entries:
        structural_keys = (
            (
                OledMaterialRegistryConflictKind.DUPLICATE_CANONICAL_SMILES,
                entry.canonical_isomeric_smiles,
            ),
            (OledMaterialRegistryConflictKind.DUPLICATE_INCHIKEY, entry.inchikey),
        )
        for conflict_key in structural_keys:
            if relevant_keys is None or conflict_key[1] in relevant_keys:
                keys.setdefault(conflict_key, set()).add(entry.material_id)
        for literal in {entry.canonical_name, *entry.aliases}:
            if relevant_keys is not None and literal not in relevant_keys:
                continue
            keys.setdefault(
                (
                    OledMaterialRegistryConflictKind.DUPLICATE_REPORTED_NAME_LITERAL,
                    literal,
                ),
                set(),
            ).add(entry.material_id)
    findings: list[OledMaterialRegistryConflictFinding] = []
    for (kind, literal), material_ids in keys.items():
        if len(material_ids) < 2:
            continue
        payload: dict[str, Any] = {
            "finding_kind": kind,
            "key_literal": literal,
            "material_ids": sorted(material_ids),
            "finding_digest": "sha256:" + "0" * 64,
            "automatic_merge_performed": False,
        }
        provisional = OledMaterialRegistryConflictFinding.model_construct(**payload)
        payload["finding_digest"] = _registry_conflict_finding_digest(provisional)
        findings.append(OledMaterialRegistryConflictFinding.model_validate(payload))
    return sorted(findings, key=lambda item: item.finding_digest)


def _relevant_registry_conflict_findings(
    entries: Sequence[OledMaterialRegistryEntry],
    eligible_groups: Sequence[OledSupplementaryAdjudicatedMaterialIdentityGroup],
) -> list[OledMaterialRegistryConflictFinding]:
    """Keep the packet bounded to conflicts that touch a PR-M accepted candidate."""

    relevant_keys: set[str] = set()
    for group in eligible_groups:
        result = group.review_item.validated_result
        if not isinstance(
            result.response_result,
            OledSupplementaryMaterialIdentityProposeStructureCandidate,
        ):
            raise ValueError("PR-M eligible group lacks a structure candidate")
        candidate = result.response_result.structure_candidate
        relevant_keys.update(
            {
                candidate.canonical_isomeric_smiles_candidate,
                candidate.inchikey_candidate,
                result.bound_identity_group.reported_subject_text,
            }
        )
    return _registry_conflict_findings(entries, relevant_keys=relevant_keys)


def _resolution_item_id(
    group: OledSupplementaryAdjudicatedMaterialIdentityGroup,
) -> str:
    identity = {
        "adjudicated_group_digest": group.adjudicated_group_digest,
        "identity_group_id": (
            group.review_item.validated_result.bound_identity_group.identity_group_id
        ),
    }
    return f"material-registry-resolution:{_stable_hash(identity)[7:31]}"


def _request_status(
    items: Sequence[OledMaterialRegistryResolutionRequestItem],
    conflicts: Sequence[OledMaterialRegistryConflictFinding],
) -> OledMaterialRegistryResolutionRequestStatus:
    if not items:
        return (
            OledMaterialRegistryResolutionRequestStatus
            .NO_REGISTRY_ELIGIBLE_CANDIDATES
        )
    if conflicts or any(
        item.match_status
        in {
            OledMaterialRegistryMatchStatus.AMBIGUOUS_DUPLICATE_STRUCTURAL_KEY,
            OledMaterialRegistryMatchStatus.CONFLICTING_STRUCTURAL_KEY_MATCHES,
        }
        for item in items
    ):
        return (
            OledMaterialRegistryResolutionRequestStatus
            .REGISTRY_CONFLICTS_REQUIRE_HUMAN_REVIEW
        )
    return (
        OledMaterialRegistryResolutionRequestStatus
        .READY_FOR_HUMAN_REGISTRY_RESOLUTION_REVIEW
    )


def _match_counts(
    items: Sequence[OledMaterialRegistryResolutionRequestItem],
) -> dict[OledMaterialRegistryMatchStatus, int]:
    return {
        status: sum(item.match_status == status for item in items)
        for status in OledMaterialRegistryMatchStatus
    }


def _alias_hit_digest(hit: OledMaterialRegistryAliasLiteralHit) -> str:
    payload = hit.model_dump(mode="json")
    payload.pop("alias_hit_digest", None)
    return _stable_hash(payload)


def _registry_conflict_finding_digest(
    finding: OledMaterialRegistryConflictFinding,
) -> str:
    payload = finding.model_dump(mode="json")
    payload.pop("finding_digest", None)
    return _stable_hash(payload)


def _validate_registry_name(value: Any, *, field_name: str, required: bool) -> str:
    clean = validate_oled_supplementary_safe_authored_text(
        value,
        field_name=field_name,
        required=required,
        max_length=4_000,
    )
    decoded = html.unescape(clean)
    if decoded != clean:
        raise ValueError(f"{field_name} must not contain HTML entities")
    return clean


def _validate_exact_chemical_literal(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 20_000:
        raise ValueError(f"{field_name} is required and bounded")
    if value != value.strip() or any(character.isspace() for character in value):
        raise ValueError(f"{field_name} must be one exact whitespace-free literal")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} contains control text")
    return value


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


def _md_id_list(values: Sequence[str]) -> str:
    if not values:
        return "`none`"
    return ", ".join(f"`{_md(value)}`" for value in values)


__all__ = [
    "OLED_MATERIAL_REGISTRY_ENTRY_VERSION",
    "OLED_MATERIAL_REGISTRY_RESOLUTION_REQUEST_VERSION",
    "OLED_MATERIAL_REGISTRY_SNAPSHOT_VERSION",
    "OledMaterialRegistryAliasField",
    "OledMaterialRegistryAliasLiteralHit",
    "OledMaterialRegistryConflictFinding",
    "OledMaterialRegistryConflictKind",
    "OledMaterialRegistryEntry",
    "OledMaterialRegistryMatchStatus",
    "OledMaterialRegistryResolutionRequestArtifact",
    "OledMaterialRegistryResolutionRequestItem",
    "OledMaterialRegistryResolutionRequestStatus",
    "OledMaterialRegistrySnapshot",
    "build_oled_material_registry_entry",
    "build_oled_material_registry_resolution_request_artifact",
    "build_oled_material_registry_snapshot",
    "oled_material_registry_entry_digest",
    "oled_material_registry_resolution_request_artifact_digest",
    "oled_material_registry_resolution_request_item_digest",
    "oled_material_registry_snapshot_digest",
    "render_oled_material_registry_resolution_request_markdown",
]
