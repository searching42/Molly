from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from ai4s_agent.domains.oled_contracts import OledCausalLayer
from ai4s_agent.domains.oled_property_ontology import DEFAULT_OLED_PROPERTY_ONTOLOGY
from ai4s_agent.domains.oled_supplementary_evidence_recovery import OledSupplementaryTargetKind
from ai4s_agent.domains.oled_supplementary_locator_adjudication import (
    OledSupplementaryLocatorAdjudicatedItem,
    OledSupplementaryLocatorAdjudicationArtifact,
    OledSupplementaryLocatorDecision,
    oled_supplementary_review_item_digest,
    validate_oled_supplementary_audit_text,
)
from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryLocatorMatchStatus,
    OledSupplementaryLocatorReviewArtifact,
    OledSupplementaryLocatorReviewItem,
    OledSupplementaryReviewTable,
)


SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION = (
    "oled_supplementary_scoped_candidate_request.v1"
)
SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION = (
    "oled_supplementary_scoped_candidate_request_ontology.v1"
)
SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_DIGEST = (
    "sha256:8e3d316497d4dd45e989c8c13604810b84f2ac381268e167d6268f4af758cd20"
)

OledSupplementaryCandidateRequestStatus = Literal["ready_for_semantic_proposal"]
OledSupplementaryDatasetScope = Literal["molecule_interaction_properties_only"]

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_DATASET_LAYERS = frozenset({OledCausalLayer.MOLECULE, OledCausalLayer.INTERACTION})
_ALLOWED_LAYER_ORDER = [OledCausalLayer.INTERACTION, OledCausalLayer.MOLECULE]

_PROPOSAL_INSTRUCTIONS = (
    "Read each bound table caption, headers, rows, and footnotes together.",
    (
        "Treat the source PDF as authoritative; the copied parsed table remains an "
        "unvalidated transcription."
    ),
    "Propose only molecule- or interaction-layer properties; exclude device-only records.",
    "Preserve every reported label and cell string verbatim, including signs and trailing zeros.",
    "Do not swap, correct, or normalize HOMO/LUMO or any other reported label or value.",
    "Treat each non-empty semantic_note as a mandatory unresolved issue for that scope.",
    (
        "Bind every proposed observation to scope_id, table_id, zero-based row_index, "
        "column_name, and the exact cell_value."
    ),
    (
        "Use only a same-row reported subject cell; do not infer canonical identity, structure, "
        "SMILES, or material role."
    ),
    "Do not force an unsupported property into the ontology; request ontology review instead.",
    "Do not infer that table cells omitted from a proposal are absent, invalid, or irrelevant.",
    (
        "Every proposal remains pending human review and must not be compiled, merged, staged, "
        "admitted, converted to gold, or written to a dataset."
    ),
    "Return data only; do not return executable code, scripts, credentials, URLs, or local paths.",
)


class OledSupplementaryScopedOntologyProperty(BaseModel):
    """Dataset-scoped ontology snapshot supplied as proposal context, never as a mapping result."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    property_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    allowed_layers: list[OledCausalLayer]
    canonical_unit: str
    physical_interpretation: str
    required_comparison_context_fields: list[str] = Field(default_factory=list)

    @field_validator("property_id")
    @classmethod
    def validate_property_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="property_id")

    @field_validator("name", "canonical_unit", "physical_interpretation")
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError(f"{info.field_name} is required")
        return clean

    @field_validator("aliases", "required_comparison_context_fields")
    @classmethod
    def validate_sorted_unique_text(cls, value: list[str], info: Any) -> list[str]:
        clean = [str(item).strip() for item in value]
        if any(not item for item in clean):
            raise ValueError(f"{info.field_name} must not contain empty values")
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @field_validator("allowed_layers")
    @classmethod
    def validate_allowed_layers(cls, value: list[OledCausalLayer]) -> list[OledCausalLayer]:
        if not value or any(layer not in _DATASET_LAYERS for layer in value):
            raise ValueError("ontology request properties must stay within dataset layers")
        if value != sorted(set(value), key=lambda layer: layer.value):
            raise ValueError("allowed_layers must be sorted and unique")
        return value


class OledSupplementaryScopedCandidateRequestScope(BaseModel):
    """One accepted locator and its literal table content, ready only for a proposal step."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    scope_id: str
    review_item_id: str
    recovery_item_id: str
    source_review_item_digest: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    parser_backend: str
    target_kind: OledSupplementaryTargetKind
    target_locator: str
    canonical_locator: str
    match_status: OledSupplementaryLocatorMatchStatus
    matched_table: OledSupplementaryReviewTable
    parser_warning_codes: list[str] = Field(default_factory=list)
    semantic_note: str = ""
    semantic_review_required: StrictBool
    dataset_scope: OledSupplementaryDatasetScope = "molecule_interaction_properties_only"
    allowed_layers: list[OledCausalLayer] = Field(
        default_factory=lambda: list(_ALLOWED_LAYER_ORDER)
    )
    proposal_instructions: list[str] = Field(
        default_factory=lambda: list(_PROPOSAL_INSTRUCTIONS)
    )
    human_review_required: StrictBool = True
    source_pdf_remains_authoritative: StrictBool = True
    parsed_table_is_authoritative: StrictBool = False
    reported_labels_must_be_preserved: StrictBool = True
    reported_values_must_be_preserved: StrictBool = True
    table_exhaustiveness_validated: StrictBool = False
    table_transcription_validated: StrictBool = False
    scientific_content_validated: StrictBool = False
    physical_semantics_validated: StrictBool = False
    schema_mapping_performed: StrictBool = False
    schema_candidates_created: StrictBool = False
    direct_admission_eligible: StrictBool = False

    @field_validator("scope_id", "review_item_id", "recovery_item_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator(
        "source_review_item_digest",
        "source_pdf_sha256",
        "parsed_document_sha256",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("parser_warning_codes")
    @classmethod
    def validate_warning_codes(cls, value: list[str]) -> list[str]:
        clean = [_validate_bound_id(item, field_name="parser_warning_code") for item in value]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("parser_warning_codes must be sorted and unique")
        return clean

    @field_validator("semantic_note")
    @classmethod
    def validate_semantic_note(cls, value: str) -> str:
        return validate_oled_supplementary_audit_text(
            value,
            field_name="semantic_note",
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_scope_integrity(self) -> OledSupplementaryScopedCandidateRequestScope:
        if self.scope_id != _scope_id(self):
            raise ValueError("supplementary candidate request scope_id mismatch")
        if self.target_kind != OledSupplementaryTargetKind.TABLE:
            raise ValueError("supplementary candidate requests support accepted tables only")
        if self.match_status != OledSupplementaryLocatorMatchStatus.EXACT_MATCH:
            raise ValueError("supplementary candidate requests require an exact locator match")
        if self.semantic_review_required != bool(self.semantic_note):
            raise ValueError("semantic_review_required must mirror semantic_note")
        if self.allowed_layers != _ALLOWED_LAYER_ORDER:
            raise ValueError("supplementary candidate requests must use dataset layers only")
        if self.proposal_instructions != list(_PROPOSAL_INSTRUCTIONS):
            raise ValueError("supplementary candidate request instructions were changed")
        fixed_true_flags = (
            "human_review_required",
            "source_pdf_remains_authoritative",
            "reported_labels_must_be_preserved",
            "reported_values_must_be_preserved",
        )
        fixed_false_flags = (
            "parsed_table_is_authoritative",
            "table_exhaustiveness_validated",
            "table_transcription_validated",
            "scientific_content_validated",
            "physical_semantics_validated",
            "schema_mapping_performed",
            "schema_candidates_created",
            "direct_admission_eligible",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true_flags):
            raise ValueError("supplementary candidate request lost a required scope flag")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary candidate request scope crossed a review boundary")
        return self


class OledSupplementaryScopedCandidateRequestArtifact(BaseModel):
    """Exact-bound request context with no response, mapping, candidate, or admission authority."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: str = SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    review_artifact_sha256: str
    review_artifact_digest: str
    adjudication_artifact_sha256: str
    adjudication_artifact_digest: str
    status: OledSupplementaryCandidateRequestStatus
    source_count: int = Field(ge=1)
    scope_count: int = Field(ge=1)
    semantic_review_required_count: int = Field(ge=0)
    ontology_version: str = SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION
    ontology: list[OledSupplementaryScopedOntologyProperty] = Field(default_factory=list)
    ontology_snapshot_digest: str
    scopes: list[OledSupplementaryScopedCandidateRequestScope] = Field(default_factory=list)
    request_digest: str
    request_only: StrictBool = True
    candidate_proposal_requested: StrictBool = True
    response_received: StrictBool = False
    response_validation_implemented: StrictBool = False
    offline_only: StrictBool = True
    human_review_required: StrictBool = True
    review_artifact_read: StrictBool = True
    adjudication_artifact_read: StrictBool = True
    matched_table_content_copied: StrictBool = True
    scientific_content_included: StrictBool = True
    parsed_output_read: StrictBool = False
    pdf_content_read: StrictBool = False
    network_accessed: StrictBool = False
    external_service_called: StrictBool = False
    llm_called: StrictBool = False
    mineru_called: StrictBool = False
    candidate_regenerated: StrictBool = False
    automatic_candidate_merge: StrictBool = False
    reviewed_evidence_staging: StrictBool = False
    device_only_admitted: StrictBool = False
    gold_records_created: StrictBool = False
    dataset_written: StrictBool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION:
            raise ValueError("unexpected supplementary scoped candidate request artifact_version")
        return value

    @field_validator("ontology_version")
    @classmethod
    def validate_ontology_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION:
            raise ValueError("unexpected supplementary candidate request ontology_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "review_artifact_sha256",
        "review_artifact_digest",
        "adjudication_artifact_sha256",
        "adjudication_artifact_digest",
        "ontology_snapshot_digest",
        "request_digest",
    )
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("generated_at")
    @classmethod
    def validate_generated_at(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("generated_at is required")
        try:
            parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("generated_at must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("generated_at must include a timezone")
        return clean

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledSupplementaryScopedCandidateRequestArtifact:
        if not self.scopes or self.scope_count != len(self.scopes):
            raise ValueError("supplementary candidate request scope_count mismatch")
        scope_ids = [scope.scope_id for scope in self.scopes]
        if scope_ids != sorted(scope_ids) or len(scope_ids) != len(set(scope_ids)):
            raise ValueError("supplementary candidate request scopes must be sorted and unique")
        review_ids = [scope.review_item_id for scope in self.scopes]
        if len(review_ids) != len(set(review_ids)):
            raise ValueError("supplementary candidate request repeats a review item")
        if self.source_count != len({scope.source_id for scope in self.scopes}):
            raise ValueError("supplementary candidate request source_count mismatch")
        semantic_count = sum(scope.semantic_review_required for scope in self.scopes)
        if self.semantic_review_required_count != semantic_count:
            raise ValueError("supplementary candidate request semantic count mismatch")
        if not self.ontology:
            raise ValueError("supplementary candidate request requires an ontology snapshot")
        property_ids = [definition.property_id for definition in self.ontology]
        if property_ids != sorted(property_ids) or len(property_ids) != len(set(property_ids)):
            raise ValueError("supplementary candidate request ontology must be sorted and unique")
        if _stable_hash([item.model_dump(mode="json") for item in self.ontology]) != (
            self.ontology_snapshot_digest
        ):
            raise ValueError("supplementary candidate request ontology digest mismatch")
        if self.ontology_snapshot_digest != (
            SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_DIGEST
        ):
            raise ValueError("supplementary candidate request ontology is not the pinned snapshot")
        expected_ontology = [
            item.model_dump(mode="json") for item in _dataset_ontology_snapshot()
        ]
        if [item.model_dump(mode="json") for item in self.ontology] != expected_ontology:
            raise ValueError("supplementary candidate request ontology content was changed")
        fixed_true_flags = (
            "request_only",
            "candidate_proposal_requested",
            "offline_only",
            "human_review_required",
            "review_artifact_read",
            "adjudication_artifact_read",
            "matched_table_content_copied",
            "scientific_content_included",
        )
        fixed_false_flags = (
            "response_received",
            "response_validation_implemented",
            "parsed_output_read",
            "pdf_content_read",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true_flags):
            raise ValueError("supplementary candidate request lost a required audit flag")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary candidate request crossed a downstream boundary")
        if _request_artifact_digest(self) != self.request_digest:
            raise ValueError("supplementary candidate request digest mismatch")
        return self


def validate_oled_supplementary_scoped_candidate_request_binding(
    *,
    review_artifact: OledSupplementaryLocatorReviewArtifact,
    review_artifact_sha256: str,
    adjudication_artifact: OledSupplementaryLocatorAdjudicationArtifact,
) -> None:
    """Fail closed unless PR-E and PR-F are the same complete, content-bound review chain."""

    review = OledSupplementaryLocatorReviewArtifact.model_validate(
        review_artifact.model_dump(mode="json")
    )
    adjudication = OledSupplementaryLocatorAdjudicationArtifact.model_validate(
        adjudication_artifact.model_dump(mode="json")
    )
    observed_review_sha256 = _normalize_sha256(
        review_artifact_sha256,
        field_name="review_artifact_sha256",
    )
    if adjudication.run_id != review.run_id or adjudication.paper_id != review.paper_id:
        raise ValueError("supplementary candidate request input identities do not match")
    if adjudication.review_artifact_sha256 != observed_review_sha256:
        raise ValueError("adjudication does not bind the exact review artifact bytes")
    if adjudication.review_artifact_digest != review.review_artifact_digest:
        raise ValueError("adjudication does not bind canonical review content")
    if adjudication.execution_artifact_sha256 != review.execution_artifact_sha256:
        raise ValueError("supplementary candidate request execution byte binding mismatch")
    if adjudication.execution_artifact_digest != review.execution_artifact_digest:
        raise ValueError("supplementary candidate request execution content binding mismatch")
    if adjudication.locator_manifest_sha256 != review.locator_manifest_sha256:
        raise ValueError("supplementary candidate request locator manifest binding mismatch")
    if adjudication.preflight_plan_digest != review.preflight_plan_digest:
        raise ValueError("supplementary candidate request preflight binding mismatch")
    review_items = {item.review_item_id: item for item in review.review_items}
    adjudicated_items = {
        item.review_item_id: item for item in adjudication.adjudicated_items
    }
    if set(adjudicated_items) != set(review_items):
        raise ValueError("supplementary candidate adjudication must exactly cover review items")
    for review_item_id, source_item in review_items.items():
        _validate_item_binding(source_item, adjudicated_items[review_item_id])
    if not any(
        item.eligible_for_later_scoped_candidate_proposal
        for item in adjudication.adjudicated_items
    ):
        raise ValueError("supplementary candidate request requires an accepted eligible locator")


def build_oled_supplementary_scoped_candidate_request_artifact(
    *,
    review_artifact: OledSupplementaryLocatorReviewArtifact,
    review_artifact_sha256: str,
    adjudication_artifact: OledSupplementaryLocatorAdjudicationArtifact,
    adjudication_artifact_sha256: str,
    generated_at: str,
) -> OledSupplementaryScopedCandidateRequestArtifact:
    review = OledSupplementaryLocatorReviewArtifact.model_validate(
        review_artifact.model_dump(mode="json")
    )
    adjudication = OledSupplementaryLocatorAdjudicationArtifact.model_validate(
        adjudication_artifact.model_dump(mode="json")
    )
    validate_oled_supplementary_scoped_candidate_request_binding(
        review_artifact=review,
        review_artifact_sha256=review_artifact_sha256,
        adjudication_artifact=adjudication,
    )
    normalized_adjudication_sha256 = _normalize_sha256(
        adjudication_artifact_sha256,
        field_name="adjudication_artifact_sha256",
    )
    review_items = {item.review_item_id: item for item in review.review_items}
    scopes = [
        _build_scope(review_items[item.review_item_id], item)
        for item in adjudication.adjudicated_items
        if item.eligible_for_later_scoped_candidate_proposal
    ]
    scopes.sort(key=lambda scope: scope.scope_id)
    ontology = _dataset_ontology_snapshot()
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION,
        "run_id": review.run_id,
        "paper_id": review.paper_id,
        "generated_at": str(generated_at or "").strip(),
        "review_artifact_sha256": _normalize_sha256(
            review_artifact_sha256,
            field_name="review_artifact_sha256",
        ),
        "review_artifact_digest": review.review_artifact_digest,
        "adjudication_artifact_sha256": normalized_adjudication_sha256,
        "adjudication_artifact_digest": adjudication.adjudication_artifact_digest,
        "status": "ready_for_semantic_proposal",
        "source_count": len({scope.source_id for scope in scopes}),
        "scope_count": len(scopes),
        "semantic_review_required_count": sum(
            scope.semantic_review_required for scope in scopes
        ),
        "ontology_version": SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION,
        "ontology": [item.model_dump(mode="json") for item in ontology],
        "ontology_snapshot_digest": _stable_hash(
            [item.model_dump(mode="json") for item in ontology]
        ),
        "scopes": [scope.model_dump(mode="json") for scope in scopes],
        "request_digest": "",
        "request_only": True,
        "candidate_proposal_requested": True,
        "response_received": False,
        "response_validation_implemented": False,
        "offline_only": True,
        "human_review_required": True,
        "review_artifact_read": True,
        "adjudication_artifact_read": True,
        "matched_table_content_copied": True,
        "scientific_content_included": True,
        "parsed_output_read": False,
        "pdf_content_read": False,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["request_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "request_digest"}
    )
    return OledSupplementaryScopedCandidateRequestArtifact.model_validate(payload)


def _validate_item_binding(
    source_item: OledSupplementaryLocatorReviewItem,
    adjudicated_item: OledSupplementaryLocatorAdjudicatedItem,
) -> None:
    if adjudicated_item.source_review_item_digest != oled_supplementary_review_item_digest(
        source_item
    ):
        raise ValueError("supplementary candidate request review item digest mismatch")
    expected_pairs = (
        ("recovery_item_id", source_item.recovery_item_id),
        ("source_id", source_item.source_id),
        ("source_pdf_sha256", source_item.source_pdf_sha256),
        ("parsed_document_sha256", source_item.parsed_document_sha256),
        ("parser_backend", source_item.parser_backend),
        ("target_kind", source_item.target_kind),
        ("target_locator", source_item.target_locator),
        ("canonical_locator", source_item.canonical_locator),
        ("match_status", source_item.match_status),
        ("parser_warning_codes", source_item.parser_warning_codes),
    )
    for field_name, expected in expected_pairs:
        if getattr(adjudicated_item, field_name) != expected:
            raise ValueError(f"supplementary candidate request {field_name} binding mismatch")
    table = source_item.matched_table
    observed_table_binding = (
        adjudicated_item.matched_table_id,
        adjudicated_item.matched_table_page,
        adjudicated_item.table_content_digest,
    )
    expected_table_binding = (
        table.table_id if table is not None else "",
        table.page if table is not None else None,
        table.table_content_digest if table is not None else "",
    )
    if observed_table_binding != expected_table_binding:
        raise ValueError("supplementary candidate request table binding mismatch")
    eligible = adjudicated_item.eligible_for_later_scoped_candidate_proposal
    accepted = adjudicated_item.decision == OledSupplementaryLocatorDecision.ACCEPT_LOCATOR
    if eligible != accepted or adjudicated_item.locator_accepted != accepted:
        raise ValueError("supplementary candidate request eligibility binding mismatch")
    if eligible and (
        source_item.match_status != OledSupplementaryLocatorMatchStatus.EXACT_MATCH
        or table is None
    ):
        raise ValueError("eligible supplementary candidate request requires exact table content")


def _build_scope(
    source_item: OledSupplementaryLocatorReviewItem,
    adjudicated_item: OledSupplementaryLocatorAdjudicatedItem,
) -> OledSupplementaryScopedCandidateRequestScope:
    table = source_item.matched_table
    if table is None:
        raise ValueError("eligible supplementary candidate request is missing table content")
    base: dict[str, Any] = {
        "scope_id": "",
        "review_item_id": source_item.review_item_id,
        "recovery_item_id": source_item.recovery_item_id,
        "source_review_item_digest": adjudicated_item.source_review_item_digest,
        "source_id": source_item.source_id,
        "source_pdf_sha256": source_item.source_pdf_sha256,
        "parsed_document_sha256": source_item.parsed_document_sha256,
        "parser_backend": source_item.parser_backend,
        "target_kind": source_item.target_kind,
        "target_locator": source_item.target_locator,
        "canonical_locator": source_item.canonical_locator,
        "match_status": source_item.match_status,
        "matched_table": table.model_dump(mode="json"),
        "parser_warning_codes": source_item.parser_warning_codes,
        "semantic_note": adjudicated_item.semantic_note,
        "semantic_review_required": adjudicated_item.semantic_review_required,
        "dataset_scope": "molecule_interaction_properties_only",
        "allowed_layers": [layer.value for layer in _ALLOWED_LAYER_ORDER],
        "proposal_instructions": list(_PROPOSAL_INSTRUCTIONS),
        "human_review_required": True,
        "source_pdf_remains_authoritative": True,
        "parsed_table_is_authoritative": False,
        "reported_labels_must_be_preserved": True,
        "reported_values_must_be_preserved": True,
        "table_exhaustiveness_validated": False,
        "table_transcription_validated": False,
        "scientific_content_validated": False,
        "physical_semantics_validated": False,
        "schema_mapping_performed": False,
        "schema_candidates_created": False,
        "direct_admission_eligible": False,
    }
    base["scope_id"] = _scope_id(base)
    return OledSupplementaryScopedCandidateRequestScope.model_validate(base)


def _dataset_ontology_snapshot() -> list[OledSupplementaryScopedOntologyProperty]:
    snapshot: list[OledSupplementaryScopedOntologyProperty] = []
    for definition in DEFAULT_OLED_PROPERTY_ONTOLOGY.list_properties():
        allowed_layers = sorted(
            definition.allowed_layers & _DATASET_LAYERS,
            key=lambda item: item.value,
        )
        if not allowed_layers:
            continue
        required_fields = definition.metadata.get("required_comparison_context_fields", [])
        snapshot.append(
            OledSupplementaryScopedOntologyProperty(
                property_id=definition.property_id,
                name=definition.name,
                aliases=sorted(definition.aliases),
                allowed_layers=allowed_layers,
                canonical_unit=definition.canonical_unit,
                physical_interpretation=definition.physical_interpretation,
                required_comparison_context_fields=sorted(
                    str(item).strip() for item in required_fields if str(item).strip()
                ),
            )
        )
    return sorted(snapshot, key=lambda item: item.property_id)


def _scope_id(value: OledSupplementaryScopedCandidateRequestScope | dict[str, Any]) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump(mode="json")
    else:
        payload = dict(value)
    identity = {
        "review_item_id": payload["review_item_id"],
        "source_review_item_digest": payload["source_review_item_digest"],
        "source_id": payload["source_id"],
        "table_id": payload["matched_table"]["table_id"],
        "table_content_digest": payload["matched_table"]["table_content_digest"],
        "semantic_note": payload["semantic_note"],
    }
    return f"supplementary-scoped-request:{_stable_hash(identity).split(':', 1)[1][:24]}"


def _validate_path_segment(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_PATH_SEGMENT_RE.fullmatch(clean) or clean in {".", ".."}:
        raise ValueError(f"{field_name} must be a safe path segment")
    return clean


def _validate_bound_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_BOUND_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} contains unsupported characters")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _stable_hash(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _request_artifact_digest(
    artifact: OledSupplementaryScopedCandidateRequestArtifact,
) -> str:
    return _stable_hash(artifact.model_dump(mode="json", exclude={"request_digest"}))


__all__ = [
    "SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ARTIFACT_VERSION",
    "SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_DIGEST",
    "SUPPLEMENTARY_SCOPED_CANDIDATE_REQUEST_ONTOLOGY_VERSION",
    "OledSupplementaryScopedCandidateRequestArtifact",
    "OledSupplementaryScopedCandidateRequestScope",
    "OledSupplementaryScopedOntologyProperty",
    "build_oled_supplementary_scoped_candidate_request_artifact",
    "validate_oled_supplementary_scoped_candidate_request_binding",
]
