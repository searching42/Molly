from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from ai4s_agent.domains.oled_supplementary_locator_review import (
    OledSupplementaryLocatorMatchStatus,
    OledSupplementaryLocatorReviewArtifact,
    OledSupplementaryLocatorReviewItem,
)
from ai4s_agent.domains.oled_supplementary_evidence_recovery import OledSupplementaryTargetKind


SUPPLEMENTARY_LOCATOR_DECISION_MANIFEST_VERSION = (
    "oled_supplementary_locator_decision_manifest.v1"
)
SUPPLEMENTARY_LOCATOR_ADJUDICATION_ARTIFACT_VERSION = (
    "oled_supplementary_locator_adjudication.v1"
)

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![0-9A-Za-z])(?:[/\\]{2}|/|~[/\\]|[A-Za-z]:[/\\])"
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"""
    (?<![0-9A-Za-z])
    [\"']?
    (?:
        x[\s_-]*api[\s_-]*key
        | api[\s_-]*key
        | access[\s_-]*token
        | auth[\s_-]*token
        | refresh[\s_-]*token
        | session[\s_-]*token
        | authorization
        | bearer
        | credentials?
        | password
        | client[\s_-]*secret
        | private[\s_-]*key
        | secret
        | token
        | cookie
    )
    [\"']?
    \s*[:=]
    """,
    re.IGNORECASE | re.VERBOSE,
)
_BEARER_TOKEN_PATTERN = r"""
    (?:
        (?=[A-Za-z0-9._~+/=-]{6,}(?![A-Za-z0-9._~+/=-]))
        (?=[A-Za-z0-9._~+/=-]*(?:[0-9]|[._~+/=][A-Za-z0-9]))
        [A-Za-z0-9._~+/=-]{6,}
        | [A-Za-z]{16,}(?![A-Za-z0-9_-])
    )
"""
_BEARER_CREDENTIAL_RE = re.compile(
    rf"""
    \A
    bearer[ \t]+
    {_BEARER_TOKEN_PATTERN}
    """,
    re.IGNORECASE | re.VERBOSE,
)
_INLINE_BEARER_CREDENTIAL_RE = re.compile(
    r"""
    (?<![0-9A-Za-z])
    (?:
        use(?:d|s|ing)?
        | credentials?
        | auth(?:entication|orization)?
        | http[ \t]+header
        | authorization[ \t]+header
        | http[ \t]+authorization
    )
    (?:
        [ \t]+(?:is|as|the|a|an|value|scheme|token|with|uses?|using|set[ \t]+to)
    ){0,3}
    [ \t]*[:=,-]?[ \t]+
    bearer[ \t]+
    """
    + _BEARER_TOKEN_PATTERN,
    re.IGNORECASE | re.VERBOSE,
)
_SECRET_KEY_RE = re.compile(
    r"(?<![0-9A-Za-z])sk-[A-Za-z0-9_-]{6,}(?![A-Za-z0-9_-])",
)


class OledSupplementaryLocatorDecision(str, Enum):
    ACCEPT_LOCATOR = "accept_locator"
    REJECT_LOCATOR = "reject_locator"
    NEEDS_SOURCE_CHECK = "needs_source_check"


class OledSupplementaryLocatorAdjudicationStatus(str, Enum):
    ALL_LOCATORS_ACCEPTED = "all_locators_accepted"
    PARTIALLY_ACCEPTED = "partially_accepted"
    NO_LOCATORS_ACCEPTED = "no_locators_accepted"


class OledSupplementaryLocatorDecisionEntry(BaseModel):
    """One human decision about a locator selection, never about data admission."""

    model_config = ConfigDict(extra="forbid")

    review_item_id: str
    decision: OledSupplementaryLocatorDecision
    reviewed_by: str
    reviewed_at: str
    review_note: str = ""
    semantic_note: str = ""

    @field_validator("review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="review_item_id")

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="reviewed_by", required=True, max_length=200)

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("reviewed_at is required")
        try:
            parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at must be ISO-8601") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("reviewed_at must include a timezone")
        return clean

    @field_validator("review_note", "semantic_note")
    @classmethod
    def validate_notes(cls, value: str, info: Any) -> str:
        return _validate_audit_text(
            value,
            field_name=str(info.field_name),
            required=False,
            max_length=2_000,
        )

    @model_validator(mode="after")
    def validate_decision_shape(self) -> OledSupplementaryLocatorDecisionEntry:
        if self.decision in {
            OledSupplementaryLocatorDecision.REJECT_LOCATOR,
            OledSupplementaryLocatorDecision.NEEDS_SOURCE_CHECK,
        } and not self.review_note:
            raise ValueError("rejected or source-check locator decisions require review_note")
        return self


class OledSupplementaryLocatorDecisionManifest(BaseModel):
    """Exact-byte-bound human decisions for every item in one PR-E artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_LOCATOR_DECISION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    review_artifact_sha256: str
    review_artifact_digest: str
    adjudication_confirmed: StrictBool = False
    decisions: list[OledSupplementaryLocatorDecisionEntry] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_LOCATOR_DECISION_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary locator decision manifest schema_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("review_artifact_sha256", "review_artifact_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledSupplementaryLocatorDecisionManifest:
        if not self.adjudication_confirmed:
            raise ValueError("supplementary locator adjudication requires adjudication_confirmed=true")
        if not self.decisions:
            raise ValueError("supplementary locator adjudication requires human decisions")
        decision_ids = [decision.review_item_id for decision in self.decisions]
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("duplicate supplementary locator review_item_id decision")
        return self


class OledSupplementaryLocatorAdjudicatedItem(BaseModel):
    """A redacted decision binding; matched table content remains in the PR-E artifact."""

    model_config = ConfigDict(extra="forbid")

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
    matched_table_id: str = ""
    matched_table_page: int | None = Field(default=None, ge=0)
    table_content_digest: str = ""
    parser_warning_codes: list[str] = Field(default_factory=list)
    decision: OledSupplementaryLocatorDecision
    reviewed_by: str
    reviewed_at: str
    review_note: str = ""
    semantic_note: str = ""
    locator_accepted: bool
    semantic_review_required: bool
    eligible_for_later_scoped_candidate_proposal: bool
    direct_admission_eligible: bool = False
    evidence_content_mutated: bool = False

    @field_validator("review_item_id", "recovery_item_id")
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
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("table_content_digest")
    @classmethod
    def validate_optional_table_digest(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        return _normalize_sha256(clean, field_name="table_content_digest")

    @field_validator("parser_warning_codes")
    @classmethod
    def validate_warning_codes(cls, value: list[str]) -> list[str]:
        clean = [_validate_bound_id(item, field_name="parser_warning_code") for item in value]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("parser_warning_codes must be sorted and unique")
        return clean

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return OledSupplementaryLocatorDecisionEntry.validate_reviewed_by(value)

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return OledSupplementaryLocatorDecisionEntry.validate_reviewed_at(value)

    @field_validator("review_note", "semantic_note")
    @classmethod
    def validate_notes(cls, value: str, info: Any) -> str:
        return OledSupplementaryLocatorDecisionEntry.validate_notes(value, info)

    @model_validator(mode="after")
    def validate_item_shape(self) -> OledSupplementaryLocatorAdjudicatedItem:
        exact_match = self.match_status == OledSupplementaryLocatorMatchStatus.EXACT_MATCH
        has_table_binding = bool(
            self.matched_table_id and self.matched_table_page is not None and self.table_content_digest
        )
        if exact_match != has_table_binding:
            raise ValueError("adjudicated locator table binding does not match source match status")
        accepted = self.decision == OledSupplementaryLocatorDecision.ACCEPT_LOCATOR
        if self.locator_accepted != accepted:
            raise ValueError("adjudicated locator accepted flag does not match decision")
        if accepted and not exact_match:
            raise ValueError("only exact supplementary locator matches may be accepted")
        if self.eligible_for_later_scoped_candidate_proposal != accepted:
            raise ValueError("supplementary locator proposal eligibility does not match decision")
        if self.semantic_review_required != bool(self.semantic_note):
            raise ValueError("supplementary locator semantic-review flag does not match semantic_note")
        if self.decision in {
            OledSupplementaryLocatorDecision.REJECT_LOCATOR,
            OledSupplementaryLocatorDecision.NEEDS_SOURCE_CHECK,
        } and not self.review_note:
            raise ValueError("rejected or source-check locator decisions require review_note")
        if self.direct_admission_eligible or self.evidence_content_mutated:
            raise ValueError("supplementary locator adjudication must not mutate or directly admit evidence")
        return self


class OledSupplementaryLocatorAdjudicationArtifact(BaseModel):
    """Content-bound human locator decisions with every downstream action disabled."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = SUPPLEMENTARY_LOCATOR_ADJUDICATION_ARTIFACT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    review_artifact_sha256: str
    review_artifact_digest: str
    decision_manifest_sha256: str
    decision_manifest_digest: str
    execution_artifact_sha256: str
    execution_artifact_digest: str
    locator_manifest_sha256: str
    preflight_plan_digest: str
    status: OledSupplementaryLocatorAdjudicationStatus
    item_count: int = Field(ge=1)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    needs_source_check_count: int = Field(ge=0)
    semantic_review_required_count: int = Field(ge=0)
    candidate_proposal_eligible_count: int = Field(ge=0)
    adjudicated_items: list[OledSupplementaryLocatorAdjudicatedItem] = Field(default_factory=list)
    adjudication_artifact_digest: str
    review_only: bool = True
    offline_only: bool = True
    human_decisions_recorded: bool = True
    adjudication_complete: bool = True
    review_artifact_read: bool = True
    matched_table_content_copied: bool = False
    table_transcription_validated: bool = False
    scientific_content_validated: bool = False
    locator_correction_applied: bool = False
    semantic_correction_applied: bool = False
    physical_semantics_validated: bool = False
    schema_mapping_performed: bool = False
    parsed_output_read: bool = False
    pdf_content_read: bool = False
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    candidate_regenerated: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_LOCATOR_ADJUDICATION_ARTIFACT_VERSION:
            raise ValueError("unexpected supplementary locator adjudication artifact_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "review_artifact_sha256",
        "review_artifact_digest",
        "decision_manifest_sha256",
        "decision_manifest_digest",
        "execution_artifact_sha256",
        "execution_artifact_digest",
        "locator_manifest_sha256",
        "preflight_plan_digest",
        "adjudication_artifact_digest",
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
    def validate_artifact_integrity(self) -> OledSupplementaryLocatorAdjudicationArtifact:
        if not self.adjudicated_items or self.item_count != len(self.adjudicated_items):
            raise ValueError("supplementary locator adjudication item_count mismatch")
        item_ids = [item.review_item_id for item in self.adjudicated_items]
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("supplementary locator adjudicated items must be sorted and unique")
        counts = {
            decision: sum(item.decision == decision for item in self.adjudicated_items)
            for decision in OledSupplementaryLocatorDecision
        }
        if self.accepted_count != counts[OledSupplementaryLocatorDecision.ACCEPT_LOCATOR]:
            raise ValueError("supplementary locator accepted_count mismatch")
        if self.rejected_count != counts[OledSupplementaryLocatorDecision.REJECT_LOCATOR]:
            raise ValueError("supplementary locator rejected_count mismatch")
        if (
            self.needs_source_check_count
            != counts[OledSupplementaryLocatorDecision.NEEDS_SOURCE_CHECK]
        ):
            raise ValueError("supplementary locator needs_source_check_count mismatch")
        semantic_count = sum(item.semantic_review_required for item in self.adjudicated_items)
        eligible_count = sum(
            item.eligible_for_later_scoped_candidate_proposal for item in self.adjudicated_items
        )
        if self.semantic_review_required_count != semantic_count:
            raise ValueError("supplementary locator semantic_review_required_count mismatch")
        if self.candidate_proposal_eligible_count != eligible_count:
            raise ValueError("supplementary locator candidate proposal eligibility count mismatch")
        expected_status = _status_for_counts(self.accepted_count, self.item_count)
        if self.status != expected_status:
            raise ValueError("supplementary locator adjudication status mismatch")
        fixed_true_flags = (
            "review_only",
            "offline_only",
            "human_decisions_recorded",
            "adjudication_complete",
            "review_artifact_read",
        )
        fixed_false_flags = (
            "matched_table_content_copied",
            "table_transcription_validated",
            "scientific_content_validated",
            "locator_correction_applied",
            "semantic_correction_applied",
            "physical_semantics_validated",
            "schema_mapping_performed",
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
            raise ValueError("supplementary locator adjudication lost a required audit flag")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary locator adjudication crossed a downstream boundary")
        if _adjudication_artifact_digest(self) != self.adjudication_artifact_digest:
            raise ValueError("supplementary locator adjudication artifact digest mismatch")
        return self


def validate_oled_supplementary_locator_decision_binding(
    review_artifact: OledSupplementaryLocatorReviewArtifact,
    manifest: OledSupplementaryLocatorDecisionManifest,
    *,
    review_artifact_sha256: str,
) -> None:
    review = OledSupplementaryLocatorReviewArtifact.model_validate(
        review_artifact.model_dump(mode="json")
    )
    manifest = OledSupplementaryLocatorDecisionManifest.model_validate(
        manifest.model_dump(mode="json")
    )
    observed_sha256 = _normalize_sha256(
        review_artifact_sha256,
        field_name="review_artifact_sha256",
    )
    if manifest.run_id != review.run_id or manifest.paper_id != review.paper_id:
        raise ValueError("supplementary locator decision identity does not match review artifact")
    if manifest.review_artifact_sha256 != observed_sha256:
        raise ValueError("supplementary locator decision does not bind exact review artifact bytes")
    if manifest.review_artifact_digest != review.review_artifact_digest:
        raise ValueError("supplementary locator decision does not bind canonical review content")
    expected_item_ids = {item.review_item_id for item in review.review_items}
    decision_item_ids = {decision.review_item_id for decision in manifest.decisions}
    if decision_item_ids != expected_item_ids:
        raise ValueError("supplementary locator decisions must exactly cover review items")


def build_oled_supplementary_locator_adjudication_artifact(
    *,
    review_artifact: OledSupplementaryLocatorReviewArtifact,
    review_artifact_sha256: str,
    decision_manifest: OledSupplementaryLocatorDecisionManifest,
    decision_manifest_sha256: str,
    generated_at: str,
) -> OledSupplementaryLocatorAdjudicationArtifact:
    review = OledSupplementaryLocatorReviewArtifact.model_validate(
        review_artifact.model_dump(mode="json")
    )
    manifest = OledSupplementaryLocatorDecisionManifest.model_validate(
        decision_manifest.model_dump(mode="json")
    )
    validate_oled_supplementary_locator_decision_binding(
        review,
        manifest,
        review_artifact_sha256=review_artifact_sha256,
    )
    decisions = {decision.review_item_id: decision for decision in manifest.decisions}
    adjudicated_items = [
        _adjudicate_item(source_item, decisions[source_item.review_item_id])
        for source_item in review.review_items
    ]
    adjudicated_items.sort(key=lambda item: item.review_item_id)
    accepted_count = sum(
        item.decision == OledSupplementaryLocatorDecision.ACCEPT_LOCATOR
        for item in adjudicated_items
    )
    rejected_count = sum(
        item.decision == OledSupplementaryLocatorDecision.REJECT_LOCATOR
        for item in adjudicated_items
    )
    source_check_count = sum(
        item.decision == OledSupplementaryLocatorDecision.NEEDS_SOURCE_CHECK
        for item in adjudicated_items
    )
    semantic_count = sum(item.semantic_review_required for item in adjudicated_items)
    eligible_count = sum(
        item.eligible_for_later_scoped_candidate_proposal for item in adjudicated_items
    )
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_LOCATOR_ADJUDICATION_ARTIFACT_VERSION,
        "run_id": review.run_id,
        "paper_id": review.paper_id,
        "generated_at": str(generated_at or "").strip(),
        "review_artifact_sha256": _normalize_sha256(
            review_artifact_sha256,
            field_name="review_artifact_sha256",
        ),
        "review_artifact_digest": review.review_artifact_digest,
        "decision_manifest_sha256": _normalize_sha256(
            decision_manifest_sha256,
            field_name="decision_manifest_sha256",
        ),
        "decision_manifest_digest": _stable_hash(manifest.model_dump(mode="json")),
        "execution_artifact_sha256": review.execution_artifact_sha256,
        "execution_artifact_digest": review.execution_artifact_digest,
        "locator_manifest_sha256": review.locator_manifest_sha256,
        "preflight_plan_digest": review.preflight_plan_digest,
        "status": _status_for_counts(accepted_count, len(adjudicated_items)).value,
        "item_count": len(adjudicated_items),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "needs_source_check_count": source_check_count,
        "semantic_review_required_count": semantic_count,
        "candidate_proposal_eligible_count": eligible_count,
        "adjudicated_items": [item.model_dump(mode="json") for item in adjudicated_items],
        "adjudication_artifact_digest": "",
        "review_only": True,
        "offline_only": True,
        "human_decisions_recorded": True,
        "adjudication_complete": True,
        "review_artifact_read": True,
        "matched_table_content_copied": False,
        "table_transcription_validated": False,
        "scientific_content_validated": False,
        "locator_correction_applied": False,
        "semantic_correction_applied": False,
        "physical_semantics_validated": False,
        "schema_mapping_performed": False,
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
    payload["adjudication_artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "adjudication_artifact_digest"}
    )
    return OledSupplementaryLocatorAdjudicationArtifact.model_validate(payload)


def oled_supplementary_review_item_digest(item: OledSupplementaryLocatorReviewItem) -> str:
    item = OledSupplementaryLocatorReviewItem.model_validate(item.model_dump(mode="json"))
    return _stable_hash(item.model_dump(mode="json"))


def _adjudicate_item(
    source_item: OledSupplementaryLocatorReviewItem,
    decision: OledSupplementaryLocatorDecisionEntry,
) -> OledSupplementaryLocatorAdjudicatedItem:
    source_item = OledSupplementaryLocatorReviewItem.model_validate(
        source_item.model_dump(mode="json")
    )
    decision = OledSupplementaryLocatorDecisionEntry.model_validate(
        decision.model_dump(mode="json")
    )
    accepted = decision.decision == OledSupplementaryLocatorDecision.ACCEPT_LOCATOR
    if accepted and source_item.match_status != OledSupplementaryLocatorMatchStatus.EXACT_MATCH:
        raise ValueError("only exact supplementary locator matches may be accepted")
    table = source_item.matched_table
    return OledSupplementaryLocatorAdjudicatedItem(
        review_item_id=source_item.review_item_id,
        recovery_item_id=source_item.recovery_item_id,
        source_review_item_digest=oled_supplementary_review_item_digest(source_item),
        source_id=source_item.source_id,
        source_pdf_sha256=source_item.source_pdf_sha256,
        parsed_document_sha256=source_item.parsed_document_sha256,
        parser_backend=source_item.parser_backend,
        target_kind=source_item.target_kind,
        target_locator=source_item.target_locator,
        canonical_locator=source_item.canonical_locator,
        match_status=source_item.match_status,
        matched_table_id=table.table_id if table is not None else "",
        matched_table_page=table.page if table is not None else None,
        table_content_digest=table.table_content_digest if table is not None else "",
        parser_warning_codes=source_item.parser_warning_codes,
        decision=decision.decision,
        reviewed_by=decision.reviewed_by,
        reviewed_at=decision.reviewed_at,
        review_note=decision.review_note,
        semantic_note=decision.semantic_note,
        locator_accepted=accepted,
        semantic_review_required=bool(decision.semantic_note),
        eligible_for_later_scoped_candidate_proposal=accepted,
        direct_admission_eligible=False,
        evidence_content_mutated=False,
    )


def _status_for_counts(
    accepted_count: int,
    item_count: int,
) -> OledSupplementaryLocatorAdjudicationStatus:
    if accepted_count == item_count:
        return OledSupplementaryLocatorAdjudicationStatus.ALL_LOCATORS_ACCEPTED
    if accepted_count:
        return OledSupplementaryLocatorAdjudicationStatus.PARTIALLY_ACCEPTED
    return OledSupplementaryLocatorAdjudicationStatus.NO_LOCATORS_ACCEPTED


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


def _validate_audit_text(
    value: Any,
    *,
    field_name: str,
    required: bool,
    max_length: int,
) -> str:
    clean = str(value or "").strip()
    if not clean:
        if required:
            raise ValueError(f"{field_name} is required")
        return ""
    if len(clean) > max_length:
        raise ValueError(f"{field_name} is too long")
    if any(ord(character) < 32 or ord(character) == 127 for character in clean):
        raise ValueError(f"{field_name} contains control characters")
    lowered = clean.lower()
    if "://" in clean or "file:" in lowered or _ABSOLUTE_PATH_RE.search(clean):
        raise ValueError(f"{field_name} must not contain a URL or absolute path")
    if _contains_credential_material(clean):
        raise ValueError(f"{field_name} contains forbidden credential-like text")
    return clean


def _contains_credential_material(value: str) -> bool:
    return any(
        pattern.search(value) is not None
        for pattern in (
            _CREDENTIAL_ASSIGNMENT_RE,
            _BEARER_CREDENTIAL_RE,
            _INLINE_BEARER_CREDENTIAL_RE,
            _SECRET_KEY_RE,
        )
    )


def _stable_hash(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _adjudication_artifact_digest(
    artifact: OledSupplementaryLocatorAdjudicationArtifact,
) -> str:
    return _stable_hash(artifact.model_dump(mode="json", exclude={"adjudication_artifact_digest"}))


__all__ = [
    "SUPPLEMENTARY_LOCATOR_ADJUDICATION_ARTIFACT_VERSION",
    "SUPPLEMENTARY_LOCATOR_DECISION_MANIFEST_VERSION",
    "OledSupplementaryLocatorAdjudicatedItem",
    "OledSupplementaryLocatorAdjudicationArtifact",
    "OledSupplementaryLocatorAdjudicationStatus",
    "OledSupplementaryLocatorDecision",
    "OledSupplementaryLocatorDecisionEntry",
    "OledSupplementaryLocatorDecisionManifest",
    "build_oled_supplementary_locator_adjudication_artifact",
    "oled_supplementary_review_item_digest",
    "validate_oled_supplementary_locator_decision_binding",
]
