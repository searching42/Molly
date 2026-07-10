from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


OledReviewCandidateType = Literal[
    "oled_compiled_record",
    "oled_schema_candidate",
    "oled_text_evidence",
    "oled_raw_candidate",
]
OledReviewPriority = Literal["high", "medium", "low"]
OledReviewStatus = Literal["pending"]
OledReviewerDecisionStatus = Literal["pending", "reviewed"]
OledReviewerDecisionValue = Literal["", "accept", "reject", "needs_more_context"]


class OledReviewItem(BaseModel):
    review_item_id: str
    paper_id: str
    candidate_type: OledReviewCandidateType
    priority: OledReviewPriority
    review_status: OledReviewStatus = "pending"
    source_candidate_id: str
    source_artifact: str
    property_id: str | None = None
    property_label: str | None = None
    raw_value: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    compound_mentions: list[str] = Field(default_factory=list)
    material_roles: list[dict[str, str]] = Field(default_factory=list)
    device_context: str = ""
    condition_text: str = ""
    evidence_text: str = ""
    evidence_page: int | None = None
    evidence_location: str = ""
    provenance: dict[str, Any] = Field(default_factory=dict)
    suggested_review_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("review_item_id", "paper_id", "source_candidate_id", "source_artifact")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean

    @field_validator("compound_mentions")
    @classmethod
    def validate_compound_mentions(cls, value: list[str]) -> list[str]:
        return _stable_unique(value)

    @field_validator("warnings", "suggested_review_questions")
    @classmethod
    def validate_stable_unique_text(cls, value: list[str]) -> list[str]:
        return _stable_unique(value)


class OledReviewPacket(BaseModel):
    schema_version: str = "oled_review_packet.v1"
    run_id: str
    generated_at: str
    source_artifacts: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)
    review_items: list[OledReviewItem] = Field(default_factory=list)

    @field_validator("run_id", "generated_at")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean


class OledReviewerDecision(BaseModel):
    review_item_id: str
    review_status: OledReviewerDecisionStatus = "pending"
    decision: OledReviewerDecisionValue = ""
    corrected_property_id: str = ""
    corrected_value: str = ""
    corrected_unit: str = ""
    corrected_compound: str = ""
    corrected_condition: str = ""
    reviewer: str = ""
    reviewed_at: str = ""
    comment: str = ""

    @field_validator("review_item_id")
    @classmethod
    def validate_review_item_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("review_item_id is required")
        return clean


class OledReviewerDecisionTemplate(BaseModel):
    schema_version: str = "oled_reviewer_decision_template.v1"
    run_id: str
    generated_at: str
    source_packet_digest: str = ""
    decisions: list[OledReviewerDecision] = Field(default_factory=list)

    @field_validator("run_id", "generated_at")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("value is required")
        return clean


def oled_review_packet_digest(packet: OledReviewPacket) -> str:
    return oled_review_payload_digest(packet.model_dump(mode="json"))


def oled_review_payload_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            ordered.append(clean)
    return ordered


__all__ = [
    "OledReviewCandidateType",
    "OledReviewItem",
    "OledReviewPacket",
    "OledReviewPriority",
    "OledReviewStatus",
    "OledReviewerDecision",
    "OledReviewerDecisionStatus",
    "OledReviewerDecisionTemplate",
    "OledReviewerDecisionValue",
    "oled_review_packet_digest",
    "oled_review_payload_digest",
]
