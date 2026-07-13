from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.domains.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryPlan,
    OledSupplementaryRecoveryStatus,
    OledSupplementaryTargetKind,
)


SUPPLEMENTARY_SOURCE_INTAKE_MANIFEST_VERSION = "oled_supplementary_source_intake_manifest.v1"
SUPPLEMENTARY_SOURCE_INTAKE_PLAN_VERSION = "oled_supplementary_source_intake_plan.v1"
DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES = 200 * 1024 * 1024

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_CREDENTIAL_MARKERS = (
    "token",
    "secret",
    "authorization",
    "password",
    "bearer",
    "cookie",
    "x-api-key",
    "api_key",
    "api key",
    "apikey",
    "private_key",
    "private key",
)
_PDF_TRAILER_SCAN_BYTES = 8192


class OledSupplementarySourceIntakeDecision(str, Enum):
    APPROVED = "approved"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class OledSupplementarySourceParseEligibility(str, Enum):
    ELIGIBLE_FOR_TARGETED_SOURCE_PARSE = "eligible_for_targeted_source_parse"
    ELIGIBLE_FOR_MANUAL_SOURCE_REVIEW = "eligible_for_manual_source_review"
    NOT_ELIGIBLE = "not_eligible"


class OledSupplementaryLocalSource(BaseModel):
    """Operator-local source metadata. The path is deliberately never serialized into output."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    local_pdf_path: str
    expected_pdf_sha256: str = ""
    provenance_category: str
    access_policy: str
    provenance_note: str = ""

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("source_id must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("local_pdf_path")
    @classmethod
    def validate_local_pdf_path(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("local_pdf_path is required")
        return clean

    @field_validator("expected_pdf_sha256")
    @classmethod
    def validate_expected_pdf_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="expected_pdf_sha256", allow_empty=True)

    @field_validator("provenance_category", "access_policy")
    @classmethod
    def validate_required_audit_text(cls, value: str, info: Any) -> str:
        return _validate_audit_text(value, field_name=info.field_name, required=True)

    @field_validator("provenance_note")
    @classmethod
    def validate_optional_audit_text(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="provenance_note", required=False)


class OledSupplementarySourceIntakeDecisionEntry(BaseModel):
    """One explicit human decision for one recovery-plan item."""

    model_config = ConfigDict(extra="forbid")

    recovery_item_id: str
    decision: OledSupplementarySourceIntakeDecision
    source_id: str | None = None
    reviewed_by: str
    reviewed_at: str
    review_note: str = ""

    @field_validator("recovery_item_id")
    @classmethod
    def validate_recovery_item_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("recovery_item_id is required")
        return clean

    @field_validator("source_id")
    @classmethod
    def validate_optional_source_id(cls, value: str | None) -> str | None:
        clean = str(value or "").strip()
        if not clean:
            return None
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("source_id must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="reviewed_by", required=True)

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("reviewed_at is required")
        try:
            datetime.fromisoformat(clean.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("reviewed_at must be ISO-8601") from exc
        return clean

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="review_note", required=False)

    @model_validator(mode="after")
    def validate_decision_shape(self) -> OledSupplementarySourceIntakeDecisionEntry:
        if self.decision == OledSupplementarySourceIntakeDecision.APPROVED:
            if not self.source_id:
                raise ValueError("approved supplementary source decision requires source_id")
        elif self.source_id is not None:
            raise ValueError("deferred or rejected supplementary source decision must not bind a source_id")
        if self.decision != OledSupplementarySourceIntakeDecision.APPROVED and not self.review_note:
            raise ValueError("deferred or rejected supplementary source decision requires review_note")
        return self


class OledSupplementarySourceIntakeManifest(BaseModel):
    """Human-authored, plan-bound manifest for operator-supplied local PDFs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_SOURCE_INTAKE_MANIFEST_VERSION
    paper_id: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    recovery_plan_digest: str
    intake_confirmed: bool = False
    sources: list[OledSupplementaryLocalSource] = Field(default_factory=list)
    decisions: list[OledSupplementarySourceIntakeDecisionEntry] = Field(default_factory=list)

    @field_validator(
        "paper_id",
        "source_request_digest",
        "source_mapping_result_digest",
        "source_context_digest",
        "recovery_plan_digest",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError(f"{info.field_name} is required")
        return clean

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_SOURCE_INTAKE_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary source intake manifest schema_version")
        return value

    @model_validator(mode="after")
    def validate_manifest_integrity(self) -> OledSupplementarySourceIntakeManifest:
        if not self.intake_confirmed:
            raise ValueError("supplementary source intake requires intake_confirmed=true")
        source_ids = [source.source_id for source in self.sources]
        decision_ids = [decision.recovery_item_id for decision in self.decisions]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("duplicate supplementary source_id")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("duplicate supplementary recovery_item_id decision")
        known_source_ids = set(source_ids)
        approved_source_ids = {
            decision.source_id
            for decision in self.decisions
            if decision.decision == OledSupplementarySourceIntakeDecision.APPROVED
        }
        if not approved_source_ids.issubset(known_source_ids):
            raise ValueError("supplementary source decision references an unknown source_id")
        if known_source_ids != approved_source_ids:
            raise ValueError("every supplementary source must be bound by an approved decision")
        return self


class OledSupplementarySourceEnvelope(BaseModel):
    """Safe metadata from structural local-PDF validation, with no path or bytes retained."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    pdf_sha256: str
    byte_size: int = Field(gt=0)
    content_type: str = "application/pdf"
    provenance_category: str
    access_policy: str
    provenance_note: str = ""
    pdf_header_valid: bool = True
    pdf_eof_marker_valid: bool = True
    page_count_validated: bool = False

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("source_id must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("pdf_sha256")
    @classmethod
    def validate_pdf_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="pdf_sha256", allow_empty=False)

    @field_validator("provenance_category", "access_policy")
    @classmethod
    def validate_required_audit_text(cls, value: str, info: Any) -> str:
        return _validate_audit_text(value, field_name=info.field_name, required=True)

    @field_validator("provenance_note")
    @classmethod
    def validate_optional_audit_text(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="provenance_note", required=False)

    @model_validator(mode="after")
    def validate_envelope_flags(self) -> OledSupplementarySourceEnvelope:
        if self.content_type != "application/pdf":
            raise ValueError("supplementary source envelope content_type must be application/pdf")
        if not self.pdf_header_valid or not self.pdf_eof_marker_valid:
            raise ValueError("supplementary source envelope must have valid PDF header and EOF marker")
        if self.page_count_validated:
            raise ValueError("source intake must not claim PDF page-count validation")
        return self


class OledSupplementarySourceIntakeItem(BaseModel):
    """A recovery item plus its explicit human source decision and parse eligibility."""

    model_config = ConfigDict(extra="forbid")

    recovery_item_id: str
    recovery_status: OledSupplementaryRecoveryStatus
    target_kind: OledSupplementaryTargetKind
    target_locator: str | None = None
    decision: OledSupplementarySourceIntakeDecision
    source_id: str | None = None
    source_pdf_sha256: str | None = None
    parse_eligibility: OledSupplementarySourceParseEligibility
    reviewed_by: str
    reviewed_at: str
    review_note: str = ""

    @field_validator("recovery_item_id")
    @classmethod
    def validate_recovery_item_id(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("recovery_item_id is required")
        return clean

    @field_validator("target_locator", "source_id", "source_pdf_sha256", mode="before")
    @classmethod
    def normalize_optional_text(cls, value: Any, info: Any) -> str | None:
        clean = str(value or "").strip()
        if not clean:
            return None
        if info.field_name == "source_id" and not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("source_id must use only letters, numbers, dot, dash, and underscore")
        if info.field_name == "source_pdf_sha256":
            return _normalize_sha256(clean, field_name="source_pdf_sha256", allow_empty=False)
        return clean

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewer(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="reviewed_by", required=True)

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return OledSupplementarySourceIntakeDecisionEntry.validate_reviewed_at(value)

    @field_validator("review_note")
    @classmethod
    def validate_review_note(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="review_note", required=False)

    @model_validator(mode="after")
    def validate_item_shape(self) -> OledSupplementarySourceIntakeItem:
        if self.recovery_status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND:
            if self.target_kind not in {
                OledSupplementaryTargetKind.TABLE,
                OledSupplementaryTargetKind.FIGURE,
            }:
                raise ValueError("explicit supplementary intake item requires a table or figure target")
            if not self.target_locator:
                raise ValueError("explicit supplementary intake item requires target_locator")
        elif self.target_locator is not None:
            raise ValueError("manual supplementary intake item must not assert target_locator")
        if self.decision == OledSupplementarySourceIntakeDecision.APPROVED:
            if not self.source_id or not self.source_pdf_sha256:
                raise ValueError("approved supplementary intake item requires source binding and PDF hash")
            if self.recovery_status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND:
                expected = OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_TARGETED_SOURCE_PARSE
            else:
                expected = OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_MANUAL_SOURCE_REVIEW
            if self.parse_eligibility != expected:
                raise ValueError("approved supplementary intake item has invalid parse eligibility")
        else:
            if self.source_id is not None or self.source_pdf_sha256 is not None:
                raise ValueError("non-approved supplementary intake item must not bind a source")
            if self.parse_eligibility != OledSupplementarySourceParseEligibility.NOT_ELIGIBLE:
                raise ValueError("non-approved supplementary intake item must not be parse eligible")
        return self


class OledSupplementarySourceIntakePlan(BaseModel):
    """Verified, review-only binding between a recovery plan and local supplementary PDFs."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_SOURCE_INTAKE_PLAN_VERSION
    paper_id: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    recovery_plan_digest: str
    intake_confirmed: bool = True
    source_envelopes: list[OledSupplementarySourceEnvelope] = Field(default_factory=list)
    items: list[OledSupplementarySourceIntakeItem] = Field(default_factory=list)
    source_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    approved_item_count: int = Field(ge=0)
    deferred_item_count: int = Field(ge=0)
    rejected_item_count: int = Field(ge=0)
    intake_plan_digest: str
    review_only: bool = True
    executable: bool = False
    offline_only: bool = True
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    pdf_content_parsed: bool = False
    pdf_page_count_validated: bool = False
    supplementary_downloaded: bool = False
    candidate_regenerated: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @field_validator(
        "schema_version",
        "paper_id",
        "source_request_digest",
        "source_mapping_result_digest",
        "source_context_digest",
        "recovery_plan_digest",
        "intake_plan_digest",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError(f"{info.field_name} is required")
        return clean

    @model_validator(mode="after")
    def validate_plan_integrity(self) -> OledSupplementarySourceIntakePlan:
        if self.schema_version != SUPPLEMENTARY_SOURCE_INTAKE_PLAN_VERSION:
            raise ValueError("unexpected supplementary source intake plan schema_version")
        if not self.intake_confirmed:
            raise ValueError("supplementary source intake plan requires confirmed human intake")
        if self.source_count != len(self.source_envelopes):
            raise ValueError("supplementary source intake source_count does not match source_envelopes")
        if self.item_count != len(self.items):
            raise ValueError("supplementary source intake item_count does not match items")
        source_ids = [source.source_id for source in self.source_envelopes]
        item_ids = [item.recovery_item_id for item in self.items]
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise ValueError("supplementary source envelopes must be sorted with unique source IDs")
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("supplementary source intake items must be sorted with unique recovery item IDs")
        approved = sum(item.decision == OledSupplementarySourceIntakeDecision.APPROVED for item in self.items)
        deferred = sum(item.decision == OledSupplementarySourceIntakeDecision.DEFERRED for item in self.items)
        rejected = sum(item.decision == OledSupplementarySourceIntakeDecision.REJECTED for item in self.items)
        if (approved, deferred, rejected) != (
            self.approved_item_count,
            self.deferred_item_count,
            self.rejected_item_count,
        ):
            raise ValueError("supplementary source intake decision counts do not match items")
        known_sources = set(source_ids)
        used_sources = {item.source_id for item in self.items if item.source_id is not None}
        if used_sources != known_sources:
            raise ValueError("supplementary source intake source envelopes must be explicitly bound")
        source_hashes = {source.source_id: source.pdf_sha256 for source in self.source_envelopes}
        for item in self.items:
            if item.source_id is not None and item.source_pdf_sha256 != source_hashes.get(item.source_id):
                raise ValueError("supplementary source intake item hash does not match source envelope")
        fixed_false_flags = (
            "executable",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "pdf_content_parsed",
            "pdf_page_count_validated",
            "supplementary_downloaded",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if not self.review_only or not self.offline_only:
            raise ValueError("supplementary source intake plan must remain review-only and offline-only")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary source intake plan unexpectedly records an execution side effect")
        if _intake_plan_digest(self) != self.intake_plan_digest:
            raise ValueError("supplementary source intake plan digest does not match canonical content")
        return self


def build_oled_supplementary_source_intake_plan(
    recovery_plan: OledSupplementaryEvidenceRecoveryPlan,
    intake_manifest: OledSupplementarySourceIntakeManifest,
    *,
    max_pdf_bytes: int = DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES,
) -> OledSupplementarySourceIntakePlan:
    """Bind human-approved local PDFs to every item in a verified recovery plan.

    This only validates a PDF envelope and hash. It never parses scientific
    content, counts pages, discovers/downloads sources, or invokes MinerU/LLM.
    """

    if max_pdf_bytes <= 0:
        raise ValueError("max_pdf_bytes must be positive")
    # Pydantic's model_copy(update=...) intentionally skips validation. Reparse
    # both caller-owned models so a stale digest, a manual-to-explicit rewrite,
    # or a cleared intake confirmation cannot cross this gate.
    recovery_plan = OledSupplementaryEvidenceRecoveryPlan.model_validate(
        recovery_plan.model_dump(mode="json")
    )
    intake_manifest = OledSupplementarySourceIntakeManifest.model_validate(
        intake_manifest.model_dump(mode="json")
    )
    _validate_manifest_binding(recovery_plan, intake_manifest)
    source_by_id = {source.source_id: source for source in intake_manifest.sources}
    decision_by_item_id = {decision.recovery_item_id: decision for decision in intake_manifest.decisions}
    approved_source_ids = sorted(
        {
            decision.source_id
            for decision in intake_manifest.decisions
            if decision.decision == OledSupplementarySourceIntakeDecision.APPROVED
        }
    )
    envelopes = [
        inspect_oled_supplementary_source_pdf(
            source_by_id[source_id],
            max_pdf_bytes=max_pdf_bytes,
        )
        for source_id in approved_source_ids
    ]
    envelope_by_source_id = {envelope.source_id: envelope for envelope in envelopes}

    items: list[OledSupplementarySourceIntakeItem] = []
    for recovery_item in recovery_plan.items:
        decision = decision_by_item_id[recovery_item.item_id]
        source_id = decision.source_id
        source_hash: str | None = None
        if decision.decision == OledSupplementarySourceIntakeDecision.APPROVED:
            source_hash = envelope_by_source_id[source_id].pdf_sha256 if source_id else None
            if recovery_item.status == OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND:
                eligibility = OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_TARGETED_SOURCE_PARSE
            else:
                eligibility = OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_MANUAL_SOURCE_REVIEW
        else:
            eligibility = OledSupplementarySourceParseEligibility.NOT_ELIGIBLE
        items.append(
            OledSupplementarySourceIntakeItem(
                recovery_item_id=recovery_item.item_id,
                recovery_status=recovery_item.status,
                target_kind=recovery_item.target_kind,
                target_locator=recovery_item.target_locator,
                decision=decision.decision,
                source_id=source_id,
                source_pdf_sha256=source_hash,
                parse_eligibility=eligibility,
                reviewed_by=decision.reviewed_by,
                reviewed_at=decision.reviewed_at,
                review_note=decision.review_note,
            )
        )

    items.sort(key=lambda item: item.recovery_item_id)
    envelopes.sort(key=lambda envelope: envelope.source_id)
    payload: dict[str, Any] = {
        "schema_version": SUPPLEMENTARY_SOURCE_INTAKE_PLAN_VERSION,
        "paper_id": recovery_plan.paper_id,
        "source_request_digest": recovery_plan.source_request_digest,
        "source_mapping_result_digest": recovery_plan.source_mapping_result_digest,
        "source_context_digest": recovery_plan.source_context_digest,
        "recovery_plan_digest": recovery_plan.plan_digest,
        "intake_confirmed": True,
        "source_envelopes": [source.model_dump(mode="json") for source in envelopes],
        "items": [item.model_dump(mode="json") for item in items],
        "source_count": len(envelopes),
        "item_count": len(items),
        "approved_item_count": sum(
            item.decision == OledSupplementarySourceIntakeDecision.APPROVED for item in items
        ),
        "deferred_item_count": sum(
            item.decision == OledSupplementarySourceIntakeDecision.DEFERRED for item in items
        ),
        "rejected_item_count": sum(
            item.decision == OledSupplementarySourceIntakeDecision.REJECTED for item in items
        ),
        "intake_plan_digest": "",
        "review_only": True,
        "executable": False,
        "offline_only": True,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "pdf_content_parsed": False,
        "pdf_page_count_validated": False,
        "supplementary_downloaded": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["intake_plan_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "intake_plan_digest"}
    )
    return OledSupplementarySourceIntakePlan.model_validate(payload)


def inspect_oled_supplementary_source_pdf(
    source: OledSupplementaryLocalSource,
    *,
    max_pdf_bytes: int = DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES,
) -> OledSupplementarySourceEnvelope:
    """Validate only a local PDF envelope and hash through one file descriptor.

    The function intentionally avoids text/table/image extraction and does not
    claim a page count. Those operations belong to a later, separately gated
    parser preflight.
    """

    if max_pdf_bytes <= 0:
        raise ValueError("max_pdf_bytes must be positive")
    path = Path(source.local_pdf_path).expanduser()
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"supplementary source {source.source_id} must use a .pdf filename")
    try:
        if path.is_symlink():
            raise ValueError(f"supplementary source {source.source_id} must not be a symlink")
        if not path.is_file():
            raise ValueError(f"supplementary source {source.source_id} must be a regular file")
        no_follow = getattr(os, "O_NOFOLLOW", None)
        if no_follow is None:
            raise ValueError("supplementary source intake requires O_NOFOLLOW support")
        flags = os.O_RDONLY | no_follow
        descriptor = os.open(path, flags)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"supplementary source {source.source_id} local PDF is unavailable") from exc

    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise ValueError(f"supplementary source {source.source_id} must be a regular file")
            byte_size = int(file_stat.st_size)
            if byte_size <= 0:
                raise ValueError(f"supplementary source {source.source_id} PDF is empty")
            if byte_size > max_pdf_bytes:
                raise ValueError(f"supplementary source {source.source_id} PDF exceeds size limit")
            header = handle.read(8)
            if not header.startswith(b"%PDF-"):
                raise ValueError(f"supplementary source {source.source_id} lacks a PDF header")
            handle.seek(max(0, byte_size - _PDF_TRAILER_SCAN_BYTES))
            trailer = handle.read(_PDF_TRAILER_SCAN_BYTES)
            if b"%%EOF" not in trailer:
                raise ValueError(f"supplementary source {source.source_id} lacks a PDF EOF marker")
            handle.seek(0)
            digest = hashlib.sha256()
            bytes_read = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                bytes_read += len(chunk)
                if bytes_read > max_pdf_bytes:
                    raise ValueError(f"supplementary source {source.source_id} PDF exceeds size limit")
                digest.update(chunk)
            final_stat = os.fstat(handle.fileno())
            if (
                final_stat.st_size != byte_size
                or final_stat.st_mtime_ns != file_stat.st_mtime_ns
                or final_stat.st_ctime_ns != file_stat.st_ctime_ns
                or bytes_read != byte_size
            ):
                raise ValueError(f"supplementary source {source.source_id} changed during inspection")
    except OSError as exc:
        raise ValueError(f"supplementary source {source.source_id} local PDF is unreadable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)

    pdf_sha256 = f"sha256:{digest.hexdigest()}"
    if source.expected_pdf_sha256 and source.expected_pdf_sha256 != pdf_sha256:
        raise ValueError(f"supplementary source {source.source_id} PDF hash does not match expected_pdf_sha256")
    return OledSupplementarySourceEnvelope(
        source_id=source.source_id,
        pdf_sha256=pdf_sha256,
        byte_size=byte_size,
        content_type="application/pdf",
        provenance_category=source.provenance_category,
        access_policy=source.access_policy,
        provenance_note=source.provenance_note,
        pdf_header_valid=True,
        pdf_eof_marker_valid=True,
        page_count_validated=False,
    )


def _validate_manifest_binding(
    recovery_plan: OledSupplementaryEvidenceRecoveryPlan,
    intake_manifest: OledSupplementarySourceIntakeManifest,
) -> None:
    expected_values = {
        "paper_id": recovery_plan.paper_id,
        "source_request_digest": recovery_plan.source_request_digest,
        "source_mapping_result_digest": recovery_plan.source_mapping_result_digest,
        "source_context_digest": recovery_plan.source_context_digest,
        "recovery_plan_digest": recovery_plan.plan_digest,
    }
    for field_name, expected in expected_values.items():
        if getattr(intake_manifest, field_name) != expected:
            raise ValueError(f"supplementary source intake manifest {field_name} does not match recovery plan")
    expected_item_ids = {item.item_id for item in recovery_plan.items}
    decision_item_ids = {decision.recovery_item_id for decision in intake_manifest.decisions}
    if decision_item_ids != expected_item_ids:
        raise ValueError("supplementary source intake decisions must cover every recovery item exactly once")


def _normalize_sha256(value: Any, *, field_name: str, allow_empty: bool) -> str:
    clean = str(value or "").strip()
    if not clean and allow_empty:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_audit_text(value: Any, *, field_name: str, required: bool) -> str:
    clean = str(value or "").strip()
    if not clean:
        if required:
            raise ValueError(f"{field_name} is required")
        return ""
    lowered = clean.lower()
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        raise ValueError(f"{field_name} contains forbidden credential-like text")
    if "://" in clean or "/" in clean or "\\" in clean:
        raise ValueError(f"{field_name} must not contain a path or URL")
    return clean


def _stable_hash(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode("utf-8")).hexdigest()}"


def _intake_plan_digest(plan: OledSupplementarySourceIntakePlan) -> str:
    payload = plan.model_dump(mode="json", exclude={"intake_plan_digest"})
    return _stable_hash(payload)


__all__ = [
    "DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES",
    "SUPPLEMENTARY_SOURCE_INTAKE_MANIFEST_VERSION",
    "SUPPLEMENTARY_SOURCE_INTAKE_PLAN_VERSION",
    "OledSupplementaryLocalSource",
    "OledSupplementarySourceEnvelope",
    "OledSupplementarySourceIntakeDecision",
    "OledSupplementarySourceIntakeDecisionEntry",
    "OledSupplementarySourceIntakeItem",
    "OledSupplementarySourceIntakeManifest",
    "OledSupplementarySourceIntakePlan",
    "OledSupplementarySourceParseEligibility",
    "build_oled_supplementary_source_intake_plan",
    "inspect_oled_supplementary_source_pdf",
]
