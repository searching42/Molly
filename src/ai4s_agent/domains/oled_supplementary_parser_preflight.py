from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.domains.oled_supplementary_evidence_recovery import (
    OledSupplementaryEvidenceRecoveryPlan,
    OledSupplementaryRecoveryStatus,
    OledSupplementaryTargetKind,
)
from ai4s_agent.domains.oled_supplementary_source_intake import (
    DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES,
    OledSupplementaryLocalSource,
    OledSupplementarySourceEnvelope,
    OledSupplementarySourceIntakeDecision,
    OledSupplementarySourceIntakePlan,
    OledSupplementarySourceParseEligibility,
)


SUPPLEMENTARY_PARSER_PREFLIGHT_MANIFEST_VERSION = "oled_supplementary_parser_preflight_manifest.v1"
SUPPLEMENTARY_PARSER_PREFLIGHT_PLAN_VERSION = "oled_supplementary_parser_preflight_plan.v1"
DEFAULT_MAX_SUPPLEMENTARY_PDF_PAGES = 1_000

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
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
_PARSE_SCOPE = "full_source_then_locator_review"
_PDF_TRAILER_SCAN_BYTES = 8192


class OledSupplementaryParserPreflightSource(BaseModel):
    """Operator-local rebinding of one approved source ID to its current PDF path."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    local_pdf_path: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="source_id")

    @field_validator("local_pdf_path")
    @classmethod
    def validate_local_pdf_path(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("local_pdf_path is required")
        return clean


class OledSupplementaryParserPreflightManifest(BaseModel):
    """Human confirmation for a narrow, non-executing supplementary parse preflight."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_PARSER_PREFLIGHT_MANIFEST_VERSION
    paper_id: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    recovery_plan_digest: str
    intake_plan_digest: str
    parse_confirmed: bool = False
    reviewed_by: str
    reviewed_at: str
    selected_recovery_item_ids: list[str] = Field(default_factory=list)
    sources: list[OledSupplementaryParserPreflightSource] = Field(default_factory=list)

    @field_validator(
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

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_PARSER_PREFLIGHT_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary parser preflight manifest schema_version")
        return value

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="reviewed_by")

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

    @field_validator("selected_recovery_item_ids")
    @classmethod
    def validate_selected_item_ids(cls, value: list[str]) -> list[str]:
        return [_validate_safe_id(item, field_name="selected_recovery_item_id") for item in value]

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledSupplementaryParserPreflightManifest:
        if not self.parse_confirmed:
            raise ValueError("supplementary parser preflight requires parse_confirmed=true")
        if not self.selected_recovery_item_ids:
            raise ValueError("supplementary parser preflight requires selected_recovery_item_ids")
        if not self.sources:
            raise ValueError("supplementary parser preflight requires local source rebindings")
        if len(self.selected_recovery_item_ids) != len(set(self.selected_recovery_item_ids)):
            raise ValueError("duplicate supplementary selected_recovery_item_id")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("duplicate supplementary parser preflight source_id")
        return self


class OledSupplementaryParserPreflightSourceEnvelope(BaseModel):
    """Redacted current-PDF facts verified for one source during preflight."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    pdf_sha256: str
    byte_size: int = Field(gt=0)
    page_count: int = Field(ge=1)
    content_type: str = "application/pdf"
    pdf_header_valid: bool = True
    pdf_eof_marker_valid: bool = True
    page_count_validated: bool = True

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_safe_id(value, field_name="source_id")

    @field_validator("pdf_sha256")
    @classmethod
    def validate_pdf_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="pdf_sha256")

    @model_validator(mode="after")
    def validate_envelope(self) -> OledSupplementaryParserPreflightSourceEnvelope:
        if self.content_type != "application/pdf":
            raise ValueError("supplementary parser preflight source content_type must be application/pdf")
        if not self.pdf_header_valid or not self.pdf_eof_marker_valid:
            raise ValueError("supplementary parser preflight source must retain a valid PDF envelope")
        if not self.page_count_validated:
            raise ValueError("supplementary parser preflight source must validate page_count")
        return self


class OledSupplementaryParserPreflightItem(BaseModel):
    """One approved explicit target that a later parser may process in full-source scope."""

    model_config = ConfigDict(extra="forbid")

    recovery_item_id: str
    source_id: str
    source_pdf_sha256: str
    target_kind: OledSupplementaryTargetKind
    target_locator: str
    parse_scope: str = _PARSE_SCOPE

    @field_validator("recovery_item_id", "source_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_safe_id(value, field_name=str(info.field_name))

    @field_validator("source_pdf_sha256")
    @classmethod
    def validate_source_pdf_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="source_pdf_sha256")

    @field_validator("target_locator")
    @classmethod
    def validate_target_locator(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("target_locator is required")
        return clean

    @model_validator(mode="after")
    def validate_item(self) -> OledSupplementaryParserPreflightItem:
        if self.target_kind not in {
            OledSupplementaryTargetKind.TABLE,
            OledSupplementaryTargetKind.FIGURE,
        }:
            raise ValueError("supplementary parser preflight item requires a table or figure target")
        if self.parse_scope != _PARSE_SCOPE:
            raise ValueError("supplementary parser preflight item must use full-source locator review")
        return self


class OledSupplementaryParserPreflightPlan(BaseModel):
    """Content-bound, non-executing plan for a future supplementary parser invocation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_PARSER_PREFLIGHT_PLAN_VERSION
    paper_id: str
    source_request_digest: str
    source_mapping_result_digest: str
    source_context_digest: str
    recovery_plan_digest: str
    intake_plan_digest: str
    parse_confirmed: bool = True
    reviewed_by: str
    reviewed_at: str
    source_envelopes: list[OledSupplementaryParserPreflightSourceEnvelope] = Field(default_factory=list)
    items: list[OledSupplementaryParserPreflightItem] = Field(default_factory=list)
    source_count: int = Field(ge=0)
    item_count: int = Field(ge=0)
    preflight_plan_digest: str
    review_only: bool = True
    executable: bool = False
    offline_only: bool = True
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    pdf_content_parsed: bool = False
    pdf_page_count_validated: bool = True
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
        "preflight_plan_digest",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError(f"{info.field_name} is required")
        return clean

    @field_validator("reviewed_by")
    @classmethod
    def validate_reviewed_by(cls, value: str) -> str:
        return _validate_audit_text(value, field_name="reviewed_by")

    @field_validator("reviewed_at")
    @classmethod
    def validate_reviewed_at(cls, value: str) -> str:
        return OledSupplementaryParserPreflightManifest.validate_reviewed_at(value)

    @model_validator(mode="after")
    def validate_plan_integrity(self) -> OledSupplementaryParserPreflightPlan:
        if self.schema_version != SUPPLEMENTARY_PARSER_PREFLIGHT_PLAN_VERSION:
            raise ValueError("unexpected supplementary parser preflight plan schema_version")
        if not self.parse_confirmed:
            raise ValueError("supplementary parser preflight plan requires parse_confirmed=true")
        if self.source_count != len(self.source_envelopes):
            raise ValueError("supplementary parser preflight source_count does not match source_envelopes")
        if self.item_count != len(self.items) or not self.items:
            raise ValueError("supplementary parser preflight item_count does not match selected items")
        source_ids = [source.source_id for source in self.source_envelopes]
        item_ids = [item.recovery_item_id for item in self.items]
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise ValueError("supplementary parser preflight source envelopes must be sorted with unique source IDs")
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("supplementary parser preflight items must be sorted with unique recovery item IDs")
        source_hashes = {source.source_id: source.pdf_sha256 for source in self.source_envelopes}
        if {item.source_id for item in self.items} != set(source_ids):
            raise ValueError("supplementary parser preflight sources must be explicitly selected by an item")
        for item in self.items:
            if item.source_pdf_sha256 != source_hashes.get(item.source_id):
                raise ValueError("supplementary parser preflight item hash does not match source envelope")
        fixed_false_flags = (
            "executable",
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "pdf_content_parsed",
            "supplementary_downloaded",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if not self.review_only or not self.offline_only or not self.pdf_page_count_validated:
            raise ValueError("supplementary parser preflight must remain offline, review-only, and page-count validated")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary parser preflight unexpectedly records an execution side effect")
        if _preflight_plan_digest(self) != self.preflight_plan_digest:
            raise ValueError("supplementary parser preflight plan digest does not match canonical content")
        return self


def build_oled_supplementary_parser_preflight_plan(
    recovery_plan: OledSupplementaryEvidenceRecoveryPlan,
    intake_plan: OledSupplementarySourceIntakePlan,
    parse_manifest: OledSupplementaryParserPreflightManifest,
    *,
    max_pdf_bytes: int = DEFAULT_MAX_SUPPLEMENTARY_PDF_BYTES,
    max_pdf_pages: int = DEFAULT_MAX_SUPPLEMENTARY_PDF_PAGES,
) -> OledSupplementaryParserPreflightPlan:
    """Rebind approved PDFs, verify their unchanged bytes and page counts, and plan no execution."""

    if max_pdf_bytes <= 0:
        raise ValueError("max_pdf_bytes must be positive")
    if max_pdf_pages <= 0:
        raise ValueError("max_pdf_pages must be positive")
    recovery_plan = OledSupplementaryEvidenceRecoveryPlan.model_validate(recovery_plan.model_dump(mode="json"))
    intake_plan = OledSupplementarySourceIntakePlan.model_validate(intake_plan.model_dump(mode="json"))
    parse_manifest = OledSupplementaryParserPreflightManifest.model_validate(parse_manifest.model_dump(mode="json"))
    _validate_recovery_and_intake_binding(recovery_plan, intake_plan)
    _validate_manifest_binding(intake_plan, parse_manifest)

    intake_items = {item.recovery_item_id: item for item in intake_plan.items}
    recovery_items = {item.item_id: item for item in recovery_plan.items}
    selected_ids = sorted(parse_manifest.selected_recovery_item_ids)
    selected_intake_items = []
    for recovery_item_id in selected_ids:
        intake_item = intake_items.get(recovery_item_id)
        recovery_item = recovery_items.get(recovery_item_id)
        if intake_item is None or recovery_item is None:
            raise ValueError("supplementary parser preflight selected item is not present in the bound source chain")
        _validate_selected_item(recovery_item, intake_item)
        selected_intake_items.append(intake_item)

    selected_source_ids = {item.source_id for item in selected_intake_items}
    if None in selected_source_ids:
        raise ValueError("supplementary parser preflight selected item is missing a source binding")
    selected_source_ids = {str(source_id) for source_id in selected_source_ids}
    manifest_sources = {source.source_id: source for source in parse_manifest.sources}
    if set(manifest_sources) != selected_source_ids:
        raise ValueError("supplementary parser preflight local source rebindings must exactly cover selected items")
    intake_sources = {source.source_id: source for source in intake_plan.source_envelopes}

    source_envelopes: list[OledSupplementaryParserPreflightSourceEnvelope] = []
    for source_id in sorted(selected_source_ids):
        source_binding = manifest_sources[source_id]
        intake_source = intake_sources.get(source_id)
        if intake_source is None:
            raise ValueError("supplementary parser preflight source is not present in the intake artifact")
        local_source = OledSupplementaryLocalSource(
            source_id=source_id,
            local_pdf_path=source_binding.local_pdf_path,
            expected_pdf_sha256=intake_source.pdf_sha256,
            provenance_category=intake_source.provenance_category,
            access_policy=intake_source.access_policy,
            provenance_note=intake_source.provenance_note,
        )
        source_envelopes.append(
            _inspect_and_count_bound_pdf(
                local_source,
                intake_source,
                max_pdf_bytes=max_pdf_bytes,
                max_pdf_pages=max_pdf_pages,
            )
        )

    source_hashes = {source.source_id: source.pdf_sha256 for source in source_envelopes}
    items = [
        OledSupplementaryParserPreflightItem(
            recovery_item_id=item.recovery_item_id,
            source_id=str(item.source_id),
            source_pdf_sha256=source_hashes[str(item.source_id)],
            target_kind=item.target_kind,
            target_locator=str(item.target_locator),
            parse_scope=_PARSE_SCOPE,
        )
        for item in selected_intake_items
    ]
    items.sort(key=lambda item: item.recovery_item_id)
    source_envelopes.sort(key=lambda source: source.source_id)
    payload: dict[str, Any] = {
        "schema_version": SUPPLEMENTARY_PARSER_PREFLIGHT_PLAN_VERSION,
        "paper_id": intake_plan.paper_id,
        "source_request_digest": intake_plan.source_request_digest,
        "source_mapping_result_digest": intake_plan.source_mapping_result_digest,
        "source_context_digest": intake_plan.source_context_digest,
        "recovery_plan_digest": intake_plan.recovery_plan_digest,
        "intake_plan_digest": intake_plan.intake_plan_digest,
        "parse_confirmed": True,
        "reviewed_by": parse_manifest.reviewed_by,
        "reviewed_at": parse_manifest.reviewed_at,
        "source_envelopes": [source.model_dump(mode="json") for source in source_envelopes],
        "items": [item.model_dump(mode="json") for item in items],
        "source_count": len(source_envelopes),
        "item_count": len(items),
        "preflight_plan_digest": "",
        "review_only": True,
        "executable": False,
        "offline_only": True,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "pdf_content_parsed": False,
        "pdf_page_count_validated": True,
        "supplementary_downloaded": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["preflight_plan_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "preflight_plan_digest"}
    )
    return OledSupplementaryParserPreflightPlan.model_validate(payload)


def _validate_recovery_and_intake_binding(
    recovery_plan: OledSupplementaryEvidenceRecoveryPlan,
    intake_plan: OledSupplementarySourceIntakePlan,
) -> None:
    expected_values = {
        "paper_id": recovery_plan.paper_id,
        "source_request_digest": recovery_plan.source_request_digest,
        "source_mapping_result_digest": recovery_plan.source_mapping_result_digest,
        "source_context_digest": recovery_plan.source_context_digest,
        "recovery_plan_digest": recovery_plan.plan_digest,
    }
    for field_name, expected in expected_values.items():
        if getattr(intake_plan, field_name) != expected:
            raise ValueError(f"supplementary parser preflight intake {field_name} does not match recovery plan")
    recovery_items = {item.item_id: item for item in recovery_plan.items}
    intake_items = {item.recovery_item_id: item for item in intake_plan.items}
    if set(recovery_items) != set(intake_items):
        raise ValueError("supplementary parser preflight intake items do not match recovery plan")
    for recovery_item_id, recovery_item in recovery_items.items():
        intake_item = intake_items[recovery_item_id]
        if (
            intake_item.recovery_status != recovery_item.status
            or intake_item.target_kind != recovery_item.target_kind
            or intake_item.target_locator != recovery_item.target_locator
        ):
            raise ValueError("supplementary parser preflight intake target does not match recovery plan")


def _validate_manifest_binding(
    intake_plan: OledSupplementarySourceIntakePlan,
    parse_manifest: OledSupplementaryParserPreflightManifest,
) -> None:
    expected_values = {
        "paper_id": intake_plan.paper_id,
        "source_request_digest": intake_plan.source_request_digest,
        "source_mapping_result_digest": intake_plan.source_mapping_result_digest,
        "source_context_digest": intake_plan.source_context_digest,
        "recovery_plan_digest": intake_plan.recovery_plan_digest,
        "intake_plan_digest": intake_plan.intake_plan_digest,
    }
    for field_name, expected in expected_values.items():
        if getattr(parse_manifest, field_name) != expected:
            raise ValueError(f"supplementary parser preflight manifest {field_name} does not match source intake")


def _validate_selected_item(recovery_item: Any, intake_item: Any) -> None:
    if recovery_item.status != OledSupplementaryRecoveryStatus.EXPLICIT_REFERENCE_FOUND:
        raise ValueError("supplementary parser preflight only permits explicit recovery targets")
    if intake_item.decision != OledSupplementarySourceIntakeDecision.APPROVED:
        raise ValueError("supplementary parser preflight only permits approved intake items")
    if intake_item.parse_eligibility != OledSupplementarySourceParseEligibility.ELIGIBLE_FOR_TARGETED_SOURCE_PARSE:
        raise ValueError("supplementary parser preflight item is not eligible for targeted source parsing")
    if intake_item.target_kind not in {
        OledSupplementaryTargetKind.TABLE,
        OledSupplementaryTargetKind.FIGURE,
    } or not intake_item.target_locator:
        raise ValueError("supplementary parser preflight requires an explicit table or figure locator")
    if not intake_item.source_id or not intake_item.source_pdf_sha256:
        raise ValueError("supplementary parser preflight selected item requires an intake source binding")


def _inspect_and_count_bound_pdf(
    source: OledSupplementaryLocalSource,
    expected: OledSupplementarySourceEnvelope,
    *,
    max_pdf_bytes: int,
    max_pdf_pages: int,
) -> OledSupplementaryParserPreflightSourceEnvelope:
    """Bind envelope, hash, and page count to one no-follow file descriptor.

    Page count cannot be collected through a second path lookup: otherwise a
    local path could be replaced temporarily and yield a different document's
    page count while the before/after hashes still describe the approved file.
    """

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
            raise ValueError("supplementary parser preflight requires O_NOFOLLOW support")
        descriptor = os.open(path, os.O_RDONLY | no_follow)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"supplementary source {source.source_id} local PDF is unavailable") from exc

    try:
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            initial_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(initial_stat.st_mode):
                raise ValueError(f"supplementary source {source.source_id} must be a regular file")
            byte_size = int(initial_stat.st_size)
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
            pdfplumber = _load_pdfplumber()
            try:
                handle.seek(0)
                with pdfplumber.open(handle, pages=tuple(range(1, max_pdf_pages + 2))) as pdf:
                    page_count = len(pdf.pages)
            except Exception as exc:
                raise ValueError("supplementary source PDF page-count inspection failed") from exc
            if page_count <= 0:
                raise ValueError("supplementary source PDF must contain at least one page")
            if page_count > max_pdf_pages:
                raise ValueError("supplementary source PDF page count exceeds the configured limit")
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
                or final_stat.st_mtime_ns != initial_stat.st_mtime_ns
                or final_stat.st_ctime_ns != initial_stat.st_ctime_ns
                or bytes_read != byte_size
            ):
                raise ValueError(f"supplementary source {source.source_id} changed during parser preflight")
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"supplementary source {source.source_id} local PDF is unreadable") from exc
    finally:
        if descriptor != -1:
            os.close(descriptor)

    pdf_sha256 = f"sha256:{digest.hexdigest()}"
    if pdf_sha256 != source.expected_pdf_sha256:
        raise ValueError(f"supplementary source {source.source_id} PDF hash does not match expected_pdf_sha256")
    if pdf_sha256 != expected.pdf_sha256 or byte_size != expected.byte_size:
        raise ValueError("supplementary source PDF no longer matches the intake artifact")
    return OledSupplementaryParserPreflightSourceEnvelope(
        source_id=source.source_id,
        pdf_sha256=pdf_sha256,
        byte_size=byte_size,
        page_count=page_count,
        content_type="application/pdf",
        pdf_header_valid=True,
        pdf_eof_marker_valid=True,
        page_count_validated=True,
    )


def _load_pdfplumber() -> Any:
    try:
        return importlib.import_module("pdfplumber")
    except ImportError as exc:
        raise ValueError("supplementary parser preflight requires pdfplumber for page-count validation") from exc


def _validate_safe_id(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_ID_RE.fullmatch(clean):
        raise ValueError(f"{field_name} must use only letters, numbers, dot, dash, underscore, and colon")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(1).lower()}"


def _validate_audit_text(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is required")
    lowered = clean.lower()
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        raise ValueError(f"{field_name} contains forbidden credential-like text")
    if "://" in clean or "/" in clean or "\\" in clean:
        raise ValueError(f"{field_name} must not contain a path or URL")
    return clean


def _stable_hash(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _preflight_plan_digest(plan: OledSupplementaryParserPreflightPlan) -> str:
    payload = plan.model_dump(mode="json", exclude={"preflight_plan_digest"})
    return _stable_hash(payload)


__all__ = [
    "DEFAULT_MAX_SUPPLEMENTARY_PDF_PAGES",
    "SUPPLEMENTARY_PARSER_PREFLIGHT_MANIFEST_VERSION",
    "SUPPLEMENTARY_PARSER_PREFLIGHT_PLAN_VERSION",
    "OledSupplementaryParserPreflightItem",
    "OledSupplementaryParserPreflightManifest",
    "OledSupplementaryParserPreflightPlan",
    "OledSupplementaryParserPreflightSource",
    "OledSupplementaryParserPreflightSourceEnvelope",
    "build_oled_supplementary_parser_preflight_plan",
]
