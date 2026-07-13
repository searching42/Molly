from __future__ import annotations

import hashlib
import html
import json
import math
import re
import unicodedata
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.domains.oled_supplementary_evidence_recovery import OledSupplementaryTargetKind
from ai4s_agent.domains.oled_supplementary_mineru_execution import (
    OledSupplementaryMineruExecutionArtifact,
    OledSupplementaryMineruExecutionStatus,
    OledSupplementaryMineruOutputKind,
    OledSupplementaryMineruSourceExecutionResult,
)
from ai4s_agent.schemas import ParsedDocument, ParsedTable


SUPPLEMENTARY_LOCATOR_MANIFEST_VERSION = "oled_supplementary_locator_manifest.v1"
SUPPLEMENTARY_LOCATOR_REVIEW_ARTIFACT_VERSION = "oled_supplementary_locator_review.v1"

MAX_PARSED_TABLES = 5_000
MAX_REVIEW_TABLE_ROWS = 2_000
MAX_REVIEW_TABLE_COLUMNS = 200
MAX_REVIEW_TABLE_TEXT_CHARS = 2_000_000

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
_SHA256_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
_HTML_TAG_RE = re.compile(r"<[^>]*>")
_LEADING_MARKUP_RE = re.compile(r"^[\s#>*_`\-\[\]()]+")
_TABLE_LOCATOR_RE = re.compile(
    r"^(?:(?:supplementary|supporting\s+information)\s+)?table\s+(S[0-9]+[A-Za-z]?)$",
    re.IGNORECASE,
)
_BARE_TABLE_LOCATOR_RE = re.compile(r"^(S[0-9]+[A-Za-z]?)$", re.IGNORECASE)


class OledSupplementaryLocatorMatchStatus(str, Enum):
    EXACT_MATCH = "exact_match"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED_TARGET_KIND = "unsupported_target_kind"
    UNSUPPORTED_LOCATOR_FORMAT = "unsupported_locator_format"


class OledSupplementaryLocatorReviewStatus(str, Enum):
    READY_FOR_HUMAN_REVIEW = "ready_for_human_review"
    MANUAL_LOCATOR_REVIEW_REQUIRED = "manual_locator_review_required"


class OledSupplementaryLocatorReviewDecision(str, Enum):
    PENDING = "pending"


class OledSupplementaryLocatorSource(BaseModel):
    """Operator-local binding to one normalized ParsedDocument JSON file."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    parsed_document_json: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("parsed_document_json")
    @classmethod
    def validate_parsed_document_json(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("parsed_document_json is required")
        return clean


class OledSupplementaryLocatorManifest(BaseModel):
    """Exact local bindings for a previously content-bound execution artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_LOCATOR_MANIFEST_VERSION
    run_id: str
    paper_id: str
    execution_artifact_sha256: str
    execution_artifact_digest: str
    sources: list[OledSupplementaryLocatorSource] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_LOCATOR_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary locator manifest schema_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("execution_artifact_sha256", "execution_artifact_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledSupplementaryLocatorManifest:
        if not self.sources:
            raise ValueError("supplementary locator manifest requires parsed-document bindings")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("duplicate supplementary locator source_id")
        return self


class OledSupplementaryReviewTable(BaseModel):
    """Reviewer-facing table content copied verbatim from a bound ParsedDocument."""

    model_config = ConfigDict(extra="forbid")

    table_id: str
    page: int = Field(ge=0)
    caption: str
    headers: list[str] = Field(default_factory=list)
    rows: list[dict[str, str]] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    source_bbox: dict[str, float] | None = None
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    table_content_digest: str

    @field_validator("table_id")
    @classmethod
    def validate_table_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="table_id")

    @field_validator("table_content_digest")
    @classmethod
    def validate_table_content_digest(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="table_content_digest")

    @field_validator("source_bbox")
    @classmethod
    def validate_source_bbox(
        cls,
        value: dict[str, float] | None,
    ) -> dict[str, float] | None:
        if value is not None and any(not math.isfinite(item) for item in value.values()):
            raise ValueError("review table source_bbox values must be finite")
        return value

    @model_validator(mode="after")
    def validate_table_shape(self) -> OledSupplementaryReviewTable:
        if self.row_count != len(self.rows):
            raise ValueError("review table row_count mismatch")
        columns = set(self.headers)
        for row in self.rows:
            columns.update(row)
        if self.column_count != len(columns):
            raise ValueError("review table column_count mismatch")
        if self.row_count > MAX_REVIEW_TABLE_ROWS or self.column_count > MAX_REVIEW_TABLE_COLUMNS:
            raise ValueError("review table exceeds bounded packet dimensions")
        text_chars = len(self.caption)
        text_chars += sum(len(item) for item in self.headers)
        text_chars += sum(len(key) + len(value) for row in self.rows for key, value in row.items())
        text_chars += sum(len(item) for item in self.footnotes)
        if text_chars > MAX_REVIEW_TABLE_TEXT_CHARS:
            raise ValueError("review table exceeds bounded packet text size")
        if _review_table_digest(self) != self.table_content_digest:
            raise ValueError("review table content digest mismatch")
        return self


class OledSupplementaryLocatorReviewItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_item_id: str
    recovery_item_id: str
    source_id: str
    source_pdf_sha256: str
    parsed_document_sha256: str
    parser_backend: str
    target_kind: OledSupplementaryTargetKind
    target_locator: str
    canonical_locator: str = ""
    match_status: OledSupplementaryLocatorMatchStatus
    candidate_table_ids: list[str] = Field(default_factory=list)
    matched_table: OledSupplementaryReviewTable | None = None
    parser_warning_codes: list[str] = Field(default_factory=list)
    review_decision: OledSupplementaryLocatorReviewDecision = (
        OledSupplementaryLocatorReviewDecision.PENDING
    )
    review_guidance: str

    @field_validator("review_item_id", "recovery_item_id")
    @classmethod
    def validate_bound_ids(cls, value: str, info: Any) -> str:
        return _validate_bound_id(value, field_name=str(info.field_name))

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("source_pdf_sha256", "parsed_document_sha256")
    @classmethod
    def validate_hashes(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("candidate_table_ids", "parser_warning_codes")
    @classmethod
    def validate_sorted_bound_ids(cls, value: list[str], info: Any) -> list[str]:
        clean = [_validate_bound_id(item, field_name=str(info.field_name)) for item in value]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError(f"{info.field_name} must be sorted and unique")
        return clean

    @model_validator(mode="after")
    def validate_match_shape(self) -> OledSupplementaryLocatorReviewItem:
        exact = self.match_status == OledSupplementaryLocatorMatchStatus.EXACT_MATCH
        if exact:
            if self.target_kind != OledSupplementaryTargetKind.TABLE:
                raise ValueError("only table targets can be resolved by the locator MVP")
            if self.matched_table is None or self.candidate_table_ids != [self.matched_table.table_id]:
                raise ValueError("exact locator match requires exactly one bound review table")
            if not self.canonical_locator:
                raise ValueError("exact locator match requires a canonical locator")
        elif self.matched_table is not None:
            raise ValueError("unresolved locator items must not select table content")
        if self.match_status == OledSupplementaryLocatorMatchStatus.AMBIGUOUS:
            if len(self.candidate_table_ids) < 2:
                raise ValueError("ambiguous locator item requires multiple candidate table ids")
        elif not exact and self.candidate_table_ids:
            raise ValueError("only exact or ambiguous locator items may list candidate tables")
        if self.review_decision != OledSupplementaryLocatorReviewDecision.PENDING:
            raise ValueError("generated locator packets must remain pending human review")
        return self


class OledSupplementaryLocatorReviewArtifact(BaseModel):
    """Content-bound, review-only artifact that cannot admit scientific records."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = SUPPLEMENTARY_LOCATOR_REVIEW_ARTIFACT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    execution_artifact_sha256: str
    execution_artifact_digest: str
    locator_manifest_sha256: str
    preflight_plan_digest: str
    endpoint_profile_name: str
    backend: str
    status: OledSupplementaryLocatorReviewStatus
    source_count: int = Field(ge=1)
    item_count: int = Field(ge=1)
    exact_match_count: int = Field(ge=0)
    unresolved_item_count: int = Field(ge=0)
    review_items: list[OledSupplementaryLocatorReviewItem] = Field(default_factory=list)
    review_artifact_digest: str
    review_only: bool = True
    human_review_required: bool = True
    offline_only: bool = True
    scientific_content_included: bool = True
    parsed_output_read: bool = True
    locator_resolution_attempted: bool = True
    locator_resolved: bool
    network_accessed: bool = False
    external_service_called: bool = False
    llm_called: bool = False
    mineru_called: bool = False
    pdf_content_read: bool = False
    candidate_regenerated: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_LOCATOR_REVIEW_ARTIFACT_VERSION:
            raise ValueError("unexpected supplementary locator review artifact_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator(
        "execution_artifact_sha256",
        "execution_artifact_digest",
        "locator_manifest_sha256",
        "preflight_plan_digest",
        "review_artifact_digest",
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
            datetime.fromisoformat(clean.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("generated_at must be ISO-8601") from exc
        return clean

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledSupplementaryLocatorReviewArtifact:
        if not self.review_items or self.item_count != len(self.review_items):
            raise ValueError("supplementary locator item_count mismatch")
        item_ids = [item.review_item_id for item in self.review_items]
        if item_ids != sorted(item_ids) or len(item_ids) != len(set(item_ids)):
            raise ValueError("supplementary locator review items must be sorted and unique")
        source_ids = {item.source_id for item in self.review_items}
        if self.source_count != len(source_ids):
            raise ValueError("supplementary locator source_count mismatch")
        source_bindings: dict[str, tuple[str, str, str]] = {}
        for item in self.review_items:
            binding = (
                item.source_pdf_sha256,
                item.parsed_document_sha256,
                item.parser_backend,
            )
            prior = source_bindings.setdefault(item.source_id, binding)
            if prior != binding:
                raise ValueError("supplementary locator source binding is inconsistent")
        exact_count = sum(
            item.match_status == OledSupplementaryLocatorMatchStatus.EXACT_MATCH
            for item in self.review_items
        )
        if self.exact_match_count != exact_count:
            raise ValueError("supplementary locator exact_match_count mismatch")
        if self.unresolved_item_count != self.item_count - exact_count:
            raise ValueError("supplementary locator unresolved_item_count mismatch")
        all_resolved = exact_count == self.item_count
        if self.locator_resolved != all_resolved:
            raise ValueError("supplementary locator resolved flag mismatch")
        expected_status = (
            OledSupplementaryLocatorReviewStatus.READY_FOR_HUMAN_REVIEW
            if all_resolved
            else OledSupplementaryLocatorReviewStatus.MANUAL_LOCATOR_REVIEW_REQUIRED
        )
        if self.status != expected_status:
            raise ValueError("supplementary locator review status mismatch")
        fixed_true_flags = (
            "review_only",
            "human_review_required",
            "offline_only",
            "scientific_content_included",
            "parsed_output_read",
            "locator_resolution_attempted",
        )
        fixed_false_flags = (
            "network_accessed",
            "external_service_called",
            "llm_called",
            "mineru_called",
            "pdf_content_read",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if any(not getattr(self, field_name) for field_name in fixed_true_flags):
            raise ValueError("supplementary locator review lost a required review-only flag")
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary locator review crossed a downstream boundary")
        if _review_artifact_digest(self) != self.review_artifact_digest:
            raise ValueError("supplementary locator review artifact digest mismatch")
        return self


def validate_oled_supplementary_locator_binding(
    execution_artifact: OledSupplementaryMineruExecutionArtifact,
    manifest: OledSupplementaryLocatorManifest,
    *,
    execution_artifact_sha256: str,
) -> None:
    execution = OledSupplementaryMineruExecutionArtifact.model_validate(
        execution_artifact.model_dump(mode="json")
    )
    manifest = OledSupplementaryLocatorManifest.model_validate(manifest.model_dump(mode="json"))
    observed_sha256 = _normalize_sha256(
        execution_artifact_sha256,
        field_name="execution_artifact_sha256",
    )
    if execution.status != OledSupplementaryMineruExecutionStatus.SUCCESS:
        raise ValueError("supplementary locator review requires a successful execution artifact")
    if manifest.run_id != execution.run_id or manifest.paper_id != execution.paper_id:
        raise ValueError("supplementary locator manifest identity does not match execution artifact")
    if manifest.execution_artifact_sha256 != observed_sha256:
        raise ValueError("supplementary locator manifest does not bind the execution artifact bytes")
    if manifest.execution_artifact_digest != execution.execution_artifact_digest:
        raise ValueError("supplementary locator manifest does not bind execution canonical content")
    expected_source_ids = {source.source_id for source in execution.source_results}
    manifest_source_ids = {source.source_id for source in manifest.sources}
    if manifest_source_ids != expected_source_ids:
        raise ValueError("supplementary locator sources must exactly cover execution sources")


def build_oled_supplementary_locator_review_artifact(
    *,
    execution_artifact: OledSupplementaryMineruExecutionArtifact,
    execution_artifact_sha256: str,
    locator_manifest: OledSupplementaryLocatorManifest,
    locator_manifest_sha256: str,
    parsed_documents: dict[str, tuple[ParsedDocument, str]],
    generated_at: str,
) -> OledSupplementaryLocatorReviewArtifact:
    execution = OledSupplementaryMineruExecutionArtifact.model_validate(
        execution_artifact.model_dump(mode="json")
    )
    manifest = OledSupplementaryLocatorManifest.model_validate(
        locator_manifest.model_dump(mode="json")
    )
    validate_oled_supplementary_locator_binding(
        execution,
        manifest,
        execution_artifact_sha256=execution_artifact_sha256,
    )
    expected_source_ids = {source.source_id for source in execution.source_results}
    if set(parsed_documents) != expected_source_ids:
        raise ValueError("parsed documents must exactly cover execution sources")

    items: list[OledSupplementaryLocatorReviewItem] = []
    for source_result in execution.source_results:
        parsed_document, parsed_sha256 = parsed_documents[source_result.source_id]
        parsed = ParsedDocument.model_validate(parsed_document.model_dump(mode="json"))
        normalized_parsed_sha256 = _normalize_sha256(
            parsed_sha256,
            field_name="parsed_document_sha256",
        )
        _validate_parsed_document_binding(
            source_result,
            parsed,
            normalized_parsed_sha256,
        )
        for target in source_result.targets:
            items.append(
                _build_review_item(
                    source_result=source_result,
                    target_kind=target.target_kind,
                    target_locator=target.target_locator,
                    recovery_item_id=target.recovery_item_id,
                    parsed_document=parsed,
                    parsed_document_sha256=normalized_parsed_sha256,
                )
            )
    items.sort(key=lambda item: item.review_item_id)
    exact_count = sum(
        item.match_status == OledSupplementaryLocatorMatchStatus.EXACT_MATCH for item in items
    )
    all_resolved = exact_count == len(items)
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_LOCATOR_REVIEW_ARTIFACT_VERSION,
        "run_id": execution.run_id,
        "paper_id": execution.paper_id,
        "generated_at": str(generated_at or "").strip(),
        "execution_artifact_sha256": _normalize_sha256(
            execution_artifact_sha256,
            field_name="execution_artifact_sha256",
        ),
        "execution_artifact_digest": execution.execution_artifact_digest,
        "locator_manifest_sha256": _normalize_sha256(
            locator_manifest_sha256,
            field_name="locator_manifest_sha256",
        ),
        "preflight_plan_digest": execution.preflight_plan_digest,
        "endpoint_profile_name": execution.endpoint_profile_name,
        "backend": execution.backend,
        "status": (
            OledSupplementaryLocatorReviewStatus.READY_FOR_HUMAN_REVIEW.value
            if all_resolved
            else OledSupplementaryLocatorReviewStatus.MANUAL_LOCATOR_REVIEW_REQUIRED.value
        ),
        "source_count": execution.source_count,
        "item_count": len(items),
        "exact_match_count": exact_count,
        "unresolved_item_count": len(items) - exact_count,
        "review_items": [item.model_dump(mode="json") for item in items],
        "review_artifact_digest": "",
        "review_only": True,
        "human_review_required": True,
        "offline_only": True,
        "scientific_content_included": True,
        "parsed_output_read": True,
        "locator_resolution_attempted": True,
        "locator_resolved": all_resolved,
        "network_accessed": False,
        "external_service_called": False,
        "llm_called": False,
        "mineru_called": False,
        "pdf_content_read": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["review_artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "review_artifact_digest"}
    )
    return OledSupplementaryLocatorReviewArtifact.model_validate(payload)


def _validate_parsed_document_binding(
    source_result: OledSupplementaryMineruSourceExecutionResult,
    parsed_document: ParsedDocument,
    parsed_document_sha256: str,
) -> None:
    output_hashes = {
        item.output_kind: item for item in source_result.output_hashes
    }
    parsed_output = output_hashes.get(OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_JSON)
    if parsed_output is None:
        raise ValueError("execution source is missing a parsed-document output binding")
    if parsed_output.sha256 != parsed_document_sha256:
        raise ValueError("parsed-document hash does not match execution artifact")
    if parsed_document.parser_backend != source_result.parser_backend:
        raise ValueError("parsed-document backend does not match execution source")
    if len(parsed_document.pages) != source_result.page_count:
        raise ValueError("parsed-document page count does not match execution source")
    if len(parsed_document.tables) > MAX_PARSED_TABLES:
        raise ValueError("parsed document exceeds the bounded table count")


def _build_review_item(
    *,
    source_result: OledSupplementaryMineruSourceExecutionResult,
    target_kind: OledSupplementaryTargetKind,
    target_locator: str,
    recovery_item_id: str,
    parsed_document: ParsedDocument,
    parsed_document_sha256: str,
) -> OledSupplementaryLocatorReviewItem:
    review_item_id = f"supplementary-locator-review:{recovery_item_id}"
    base: dict[str, Any] = {
        "review_item_id": review_item_id,
        "recovery_item_id": recovery_item_id,
        "source_id": source_result.source_id,
        "source_pdf_sha256": source_result.source_pdf_sha256,
        "parsed_document_sha256": parsed_document_sha256,
        "parser_backend": source_result.parser_backend,
        "target_kind": target_kind,
        "target_locator": target_locator,
        "parser_warning_codes": source_result.warning_codes,
        "review_decision": OledSupplementaryLocatorReviewDecision.PENDING,
    }
    if target_kind != OledSupplementaryTargetKind.TABLE:
        return OledSupplementaryLocatorReviewItem(
            **base,
            match_status=OledSupplementaryLocatorMatchStatus.UNSUPPORTED_TARGET_KIND,
            review_guidance="This MVP does not resolve figure targets; locate the evidence manually.",
        )
    canonical_locator = _canonical_table_locator(target_locator)
    if not canonical_locator:
        return OledSupplementaryLocatorReviewItem(
            **base,
            match_status=OledSupplementaryLocatorMatchStatus.UNSUPPORTED_LOCATOR_FORMAT,
            review_guidance="The table locator format is unsupported; locate the evidence manually.",
        )
    candidates = [
        table
        for table in parsed_document.tables
        if _caption_matches_table_locator(table.caption, canonical_locator)
    ]
    candidates.sort(key=lambda table: (table.page, table.table_id))
    candidate_ids = sorted(table.table_id for table in candidates)
    if not candidates:
        return OledSupplementaryLocatorReviewItem(
            **base,
            canonical_locator=canonical_locator,
            match_status=OledSupplementaryLocatorMatchStatus.NOT_FOUND,
            review_guidance="No exact anchored caption matched; inspect the bound parsed output manually.",
        )
    if len(candidates) > 1:
        return OledSupplementaryLocatorReviewItem(
            **base,
            canonical_locator=canonical_locator,
            match_status=OledSupplementaryLocatorMatchStatus.AMBIGUOUS,
            candidate_table_ids=candidate_ids,
            review_guidance="Multiple exact anchored captions matched; select no table until manual review.",
        )
    review_table = _review_table(candidates[0])
    return OledSupplementaryLocatorReviewItem(
        **base,
        canonical_locator=canonical_locator,
        match_status=OledSupplementaryLocatorMatchStatus.EXACT_MATCH,
        candidate_table_ids=[review_table.table_id],
        matched_table=review_table,
        review_guidance=(
            "Compare the caption, headers, rows, footnotes, and page anchor with the source; "
            "this packet remains unadmitted until a later explicit review decision."
        ),
    )


def _canonical_table_locator(value: str) -> str:
    normalized = _normalize_caption(value)
    match = _BARE_TABLE_LOCATOR_RE.fullmatch(normalized) or _TABLE_LOCATOR_RE.fullmatch(normalized)
    if not match:
        return ""
    return match.group(1).upper()


def _caption_matches_table_locator(caption: str, locator: str) -> bool:
    normalized = _normalize_caption(caption)
    prefix = r"(?:(?:supplementary|supporting\s+information)\s+)?table\s+"
    return re.match(prefix + re.escape(locator) + r"(?=$|[^0-9A-Za-z])", normalized, re.I) is not None


def _normalize_caption(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(str(value or "")))
    normalized = _HTML_TAG_RE.sub(" ", normalized)
    normalized = _LEADING_MARKUP_RE.sub("", normalized)
    return " ".join(normalized.split()).strip()


def _review_table(table: ParsedTable) -> OledSupplementaryReviewTable:
    columns = set(table.headers)
    for row in table.rows:
        columns.update(row)
    payload: dict[str, Any] = {
        "table_id": table.table_id,
        "page": table.page,
        "caption": table.caption,
        "headers": list(table.headers),
        "rows": [dict(row) for row in table.rows],
        "footnotes": list(table.footnotes),
        "source_bbox": dict(table.source_bbox) if table.source_bbox is not None else None,
        "row_count": len(table.rows),
        "column_count": len(columns),
        "table_content_digest": "",
    }
    payload["table_content_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "table_content_digest"}
    )
    return OledSupplementaryReviewTable.model_validate(payload)


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


def _stable_hash(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _review_table_digest(table: OledSupplementaryReviewTable) -> str:
    return _stable_hash(table.model_dump(mode="json", exclude={"table_content_digest"}))


def _review_artifact_digest(artifact: OledSupplementaryLocatorReviewArtifact) -> str:
    return _stable_hash(artifact.model_dump(mode="json", exclude={"review_artifact_digest"}))


__all__ = [
    "SUPPLEMENTARY_LOCATOR_MANIFEST_VERSION",
    "SUPPLEMENTARY_LOCATOR_REVIEW_ARTIFACT_VERSION",
    "OledSupplementaryLocatorManifest",
    "OledSupplementaryLocatorMatchStatus",
    "OledSupplementaryLocatorReviewArtifact",
    "OledSupplementaryLocatorReviewDecision",
    "OledSupplementaryLocatorReviewItem",
    "OledSupplementaryLocatorReviewStatus",
    "OledSupplementaryLocatorSource",
    "OledSupplementaryReviewTable",
    "build_oled_supplementary_locator_review_artifact",
    "validate_oled_supplementary_locator_binding",
]
