from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent.domains.oled_supplementary_evidence_recovery import OledSupplementaryTargetKind
from ai4s_agent.domains.oled_supplementary_parser_preflight import OledSupplementaryParserPreflightPlan


SUPPLEMENTARY_MINERU_EXECUTION_MANIFEST_VERSION = "oled_supplementary_mineru_execution_manifest.v1"
SUPPLEMENTARY_MINERU_EXECUTION_ARTIFACT_VERSION = "oled_supplementary_mineru_execution.v1"

_SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SAFE_BOUND_ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
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


class OledSupplementaryMineruExecutionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class OledSupplementaryMineruOutputKind(str, Enum):
    PARSED_DOCUMENT_JSON = "parsed_document_json"
    PARSED_DOCUMENT_MARKDOWN = "parsed_document_markdown"
    PARSER_AUDIT_JSON = "parser_audit_json"
    CONTENT_LIST_JSON = "content_list_json"
    CONTENT_LIST_V2_JSON = "content_list_v2_json"
    MIDDLE_JSON = "middle_json"


class OledSupplementaryMineruExecutionSource(BaseModel):
    """Operator-local rebinding of an approved source to the PDF used for execution."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    local_pdf_path: str

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("local_pdf_path")
    @classmethod
    def validate_local_pdf_path(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("local_pdf_path is required")
        return clean


class OledSupplementaryMineruExecutionManifest(BaseModel):
    """Human confirmation bound to one parser preflight and endpoint preflight report."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SUPPLEMENTARY_MINERU_EXECUTION_MANIFEST_VERSION
    run_id: str
    paper_id: str
    preflight_plan_digest: str
    execution_confirmed: bool = False
    reviewed_by: str
    reviewed_at: str
    endpoint_profile_name: str
    endpoint_preflight_sha256: str
    sources: list[OledSupplementaryMineruExecutionSource] = Field(default_factory=list)

    @field_validator("schema_version")
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MINERU_EXECUTION_MANIFEST_VERSION:
            raise ValueError("unexpected supplementary MinerU execution manifest schema_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("endpoint_profile_name")
    @classmethod
    def validate_endpoint_profile_name(cls, value: str) -> str:
        clean = _validate_path_segment(value, field_name="endpoint_profile_name")
        return _validate_non_sensitive_text(clean, field_name="endpoint_profile_name")

    @field_validator("preflight_plan_digest", "endpoint_preflight_sha256")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

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

    @model_validator(mode="after")
    def validate_manifest_shape(self) -> OledSupplementaryMineruExecutionManifest:
        if not self.execution_confirmed:
            raise ValueError("supplementary MinerU execution requires execution_confirmed=true")
        if not self.sources:
            raise ValueError("supplementary MinerU execution requires local source rebindings")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("duplicate supplementary MinerU execution source_id")
        return self


class OledSupplementaryMineruExecutionTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recovery_item_id: str
    target_kind: OledSupplementaryTargetKind
    target_locator: str

    @field_validator("recovery_item_id")
    @classmethod
    def validate_recovery_item_id(cls, value: str) -> str:
        return _validate_bound_id(value, field_name="recovery_item_id")

    @field_validator("target_locator")
    @classmethod
    def validate_target_locator(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("target_locator is required")
        if "/" in clean or "\\" in clean:
            raise ValueError("target_locator must not contain a path")
        return clean

    @model_validator(mode="after")
    def validate_target_kind(self) -> OledSupplementaryMineruExecutionTarget:
        if self.target_kind not in {
            OledSupplementaryTargetKind.TABLE,
            OledSupplementaryTargetKind.FIGURE,
        }:
            raise ValueError("supplementary MinerU execution target must be a table or figure")
        return self


class OledSupplementaryMineruOutputHash(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_kind: OledSupplementaryMineruOutputKind
    sha256: str
    byte_size: int = Field(gt=0)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="sha256")


class OledSupplementaryMineruSourceExecutionResult(BaseModel):
    """Redacted outcome for one approved source; raw parser content stays outside this artifact."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_pdf_sha256: str
    byte_size: int = Field(gt=0)
    page_count: int = Field(ge=1)
    targets: list[OledSupplementaryMineruExecutionTarget] = Field(default_factory=list)
    status: OledSupplementaryMineruExecutionStatus
    mineru_called: bool = False
    provider: str = ""
    parser_backend: str = ""
    mineru_version: str = ""
    protocol_version: str = ""
    output_hashes: list[OledSupplementaryMineruOutputHash] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    error_code: str = ""
    error_message: str = ""
    full_source_parse: bool = True
    locator_resolved: bool = False
    candidate_regenerated: bool = False

    @field_validator("source_id")
    @classmethod
    def validate_source_id(cls, value: str) -> str:
        return _validate_path_segment(value, field_name="source_id")

    @field_validator("source_pdf_sha256")
    @classmethod
    def validate_source_pdf_sha256(cls, value: str) -> str:
        return _normalize_sha256(value, field_name="source_pdf_sha256")

    @field_validator("provider", "parser_backend", "mineru_version", "protocol_version", mode="before")
    @classmethod
    def normalize_parser_text(cls, value: Any, info: Any) -> str:
        return _validate_non_sensitive_text(value, field_name=str(info.field_name), allow_empty=True)

    @field_validator("warning_codes")
    @classmethod
    def validate_warning_codes(cls, value: list[str]) -> list[str]:
        clean = [_validate_bound_id(item, field_name="warning_code") for item in value]
        if clean != sorted(clean) or len(clean) != len(set(clean)):
            raise ValueError("warning_codes must be sorted and unique")
        return clean

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            return ""
        clean = _validate_bound_id(clean, field_name="error_code")
        if any(marker in clean.lower() for marker in _CREDENTIAL_MARKERS):
            raise ValueError("error_code contains forbidden credential-like text")
        return clean

    @field_validator("error_message")
    @classmethod
    def validate_error_message(cls, value: str) -> str:
        clean = str(value or "").strip()
        if len(clean) > 500:
            raise ValueError("error_message is too long")
        lowered = clean.lower()
        if any(marker in lowered for marker in _CREDENTIAL_MARKERS) or "/" in clean or "\\" in clean:
            raise ValueError("error_message contains sensitive path or credential-like text")
        return clean

    @model_validator(mode="after")
    def validate_result_shape(self) -> OledSupplementaryMineruSourceExecutionResult:
        if not self.targets:
            raise ValueError("supplementary MinerU source result requires at least one target")
        target_ids = [target.recovery_item_id for target in self.targets]
        if target_ids != sorted(target_ids) or len(target_ids) != len(set(target_ids)):
            raise ValueError("supplementary MinerU targets must be sorted and unique")
        output_kinds = [item.output_kind.value for item in self.output_hashes]
        if output_kinds != sorted(output_kinds) or len(output_kinds) != len(set(output_kinds)):
            raise ValueError("supplementary MinerU output hashes must be sorted and unique")
        if not self.full_source_parse or self.locator_resolved or self.candidate_regenerated:
            raise ValueError("supplementary MinerU execution must remain full-source and pre-locator")
        if self.status == OledSupplementaryMineruExecutionStatus.SUCCESS:
            required_kinds = {
                OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_JSON,
                OledSupplementaryMineruOutputKind.PARSED_DOCUMENT_MARKDOWN,
                OledSupplementaryMineruOutputKind.PARSER_AUDIT_JSON,
            }
            if not self.mineru_called or self.provider != "mineru_api":
                raise ValueError("successful supplementary execution requires an explicit MinerU call")
            if not self.parser_backend.startswith("mineru_api:"):
                raise ValueError("successful supplementary execution requires a MinerU API backend")
            if not self.mineru_version or not self.protocol_version:
                raise ValueError("successful supplementary execution requires parser version evidence")
            if not required_kinds.issubset({item.output_kind for item in self.output_hashes}):
                raise ValueError("successful supplementary execution is missing required output hashes")
            if self.error_code or self.error_message:
                raise ValueError("successful supplementary execution must not record an error")
        elif self.status == OledSupplementaryMineruExecutionStatus.FAILED:
            if not self.error_code or not self.error_message:
                raise ValueError("failed supplementary execution requires a redacted error")
        else:
            if self.mineru_called or self.provider or self.parser_backend or self.output_hashes:
                raise ValueError("skipped supplementary execution must not record parser activity")
            if self.error_code != "skipped_after_prior_failure":
                raise ValueError("skipped supplementary execution requires the expected skip reason")
        return self


class OledSupplementaryMineruExecutionArtifact(BaseModel):
    """Content-bound audit artifact for an explicitly confirmed MinerU execution."""

    model_config = ConfigDict(extra="forbid")

    artifact_version: str = SUPPLEMENTARY_MINERU_EXECUTION_ARTIFACT_VERSION
    run_id: str
    paper_id: str
    generated_at: str
    preflight_plan_digest: str
    endpoint_profile_name: str
    endpoint_preflight_sha256: str
    redacted_api_origin: str
    backend: str
    effort: str
    parse_method: str
    formula_enabled: bool = True
    table_enabled: bool = True
    image_analysis_enabled: bool = False
    full_source_parse: bool = True
    status: OledSupplementaryMineruExecutionStatus
    source_results: list[OledSupplementaryMineruSourceExecutionResult] = Field(default_factory=list)
    source_count: int = Field(ge=1)
    successful_source_count: int = Field(ge=0)
    failed_source_count: int = Field(ge=0)
    skipped_source_count: int = Field(ge=0)
    execution_artifact_digest: str
    audit_only: bool = True
    mineru_called: bool
    network_accessed: bool
    external_service_called: bool
    pdf_content_parsed: bool
    locator_resolved: bool = False
    candidate_regenerated: bool = False
    automatic_candidate_merge: bool = False
    reviewed_evidence_staging: bool = False
    device_only_admitted: bool = False
    gold_records_created: bool = False
    dataset_written: bool = False

    @field_validator("artifact_version")
    @classmethod
    def validate_artifact_version(cls, value: str) -> str:
        if value != SUPPLEMENTARY_MINERU_EXECUTION_ARTIFACT_VERSION:
            raise ValueError("unexpected supplementary MinerU execution artifact_version")
        return value

    @field_validator("run_id", "paper_id")
    @classmethod
    def validate_safe_segments(cls, value: str, info: Any) -> str:
        return _validate_path_segment(value, field_name=str(info.field_name))

    @field_validator("endpoint_profile_name")
    @classmethod
    def validate_endpoint_profile_name(cls, value: str) -> str:
        clean = _validate_path_segment(value, field_name="endpoint_profile_name")
        return _validate_non_sensitive_text(clean, field_name="endpoint_profile_name")

    @field_validator("preflight_plan_digest", "endpoint_preflight_sha256", "execution_artifact_digest")
    @classmethod
    def validate_digests(cls, value: str, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("redacted_api_origin")
    @classmethod
    def validate_redacted_api_origin(cls, value: str) -> str:
        parsed = urlparse(str(value or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("redacted_api_origin must be an HTTP(S) origin")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("redacted_api_origin must not contain credentials, query, or fragment")
        if parsed.path not in {"", "/"}:
            raise ValueError("redacted_api_origin must not contain a path")
        return f"{parsed.scheme}://{parsed.netloc}"

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

    @field_validator("backend", "effort", "parse_method")
    @classmethod
    def validate_parse_settings(cls, value: str, info: Any) -> str:
        return _validate_non_sensitive_text(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def validate_artifact_integrity(self) -> OledSupplementaryMineruExecutionArtifact:
        if not self.source_results or self.source_count != len(self.source_results):
            raise ValueError("supplementary MinerU source_count does not match source results")
        source_ids = [item.source_id for item in self.source_results]
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise ValueError("supplementary MinerU source results must be sorted and unique")
        counts = {
            status: sum(1 for item in self.source_results if item.status == status)
            for status in OledSupplementaryMineruExecutionStatus
        }
        if self.successful_source_count != counts[OledSupplementaryMineruExecutionStatus.SUCCESS]:
            raise ValueError("supplementary MinerU successful_source_count mismatch")
        if self.failed_source_count != counts[OledSupplementaryMineruExecutionStatus.FAILED]:
            raise ValueError("supplementary MinerU failed_source_count mismatch")
        if self.skipped_source_count != counts[OledSupplementaryMineruExecutionStatus.SKIPPED]:
            raise ValueError("supplementary MinerU skipped_source_count mismatch")
        expected_status = (
            OledSupplementaryMineruExecutionStatus.SUCCESS
            if self.successful_source_count == self.source_count
            else OledSupplementaryMineruExecutionStatus.FAILED
        )
        if self.status != expected_status:
            raise ValueError("supplementary MinerU artifact status does not match source results")
        any_called = any(item.mineru_called for item in self.source_results)
        any_parsed = any(item.status == OledSupplementaryMineruExecutionStatus.SUCCESS for item in self.source_results)
        if (
            self.mineru_called != any_called
            or self.network_accessed != any_called
            or self.external_service_called != any_called
        ):
            raise ValueError("supplementary MinerU execution-side-effect flags do not match source results")
        if self.pdf_content_parsed != any_parsed:
            raise ValueError("supplementary MinerU parsed-content flag does not match source results")
        if not self.audit_only or not self.full_source_parse or not self.formula_enabled or not self.table_enabled:
            raise ValueError("supplementary MinerU execution must retain its fixed audit and parse settings")
        fixed_false_flags = (
            "image_analysis_enabled",
            "locator_resolved",
            "candidate_regenerated",
            "automatic_candidate_merge",
            "reviewed_evidence_staging",
            "device_only_admitted",
            "gold_records_created",
            "dataset_written",
        )
        if any(getattr(self, field_name) for field_name in fixed_false_flags):
            raise ValueError("supplementary MinerU execution crossed a downstream admission boundary")
        if _execution_artifact_digest(self) != self.execution_artifact_digest:
            raise ValueError("supplementary MinerU execution artifact digest does not match canonical content")
        return self


def validate_oled_supplementary_mineru_execution_binding(
    preflight_plan: OledSupplementaryParserPreflightPlan,
    manifest: OledSupplementaryMineruExecutionManifest,
) -> None:
    """Reject any execution manifest that does not exactly cover the approved preflight sources."""

    preflight_plan = OledSupplementaryParserPreflightPlan.model_validate(
        preflight_plan.model_dump(mode="json")
    )
    manifest = OledSupplementaryMineruExecutionManifest.model_validate(manifest.model_dump(mode="json"))
    if manifest.paper_id != preflight_plan.paper_id:
        raise ValueError("supplementary MinerU execution paper_id does not match parser preflight")
    if manifest.preflight_plan_digest != preflight_plan.preflight_plan_digest:
        raise ValueError("supplementary MinerU execution preflight_plan_digest does not match parser preflight")
    expected_source_ids = {source.source_id for source in preflight_plan.source_envelopes}
    manifest_source_ids = {source.source_id for source in manifest.sources}
    if manifest_source_ids != expected_source_ids:
        raise ValueError("supplementary MinerU execution sources must exactly cover parser preflight sources")


def build_oled_supplementary_mineru_execution_artifact(
    *,
    manifest: OledSupplementaryMineruExecutionManifest,
    generated_at: str,
    redacted_api_origin: str,
    backend: str,
    effort: str,
    parse_method: str,
    source_results: list[OledSupplementaryMineruSourceExecutionResult],
) -> OledSupplementaryMineruExecutionArtifact:
    manifest = OledSupplementaryMineruExecutionManifest.model_validate(manifest.model_dump(mode="json"))
    results = [
        OledSupplementaryMineruSourceExecutionResult.model_validate(item.model_dump(mode="json"))
        for item in source_results
    ]
    results.sort(key=lambda item: item.source_id)
    successful = sum(1 for item in results if item.status == OledSupplementaryMineruExecutionStatus.SUCCESS)
    failed = sum(1 for item in results if item.status == OledSupplementaryMineruExecutionStatus.FAILED)
    skipped = sum(1 for item in results if item.status == OledSupplementaryMineruExecutionStatus.SKIPPED)
    any_called = any(item.mineru_called for item in results)
    any_parsed = successful > 0
    payload: dict[str, Any] = {
        "artifact_version": SUPPLEMENTARY_MINERU_EXECUTION_ARTIFACT_VERSION,
        "run_id": manifest.run_id,
        "paper_id": manifest.paper_id,
        "generated_at": str(generated_at or "").strip(),
        "preflight_plan_digest": manifest.preflight_plan_digest,
        "endpoint_profile_name": manifest.endpoint_profile_name,
        "endpoint_preflight_sha256": manifest.endpoint_preflight_sha256,
        "redacted_api_origin": redacted_api_origin,
        "backend": str(backend or "").strip(),
        "effort": str(effort or "").strip(),
        "parse_method": str(parse_method or "").strip(),
        "formula_enabled": True,
        "table_enabled": True,
        "image_analysis_enabled": False,
        "full_source_parse": True,
        "status": "success" if successful == len(results) else "failed",
        "source_results": [item.model_dump(mode="json") for item in results],
        "source_count": len(results),
        "successful_source_count": successful,
        "failed_source_count": failed,
        "skipped_source_count": skipped,
        "execution_artifact_digest": "",
        "audit_only": True,
        "mineru_called": any_called,
        "network_accessed": any_called,
        "external_service_called": any_called,
        "pdf_content_parsed": any_parsed,
        "locator_resolved": False,
        "candidate_regenerated": False,
        "automatic_candidate_merge": False,
        "reviewed_evidence_staging": False,
        "device_only_admitted": False,
        "gold_records_created": False,
        "dataset_written": False,
    }
    payload["execution_artifact_digest"] = _stable_hash(
        {key: value for key, value in payload.items() if key != "execution_artifact_digest"}
    )
    return OledSupplementaryMineruExecutionArtifact.model_validate(payload)


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


def _validate_non_sensitive_text(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    clean = str(value or "").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError(f"{field_name} is required")
    if len(clean) > 200:
        raise ValueError(f"{field_name} is too long")
    lowered = clean.lower()
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        raise ValueError(f"{field_name} contains forbidden credential-like text")
    if "://" in clean or "/" in clean or "\\" in clean:
        raise ValueError(f"{field_name} must not contain a path or URL")
    return clean


def _stable_hash(value: dict[str, Any]) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _execution_artifact_digest(artifact: OledSupplementaryMineruExecutionArtifact) -> str:
    return _stable_hash(artifact.model_dump(mode="json", exclude={"execution_artifact_digest"}))


__all__ = [
    "SUPPLEMENTARY_MINERU_EXECUTION_ARTIFACT_VERSION",
    "SUPPLEMENTARY_MINERU_EXECUTION_MANIFEST_VERSION",
    "OledSupplementaryMineruExecutionArtifact",
    "OledSupplementaryMineruExecutionManifest",
    "OledSupplementaryMineruExecutionSource",
    "OledSupplementaryMineruExecutionStatus",
    "OledSupplementaryMineruExecutionTarget",
    "OledSupplementaryMineruOutputHash",
    "OledSupplementaryMineruOutputKind",
    "OledSupplementaryMineruSourceExecutionResult",
    "build_oled_supplementary_mineru_execution_artifact",
    "validate_oled_supplementary_mineru_execution_binding",
]
