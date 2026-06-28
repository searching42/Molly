from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ai4s_agent._utils import write_json


AdmissionDecision = Literal["eligible", "ineligible", "needs_review"]
AdmissionRecordAction = Literal["admit", "exclude", "needs_review"]
AdmissionScope = Literal["record", "field"]
ReviewDecision = Literal["accept", "reject", "needs_review"]

_SCHEMA_VERSION = "custom_corpus_admission.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
_CREDENTIAL_MARKERS = ("token", "secret", "authorization", "password", "bearer", "cookie", "x-api-key")
_PRIVATE_PATH_MARKERS = ("/Users/", "/home/", "C:\\")
_FREE_TEXT_LIMIT = 500


class CustomCorpusAdmissionError(ValueError):
    pass


class AdmissionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admission_record_id: str
    corpus_id: str
    dry_run_id: str
    review_manifest_id: str
    document_id: str
    record_id: str
    field_name: str = ""
    admission_scope: AdmissionScope = "record"
    review_id: str
    review_decision: ReviewDecision
    action: AdmissionRecordAction
    admission_reason: str = ""
    exclusion_reason: str = ""
    source_artifact_sha256: str
    review_artifact_sha256: str
    provenance_summary: str = ""
    normalized_value_summary: str = ""
    notes: str = ""

    @field_validator(
        "admission_record_id",
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "document_id",
        "record_id",
        "review_id",
        mode="before",
    )
    @classmethod
    def _clean_required_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator(
        "field_name",
        "admission_reason",
        "exclusion_reason",
        "provenance_summary",
        "normalized_value_summary",
        "notes",
        mode="before",
    )
    @classmethod
    def _clean_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)

    @field_validator("source_artifact_sha256", "review_artifact_sha256", mode="before")
    @classmethod
    def _clean_sha(cls, value: Any, info: Any) -> str:
        return _normalize_required_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_action_rules(self) -> "AdmissionRecord":
        if self.review_decision == "reject" and self.action != "exclude":
            raise ValueError("review_decision=reject must use action=exclude")
        if self.review_decision == "needs_review" and self.action not in {"needs_review", "exclude"}:
            raise ValueError("review_decision=needs_review must use action=needs_review or exclude")
        if self.action == "admit" and self.review_decision != "accept":
            raise ValueError("action=admit requires review_decision=accept")
        if self.action == "admit" and not self.admission_reason:
            raise ValueError("action=admit requires admission_reason")
        if self.action == "exclude" and not self.exclusion_reason:
            raise ValueError("action=exclude requires exclusion_reason")
        if self.action == "needs_review" and not self.notes:
            raise ValueError("action=needs_review requires notes")
        return self

    def admission_target(self) -> tuple[str, str, str, str]:
        return (self.document_id, self.record_id, self.field_name, self.admission_scope)


class AdmissionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    admission_request_id: str
    corpus_id: str
    dry_run_id: str
    created_at: str
    created_by: str
    source_manifest_sha256: str
    source_dry_run_report_sha256: str
    source_review_manifest_sha256: str
    review_manifest_id: str
    admission_policy: str
    dataset_target: str
    admission_records: list[AdmissionRecord]

    @field_validator("schema_version", "created_at", "admission_policy", mode="before")
    @classmethod
    def _clean_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)

    @field_validator("admission_request_id", "corpus_id", "dry_run_id", "review_manifest_id", mode="before")
    @classmethod
    def _clean_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("created_by", mode="before")
    @classmethod
    def _clean_created_by(cls, value: Any) -> str:
        return _clean_text_value(value, field_name="created_by", max_length=_FREE_TEXT_LIMIT)

    @field_validator("created_by")
    @classmethod
    def _validate_created_by(cls, value: str) -> str:
        if not value:
            raise ValueError("created_by is required")
        if "@" in value and "redacted" not in value.lower():
            raise ValueError("created_by must be redacted and must not look like a private email address")
        return value

    @field_validator("dataset_target", mode="before")
    @classmethod
    def _clean_dataset_target(cls, value: Any) -> str:
        clean = _clean_text_value(value, field_name="dataset_target", max_length=_FREE_TEXT_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("dataset_target must be a safe label")
        return clean

    @field_validator("source_manifest_sha256", "source_dry_run_report_sha256", "source_review_manifest_sha256", mode="before")
    @classmethod
    def _clean_sha(cls, value: Any, info: Any) -> str:
        return _normalize_required_sha256(value, field_name=str(info.field_name))

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError("schema_version must be custom_corpus_admission.v1")
        return value

    @field_validator("admission_records")
    @classmethod
    def _validate_records_present(cls, value: list[AdmissionRecord]) -> list[AdmissionRecord]:
        if not value:
            raise ValueError("admission_records must be non-empty")
        duplicate_record_id = _first_duplicate([record.admission_record_id for record in value])
        if duplicate_record_id:
            raise ValueError("duplicate admission_record_id")
        duplicate_target = _first_duplicate([record.admission_target() for record in value])
        if duplicate_target:
            raise ValueError("duplicate admission target")
        return value

    @model_validator(mode="after")
    def _validate_record_binding(self) -> "AdmissionRequest":
        for record in self.admission_records:
            if record.corpus_id != self.corpus_id:
                raise ValueError("admission record corpus_id must match request corpus_id")
            if record.dry_run_id != self.dry_run_id:
                raise ValueError("admission record dry_run_id must match request dry_run_id")
            if record.review_manifest_id != self.review_manifest_id:
                raise ValueError("admission record review_manifest_id must match request review_manifest_id")
        return self


def load_admission_request(path: str | Path) -> AdmissionRequest:
    request_path = Path(path).expanduser()
    try:
        payload = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusAdmissionError(f"could not read admission request: {exc.__class__.__name__}") from exc
    return validate_admission_request(payload)


def validate_admission_request(value: Any) -> AdmissionRequest:
    try:
        return AdmissionRequest.model_validate(value)
    except CustomCorpusAdmissionError:
        raise
    except Exception as exc:
        raise CustomCorpusAdmissionError(_safe_error_message(str(exc))) from exc


def admission_validation_summary(request: AdmissionRequest, path: str | Path | None = None) -> dict[str, Any]:
    request_path = ""
    request_sha = ""
    if path is not None:
        request_path = Path(path).name or "admission_request.json"
        try:
            request_sha = sha256_file(path)
        except Exception:
            request_sha = ""
    actions = [record.action for record in request.admission_records]
    admit_count = actions.count("admit")
    needs_review_count = actions.count("needs_review")
    blocking_reasons: list[str] = []
    if needs_review_count:
        decision: AdmissionDecision = "needs_review"
        blocking_reasons.append("records_need_review")
    elif admit_count:
        decision = "eligible"
    else:
        decision = "ineligible"
        blocking_reasons.append("no_records_admitted")
    return {
        "admission_request_path": request_path,
        "admission_request_sha256": request_sha,
        "admission_request_id": request.admission_request_id,
        "corpus_id": request.corpus_id,
        "dry_run_id": request.dry_run_id,
        "review_manifest_id": request.review_manifest_id,
        "dataset_target": request.dataset_target,
        "admission_policy": request.admission_policy,
        "admission_record_count": len(request.admission_records),
        "admit_count": admit_count,
        "exclude_count": actions.count("exclude"),
        "needs_review_count": needs_review_count,
        "eligible_record_count": admit_count,
        "source_manifest_sha256": request.source_manifest_sha256,
        "source_dry_run_report_sha256": request.source_dry_run_report_sha256,
        "source_review_manifest_sha256": request.source_review_manifest_sha256,
        "decision": decision,
        "blocking_reasons": blocking_reasons,
        "warnings": [],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main(
    argv: list[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    output = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = _parser()
    if stderr is None:
        args = parser.parse_args(argv)
    else:
        with redirect_stderr(stderr):
            args = parser.parse_args(argv)
    try:
        request = load_admission_request(args.admission_request)
        summary = admission_validation_summary(request, path=args.admission_request)
        if args.output_summary:
            write_json(Path(args.output_summary).expanduser(), summary)
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 0
    except CustomCorpusAdmissionError as exc:
        err.write(f"admission request invalid: {exc}\n")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_admission",
        description="Validate a custom corpus dataset admission request offline.",
    )
    parser.add_argument("--admission-request", required=True)
    parser.add_argument("--output-summary", default="")
    return parser


def _clean_text_value(value: Any, *, field_name: str, max_length: int) -> str:
    clean = str(value or "").strip()
    lowered = clean.lower()
    if len(clean) > max_length:
        raise ValueError(f"{field_name} is too long")
    if any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        raise ValueError(f"{field_name} contains forbidden credential-like value")
    if any(marker.lower() in lowered for marker in _PRIVATE_PATH_MARKERS):
        raise ValueError(f"{field_name} contains forbidden private path-like value")
    if _contains_url_query(clean):
        raise ValueError(f"{field_name} must not contain URL query strings")
    return clean


def _normalize_required_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is required")
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(2).lower()}"


def _contains_url_query(value: str) -> bool:
    lowered = value.lower()
    return ("http://" in lowered or "https://" in lowered) and "?" in value


def _first_duplicate(values: list[Any]) -> Any:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _safe_error_message(message: str) -> str:
    clean = str(message or "").strip()
    lowered = clean.lower()
    if "duplicate admission_record_id" in lowered:
        return "duplicate admission_record_id"
    if "duplicate admission target" in lowered:
        return "duplicate admission target"
    if "review_decision" in lowered:
        return "review_decision and action are incompatible"
    if "admission_reason" in lowered:
        return "action=admit requires admission_reason"
    if "exclusion_reason" in lowered:
        return "action=exclude requires exclusion_reason"
    if "needs_review" in lowered:
        return "action=needs_review requires notes"
    if "dataset_target" in lowered:
        if "private path-like" in lowered:
            return "dataset_target contains forbidden private path-like value"
        return "dataset_target is invalid"
    if "created_by" in lowered:
        return "created_by is invalid or must be redacted"
    for field in (
        "schema_version",
        "admission_request_id",
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "admission_record_id",
        "document_id",
        "record_id",
        "review_id",
        "source_manifest_sha256",
        "source_dry_run_report_sha256",
        "source_review_manifest_sha256",
        "source_artifact_sha256",
        "review_artifact_sha256",
    ):
        if field in lowered:
            if "credential-like" in lowered:
                return f"{field} contains forbidden credential-like value"
            if "private path-like" in lowered:
                return f"{field} contains forbidden private path-like value"
            return f"{field} is invalid"
    if "credential-like" in lowered or any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        return "admission request contains forbidden credential-like value"
    if "private path-like" in lowered or any(marker.lower() in lowered for marker in _PRIVATE_PATH_MARKERS):
        return "admission request contains forbidden private path-like value"
    if "url query" in lowered:
        return "admission request contains forbidden URL query string"
    if "too long" in lowered:
        return "admission request contains overlong text"
    return "admission request is invalid"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
