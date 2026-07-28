from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4s_agent._utils import write_json


ReviewDecision = Literal["accept", "reject", "needs_review"]
ReviewScope = Literal["record", "field", "document", "corpus"]

_SCHEMA_VERSION = "custom_corpus_review.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
_CREDENTIAL_MARKERS = ("token", "secret", "authorization", "password", "bearer", "cookie", "x-api-key")
_PRIVATE_PATH_MARKERS = ("/Users/", "/home/", "C:\\")
_SHORT_SUMMARY_FIELDS = {"extracted_value_summary", "normalized_value_summary"}
_FREE_TEXT_LIMIT = 500
_SUMMARY_TEXT_LIMIT = 300


class CustomCorpusReviewError(ValueError):
    pass


class ReviewedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str
    corpus_id: str
    dry_run_id: str
    document_id: str
    record_id: str
    field_name: str = ""
    review_scope: ReviewScope = "record"
    decision: ReviewDecision
    rejection_reason: str = ""
    reviewer_label: str
    reviewed_at: str
    source_artifact_sha256: str
    extracted_value_summary: str = ""
    normalized_value_summary: str = ""
    confidence_note: str = ""
    provenance_note: str = ""
    notes: str = ""

    @field_validator(
        "review_id",
        "corpus_id",
        "dry_run_id",
        "document_id",
        "record_id",
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
        "rejection_reason",
        "reviewer_label",
        "reviewed_at",
        "confidence_note",
        "provenance_note",
        "notes",
        mode="before",
    )
    @classmethod
    def _clean_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)

    @field_validator("extracted_value_summary", "normalized_value_summary", mode="before")
    @classmethod
    def _clean_summary_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_SUMMARY_TEXT_LIMIT)

    @field_validator("source_artifact_sha256", mode="before")
    @classmethod
    def _clean_source_sha(cls, value: Any) -> str:
        return _normalize_sha256(value, field_name="source_artifact_sha256")

    @field_validator("reviewer_label")
    @classmethod
    def _validate_reviewer_label(cls, value: str) -> str:
        if not value:
            raise ValueError("reviewer_label is required")
        if "@" in value and "redacted" not in value.lower():
            raise ValueError("reviewer_label must be redacted and must not look like a private email address")
        return value

    @model_validator(mode="after")
    def _validate_decision_fields(self) -> "ReviewedRecord":
        if self.decision == "reject" and not self.rejection_reason:
            raise ValueError("decision=reject requires rejection_reason")
        if self.decision == "accept" and self.rejection_reason:
            raise ValueError("decision=accept must not include rejection_reason")
        if self.decision == "needs_review" and not (self.notes or self.confidence_note):
            raise ValueError("decision=needs_review requires notes or confidence_note")
        return self

    def review_target(self) -> tuple[str, str, str, str]:
        return (self.document_id, self.record_id, self.field_name, self.review_scope)


class ReviewManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    review_manifest_id: str
    corpus_id: str
    dry_run_id: str
    created_at: str
    created_by: str
    source_dry_run_report_sha256: str
    source_manifest_sha256: str = ""
    review_policy: str
    review_records: list[ReviewedRecord]

    @field_validator("schema_version", "created_at", "created_by", "review_policy", mode="before")
    @classmethod
    def _clean_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)

    @field_validator("review_manifest_id", "corpus_id", "dry_run_id", mode="before")
    @classmethod
    def _clean_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_FREE_TEXT_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("source_dry_run_report_sha256", "source_manifest_sha256", mode="before")
    @classmethod
    def _clean_sha(cls, value: Any, info: Any) -> str:
        return _normalize_sha256(value, field_name=str(info.field_name))

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError("schema_version must be custom_corpus_review.v1")
        return value

    @field_validator("review_records")
    @classmethod
    def _validate_review_records_present(cls, value: list[ReviewedRecord]) -> list[ReviewedRecord]:
        if not value:
            raise ValueError("review_records must be non-empty")
        duplicate_review_id = _first_duplicate([record.review_id for record in value])
        if duplicate_review_id:
            raise ValueError("duplicate review_id")
        duplicate_target = _first_duplicate([record.review_target() for record in value])
        if duplicate_target:
            raise ValueError("duplicate review target")
        return value

    @model_validator(mode="after")
    def _validate_record_binding(self) -> "ReviewManifest":
        for record in self.review_records:
            if record.corpus_id != self.corpus_id:
                raise ValueError("review record corpus_id must match manifest corpus_id")
            if record.dry_run_id != self.dry_run_id:
                raise ValueError("review record dry_run_id must match manifest dry_run_id")
        return self


def load_review_manifest(path: str | Path) -> ReviewManifest:
    manifest_path = Path(path).expanduser()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusReviewError(f"could not read review manifest: {exc.__class__.__name__}") from exc
    return validate_review_manifest(payload)


def validate_review_manifest(value: Any) -> ReviewManifest:
    try:
        return ReviewManifest.model_validate(value)
    except CustomCorpusReviewError:
        raise
    except Exception as exc:
        raise CustomCorpusReviewError(_safe_error_message(str(exc))) from exc


def review_manifest_summary(manifest: ReviewManifest, path: str | Path | None = None) -> dict[str, Any]:
    review_path = ""
    review_sha = ""
    if path is not None:
        review_path = Path(path).name or "review_manifest.json"
        try:
            review_sha = sha256_file(path)
        except Exception:
            review_sha = ""
    documents = {record.document_id for record in manifest.review_records}
    decisions = [record.decision for record in manifest.review_records]
    return {
        "review_manifest_path": review_path,
        "review_manifest_sha256": review_sha,
        "corpus_id": manifest.corpus_id,
        "dry_run_id": manifest.dry_run_id,
        "review_record_count": len(manifest.review_records),
        "accepted_count": decisions.count("accept"),
        "rejected_count": decisions.count("reject"),
        "needs_review_count": decisions.count("needs_review"),
        "reviewed_document_count": len(documents),
        "source_dry_run_report_sha256": manifest.source_dry_run_report_sha256,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "review_policy": manifest.review_policy,
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
        manifest = load_review_manifest(args.review_manifest)
        summary = review_manifest_summary(manifest, path=args.review_manifest)
        if args.output_summary:
            write_json(Path(args.output_summary).expanduser(), summary)
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 0
    except CustomCorpusReviewError as exc:
        err.write(f"review manifest invalid: {exc}\n")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_review",
        description="Validate a custom corpus human review manifest offline.",
    )
    parser.add_argument("--review-manifest", required=True)
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
    if field_name in {"provenance_note", "notes"} and _contains_url_query(clean):
        raise ValueError(f"{field_name} must not contain URL query strings")
    return clean


def _normalize_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be empty or a SHA-256 digest")
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
    if "duplicate review_id" in lowered:
        return "duplicate review_id"
    if "duplicate review target" in lowered:
        return "duplicate review target"
    if "needs_review" in lowered:
        return "decision=needs_review requires notes or confidence_note"
    if "rejection_reason" in lowered:
        return "decision rejection_reason constraint failed"
    if "reviewer_label" in lowered:
        return "reviewer_label is invalid or must be redacted"
    for field in (
        "schema_version",
        "review_manifest_id",
        "corpus_id",
        "dry_run_id",
        "document_id",
        "record_id",
        "review_id",
        "source_artifact_sha256",
        "source_dry_run_report_sha256",
        "source_manifest_sha256",
    ):
        if field in lowered:
            if "credential-like" in lowered:
                return f"{field} contains forbidden credential-like value"
            if "private path-like" in lowered:
                return f"{field} contains forbidden private path-like value"
            return f"{field} is invalid"
    if "credential-like" in lowered or any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        return "review manifest contains forbidden credential-like value"
    if "private path-like" in lowered or any(marker.lower() in lowered for marker in _PRIVATE_PATH_MARKERS):
        return "review manifest contains forbidden private path-like value"
    if "url query" in lowered:
        return "review manifest contains forbidden URL query string"
    if "too long" in lowered:
        return "review manifest contains overlong text"
    if "review record corpus_id" in lowered:
        return "review record corpus_id must match manifest corpus_id"
    if "review record dry_run_id" in lowered:
        return "review record dry_run_id must match manifest dry_run_id"
    return "review manifest is invalid"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
