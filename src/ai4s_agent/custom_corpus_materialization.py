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


MaterializationMode = Literal["candidate_only"]
MaterializationDecision = Literal["planned", "blocked"]
MaterializationRecordAction = Literal["materialize_candidate", "exclude"]

_SCHEMA_VERSION = "custom_corpus_materialization.v1"
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256_RE = re.compile(r"^(sha256:)?([0-9a-fA-F]{64})$")
_CREDENTIAL_MARKERS = ("token", "secret", "authorization", "password", "bearer", "cookie", "x-api-key")
_PRIVATE_PATH_MARKERS = ("/Users/", "/home/", "C:\\")
_LABEL_LIMIT = 200
_SUMMARY_LIMIT = 500


class CustomCorpusMaterializationError(ValueError):
    pass


class MaterializationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    materialization_record_id: str
    corpus_id: str
    dry_run_id: str
    review_manifest_id: str
    admission_request_id: str
    admission_record_id: str
    review_id: str
    document_id: str
    record_id: str
    field_name: str = ""
    action: MaterializationRecordAction
    admission_action: str
    review_decision: str
    source_artifact_sha256: str
    review_artifact_sha256: str
    admission_request_sha256: str
    package_validation_sha256: str
    normalized_value_summary: str = ""
    provenance_summary: str = ""
    materialization_reason: str = ""
    exclusion_reason: str = ""
    notes: str = ""

    @field_validator(
        "materialization_record_id",
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "admission_request_id",
        "admission_record_id",
        "review_id",
        "document_id",
        "record_id",
        mode="before",
    )
    @classmethod
    def _clean_required_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("field_name", mode="before")
    @classmethod
    def _clean_field_name(cls, value: Any) -> str:
        clean = _clean_text_value(value, field_name="field_name", max_length=_LABEL_LIMIT)
        if clean and not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("field_name must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("admission_action", "review_decision", mode="before")
    @classmethod
    def _clean_safe_label(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)

    @field_validator(
        "normalized_value_summary",
        "provenance_summary",
        "materialization_reason",
        "exclusion_reason",
        "notes",
        mode="before",
    )
    @classmethod
    def _clean_summary(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_SUMMARY_LIMIT)

    @field_validator(
        "source_artifact_sha256",
        "review_artifact_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        mode="before",
    )
    @classmethod
    def _clean_sha(cls, value: Any, info: Any) -> str:
        return _normalize_required_sha256(value, field_name=str(info.field_name))

    @model_validator(mode="after")
    def _validate_action_rules(self) -> "MaterializationRecord":
        if self.action == "materialize_candidate":
            if self.admission_action != "admit":
                raise ValueError("materialize_candidate requires admission_action=admit")
            if self.review_decision != "accept":
                raise ValueError("materialize_candidate requires review_decision=accept")
            if not self.normalized_value_summary:
                raise ValueError("materialize_candidate requires normalized_value_summary")
            if not self.provenance_summary:
                raise ValueError("materialize_candidate requires provenance_summary")
            if not self.materialization_reason:
                raise ValueError("materialize_candidate requires materialization_reason")
        if self.action == "exclude" and not self.exclusion_reason:
            raise ValueError("action=exclude requires exclusion_reason")
        if self.review_decision in {"reject", "needs_review"} and self.action == "materialize_candidate":
            raise ValueError("rejected or needs_review records cannot be materialized")
        if self.admission_action in {"exclude", "needs_review"} and self.action == "materialize_candidate":
            raise ValueError("excluded or needs_review admission records cannot be materialized")
        return self

    def materialization_target(self) -> tuple[str, str, str]:
        return (self.document_id, self.record_id, self.field_name)


class CustomCorpusMaterializationConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confirmed: bool
    confirmed_by: str
    confirmed_at: str
    confirmation_source: str
    manifest_sha256: str
    dry_run_report_sha256: str
    review_manifest_sha256: str
    admission_request_sha256: str
    package_validation_sha256: str
    corpus_id: str
    dry_run_id: str
    review_manifest_id: str
    admission_request_id: str
    reason: str

    @field_validator("confirmed_by", "confirmed_at", "confirmation_source", "reason", mode="before")
    @classmethod
    def _clean_summary_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_SUMMARY_LIMIT)

    @field_validator("corpus_id", "dry_run_id", "review_manifest_id", "admission_request_id", mode="before")
    @classmethod
    def _clean_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator(
        "manifest_sha256",
        "dry_run_report_sha256",
        "review_manifest_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        mode="before",
    )
    @classmethod
    def _clean_sha(cls, value: Any, info: Any) -> str:
        return _normalize_required_sha256(value, field_name=str(info.field_name))

    @field_validator("confirmed_by")
    @classmethod
    def _validate_confirmed_by(cls, value: str) -> str:
        if not value:
            raise ValueError("confirmed_by is required")
        if "@" in value and "redacted" not in value.lower():
            raise ValueError("confirmed_by must be redacted and must not look like a private email address")
        return value


class MaterializationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    materialization_plan_id: str
    materialization_run_id: str
    created_at: str
    created_by: str
    corpus_id: str
    dry_run_id: str
    review_manifest_id: str
    admission_request_id: str
    materialization_mode: MaterializationMode
    materialization_decision: MaterializationDecision
    dataset_target: str
    source_manifest_sha256: str
    source_dry_run_report_sha256: str
    source_review_manifest_sha256: str
    source_admission_request_sha256: str
    source_package_validation_sha256: str
    package_validation_status: str
    package_admission_decision: str
    dry_run_phase1_status: str
    dry_run_dataset_confirmation_confirmed: bool
    dry_run_training_dataset_admitted: bool
    confirmation: CustomCorpusMaterializationConfirmation
    materialization_records: list[MaterializationRecord]
    rollback_policy: str
    redaction_policy: str

    @field_validator("schema_version", "created_at", "rollback_policy", "redaction_policy", mode="before")
    @classmethod
    def _clean_summary_text(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_SUMMARY_LIMIT)

    @field_validator(
        "materialization_plan_id",
        "materialization_run_id",
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "admission_request_id",
        mode="before",
    )
    @classmethod
    def _clean_id(cls, value: Any, info: Any) -> str:
        clean = _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError(f"{info.field_name} must use only letters, numbers, dot, dash, and underscore")
        return clean

    @field_validator("created_by", mode="before")
    @classmethod
    def _clean_created_by(cls, value: Any) -> str:
        return _clean_text_value(value, field_name="created_by", max_length=_SUMMARY_LIMIT)

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
        clean = _clean_text_value(value, field_name="dataset_target", max_length=_LABEL_LIMIT)
        lowered = clean.lower()
        if not _SAFE_ID_RE.fullmatch(clean):
            raise ValueError("dataset_target must be a safe label")
        if "/" in clean or "\\" in clean or lowered.endswith(".csv") or lowered.endswith(".jsonl"):
            raise ValueError("dataset_target must be a safe label and not an output path")
        return clean

    @field_validator("package_validation_status", "package_admission_decision", "dry_run_phase1_status", mode="before")
    @classmethod
    def _clean_status(cls, value: Any, info: Any) -> str:
        return _clean_text_value(value, field_name=str(info.field_name), max_length=_LABEL_LIMIT)

    @field_validator(
        "source_manifest_sha256",
        "source_dry_run_report_sha256",
        "source_review_manifest_sha256",
        "source_admission_request_sha256",
        "source_package_validation_sha256",
        mode="before",
    )
    @classmethod
    def _clean_sha(cls, value: Any, info: Any) -> str:
        return _normalize_required_sha256(value, field_name=str(info.field_name))

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: str) -> str:
        if value != _SCHEMA_VERSION:
            raise ValueError("schema_version must be custom_corpus_materialization.v1")
        return value

    @field_validator("materialization_records")
    @classmethod
    def _validate_records_present(cls, value: list[MaterializationRecord]) -> list[MaterializationRecord]:
        if not value:
            raise ValueError("materialization_records must be non-empty")
        duplicate_record_id = _first_duplicate([record.materialization_record_id for record in value])
        if duplicate_record_id:
            raise ValueError("duplicate materialization_record_id")
        duplicate_target = _first_duplicate([record.materialization_target() for record in value])
        if duplicate_target:
            raise ValueError("duplicate materialization target")
        return value

    @model_validator(mode="after")
    def _validate_plan_boundary(self) -> "MaterializationPlan":
        if self.materialization_decision == "planned":
            if self.package_validation_status != "passed":
                raise ValueError("planned materialization requires package_validation_status=passed")
            if self.package_admission_decision != "eligible":
                raise ValueError("planned materialization requires package_admission_decision=eligible")
            if self.dry_run_phase1_status != "not_run":
                raise ValueError("planned materialization requires dry_run_phase1_status=not_run")
            if self.dry_run_dataset_confirmation_confirmed is not False:
                raise ValueError("planned materialization requires DatasetConfirmation to remain false")
            if self.dry_run_training_dataset_admitted is not False:
                raise ValueError("planned materialization requires dry_run_training_dataset_admitted=false")
            if self.confirmation.confirmed is not True:
                raise ValueError("planned materialization requires explicit confirmation")
            if not self.confirmation.confirmation_source:
                raise ValueError("planned materialization requires confirmation_source")
            if not self.confirmation.reason:
                raise ValueError("planned materialization requires confirmation reason")
        else:
            if self.package_validation_status == "passed" and self.package_admission_decision == "eligible":
                raise ValueError("blocked materialization must not be used for an eligible package")
        self._validate_confirmation_binding()
        self._validate_record_binding()
        return self

    def _validate_confirmation_binding(self) -> None:
        pairs = [
            ("corpus_id", self.confirmation.corpus_id, self.corpus_id),
            ("dry_run_id", self.confirmation.dry_run_id, self.dry_run_id),
            ("review_manifest_id", self.confirmation.review_manifest_id, self.review_manifest_id),
            ("admission_request_id", self.confirmation.admission_request_id, self.admission_request_id),
            ("manifest_sha256", self.confirmation.manifest_sha256, self.source_manifest_sha256),
            ("dry_run_report_sha256", self.confirmation.dry_run_report_sha256, self.source_dry_run_report_sha256),
            ("review_manifest_sha256", self.confirmation.review_manifest_sha256, self.source_review_manifest_sha256),
            ("admission_request_sha256", self.confirmation.admission_request_sha256, self.source_admission_request_sha256),
            ("package_validation_sha256", self.confirmation.package_validation_sha256, self.source_package_validation_sha256),
        ]
        for field_name, observed, expected in pairs:
            if observed != expected:
                raise ValueError(f"confirmation {field_name} must match plan {field_name}")

    def _validate_record_binding(self) -> None:
        for record in self.materialization_records:
            for field_name in ("corpus_id", "dry_run_id", "review_manifest_id", "admission_request_id"):
                if getattr(record, field_name) != getattr(self, field_name):
                    raise ValueError(f"record {field_name} must match plan {field_name}")
            if record.admission_request_sha256 != self.source_admission_request_sha256:
                raise ValueError("record admission_request_sha256 must match plan source_admission_request_sha256")
            if record.package_validation_sha256 != self.source_package_validation_sha256:
                raise ValueError("record package_validation_sha256 must match plan source_package_validation_sha256")


def load_materialization_plan(path: str | Path) -> MaterializationPlan:
    plan_path = Path(path).expanduser()
    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CustomCorpusMaterializationError(f"could not read materialization plan: {exc.__class__.__name__}") from exc
    return validate_materialization_plan(payload)


def validate_materialization_plan(value: Any) -> MaterializationPlan:
    try:
        return MaterializationPlan.model_validate(value)
    except CustomCorpusMaterializationError:
        raise
    except Exception as exc:
        raise CustomCorpusMaterializationError(_safe_error_message(str(exc))) from exc


def materialization_plan_summary(plan: MaterializationPlan, path: str | Path | None = None) -> dict[str, Any]:
    plan_path = ""
    plan_sha = ""
    if path is not None:
        plan_path = Path(path).name or "materialization_plan.json"
        try:
            plan_sha = sha256_file(path)
        except Exception:
            plan_sha = ""
    actions = [record.action for record in plan.materialization_records]
    return {
        "schema_version": plan.schema_version,
        "materialization_plan_path": plan_path,
        "materialization_plan_sha256": plan_sha,
        "materialization_plan_id": plan.materialization_plan_id,
        "materialization_run_id": plan.materialization_run_id,
        "corpus_id": plan.corpus_id,
        "dry_run_id": plan.dry_run_id,
        "review_manifest_id": plan.review_manifest_id,
        "admission_request_id": plan.admission_request_id,
        "materialization_mode": plan.materialization_mode,
        "materialization_decision": plan.materialization_decision,
        "dataset_target": plan.dataset_target,
        "package_validation_status": plan.package_validation_status,
        "package_admission_decision": plan.package_admission_decision,
        "dry_run_phase1_status": plan.dry_run_phase1_status,
        "dry_run_dataset_confirmation_confirmed": plan.dry_run_dataset_confirmation_confirmed,
        "dry_run_training_dataset_admitted": plan.dry_run_training_dataset_admitted,
        "confirmation_present": bool(plan.confirmation.confirmed),
        "confirmation_source": plan.confirmation.confirmation_source,
        "materialization_record_count": len(plan.materialization_records),
        "candidate_record_count": actions.count("materialize_candidate"),
        "excluded_record_count": actions.count("exclude"),
        "source_manifest_sha256": plan.source_manifest_sha256,
        "source_dry_run_report_sha256": plan.source_dry_run_report_sha256,
        "source_review_manifest_sha256": plan.source_review_manifest_sha256,
        "source_admission_request_sha256": plan.source_admission_request_sha256,
        "source_package_validation_sha256": plan.source_package_validation_sha256,
        "rollback_policy": plan.rollback_policy,
        "redaction_policy": plan.redaction_policy,
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
        plan = load_materialization_plan(args.materialization_plan)
        summary = materialization_plan_summary(plan, path=args.materialization_plan)
        if args.output_summary:
            write_json(Path(args.output_summary).expanduser(), summary)
        output.write(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        output.write("\n")
        return 0
    except CustomCorpusMaterializationError as exc:
        err.write(f"materialization plan invalid: {exc}\n")
        return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ai4s_agent.custom_corpus_materialization",
        description="Validate a custom corpus materialization plan offline.",
    )
    parser.add_argument("--materialization-plan", required=True)
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
    if _contains_url_query_or_signature(clean):
        raise ValueError(f"{field_name} must not contain URL query strings or signed URL values")
    return clean


def _normalize_required_sha256(value: Any, *, field_name: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{field_name} is required")
    match = _SHA256_RE.fullmatch(clean)
    if not match:
        raise ValueError(f"{field_name} must be a SHA-256 digest")
    return f"sha256:{match.group(2).lower()}"


def _contains_url_query_or_signature(value: str) -> bool:
    lowered = value.lower()
    if ("http://" in lowered or "https://" in lowered) and "?" in value:
        return True
    return any(marker in lowered for marker in ("x-amz-signature", "signature=", "sig=", "signedurl", "signed-url"))


def _first_duplicate(values: list[Any]) -> Any:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _safe_error_message(message: str) -> str:
    lowered = str(message or "").lower()
    if "duplicate materialization_record_id" in lowered:
        return "duplicate materialization_record_id"
    if "duplicate materialization target" in lowered:
        return "duplicate materialization target"
    if "package_validation_status=passed" in lowered:
        return "package_validation_status is invalid"
    if "package_admission_decision=eligible" in lowered:
        return "package_admission_decision is invalid"
    if "dry_run_phase1_status=not_run" in lowered:
        return "dry_run_phase1_status is invalid"
    if "datasetconfirmation" in lowered:
        return "DatasetConfirmation must remain false"
    if "dry_run_training_dataset_admitted=false" in lowered:
        return "dry_run_training_dataset_admitted is invalid"
    if "explicit confirmation" in lowered:
        return "confirmation is required"
    if "confirmation manifest_sha256" in lowered:
        return "manifest_sha256 is invalid"
    if "confirmation dry_run_report_sha256" in lowered:
        return "dry_run_report_sha256 is invalid"
    if "confirmation review_manifest_sha256" in lowered:
        return "review_manifest_sha256 is invalid"
    if "confirmation admission_request_sha256" in lowered:
        return "admission_request_sha256 is invalid"
    if "confirmation package_validation_sha256" in lowered:
        return "package_validation_sha256 is invalid"
    if "confirmation corpus_id" in lowered:
        return "corpus_id is invalid"
    if "confirmation dry_run_id" in lowered:
        return "dry_run_id is invalid"
    if "confirmation review_manifest_id" in lowered:
        return "review_manifest_id is invalid"
    if "confirmation admission_request_id" in lowered:
        return "admission_request_id is invalid"
    for field in (
        "materialization_mode",
        "materialization_decision",
        "package_validation_status",
        "package_admission_decision",
        "dry_run_phase1_status",
        "dry_run_training_dataset_admitted",
        "dataset_target",
        "created_by",
        "confirmed_by",
        "confirmation",
        "corpus_id",
        "dry_run_id",
        "review_manifest_id",
        "admission_request_id",
        "source_manifest_sha256",
        "source_dry_run_report_sha256",
        "source_review_manifest_sha256",
        "source_admission_request_sha256",
        "source_package_validation_sha256",
        "manifest_sha256",
        "dry_run_report_sha256",
        "review_manifest_sha256",
        "admission_request_sha256",
        "package_validation_sha256",
        "admission_request_sha256",
        "source_artifact_sha256",
        "review_artifact_sha256",
        "normalized_value_summary",
        "provenance_summary",
        "materialization_reason",
        "exclusion_reason",
        "review_decision",
        "admission_action",
        "schema_version",
    ):
        if field.lower() in lowered:
            if "credential-like" in lowered:
                return f"{field} contains forbidden credential-like value"
            if "private path-like" in lowered:
                return f"{field} contains forbidden private path-like value"
            if "url query" in lowered or "signed url" in lowered:
                return f"{field} contains forbidden URL value"
            return f"{field} is invalid"
    if "credential-like" in lowered or any(marker in lowered for marker in _CREDENTIAL_MARKERS):
        return "materialization plan contains forbidden credential-like value"
    if "private path-like" in lowered or any(marker.lower() in lowered for marker in _PRIVATE_PATH_MARKERS):
        return "materialization plan contains forbidden private path-like value"
    if "too long" in lowered:
        return "materialization plan contains overlong text"
    return "materialization plan is invalid"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
