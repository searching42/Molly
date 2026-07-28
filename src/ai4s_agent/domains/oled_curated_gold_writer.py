from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

from ai4s_agent.domains.oled_gold_validation import (
    OledGoldDatasetRecord,
    validate_oled_gold_dataset,
)
from ai4s_agent.domains.oled_mineru_acceptance_harness import redact_oled_mineru_acceptance_path
from ai4s_agent.domains.oled_reviewed_gold_candidates import (
    OledReviewedGoldCandidate,
    OledReviewedGoldCandidateStatus,
)


class OledCuratedGoldWriterPolicy(BaseModel):
    require_confirmation: bool = True
    require_converted_status: bool = True
    allow_converted_with_warnings: bool = False
    allow_validation_warnings: bool = False
    require_no_validation_errors: bool = True
    require_evidence_refs: bool = True
    require_reviewer: bool = False
    require_candidate_only_source: bool = True
    write_training_data: bool = False
    run_dataset_views: bool = False
    run_model_backends: bool = False


class OledCuratedGoldWriteStatus(str, Enum):
    WRITTEN = "written"
    SKIPPED = "skipped"
    REJECTED = "rejected"


class OledCuratedGoldWriteResult(BaseModel):
    gold_candidate_id: str
    gold_record_id: str | None = None
    source_reviewed_candidate_id: str | None = None
    source_packet_id: str | None = None

    status: OledCuratedGoldWriteStatus
    reason_codes: list[str] = Field(default_factory=list)

    validation_error_codes: list[str] = Field(default_factory=list)
    validation_warning_codes: list[str] = Field(default_factory=list)

    source_evidence_anchors: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "reason_codes",
        "validation_error_codes",
        "validation_warning_codes",
        "source_evidence_anchors",
    )
    @classmethod
    def validate_sorted_unique_strings(cls, value: list[str]) -> list[str]:
        return sorted({str(item).strip() for item in value if str(item).strip()})


class OledCuratedGoldManifest(BaseModel):
    manifest_id: str
    output_record_count: int
    input_candidate_count: int

    output_jsonl_path: str | None = None
    output_sha256: str | None = None

    status_counts: dict[str, int] = Field(default_factory=dict)
    reason_code_counts: dict[str, int] = Field(default_factory=dict)

    written_record_ids: list[str] = Field(default_factory=list)
    write_results: list[OledCuratedGoldWriteResult] = Field(default_factory=list)

    policy: OledCuratedGoldWriterPolicy
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return all(result.status == OledCuratedGoldWriteStatus.WRITTEN for result in self.write_results)


class OledCuratedGoldWriterFinding(BaseModel):
    code: str
    severity: Literal["error", "warning"] = "warning"
    message: str
    gold_candidate_id: str | None = None
    gold_record_id: str | None = None


class OledCuratedGoldWriterReport(BaseModel):
    manifest: OledCuratedGoldManifest
    records: list[OledGoldDatasetRecord] = Field(default_factory=list)
    findings: list[OledCuratedGoldWriterFinding] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    @property
    def error_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "error"]

    @property
    def warning_codes(self) -> list[str]:
        return [finding.code for finding in self.findings if finding.severity == "warning"]


def load_oled_reviewed_gold_candidates_jsonl(
    path: str | Path,
) -> list[OledReviewedGoldCandidate]:
    candidate_path = Path(path)
    _reject_forbidden_input(candidate_path)
    if not candidate_path.exists():
        raise ValueError(f"missing_gold_candidate_jsonl:{redact_oled_mineru_acceptance_path(candidate_path)}")
    candidates: list[OledReviewedGoldCandidate] = []
    for line_number, raw_line in enumerate(candidate_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            candidate = OledReviewedGoldCandidate.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"invalid_gold_candidate_jsonl:line-{line_number}") from exc
        if _contains_absolute_path(candidate.metadata):
            raise ValueError(f"absolute_path_in_gold_candidate_metadata:{candidate.candidate_id}")
        candidates.append(candidate)
    return candidates


def select_oled_curated_gold_records(
    candidates: Iterable[OledReviewedGoldCandidate],
    *,
    policy: OledCuratedGoldWriterPolicy | None = None,
    confirm_curated_gold_write: bool = False,
) -> OledCuratedGoldWriterReport:
    writer_policy = policy or OledCuratedGoldWriterPolicy()
    if writer_policy.require_confirmation and not confirm_curated_gold_write:
        raise ValueError("confirmation_required:curated_gold_write")

    source_candidates = sorted(list(candidates), key=lambda item: item.candidate_id)
    results: list[OledCuratedGoldWriteResult] = []
    findings: list[OledCuratedGoldWriterFinding] = []
    selected_records_by_candidate_id: dict[str, OledGoldDatasetRecord] = {}

    for candidate in source_candidates:
        reason_codes = _policy_rejection_reasons(candidate, writer_policy)
        if reason_codes:
            result = _write_result(
                candidate,
                status=OledCuratedGoldWriteStatus.REJECTED,
                reason_codes=reason_codes,
            )
            results.append(result)
            findings.extend(_findings_for_reasons(reason_codes, candidate, result))
            continue

        assert candidate.gold_record is not None
        result = _write_result(
            candidate,
            status=OledCuratedGoldWriteStatus.WRITTEN,
            reason_codes=["selected_for_write"],
        )
        results.append(result)
        selected_records_by_candidate_id[candidate.candidate_id] = candidate.gold_record

    results, selected_records, validation_findings = _apply_post_selection_validation(
        results,
        source_candidates,
        selected_records_by_candidate_id,
        writer_policy,
    )
    findings.extend(validation_findings)
    manifest = _manifest(
        candidates=source_candidates,
        policy=writer_policy,
        results=results,
        records=selected_records,
    )
    return OledCuratedGoldWriterReport(
        manifest=manifest,
        records=selected_records,
        findings=findings,
    )


def write_oled_curated_gold_records_jsonl(
    records: Iterable[OledGoldDatasetRecord],
    path: str | Path,
) -> str:
    lines = [
        json.dumps(
            _sanitize_for_output(record.model_dump(mode="json", exclude_none=True)),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in sorted(records, key=lambda item: item.record_id)
    ]
    payload = "\n".join(lines) + ("\n" if lines else "")
    encoded = payload.encode("utf-8")
    Path(path).write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_oled_curated_gold_manifest_json(
    manifest: OledCuratedGoldManifest,
    path: str | Path,
) -> None:
    payload = _sanitize_for_output(manifest.model_dump(mode="json", exclude_none=True))
    Path(path).write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run_oled_curated_gold_writer(
    candidates: Iterable[OledReviewedGoldCandidate],
    *,
    output_jsonl_path: str | Path | None = None,
    output_manifest_path: str | Path | None = None,
    policy: OledCuratedGoldWriterPolicy | None = None,
    confirm_curated_gold_write: bool = False,
) -> OledCuratedGoldWriterReport:
    report = select_oled_curated_gold_records(
        candidates,
        policy=policy,
        confirm_curated_gold_write=confirm_curated_gold_write,
    )
    manifest_updates: dict[str, Any] = {}
    if report.is_valid and output_jsonl_path is not None:
        output_hash = write_oled_curated_gold_records_jsonl(report.records, output_jsonl_path)
        manifest_updates["output_jsonl_path"] = redact_oled_mineru_acceptance_path(output_jsonl_path)
        manifest_updates["output_sha256"] = output_hash
    if manifest_updates:
        report = report.model_copy(update={"manifest": report.manifest.model_copy(update=manifest_updates)})
    if output_manifest_path is not None:
        write_oled_curated_gold_manifest_json(report.manifest, output_manifest_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write curated OLED gold records under explicit gates.")
    parser.add_argument("--gold-candidates", required=True, help="Path to reviewed gold candidates JSONL.")
    parser.add_argument("--output-jsonl", help="Optional path for curated gold records JSONL.")
    parser.add_argument("--output-manifest", help="Optional path for the writer audit manifest.")
    parser.add_argument("--confirm-curated-gold-write", action="store_true", help="Confirm curated gold output writing.")
    parser.add_argument("--dry-run", action="store_true", help="Run selection/preflight without writing record JSONL.")
    args = parser.parse_args(argv)

    if not args.output_jsonl and not args.output_manifest:
        print("output_required:jsonl_or_manifest", file=sys.stderr)
        return 1
    if not args.dry_run and not args.confirm_curated_gold_write:
        print("confirmation_required:curated_gold_write", file=sys.stderr)
        return 1
    try:
        candidates = load_oled_reviewed_gold_candidates_jsonl(args.gold_candidates)
        policy = OledCuratedGoldWriterPolicy(require_confirmation=not args.dry_run)
        report = run_oled_curated_gold_writer(
            candidates,
            output_jsonl_path=None if args.dry_run else args.output_jsonl,
            output_manifest_path=args.output_manifest,
            policy=policy,
            confirm_curated_gold_write=args.confirm_curated_gold_write,
        )
        summary = {
            "input_candidate_count": report.manifest.input_candidate_count,
            "output_record_count": report.manifest.output_record_count,
            "status_counts": report.manifest.status_counts,
            "reason_code_counts": report.manifest.reason_code_counts,
            "error_codes": report.error_codes,
            "warning_codes": report.warning_codes,
        }
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
        return 0 if report.is_valid else 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _policy_rejection_reasons(
    candidate: OledReviewedGoldCandidate,
    policy: OledCuratedGoldWriterPolicy,
) -> list[str]:
    reasons: list[str] = []
    if candidate.gold_record is None:
        reasons.append("missing_gold_record")
    if _status_not_writable(candidate, policy):
        reasons.append("status_not_writable")
    if policy.require_no_validation_errors and candidate.validation_error_codes:
        reasons.append("validation_errors_present")
    if not policy.allow_validation_warnings and candidate.validation_warning_codes:
        reasons.append("validation_warnings_present")
    if policy.require_evidence_refs and _missing_evidence_refs(candidate):
        reasons.append("missing_evidence_refs")
    if policy.require_reviewer and not (candidate.gold_record and candidate.gold_record.reviewer):
        reasons.append("missing_reviewer")
    if policy.require_candidate_only_source and not _is_candidate_only_source(candidate):
        reasons.append("source_not_candidate_only")
    if _source_claims_final_gold_dataset(candidate):
        reasons.append("source_claims_final_gold_dataset")
    return sorted(set(reasons))


def _status_not_writable(
    candidate: OledReviewedGoldCandidate,
    policy: OledCuratedGoldWriterPolicy,
) -> bool:
    if not policy.require_converted_status:
        return False
    if candidate.status == OledReviewedGoldCandidateStatus.CONVERTED:
        return False
    if (
        candidate.status == OledReviewedGoldCandidateStatus.CONVERTED_WITH_WARNINGS
        and policy.allow_converted_with_warnings
    ):
        return False
    return True


def _missing_evidence_refs(candidate: OledReviewedGoldCandidate) -> bool:
    if not candidate.source_evidence_anchors:
        return True
    if candidate.gold_record is None or not candidate.gold_record.evidence_refs:
        return True
    return False


def _is_candidate_only_source(candidate: OledReviewedGoldCandidate) -> bool:
    if candidate.metadata.get("candidate_only") is not True:
        return False
    if candidate.gold_record is not None and candidate.gold_record.metadata.get("candidate_only") is not True:
        return False
    return True


def _source_claims_final_gold_dataset(candidate: OledReviewedGoldCandidate) -> bool:
    if candidate.metadata.get("final_gold_dataset") is True:
        return True
    if candidate.gold_record is not None and candidate.gold_record.metadata.get("final_gold_dataset") is True:
        return True
    return False


def _apply_post_selection_validation(
    results: list[OledCuratedGoldWriteResult],
    candidates: list[OledReviewedGoldCandidate],
    selected_records_by_candidate_id: dict[str, OledGoldDatasetRecord],
    policy: OledCuratedGoldWriterPolicy,
) -> tuple[list[OledCuratedGoldWriteResult], list[OledGoldDatasetRecord], list[OledCuratedGoldWriterFinding]]:
    if not selected_records_by_candidate_id:
        return results, [], []

    validation_report = validate_oled_gold_dataset(selected_records_by_candidate_id.values())
    errors_by_record: dict[str, list[str]] = defaultdict(list)
    warnings_by_record: dict[str, list[str]] = defaultdict(list)
    validation_findings: list[OledCuratedGoldWriterFinding] = []
    candidates_by_record_id = {
        record.record_id: candidate
        for candidate in candidates
        if (record := selected_records_by_candidate_id.get(candidate.candidate_id)) is not None
    }
    for finding in validation_report.findings:
        if finding.severity == "error":
            errors_by_record[finding.record_id].append(finding.code)
            severity: Literal["error", "warning"] = "error"
        else:
            warnings_by_record[finding.record_id].append(finding.code)
            severity = "warning"
        candidate = candidates_by_record_id.get(finding.record_id)
        validation_findings.append(
            OledCuratedGoldWriterFinding(
                code=finding.code,
                severity=severity,
                message=finding.message,
                gold_candidate_id=candidate.candidate_id if candidate is not None else None,
                gold_record_id=finding.record_id,
            )
        )

    refreshed_results: list[OledCuratedGoldWriteResult] = []
    selected_records: list[OledGoldDatasetRecord] = []
    for result in results:
        if result.status != OledCuratedGoldWriteStatus.WRITTEN:
            refreshed_results.append(result)
            continue
        record = selected_records_by_candidate_id.get(result.gold_candidate_id)
        if record is None:
            refreshed_results.append(result)
            continue
        validation_errors = sorted(set(errors_by_record.get(record.record_id, [])))
        validation_warnings = sorted(set(warnings_by_record.get(record.record_id, [])))
        rejection_reasons: list[str] = []
        if policy.require_no_validation_errors and validation_errors:
            rejection_reasons.append("post_selection_validation_error")
        if not policy.allow_validation_warnings and validation_warnings:
            rejection_reasons.append("validation_warnings_present")
        if rejection_reasons:
            refreshed_results.append(
                result.model_copy(
                    update={
                        "status": OledCuratedGoldWriteStatus.REJECTED,
                        "reason_codes": sorted({*result.reason_codes, *rejection_reasons}),
                        "validation_error_codes": validation_errors,
                        "validation_warning_codes": validation_warnings,
                    }
                )
            )
            validation_findings.extend(
                _post_selection_policy_findings(
                    result,
                    record,
                    rejection_reasons,
                    validation_errors,
                    validation_warnings,
                )
            )
            continue
        refreshed_results.append(
            result.model_copy(
                update={
                    "validation_error_codes": validation_errors,
                    "validation_warning_codes": validation_warnings,
                }
            )
        )
        selected_records.append(record)
    return refreshed_results, sorted(selected_records, key=lambda item: item.record_id), validation_findings


def _post_selection_policy_findings(
    result: OledCuratedGoldWriteResult,
    record: OledGoldDatasetRecord,
    rejection_reasons: list[str],
    validation_errors: list[str],
    validation_warnings: list[str],
) -> list[OledCuratedGoldWriterFinding]:
    findings: list[OledCuratedGoldWriterFinding] = []
    if "post_selection_validation_error" in rejection_reasons:
        findings.append(
            OledCuratedGoldWriterFinding(
                code="post_selection_validation_error",
                severity="error",
                message="selected gold record failed post-selection gold validation",
                gold_candidate_id=result.gold_candidate_id,
                gold_record_id=record.record_id,
            )
        )
    if "validation_warnings_present" in rejection_reasons:
        findings.append(
            OledCuratedGoldWriterFinding(
                code="validation_warnings_present",
                severity="error",
                message="selected gold record has validation warnings disallowed by policy",
                gold_candidate_id=result.gold_candidate_id,
                gold_record_id=record.record_id,
            )
        )
    for code in validation_errors:
        findings.append(
            OledCuratedGoldWriterFinding(
                code=code,
                severity="error",
                message=f"post-selection validation error `{code}`",
                gold_candidate_id=result.gold_candidate_id,
                gold_record_id=record.record_id,
            )
        )
    for code in validation_warnings:
        findings.append(
            OledCuratedGoldWriterFinding(
                code=code,
                severity="warning",
                message=f"post-selection validation warning `{code}`",
                gold_candidate_id=result.gold_candidate_id,
                gold_record_id=record.record_id,
            )
        )
    return findings


def _manifest(
    *,
    candidates: list[OledReviewedGoldCandidate],
    policy: OledCuratedGoldWriterPolicy,
    results: list[OledCuratedGoldWriteResult],
    records: list[OledGoldDatasetRecord],
) -> OledCuratedGoldManifest:
    return OledCuratedGoldManifest(
        manifest_id=_manifest_id(candidates, policy),
        output_record_count=len(records),
        input_candidate_count=len(candidates),
        status_counts=dict(sorted(Counter(result.status.value for result in results).items())),
        reason_code_counts=dict(sorted(Counter(code for result in results for code in result.reason_codes).items())),
        written_record_ids=sorted(record.record_id for record in records),
        write_results=sorted(results, key=lambda item: item.gold_candidate_id),
        policy=policy,
        metadata=_safety_metadata(),
    )


def _manifest_id(
    candidates: list[OledReviewedGoldCandidate],
    policy: OledCuratedGoldWriterPolicy,
) -> str:
    payload = {
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "policy": policy.model_dump(mode="json"),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"oled-curated-gold-writer:{digest[:16]}"


def _write_result(
    candidate: OledReviewedGoldCandidate,
    *,
    status: OledCuratedGoldWriteStatus,
    reason_codes: list[str],
) -> OledCuratedGoldWriteResult:
    return OledCuratedGoldWriteResult(
        gold_candidate_id=candidate.candidate_id,
        gold_record_id=candidate.gold_record.record_id if candidate.gold_record is not None else None,
        source_reviewed_candidate_id=candidate.source_reviewed_candidate_id,
        source_packet_id=candidate.source_packet_id,
        status=status,
        reason_codes=reason_codes,
        validation_error_codes=candidate.validation_error_codes,
        validation_warning_codes=candidate.validation_warning_codes,
        source_evidence_anchors=candidate.source_evidence_anchors,
        metadata={
            "candidate_only_source": candidate.metadata.get("candidate_only") is True,
            "curated_dataset_written": False,
            "training_data_written": False,
        },
    )


def _findings_for_reasons(
    reason_codes: list[str],
    candidate: OledReviewedGoldCandidate,
    result: OledCuratedGoldWriteResult,
) -> list[OledCuratedGoldWriterFinding]:
    return [
        OledCuratedGoldWriterFinding(
            code=reason,
            severity=_reason_severity(reason),
            message=_REASON_MESSAGES.get(reason, f"candidate rejected by policy: {reason}"),
            gold_candidate_id=candidate.candidate_id,
            gold_record_id=result.gold_record_id,
        )
        for reason in reason_codes
    ]


def _reason_severity(reason: str) -> Literal["error", "warning"]:
    if reason == "selected_for_write":
        return "warning"
    return "error"


def _safety_metadata() -> dict[str, Any]:
    return {
        "curated_gold_writer": True,
        "training_data_written": False,
        "dataset_views_run": False,
        "leakage_splits_run": False,
        "feature_materialization_run": False,
        "model_backends_run": False,
        "llm_called": False,
        "mineru_called": False,
        "pdfs_read": False,
        "images_read": False,
    }


def _reject_forbidden_input(path: str | Path) -> None:
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        raise ValueError(f"forbidden_pdf_input:{redact_oled_mineru_acceptance_path(path)}")
    if suffix in _FORBIDDEN_IMAGE_SUFFIXES:
        raise ValueError(f"forbidden_image_input:{redact_oled_mineru_acceptance_path(path)}")


def _contains_absolute_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_absolute_path(item) for item in value)
    if isinstance(value, str):
        return Path(value).is_absolute()
    return False


def _sanitize_for_output(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_forbidden_payload_key(key):
                continue
            sanitized_value = _sanitize_for_output(raw_value)
            if sanitized_value in (None, {}, []):
                continue
            output[key] = sanitized_value
        return output
    if isinstance(value, list):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, tuple):
        output = []
        for item in value:
            sanitized_item = _sanitize_for_output(item)
            if sanitized_item not in (None, {}, []):
                output.append(sanitized_item)
        return output
    if isinstance(value, str):
        if Path(value).is_absolute():
            return redact_oled_mineru_acceptance_path(value)
        if len(value) > _MAX_OUTPUT_STRING_LENGTH:
            return value[: _MAX_OUTPUT_STRING_LENGTH - 3] + "..."
        return value
    return value


def _is_forbidden_payload_key(key: str) -> bool:
    normalized = key.lower()
    return any(
        token in normalized
        for token in (
            "raw_text",
            "full_text",
            "parsed_json",
            "table_body",
            "html_table",
            "markdown_table",
        )
    )


_REASON_MESSAGES = {
    "missing_gold_record": "gold candidate has no OledGoldDatasetRecord payload",
    "status_not_writable": "gold candidate status is not writable under the current policy",
    "validation_errors_present": "gold candidate carries validation error codes",
    "validation_warnings_present": "gold candidate carries validation warnings disallowed by policy",
    "missing_evidence_refs": "gold candidate or gold record is missing evidence refs",
    "missing_reviewer": "gold candidate is missing reviewer provenance required by policy",
    "source_not_candidate_only": "gold candidate lacks candidate-only provenance required by policy",
    "source_claims_final_gold_dataset": "gold candidate source claims final gold dataset status",
}

_MAX_OUTPUT_STRING_LENGTH = 240

_FORBIDDEN_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    ".svg",
}


__all__ = [
    "OledCuratedGoldWriterPolicy",
    "OledCuratedGoldWriteStatus",
    "OledCuratedGoldWriteResult",
    "OledCuratedGoldManifest",
    "OledCuratedGoldWriterFinding",
    "OledCuratedGoldWriterReport",
    "load_oled_reviewed_gold_candidates_jsonl",
    "select_oled_curated_gold_records",
    "write_oled_curated_gold_records_jsonl",
    "write_oled_curated_gold_manifest_json",
    "run_oled_curated_gold_writer",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
