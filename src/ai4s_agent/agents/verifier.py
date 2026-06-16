from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.schemas import RunObservation, VerificationFinding, VerificationReport
from ai4s_agent.storage import ProjectStorage


class VerifierAgent:
    """Rule-based Phase 4 verifier for audited run observations."""

    def __init__(self, *, conflict_rate_threshold: float = 0.25) -> None:
        self.conflict_rate_threshold = conflict_rate_threshold

    def verify(self, observation: RunObservation) -> VerificationReport:
        findings: list[VerificationFinding] = []
        findings.extend(self._artifact_findings(observation))
        findings.extend(self._extraction_findings(observation.reports))
        findings.extend(self._conflict_findings(observation.reports))
        findings.extend(self._unit_findings(observation.reports))
        findings.extend(self._leakage_findings(observation.reports))
        findings.extend(self._provenance_findings(observation.reports, observation.asset_manifests))
        findings.extend(self._approval_findings(observation))
        findings.extend(self._trainability_findings(observation.reports))
        findings.extend(self._model_metric_findings(observation.reports))

        decision = self._overall_decision(findings)
        stage = observation.stage_state.stage if observation.stage_state is not None else ""
        status = observation.stage_state.status.value if observation.stage_state is not None else ""
        return VerificationReport(
            project_id=observation.project_id,
            run_id=observation.run_id,
            generated_at=now_iso(),
            observed_stage=stage,
            observed_status=status,
            overall_decision=decision,
            findings=findings,
            summary=self._summary(decision, findings),
        )

    def write_reports(
        self,
        storage: ProjectStorage,
        project_id: str,
        run_id: str,
        report: VerificationReport,
    ) -> tuple[Path, Path]:
        run_dir = storage.run_dir(project_id, run_id)
        json_path = write_json(run_dir / "verification_report.json", report.model_dump(mode="json"))
        md_path = run_dir / "verification_report.md"
        md_path.write_text(self._render_markdown(report), encoding="utf-8")
        storage.register_artifact_path(project_id, run_id, "verification_report_json", json_path.name)
        storage.register_artifact_path(project_id, run_id, "verification_report_md", md_path.name)
        return json_path, md_path

    def _artifact_findings(self, observation: RunObservation) -> list[VerificationFinding]:
        findings: list[VerificationFinding] = []
        for artifact in observation.artifacts:
            if not artifact.exists:
                findings.append(
                    self._finding(
                        category="missing_artifact",
                        severity="error",
                        decision="ask_user",
                        message=f"Artifact `{artifact.artifact_id}` is registered but missing on disk.",
                        evidence={"artifact_id": artifact.artifact_id, "relative_path": artifact.relative_path},
                    )
                )
        return findings

    def _extraction_findings(self, reports: dict[str, dict[str, Any]]) -> list[VerificationFinding]:
        report = self._find_report(reports, "extraction_confidence_report")
        if not report:
            return []
        findings: list[VerificationFinding] = []
        attempted = self._int_field(report, "attempted_hit_count", "extraction_confidence_report", findings)
        extracted = self._int_field(report, "extracted_record_count", "extraction_confidence_report", findings)
        high = self._int_field(report, "high_confidence_count", "extraction_confidence_report", findings)
        low = self._int_field(report, "low_confidence_count", "extraction_confidence_report", findings)
        if attempted > 0 and extracted == 0:
            findings.append(
                self._finding(
                    category="empty_extraction",
                    severity="error",
                    decision="retry",
                    message="Extraction attempted evidence hits but produced no usable records.",
                    evidence={"attempted_hit_count": attempted, "extracted_record_count": extracted},
                )
            )
        if attempted > 0 and low > 0 and (high == 0 or low > high):
            findings.append(
                self._finding(
                    category="low_confidence",
                    severity="warning",
                    decision="ask_user",
                    message="Extraction confidence is too low for automatic promotion.",
                    evidence={"high_confidence_count": high, "low_confidence_count": low},
                )
            )
        return findings

    def _conflict_findings(self, reports: dict[str, dict[str, Any]]) -> list[VerificationFinding]:
        report = self._find_report(reports, "conflict_report")
        if not report:
            return []
        findings: list[VerificationFinding] = []
        input_count = self._int_field(report, "input_record_count", "conflict_report", findings)
        conflict_count = self._int_field(report, "conflict_count", "conflict_report", findings)
        rate = conflict_count / input_count if input_count else 0.0
        if rate < self.conflict_rate_threshold:
            return findings
        findings.append(
            self._finding(
                category="high_conflict_rate",
                severity="warning",
                decision="ask_user",
                message="Merged literature records have a high conflict rate.",
                evidence={"input_record_count": input_count, "conflict_count": conflict_count, "conflict_rate": rate},
            )
        )
        return findings

    def _unit_findings(self, reports: dict[str, dict[str, Any]]) -> list[VerificationFinding]:
        report = self._find_report(reports, "unit_normalization_report")
        if not report:
            return []
        findings: list[VerificationFinding] = []
        warning_count = self._int_field(report, "warning_count", "unit_normalization_report", findings)
        if warning_count <= 0:
            return findings
        findings.append(
            self._finding(
                category="invalid_units",
                severity="warning",
                decision="ask_user",
                message="Unit normalization produced warnings that require review.",
                evidence={"warning_count": warning_count, "warnings": report.get("warnings", [])},
            )
        )
        return findings

    def _leakage_findings(self, reports: dict[str, dict[str, Any]]) -> list[VerificationFinding]:
        report = self._find_report(reports, "leakage_report")
        if not report:
            return []
        findings: list[VerificationFinding] = []
        overlap_count = self._int_field(report, "overlap_count", "leakage_report", findings)
        if overlap_count <= 0:
            return findings
        findings.append(
            self._finding(
                category="data_leakage",
                severity="error",
                decision="ask_user",
                message="Leakage check found overlapping molecules that require review.",
                evidence={"overlap_count": overlap_count, "overlap_smiles": report.get("overlap_smiles", [])},
            )
        )
        return findings

    def _provenance_findings(
        self,
        reports: dict[str, dict[str, Any]],
        asset_manifests: list[Any],
    ) -> list[VerificationFinding]:
        report = self._find_report(reports, "citation_license_report")
        if not report:
            return []
        findings: list[VerificationFinding] = []
        unknown_license = self._int_field(report, "unknown_license_count", "citation_license_report", findings)
        sources = report.get("sources", []) if report else []
        missing_source_fields = 0
        if isinstance(sources, list):
            for source in sources:
                if not isinstance(source, dict):
                    continue
                if not any(str(source.get(key) or "").strip() for key in ("doi", "citation", "source_path", "title")):
                    missing_source_fields += 1
        if unknown_license <= 0 and missing_source_fields <= 0:
            return findings
        findings.append(
            self._finding(
                category="missing_provenance",
                severity="warning",
                decision="ask_user",
                message="Source provenance or licensing is incomplete.",
                evidence={
                    "unknown_license_count": unknown_license,
                    "missing_source_fields": missing_source_fields,
                    "asset_manifest_count": len(asset_manifests),
                },
            )
        )
        return findings

    def _approval_findings(self, observation: RunObservation) -> list[VerificationFinding]:
        if not observation.approval_records:
            return []
        stage_started_at = observation.stage_state.started_at if observation.stage_state is not None else ""
        stage_start = self._parse_iso_timestamp(stage_started_at)
        findings: list[VerificationFinding] = []
        for record in observation.approval_records:
            approved_at_raw = str(record.get("approved_at") or "").strip()
            approved_at = self._parse_iso_timestamp(approved_at_raw)
            if stage_start is None:
                findings.append(
                    self._stale_approval_finding(
                        record=record,
                        stage_started_at=stage_started_at,
                        reason="stage timestamp is missing or invalid",
                    )
                )
                continue
            if approved_at is None:
                findings.append(
                    self._stale_approval_finding(
                        record=record,
                        stage_started_at=stage_started_at,
                        reason="approval timestamp is missing or invalid",
                    )
                )
                continue
            if stage_start is not None and approved_at < stage_start:
                findings.append(
                    self._stale_approval_finding(
                        record=record,
                        stage_started_at=stage_started_at,
                        reason="approval predates the current stage attempt",
                    )
                )
        return findings

    def _trainability_findings(self, reports: dict[str, dict[str, Any]]) -> list[VerificationFinding]:
        report = self._find_report(reports, "trainability_report")
        if not report:
            return []
        status = str(report.get("overall_status") or "").upper()
        if status not in {"BLOCKED", "NOT_READY"}:
            return []
        return [
            self._finding(
                category="poor_trainability",
                severity="error",
                decision="replan",
                message="Training data is not ready for model training.",
                evidence={"overall_status": status, "properties": report.get("properties", [])},
            )
        ]

    def _model_metric_findings(self, reports: dict[str, dict[str, Any]]) -> list[VerificationFinding]:
        report = self._find_report(reports, "model_metrics")
        if not report:
            return []
        weak_metrics: list[dict[str, Any]] = []
        for item in report.get("properties", []):
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            r2 = metrics.get("r2")
            try:
                r2_value = float(r2)
            except (TypeError, ValueError):
                continue
            if r2_value < 0.0:
                weak_metrics.append({"property_id": item.get("property_id", ""), "r2": r2_value})
        if not weak_metrics:
            return []
        return [
            self._finding(
                category="abnormal_model_metrics",
                severity="warning",
                decision="replan",
                message="Model metrics are abnormal and should not advance without review.",
                evidence={"weak_metrics": weak_metrics},
            )
        ]

    def _finding(
        self,
        *,
        category: str,
        severity: str,
        decision: str,
        message: str,
        evidence: dict[str, Any],
    ) -> VerificationFinding:
        digest = hashlib.sha1(f"{category}:{message}".encode("utf-8")).hexdigest()[:10]
        return VerificationFinding(
            finding_id=f"{category}_{digest}",
            category=category,
            severity=severity,
            decision=decision,
            message=message,
            evidence=evidence,
        )

    def _malformed_report_finding(self, *, report_key: str, field: str, value: Any) -> VerificationFinding:
        return self._finding(
            category="malformed_report",
            severity="warning",
            decision="ask_user",
            message=f"Report `{report_key}` has a non-integer `{field}` value.",
            evidence={"report": report_key, "field": field, "value": str(value)},
        )

    def _int_field(
        self,
        report: dict[str, Any],
        field: str,
        report_key: str,
        findings: list[VerificationFinding],
    ) -> int:
        raw = report.get(field)
        if raw in (None, ""):
            return 0
        if isinstance(raw, bool):
            findings.append(self._malformed_report_finding(report_key=report_key, field=field, value=raw))
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            findings.append(self._malformed_report_finding(report_key=report_key, field=field, value=raw))
            return 0

    def _stale_approval_finding(
        self,
        *,
        record: dict[str, Any],
        stage_started_at: str,
        reason: str,
    ) -> VerificationFinding:
        return self._finding(
            category="stale_approval",
            severity="warning",
            decision="ask_user",
            message="Approval must be reviewed before advancing this run stage.",
            evidence={
                "reason": reason,
                "approval_type": str(record.get("approval_type") or ""),
                "gate": str(record.get("gate") or ""),
                "asset_id": str(record.get("asset_id") or ""),
                "approved_at": str(record.get("approved_at") or ""),
                "stage_started_at": stage_started_at,
                "source_file": str(record.get("source_file") or ""),
            },
        )

    @staticmethod
    def _parse_iso_timestamp(value: str) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    @staticmethod
    def _find_report(reports: dict[str, dict[str, Any]], suffix: str) -> dict[str, Any]:
        if suffix in reports:
            return reports[suffix]
        for key, report in reports.items():
            if key.endswith(suffix):
                return report
        return {}

    @staticmethod
    def _overall_decision(findings: list[VerificationFinding]) -> str:
        if not findings:
            return "continue"
        priority = {"continue": 0, "retry": 1, "replan": 2, "ask_user": 3, "abort": 4}
        return max((finding.decision for finding in findings), key=lambda item: priority[item])

    @staticmethod
    def _summary(decision: str, findings: list[VerificationFinding]) -> str:
        if not findings:
            return "No verifier findings; the run can continue."
        return f"{len(findings)} verifier finding(s); recommended decision: {decision}."

    @staticmethod
    def _render_markdown(report: VerificationReport) -> str:
        lines = [
            "# Verification Report",
            "",
            f"- Run: `{report.run_id}`",
            f"- Project: `{report.project_id}`",
            f"- Decision: `{report.overall_decision}`",
            f"- Summary: {report.summary}",
            "",
            "| Severity | Category | Decision | Message |",
            "| --- | --- | --- | --- |",
        ]
        for finding in report.findings:
            lines.append(
                f"| {finding.severity} | {finding.category} | {finding.decision} | {finding.message} |"
            )
        return "\n".join(lines) + "\n"
