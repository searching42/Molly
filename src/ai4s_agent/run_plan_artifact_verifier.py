from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4s_agent._utils import now_iso
from ai4s_agent.run_plan_queue_lifecycle import internal_run_plan_queue_dir, read_run_plan_queue_status
from ai4s_agent.storage import ProjectStorage
from ai4s_agent.worker_queue import JsonWorkerQueueStore, WorkerQueue


RunPlanArtifactDecision = Literal["continue", "needs_review", "rerun_recommended", "blocked"]
RunPlanArtifactSeverity = Literal["info", "warning", "error", "critical"]


class RunPlanArtifactFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    category: str
    severity: RunPlanArtifactSeverity
    decision: RunPlanArtifactDecision
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)

    @field_validator("category", "message")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("field must be non-empty")
        return clean


class RunPlanArtifactVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    run_id: str
    generated_at: str
    decision: RunPlanArtifactDecision
    summary: str
    findings: list[RunPlanArtifactFinding] = Field(default_factory=list)
    observed: dict[str, Any] = Field(default_factory=dict)

    @field_validator("project_id", "run_id", "summary")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        clean = str(value or "").strip()
        if not clean:
            raise ValueError("field must be non-empty")
        return clean


def verify_run_plan_artifacts(
    *,
    workspace_dir: str | Path,
    project_id: str,
    run_id: str,
    queue_summary: dict[str, Any] | None = None,
    queue_status: dict[str, Any] | None = None,
    audit_records: list[dict[str, Any]] | None = None,
    artifact_registry: dict[str, str] | None = None,
) -> RunPlanArtifactVerification:
    """Read run artifacts and produce a fixed observer-verifier decision.

    This function is intentionally read-only. It does not execute adapters,
    invoke LLMs, mutate queues, or propose a revised plan.
    """

    workspace = Path(workspace_dir).expanduser().resolve()
    storage = ProjectStorage(workspace)
    run_dir = storage.run_dir(project_id, run_id)
    registry = _load_artifact_registry(storage, project_id, run_id, artifact_registry)
    queue_observation = _observe_queue(workspace, project_id, run_id, queue_summary=queue_summary, queue_status=queue_status)
    audit_observation = _observe_audit(workspace, project_id, run_id, audit_records=audit_records)
    artifacts_observation = _observe_artifacts(run_dir, registry)
    reports_observation = _observe_reports(run_dir, artifacts_observation["artifacts"])

    observed = {
        "queue": queue_observation,
        "audit": audit_observation,
        "artifacts": artifacts_observation,
        "reports": reports_observation,
    }
    findings: list[RunPlanArtifactFinding] = []
    findings.extend(_queue_findings(queue_observation))
    findings.extend(_audit_findings(audit_observation, queue_observation))
    findings.extend(_artifact_findings(artifacts_observation))
    findings.extend(_trainability_findings(reports_observation))
    findings.extend(_model_metric_findings(reports_observation))
    findings.extend(_generation_findings(reports_observation))
    findings.extend(_extraction_benchmark_findings(reports_observation))
    findings.extend(_multiobjective_ranking_findings(reports_observation))
    decision = _overall_decision(findings)
    return RunPlanArtifactVerification(
        project_id=project_id,
        run_id=run_id,
        generated_at=now_iso(),
        decision=decision,
        summary=_summary(decision, findings),
        findings=findings,
        observed=observed,
    )


def _load_artifact_registry(
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    explicit: dict[str, str] | None,
) -> dict[str, str]:
    registry = dict(storage.read_artifact_registry(project_id, run_id))
    if explicit:
        registry.update({str(key): str(value) for key, value in explicit.items()})
    return registry


def _observe_queue(
    workspace: Path,
    project_id: str,
    run_id: str,
    *,
    queue_summary: dict[str, Any] | None,
    queue_status: dict[str, Any] | None,
) -> dict[str, Any]:
    status = dict(queue_status or {})
    if not status:
        queue_dir = internal_run_plan_queue_dir(workspace, project_id, run_id)
        if (queue_dir / "worker_queue.json").exists() or (queue_dir / "worker_leases.json").exists():
            status = read_run_plan_queue_status(WorkerQueue(JsonWorkerQueueStore(queue_dir)))
    return {
        "summary": dict(queue_summary or {}),
        "status": status,
    }


def _observe_audit(
    workspace: Path,
    project_id: str,
    run_id: str,
    *,
    audit_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    records = list(audit_records) if audit_records is not None else _read_internal_audit_records(workspace)
    relevant = [
        record
        for record in records
        if str(record.get("project_id") or "") == project_id
        and str(record.get("run_id") or "") == run_id
        and str(record.get("event") or "") == "internal_run_plan_queue_execute"
    ]
    terminal = next(
        (
            record
            for record in reversed(relevant)
            if str(record.get("outcome") or "") not in {"requested", ""}
        ),
        None,
    )
    return {
        "records": relevant,
        "record_count": len(relevant),
        "terminal_outcome": str(terminal.get("outcome") or "") if isinstance(terminal, dict) else "",
        "terminal_record": terminal or {},
    }


def _read_internal_audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / ".ai4s_internal" / "audit" / "internal_run_plan_queue_audit.jsonl"
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            records.append(loaded)
    return records


def _observe_artifacts(run_dir: Path, registry: dict[str, str]) -> dict[str, Any]:
    artifacts: dict[str, dict[str, Any]] = {}
    for artifact_id, raw_path in sorted(registry.items()):
        path = _artifact_path(run_dir, raw_path)
        inside_run_dir = path == run_dir or path.is_relative_to(run_dir)
        exists = False
        size_bytes = 0
        if path.exists() and (path.is_file() or path.is_dir()):
            exists = True
            size_bytes = path.stat().st_size if path.is_file() else 0
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "path": str(path),
            "relative_path": str(path.relative_to(run_dir)) if inside_run_dir else str(path),
            "exists": exists,
            "size_bytes": size_bytes,
            "inside_run_dir": inside_run_dir,
        }
    return {
        "registry_count": len(registry),
        "artifacts": artifacts,
        "missing_artifacts": [
            artifact_id
            for artifact_id, item in artifacts.items()
            if not bool(item.get("exists"))
        ],
    }


def _artifact_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    return path.resolve()


def _observe_reports(run_dir: Path, artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    json_payloads: dict[str, dict[str, Any]] = {}
    csv_payloads: dict[str, list[dict[str, str]]] = {}
    for artifact_id, item in artifacts.items():
        if not item.get("exists"):
            continue
        path = Path(str(item["path"]))
        if path.suffix.lower() == ".json":
            payload = _read_json_object(path)
            if payload:
                json_payloads[artifact_id] = payload
        elif path.suffix.lower() == ".csv":
            csv_payloads[artifact_id] = _read_csv_rows(path)

    trainability = _first_json(json_payloads, ["trainability_report"])
    if trainability:
        reports["trainability_report"] = _unwrap_trainability_report(trainability)

    model_metrics = _first_json(json_payloads, ["baseline_metrics", "model_metrics", "multi_property_model_metrics"])
    if model_metrics:
        reports["model_metrics"] = model_metrics

    generation = _first_json(json_payloads, ["generation_report"])
    if generation:
        reports["generation_report"] = generation

    extraction_benchmark = _first_json(json_payloads, ["extraction_benchmark_report"])
    if extraction_benchmark:
        reports["extraction_benchmark"] = extraction_benchmark

    ranking_rows = _first_csv(csv_payloads, ["multiobjective_ranked_candidates"])
    if ranking_rows is not None:
        reports["multiobjective_ranking"] = {
            "row_count": len(ranking_rows),
            "columns": list(ranking_rows[0].keys()) if ranking_rows else [],
            "top_weighted_score": _top_weighted_score(ranking_rows),
        }

    return reports


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_json(payloads: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    for key in keys:
        payload = payloads.get(key)
        if isinstance(payload, dict) and payload:
            return payload
    return {}


def _first_csv(payloads: dict[str, list[dict[str, str]]], keys: list[str]) -> list[dict[str, str]] | None:
    for key in keys:
        if key in payloads:
            return payloads[key]
    return None


def _unwrap_trainability_report(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("trainability_report")
    return nested if isinstance(nested, dict) else payload


def _top_weighted_score(rows: list[dict[str, str]]) -> float | None:
    if not rows:
        return None
    return _safe_float(rows[0].get("weighted_score"))


def _queue_findings(observation: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    findings: list[RunPlanArtifactFinding] = []
    summary = observation.get("summary") if isinstance(observation.get("summary"), dict) else {}
    status = observation.get("status") if isinstance(observation.get("status"), dict) else {}
    if summary:
        if bool(summary.get("waiting_user")):
            findings.append(
                _finding(
                    category="waiting_user",
                    severity="warning",
                    decision="needs_review",
                    message="Queued execution is waiting for user review.",
                    evidence={
                        "waiting_task": str(summary.get("waiting_task") or ""),
                        "required_gates": list(summary.get("required_gates") or []),
                    },
                )
            )
        final_job = summary.get("final_job") if isinstance(summary.get("final_job"), dict) else {}
        if bool(summary.get("ok")) is False or str(final_job.get("status") or "") == "failed":
            findings.append(
                _finding(
                    category="queue_failed",
                    severity="critical",
                    decision="blocked",
                    message="Queued run-plan execution failed.",
                    evidence={"summary_error": summary.get("error"), "final_job": final_job},
                )
            )
        if summary.get("terminal") is False:
            findings.append(
                _finding(
                    category="queue_not_terminal",
                    severity="warning",
                    decision="needs_review",
                    message="Queued run-plan execution has not reached a terminal state.",
                    evidence={"queued_job_id": str(summary.get("queued_job_id") or "")},
                )
            )
    counts = status.get("counts") if isinstance(status.get("counts"), dict) else {}
    if int(counts.get("failed") or 0) > 0:
        findings.append(
            _finding(
                category="queue_failed",
                severity="critical",
                decision="blocked",
                message="Queue status contains failed jobs.",
                evidence={"counts": counts},
            )
        )
    if int(counts.get("waiting_user") or 0) > 0:
        findings.append(
            _finding(
                category="waiting_user",
                severity="warning",
                decision="needs_review",
                message="Queue status contains waiting-user jobs.",
                evidence={"counts": counts, "waiting_user_jobs": status.get("waiting_user_jobs", [])},
            )
        )
    return findings


def _audit_findings(audit: dict[str, Any], queue: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    if not audit.get("records") and not queue.get("summary") and not queue.get("status"):
        return []
    outcome = str(audit.get("terminal_outcome") or "")
    if not outcome:
        return [
            _finding(
                category="missing_audit",
                severity="warning",
                decision="needs_review",
                message="No terminal internal run-plan queue audit outcome was found.",
                evidence={"record_count": audit.get("record_count", 0)},
            )
        ]
    if outcome in {"failed", "permission_denied", "validation_error"}:
        return [
            _finding(
                category="audit_terminal_failure",
                severity="error",
                decision="blocked",
                message="Internal run-plan queue audit ended in a failure outcome.",
                evidence={"terminal_outcome": outcome, "terminal_record": audit.get("terminal_record", {})},
            )
        ]
    if outcome == "waiting_user":
        return [
            _finding(
                category="waiting_user",
                severity="warning",
                decision="needs_review",
                message="Audit outcome indicates the run is waiting for user review.",
                evidence={"terminal_record": audit.get("terminal_record", {})},
            )
        ]
    return []


def _artifact_findings(observation: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    missing = list(observation.get("missing_artifacts") or [])
    if not missing:
        return []
    return [
        _finding(
            category="missing_artifact",
            severity="critical",
            decision="blocked",
            message="Registered artifacts are missing on disk.",
            evidence={"missing_artifacts": missing},
        )
    ]


def _trainability_findings(reports: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    report = reports.get("trainability_report")
    if not isinstance(report, dict) or not report:
        return []
    status = str(report.get("overall_status") or "").upper()
    properties = report.get("properties", []) if isinstance(report.get("properties"), list) else []
    if status in {"BLOCKED", "NOT_READY"}:
        return [
            _finding(
                category="poor_trainability",
                severity="error",
                decision="blocked",
                message="Trainability report says the dataset is not ready.",
                evidence={"overall_status": status, "properties": properties},
            )
        ]
    if not properties:
        return [
            _finding(
                category="missing_trainability_properties",
                severity="warning",
                decision="needs_review",
                message="Trainability report does not list any properties.",
                evidence={"overall_status": status},
            )
        ]
    return []


def _model_metric_findings(reports: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    report = reports.get("model_metrics")
    if not isinstance(report, dict) or not report:
        return []
    weak_metrics: list[dict[str, Any]] = []
    for item in report.get("properties", []) if isinstance(report.get("properties"), list) else []:
        if not isinstance(item, dict):
            continue
        metrics = item.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        r2 = _safe_float(metrics.get("r2"))
        mae = _safe_float(metrics.get("mae"))
        if r2 is not None and r2 < 0.0:
            weak_metrics.append({"property_id": str(item.get("property_id") or ""), "r2": r2})
        if mae is not None and not math.isfinite(mae):
            weak_metrics.append({"property_id": str(item.get("property_id") or ""), "mae": str(metrics.get("mae"))})
    if not weak_metrics:
        return []
    return [
        _finding(
            category="poor_model_metrics",
            severity="warning",
            decision="rerun_recommended",
            message="Model metrics are weak enough to recommend a rerun before promotion.",
            evidence={"weak_metrics": weak_metrics},
        )
    ]


def _generation_findings(reports: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    report = reports.get("generation_report")
    if not isinstance(report, dict) or not report:
        return []
    generated_count = _safe_int(report.get("generated_count"))
    if generated_count is not None and generated_count <= 0:
        return [
            _finding(
                category="empty_generation",
                severity="warning",
                decision="needs_review",
                message="Candidate generation did not produce any candidates.",
                evidence={"generated_count": generated_count, "backend": report.get("backend")},
            )
        ]
    return []


def _extraction_benchmark_findings(reports: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    report = reports.get("extraction_benchmark")
    if not isinstance(report, dict) or not report:
        return []
    findings: list[RunPlanArtifactFinding] = []
    conflict_rate = _safe_float(report.get("conflict_rate"))
    if conflict_rate is not None and conflict_rate >= 0.25:
        findings.append(
            _finding(
                category="high_extraction_conflict_rate",
                severity="warning",
                decision="needs_review",
                message="Extraction benchmark reports a high conflict rate.",
                evidence={"conflict_rate": conflict_rate},
            )
        )
    labels = _safe_int(report.get("trainable_labels_gained"))
    if labels is not None and labels <= 0:
        findings.append(
            _finding(
                category="no_trainable_labels_gained",
                severity="warning",
                decision="rerun_recommended",
                message="Extraction benchmark did not add trainable labels.",
                evidence={"trainable_labels_gained": labels},
            )
        )
    return findings


def _multiobjective_ranking_findings(reports: dict[str, Any]) -> list[RunPlanArtifactFinding]:
    report = reports.get("multiobjective_ranking")
    if not isinstance(report, dict) or not report:
        return []
    if int(report.get("row_count") or 0) <= 0:
        return [
            _finding(
                category="empty_ranking",
                severity="warning",
                decision="needs_review",
                message="Multi-objective ranking did not produce ranked candidates.",
                evidence=report,
            )
        ]
    columns = set(report.get("columns", []) if isinstance(report.get("columns"), list) else [])
    if "weighted_score" not in columns:
        return [
            _finding(
                category="missing_weighted_score",
                severity="warning",
                decision="needs_review",
                message="Multi-objective ranking is missing weighted_score.",
                evidence={"columns": sorted(columns)},
            )
        ]
    return []


def _overall_decision(findings: list[RunPlanArtifactFinding]) -> RunPlanArtifactDecision:
    for decision in ("blocked", "rerun_recommended", "needs_review"):
        if any(finding.decision == decision for finding in findings):
            return decision  # type: ignore[return-value]
    return "continue"


def _summary(decision: str, findings: list[RunPlanArtifactFinding]) -> str:
    if not findings:
        return "Observer-verifier found no blocking, review, or rerun signals."
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.decision] = counts.get(finding.decision, 0) + 1
    parts = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    return f"Observer-verifier decision `{decision}` from {len(findings)} finding(s): {parts}."


def _finding(
    *,
    category: str,
    severity: RunPlanArtifactSeverity,
    decision: RunPlanArtifactDecision,
    message: str,
    evidence: dict[str, Any],
) -> RunPlanArtifactFinding:
    digest = hashlib.sha1(f"{category}:{message}:{json.dumps(evidence, sort_keys=True, default=str)}".encode("utf-8")).hexdigest()[:10]
    return RunPlanArtifactFinding(
        finding_id=f"{category}_{digest}",
        category=category,
        severity=severity,
        decision=decision,
        message=message,
        evidence=evidence,
    )


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
