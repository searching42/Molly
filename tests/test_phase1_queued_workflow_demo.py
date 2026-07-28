from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai4s_agent.adapters.phase1 import (
    filter_rank_adapter,
    predict_candidates_baseline_adapter,
    render_report_adapter,
    train_model_baseline_adapter,
)
from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.run_plan_queue_summary import RunPlanQueueExecutionSummary
from ai4s_agent.schemas import PlannedTask, RunPlan, RunStatus
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase1_queued_workflow_demo"
PERMISSION_ACTION = "run_plan_queue_execute"


class Phase1QueuedDemoExecutor:
    def __init__(self, *, storage: ProjectStorage, calls: list[dict[str, Any]]) -> None:
        self.storage = storage
        self.calls = calls

    def execute(
        self,
        *,
        project_id: str,
        run_plan: RunPlan,
        input_artifacts: dict[str, Any] | None = None,
        task_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        artifacts = {str(key): str(value) for key, value in (input_artifacts or {}).items()}
        options = task_options or {}
        self.calls.append(
            {
                "project_id": project_id,
                "run_id": run_plan.run_id,
                "input_artifacts": dict(artifacts),
                "task_options": dict(options),
            }
        )
        run_dir = self.storage.run_dir(project_id, run_plan.run_id)

        baseline_plan = RunPlan(
            run_id=run_plan.run_id,
            requested_tasks=["run_baseline"],
            tasks=[
                PlannedTask(task_id=task_id)
                for task_id in ("inspect_dataset", "clean_dataset", "check_trainability", "run_baseline")
            ],
            available_artifacts=["uploaded_dataset"],
            missing_artifacts=[],
        )
        baseline_result = RunPlanExecutor(storage=self.storage).execute(
            project_id=project_id,
            run_plan=baseline_plan,
            input_artifacts={"uploaded_dataset": artifacts["uploaded_dataset"]},
            task_options={key: value for key, value in options.items() if key in {"clean_dataset", "run_baseline"}},
        )
        if baseline_result.get("status") != RunStatus.SUCCEEDED.value:
            return {"ok": False, **baseline_result}

        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        baseline_report = _artifact_path(run_dir, registry, "baseline_report")
        baseline_payload = json.loads(baseline_report.read_text(encoding="utf-8"))
        baseline_outputs = baseline_payload.get("output_paths") if isinstance(baseline_payload.get("output_paths"), dict) else {}
        if baseline_outputs.get("model_metrics_json"):
            _register(
                self.storage,
                project_id,
                run_plan.run_id,
                run_dir,
                "baseline_metrics",
                Path(str(baseline_outputs["model_metrics_json"])),
            )
            registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        cleaned_csv = _artifact_path(run_dir, registry, "cleaned_train_dataset")

        train_options = _without_adapter(options.get("train_model", {}))
        train_result = train_model_baseline_adapter(
            {
                "run_id": run_plan.run_id,
                "cleaned_master_csv": str(cleaned_csv),
                "property_id": str(train_options.pop("property_id", "plqy")),
                "model_root": str(run_dir / "04_models"),
                **train_options,
            }
        )
        _require_success(train_result, "train_model_baseline_adapter")
        model_metadata = train_result["model_metadata"]
        model_dir = Path(str(model_metadata["model_dir"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "trained_model", model_dir)
        _register(self.storage, project_id, run_plan.run_id, run_dir, "model_metadata", model_dir / "model_metadata.json")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "model_manifest", model_dir / "model_manifest.json")
        _register(
            self.storage,
            project_id,
            run_plan.run_id,
            run_dir,
            "domain_model_manifest",
            model_dir / "domain_model_manifest.json",
        )

        prediction_csv = run_dir / "06_prediction" / f"{run_plan.run_id}_plqy_predictions.csv"
        prediction_result = predict_candidates_baseline_adapter(
            {
                "candidate_csv": artifacts["candidate_dataset"],
                "property_id": "plqy",
                "model_path": str(model_metadata["model_path"]),
                "output_csv": str(prediction_csv),
                **_without_adapter(options.get("predict_candidates", {})),
            }
        )
        _require_success(prediction_result, "predict_candidates_baseline_adapter")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "candidate_predictions", prediction_csv)

        rank_options = _without_adapter(options.get("filter_rank", {}))
        ranked_csv = run_dir / "07_rank" / f"{run_plan.run_id}_ranked_candidates.csv"
        rank_result = filter_rank_adapter(
            {
                "run_id": run_plan.run_id,
                "prediction_csv": str(prediction_csv),
                "output_csv": str(ranked_csv),
                "topn": 3,
                "score_columns": ["plqy_pred"],
                **rank_options,
            }
        )
        _require_success(rank_result, "filter_rank_adapter")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "ranked_candidates", ranked_csv)
        _register(self.storage, project_id, run_plan.run_id, run_dir, "topn_export", ranked_csv)

        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        report_options = _without_adapter(options.get("render_report", {}))
        report_result = render_report_adapter(
            {
                "run_id": run_plan.run_id,
                "output_dir": str(run_dir / "08_report"),
                "sections": {
                    "Summary": [
                        "Phase 1 queued workflow fixture completed.",
                        "Artifacts include cleaned data, baseline metrics, predictions, ranking, and report outputs.",
                    ],
                    "Ranking": rank_result["summary"],
                },
                "artifacts": dict(registry),
                **report_options,
            }
        )
        _require_success(report_result, "render_report_adapter")
        outputs = report_result["outputs"]
        _register(self.storage, project_id, run_plan.run_id, run_dir, "report_markdown", Path(outputs["markdown"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "report_html", Path(outputs["html"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "report_json", Path(outputs["json"]))

        final_registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        return {
            "ok": True,
            "run_id": run_plan.run_id,
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": [task.task_id for task in run_plan.tasks],
            "artifacts": final_registry,
        }


def test_phase1_queued_workflow_demo_success(tmp_path: Path) -> None:
    response, summary, calls, _app = _execute_demo(tmp_path)

    assert response.status_code == 200
    assert summary.ok is True
    assert summary.terminal is True
    assert summary.final_job is not None
    assert summary.final_job["status"] == "succeeded"
    assert summary.final_lease is not None
    assert summary.final_lease["status"] == "completed"
    assert summary.loop_results == ["completed", "idle"]
    assert calls[0]["project_id"] == "phase1-demo-project"
    assert calls[0]["run_id"] == "phase1-demo-run"


def test_phase1_queued_workflow_demo_outputs_artifacts(tmp_path: Path) -> None:
    _response, _summary, _calls, _app = _execute_demo(tmp_path)
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir("phase1-demo-project", "phase1-demo-run")
    registry = storage.read_artifact_registry("phase1-demo-project", "phase1-demo-run")

    required = {
        "cleaned_train_dataset",
        "baseline_report",
        "baseline_metrics",
        "model_metadata",
        "candidate_predictions",
        "ranked_candidates",
        "report_markdown",
        "report_json",
    }
    assert required <= set(registry)
    cleaned = _artifact_path(run_dir, registry, "cleaned_train_dataset")
    baseline = _artifact_path(run_dir, registry, "baseline_report")
    metrics = _artifact_path(run_dir, registry, "baseline_metrics")
    predictions = _artifact_path(run_dir, registry, "candidate_predictions")
    ranked = _artifact_path(run_dir, registry, "ranked_candidates")
    report_md = _artifact_path(run_dir, registry, "report_markdown")
    report_json = _artifact_path(run_dir, registry, "report_json")
    for path in (cleaned, baseline, metrics, predictions, ranked, report_md, report_json):
        assert path.exists(), path

    cleaned_rows = _csv_rows(cleaned)
    prediction_rows = _csv_rows(predictions)
    ranked_rows = _csv_rows(ranked)
    baseline_payload = json.loads(baseline.read_text(encoding="utf-8"))
    metrics_payload = json.loads(metrics.read_text(encoding="utf-8"))
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    report_markdown = report_md.read_text(encoding="utf-8")

    assert cleaned_rows
    assert "plqy" in cleaned_rows[0]
    assert any(item["property_id"] == "plqy" for item in baseline_payload["properties"])
    assert any(item["property_id"] == "plqy" and "metrics" in item for item in metrics_payload["properties"])
    assert prediction_rows
    assert "plqy_pred" in prediction_rows[0]
    assert ranked_rows
    assert "weighted_score" in ranked_rows[0]
    assert "Phase 1 queued workflow fixture completed" in report_markdown
    assert "ranked_candidates" in report_payload["artifacts"]


def test_phase1_queued_workflow_demo_status_route_reports_succeeded(tmp_path: Path) -> None:
    _response, _summary, calls, app = _execute_demo(tmp_path)
    calls.clear()
    client = app.test_client()

    response = client.get(
        "/api/internal/run-plan/queue/status?project_id=phase1-demo-project&run_id=phase1-demo-run",
        headers={"X-Actor": "phase1-demo-user"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["status"]["counts"]["succeeded"] == 1
    assert payload["status"]["counts"]["terminal_leases"] == 1
    assert payload["status"]["has_terminal_jobs"] is True
    assert calls == []


def test_phase1_queued_workflow_demo_audit_records_requested_succeeded(tmp_path: Path) -> None:
    _response, summary, _calls, _app = _execute_demo(tmp_path)

    audit = _audit_records(tmp_path)
    outcomes = [record["outcome"] for record in audit if record.get("event") == "internal_run_plan_queue_execute"]
    assert outcomes[-2:] == ["requested", "succeeded"]
    requested, succeeded = audit[-2], audit[-1]
    assert requested["actor"] == "phase1-demo-user"
    assert requested["actor_source"] == "header:X-Actor"
    assert requested["project_id"] == "phase1-demo-project"
    assert requested["run_id"] == "phase1-demo-run"
    assert requested["permission_allowed"] is True
    assert requested["permission_action"] == PERMISSION_ACTION
    assert requested["queued_job_id"] == ""
    assert succeeded["queued_job_id"] == summary.queued_job_id
    assert succeeded["outcome"] == "succeeded"
    assert succeeded["permission_allowed"] is True


def test_phase1_queued_workflow_demo_permission_required(tmp_path: Path) -> None:
    app, calls = _app_with_demo_executor(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/queue/execute",
        json=_payload(),
        headers={"X-Actor": "phase1-demo-user"},
    )

    assert response.status_code == 403
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.error is not None
    assert summary.error["type"] == "permission_denied"
    assert calls == []
    queue_dir = tmp_path / ".ai4s_internal" / "run_plan_queues" / "phase1-demo-project" / "phase1-demo-run"
    assert not (queue_dir / "worker_queue.json").exists()
    assert not (queue_dir / "worker_leases.json").exists()


def test_phase1_queued_workflow_demo_does_not_modify_default_execute_route(tmp_path: Path) -> None:
    app, _calls = _app_with_demo_executor(tmp_path)
    client = app.test_client()

    response = client.post("/api/run-plan/execute", json={})

    assert response.status_code == 400
    assert response.get_json() == {"ok": False, "error": "project_id required"}


def _execute_demo(tmp_path: Path):
    app, calls = _app_with_demo_executor(tmp_path)
    _grant_run_plan_queue_permission(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/queue/execute",
        json=_payload(),
        headers={"X-Actor": "phase1-demo-user"},
    )

    return response, RunPlanQueueExecutionSummary.model_validate(response.get_json()), calls, app


def _app_with_demo_executor(tmp_path: Path):
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []

    def factory(storage: ProjectStorage) -> Phase1QueuedDemoExecutor:
        return Phase1QueuedDemoExecutor(storage=storage, calls=calls)

    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = factory
    return app, calls


def _payload() -> dict[str, Any]:
    return {
        "project_id": "phase1-demo-project",
        "run_plan": _read_json(FIXTURE_DIR / "run_plan.json"),
        "input_artifacts": _resolved_input_artifacts(),
        "task_options": _read_json(FIXTURE_DIR / "task_options.json"),
        "max_iterations": 10,
    }


def _resolved_input_artifacts() -> dict[str, str]:
    raw = _read_json(FIXTURE_DIR / "input_artifacts.json")
    return {str(key): str((FIXTURE_DIR / str(value)).resolve()) for key, value in raw.items()}


def _grant_run_plan_queue_permission(tmp_path: Path) -> dict[str, Any]:
    return ServerPermissionStore(tmp_path).create_grant(
        "phase1-demo-project",
        PERMISSION_ACTION,
        actor="phase1-demo-admin",
        actor_source="test",
        run_id="phase1-demo-run",
        reason="phase1 queued workflow fixture",
    )


def _audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / ".ai4s_internal" / "audit" / "internal_run_plan_queue_audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact_path(run_dir: Path, registry: dict[str, str], artifact_id: str) -> Path:
    relative = registry.get(artifact_id)
    assert relative, f"missing artifact: {artifact_id}"
    return (run_dir / relative).resolve()


def _register(
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    run_dir: Path,
    artifact_id: str,
    path: Path,
) -> None:
    storage.register_artifact_path(project_id, run_id, artifact_id, str(path.resolve().relative_to(run_dir.resolve())))


def _without_adapter(options: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(options, dict):
        return {}
    return {str(key): value for key, value in options.items() if str(key) != "adapter"}


def _require_success(result: dict[str, Any], label: str) -> None:
    if result.get("status") != "success":
        raise AssertionError(f"{label} failed: {json.dumps(result, sort_keys=True)}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
