from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai4s_agent.adapters.phase1 import (
    filter_rank_adapter,
    generate_candidates_stub_adapter,
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


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase2_generation_screening_demo"
PROJECT_ID = "phase2-demo-project"
RUN_ID = "phase2-demo-run"
PERMISSION_ACTION = "run_plan_queue_execute"


class Phase2GenerationScreeningExecutor:
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
        self._run_phase1_baseline_chain(project_id, run_plan, artifacts, options)
        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        cleaned_csv = _artifact_path(run_dir, registry, "cleaned_train_dataset")
        model_metadata = self._train_lightweight_baseline_model(project_id, run_plan, run_dir, cleaned_csv, options)
        generated_csv, generation_report = self._generate_candidates(project_id, run_plan, run_dir, cleaned_csv, options)
        predictions_csv = self._predict_candidates(project_id, run_plan, run_dir, generated_csv, model_metadata, options)
        ranked_csv, rank_result = self._rank_candidates(project_id, run_plan, run_dir, predictions_csv, options)
        self._render_report(project_id, run_plan, run_dir, generation_report, rank_result, options)
        final_registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        return {
            "ok": True,
            "run_id": run_plan.run_id,
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": [task.task_id for task in run_plan.tasks],
            "artifacts": final_registry,
        }

    def _run_phase1_baseline_chain(
        self,
        project_id: str,
        run_plan: RunPlan,
        artifacts: dict[str, str],
        options: dict[str, dict[str, Any]],
    ) -> None:
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
        result = RunPlanExecutor(storage=self.storage).execute(
            project_id=project_id,
            run_plan=baseline_plan,
            input_artifacts={"uploaded_dataset": artifacts["uploaded_dataset"]},
            task_options={key: value for key, value in options.items() if key in {"clean_dataset", "run_baseline"}},
        )
        if result.get("status") != RunStatus.SUCCEEDED.value:
            raise AssertionError(f"phase1 baseline chain failed: {json.dumps(result, sort_keys=True)}")
        run_dir = self.storage.run_dir(project_id, run_plan.run_id)
        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        baseline_report = _artifact_path(run_dir, registry, "baseline_report")
        baseline_payload = json.loads(baseline_report.read_text(encoding="utf-8"))
        outputs = baseline_payload.get("output_paths") if isinstance(baseline_payload.get("output_paths"), dict) else {}
        if outputs.get("model_metrics_json"):
            _register(self.storage, project_id, run_plan.run_id, run_dir, "baseline_metrics", Path(str(outputs["model_metrics_json"])))

    def _train_lightweight_baseline_model(
        self,
        project_id: str,
        run_plan: RunPlan,
        run_dir: Path,
        cleaned_csv: Path,
        options: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        train_options = _without_adapter(options.get("train_model", {}))
        result = train_model_baseline_adapter(
            {
                "run_id": run_plan.run_id,
                "cleaned_master_csv": str(cleaned_csv),
                "property_id": str(train_options.pop("property_id", "plqy")),
                "model_root": str(run_dir / "04_models"),
                **train_options,
            }
        )
        _require_success(result, "train_model_baseline_adapter")
        metadata = result["model_metadata"]
        model_dir = Path(str(metadata["model_dir"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "trained_model", model_dir)
        _register(self.storage, project_id, run_plan.run_id, run_dir, "model_metadata", model_dir / "model_metadata.json")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "model_manifest", model_dir / "model_manifest.json")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "domain_model_manifest", model_dir / "domain_model_manifest.json")
        return metadata

    def _generate_candidates(
        self,
        project_id: str,
        run_plan: RunPlan,
        run_dir: Path,
        cleaned_csv: Path,
        options: dict[str, dict[str, Any]],
    ) -> tuple[Path, Path]:
        result = generate_candidates_stub_adapter(
            {
                "run_id": run_plan.run_id,
                "output_dir": str(run_dir / "05_generation"),
                "backend": "deterministic_stub",
                "count": 8,
                "seed": 7,
                "reference_csv": str(cleaned_csv),
                **_without_adapter(options.get("generate_candidates", {})),
            }
        )
        _require_success(result, "generate_candidates_stub_adapter")
        generated_csv = Path(str(result["outputs"]["candidate_csv"]))
        generation_report = Path(str(result["outputs"]["generation_report_json"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "candidate_dataset", generated_csv)
        _register(self.storage, project_id, run_plan.run_id, run_dir, "generated_candidates", generated_csv)
        _register(self.storage, project_id, run_plan.run_id, run_dir, "generation_report", generation_report)
        return generated_csv, generation_report

    def _predict_candidates(
        self,
        project_id: str,
        run_plan: RunPlan,
        run_dir: Path,
        generated_csv: Path,
        model_metadata: dict[str, Any],
        options: dict[str, dict[str, Any]],
    ) -> Path:
        prediction_csv = run_dir / "06_prediction" / f"{run_plan.run_id}_plqy_predictions.csv"
        result = predict_candidates_baseline_adapter(
            {
                "candidate_csv": str(generated_csv),
                "property_id": "plqy",
                "model_path": str(model_metadata["model_path"]),
                "output_csv": str(prediction_csv),
                **_without_adapter(options.get("predict_candidates", {})),
            }
        )
        _require_success(result, "predict_candidates_baseline_adapter")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "candidate_predictions", prediction_csv)
        return prediction_csv

    def _rank_candidates(
        self,
        project_id: str,
        run_plan: RunPlan,
        run_dir: Path,
        predictions_csv: Path,
        options: dict[str, dict[str, Any]],
    ) -> tuple[Path, dict[str, Any]]:
        ranked_csv = run_dir / "07_rank" / f"{run_plan.run_id}_ranked_candidates.csv"
        result = filter_rank_adapter(
            {
                "run_id": run_plan.run_id,
                "prediction_csv": str(predictions_csv),
                "output_csv": str(ranked_csv),
                "topn": 4,
                "score_columns": ["plqy_pred"],
                **_without_adapter(options.get("filter_rank", {})),
            }
        )
        _require_success(result, "filter_rank_adapter")
        _register(self.storage, project_id, run_plan.run_id, run_dir, "ranked_candidates", ranked_csv)
        _register(self.storage, project_id, run_plan.run_id, run_dir, "topn_export", ranked_csv)
        return ranked_csv, result

    def _render_report(
        self,
        project_id: str,
        run_plan: RunPlan,
        run_dir: Path,
        generation_report: Path,
        rank_result: dict[str, Any],
        options: dict[str, dict[str, Any]],
    ) -> None:
        registry = self.storage.read_artifact_registry(project_id, run_plan.run_id)
        report_options = _without_adapter(options.get("render_report", {}))
        result = render_report_adapter(
            {
                "run_id": run_plan.run_id,
                "output_dir": str(run_dir / "08_report"),
                "sections": {
                    "Summary": [
                        "Phase 2 deterministic generation screening fixture completed.",
                        "Generated candidates were screened through Phase 1 prediction, ranking, and reporting.",
                    ],
                    "Generation": [f"generation_report: {generation_report}"],
                    "Ranking": rank_result["summary"],
                },
                "artifacts": dict(registry),
                **report_options,
            }
        )
        _require_success(result, "render_report_adapter")
        outputs = result["outputs"]
        _register(self.storage, project_id, run_plan.run_id, run_dir, "report_markdown", Path(outputs["markdown"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "report_html", Path(outputs["html"]))
        _register(self.storage, project_id, run_plan.run_id, run_dir, "report_json", Path(outputs["json"]))


def test_phase2_generation_screening_demo_success(tmp_path: Path) -> None:
    response, summary, calls, _app = _execute_demo(tmp_path)

    assert response.status_code == 200
    assert summary.ok is True
    assert summary.terminal is True
    assert summary.final_job is not None
    assert summary.final_job["status"] == "succeeded"
    assert summary.final_lease is not None
    assert summary.final_lease["status"] == "completed"
    assert summary.loop_results == ["completed", "idle"]
    assert calls[0]["project_id"] == PROJECT_ID
    assert calls[0]["run_id"] == RUN_ID


def test_phase2_generation_screening_demo_outputs_artifacts(tmp_path: Path) -> None:
    _response, _summary, _calls, _app = _execute_demo(tmp_path)
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)

    required = {
        "generation_report",
        "generated_candidates",
        "candidate_dataset",
        "candidate_predictions",
        "ranked_candidates",
        "report_markdown",
        "report_json",
    }
    assert required <= set(registry)
    generation_report = _artifact_path(run_dir, registry, "generation_report")
    generated = _artifact_path(run_dir, registry, "generated_candidates")
    predictions = _artifact_path(run_dir, registry, "candidate_predictions")
    ranked = _artifact_path(run_dir, registry, "ranked_candidates")
    report_md = _artifact_path(run_dir, registry, "report_markdown")
    report_json = _artifact_path(run_dir, registry, "report_json")
    for path in (generation_report, generated, predictions, ranked, report_md, report_json):
        assert path.exists(), path

    generation_payload = json.loads(generation_report.read_text(encoding="utf-8"))
    ranked_rows = _csv_rows(ranked)
    report_payload = json.loads(report_json.read_text(encoding="utf-8"))
    report_markdown = report_md.read_text(encoding="utf-8")

    assert generation_payload["backend"] == "deterministic_stub"
    assert generation_payload["generated_count"] > 0
    assert ranked_rows
    assert "weighted_score" in ranked_rows[0]
    assert "Phase 2 deterministic generation screening fixture completed" in report_markdown
    assert "generation_report" in report_payload["artifacts"]
    assert "ranked_candidates" in report_payload["artifacts"]


def test_phase2_generation_screening_demo_generation_flows_into_prediction(tmp_path: Path) -> None:
    _response, _summary, _calls, _app = _execute_demo(tmp_path)
    storage = ProjectStorage(tmp_path)
    run_dir = storage.run_dir(PROJECT_ID, RUN_ID)
    registry = storage.read_artifact_registry(PROJECT_ID, RUN_ID)
    generated_rows = _csv_rows(_artifact_path(run_dir, registry, "generated_candidates"))
    prediction_rows = _csv_rows(_artifact_path(run_dir, registry, "candidate_predictions"))
    ranked_rows = _csv_rows(_artifact_path(run_dir, registry, "ranked_candidates"))

    generated_ids = {row["candidate_id"] for row in generated_rows}
    generated_smiles = {row["SMILES"] for row in generated_rows}
    prediction_ids = {row["candidate_id"] for row in prediction_rows}
    prediction_smiles = {row["SMILES"] for row in prediction_rows}
    ranked_ids = {row["candidate_id"] for row in ranked_rows}

    assert generated_ids <= prediction_ids
    assert generated_smiles <= prediction_smiles
    assert ranked_ids <= prediction_ids
    assert all(row.get("plqy_pred") not in {"", None} for row in prediction_rows)


def test_phase2_generation_screening_demo_status_and_audit(tmp_path: Path) -> None:
    _response, summary, calls, app = _execute_demo(tmp_path)
    calls.clear()
    client = app.test_client()

    status = client.get(
        f"/api/internal/run-plan/queue/status?project_id={PROJECT_ID}&run_id={RUN_ID}",
        headers={"X-Actor": "phase2-demo-user"},
    )

    assert status.status_code == 200
    payload = status.get_json()
    assert payload["status"]["counts"]["succeeded"] == 1
    assert payload["status"]["counts"]["terminal_leases"] == 1
    assert calls == []
    audit = _audit_records(tmp_path)
    outcomes = [record["outcome"] for record in audit if record.get("event") == "internal_run_plan_queue_execute"]
    assert outcomes[-2:] == ["requested", "succeeded"]
    requested, succeeded = audit[-2], audit[-1]
    assert requested["actor"] == "phase2-demo-user"
    assert requested["permission_allowed"] is True
    assert requested["permission_action"] == PERMISSION_ACTION
    assert succeeded["queued_job_id"] == summary.queued_job_id
    assert succeeded["permission_allowed"] is True


def test_phase2_generation_screening_demo_permission_required(tmp_path: Path) -> None:
    app, calls = _app_with_demo_executor(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/internal/run-plan/queue/execute",
        json=_payload(),
        headers={"X-Actor": "phase2-demo-user"},
    )

    assert response.status_code == 403
    summary = RunPlanQueueExecutionSummary.model_validate(response.get_json())
    assert summary.ok is False
    assert summary.error is not None
    assert summary.error["type"] == "permission_denied"
    assert calls == []
    queue_dir = tmp_path / ".ai4s_internal" / "run_plan_queues" / PROJECT_ID / RUN_ID
    assert not (queue_dir / "worker_queue.json").exists()
    assert not (queue_dir / "worker_leases.json").exists()


def test_phase2_generation_screening_demo_does_not_modify_default_execute_route(tmp_path: Path) -> None:
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
        headers={"X-Actor": "phase2-demo-user"},
    )

    return response, RunPlanQueueExecutionSummary.model_validate(response.get_json()), calls, app


def _app_with_demo_executor(tmp_path: Path):
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []

    def factory(storage: ProjectStorage) -> Phase2GenerationScreeningExecutor:
        return Phase2GenerationScreeningExecutor(storage=storage, calls=calls)

    app.config["AI4S_ENABLE_INTERNAL_RUN_PLAN_QUEUE_ROUTE"] = True
    app.config["AI4S_RUN_PLAN_QUEUE_EXECUTOR_FACTORY"] = factory
    return app, calls


def _payload() -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
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
        PROJECT_ID,
        PERMISSION_ACTION,
        actor="phase2-demo-admin",
        actor_source="test",
        run_id=RUN_ID,
        reason="phase2 deterministic generation screening fixture",
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
