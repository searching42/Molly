from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ai4s_agent.adapters.phase1 import (
    check_trainability_service,
    filter_rank_adapter,
    inspect_dataset_service,
    predict_candidates_baseline_adapter,
    render_report_adapter,
    train_model_baseline_adapter,
)
from ai4s_agent.app import create_app
from ai4s_agent.run_plan_queue_summary import RunPlanQueueExecutionSummary
from ai4s_agent.schemas import RunPlan, RunStatus
from ai4s_agent.server_permissions import ServerPermissionStore
from ai4s_agent.storage import ProjectStorage


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "oled_multiobjective_screening_demo"
PROFILE_PATH = Path(__file__).parent / "fixtures" / "oled_property_profiles" / "oled_properties.json"
PROJECT_ID = "oled-multiobjective-demo-project"
RUN_ID = "oled-multiobjective-demo-run"
PERMISSION_ACTION = "run_plan_queue_execute"
TARGET_PROPERTIES = {"plqy", "lambda_em_nm", "homo_ev", "lumo_ev", "delta_e_st_ev"}


class OLEDMultiObjectiveScreeningExecutor:
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
        profile = _read_json(Path(artifacts["property_profile"]))
        target_properties = [str(item) for item in options.get("target_properties", []) if str(item)]
        if not target_properties:
            target_properties = [item["property_id"] for item in profile["properties"]]

        profile_report = self._write_property_profile_report(project_id, run_plan.run_id, run_dir, profile, target_properties)
        trainability_report = self._write_trainability_report(project_id, run_plan.run_id, run_dir, artifacts["train_dataset"], target_properties)
        model_payload = self._train_property_models(project_id, run_plan.run_id, run_dir, artifacts["train_dataset"], target_properties, options)
        predictions_csv = self._predict_all_properties(run_dir, artifacts["candidate_dataset"], model_payload["models"], target_properties)
        scored_csv = self._write_multiobjective_scores(run_dir, predictions_csv, profile, target_properties)
        ranked_csv, rank_result = self._rank_multiobjective(run_dir, scored_csv, profile, target_properties, options)
        self._render_report(
            project_id,
            run_plan.run_id,
            run_dir,
            profile,
            target_properties,
            profile_report,
            trainability_report,
            model_payload,
            predictions_csv,
            ranked_csv,
            rank_result,
            options,
        )
        return {
            "ok": True,
            "run_id": run_plan.run_id,
            "status": RunStatus.SUCCEEDED.value,
            "executed_tasks": [task.task_id for task in run_plan.tasks],
            "artifacts": self.storage.read_artifact_registry(project_id, run_plan.run_id),
        }

    def _write_property_profile_report(
        self,
        project_id: str,
        run_id: str,
        run_dir: Path,
        profile: dict[str, Any],
        target_properties: list[str],
    ) -> Path:
        by_id = {item["property_id"]: item for item in profile["properties"]}
        report = {
            "profile_id": profile["profile_id"],
            "version": profile["version"],
            "target_properties": target_properties,
            "properties": [by_id[property_id] for property_id in target_properties],
            "notes": [
                "OLED property profile is data configuration, not a core schema enum.",
                "LLM agents may inspect this profile for planning but do not generate executable code at runtime.",
            ],
        }
        path = _write_json(run_dir / "01_profile" / "property_profile_report.json", report)
        _register(self.storage, project_id, run_id, run_dir, "property_profile_report", path)
        return path

    def _write_trainability_report(
        self,
        project_id: str,
        run_id: str,
        run_dir: Path,
        train_dataset: str,
        target_properties: list[str],
    ) -> Path:
        inspect = inspect_dataset_service(
            {
                "input_csv": train_dataset,
                "min_numeric_ratio": 0.8,
                "min_nonempty": 5,
            }
        )
        _require_success(inspect, "inspect_dataset_service")
        properties = [
            {
                "property_id": item["property_id"],
                "effective_labels": item["nonempty_count"],
                "numeric_ratio": item["numeric_ratio"],
            }
            for item in inspect["property_candidates"]
            if item["property_id"] in target_properties
        ]
        trainability = check_trainability_service({"properties": properties})
        _require_success(trainability, "check_trainability_service")
        path = _write_json(
            run_dir / "02_trainability" / "trainability_report.json",
            {
                "dataset_profile": inspect["dataset_profile"],
                "property_candidates": inspect["property_candidates"],
                "trainability_report": trainability["trainability_report"],
            },
        )
        _register(self.storage, project_id, run_id, run_dir, "trainability_report", path)
        return path

    def _train_property_models(
        self,
        project_id: str,
        run_id: str,
        run_dir: Path,
        train_dataset: str,
        target_properties: list[str],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        train_options = _without_adapter(options.get("train_model", {}))
        models: dict[str, dict[str, Any]] = {}
        metrics: dict[str, Any] = {}
        for property_id in target_properties:
            result = train_model_baseline_adapter(
                {
                    "run_id": f"{run_id}_{property_id}",
                    "cleaned_master_csv": train_dataset,
                    "property_id": property_id,
                    "model_root": str(run_dir / "04_models"),
                    "domain": "oled",
                    "use_case": "multi_objective_fixture_screening",
                    **train_options,
                }
            )
            _require_success(result, f"train_model_baseline_adapter:{property_id}")
            metadata = result["model_metadata"]
            models[property_id] = metadata
            metrics[property_id] = metadata.get("metrics", {})
            model_dir = Path(str(metadata["model_dir"]))
            _register(self.storage, project_id, run_id, run_dir, f"model_metadata_{property_id}", model_dir / "model_metadata.json")
        metrics_path = _write_json(
            run_dir / "04_models" / "multi_property_model_metrics.json",
            {
                "properties": [
                    {
                        "property_id": property_id,
                        "metrics": metrics[property_id],
                        "model_metadata": models[property_id],
                    }
                    for property_id in target_properties
                ]
            },
        )
        _register(self.storage, project_id, run_id, run_dir, "multi_property_model_metrics", metrics_path)
        return {"models": models, "metrics_path": metrics_path}

    def _predict_all_properties(
        self,
        run_dir: Path,
        candidate_dataset: str,
        models: dict[str, dict[str, Any]],
        target_properties: list[str],
    ) -> Path:
        current_csv = Path(candidate_dataset)
        prediction_dir = run_dir / "06_prediction"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        for property_id in target_properties:
            output_csv = prediction_dir / f"{property_id}_predictions.csv"
            result = predict_candidates_baseline_adapter(
                {
                    "candidate_csv": str(current_csv),
                    "property_id": property_id,
                    "model_path": str(models[property_id]["model_path"]),
                    "output_csv": str(output_csv),
                }
            )
            _require_success(result, f"predict_candidates_baseline_adapter:{property_id}")
            current_csv = output_csv
        final_csv = prediction_dir / "multi_property_predictions.csv"
        final_csv.write_text(current_csv.read_text(encoding="utf-8"), encoding="utf-8")
        return final_csv

    def _write_multiobjective_scores(
        self,
        run_dir: Path,
        predictions_csv: Path,
        profile: dict[str, Any],
        target_properties: list[str],
    ) -> Path:
        rows = _csv_rows(predictions_csv)
        by_id = {item["property_id"]: item for item in profile["properties"]}
        scored_rows: list[dict[str, Any]] = []
        for row in rows:
            scored = dict(row)
            for property_id in target_properties:
                spec = by_id[property_id]
                value = float(row[f"{property_id}_pred"])
                scored[f"{property_id}_score"] = round(_objective_score(value, spec), 8)
            scored_rows.append(scored)
        output = run_dir / "06_prediction" / "multi_property_scored_predictions.csv"
        _write_csv(output, scored_rows)
        return output

    def _rank_multiobjective(
        self,
        run_dir: Path,
        scored_csv: Path,
        profile: dict[str, Any],
        target_properties: list[str],
        options: dict[str, Any],
    ) -> tuple[Path, dict[str, Any]]:
        by_id = {item["property_id"]: item for item in profile["properties"]}
        score_columns = [f"{property_id}_score" for property_id in target_properties]
        weights = {
            f"{property_id}_score": float(by_id[property_id].get("ranking_default", {}).get("weight") or 1.0)
            for property_id in target_properties
        }
        output = run_dir / "07_rank" / "multiobjective_ranked_candidates.csv"
        result = filter_rank_adapter(
            {
                "run_id": RUN_ID,
                "prediction_csv": str(scored_csv),
                "output_csv": str(output),
                "topn": int(_without_adapter(options.get("filter_rank", {})).get("topn", 5)),
                "score_columns": score_columns,
                "directions": {column: "maximize" for column in score_columns},
                "weights": weights,
            }
        )
        _require_success(result, "filter_rank_adapter")
        return output, result

    def _render_report(
        self,
        project_id: str,
        run_id: str,
        run_dir: Path,
        profile: dict[str, Any],
        target_properties: list[str],
        profile_report: Path,
        trainability_report: Path,
        model_payload: dict[str, Any],
        predictions_csv: Path,
        ranked_csv: Path,
        rank_result: dict[str, Any],
        options: dict[str, Any],
    ) -> None:
        artifacts = self.storage.read_artifact_registry(project_id, run_id)
        report_options = _without_adapter(options.get("render_report", {}))
        sections = {
            "Summary": [
                "OLED multi-objective screening fixture completed.",
                "This is fixture-only, not full multi-task model training or inverse design.",
                f"Property profile: {profile['profile_id']} version {profile['version']}",
                "multi-objective screening uses multiple single-property predictions plus weighted ranking.",
            ],
            "Properties": target_properties,
            "Ranking": rank_result["summary"],
        }
        extra_sections = report_options.get("sections", {})
        if isinstance(extra_sections, dict):
            extra_summary = extra_sections.get("Summary")
            if isinstance(extra_summary, list):
                sections["Summary"].extend(str(item) for item in extra_summary)
            for key, value in extra_sections.items():
                if key != "Summary":
                    sections[str(key)] = value
        result = render_report_adapter(
            {
                "run_id": run_id,
                "output_dir": str(run_dir / "08_report"),
                "sections": sections,
                "artifacts": {
                    **artifacts,
                    "property_profile_report": str(profile_report),
                    "trainability_report": str(trainability_report),
                    "multi_property_model_metrics": str(model_payload["metrics_path"]),
                    "multi_property_predictions": str(predictions_csv),
                    "multiobjective_ranked_candidates": str(ranked_csv),
                },
            }
        )
        _require_success(result, "render_report_adapter")
        outputs = result["outputs"]
        _register(self.storage, project_id, run_id, run_dir, "multi_property_predictions", predictions_csv)
        _register(self.storage, project_id, run_id, run_dir, "multiobjective_ranked_candidates", ranked_csv)
        _register(self.storage, project_id, run_id, run_dir, "report_markdown", Path(outputs["markdown"]))
        _register(self.storage, project_id, run_id, run_dir, "report_json", Path(outputs["json"]))


def test_oled_property_profile_loads_and_validates() -> None:
    profile = _read_json(PROFILE_PATH)
    properties = profile["properties"]
    property_ids = [item["property_id"] for item in properties]

    assert TARGET_PROPERTIES <= set(property_ids)
    assert len(property_ids) == len(set(property_ids))
    assert "core_schema_enum" not in profile
    assert profile.get("restricts_property_ids") is not True
    for item in properties:
        assert item["aliases"]
        assert item["canonical_unit"]
        assert item["direction"] in {"maximize", "minimize", "target", "range"}
        assert item["task_type"] == "numeric_regression"
        assert item["property_family"]


def test_oled_multiobjective_dataset_inspection_detects_properties() -> None:
    inspect = _inspect_fixture_dataset()

    property_ids = {item["property_id"] for item in inspect["property_candidates"]}
    assert TARGET_PROPERTIES <= property_ids


def test_oled_multiobjective_trainability_checks_all_targets() -> None:
    trainability = _fixture_trainability()
    properties = {item["property_id"]: item for item in trainability["trainability_report"]["properties"]}

    assert TARGET_PROPERTIES <= set(properties)
    for property_id in TARGET_PROPERTIES:
        item = properties[property_id]
        assert item["effective_labels"] > 0
        assert item["numeric_ratio"] > 0
        assert "status" in item
        assert "reason" in item


def test_oled_multiobjective_screening_outputs_predictions_and_ranking(tmp_path: Path) -> None:
    response, summary, _calls, _app = _execute_demo(tmp_path)

    assert response.status_code == 200
    assert summary.ok is True
    assert summary.terminal is True
    assert summary.final_job is not None
    assert summary.final_job["status"] == "succeeded"
    assert summary.final_lease is not None
    assert summary.final_lease["status"] == "completed"

    run_dir = tmp_path / "projects" / PROJECT_ID / "runs" / RUN_ID
    profile_report = _read_json(run_dir / "01_profile" / "property_profile_report.json")
    trainability = _read_json(run_dir / "02_trainability" / "trainability_report.json")
    metrics = _read_json(run_dir / "04_models" / "multi_property_model_metrics.json")
    predictions = _csv_rows(run_dir / "06_prediction" / "multi_property_predictions.csv")
    ranked = _csv_rows(run_dir / "07_rank" / "multiobjective_ranked_candidates.csv")

    assert set(profile_report["target_properties"]) == TARGET_PROPERTIES
    assert {
        item["property_id"]
        for item in trainability["trainability_report"]["properties"]
    } >= TARGET_PROPERTIES
    assert {item["property_id"] for item in metrics["properties"]} == TARGET_PROPERTIES
    assert predictions
    assert ranked
    for property_id in TARGET_PROPERTIES:
        assert f"{property_id}_pred" in predictions[0]
        assert f"{property_id}_score" in ranked[0]
    assert "weighted_score" in ranked[0]


def test_oled_multiobjective_report_mentions_property_profile(tmp_path: Path) -> None:
    _response, _summary, _calls, _app = _execute_demo(tmp_path)
    run_dir = tmp_path / "projects" / PROJECT_ID / "runs" / RUN_ID
    report_md = (run_dir / "08_report" / f"{RUN_ID}_final_summary.md").read_text(encoding="utf-8")
    report_json = _read_json(run_dir / "08_report" / f"{RUN_ID}_final_summary.json")

    assert "oled_properties" in report_md
    assert "multi-objective screening" in report_md
    assert "fixture-only" in report_md
    for property_id in TARGET_PROPERTIES:
        assert property_id in report_md
    assert "property_profile_report" in report_json["artifacts"]


def test_oled_multiobjective_status_and_audit(tmp_path: Path) -> None:
    _response, summary, calls, app = _execute_demo(tmp_path)
    calls.clear()
    client = app.test_client()

    status = client.get(
        f"/api/internal/run-plan/queue/status?project_id={PROJECT_ID}&run_id={RUN_ID}",
        headers={"X-Actor": "oled-demo-user"},
    )

    assert status.status_code == 200
    payload = status.get_json()
    assert payload["status"]["counts"]["succeeded"] == 1
    assert payload["status"]["counts"]["terminal_leases"] == 1
    assert calls == []

    audit = _audit_records(tmp_path)
    outcomes = [record["outcome"] for record in audit if record.get("event") == "internal_run_plan_queue_execute"]
    assert outcomes[-2:] == ["requested", "succeeded"]
    assert audit[-2]["permission_allowed"] is True
    assert audit[-1]["queued_job_id"] == summary.queued_job_id


def test_oled_multiobjective_not_runtime_code_generation() -> None:
    profile = _read_json(PROFILE_PATH)
    task_options = _read_json(FIXTURE_DIR / "task_options.json")
    serialized_config = json.dumps({"profile": profile, "task_options": task_options}, sort_keys=True)

    assert "source_code" not in serialized_config
    assert "python_code" not in serialized_config
    assert "eval(" not in serialized_config
    assert "exec(" not in serialized_config
    assert "command" not in serialized_config
    assert all("ranking_default" in item for item in profile["properties"])


def test_oled_multiobjective_does_not_modify_default_execute_route(tmp_path: Path) -> None:
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
        headers={"X-Actor": "oled-demo-user"},
    )

    return response, RunPlanQueueExecutionSummary.model_validate(response.get_json()), calls, app


def _app_with_demo_executor(tmp_path: Path):
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    calls: list[dict[str, Any]] = []

    def factory(storage: ProjectStorage) -> OLEDMultiObjectiveScreeningExecutor:
        return OLEDMultiObjectiveScreeningExecutor(storage=storage, calls=calls)

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
    resolved: dict[str, str] = {}
    for key, value in raw.items():
        path = (FIXTURE_DIR / str(value)).resolve()
        resolved[str(key)] = str(path)
    return resolved


def _inspect_fixture_dataset() -> dict[str, Any]:
    return inspect_dataset_service(
        {
            "input_csv": str(FIXTURE_DIR / "train_dataset.csv"),
            "min_numeric_ratio": 0.8,
            "min_nonempty": 5,
        }
    )


def _fixture_trainability() -> dict[str, Any]:
    inspect = _inspect_fixture_dataset()
    properties = [
        {
            "property_id": item["property_id"],
            "effective_labels": item["nonempty_count"],
            "numeric_ratio": item["numeric_ratio"],
        }
        for item in inspect["property_candidates"]
        if item["property_id"] in TARGET_PROPERTIES
    ]
    return check_trainability_service({"properties": properties})


def _grant_run_plan_queue_permission(tmp_path: Path) -> dict[str, Any]:
    return ServerPermissionStore(tmp_path).create_grant(
        PROJECT_ID,
        PERMISSION_ACTION,
        actor="oled-demo-admin",
        actor_source="test",
        run_id=RUN_ID,
        reason="OLED multi-objective screening fixture",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _audit_records(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / ".ai4s_internal" / "audit" / "internal_run_plan_queue_audit.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise AssertionError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _objective_score(value: float, spec: dict[str, Any]) -> float:
    direction = str(spec.get("direction") or "maximize")
    typical = spec.get("typical_range") if isinstance(spec.get("typical_range"), list) else [value, value]
    lo = float(typical[0])
    hi = float(typical[1])
    span = max(abs(hi - lo), 1e-9)
    if direction == "minimize":
        return _clamp((hi - value) / span)
    if direction == "target":
        target = float(spec.get("ranking_default", {}).get("target_value"))
        return _clamp(1.0 - abs(value - target) / (span / 2.0))
    if direction == "range":
        range_spec = spec.get("ranking_default", {}).get("range")
        if isinstance(range_spec, list) and len(range_spec) == 2:
            range_lo = float(range_spec[0])
            range_hi = float(range_spec[1])
            if range_lo <= value <= range_hi:
                return 1.0
            distance = min(abs(value - range_lo), abs(value - range_hi))
            return _clamp(1.0 - distance / span)
    return _clamp((value - lo) / span)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _register(
    storage: ProjectStorage,
    project_id: str,
    run_id: str,
    run_dir: Path,
    artifact_id: str,
    path: Path,
) -> None:
    storage.register_artifact_path(project_id, run_id, artifact_id, str(path.resolve().relative_to(run_dir.resolve())))


def _without_adapter(options: Any) -> dict[str, Any]:
    if not isinstance(options, dict):
        return {}
    return {str(key): value for key, value in options.items() if str(key) != "adapter"}


def _require_success(result: dict[str, Any], label: str) -> None:
    if result.get("status") != "success":
        raise AssertionError(f"{label} failed: {json.dumps(result, sort_keys=True)}")
