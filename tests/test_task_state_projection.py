from __future__ import annotations

from ai4s_agent.task_state_projection import (
    TASK_STATE_PROJECTION_VERSION,
    build_task_state_projection,
)


def test_task_state_projection_uses_semantic_values_and_actual_regular_files(tmp_path) -> None:
    (tmp_path / "stage.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "job_state.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "background_job_state.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "artifact_registry.json").symlink_to(tmp_path / "stage.json")
    (tmp_path / "job.json").write_text("{}\n", encoding="utf-8")

    result = build_task_state_projection(
        run_path=tmp_path,
        project_id="proj-a",
        run_id="run-a",
        stage_payload={
            "status": "private.compute.invalid",
            "stage": "10.0.0.1",
            "history": [
                {"stage": "inspect_dataset", "status": "SUCCEEDED"},
                {"stage": "internal-node_42", "status": "RUNNING"},
            ],
            "artifacts": [
                {"artifact_id": "model_metadata"},
                {"artifact_id": "private.compute.invalid"},
            ],
        },
        artifact_payload={
            "artifacts": {
                "trained_model": "03_training/model.joblib",
                "internal-node_42": "private",
            }
        },
    )

    assert result == {
        "projection_version": TASK_STATE_PROJECTION_VERSION,
        "project_id": "proj-a",
        "run_id": "run-a",
        "status": "unavailable",
        "current_stage": "unavailable",
        "history": [{"stage": "inspect_dataset", "status": "SUCCEEDED"}],
        "artifact_ids": ["model_metadata", "trained_model"],
        "state_files": ["background_job_state.json", "job_state.json", "stage.json"],
    }


def test_task_state_projection_does_not_infer_registry_from_empty_payload(tmp_path) -> None:
    (tmp_path / "stage.json").write_text("{}\n", encoding="utf-8")

    result = build_task_state_projection(
        run_path=tmp_path,
        project_id="proj-a",
        run_id="run-a",
        stage_payload={"stage": "train_model", "status": "RUNNING"},
        artifact_payload={},
    )

    assert result["artifact_ids"] == []
    assert result["state_files"] == ["stage.json"]
