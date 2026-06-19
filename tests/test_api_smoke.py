from __future__ import annotations

import json
from pathlib import Path

from ai4s_agent._utils import now_iso, write_json
from ai4s_agent.app import create_app
from ai4s_agent.schemas import RunStatus, StageState
from ai4s_agent.storage import ProjectStorage


def test_retry_run_refreshes_stage_start_time(tmp_path) -> None:
    app = create_app(base_runs_dir=tmp_path)
    client = app.test_client()
    plan = client.post("/api/plan", json={"project_id": "proj-a", "run_id": "run-1", "prompt": "test"})
    assert plan.status_code == 200
    assert client.post("/api/projects/proj-a/runs/run-1/stop").status_code == 200
    storage = ProjectStorage(workspace_dir=tmp_path)
    storage.write_stage_state(
        "proj-a",
        "run-1",
        StageState(
            stage="train_model",
            next_stage="predict_candidates",
            status=RunStatus.FAILED,
            started_at="2026-05-28T10:00:00Z",
            ended_at="2026-05-28T10:05:00Z",
            updated_at="2026-05-28T10:05:00Z",
            error={"retryable": True},
        ),
    )

    resp = client.post("/api/runs/run-1/retry", json={"project_id": "proj-a"})

    assert resp.status_code == 200
    state = storage.read_stage_state("proj-a", "run-1")
    assert state is not None
    assert state.status == RunStatus.PENDING
    assert state.ended_at is None
    assert state.started_at != "2026-05-28T10:00:00Z"
    assert state.started_at == state.updated_at
