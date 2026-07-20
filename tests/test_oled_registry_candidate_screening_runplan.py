from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent import adapters
from ai4s_agent.adapters.oled_registry_screening import (
    execute_oled_registry_candidate_screening_adapter,
)
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import GateName, RiskLevel, RunStatus
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_registry_candidate_screening import _screening_inputs


TASK_ID = "execute_oled_registry_candidate_screening"
ADAPTER_NAME = "execute_oled_registry_candidate_screening_adapter"
INPUT_ARTIFACT_IDS = (
    "oled_phase1_execution_dir",
    "oled_dataset_snapshot",
    "oled_registry_snapshot",
)
OUTPUT_FILENAMES = {
    "oled_registry_screening_receipt": "screening.json",
    "oled_registry_screening_shortlist": "ranked_shortlist.csv",
    "oled_registry_screening_predictions": "predictions.jsonl",
    "oled_registry_screening_exclusions": "excluded_candidates.jsonl",
    "oled_registry_screening_eligible_candidates": "eligible_candidates.csv",
    "oled_registry_screening_report": "report.md",
}
EXECUTION_RECORD_ARTIFACT_ID = "oled_registry_screening_execution_record"


def _run_plan(run_id: str) -> object:
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=list(INPUT_ARTIFACT_IDS),
    )


def _input_artifacts(*, execution_dir: Path, dataset_snapshot: Path, registry_snapshot: Path) -> dict[str, str]:
    return {
        "oled_phase1_execution_dir": str(execution_dir),
        "oled_dataset_snapshot": str(dataset_snapshot),
        "oled_registry_snapshot": str(registry_snapshot),
    }


def _screening_options() -> dict[str, dict[str, object]]:
    return {
        TASK_ID: {
            "minimums": ["s1_ev=0.0"],
            "maximums": ["delta_e_st_ev=1.0"],
        }
    }


def _placeholder_inputs(tmp_path: Path) -> dict[str, str]:
    execution_dir = tmp_path / "phase1-execution"
    execution_dir.mkdir()
    (execution_dir / "execution.json").write_text("{}\n", encoding="utf-8")
    dataset_snapshot = tmp_path / "dataset.json"
    registry_snapshot = tmp_path / "registry.json"
    dataset_snapshot.write_text("{}\n", encoding="utf-8")
    registry_snapshot.write_text("{}\n", encoding="utf-8")
    return _input_artifacts(
        execution_dir=execution_dir,
        dataset_snapshot=dataset_snapshot,
        registry_snapshot=registry_snapshot,
    )


def test_registry_screening_is_an_agent_selectable_gated_task_and_chat_preview() -> None:
    task = AtomicTaskRegistry().get(TASK_ID)

    assert task.required_artifacts == list(INPUT_ARTIFACT_IDS)
    assert task.output_artifacts == [*OUTPUT_FILENAMES, EXECUTION_RECORD_ARTIFACT_ID]
    assert task.risk_level == RiskLevel.MEDIUM
    assert task.gates == [GateName.FINAL_THRESHOLD.value]
    assert task.default_adapter == ADAPTER_NAME

    proposal = PlannerAgent().propose_plan(
        run_id="registry-agent-plan",
        goal="Screen independent candidates from the approved OLED material Registry.",
    )

    assert proposal.run_plan.requested_tasks == [TASK_ID]
    assert proposal.status == "needs_clarification"
    assert proposal.run_plan.missing_artifacts == sorted(INPUT_ARTIFACT_IDS)
    assert proposal.rationales[0].task_id == TASK_ID
    assert proposal.required_gates == [GateName.FINAL_THRESHOLD.value]


def test_registry_screening_chat_preview_does_not_fall_back_to_generic_report(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    client = app.test_client()

    response = client.post(
        "/api/agent/conversation/run-plan-preview",
        json={
            "project_id": "registry-project",
            "run_id": "registry-chat-preview",
            "modeling_plan_payload": {
                "goal": "Screen candidates from the OLED material registry.",
            },
            "available_artifacts": [],
        },
    )

    assert response.status_code == 200
    body = response.json
    assert body["run_plan"]["requested_tasks"] == [TASK_ID]
    assert body["preview"]["status"] == "blocked_missing_artifacts"
    assert body["preview"]["missing_artifacts"] == sorted(INPUT_ARTIFACT_IDS)
    assert body["preview"]["required_gates"] == [GateName.FINAL_THRESHOLD.value]


def test_registry_screening_adapter_cannot_bypass_runplan_snapshot(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    response = app.test_client().post(
        "/api/adapters/execute",
        json={
            "run_id": "registry-direct-adapter",
            "adapter": ADAPTER_NAME,
            "confirmed": True,
            "actor": "reviewer",
            "payload": {"confirmed": True, "actor": "reviewer"},
        },
    )

    assert response.status_code == 400
    assert response.json["error"] == "gated adapter execution requires run-plan snapshot approval"
    assert response.json["required_gates"] == [GateName.FINAL_THRESHOLD.value]


@pytest.mark.parametrize("change", ["input", "constraint"])
def test_registry_screening_gate_snapshot_rejects_changed_inputs_before_adapter_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = f"registry-snapshot-{change}"
    run_plan = _run_plan(run_id)
    input_artifacts = _placeholder_inputs(tmp_path)
    calls: list[dict[str, object]] = []

    def unexpected_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("adapter must not run before a valid gate resume")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected_adapter)
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="registry-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_screening_options(),
    )

    assert first["status"] == RunStatus.WAITING_USER.value
    assert first["waiting_task"] == TASK_ID
    assert calls == []
    run_dir = storage.run_dir("registry-project", run_id)
    assert not (run_dir / "oled_registry_screening").exists()
    state = storage.read_stage_state("registry-project", run_id)
    assert state is not None
    snapshot = state.details["execution_snapshot"]
    manifest = snapshot["resource_manifest"]
    bound_artifacts = {
        item["artifact_id"]
        for item in manifest.values()
        if isinstance(item, dict) and item.get("artifact_id")
    }
    assert set(INPUT_ARTIFACT_IDS).issubset(bound_artifacts)

    changed_options = _screening_options()
    if change == "input":
        Path(input_artifacts["oled_dataset_snapshot"]).write_text("{\"changed\":true}\n", encoding="utf-8")
    else:
        changed_options[TASK_ID]["minimums"] = ["s1_ev=0.1"]

    with pytest.raises(ValueError, match="execution snapshot changed"):
        executor.resume_after_gate(
            project_id="registry-project",
            run_plan=run_plan,
            approved_gates=[GateName.FINAL_THRESHOLD.value],
            actor="reviewer",
            input_artifacts=input_artifacts,
            task_options=changed_options,
        )

    assert calls == []
    assert not (run_dir / "oled_registry_screening").exists()


def test_registry_screening_rejects_output_or_input_override_options(tmp_path: Path) -> None:
    storage = ProjectStorage(tmp_path / "workspace")
    input_artifacts = _placeholder_inputs(tmp_path)
    run_plan = _run_plan("registry-invalid-options")

    with pytest.raises(ValueError, match="cannot override artifact identity keys"):
        RunPlanExecutor(storage=storage).execute(
            project_id="registry-project",
            run_plan=run_plan,
            input_artifacts=input_artifacts,
            task_options={TASK_ID: {"output_root": str(tmp_path / "outside")}},
        )

    with pytest.raises(ValueError, match="unsupported Registry screening task option"):
        RunPlanExecutor(storage=storage).execute(
            project_id="registry-project",
            run_plan=run_plan.model_copy(update={"run_id": "registry-invalid-input"}),
            input_artifacts=input_artifacts,
            task_options={TASK_ID: {"phase1_execution_dir": str(tmp_path / "outside")}},
        )


def test_registry_screening_snapshot_binds_relative_execution_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    storage = ProjectStorage(tmp_path / "workspace")
    input_artifacts = _placeholder_inputs(tmp_path)
    relative_artifacts = {
        artifact_id: str(Path(path).relative_to(tmp_path))
        for artifact_id, path in input_artifacts.items()
    }
    run_id = "registry-relative-inputs"

    first = RunPlanExecutor(storage=storage).execute(
        project_id="registry-project",
        run_plan=_run_plan(run_id),
        input_artifacts=relative_artifacts,
        task_options=_screening_options(),
    )

    assert first["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("registry-project", run_id)
    assert state is not None
    manifest = state.details["execution_snapshot"]["resource_manifest"]
    bound_artifacts = {
        item["artifact_id"]
        for item in manifest.values()
        if isinstance(item, dict) and item.get("artifact_id")
    }
    assert set(INPUT_ARTIFACT_IDS).issubset(bound_artifacts)


def test_registry_screening_executes_after_gate_and_surfaces_all_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    storage = ProjectStorage(workspace)
    run_id = "registry-screening-success"
    run_plan = _run_plan(run_id)
    input_artifacts = _input_artifacts(
        execution_dir=inputs.execution_dir,
        dataset_snapshot=inputs.dataset_snapshot,
        registry_snapshot=inputs.registry_snapshot,
    )
    options = _screening_options()
    executor = RunPlanExecutor(storage=storage)

    first = executor.execute(
        project_id="registry-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=options,
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("registry-project", run_id)
    assert state is not None
    snapshot = state.details["execution_snapshot"]

    result = executor.resume_after_gate(
        project_id="registry-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        note="screen this exact Registry snapshot",
        input_artifacts=input_artifacts,
        task_options=options,
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == [TASK_ID]
    registry = storage.read_artifact_registry("registry-project", run_id)
    assert set(registry) == {*OUTPUT_FILENAMES, EXECUTION_RECORD_ARTIFACT_ID}
    run_dir = storage.run_dir("registry-project", run_id)
    for artifact_id, filename in OUTPUT_FILENAMES.items():
        artifact_path = run_dir / registry[artifact_id]
        assert artifact_path.is_file()
        assert artifact_path.name == filename
        assert artifact_path.is_relative_to(run_dir)
    assert (run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]).name == "adapter_result.json"

    receipt_path = run_dir / registry["oled_registry_screening_receipt"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["counts"] == {
        "eligible_candidate_count": 2,
        "excluded_candidate_count": 2,
        "prediction_count": 2,
        "registry_candidate_count": 4,
        "shortlist_count": 2,
    }
    assert receipt["claims"]["experimental_validation_claimed"] is False
    assert receipt["claims"]["registry_mutated"] is False
    assert receipt["claims"]["model_registered"] is False

    adapter_result = json.loads(
        (run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]).read_text(encoding="utf-8")
    )
    assert adapter_result["status"] == "success"
    assert adapter_result["adapter"] == ADAPTER_NAME
    assert adapter_result["summary"]["screening_id"] == receipt["screening_id"]
    assert adapter_result["summary"]["shortlist_count"] == 2

    decisions = storage.read_gate_decisions("registry-project", run_id)
    assert len(decisions) == 1
    assert decisions[0]["gate"] == GateName.FINAL_THRESHOLD.value
    assert decisions[0]["approved_snapshot_id"] == snapshot["snapshot_id"]
    confirmations = storage.read_execution_confirmations("registry-project", run_id)
    assert len(confirmations) == 1
    assert confirmations[0]["snapshot_hash"] == snapshot["snapshot_hash"]
    assert confirmations[0]["task_id"] == TASK_ID

    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    feedback = app.test_client().post(
        "/api/agent/conversation/execution-feedback",
        json={"project_id": "registry-project", "run_id": run_id},
    )
    assert feedback.status_code == 200
    assert feedback.json["feedback"]["artifact_registry"] == registry
    assert "review_artifacts" in feedback.json["feedback"]["next_actions"]


def test_registry_screening_no_replace_retry_preserves_first_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "registry-screening-no-replace"
    run_plan = _run_plan(run_id)
    input_artifacts = _input_artifacts(
        execution_dir=inputs.execution_dir,
        dataset_snapshot=inputs.dataset_snapshot,
        registry_snapshot=inputs.registry_snapshot,
    )
    options = _screening_options()
    executor = RunPlanExecutor(storage=storage)

    assert executor.execute(
        project_id="registry-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=options,
    )["status"] == RunStatus.WAITING_USER.value
    assert executor.resume_after_gate(
        project_id="registry-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=options,
    )["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("registry-project", run_id)
    receipt_path = storage.run_dir("registry-project", run_id) / registry["oled_registry_screening_receipt"]
    first_receipt_bytes = receipt_path.read_bytes()

    assert executor.execute(
        project_id="registry-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=options,
    )["status"] == RunStatus.WAITING_USER.value
    failed = executor.resume_after_gate(
        project_id="registry-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=options,
    )

    assert failed["status"] == RunStatus.FAILED.value
    assert failed["failed_task"] == TASK_ID
    assert failed["result"]["error"]["code"] == "registry_candidate_screening_failed"
    assert receipt_path.read_bytes() == first_receipt_bytes


def test_registry_screening_adapter_redacts_failure_paths_and_requires_confirmation(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    output_root = run_dir / "oled_registry_screening"
    payload = {
        "run_id": "registry-adapter-failure",
        "run_dir": str(run_dir),
        "output_root": str(output_root),
        "phase1_execution_dir": str(tmp_path / "private-execution"),
        "dataset_snapshot_json": str(tmp_path / "private-dataset.json"),
        "registry_snapshot_json": str(tmp_path / "private-registry.json"),
        "confirmed": True,
        "actor": "reviewer",
    }

    failed = execute_oled_registry_candidate_screening_adapter(payload)
    assert failed["status"] == "failed"
    assert failed["error"] == {
        "code": "registry_candidate_screening_failed",
        "message": "Registry candidate screening failed.",
    }
    assert str(tmp_path) not in json.dumps(failed, sort_keys=True)

    unconfirmed = execute_oled_registry_candidate_screening_adapter({**payload, "confirmed": False})
    assert unconfirmed["status"] == "failed"
    assert unconfirmed["error"]["code"] == "confirmation_required"
