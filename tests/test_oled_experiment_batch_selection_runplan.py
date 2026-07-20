from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from ai4s_agent import adapters
from ai4s_agent.adapters.oled_experiment_batch_selection import (
    execute_oled_experiment_batch_selection_adapter,
)
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.app import create_app
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.oled_registry_candidate_screening import (
    run_oled_registry_candidate_screening_from_files,
)
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import GateName, RiskLevel, RunStatus
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_registry_candidate_screening import _screening_inputs


TASK_ID = "execute_oled_experiment_batch_selection"
ADAPTER_NAME = "execute_oled_experiment_batch_selection_adapter"
INPUT_ARTIFACT_IDS = (
    "oled_registry_screening_receipt",
    "oled_registry_screening_shortlist",
)
OUTPUT_FILENAMES = {
    "oled_experiment_batch_receipt": "batch_selection.json",
    "oled_experiment_batch_handoff": "experiment_batch.csv",
    "oled_experiment_batch_report": "experiment_handoff.md",
}
EXECUTION_RECORD_ARTIFACT_ID = "oled_experiment_batch_execution_record"


def _run_plan(run_id: str) -> object:
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=list(INPUT_ARTIFACT_IDS),
    )


def _selection_options(*, target_batch_size: int = 1) -> dict[str, dict[str, object]]:
    options: dict[str, object] = {
        "target_batch_size": target_batch_size,
        "minimums": ["s1_ev=0.0"],
        "maximums": ["delta_e_st_ev=1.0"],
    }
    if target_batch_size > 1:
        options["max_pairwise_tanimoto"] = 0.7
    return {TASK_ID: options}


def _screening_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_name: str = "screenings",
    minimums: list[str] | None = None,
) -> dict[str, str]:
    inputs = _screening_inputs(tmp_path, monkeypatch)
    result = run_oled_registry_candidate_screening_from_files(
        phase1_execution_dir=inputs.execution_dir,
        dataset_snapshot_json=inputs.dataset_snapshot,
        registry_snapshot_json=inputs.registry_snapshot,
        output_root=tmp_path / output_name,
        minimums=minimums or ["s1_ev=0.0"],
        maximums=["delta_e_st_ev=1.0"],
        generated_at="2026-07-20T10:00:00+08:00",
    )
    return {
        "oled_registry_screening_receipt": str(result.output_dir / "screening.json"),
        "oled_registry_screening_shortlist": str(result.output_dir / "ranked_shortlist.csv"),
    }


def _cost_manifest_path(
    tmp_path: Path,
    input_artifacts: dict[str, str],
) -> Path:
    receipt_path = Path(input_artifacts["oled_registry_screening_receipt"])
    shortlist_path = Path(input_artifacts["oled_registry_screening_shortlist"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with shortlist_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    manifest = {
        "cost_manifest_version": "oled_candidate_cost_manifest.v1",
        "screening_id": receipt["screening_id"],
        "ranked_shortlist_sha256": "sha256:"
        + hashlib.sha256(shortlist_path.read_bytes()).hexdigest(),
        "currency": "USD",
        "entries": [
            {
                "material_id": row["material_id"],
                "registry_entry_digest": row["registry_entry_digest"],
                "cost_minor": 100,
            }
            for row in rows
        ],
    }
    path = tmp_path / "candidate-cost-manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_experiment_batch_is_agent_selectable_gated_task_and_chat_preview(
    tmp_path: Path,
) -> None:
    task = AtomicTaskRegistry().get(TASK_ID)

    assert task.required_artifacts == list(INPUT_ARTIFACT_IDS)
    assert task.output_artifacts == [*OUTPUT_FILENAMES, EXECUTION_RECORD_ARTIFACT_ID]
    assert task.risk_level == RiskLevel.MEDIUM
    assert task.gates == [GateName.FINAL_THRESHOLD.value]
    assert task.default_adapter == ADAPTER_NAME

    proposal = PlannerAgent().propose_plan(
        run_id="experiment-batch-agent-plan",
        goal="Select an experimental validation batch from the approved shortlist.",
        available_artifacts=list(INPUT_ARTIFACT_IDS),
    )

    assert proposal.run_plan.requested_tasks == [TASK_ID]
    assert proposal.status == "needs_confirmation"
    assert proposal.run_plan.missing_artifacts == []
    assert proposal.rationales[0].task_id == TASK_ID
    assert proposal.required_gates == [GateName.FINAL_THRESHOLD.value]

    workspace = tmp_path / "workspace"
    app = create_app(base_runs_dir=workspace / "runs", workspace_dir=workspace)
    response = app.test_client().post(
        "/api/agent/conversation/run-plan-preview",
        json={
            "project_id": "experiment-batch-project",
            "run_id": "experiment-batch-chat-preview",
            "modeling_plan_payload": {"goal": "从 shortlist 选择待验证批次"},
            "available_artifacts": list(INPUT_ARTIFACT_IDS),
        },
    )

    assert response.status_code == 200
    body = response.json
    assert body["run_plan"]["requested_tasks"] == [TASK_ID]
    assert body["preview"]["status"] == "ready_for_controlled_execution"
    assert body["preview"]["missing_artifacts"] == []
    assert body["preview"]["required_gates"] == [GateName.FINAL_THRESHOLD.value]


def test_experiment_batch_adapter_cannot_bypass_runplan_snapshot(tmp_path: Path) -> None:
    app = create_app(base_runs_dir=tmp_path / "runs", workspace_dir=tmp_path)
    response = app.test_client().post(
        "/api/adapters/execute",
        json={
            "run_id": "experiment-batch-direct-adapter",
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
def test_experiment_batch_gate_snapshot_rejects_changed_inputs_before_adapter_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    input_artifacts = _screening_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = f"experiment-batch-snapshot-{change}"
    run_plan = _run_plan(run_id)
    calls: list[dict[str, object]] = []

    def unexpected_adapter(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("adapter must not run before a valid gate resume")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected_adapter)
    executor = RunPlanExecutor(storage=storage)
    first = executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )

    assert first["status"] == RunStatus.WAITING_USER.value
    assert first["waiting_task"] == TASK_ID
    assert calls == []
    run_dir = storage.run_dir("experiment-batch-project", run_id)
    assert not (run_dir / "oled_experiment_batch").exists()
    state = storage.read_stage_state("experiment-batch-project", run_id)
    assert state is not None
    snapshot = state.details["execution_snapshot"]
    manifest = snapshot["resource_manifest"]
    bound_artifacts = {
        item["artifact_id"]
        for item in manifest.values()
        if isinstance(item, dict) and item.get("artifact_id")
    }
    assert set(INPUT_ARTIFACT_IDS).issubset(bound_artifacts)

    changed_options = _selection_options()
    if change == "input":
        receipt = Path(input_artifacts["oled_registry_screening_receipt"])
        receipt.write_bytes(receipt.read_bytes() + b"\n")
    else:
        changed_options[TASK_ID]["target_batch_size"] = 2
        changed_options[TASK_ID]["max_pairwise_tanimoto"] = 0.7

    with pytest.raises(ValueError, match="execution snapshot changed"):
        executor.resume_after_gate(
            project_id="experiment-batch-project",
            run_plan=run_plan,
            approved_gates=[GateName.FINAL_THRESHOLD.value],
            actor="reviewer",
            input_artifacts=input_artifacts,
            task_options=changed_options,
        )

    assert calls == []
    assert not (run_dir / "oled_experiment_batch").exists()


def test_experiment_batch_freezes_exact_inputs_and_rechecks_source_before_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _screening_artifacts(tmp_path, monkeypatch)
    original_receipt = Path(input_artifacts["oled_registry_screening_receipt"])
    original_shortlist = Path(input_artifacts["oled_registry_screening_shortlist"])
    replacement_root = tmp_path / "replacement-inputs"
    replacement_root.mkdir()
    replacement = _screening_artifacts(
        replacement_root,
        monkeypatch,
        output_name="replacement-screenings",
        minimums=["s1_ev=0.1"],
    )
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "experiment-batch-post-recheck-swap"
    run_plan = _run_plan(run_id)
    executor = RunPlanExecutor(storage=storage)
    assert executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )["status"] == RunStatus.WAITING_USER.value

    adapter_calls: list[dict[str, object]] = []
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def tracking_adapter(payload: dict[str, object]) -> dict[str, object]:
        adapter_calls.append(payload)
        return real_adapter(payload)

    monkeypatch.setattr(adapters, ADAPTER_NAME, tracking_adapter)
    original_validate = executor._validate_waiting_execution_snapshot

    def validate_then_swap(**kwargs: object) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
        validated = original_validate(**kwargs)  # type: ignore[arg-type]
        original_receipt.write_bytes(
            Path(replacement["oled_registry_screening_receipt"]).read_bytes()
        )
        original_shortlist.write_bytes(
            Path(replacement["oled_registry_screening_shortlist"]).read_bytes()
        )
        return validated

    monkeypatch.setattr(executor, "_validate_waiting_execution_snapshot", validate_then_swap)
    run_dir = storage.run_dir("experiment-batch-project", run_id)
    with pytest.raises(
        ValueError,
        match="Experiment batch source binding changed after gate snapshot",
    ):
        executor.resume_after_gate(
            project_id="experiment-batch-project",
            run_plan=run_plan,
            approved_gates=[GateName.FINAL_THRESHOLD.value],
            actor="reviewer",
            input_artifacts=input_artifacts,
            task_options=_selection_options(),
        )

    assert adapter_calls == []
    assert not (run_dir / "oled_experiment_batch").exists()


def test_experiment_batch_executes_after_gate_and_surfaces_immutable_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _screening_artifacts(tmp_path, monkeypatch)
    workspace = tmp_path / "workspace"
    storage = ProjectStorage(workspace)
    run_id = "experiment-batch-success"
    run_plan = _run_plan(run_id)
    executor = RunPlanExecutor(storage=storage)

    first = executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )
    assert first["status"] == RunStatus.WAITING_USER.value
    run_dir = storage.run_dir("experiment-batch-project", run_id)
    state = storage.read_stage_state("experiment-batch-project", run_id)
    assert state is not None
    snapshot = state.details["execution_snapshot"]
    snapshot_payload = snapshot["execution_payload"]
    assert snapshot_payload["source_screening_receipt_json"] == input_artifacts[
        "oled_registry_screening_receipt"
    ]
    assert snapshot_payload["screening_receipt_json"] != input_artifacts[
        "oled_registry_screening_receipt"
    ]
    frozen_receipt_path = Path(snapshot_payload["screening_receipt_json"])
    assert frozen_receipt_path.is_file()
    assert frozen_receipt_path.is_relative_to(run_dir / TASK_ID)

    adapter_payloads: list[dict[str, object]] = []
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def tracking_adapter(payload: dict[str, object]) -> dict[str, object]:
        adapter_payloads.append(payload)
        return real_adapter(payload)

    monkeypatch.setattr(adapters, ADAPTER_NAME, tracking_adapter)

    result = executor.resume_after_gate(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        note="select this exact shortlist for a recommendation-only handoff",
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert result["executed_tasks"] == [TASK_ID]
    assert len(adapter_payloads) == 1
    assert not any(key.startswith("source_") for key in adapter_payloads[0])
    assert Path(str(adapter_payloads[0]["screening_receipt_json"])).is_relative_to(
        run_dir / TASK_ID
    )
    assert Path(str(adapter_payloads[0]["ranked_shortlist_csv"])).is_relative_to(
        run_dir / TASK_ID
    )
    registry = storage.read_artifact_registry("experiment-batch-project", run_id)
    assert set(registry) == {*OUTPUT_FILENAMES, EXECUTION_RECORD_ARTIFACT_ID}
    for artifact_id, filename in OUTPUT_FILENAMES.items():
        artifact_path = run_dir / registry[artifact_id]
        assert artifact_path.is_file()
        assert artifact_path.name == filename
        assert artifact_path.is_relative_to(run_dir)
    record_path = run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]
    assert record_path.name.startswith("adapter_result_")

    receipt = json.loads(
        (run_dir / registry["oled_experiment_batch_receipt"]).read_text(encoding="utf-8")
    )
    assert receipt["claims"]["experimental_validation_claimed"] is False
    assert receipt["claims"]["recommendation_only"] is True
    assert receipt["claims"]["registry_mutated"] is False

    adapter_result = json.loads(record_path.read_text(encoding="utf-8"))
    assert adapter_result["status"] == "success"
    assert adapter_result["adapter"] == ADAPTER_NAME
    assert adapter_result["summary"]["batch_id"] == receipt["batch_id"]
    assert adapter_result["summary"]["experiment_started"] is False
    assert adapter_result["summary"]["procurement_started"] is False
    assert adapter_result["summary"]["experiment_executed"] is False
    assert adapter_result["summary"]["procurement_performed"] is False
    assert adapter_result["summary"]["synthesis_performed"] is False
    assert adapter_result["summary"]["measurement_performed"] is False

    decisions = storage.read_gate_decisions("experiment-batch-project", run_id)
    assert len(decisions) == 1
    assert decisions[0]["gate"] == GateName.FINAL_THRESHOLD.value
    assert decisions[0]["approved_snapshot_id"] == snapshot["snapshot_id"]
    confirmations = storage.read_execution_confirmations("experiment-batch-project", run_id)
    assert len(confirmations) == 1
    assert confirmations[0]["snapshot_hash"] == snapshot["snapshot_hash"]
    assert confirmations[0]["task_id"] == TASK_ID


def test_experiment_batch_not_ready_is_a_successful_advisory_without_partial_handoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _screening_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "experiment-batch-not-ready"
    run_plan = _run_plan(run_id)
    options = _selection_options(target_batch_size=3)
    executor = RunPlanExecutor(storage=storage)

    assert executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=options,
    )["status"] == RunStatus.WAITING_USER.value
    result = executor.resume_after_gate(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=options,
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    run_dir = storage.run_dir("experiment-batch-project", run_id)
    registry = storage.read_artifact_registry("experiment-batch-project", run_id)
    receipt = json.loads(
        (run_dir / registry["oled_experiment_batch_receipt"]).read_text(encoding="utf-8")
    )
    assert receipt["status"] == "not_ready"
    assert receipt["counts"]["selected_candidate_count"] == 0
    assert (run_dir / registry["oled_experiment_batch_handoff"]).read_text(
        encoding="utf-8"
    ).count("\n") == 1
    adapter_result = json.loads(
        (run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]).read_text(encoding="utf-8")
    )
    assert adapter_result["status"] == "success"
    assert adapter_result["summary"]["batch_status"] == "not_ready"


def test_experiment_batch_freezes_optional_cost_manifest_for_money_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _screening_artifacts(tmp_path, monkeypatch)
    cost_manifest = _cost_manifest_path(tmp_path, input_artifacts)
    input_artifacts["oled_candidate_cost_manifest"] = str(cost_manifest)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "experiment-batch-cost-manifest"
    run_plan = _run_plan(run_id)
    options = _selection_options()
    options[TASK_ID]["max_budget_minor"] = 200
    executor = RunPlanExecutor(storage=storage)

    assert executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=options,
    )["status"] == RunStatus.WAITING_USER.value
    state = storage.read_stage_state("experiment-batch-project", run_id)
    assert state is not None
    snapshot_payload = state.details["execution_snapshot"]["execution_payload"]
    assert snapshot_payload["source_candidate_cost_manifest_json"] == str(cost_manifest)
    frozen_cost_path = Path(snapshot_payload["candidate_cost_manifest_json"])
    assert frozen_cost_path.is_file()

    adapter_payloads: list[dict[str, object]] = []
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def tracking_adapter(payload: dict[str, object]) -> dict[str, object]:
        adapter_payloads.append(payload)
        return real_adapter(payload)

    monkeypatch.setattr(adapters, ADAPTER_NAME, tracking_adapter)
    result = executor.resume_after_gate(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=options,
    )

    assert result["status"] == RunStatus.SUCCEEDED.value
    assert len(adapter_payloads) == 1
    assert "source_candidate_cost_manifest_json" not in adapter_payloads[0]
    assert Path(str(adapter_payloads[0]["candidate_cost_manifest_json"])).is_relative_to(
        storage.run_dir("experiment-batch-project", run_id) / TASK_ID
    )
    registry = storage.read_artifact_registry("experiment-batch-project", run_id)
    receipt = json.loads(
        (
            storage.run_dir("experiment-batch-project", run_id)
            / registry["oled_experiment_batch_receipt"]
        ).read_text(encoding="utf-8")
    )
    assert receipt["sources"]["candidate_cost_manifest_sha256"] == "sha256:" + hashlib.sha256(
        cost_manifest.read_bytes()
    ).hexdigest()


def test_experiment_batch_retry_preserves_first_execution_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_artifacts = _screening_artifacts(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    run_id = "experiment-batch-no-replace"
    run_plan = _run_plan(run_id)
    executor = RunPlanExecutor(storage=storage)

    assert executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )["status"] == RunStatus.WAITING_USER.value
    assert executor.resume_after_gate(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("experiment-batch-project", run_id)
    first_registry = dict(registry)
    run_dir = storage.run_dir("experiment-batch-project", run_id)
    first_receipt_path = run_dir / registry["oled_experiment_batch_receipt"]
    first_receipt_bytes = first_receipt_path.read_bytes()
    first_record_path = run_dir / registry[EXECUTION_RECORD_ARTIFACT_ID]
    first_record_bytes = first_record_path.read_bytes()

    assert executor.execute(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )["status"] == RunStatus.WAITING_USER.value
    failed = executor.resume_after_gate(
        project_id="experiment-batch-project",
        run_plan=run_plan,
        approved_gates=[GateName.FINAL_THRESHOLD.value],
        actor="reviewer",
        input_artifacts=input_artifacts,
        task_options=_selection_options(),
    )

    assert failed["status"] == RunStatus.FAILED.value
    assert failed["failed_task"] == TASK_ID
    assert failed["result"]["error"]["code"] == "experiment_batch_selection_failed"
    assert first_receipt_path.read_bytes() == first_receipt_bytes
    assert first_record_path.read_bytes() == first_record_bytes
    assert storage.read_artifact_registry("experiment-batch-project", run_id) == first_registry

    state = storage.read_stage_state("experiment-batch-project", run_id)
    assert state is not None
    assert state.status == RunStatus.FAILED
    assert len(state.artifacts) == 1
    failed_record_path = run_dir / state.artifacts[0].relative_path
    assert failed_record_path != first_record_path
    assert json.loads(failed_record_path.read_text(encoding="utf-8"))["status"] == "failed"
    attempt_records = sorted((run_dir / TASK_ID).glob("adapter_result_*.json"))
    assert attempt_records == sorted([first_record_path, failed_record_path])


def test_experiment_batch_adapter_requires_diversity_or_cost_inputs_when_needed(
    tmp_path: Path,
) -> None:
    payload = {
        "run_id": "experiment-batch-adapter-validation",
        "output_root": str(tmp_path / "run" / "oled_experiment_batch"),
        "screening_receipt_json": str(tmp_path / "private-screening.json"),
        "ranked_shortlist_csv": str(tmp_path / "private-shortlist.csv"),
        "target_batch_size": 2,
        "confirmed": True,
        "actor": "reviewer",
    }

    missing_diversity = execute_oled_experiment_batch_selection_adapter(payload)
    assert missing_diversity["status"] == "failed"
    assert missing_diversity["error"]["code"] == "diversity_threshold_required"

    missing_cost = execute_oled_experiment_batch_selection_adapter(
        {**payload, "target_batch_size": 1, "max_budget_minor": 100}
    )
    assert missing_cost["status"] == "failed"
    assert missing_cost["error"]["code"] == "cost_manifest_required"
