from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4s_agent import adapters
from ai4s_agent.agents.planner import PlannerAgent
from ai4s_agent.executor import RunPlanExecutor
from ai4s_agent.oled_real_phase1_execution import _json_bytes
from ai4s_agent.planner import AtomicTaskRegistry, expand_run_plan
from ai4s_agent.schemas import RiskLevel, RunStatus
from ai4s_agent.storage import ProjectStorage
from tests.test_oled_bounded_discovery_controller import _request


TASK_ID = "execute_oled_bounded_discovery_controller"
ADAPTER_NAME = "execute_oled_bounded_discovery_controller_adapter"
OUTPUTS = {
    "oled_bounded_controller_receipt": "controller.json",
    "oled_bounded_controller_report": "report.md",
}
RECORD_ID = "oled_bounded_controller_execution_record"


def _plan(run_id: str) -> object:
    return expand_run_plan(
        run_id=run_id,
        requested_tasks=[TASK_ID],
        available_artifacts=["oled_bounded_controller_request"],
    )


def test_controller_is_low_risk_plannable_and_does_not_own_generation_gate() -> None:
    task = AtomicTaskRegistry().get(TASK_ID)
    assert task.required_artifacts == ["oled_bounded_controller_request"]
    assert task.output_artifacts == [*OUTPUTS, RECORD_ID]
    assert task.risk_level == RiskLevel.LOW
    assert task.gates == []
    assert task.default_adapter == ADAPTER_NAME

    proposal = PlannerAgent().propose_plan(
        run_id="bounded-controller-plan",
        goal="Run the PR-AU bounded closed-loop discovery controller.",
        available_artifacts=["oled_bounded_controller_request"],
    )
    assert proposal.run_plan.requested_tasks == [TASK_ID]
    assert proposal.required_gates == []
    assert "gated-generation request" in proposal.rationales[0].reason


def test_executor_registers_controller_and_retry_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    executor = RunPlanExecutor(storage=storage)
    run_id = "bounded-controller-success"
    inputs = {"oled_bounded_controller_request": str(request)}

    result = executor.execute(
        project_id="bounded-controller-project",
        run_plan=_plan(run_id),
        input_artifacts=inputs,
    )
    assert result["status"] == RunStatus.SUCCEEDED.value
    registry = storage.read_artifact_registry("bounded-controller-project", run_id)
    assert set(registry) == {*OUTPUTS, RECORD_ID}
    run_dir = storage.run_dir("bounded-controller-project", run_id)
    receipt = run_dir / registry["oled_bounded_controller_receipt"]
    before = receipt.read_bytes()

    calls: list[dict[str, object]] = []

    def unexpected(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        raise AssertionError("completed immutable controller must not dispatch again")

    monkeypatch.setattr(adapters, ADAPTER_NAME, unexpected)
    retry = executor.execute(
        project_id="bounded-controller-project",
        run_plan=_plan(run_id),
        input_artifacts=inputs,
    )
    assert retry["status"] == RunStatus.SUCCEEDED.value
    assert retry["result"]["already_completed"] is True
    assert calls == []
    assert receipt.read_bytes() == before


def test_executor_rejects_fully_resigned_controller_route_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, monkeypatch)
    storage = ProjectStorage(tmp_path / "workspace")
    executor = RunPlanExecutor(storage=storage)
    real_adapter = getattr(adapters, ADAPTER_NAME)

    def forged(payload: dict[str, object]) -> dict[str, object]:
        result = real_adapter(payload)
        receipt_path = Path(str(result["outputs"]["oled_bounded_controller_receipt"]))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["route"]["next_action"] = "execute_generation_without_gate"
        receipt["route"]["requires_human_approval"] = False
        receipt_path.write_bytes(_json_bytes(receipt))
        return result

    monkeypatch.setattr(adapters, ADAPTER_NAME, forged)
    run_id = "bounded-controller-forged"
    result = executor.execute(
        project_id="bounded-controller-project",
        run_plan=_plan(run_id),
        input_artifacts={"oled_bounded_controller_request": str(request)},
    )
    assert result["status"] == RunStatus.FAILED.value
    assert result["result"]["error"]["code"] == "artifact_collection_failed"
    assert storage.read_artifact_registry("bounded-controller-project", run_id) == {}
