import json

from ai4s_agent.orchestrator import Orchestrator
from ai4s_agent.schemas import GateName


def test_orchestrator_stops_at_first_unapproved_gate(tmp_path) -> None:
    orch = Orchestrator(base_runs_dir=tmp_path)
    status = orch.start_run(run_id="r1", prompt="opt")
    assert status["state"] == "WAITING_USER"
    assert status["gate"] == "gate_1_task_parse"


def test_orchestrator_writes_plan_artifact(tmp_path) -> None:
    orch = Orchestrator(base_runs_dir=tmp_path)
    orch.start_run(run_id="r1", prompt="opt")
    assert (tmp_path / "r1" / "plan.json").exists()


def test_orchestrator_rejects_reused_run_id_with_existing_state(tmp_path) -> None:
    orch = Orchestrator(base_runs_dir=tmp_path)
    orch.start_run(run_id="r1", prompt="opt")
    orch.approve_gate(run_id="r1", gate=GateName.TASK_PARSE, actor="user")

    try:
        orch.start_run(run_id="r1", prompt="new prompt")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "already exists" in str(exc)


def test_orchestrator_approval_state_survives_restart(tmp_path) -> None:
    first = Orchestrator(base_runs_dir=tmp_path)
    first.start_run(run_id="r1", prompt="opt")
    approved = first.approve_gate(
        run_id="r1",
        gate=GateName.TASK_PARSE,
        actor="user",
    )
    assert approved["state"] == "WAITING_USER"
    assert approved["next_gate"] == GateName.DATA_MINING.value

    second = Orchestrator(base_runs_dir=tmp_path)
    assert second.gates.can_advance("r1", GateName.TASK_PARSE) is True
    assert second.gates.is_next_gate("r1", GateName.DATA_MINING) is True


def test_orchestrator_gate_decisions_include_approval_timestamp(tmp_path) -> None:
    orch = Orchestrator(base_runs_dir=tmp_path)
    orch.start_run(run_id="r1", prompt="opt")

    orch.approve_gate(run_id="r1", gate=GateName.TASK_PARSE, actor="user")

    payload = json.loads((tmp_path / "r1" / "gate_decisions.json").read_text(encoding="utf-8"))
    assert payload["decisions"][0]["approved_at"].endswith("Z")


def test_orchestrator_rejects_out_of_order_gate_approval(tmp_path) -> None:
    orch = Orchestrator(base_runs_dir=tmp_path)
    orch.start_run(run_id="r1", prompt="opt")

    try:
        orch.approve_gate(run_id="r1", gate=GateName.FINAL_THRESHOLD, actor="user")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "out of order" in str(exc)
